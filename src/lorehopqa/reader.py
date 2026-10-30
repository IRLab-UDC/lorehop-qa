"""The reader side of the collection: one item, one prompt, one scored prediction.

Everything the demonstration experiments (E1-E4) share lives here — how an item is loaded from
either collection, how a condition rewrites its context, how the prompt is built, and what shape
the model must answer in. The experiment scripts choose items and conditions; none of them
builds a prompt of its own, because a template that drifted between conditions would make the
closed-book floor incomparable with the context conditions it is supposed to calibrate.

Two decisions here are worth stating up front, because both are visible in the numbers.

SENTENCES ARE ADDRESSED BY NUMBER, NOT BY TITLE.  The official metric compares
`(title, sentence index)` pairs, so the obvious prompt asks the model to write the title back.
That measures transcription as much as reading: a reader that identifies the right sentence and
misspells a fictional title scores zero on supporting facts, and invented titles are exactly the
strings a model is worst at copying. The prompt therefore numbers the documents and asks for
`(document number, sentence number)`, and this module maps the numbers back to titles before
scoring. The help is identical for every reader in the panel, and it removes a source of noise
that would fall hardest on the collection whose names are unfamiliar — which is ours.

AN OUT-OF-RANGE REFERENCE IS SCORED, NOT DROPPED.  A document number the prompt never offered
maps to a sentinel title that cannot match any gold pair, so it counts as a false positive.
Discarding such references instead would silently reward a model for guessing widely.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .eval_metrics import SupportRef

# Document numbers the model may not use. Kept out of the title namespace of both collections
# so it can never collide with a real gold reference.
INVALID_TITLE = "<out-of-range>"


# --------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class EvalItem:
    """One question with its evidence, in whichever layout the current condition produced.

    `meta` carries the breakdown axes the experiments report over — origin, world, generator
    family, hop count — so a prediction row can be grouped without rejoining the source files.
    """

    id: str
    question: str
    answer: str
    type: str
    level: str | None
    context: tuple[Document, ...]
    supporting_facts: tuple[SupportRef, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def supporting_titles(self) -> tuple[str, ...]:
        """Distinct supporting document titles, in context order."""
        marked = {title for title, _ in self.supporting_facts}
        return tuple(doc.title for doc in self.context if doc.title in marked)

    @property
    def is_yes_no(self) -> bool:
        return self.answer.strip().lower() in {"yes", "no"}

    @property
    def answer_in_question(self) -> bool:
        """Whether the question already contains the answer string.

        A comparison question names every candidate — "which of X or Y ..." — so a reader with no
        evidence at all picks between two strings it was handed and is right about half the time.
        That is the other half of the guessing floor, alongside yes/no items, and it has to be
        measured rather than assumed: without it a closed-book score reads as leakage when it is
        arithmetic. Normalised on both sides with the official answer normalisation, so that
        punctuation and articles do not decide the test.
        """
        from .eval_metrics import normalize_answer

        answer = normalize_answer(self.answer)
        return bool(answer) and answer in normalize_answer(self.question)

    def facts_for(self, titles: Iterable[str]) -> tuple[SupportRef, ...]:
        keep = set(titles)
        return tuple((title, index) for title, index in self.supporting_facts if title in keep)


def _documents(context: Sequence[Any]) -> tuple[Document, ...]:
    return tuple(Document(title=str(entry[0]), sentences=tuple(str(s) for s in entry[1])) for entry in context)


def _facts(supporting_facts: Sequence[Any]) -> tuple[SupportRef, ...]:
    return tuple((str(entry[0]), int(entry[1])) for entry in supporting_facts)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_lorehopqa(
    split_path: Path, provenance_path: Path | None = None
) -> list[EvalItem]:
    """Load a released split, joining the provenance sidecar when it is available.

    Provenance supplies `origin` and the generator family, which E1's fairness decomposition is
    reported over and which the released records deliberately do not carry. Without it the items
    still load and score; only the breakdowns are unavailable.
    """
    provenance: dict[str, dict[str, Any]] = {}
    if provenance_path is not None and provenance_path.exists():
        provenance = {row["_id"]: row for row in read_jsonl(provenance_path)}

    items: list[EvalItem] = []
    for row in read_jsonl(split_path):
        source = provenance.get(row["_id"], {})
        supporting_facts = _facts(row["supporting_facts"])
        context = _documents(row["context"])
        items.append(
            EvalItem(
                id=row["_id"],
                question=row["question"],
                answer=row["answer"],
                type=row["type"],
                level=row.get("level"),
                context=context,
                supporting_facts=supporting_facts,
                meta={
                    "collection": "lorehopqa",
                    "origin": source.get("origin"),
                    "generator_model": source.get("generator_model"),
                    "judge_model": source.get("judge_model"),
                    "source_id": source.get("source_id"),
                    "lore_index": source.get("lore_index"),
                    "documents": len(context),
                    "supporting_documents": len({title for title, _ in supporting_facts}),
                    "supporting_facts": len(supporting_facts),
                },
            )
        )
    return items


def load_hotpot(path: Path, keep_ids: Iterable[str] | None = None) -> list[EvalItem]:
    """Load a HotpotQA distractor split. `keep_ids` restricts it to the paired subset."""
    wanted = set(keep_ids) if keep_ids is not None else None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items: list[EvalItem] = []
    for row in payload:
        if wanted is not None and row["_id"] not in wanted:
            continue
        supporting_facts = _facts(row["supporting_facts"])
        context = _documents(row["context"])
        items.append(
            EvalItem(
                id=row["_id"],
                question=row["question"],
                answer=row["answer"],
                type=row.get("type", "bridge"),
                level=row.get("level"),
                context=context,
                supporting_facts=supporting_facts,
                meta={
                    "collection": "hotpotqa",
                    "origin": "hotpotqa",
                    "documents": len(context),
                    "supporting_documents": len({title for title, _ in supporting_facts}),
                    "supporting_facts": len(supporting_facts),
                },
            )
        )
    return items


# --------------------------------------------------------------------------------------
# Worlds
# --------------------------------------------------------------------------------------

_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{3,}\b")


def _proper_nouns(text: str) -> set[str]:
    return set(_PROPER_NOUN.findall(text))


def assign_worlds(
    provenance_path: Path, lore_dir: Path
) -> dict[str, str]:
    """Map every item id to the invented world it belongs to.

    Novel items record `lore_index` directly. Transposed items do not: the batch pipeline
    assigned each task a lore but wrote only the per-item elaboration of it into
    `generation_lore`, so the base world has to be recovered. It is recovered by proper-noun
    overlap against the 61 lore files, which is unambiguous in practice — an elaboration inherits
    the world's place and institution names, and the mean overlap with the correct lore is an
    order of magnitude above the runner-up.

    The assignment exists so that E2's replacement distractors come from the same fiction as the
    document they replace. Getting one wrong costs a slightly out-of-place paragraph in one
    ablation variant, not a wrong measurement.
    """
    lores = {path.stem: _proper_nouns(path.read_text(encoding="utf-8")) for path in sorted(lore_dir.glob("*.txt"))}
    if not lores:
        raise SystemExit(f"No lore files under {lore_dir}")

    worlds: dict[str, str] = {}
    for row in read_jsonl(provenance_path):
        index = row.get("lore_index")
        if index is not None:
            worlds[row["_id"]] = f"lore_{index}"
            continue
        nouns = _proper_nouns(row.get("generation_lore") or "")
        best, overlap = max(
            ((name, len(nouns & entities)) for name, entities in lores.items()),
            key=lambda pair: pair[1],
        )
        # No overlap at all means the elaboration shares no names with any lore. That item gets
        # its own singleton world rather than being filed under an arbitrary winner.
        worlds[row["_id"]] = best if overlap > 0 else f"unassigned:{row['_id']}"
    return worlds


# --------------------------------------------------------------------------------------
# Distractor pools
# --------------------------------------------------------------------------------------


@dataclass
class DistractorPool:
    """Every non-supporting paragraph in the collection, grouped by world.

    E2 replaces a withheld supporting document rather than deleting it, so it needs a supply of
    paragraphs that belong to the same fiction and carry no part of the answer. Distractors from
    other items in the same world are exactly that: written by the same generator, from the same
    lore, and already judged not to answer their own question.
    """

    by_world: dict[str, list[tuple[str, Document]]] = field(default_factory=dict)

    @classmethod
    def build(cls, items: Sequence[EvalItem], worlds: dict[str, str] | None = None) -> DistractorPool:
        pool = cls()
        for item in items:
            world = (worlds or {}).get(item.id, item.meta.get("collection", "default"))
            marked = {title for title, _ in item.supporting_facts}
            for document in item.context:
                if document.title not in marked:
                    pool.by_world.setdefault(world, []).append((item.id, document))
        return pool

    def draw(
        self,
        *,
        world: str,
        rng: random.Random,
        exclude_titles: set[str],
        forbidden: str,
        target_sentences: int,
        considered: int = 64,
    ) -> Document | None:
        """One replacement paragraph: same world, unused title, and not carrying the answer.

        `forbidden` is the gold answer. A distractor that happened to contain it would hand the
        reader the very string the ablation withheld, turning a leave-one-out variant into a
        differently-shaped full-context one, so candidates are rejected on a normalised substring
        test. Length is matched — nearest sentence count among the candidates considered —
        because a replacement markedly shorter than the document it stands in for changes the
        context length the ablation exists to hold fixed.

        The scan walks a random permutation until it has gathered `considered` valid candidates
        or exhausted the world, rather than taking a fixed number of independent random draws. An
        earlier version did the latter and silently returned nothing whenever the draws all
        happened to land on excluded titles, which cost 11% of the variants in a small pool and
        would have shortened those contexts instead of replacing a document in them.
        """
        needle = forbidden.strip().lower()

        def pick(entries: Sequence[tuple[str, Document]]) -> Document | None:
            if not entries:
                return None
            # Bounded work per call: a full shuffle of the whole-collection fallback would be
            # tens of thousands of swaps for one paragraph.
            if len(entries) > 4 * considered:
                order = rng.sample(range(len(entries)), 4 * considered)
            else:
                order = list(range(len(entries)))
                rng.shuffle(order)

            valid: list[Document] = []
            for index in order:
                _, document = entries[index]
                if document.title in exclude_titles:
                    continue
                if needle and any(needle in sentence.lower() for sentence in document.sentences):
                    continue
                valid.append(document)
                if len(valid) >= considered:
                    break
            if not valid:
                return None
            return min(valid, key=lambda doc: abs(len(doc.sentences) - target_sentences))

        drawn = pick(self.by_world.get(world) or [])
        if drawn is not None:
            return drawn
        # Nothing usable in this world. A paragraph from another one is a smaller distortion than
        # a context one document short, so the fallback is over everything.
        return pick([entry for entries in self.by_world.values() for entry in entries])


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------


def full_context(item: EvalItem) -> EvalItem:
    """The item as released. Present for symmetry, so every condition is named in the runner."""
    return item


def pad_context(
    item: EvalItem,
    *,
    documents: int,
    pool: DistractorPool,
    world: str,
    rng: random.Random,
) -> EvalItem:
    """Grow the context to `documents` paragraphs with same-world distractors.

    This is one half of the layout control. Transposed items ship four documents where a HotpotQA
    distractor item ships ten, and no comparison across the two collections is meaningful until
    that is equalised. Padding ours is the direction that makes ours harder; the other direction
    is `restrict_context`, and Section 5.1 of the paper reports both because neither is neutral.
    """
    if len(item.context) >= documents:
        return item

    context = list(item.context)
    used = {document.title for document in context}
    # Match the item's typical document, not its longest: padding that ran to the outlier
    # length would make every added distractor conspicuously the biggest thing present.
    lengths = sorted(len(document.sentences) for document in context)
    target = lengths[len(lengths) // 2] if lengths else 3
    while len(context) < documents:
        drawn = pool.draw(
            world=world,
            rng=rng,
            exclude_titles=used,
            forbidden=item.answer,
            target_sentences=target,
        )
        if drawn is None:
            break
        used.add(drawn.title)
        context.append(drawn)

    rng.shuffle(context)
    return replace(item, context=tuple(context), meta={**item.meta, "documents": len(context)})


def restrict_context(item: EvalItem, *, distractors: int, rng: random.Random) -> EvalItem:
    """Keep the supporting documents and `distractors` sampled others.

    The other half of the layout control, applied to HotpotQA: ten documents down to the gold
    pair plus two, which is the transposed subset's layout. This direction makes HotpotQA easier
    than the collection it came from, which is why it is reported alongside `pad_context` rather
    than instead of it.
    """
    marked = {title for title, _ in item.supporting_facts}
    supporting = [document for document in item.context if document.title in marked]
    others = [document for document in item.context if document.title not in marked]
    rng.shuffle(others)

    context = supporting + others[:distractors]
    rng.shuffle(context)
    return replace(item, context=tuple(context), meta={**item.meta, "documents": len(context)})


def leave_one_out(
    item: EvalItem,
    *,
    withheld: str,
    pool: DistractorPool,
    world: str,
    rng: random.Random,
) -> EvalItem | None:
    """Withhold one supporting document and replace it with a same-world distractor.

    Replacement rather than deletion, in the same slot: deleting shortens the context and drops
    the number of candidates by one, so a fall in accuracy would confound the loss of evidence
    with the change in layout. The gold answer is left untouched — the question of E2 is whether
    the reader still produces it — while `supporting_facts` is narrowed to the facts that survive,
    so the supporting-fact metrics remain interpretable as "did it point at what is still there".
    """
    positions = [index for index, document in enumerate(item.context) if document.title == withheld]
    if not positions:
        return None
    position = positions[0]

    used = {document.title for document in item.context}
    replacement = pool.draw(
        world=world,
        rng=rng,
        exclude_titles=used,
        forbidden=item.answer,
        target_sentences=len(item.context[position].sentences),
    )
    if replacement is None:
        return None

    context = list(item.context)
    context[position] = replacement
    remaining = tuple((title, index) for title, index in item.supporting_facts if title != withheld)

    return replace(
        item,
        id=f"{item.id}::loo::{position}",
        context=tuple(context),
        supporting_facts=remaining,
        meta={
            **item.meta,
            "ablation_of": item.id,
            "withheld_title": withheld,
            "withheld_position": position,
            "replacement_title": replacement.title,
            "withheld_facts": len(item.supporting_facts) - len(remaining),
        },
    )


# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------


class SentenceRef(BaseModel):
    """One supporting sentence, addressed by the numbers the prompt printed."""

    model_config = ConfigDict(extra="forbid")

    document: int = Field(
        ge=1, le=64, description="Number of the document, as printed in DOCUMENTS (starts at 1)."
    )
    sentence: int = Field(
        ge=0, le=127, description="Number of the sentence inside that document (starts at 0)."
    )


class ReaderAnswer(BaseModel):
    """The open-book reply: an answer, and the sentences it rests on."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "The shortest span that answers the question, copied from the documents. "
            "For a yes/no question answer exactly 'yes' or 'no'. No explanation."
        ),
    )
    supporting_facts: list[SentenceRef] = Field(
        min_length=1,
        max_length=12,
        description=(
            "Every sentence needed to establish the answer, and no others. A multi-hop question "
            "needs at least one sentence from each document along the chain."
        ),
    )


class ClosedBookAnswer(BaseModel):
    """The closed-book reply. No documents were shown, so there is nothing to cite."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Your best answer to the question. For a yes/no question answer exactly 'yes' or "
            "'no'. Answer even if you are unsure; do not refuse and do not explain."
        ),
    )


READER_SYSTEM = (
    "You answer multi-hop questions from a supplied set of documents. The answer is always "
    "present in the documents, and reaching it requires combining facts from more than one of "
    "them. Use only what the documents say, never outside knowledge, and never your own "
    "assumptions about what is plausible."
)

CLOSED_BOOK_SYSTEM = (
    "You answer questions from your own knowledge. No documents are supplied. Give your single "
    "best answer even when you are unsure — a guess is more useful here than a refusal."
)


def render_context(item: EvalItem) -> str:
    """The documents, numbered from 1, with sentences numbered from 0 inside each."""
    blocks: list[str] = []
    for document_number, document in enumerate(item.context, start=1):
        lines = [f"[{document_number}] {document.title}"]
        lines += [
            f"  ({sentence_number}) {sentence}"
            for sentence_number, sentence in enumerate(document.sentences)
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def reader_user(item: EvalItem) -> str:
    return (
        "=== DOCUMENTS ===\n"
        f"{render_context(item)}\n\n"
        "=== QUESTION ===\n"
        f"{item.question}\n\n"
        "Answer the question from the documents above, then list the supporting sentences by "
        "document number and sentence number."
    )


def closed_book_user(item: EvalItem) -> str:
    return f"=== QUESTION ===\n{item.question}\n\nAnswer the question."


def resolve_support(item: EvalItem, refs: Sequence[SentenceRef]) -> list[SupportRef]:
    """Turn the model's (document, sentence) numbers back into (title, index) pairs.

    A document number outside the range printed for this item becomes `INVALID_TITLE`, which
    matches no gold pair and so is scored as a false positive. A sentence number outside its
    document is kept as written for the same reason: both are wrong answers, not parse errors.
    """
    resolved: list[SupportRef] = []
    for ref in refs:
        if 1 <= ref.document <= len(item.context):
            resolved.append((item.context[ref.document - 1].title, ref.sentence))
        else:
            resolved.append((INVALID_TITLE, ref.sentence))
    return resolved
