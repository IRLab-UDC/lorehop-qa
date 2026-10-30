"""Shared machinery for the demonstration experiments E1-E4.

The four `scripts/experiment_*.py` entry points differ only in which items they build and which
condition they put those items under. Everything else — loading the reader panel, laying out the
run directory, submitting a sweep to one loaded model, parsing and scoring the replies, and
resuming a job that stopped halfway — is here, so that the closed-book floor and the conditions
it calibrates cannot drift apart through a copy-and-edit.

RUN LAYOUT.  Under the output directory each experiment owns a tree:

    <exp>/tasks/<dataset>.jsonl          the items, exactly as every reader will see them
    <exp>/predictions/<dataset>.<reader>.jsonl   one row per item, raw reply and parsed score
    <exp>/metrics/<dataset>.<reader>.json        the aggregates and breakdowns for that pair

`tasks` is written once and re-read by every reader. That is the point: a condition that
sampled its distractors afresh per reader would give each model a different collection, and the
per-item pairing that E1 and E2 rest on would be worthless. It also makes the ablation variants
themselves an inspectable artefact rather than a side effect of a run.

RESUMING.  Predictions are appended, and a re-run skips ids already present. A job that hits the
wall clock loses at most one checkpoint chunk, and a reader added to the panel later costs only
its own sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runner import ChatRequest, VLLMModelConfig, VLLMRunner, parse_structured
from .eval_metrics import ItemScore, Tally, breakdown, score_item
from .reader import (
    CLOSED_BOOK_SYSTEM,
    READER_SYSTEM,
    ClosedBookAnswer,
    DistractorPool,
    EvalItem,
    ReaderAnswer,
    closed_book_user,
    reader_user,
    resolve_support,
)


# --------------------------------------------------------------------------------------
# Reader panel
# --------------------------------------------------------------------------------------


# Marks a `model:` in the panel that has not been filled in yet. A fine-tuned checkpoint lives
# wherever it was trained, which this repository cannot know, so E4's panel ships with the paths
# unset and refuses to load one rather than handing vLLM a nonexistent directory an hour into a
# queue wait.
PLACEHOLDER = "/PATH/TO/"


class ReaderModelConfig(VLLMModelConfig):
    """A reader in the panel: a vLLM model plus the two facts the fairness analysis needs.

    `pool` records whether the family served as a generator or juror during construction, which
    is the axis R6 is about. `parameters_b` is carried so that a pool/control comparison can be
    reported at matched scale rather than across whatever sizes happened to be convenient.
    """

    pool: bool = Field(description="True if this family generated or judged during construction.")
    parameters_b: float | None = Field(
        default=None, description="Parameter count in billions, for scale-matched comparisons."
    )
    family: str | None = Field(default=None, description="Display name for the tables.")
    checkpoint: str = Field(
        default="base",
        description="'base' or 'fine-tuned'. E4 reports the two side by side for one family.",
    )
    base_reader: str | None = Field(
        default=None,
        description="For a fine-tuned entry, the name of the base reader it was trained from.",
    )

    @property
    def label(self) -> str:
        return self.family or self.name


class ReadersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readers: list[ReaderModelConfig]
    checkpoint_every: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def validate_readers(self) -> ReadersConfig:
        if not self.readers:
            raise ValueError("readers must not be empty")
        names = [reader.name for reader in self.readers]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate reader names: {sorted(duplicates)}")
        return self

    def get(self, name: str) -> ReaderModelConfig:
        for reader in self.readers:
            if reader.name == name:
                if reader.model.startswith(PLACEHOLDER):
                    raise SystemExit(
                        f"Reader {name!r} still points at a placeholder path ({reader.model}). "
                        "Fill in the checkpoint location in the panel YAML before running it."
                    )
                return reader
        available = ", ".join(reader.name for reader in self.readers)
        raise SystemExit(f"No reader named {name!r}. Available: {available}")


def load_readers_config(path: Path) -> ReadersConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ReadersConfig.model_validate(payload)


# --------------------------------------------------------------------------------------
# Task and prediction storage
# --------------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write atomically, and survive another job writing the same file at the same moment.

    Task lists are shared across readers, and the launcher submits every reader for an experiment
    at once, so several jobs routinely reach an unbuilt task list together. A fixed `.tmp` name
    made them collide: the first to finish renamed the shared temporary away and the others died
    on a missing file, after the queue wait and before touching a GPU. The temporary is therefore
    per-process, and losing the race is not an error — the content is a deterministic function of
    the split and the seed, so whichever copy lands is the same collection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as handle:
            for row in rows:
                handle.write(orjson.dumps(row))
                handle.write(b"\n")
        tmp.replace(path)
    except OSError:
        if not path.exists():
            raise
        print(f"[tasks] {path.name}: another job wrote it first, using that copy", flush=True)
    finally:
        tmp.unlink(missing_ok=True)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def item_to_row(item: EvalItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "question": item.question,
        "answer": item.answer,
        "type": item.type,
        "level": item.level,
        "context": [[document.title, list(document.sentences)] for document in item.context],
        "supporting_facts": [list(fact) for fact in item.supporting_facts],
        "meta": item.meta,
    }


def row_to_item(row: dict[str, Any]) -> EvalItem:
    from .reader import Document

    return EvalItem(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        type=row["type"],
        level=row.get("level"),
        context=tuple(
            Document(title=entry[0], sentences=tuple(entry[1])) for entry in row["context"]
        ),
        supporting_facts=tuple((fact[0], int(fact[1])) for fact in row["supporting_facts"]),
        meta=row.get("meta", {}),
    )


@dataclass
class TaskSet:
    """The frozen item list for one dataset under one condition."""

    name: str
    path: Path
    items: list[EvalItem]

    @classmethod
    def materialise(
        cls, name: str, path: Path, build: Callable[[], list[EvalItem]], *, rebuild: bool = False
    ) -> TaskSet:
        """Load the task list, building and writing it the first time.

        Deliberately not rebuilt on every run: the ablation variants and the padded layouts are
        sampled, and a second sampling would silently break the pairing between readers that have
        already run and readers that have not. `--rebuild` is the explicit way to change them.
        """
        if path.exists() and not rebuild:
            items = [row_to_item(row) for row in read_jsonl(path)]
            print(f"[tasks] {name}: {len(items)} items (existing {path})", flush=True)
            return cls(name=name, path=path, items=items)

        items = build()
        write_jsonl(path, [item_to_row(item) for item in items])
        print(f"[tasks] {name}: {len(items)} items (built -> {path})", flush=True)
        return cls(name=name, path=path, items=items)


class PredictionStore:
    """Append-only predictions for one (dataset, reader) pair, with resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = {row["id"] for row in read_jsonl(path)}

    def append(self, rows: Sequence[dict[str, Any]]) -> None:
        with self.path.open("ab") as handle:
            for row in rows:
                handle.write(orjson.dumps(row))
                handle.write(b"\n")
                self.done.add(row["id"])

    def rows(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path))


# --------------------------------------------------------------------------------------
# Running one condition against one loaded model
# --------------------------------------------------------------------------------------

OPEN_BOOK = "open_book"
CLOSED_BOOK = "closed_book"


def build_request(item: EvalItem, mode: str) -> ChatRequest:
    if mode == CLOSED_BOOK:
        return ChatRequest(
            system=CLOSED_BOOK_SYSTEM,
            user=closed_book_user(item),
            output_model=ClosedBookAnswer,
            temperature=0.0,
        )
    return ChatRequest(
        system=READER_SYSTEM,
        user=reader_user(item),
        output_model=ReaderAnswer,
        temperature=0.0,
    )


def _best_answer_score(predicted: str, item: EvalItem):
    """Score against the best-matching gold answer.

    A relational query can have several correct fillers, and PhantomWiki records all of them.
    Scoring against one arbitrarily would mark a correct reply wrong whenever the model named a
    different valid entity, which is the convention multi-answer question answering exists to
    avoid. Collections with a single answer are unaffected: the list is then one element.
    """
    from .eval_metrics import score_item as _score

    golds = item.meta.get("answers") or [item.answer]
    return max((_score(predicted, gold) for gold in golds), key=lambda s: (s.answer_f1, s.answer_em))


def score_reply(item: EvalItem, text: str, mode: str) -> dict[str, Any]:
    """Parse one reply and score it. An unparsable reply is a wrong answer, not a dropped item.

    Grammar-constrained decoding makes a parse failure rare — it means the reply hit the token
    budget mid-object — but silently discarding those would flatter whichever model truncates
    most, so they are scored as an empty answer with no supporting facts.
    """
    if mode == CLOSED_BOOK:
        parsed, error = parse_structured(text, ClosedBookAnswer)
        predicted_answer = parsed.answer if parsed else ""
        score = _best_answer_score(predicted_answer, item)
        return {
            "id": item.id,
            "predicted_answer": predicted_answer,
            "predicted_support": None,
            "parse_error": error,
            **score.as_dict(),
        }

    parsed, error = parse_structured(text, ReaderAnswer)
    predicted_answer = parsed.answer if parsed else ""
    predicted_support = resolve_support(item, parsed.supporting_facts) if parsed else []
    score = score_item(
        predicted_answer, item.answer, predicted_support, item.supporting_facts
    )
    return {
        "id": item.id,
        "predicted_answer": predicted_answer,
        "predicted_support": [list(pair) for pair in predicted_support],
        "parse_error": error,
        **score.as_dict(),
    }


def run_sweep(
    runner: VLLMRunner,
    tasks: TaskSet,
    store: PredictionStore,
    *,
    mode: str,
    checkpoint_every: int,
    progress: bool = True,
) -> int:
    """Answer every item in `tasks` not already in `store`, checkpointing as it goes."""
    pending = [item for item in tasks.items if item.id not in store.done]
    if not pending:
        print(f"[run] {tasks.name}: nothing to do ({len(store.done)} already recorded)", flush=True)
        return 0

    print(
        f"[run] {tasks.name}: {len(pending)} pending of {len(tasks.items)} "
        f"({len(store.done)} recorded)",
        flush=True,
    )

    started = time.monotonic()
    written = 0
    for offset in range(0, len(pending), checkpoint_every):
        chunk = pending[offset : offset + checkpoint_every]
        generations = runner.generate(
            [build_request(item, mode) for item in chunk], progress=progress
        )
        rows = []
        for item, generation in zip(chunk, generations, strict=True):
            row = score_reply(item, generation.text, mode)
            row.update(
                {
                    "truncated": generation.truncated,
                    "prompt_too_long": generation.finish_reason == "prompt_too_long",
                    "gold_answer": item.answer,
                    "type": item.type,
                    "level": item.level,
                    "origin": item.meta.get("origin"),
                    "source_id": item.meta.get("source_id"),
                    "generator_model": item.meta.get("generator_model"),
                    "documents": item.meta.get("documents"),
                    "supporting_documents": item.meta.get("supporting_documents"),
                    "full_support": item.meta.get("full_support"),
                    "gold_retrieved": item.meta.get("gold_retrieved"),
                    "gold_documents": item.meta.get("gold_documents"),
                    "is_yes_no": item.is_yes_no,
                    "answer_in_question": item.answer_in_question,
                    "ablation_of": item.meta.get("ablation_of"),
                    "withheld_title": item.meta.get("withheld_title"),
                }
            )
            rows.append(row)
        store.append(rows)
        written += len(rows)
        elapsed = time.monotonic() - started
        rate = written / elapsed if elapsed > 0 else 0.0
        print(
            f"[run] {tasks.name}: {written}/{len(pending)} "
            f"({rate:.1f} items/s, {elapsed / 60:.1f} min)",
            flush=True,
        )
    return written


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

_SCORE_FIELDS = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")


def _score_of(row: dict[str, Any]) -> ItemScore:
    return ItemScore(
        answer_em=row["answer_em"],
        answer_f1=row["answer_f1"],
        answer_precision=0.0,
        answer_recall=0.0,
        sp_em=row["sp_em"],
        sp_f1=row["sp_f1"],
        sp_precision=0.0,
        sp_recall=0.0,
        joint_em=row["joint_em"],
        joint_f1=row["joint_f1"],
        joint_precision=0.0,
        joint_recall=0.0,
    )


def summarise(rows: Sequence[dict[str, Any]], *, axes: Sequence[str] = ()) -> dict[str, Any]:
    """Overall means, the same means with yes/no items removed, and the requested breakdowns.

    Yes/no items are reported separately in every table because they can be answered above chance
    with no evidence at all: a closed-book score of zero is not the expected result even on a
    wholly invented collection, and a number that mixes the two cannot be read against a floor.
    """
    scored = [{**row, "score": _score_of(row)} for row in rows]

    overall = Tally()
    without_yes_no = Tally()
    without_guessable = Tally()
    for row in scored:
        overall.add(row["score"])
        if not row.get("is_yes_no"):
            without_yes_no.add(row["score"])
        # The full guessing floor: an item is guessable if the answer is one of two words the
        # question itself offers (yes/no) or a string the question already contains (a comparison
        # naming every candidate). Both can be answered above chance with no evidence at all.
        if not row.get("is_yes_no") and not row.get("answer_in_question"):
            without_guessable.add(row["score"])

    summary: dict[str, Any] = {
        "overall": overall.means(),
        "excluding_yes_no": without_yes_no.means(),
        "excluding_guessable": without_guessable.means(),
        "guessable_share": (
            sum(1 for row in rows if row.get("is_yes_no") or row.get("answer_in_question"))
            / len(rows)
            if rows
            else 0.0
        ),
        "parse_failures": sum(1 for row in rows if row.get("parse_error")),
        "truncated": sum(1 for row in rows if row.get("truncated")),
        "prompt_too_long": sum(1 for row in rows if row.get("prompt_too_long")),
    }
    for axis in axes:
        summary[f"by_{axis}"] = breakdown(scored, key=axis)
    return summary


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"[metrics] -> {path}", flush=True)


# --------------------------------------------------------------------------------------
# CLI plumbing shared by the experiment scripts
# --------------------------------------------------------------------------------------

DEFAULT_AXES = ("origin", "type", "level", "generator_model", "supporting_documents")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Reader panel YAML.")
    parser.add_argument("--model", type=str, required=True, help="Reader handle from the panel.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run tree root.")
    parser.add_argument(
        "--dist-dir", type=Path, required=True, help="Directory holding lorehopqa/ and lorehopqa-private/."
    )
    parser.add_argument("--hotpot", type=Path, default=None, help="HotpotQA dev distractor JSON.")
    parser.add_argument("--lore-dir", type=Path, default=None, help="Directory of lore .txt files.")
    parser.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N items only.")
    parser.add_argument("--seed", type=int, default=20260901, help="Seed for sampled layouts.")
    parser.add_argument("--rebuild-tasks", action="store_true", help="Re-sample the task lists.")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--metrics-only", action="store_true", help="Score what exists, load nothing.")
    # Several experiments discover their datasets by globbing the task directory, so a task list
    # built for a different purpose (a ranking measured for retrieval quality alone) would be
    # swept end-to-end by every reader as soon as it appeared. --only restricts the run to the
    # datasets actually wanted, without hiding the other task lists from the tree.
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="GLOB",
        help="Run only datasets matching this glob; repeatable.",
    )


@dataclass
class Paths:
    """Where an experiment's inputs live and where its outputs go."""

    root: Path
    dist: Path
    hotpot: Path | None
    lore_dir: Path
    split: str

    @property
    def lorehopqa(self) -> Path:
        return self.dist / "lorehopqa" / f"{self.split}.jsonl"

    @property
    def provenance(self) -> Path:
        return self.dist / "lorehopqa-private" / f"{self.split}.provenance.jsonl"

    # Everything an experiment writes hangs off <root>/<split>/. The split has to be in the path
    # rather than only in the file contents: a 64-item smoke run on validation writes the same
    # dataset names as the real run on test, and because task lists are deliberately reused
    # across readers, a shared directory would have silently handed the full sweep the smoke
    # test's truncated collection.
    @property
    def base(self) -> Path:
        return self.root / self.split

    def tasks(self, name: str) -> Path:
        return self.base / "tasks" / f"{name}.jsonl"

    def predictions(self, name: str, reader: str) -> Path:
        return self.base / "predictions" / f"{name}.{reader}.jsonl"

    def metrics(self, name: str, reader: str) -> Path:
        return self.base / "metrics" / f"{name}.{reader}.json"


def resolve_paths(args: argparse.Namespace, experiment: str) -> Paths:
    repo = Path(__file__).resolve().parent.parent.parent
    lore_dir = args.lore_dir or repo / "examples" / "lores"
    paths = Paths(
        root=args.output_dir / experiment,
        dist=args.dist_dir,
        hotpot=args.hotpot,
        lore_dir=lore_dir,
        split=args.split,
    )
    if not paths.lorehopqa.exists():
        raise SystemExit(f"Missing required input: {paths.lorehopqa}")
    if args.hotpot is not None and not args.hotpot.exists():
        raise SystemExit(f"Missing HotpotQA input: {args.hotpot}")
    # The provenance sidecar and the world specifications are not part of the public release.
    # Without them the items still load and score; the per-origin and per-generator breakdowns
    # are unavailable, and the experiments that draw same-world replacement documents cannot be
    # reproduced. Each caller that needs one checks for it, so experiments that do not are not
    # held back by its absence.
    paths.base.mkdir(parents=True, exist_ok=True)
    return paths


def require_withheld(paths: "Paths", *, lore: bool = False) -> None:
    """Fail early, and legibly, in the experiments that need material the release withholds."""
    missing = [str(paths.provenance)] if not paths.provenance.exists() else []
    if lore and not paths.lore_dir.is_dir():
        missing.append(str(paths.lore_dir))
    if missing:
        raise SystemExit(
            "This experiment needs inputs that are not part of the public release:\n  "
            + "\n  ".join(missing)
            + "\nThey are retained by the authors; see the README."
        )


def limited(items: list[EvalItem], limit: int | None) -> list[EvalItem]:
    return items if limit is None else items[:limit]


def rng_for(seed: int, *parts: str) -> random.Random:
    """A generator whose stream depends only on the seed and the names given.

    Sampled layouts have to be reproducible from the seed alone — not from the order in which
    datasets happened to be built — or a rerun that skips one dataset would resample another.
    """
    return random.Random(f"{seed}:{'|'.join(parts)}")


def run_experiment(
    args: argparse.Namespace,
    *,
    experiment: str,
    datasets: Callable[[Paths, argparse.Namespace], dict[str, tuple[TaskSet, str]]],
    axes: Sequence[str] = DEFAULT_AXES,
) -> int:
    """Load one reader, sweep it over every dataset the experiment defines, and score.

    The model is loaded once and reused across datasets. Loading is the expensive part of a
    short sweep — a 70B checkpoint takes minutes — so an experiment that puts one reader through
    four conditions must not pay for it four times.
    """
    config = load_readers_config(args.config)
    reader = config.get(args.model)
    paths = resolve_paths(args, experiment)

    built = datasets(paths, args)
    if not built:
        raise SystemExit(f"{experiment}: no datasets to run")

    patterns = getattr(args, "only", None)
    if patterns:
        kept = {n: v for n, v in built.items() if any(fnmatch(n, p) for p in patterns)}
        if not kept:
            raise SystemExit(
                f"{experiment}: --only {' '.join(patterns)} matched none of: "
                + ", ".join(sorted(built))
            )
        dropped = sorted(set(built) - set(kept))
        if dropped:
            print(f"[run] --only: skipping {', '.join(dropped)}", flush=True)
        built = kept

    runner: VLLMRunner | None = None
    for name, (tasks, mode) in built.items():
        store = PredictionStore(paths.predictions(name, reader.name))
        pending = [item for item in tasks.items if item.id not in store.done]

        if pending and not args.metrics_only:
            if runner is None:
                runner = VLLMRunner(reader)
            run_sweep(
                runner,
                tasks,
                store,
                mode=mode,
                checkpoint_every=config.checkpoint_every,
                progress=args.progress,
            )
        elif pending:
            print(f"[run] {name}: {len(pending)} pending, skipped (--metrics-only)", flush=True)

        rows = store.rows()
        if not rows:
            continue
        write_metrics(
            paths.metrics(name, reader.name),
            {
                "experiment": experiment,
                "dataset": name,
                "mode": mode,
                "reader": reader.name,
                "reader_model": reader.model,
                "reader_label": reader.label,
                "pool": reader.pool,
                "parameters_b": reader.parameters_b,
                "checkpoint": reader.checkpoint,
                "base_reader": reader.base_reader,
                "split": args.split,
                "items": len(tasks.items),
                "scored": len(rows),
                **summarise(rows, axes=axes),
            },
        )

    return 0


def main_guard(entry: Callable[[], int]) -> None:
    try:
        sys.exit(entry())
    except KeyboardInterrupt:
        sys.exit(130)


def stratified_sample(
    items: Sequence[EvalItem], *, size: int, keys: Sequence[str], rng: random.Random
) -> list[EvalItem]:
    """A subsample that preserves the composition of the split along `keys`.

    E2 expands every item into one variant per supporting document, so running it over the whole
    test split costs about 2.3 times what a full-context sweep does, for a quantity — the drop
    from full context to leave-one-out — that a few thousand items already pin down. What a plain
    random subsample would put at risk is the breakdowns: convergent items are 12% of the split
    and easy items 3%, and a stratum that small drifts noticeably under simple sampling. Strata
    are allocated their exact share and the remainder goes to the largest ones, so every reported
    cell keeps its proportion.
    """
    if size >= len(items):
        return list(items)

    strata: dict[tuple[Any, ...], list[EvalItem]] = {}
    for item in items:
        signature = tuple(
            item.meta.get(key) if key in item.meta else getattr(item, key, None) for key in keys
        )
        strata.setdefault(signature, []).append(item)

    total = len(items)
    quotas: dict[tuple[Any, ...], int] = {}
    for signature, members in strata.items():
        quotas[signature] = int(len(members) * size / total)

    # Integer truncation leaves a shortfall; hand it to the largest strata, which are the ones
    # that lost the most to rounding.
    shortfall = size - sum(quotas.values())
    for signature in sorted(strata, key=lambda key: -len(strata[key]))[: max(shortfall, 0)]:
        quotas[signature] += 1

    sampled: list[EvalItem] = []
    for signature, members in strata.items():
        take = min(quotas[signature], len(members))
        if take <= 0:
            continue
        ordered = sorted(members, key=lambda item: item.id)
        rng.shuffle(ordered)
        sampled.extend(ordered[:take])

    sampled.sort(key=lambda item: item.id)
    return sampled


def build_ablations(
    items: Sequence[EvalItem],
    *,
    pool: DistractorPool,
    worlds: dict[str, str],
    seed: int,
    tag: str,
) -> list[EvalItem]:
    """One leave-one-out variant per supporting document, across every item given.

    An item with three supporting documents yields three variants, each missing a different one.
    Items whose replacement could not be drawn are skipped rather than shortened, and the skip is
    reported: a variant with one document fewer than its siblings would answer a different
    question about layout than the one E2 asks.
    """
    from .reader import leave_one_out

    variants: list[EvalItem] = []
    skipped = 0
    for item in items:
        world = worlds.get(item.id, item.meta.get("collection", "default"))
        for title in item.supporting_titles:
            variant = leave_one_out(
                item,
                withheld=title,
                pool=pool,
                world=world,
                rng=rng_for(seed, tag, item.id, title),
            )
            if variant is None:
                skipped += 1
                continue
            variants.append(variant)
    if skipped:
        print(f"[tasks] {tag}: {skipped} variants skipped (no replacement available)", flush=True)
    return variants
