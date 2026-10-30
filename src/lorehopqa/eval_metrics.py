"""The official HotpotQA scorer, ported unchanged.

`hotpot_evaluate_v1.py` is distributed as a script that reads two files and prints a dict; the
experiments need the same arithmetic per item, so the per-item half is lifted here verbatim and
the aggregation is rebuilt around it. Every formula below — the answer normalisation, the
yes/no/noanswer short circuit in `f1_score`, the supporting-fact set arithmetic, and the
multiplicative joint score — is the official one. Changing any of them would make the numbers
incomparable with every HotpotQA result ever published, which is the whole reason for
reproducing rather than improving it.

The one addition is `Tally`, which accumulates per-item scores so a run can be broken down by
subset, question family or hop count without re-scoring.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SupportRef = tuple[str, int]

# The three strings the official scorer treats as categorical rather than as token bags. A
# predicted "yes" against a gold "no" shares no tokens anyway, but a gold answer of "no" against
# a prediction of "no answer" would otherwise score partial credit on the token overlap.
_CATEGORICAL = ("yes", "no", "noanswer")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace. Official, unchanged."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-overlap F1, precision, recall over normalised answers."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero = (0.0, 0.0, 0.0)
    # A categorical answer is right or wrong; it never earns partial credit from token overlap.
    if normalized_prediction in _CATEGORICAL and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in _CATEGORICAL and normalized_prediction != normalized_ground_truth:
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


def sp_score(
    prediction: Iterable[SupportRef], ground_truth: Iterable[SupportRef]
) -> tuple[float, float, float, float]:
    """Supporting-fact EM, F1, precision, recall over (title, sentence index) sets."""
    predicted = set(prediction)
    gold = set(ground_truth)

    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)

    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    em = 1.0 if false_positive + false_negative == 0 else 0.0
    return em, f1, precision, recall


@dataclass(frozen=True)
class ItemScore:
    """Every official metric for one item, plus the joint scores built from them.

    Joint precision and recall are the *products* of the answer and supporting-fact halves, not
    their averages: the official metric credits an item only to the extent that both were right,
    so a perfect answer with no correct evidence scores zero rather than one half.
    """

    answer_em: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    sp_em: float
    sp_f1: float
    sp_precision: float
    sp_recall: float
    joint_em: float
    joint_f1: float
    joint_precision: float
    joint_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "answer_em": self.answer_em,
            "answer_f1": self.answer_f1,
            "sp_em": self.sp_em,
            "sp_f1": self.sp_f1,
            "joint_em": self.joint_em,
            "joint_f1": self.joint_f1,
        }


def score_item(
    predicted_answer: str,
    gold_answer: str,
    predicted_support: Iterable[SupportRef] | None = None,
    gold_support: Iterable[SupportRef] | None = None,
) -> ItemScore:
    """Score one prediction.

    `predicted_support=None` is the closed-book case: no documents were shown, so no supporting
    set could be named. The supporting-fact and joint metrics are then zero by construction —
    which is the honest reading, and why E1 reports answer metrics alone.
    """
    answer_em = exact_match_score(predicted_answer, gold_answer)
    answer_f1, answer_precision, answer_recall = f1_score(predicted_answer, gold_answer)

    if predicted_support is None or gold_support is None:
        sp_em = sp_f1 = sp_precision = sp_recall = 0.0
    else:
        sp_em, sp_f1, sp_precision, sp_recall = sp_score(predicted_support, gold_support)

    joint_precision = answer_precision * sp_precision
    joint_recall = answer_recall * sp_recall
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall > 0
        else 0.0
    )

    return ItemScore(
        answer_em=answer_em,
        answer_f1=answer_f1,
        answer_precision=answer_precision,
        answer_recall=answer_recall,
        sp_em=sp_em,
        sp_f1=sp_f1,
        sp_precision=sp_precision,
        sp_recall=sp_recall,
        joint_em=answer_em * sp_em,
        joint_f1=joint_f1,
        joint_precision=joint_precision,
        joint_recall=joint_recall,
    )


_FIELDS = (
    "answer_em",
    "answer_f1",
    "answer_precision",
    "answer_recall",
    "sp_em",
    "sp_f1",
    "sp_precision",
    "sp_recall",
    "joint_em",
    "joint_f1",
    "joint_precision",
    "joint_recall",
)


@dataclass
class Tally:
    """Running means over a set of items, so one pass can serve many breakdowns."""

    count: int = 0
    totals: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_FIELDS, 0.0))

    def add(self, score: ItemScore) -> None:
        self.count += 1
        for name in _FIELDS:
            self.totals[name] += getattr(score, name)

    def means(self) -> dict[str, float]:
        if self.count == 0:
            return {"n": 0}
        return {"n": self.count, **{name: self.totals[name] / self.count for name in _FIELDS}}


def aggregate(scores: Sequence[ItemScore]) -> dict[str, float]:
    tally = Tally()
    for score in scores:
        tally.add(score)
    return tally.means()


def breakdown(
    rows: Sequence[dict[str, Any]], *, key: str, score_key: str = "score"
) -> dict[str, dict[str, float]]:
    """Means grouped by one field of the prediction rows, e.g. `origin` or `type`."""
    tallies: dict[str, Tally] = {}
    for row in rows:
        group = str(row.get(key))
        tallies.setdefault(group, Tally()).add(row[score_key])
    return {group: tally.means() for group, tally in sorted(tallies.items())}
