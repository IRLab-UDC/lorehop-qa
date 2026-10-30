#!/usr/bin/env python
"""Aggregate the E1-E4 runs into the paper's tables.

Reads the prediction files every experiment wrote, recomputes the aggregates it needs — rather
than trusting the per-run metric files, so that a breakdown added later does not require the
sweeps to be repeated — and emits three things:

    results/summary.json    every number, including the ones that did not make a table
    results/tables.tex      the four table bodies, ready to paste into the manuscript
    stdout                  a readable report

SIGNIFICANCE.  Transposed items and their HotpotQA twins form matched pairs — same skeleton, same
question family, same hop structure, differing only in whether the world is real — so differences
across a pair are tested paired. The bootstrap resamples pairs, not items from two independent
populations, and the sign test is carried alongside it as a distribution-free check that does not
assume the per-pair differences are symmetric. Both are run on answer F1, which is the metric the
leakage claim is stated in.

    PYTHONPATH=src python scripts/analyze_experiments.py --output-dir experiments
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import orjson  # noqa: E402

from lorehopqa.eval_metrics import ItemScore, Tally, normalize_answer  # noqa: E402

EXPERIMENTS = {
    "e1_closed_book": "E1",
    "e2_ablated_evidence": "E2",
    "e3_full_context": "E3",
    "e4_transfer": "E4",
    "e5_bm25_retrieval": "E5",
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predictions(
    root: Path, split: str, *, warn_partial: bool = True
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    """Every prediction file for one split, keyed by experiment then (dataset, reader).

    Prediction files are appended as a sweep runs, so a file belonging to a job still in flight
    is a valid prefix of the finished one and scores perfectly happily --- at whatever the reader
    had reached. Every incomplete file is therefore reported against the length of its task list,
    loudly, because a partial sweep read as a finished one produces a number that looks exactly
    like a result.
    """
    loaded: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    incomplete: list[str] = []
    for experiment in EXPERIMENTS:
        directory = root / experiment / split / "predictions"
        if not directory.is_dir():
            continue
        tasks_dir = root / experiment / split / "tasks"
        expected = {
            path.stem: sum(1 for line in path.open() if line.strip())
            for path in tasks_dir.glob("*.jsonl")
        } if tasks_dir.is_dir() else {}
        bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for path in sorted(directory.glob("*.jsonl")):
            # <dataset>.<reader>.jsonl — the reader handle is the last dotted field, because a
            # dataset name may itself contain a dash but never a dot.
            stem = path.stem
            dataset, _, reader = stem.rpartition(".")
            if not dataset:
                continue
            rows = read_jsonl(path)
            bucket[(dataset, reader)] = rows
            total = expected.get(dataset)
            if total and len(rows) < total:
                incomplete.append(
                    f"{experiment}/{dataset}.{reader}: {len(rows)}/{total} "
                    f"({len(rows) / total:.0%})"
                )
        if bucket:
            loaded[experiment] = bucket

    if incomplete and warn_partial:
        print(
            f"WARNING: {len(incomplete)} prediction file(s) are incomplete; the numbers below "
            "include partial sweeps and are not final:",
            file=sys.stderr,
        )
        for line in incomplete:
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)
    return loaded


def score_of(row: dict[str, Any]) -> ItemScore:
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


def means(
    rows: Iterable[dict[str, Any]],
    *,
    exclude_yes_no: bool = False,
    exclude_guessable: bool = False,
) -> dict[str, float]:
    """Means over the rows, optionally dropping the items a reader can guess without evidence.

    `exclude_guessable` drops yes/no items and comparison items whose question already contains
    the answer. Those are the two shapes that score above chance closed-book on any collection,
    invented or not, so a floor computed over them measures arithmetic rather than leakage.
    """
    tally = Tally()
    for row in rows:
        if exclude_yes_no and row.get("is_yes_no"):
            continue
        if exclude_guessable and (row.get("is_yes_no") or row.get("answer_in_question")):
            continue
        tally.add(score_of(row))
    return tally.means()


def spacing_only_misses(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Predictions that are the gold answer with the word boundaries in different places.

    The official scorer tokenises on whitespace, so "SluisTerrier" and "Sluis Terrier" share no
    tokens and score zero F1. On real-world data this is rare, because the model has seen the
    entity written and copies the spelling; on an invented world it is a live failure mode, and
    it falls entirely on our side of the comparison. That direction matters: it depresses
    closed-book scores on LoreHopQA and therefore *widens* the leakage gap we report, so it is a bias
    in favour of our own claim and is quantified rather than left implicit.
    """
    misses = 0
    for row in rows:
        predicted = normalize_answer(row.get("predicted_answer") or "")
        gold = normalize_answer(row.get("gold_answer") or "")
        if predicted and gold and predicted != gold:
            if predicted.replace(" ", "") == gold.replace(" ", ""):
                misses += 1
    return {"count": misses, "share": misses / len(rows) if rows else 0.0}


def where(rows: Sequence[dict[str, Any]], **conditions: Any) -> list[dict[str, Any]]:
    return [
        row for row in rows if all(row.get(key) == value for key, value in conditions.items())
    ]


# --------------------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------------------


def paired_bootstrap(
    differences: Sequence[float], *, resamples: int, seed: int = 20260901
) -> dict[str, float]:
    """Percentile confidence interval for the mean of the per-pair differences."""
    if not differences:
        return {}
    rng = random.Random(seed)
    size = len(differences)
    observed = statistics.fmean(differences)
    draws = []
    for _ in range(resamples):
        draws.append(statistics.fmean(rng.choices(differences, k=size)))
    draws.sort()
    return {
        "mean_difference": observed,
        "ci_low": draws[int(0.025 * resamples)],
        "ci_high": draws[min(int(0.975 * resamples), resamples - 1)],
        "resamples": resamples,
        "pairs": size,
    }


def sign_test(differences: Sequence[float]) -> dict[str, float]:
    """Two-sided sign test on the non-zero differences.

    A normal approximation with a continuity correction, not the exact binomial: with several
    thousand pairs the two agree far beyond the precision anything here is reported to, and the
    exact tail would cost a sum over thousands of binomial coefficients per comparison.
    """
    positive = sum(1 for value in differences if value > 0)
    negative = sum(1 for value in differences if value < 0)
    trials = positive + negative
    if trials == 0:
        return {"positive": 0, "negative": 0, "p_value": 1.0}

    # z = (|k - n/2| - 1/2) / sqrt(n/4)
    z = (abs(positive - trials / 2) - 0.5) / math.sqrt(trials / 4)
    p_value = math.erfc(z / math.sqrt(2))
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - trials,
        "z": z,
        "p_value": min(p_value, 1.0),
    }


def pair_by_source(
    lore_rows: Sequence[dict[str, Any]], hotpot_rows: Sequence[dict[str, Any]], field: str
) -> list[float]:
    """Per-pair (HotpotQA minus \\ds) differences on one metric, joined on the source item."""
    hotpot = {row["id"]: row for row in hotpot_rows}
    differences = []
    for row in lore_rows:
        source = row.get("source_id")
        twin = hotpot.get(source) if source else None
        if twin is not None:
            differences.append(twin[field] - row[field])
    return differences


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------


def analyse_e1(
    predictions: dict[tuple[str, str], list[dict[str, Any]]],
    panel: dict[str, dict[str, Any]],
    *,
    resamples: int,
) -> dict[str, Any]:
    """Closed-book quality per reader, and the leakage gap against the paired HotpotQA items."""
    readers = sorted({reader for _, reader in predictions})
    report: dict[str, Any] = {"readers": {}}

    for reader in readers:
        lore = predictions.get(("lorehopqa", reader), [])
        hotpot = predictions.get(("hotpot-paired", reader), [])
        if not lore:
            continue

        transposed = where(lore, origin="transposed")
        entry: dict[str, Any] = {
            "all": means(lore),
            "all_no_yesno": means(lore, exclude_yes_no=True),
            "all_no_guessable": means(lore, exclude_guessable=True),
            "transposed": means(transposed),
            "transposed_no_guessable": means(transposed, exclude_guessable=True),
            "novel": means(where(lore, origin="novel")),
            "novel_no_guessable": means(where(lore, origin="novel"), exclude_guessable=True),
            "hotpot_paired": means(hotpot) if hotpot else {},
            "hotpot_paired_no_yesno": means(hotpot, exclude_yes_no=True) if hotpot else {},
            "hotpot_paired_no_guessable": means(hotpot, exclude_guessable=True) if hotpot else {},
            "guessable_share": (
                sum(1 for row in lore if row.get("is_yes_no") or row.get("answer_in_question"))
                / len(lore)
            ),
            "spacing_only_misses": spacing_only_misses(lore),
            "hotpot_spacing_only_misses": spacing_only_misses(hotpot) if hotpot else {},
        }

        if hotpot:
            # The leakage estimate: same skeletons, one real world and one invented.
            differences = pair_by_source(transposed, hotpot, "answer_f1")
            entry["leakage_gap_f1"] = {
                **paired_bootstrap(differences, resamples=resamples),
                **sign_test(differences),
            }
            em_differences = pair_by_source(transposed, hotpot, "answer_em")
            entry["leakage_gap_em"] = paired_bootstrap(em_differences, resamples=resamples)

            # The same pairing restricted to items neither side can guess. This is the number the
            # leakage claim rests on: on guessable items both collections hand the reader free
            # credit, and including them measures question shape as much as memory.
            guessable = {
                row["id"]
                for row in hotpot
                if row.get("is_yes_no") or row.get("answer_in_question")
            }
            hard_lore = [
                row
                for row in transposed
                if not (row.get("is_yes_no") or row.get("answer_in_question"))
                and row.get("source_id") not in guessable
            ]
            hard_differences = pair_by_source(hard_lore, hotpot, "answer_f1")
            entry["leakage_gap_f1_no_guessable"] = {
                **paired_bootstrap(hard_differences, resamples=resamples),
                **sign_test(hard_differences),
            }

        # The fairness decomposition. Closed-book is the sharpest test of R6 because the context
        # is absent: a reader scoring above floor on items its own family wrote is exhibiting
        # memory of authorship, which is exactly what the rotation exists to remove.
        by_generator: dict[str, list[dict[str, Any]]] = {}
        for row in lore:
            generator = row.get("generator_model")
            if generator:
                by_generator.setdefault(generator, []).append(row)
        entry["by_generator"] = {
            generator: means(rows) for generator, rows in sorted(by_generator.items())
        }
        by_generator_hard: dict[str, list[dict[str, Any]]] = {}
        for row in lore:
            generator = row.get("generator_model")
            if generator and not (row.get("is_yes_no") or row.get("answer_in_question")):
                by_generator_hard.setdefault(generator, []).append(row)
        entry["by_generator_no_guessable"] = {
            generator: means(rows) for generator, rows in sorted(by_generator_hard.items())
        }
        report["readers"][reader] = entry

    # R6: the closed-book condition is the sharpest test of it, because the context is absent.
    # Reported twice — over everything, and over the items a reader cannot simply guess, where a
    # residual advantage could not be explained by question shape.
    report["authorship"] = authorship_advantage(report["readers"], panel)
    report["authorship_no_guessable"] = authorship_advantage(
        report["readers"], panel, field="by_generator_no_guessable"
    )
    return report


def authorship_advantage(
    readers: dict[str, dict[str, Any]],
    panel: dict[str, dict[str, Any]],
    *,
    field: str = "by_generator",
) -> dict[str, Any]:
    """Does a reader do better on items its own family wrote than other readers do on them?

    The raw generator-by-reader table cannot answer that on its own, because it confounds two
    effects that both move a cell: some readers guess better than others everywhere, and some
    generators write items that are easier to guess for everyone. Reading the diagonal against
    the off-diagonal therefore measures reader strength as much as authorship.

    Each column is standardised across readers first, which removes the item-difficulty effect
    and leaves each cell as "how well this reader did on this generator's items, relative to the
    other readers on the same items". A pool reader's advantage is then its own column's z-score
    minus its mean z-score elsewhere. Under R6 that quantity is indistinguishable from zero; a
    clearly positive value for a family would mean the generator/juror rotation failed to remove
    the advantage of having authored the fiction.
    """
    generators = sorted({g for entry in readers.values() for g in entry.get(field, {})})
    if len(generators) < 2 or len(readers) < 2:
        return {}

    table = {
        reader: {
            generator: entry[field].get(generator, {}).get("answer_f1")
            for generator in generators
        }
        for reader, entry in readers.items()
    }

    # Standardise within a generator column: same items, different readers.
    z: dict[str, dict[str, float]] = {reader: {} for reader in table}
    for generator in generators:
        column = [table[reader][generator] for reader in table if table[reader][generator] is not None]
        if len(column) < 2:
            continue
        mean = statistics.fmean(column)
        spread = statistics.pstdev(column)
        for reader in table:
            value = table[reader][generator]
            if value is not None:
                z[reader][generator] = (value - mean) / spread if spread > 0 else 0.0

    report: dict[str, Any] = {"table": table, "z": z, "advantage": {}}
    diagonal, off_diagonal = [], []
    for reader, scores in z.items():
        own = panel.get(reader, {}).get("model")
        if own not in scores:
            # A control family authored nothing, so it has no diagonal cell. Its spread across
            # generators is still reported, as the reference for what "no advantage" looks like.
            report["advantage"][reader] = {
                "pool": panel.get(reader, {}).get("pool"),
                "own_generator": None,
                "z_spread": max(scores.values()) - min(scores.values()) if scores else None,
            }
            continue
        others = [value for generator, value in scores.items() if generator != own]
        advantage = scores[own] - statistics.fmean(others) if others else None
        report["advantage"][reader] = {
            "pool": panel.get(reader, {}).get("pool"),
            "own_generator": own,
            "z_on_own": scores[own],
            "mean_z_on_others": statistics.fmean(others) if others else None,
            "authorship_advantage": advantage,
            "answer_f1_on_own": table[reader][own],
            "mean_answer_f1_on_others": statistics.fmean(
                [table[reader][g] for g in generators if g != own and table[reader][g] is not None]
            ),
        }
        diagonal.append(table[reader][own])
        off_diagonal.append(
            statistics.fmean([table[reader][g] for g in generators if g != own and table[reader][g] is not None])
        )

    if diagonal:
        report["diagonal_mean_answer_f1"] = statistics.fmean(diagonal)
        report["off_diagonal_mean_answer_f1"] = statistics.fmean(off_diagonal)
        report["diagonal_minus_off_diagonal"] = report["diagonal_mean_answer_f1"] - report["off_diagonal_mean_answer_f1"]
        advantages = [
            entry["authorship_advantage"]
            for entry in report["advantage"].values()
            if entry.get("authorship_advantage") is not None
        ]
        report["mean_authorship_advantage_z"] = statistics.fmean(advantages)
        report["max_authorship_advantage_z"] = max(advantages)
        report["permutation_test"] = authorship_permutation_test(z, panel)
    return report


def authorship_permutation_test(
    z: dict[str, dict[str, float]], panel: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Exact test of whether readers score highest on the items their own family wrote.

    The statistic is the trace of the standardised table: the sum of each pool reader's z-score
    on its own generator's column. Because every column is standardised across readers, its
    entries sum to zero, so under an additive model — readers differ, generators differ, but
    authorship carries no advantage — the expected trace is exactly zero. The null distribution
    is obtained by permuting which reader is said to own which column, which is exact rather than
    asymptotic: with four pool families there are only 24 assignments, and enumerating them is
    cheaper and more honest than a normal approximation on four points.

    A one-sided p-value is reported, because R6 predicts the direction: the concern is a reader
    advantaged on its own family's items, not disadvantaged.
    """
    import itertools

    pool = [reader for reader in z if panel.get(reader, {}).get("model") in z[reader]]
    if len(pool) < 3:
        return {}

    columns = [panel[reader]["model"] for reader in pool]
    observed = sum(z[reader][panel[reader]["model"]] for reader in pool)

    traces = [
        sum(z[reader][column] for reader, column in zip(pool, assignment, strict=True))
        for assignment in itertools.permutations(columns)
    ]
    at_least = sum(1 for value in traces if value >= observed - 1e-12)

    return {
        "statistic": "trace of the column-standardised reader x generator table",
        "observed_trace": observed,
        "permutations": len(traces),
        "mean_null_trace": statistics.fmean(traces),
        "max_null_trace": max(traces),
        "p_value_one_sided": at_least / len(traces),
        "families": pool,
    }


def answer_survives_ablation(tasks_dir: Path, loo_name: str) -> dict[str, bool]:
    """For each leave-one-out variant, is the gold answer still printed in what remains?

    This is the decisive split, and a weaker version of it is misleading. Asking whether the
    *withheld* document carried the answer is not enough: an answer string often appears in more
    than one document — the bridge entity is named by both hops, a title recurs, a date is
    repeated — so removing one copy leaves the reader able to read the answer off the context
    without performing any hop at all. Scoring those variants as failed ablations would charge
    the collection for leakage that is really redundancy in the evidence.

    Computed from the ablated contexts themselves rather than inferred, so it reflects exactly
    what each reader was shown. Where the answer is absent from the remaining context, a correct
    reply is either parametric recall or a guess at a categorical answer, and that is the number
    the hop-necessity claim rests on.
    """
    path = tasks_dir / f"{loo_name}.jsonl"
    if not path.exists():
        return {}
    survives: dict[str, bool] = {}
    for row in read_jsonl(path):
        answer = normalize_answer(row["answer"])
        if not answer:
            continue
        text = normalize_answer(
            " ".join(sentence for _, sentences in row["context"] for sentence in sentences)
        )
        survives[row["id"]] = answer in text
    return survives


def analyse_e2(
    predictions: dict[tuple[str, str], list[dict[str, Any]]],
    floors: dict[str, dict[str, float]],
    tasks_dir: Path,
) -> dict[str, Any]:
    """Full context against leave-one-out, per collection and per layout.

    The pairs are (`*-full`, `*-loo`). The drop is what E2 measures, and the floor from E1 is the
    level it should reach on an invented world: a leave-one-out score that stays near full context
    means the withheld evidence was not required.
    """
    layouts = {
        "lore": ("lore-full", "lore-loo"),
        "lore10": ("lore10-full", "lore10-loo"),
        "hotpot": ("hotpot-full", "hotpot-loo"),
        "hotpot4": ("hotpot4-full", "hotpot4-loo"),
    }
    readers = sorted({reader for _, reader in predictions})
    report: dict[str, Any] = {"readers": {}}

    for reader in readers:
        entry: dict[str, Any] = {}
        for layout, (full_name, loo_name) in layouts.items():
            full = predictions.get((full_name, reader), [])
            loo = predictions.get((loo_name, reader), [])
            if not full or not loo:
                continue

            full_means = means(full)
            loo_means = means(loo)
            # Two floors, and picking the wrong one inverts the answer. A HotpotQA layout must be
            # read against the closed-book score on HotpotQA, not on \ds — the whole point is
            # that those differ by 0.2 F1. And a subset restricted to non-guessable items must be
            # read against the floor computed on non-guessable items, or the comparison charges
            # the ablation for guessing credit it was never given.
            collection = "hotpot" if layout.startswith("hotpot") else "lore"
            reader_floors = floors.get(reader, {})
            floor = reader_floors.get(f"{collection}_all")
            cell: dict[str, Any] = {
                "full": full_means,
                "leave_one_out": loo_means,
                "drop_answer_f1": full_means["answer_f1"] - loo_means["answer_f1"],
                "floor_answer_f1": floor,
            }
            if floor is not None:
                # How far the ablated score falls towards the floor. 1.0 means the evidence was
                # strictly required; a small value means the answer survived its removal.
                span = full_means["answer_f1"] - floor
                cell["fraction_of_span_lost"] = (
                    (full_means["answer_f1"] - loo_means["answer_f1"]) / span if span > 0 else None
                )
                cell["residual_above_floor"] = loo_means["answer_f1"] - floor

            # Convergent items need separate treatment: each supporting document contributes one
            # constraint, so removing one leaves the answer under-determined rather than
            # unavailable, and a reader may still land on it by elimination.
            by_type: dict[str, Any] = {}
            for question_type in sorted({row.get("type") for row in loo if row.get("type")}):
                by_type[question_type] = {
                    "full": means(where(full, type=question_type)),
                    "leave_one_out": means(where(loo, type=question_type)),
                }
            cell["by_type"] = by_type

            by_hops: dict[str, Any] = {}
            for hops in sorted({row.get("supporting_documents") for row in loo if row.get("supporting_documents")}):
                by_hops[str(hops)] = {
                    "full": means(where(full, supporting_documents=hops)),
                    "leave_one_out": means(where(loo, supporting_documents=hops)),
                }
            cell["by_supporting_documents"] = by_hops

            # Split the ablation by whether the answer is still printed in what remains.
            survives = answer_survives_ablation(tasks_dir, loo_name)
            if survives:
                still_there = [row for row in loo if survives.get(row["id"]) is True]
                gone = [row for row in loo if survives.get(row["id"]) is False]
                gone_hard = [
                    row for row in gone if not (row.get("is_yes_no") or row.get("answer_in_question"))
                ]
                cell["answer_still_in_context"] = {
                    **means(still_there),
                    "share": len(still_there) / len(loo) if loo else 0.0,
                }
                cell["answer_absent_from_context"] = {
                    **means(gone),
                    "share": len(gone) / len(loo) if loo else 0.0,
                }
                # The strictest reading: the answer is gone from the page and cannot be guessed
                # from the question's own wording. Anything scored here came from the weights.
                cell["answer_absent_not_guessable"] = {
                    **means(gone_hard),
                    "share": len(gone_hard) / len(loo) if loo else 0.0,
                }
                hard_floor = reader_floors.get(f"{collection}_no_guessable")
                cell["floor_no_guessable"] = hard_floor
                if hard_floor is not None and gone_hard:
                    cell["residual_above_floor"] = means(gone_hard)["answer_f1"] - hard_floor
            entry[layout] = cell
        report["readers"][reader] = entry
    return report


def analyse_e3(
    predictions: dict[tuple[str, str], list[dict[str, Any]]], floors: dict[str, float]
) -> dict[str, Any]:
    """The official six metrics with the full context, overall and by every reported axis."""
    readers = sorted({reader for _, reader in predictions})
    report: dict[str, Any] = {"readers": {}}

    for reader in readers:
        lore = predictions.get(("lore", reader), [])
        if not lore:
            continue
        transposed = where(lore, origin="transposed")
        novel = where(lore, origin="novel")

        entry: dict[str, Any] = {
            "all": means(lore),
            "all_no_yesno": means(lore, exclude_yes_no=True),
            "transposed": means(transposed),
            "novel": means(novel),
            "floor_answer_f1": floors.get(reader, {}).get("lore_all"),
            "padded_layout": means(predictions.get(("lore10", reader), [])),
        }

        # The layout control is only interpretable against the transposed subset, which is what
        # was padded — not against the whole split, whose average includes the novel items and
        # can coincide with the padded figure for no reason at all.
        padded = entry["padded_layout"]
        if padded.get("n"):
            entry["layout_control"] = {
                "native": entry["transposed"],
                "padded_to_ten": padded,
                "delta_answer_f1": padded["answer_f1"] - entry["transposed"]["answer_f1"],
                "delta_sp_f1": padded["sp_f1"] - entry["transposed"]["sp_f1"],
                "delta_joint_f1": padded["joint_f1"] - entry["transposed"]["joint_f1"],
            }

        entry["transposed_by_type"] = {
            question_type: means(where(transposed, type=question_type))
            for question_type in sorted({row.get("type") for row in transposed if row.get("type")})
        }
        entry["novel_by_type"] = {
            question_type: means(where(novel, type=question_type))
            for question_type in sorted({row.get("type") for row in novel if row.get("type")})
        }
        entry["novel_by_level"] = {
            str(level): means(where(novel, level=level))
            for level in sorted({row.get("level") for row in novel if row.get("level")})
        }
        entry["by_supporting_documents"] = {
            str(hops): means(where(lore, supporting_documents=hops))
            for hops in sorted({row.get("supporting_documents") for row in lore if row.get("supporting_documents")})
        }
        # The evidence question: how far answer F1 runs ahead of joint F1 is how often a reader
        # reaches the right string while pointing at the wrong sentences.
        entry["answer_minus_joint_f1"] = entry["all"]["answer_f1"] - entry["all"]["joint_f1"]
        report["readers"][reader] = entry
    return report


def analyse_e5(
    predictions: dict[tuple[str, str], list[dict[str, Any]]],
    floors: dict[str, dict[str, float]],
    retrieval: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """How the choice of ranker moves end-to-end answer and supporting-fact quality.

    The question is not which ranker retrieves better, which recall already answers, but how much
    of a retrieval difference reaches the answer. On a collection where the answer exists only in
    the retrieved passages, a change of ranker should move end-to-end quality by something close
    to the change in retrieval; where the model knows the entities, part of the difference is
    absorbed before it reaches the answer. The ratio between the two is reported per system as
    `transfer`, and it is the quantity the collection exists to make measurable.
    """
    readers = sorted({reader for _, reader in predictions})
    systems = ("bm25", "dense", "hybrid")
    report: dict[str, Any] = {"readers": {}, "retrieval": retrieval}

    for reader in readers:
        entry: dict[str, Any] = {}
        for collection in ("lore", "hotpot"):
            cell: dict[str, Any] = {}
            for system in systems:
                rows = predictions.get((f"{collection}-{system}", reader), [])
                if not rows:
                    continue
                complete = [row for row in rows if row.get("full_support")]
                missed = [
                    row for row in rows
                    if row.get("gold_retrieved") == 0
                    and not (row.get("is_yes_no") or row.get("answer_in_question"))
                ]
                cell[system] = {
                    "overall": means(rows),
                    "full_support": {**means(complete), "share": len(complete) / len(rows)},
                    "no_support": {**means(missed), "share": len(missed) / len(rows)},
                    "recall_at_10": retrieval.get(f"{collection}-{system}", {}).get("recall@10"),
                }
            if not cell:
                continue

            # Everything is expressed against the lexical baseline, so a system's effect is a
            # difference rather than a level, and the two collections can be compared even though
            # their absolute scores differ.
            base = cell.get("bm25")
            if base and base.get("recall_at_10") is not None:
                for system, values in cell.items():
                    if system == "bm25" or values.get("recall_at_10") is None:
                        continue
                    d_recall = values["recall_at_10"] - base["recall_at_10"]
                    d_answer = values["overall"]["answer_f1"] - base["overall"]["answer_f1"]
                    d_support = values["overall"]["sp_f1"] - base["overall"]["sp_f1"]
                    values["vs_bm25"] = {
                        "delta_recall_at_10": d_recall,
                        "delta_answer_f1": d_answer,
                        "delta_sp_f1": d_support,
                        "delta_joint_f1": values["overall"]["joint_f1"] - base["overall"]["joint_f1"],
                        # How much of the retrieval change arrives at the answer. Near 1 means the
                        # ranker's effect is passed through; near 0 means it is absorbed.
                        "transfer_answer": d_answer / d_recall if abs(d_recall) > 1e-6 else None,
                        "transfer_sp": d_support / d_recall if abs(d_recall) > 1e-6 else None,
                    }
            entry[collection] = cell
        report["readers"][reader] = entry
    return report


def analyse_e4(
    predictions: dict[tuple[str, str], list[dict[str, Any]]], panel: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """In-domain HotpotQA against the skeleton-matched \\ds items, base and fine-tuned.

    The gap between the two is the quantity of interest, and the change in that gap from base to
    fine-tuned is what the prediction in the paper is about: if fine-tuning on HotpotQA teaches a
    reliance on memory, the gap should widen rather than narrow.
    """
    readers = sorted({reader for _, reader in predictions})
    report: dict[str, Any] = {"readers": {}}

    for reader in readers:
        pairs = {
            "native": ("hotpot-dev", "lore-paired10"),
            "matched": ("hotpot-dev4", "lore-paired"),
        }
        entry: dict[str, Any] = {
            "checkpoint": panel.get(reader, {}).get("checkpoint", "base"),
            "base_reader": panel.get(reader, {}).get("base_reader"),
        }
        for layout, (hotpot_name, lore_name) in pairs.items():
            hotpot = predictions.get((hotpot_name, reader), [])
            lore = predictions.get((lore_name, reader), [])
            if not hotpot or not lore:
                continue
            hotpot_means = means(hotpot)
            lore_means = means(lore)
            entry[layout] = {
                "hotpot": hotpot_means,
                "lorehopqa": lore_means,
                "gap_answer_f1": hotpot_means["answer_f1"] - lore_means["answer_f1"],
                "gap_joint_f1": hotpot_means["joint_f1"] - lore_means["joint_f1"],
            }
        report["readers"][reader] = entry

    # The change in gap, base to fine-tuned, for every family that has both.
    changes: dict[str, Any] = {}
    for reader, entry in report["readers"].items():
        base = entry.get("base_reader")
        if entry.get("checkpoint") != "fine-tuned" or base not in report["readers"]:
            continue
        baseline = report["readers"][base]
        for layout in ("native", "matched"):
            if layout in entry and layout in baseline:
                changes.setdefault(reader, {})[layout] = {
                    "base_gap_answer_f1": baseline[layout]["gap_answer_f1"],
                    "tuned_gap_answer_f1": entry[layout]["gap_answer_f1"],
                    "change": entry[layout]["gap_answer_f1"] - baseline[layout]["gap_answer_f1"],
                }
    report["gap_change"] = changes
    return report


# --------------------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------------------


def fmt(value: Any, places: int = 3) -> str:
    if value is None or value == {} or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, dict):
        return "--"
    return f"{value:.{places}f}"


def latex_e1(report: dict[str, Any], panel: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% E1: closed-book answer quality (no documents).",
        "\\begin{tabular}{llccccc}",
        "\\hline",
        "Reader family & Pool & \\ds\\ all & transposed & novel & HotpotQA (paired) & $\\Delta$\\\\",
        "\\hline",
    ]
    for reader, entry in report["readers"].items():
        info = panel.get(reader, {})
        label = info.get("label", reader)
        pool = "yes" if info.get("pool") else "no"
        gap = entry.get("leakage_gap_f1", {}).get("mean_difference")
        lines.append(
            f"{label} & {pool} & {fmt(entry['all'].get('answer_f1'))} & "
            f"{fmt(entry['transposed'].get('answer_f1'))} & {fmt(entry['novel'].get('answer_f1'))} & "
            f"{fmt(entry.get('hotpot_paired', {}).get('answer_f1'))} & {fmt(gap)}\\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def latex_e3(report: dict[str, Any], panel: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% E3: full-context reading with the official HotpotQA metrics.",
        "\\begin{tabular}{llcccccc}",
        "\\hline",
        "Reader & Subset & Ans EM & Ans F1 & Sup EM & Sup F1 & Joint F1 & Floor\\\\",
        "\\hline",
    ]
    for reader, entry in report["readers"].items():
        label = panel.get(reader, {}).get("label", reader)
        for subset in ("all", "transposed", "novel"):
            cell = entry.get(subset, {})
            if not cell:
                continue
            lines.append(
                f"{label} & {subset} & {fmt(cell.get('answer_em'))} & {fmt(cell.get('answer_f1'))} & "
                f"{fmt(cell.get('sp_em'))} & {fmt(cell.get('sp_f1'))} & {fmt(cell.get('joint_f1'))} & "
                f"{fmt(entry.get('floor_answer_f1'))}\\\\"
            )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def latex_results(
    e1: dict[str, Any], e3: dict[str, Any], panel: dict[str, dict[str, Any]]
) -> str:
    """The wide open-book/closed-book table the manuscript carries as tab:results."""
    lines = [
        "% Answer, supporting fact and joint F1: open-book and closed-book, by subset.",
        "\\begin{tabular}{l|ccc|ccc||ccc|ccc|ccc}",
        "\\toprule",
        "& \\multicolumn{6}{c||}{\\textbf{Open-book}} & \\multicolumn{9}{c}{\\textbf{Closed-book}} \\\\",
        "\\cmidrule(lr){2-7} \\cmidrule(lr){8-16}",
        "\\textbf{Model} & \\multicolumn{3}{c|}{\\textbf{Novel}} & \\multicolumn{3}{c||}{\\textbf{Trans}}"
        " & \\multicolumn{3}{c|}{\\textbf{Novel}} & \\multicolumn{3}{c|}{\\textbf{Trans}}"
        " & \\multicolumn{3}{c}{\\textbf{HotpotQA}} \\\\",
        "& \\textbf{Ans F1} & \\textbf{SP F1} & \\textbf{Joint F1}" * 1
        + " & \\textbf{Ans F1} & \\textbf{SP F1} & \\textbf{Joint F1}"
        + " & \\textbf{Ans F1} & \\textbf{SP F1} & \\textbf{Joint F1}"
        + " & \\textbf{Ans F1} & \\textbf{SP F1} & \\textbf{Joint F1}"
        + " & \\textbf{Ans F1} & \\textbf{SP F1} & \\textbf{Joint F1} \\\\",
        "\\midrule",
    ]
    for reader in e3.get("readers", {}):
        label = panel.get(reader, {}).get("label", reader)
        open_book = e3["readers"][reader]
        closed = e1.get("readers", {}).get(reader, {})
        cells = []
        for cell in (open_book.get("novel", {}), open_book.get("transposed", {})):
            cells += [fmt(cell.get("answer_f1")), fmt(cell.get("sp_f1")), fmt(cell.get("joint_f1"))]
        for cell in (
            closed.get("novel", {}),
            closed.get("transposed", {}),
            closed.get("hotpot_paired", {}),
        ):
            cells += [fmt(cell.get("answer_f1")), fmt(cell.get("sp_f1")), fmt(cell.get("joint_f1"))]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def latex_e2(report: dict[str, Any], panel: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% E2: answer F1 under full context, leave-one-out, and the closed-book floor.",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Reader & Collection & Full & Leave-one-out & Floor & Drop\\\\",
        "\\hline",
    ]
    names = {
        "lore": "\\ds\\ (4 doc)",
        "lore10": "\\ds\\ (10 doc)",
        "hotpot4": "HotpotQA (4 doc)",
        "hotpot": "HotpotQA (10 doc)",
    }
    for reader, entry in report["readers"].items():
        label = panel.get(reader, {}).get("label", reader)
        for layout, display in names.items():
            cell = entry.get(layout)
            if not cell:
                continue
            lines.append(
                f"{label} & {display} & {fmt(cell['full'].get('answer_f1'))} & "
                f"{fmt(cell['leave_one_out'].get('answer_f1'))} & {fmt(cell.get('floor_answer_f1'))} & "
                f"{fmt(cell.get('drop_answer_f1'))}\\\\"
            )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def latex_e4(report: dict[str, Any], panel: dict[str, dict[str, Any]]) -> str:
    lines = [
        "% E4: transfer from HotpotQA fine-tuning, under the matched layout control.",
        "\\begin{tabular}{llccccc}",
        "\\hline",
        " & & \\multicolumn{2}{c}{HotpotQA dev} & \\multicolumn{2}{c}{\\ds\\ (paired)} & \\\\",
        "Reader & Checkpoint & Ans F1 & Joint F1 & Ans F1 & Joint F1 & Gap\\\\",
        "\\hline",
    ]
    for reader, entry in report["readers"].items():
        cell = entry.get("matched")
        if not cell:
            continue
        label = panel.get(reader, {}).get("label", reader)
        lines.append(
            f"{label} & {entry.get('checkpoint', 'base')} & "
            f"{fmt(cell['hotpot'].get('answer_f1'))} & {fmt(cell['hotpot'].get('joint_f1'))} & "
            f"{fmt(cell['lorehopqa'].get('answer_f1'))} & {fmt(cell['lorehopqa'].get('joint_f1'))} & "
            f"{fmt(cell.get('gap_answer_f1'))}\\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def reader_panel(config_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Display names, pool membership and checkpoint role, keyed by reader handle."""
    import yaml

    panel: dict[str, dict[str, Any]] = {}
    for path in config_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        for entry in payload.get("readers", []):
            panel[entry["name"]] = {
                "label": entry.get("family", entry["name"]),
                # The checkpoint string, which is also how construction recorded the generator
                # that authored an item — the join the authorship analysis needs.
                "model": entry.get("model"),
                "pool": entry.get("pool"),
                "parameters_b": entry.get("parameters_b"),
                "checkpoint": entry.get("checkpoint", "base"),
                "base_reader": entry.get("base_reader"),
            }
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate E1-E4 into the paper's tables.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run tree root.")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        default=[Path("configs/readers.yaml")],
    )
    parser.add_argument("--resamples", type=int, default=5000, help="Bootstrap resamples.")
    parser.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    args = parser.parse_args()

    panel = reader_panel(args.config)
    loaded = load_predictions(args.output_dir, args.split)
    if not loaded:
        raise SystemExit(f"No predictions under {args.output_dir} for split {args.split!r}")

    summary: dict[str, Any] = {}

    e1 = (
        analyse_e1(loaded["e1_closed_book"], panel, resamples=args.resamples)
        if "e1_closed_book" in loaded
        else {}
    )
    if e1:
        summary["e1"] = e1
    # Every later experiment reads its floor from E1, so a number can be read against the floor
    # rather than against zero.
    # Every later experiment reads its floor from E1. Four numbers per reader, not one: the two
    # collections have closed-book scores an order of magnitude apart, and within each the
    # guessable items carry credit that a conditioned comparison must not be charged for.
    floors = {
        reader: {
            "lore_all": entry["all"]["answer_f1"],
            "lore_no_guessable": entry["all_no_guessable"]["answer_f1"],
            "hotpot_all": entry.get("hotpot_paired", {}).get("answer_f1"),
            "hotpot_no_guessable": entry.get("hotpot_paired_no_guessable", {}).get("answer_f1"),
        }
        for reader, entry in e1.get("readers", {}).items()
    }

    if "e2_ablated_evidence" in loaded:
        summary["e2"] = analyse_e2(
            loaded["e2_ablated_evidence"],
            floors,
            args.output_dir / "e2_ablated_evidence" / args.split / "tasks",
        )
    if "e3_full_context" in loaded:
        summary["e3"] = analyse_e3(loaded["e3_full_context"], floors)
    if "e5_bm25_retrieval" in loaded:
        retrieval = {}
        for report_name in ("retrieval.json", "retrieval-dense.json"):
            path = args.output_dir / "e5_bm25_retrieval" / args.split / report_name
            if path.exists():
                blob = json.loads(path.read_text())
                retrieval.update(blob.get("collections") or blob.get("systems") or {})
        summary["e5"] = analyse_e5(loaded["e5_bm25_retrieval"], floors, retrieval)
    if "e4_transfer" in loaded:
        summary["e4"] = analyse_e4(loaded["e4_transfer"], panel)

    results = args.output_dir / "results" / args.split
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    tables = []
    if "e1" in summary:
        tables += [latex_e1(summary["e1"], panel), ""]
    if "e2" in summary:
        tables += [latex_e2(summary["e2"], panel), ""]
    if "e3" in summary:
        tables += [latex_e3(summary["e3"], panel), ""]
    if "e1" in summary and "e3" in summary:
        tables += [latex_results(summary["e1"], summary["e3"], panel), ""]
    if "e4" in summary:
        tables += [latex_e4(summary["e4"], panel), ""]
    (results / "tables.tex").write_text("\n".join(tables), encoding="utf-8")

    print(f"[analyse] summary -> {results / 'summary.json'}")
    print(f"[analyse] tables  -> {results / 'tables.tex'}")
    print()
    for experiment, report in summary.items():
        print(f"=== {experiment.upper()} ===")
        for reader, entry in report.get("readers", {}).items():
            label = panel.get(reader, {}).get("label", reader)
            if experiment == "e1":
                gap = entry.get("leakage_gap_f1", {})
                print(
                    f"  {label:24s} lore {fmt(entry['all'].get('answer_f1'))} "
                    f"(trans {fmt(entry['transposed'].get('answer_f1'))}, "
                    f"novel {fmt(entry['novel'].get('answer_f1'))})  "
                    f"hotpot {fmt(entry.get('hotpot_paired', {}).get('answer_f1'))}  "
                    f"gap {fmt(gap.get('mean_difference'))} "
                    f"[{fmt(gap.get('ci_low'))}, {fmt(gap.get('ci_high'))}] "
                    f"p={gap.get('p_value', float('nan')):.2g}"
                )
            elif experiment == "e3":
                print(
                    f"  {label:24s} ans F1 {fmt(entry['all'].get('answer_f1'))} "
                    f"sup F1 {fmt(entry['all'].get('sp_f1'))} "
                    f"joint F1 {fmt(entry['all'].get('joint_f1'))}"
                )
            elif experiment == "e2":
                for layout, cell in entry.items():
                    print(
                        f"  {label:24s} {layout:8s} full {fmt(cell['full'].get('answer_f1'))} "
                        f"-> loo {fmt(cell['leave_one_out'].get('answer_f1'))} "
                        f"(floor {fmt(cell.get('floor_answer_f1'))}, drop {fmt(cell.get('drop_answer_f1'))})"
                    )
            elif experiment == "e5":
                for collection, systems in entry.items():
                    for system, cell in systems.items():
                        delta = cell.get("vs_bm25", {})
                        extra = ""
                        if delta:
                            extra = (f"  dR@10 {delta['delta_recall_at_10']:+.3f} -> "
                                     f"dAns {delta['delta_answer_f1']:+.3f} "
                                     f"dSP {delta['delta_sp_f1']:+.3f}")
                        print(f"  {label:20s} {collection:6s} {system:6s} "
                              f"Ans {fmt(cell['overall'].get('answer_f1'))} "
                              f"SP {fmt(cell['overall'].get('sp_f1'))} "
                              f"Joint {fmt(cell['overall'].get('joint_f1'))}{extra}")
            elif experiment == "e4":
                cell = entry.get("matched")
                if cell:
                    print(
                        f"  {label:24s} {entry.get('checkpoint'):11s} "
                        f"hotpot {fmt(cell['hotpot'].get('answer_f1'))} "
                        f"lore {fmt(cell['lorehopqa'].get('answer_f1'))} "
                        f"gap {fmt(cell.get('gap_answer_f1'))}"
                    )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
