#!/usr/bin/env python3
"""Ablated accuracy against closed-book accuracy on the same items.

The ablated variants are a selected subset: they are the items whose answer the ablation actually
removes, and they are drawn from a subsample. Comparing their score against a closed-book average
over the whole split measures that selection as well as the effect. This restricts both conditions
to the items underlying the ablated variants and reports the paired difference.

    python3 scripts/ablation_paired.py --experiments experiments --split test
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import re
import string
from collections import defaultdict
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

# The reported panel, which must match PARTICIPATING + CONTROL in paper_tables.py.
# paper_tables.py reads this file's output and fails naming any reader it expects and
# does not find, so the two lists cannot drift silently in that direction.
MODELS = list(PANEL_NAMES)


def normalize_answer(s: str) -> str:
    """The official HotpotQA normalisation, so the survival test matches the aggregate's."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_survives(tasks_dir: Path, loo_name: str) -> dict[str, bool]:
    """Whether the gold answer is still printed in what the ablation leaves.

    Reimplemented from analyze_experiments.answer_survives_ablation rather than imported, since
    that module pulls in dependencies the access node does not carry. The two must agree: this is
    the split the hop-necessity claim rests on.
    """
    path = tasks_dir / f"{loo_name}.jsonl"
    if not path.exists():
        return {}
    survives: dict[str, bool] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        answer = normalize_answer(row["answer"])
        if not answer:
            continue
        text = normalize_answer(
            " ".join(sentence for _, sentences in row["context"] for sentence in sentences))
        survives[row["id"]] = answer in text
    return survives


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def bootstrap(diffs, resamples=2000, seed=20260901):
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, dict] = {}
    # The two sides of one paired comparison, and they have to be the two sides of the *same*
    # pairing. hotpot4 is HotpotQA reduced to the matched four-document layout over 1,982
    # items; its \ds counterpart is lore10, over the same 1,982 items. lore-loo is a
    # different, larger sweep (5,000 items), and pairing it against hotpot4 compares two
    # populations while reporting one.
    for label, ablation_set, closed in (("lore10", "lore10-loo", "lorehopqa"),
                                        ("hotpot4", "hotpot4-loo", "hotpot-paired")):
        print(f"=== {label}: variants whose ablation removes the answer ===")
        print("{:9s}{:>9s}{:>8s}{:>11s}{:>13s}{:>9s}{:>19s}".format(
            "model", "variants", "items", "ablated", "closed-book", "diff", "95% CI"))
        for model in MODELS:
            variants = rows(args.experiments / "e2_ablated_evidence" / args.split /
                            "predictions" / f"{ablation_set}.{model}.jsonl")
            book = {r["id"]: r for r in
                    rows(args.experiments / "e1_closed_book" / args.split /
                         "predictions" / f"{closed}.{model}.jsonl")}
            if not variants or not book:
                continue
            # The ablated condition, defined exactly as the aggregate defines it: the answer is
            # absent from what remains, and the question is not answerable by chance.
            survives = answer_survives(
                args.experiments / "e2_ablated_evidence" / args.split / "tasks", ablation_set)
            by_item: dict[str, list[float]] = defaultdict(list)
            for r in variants:
                if r.get("is_yes_no") or r.get("answer_in_question"):
                    continue
                if survives.get(r["id"]) is not False:
                    continue
                source = r.get("ablation_of") or r.get("id")
                if source in book:
                    by_item[source].append(r["answer_f1"])
            if not by_item:
                continue
            # One score per item, so an item with several variants does not count several times.
            ablated = {i: statistics.fmean(v) for i, v in by_item.items()}
            diffs = [ablated[i] - book[i]["answer_f1"] for i in ablated]
            low, high = bootstrap(diffs)
            print("{:9s}{:>9d}{:>8d}{:>11.3f}{:>13.3f}{:>9.3f}{:>19s}".format(
                model, sum(len(v) for v in by_item.values()), len(ablated),
                statistics.fmean(ablated.values()),
                statistics.fmean(book[i]["answer_f1"] for i in ablated),
                statistics.fmean(diffs), f"[{low:+.3f}, {high:+.3f}]"))
            report.setdefault(label, {})[model] = {
                "items": len(ablated),
                "ablated_f1": round(statistics.fmean(ablated.values()), 4),
                "closed_book_f1": round(statistics.fmean(book[i]["answer_f1"] for i in ablated), 4),
                "difference": round(statistics.fmean(diffs), 4),
                "ci_low": round(low, 4), "ci_high": round(high, 4),
            }
        print()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"[report] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
