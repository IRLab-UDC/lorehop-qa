#!/usr/bin/env python3
"""Closed-book versus zero-relevant-retrieved, on identical items.

The questions for which a retriever returns no relevant document are not a random sample: they
are selected by retrieval failure, and may be harder or differently worded than the collection
as a whole. Comparing their end-to-end score against a closed-book average over all items
therefore measures the selection as much as the effect.

This restricts both conditions to the same item identifiers and reports the paired difference,
which is what the claim about corpus-internal access rests on.

    python3 scripts/no_support_paired.py --experiments experiments --split test
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
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
PAIRS = [("lore", "lore-bm25", "lorehopqa"), ("hotpot", "hotpot-bm25", "hotpot-paired")]


def rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record
    return out


def bootstrap(diffs: list[float], resamples: int = 2000, seed: int = 20260901) -> tuple:
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
    for label, system, closed in PAIRS:
        print(f"=== {label}: items where BM25 returned no relevant document ===")
        print("{:9s}{:>7s}{:>11s}{:>13s}{:>10s}{:>20s}".format(
            "model", "n", "retrieved", "closed-book", "diff", "95% CI"))
        for model in MODELS:
            retrieved = rows(args.experiments / "e5_bm25_retrieval" / args.split /
                             "predictions" / f"{system}.{model}.jsonl")
            book = rows(args.experiments / "e1_closed_book" / args.split /
                        "predictions" / f"{closed}.{model}.jsonl")
            if not retrieved or not book:
                continue
            # The same exclusion the aggregate uses: an item answerable by chance carries credit
            # that neither condition earned.
            ids = [i for i, r in retrieved.items()
                   if r.get("gold_retrieved") == 0
                   and not (r.get("is_yes_no") or r.get("answer_in_question"))
                   and i in book]
            if not ids:
                continue
            diffs = [retrieved[i]["answer_f1"] - book[i]["answer_f1"] for i in ids]
            low, high = bootstrap(diffs)
            mean_r = statistics.fmean(retrieved[i]["answer_f1"] for i in ids)
            mean_b = statistics.fmean(book[i]["answer_f1"] for i in ids)
            print("{:9s}{:>7d}{:>11.3f}{:>13.3f}{:>10.3f}{:>20s}".format(
                model, len(ids), mean_r, mean_b, statistics.fmean(diffs),
                f"[{low:+.3f}, {high:+.3f}]"))
            report.setdefault(label, {})[model] = {
                "n": len(ids), "retrieved_f1": round(mean_r, 4),
                "closed_book_f1": round(mean_b, 4),
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
