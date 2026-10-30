#!/usr/bin/env python3
"""Among items no LLM answers correctly with full context, how far do the LLMs agree?

Unanimous agreement on a reply that scores zero and disagreement across six models are
different situations. This counts them, and reports nothing about which situation means
what: an item where every model returns the same string is a candidate for one of several
explanations -- the context genuinely supports that string, the reference is written
differently, the question admits two readings -- and separating those is what reading the
items is for.

Agreement is measured on the official HotpotQA normalisation, so "The Valozian Highlanders"
and "valozian highlanders" are one answer rather than two.

    python scripts/unanswered_agreement.py --experiments <experiments> --dist dist \\
        --output <experiments>/results/test/unanswered_agreement.json
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

MODELS = PANEL_NAMES

# The four families that produced items, keyed by the checkpoint recorded on each item.
# Written out rather than parsed from the string so an unrecognised generator is visible
# rather than folded into an "other" bucket.
GENERATORS = {
    "Qwen/Qwen3.6-27B": "Qwen",
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": "Llama",
    "google/gemma-4-31B-it": "Gemma",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral",
}

# The threshold select_for_review.py uses, so "no LLM answered it" means the same thing in
# both places.
F1_CORRECT = 0.75


def tokens(text: str) -> list[str]:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def verbatim_overlap(prediction: str, reference: str) -> bool:
    """Whether either string occurs verbatim inside the other, as select_for_review.py does.

    A screening test, not a correctness test: it says only that one string appears inside the
    other, and the longer text may deny the shorter or attach it to the wrong entity. It is
    reported here because the review set was built with it, so "no LLM answered" means a
    smaller population there than under the F1 threshold alone.
    """
    gold, said = tokens(reference), tokens(prediction)
    if not gold or not said or gold == said:
        return False
    short, long = (gold, said) if len(gold) < len(said) else (said, gold)
    span = len(short)
    return any(long[i:i + span] == short for i in range(len(long) - span + 1))


def normalise(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"no predictions at {path}")
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path, default=None,
                        help="Write the unanimous items here for reading.")
    args = parser.parse_args()

    base = args.experiments / "e3_full_context" / args.split / "predictions"
    full = {m: rows(base / f"lore.{m}.jsonl") for m in MODELS}

    shared = set.intersection(*(set(v) for v in full.values()))
    unanimous, split_verdict, sizes = [], 0, Counter()
    screened = screened_unanimous = 0
    chance_excluded = 0
    unanswered = 0

    for item_id in sorted(shared):
        replies = [full[m][item_id] for m in MODELS]
        # Chance-answerable items are excluded throughout the paper; keeping them here would
        # mix in yes/no questions, where six models agreeing is close to inevitable.
        if any(r.get("is_yes_no") or r.get("answer_in_question") for r in replies):
            chance_excluded += 1
            continue
        if any(float(r.get("answer_f1") or 0.0) > F1_CORRECT for r in replies):
            continue
        unanswered += 1
        if not any(verbatim_overlap(r.get("predicted_answer"), r.get("gold_answer"))
                   for r in replies):
            screened += 1
            if len(Counter(normalise(r.get("predicted_answer")) for r in replies)) == 1:
                screened_unanimous += 1

        answers = [normalise(r.get("predicted_answer")) for r in replies]
        counts = Counter(answers)
        sizes[len(counts)] += 1
        if len(counts) == 1:
            unanimous.append({
                "id": item_id,
                "question": replies[0].get("question"),
                "gold_answer": replies[0]["gold_answer"],
                "agreed_answer": replies[0].get("predicted_answer"),
                "type": replies[0].get("type"),
                "origin": replies[0].get("origin"),
                "max_answer_f1": round(max(float(r.get("answer_f1") or 0.0) for r in replies), 3),
            })
        else:
            split_verdict += 1

    largest = Counter()
    by_type = defaultdict(lambda: {"unanswered": 0, "unanimous": 0})
    # Counted against each path's own eligible population: transposition and induction
    # contribute very different numbers of items, so a raw count says mostly which path is
    # larger.
    by_origin = defaultdict(lambda: {"eligible": 0, "unanswered": 0, "unanimous": 0})
    by_generator = defaultdict(lambda: {"eligible": 0, "unanswered": 0, "unanimous": 0})
    by_generator_origin = defaultdict(lambda: {"eligible": 0, "unanswered": 0})
    for item_id in sorted(shared):
        replies = [full[m][item_id] for m in MODELS]
        if any(r.get("is_yes_no") or r.get("answer_in_question") for r in replies):
            continue
        origin = replies[0].get("origin") or "unknown"
        checkpoint = replies[0].get("generator_model")
        family = GENERATORS.get(checkpoint, checkpoint or "unknown")
        by_origin[origin]["eligible"] += 1
        by_generator[family]["eligible"] += 1
        by_generator_origin[(family, origin)]["eligible"] += 1
        if any(float(r.get("answer_f1") or 0.0) > F1_CORRECT for r in replies):
            continue
        by_origin[origin]["unanswered"] += 1
        by_generator[family]["unanswered"] += 1
        by_generator_origin[(family, origin)]["unanswered"] += 1
        counts = Counter(normalise(r.get("predicted_answer")) for r in replies)
        if len(counts) == 1:
            by_origin[origin]["unanimous"] += 1
            by_generator[family]["unanimous"] += 1
        largest[counts.most_common(1)[0][1]] += 1
        kind = replies[0].get("type") or "unknown"
        by_type[kind]["unanswered"] += 1
        if len(counts) == 1:
            by_type[kind]["unanimous"] += 1

    report = {
        "split": args.split,
        "models": list(MODELS),
        "f1_correct_threshold": F1_CORRECT,
        "items_scored_by_all": len(shared),
        "chance_answerable_excluded": chance_excluded,
        "no_llm_correct": unanswered,
        "no_llm_correct_containment_screened": screened,
        "containment_screened_unanimous": screened_unanimous,
        "unanimous": len(unanimous),
        "unanimous_share": round(len(unanimous) / unanswered, 4) if unanswered else None,
        "disagreeing": split_verdict,
        "distinct_answers_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "largest_cluster_distribution": {str(k): v for k, v in sorted(largest.items())},
        "by_type": {k: v for k, v in sorted(by_type.items())},
        "by_origin": {
            k: {**v,
                "unanswered_rate": round(v["unanswered"] / v["eligible"], 4) if v["eligible"] else None,
                "unanimous_share_of_unanswered": round(v["unanimous"] / v["unanswered"], 4) if v["unanswered"] else None}
            for k, v in sorted(by_origin.items())},
        "by_generator": {
            k: {**v,
                "unanswered_rate": round(v["unanswered"] / v["eligible"], 4) if v["eligible"] else None,
                "unanimous_share_of_unanswered": round(v["unanimous"] / v["unanswered"], 4) if v["unanswered"] else None}
            for k, v in sorted(by_generator.items())},
        "by_generator_and_origin": {
            f"{fam}/{org}": {**v,
                             "unanswered_rate": round(v["unanswered"] / v["eligible"], 4) if v["eligible"] else None}
            for (fam, org), v in sorted(by_generator_origin.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    if args.examples:
        args.examples.write_text(json.dumps(
            {"note": "items no LLM answered where all six returned the same string",
             "items": unanimous}, indent=1, ensure_ascii=False) + "\n")

    print(f"items scored by all {len(MODELS)} models : {len(shared):,}")
    print(f"chance-answerable excluded          : {chance_excluded:,}")
    print(f"no LLM correct with full context    : {unanswered:,}")
    print(f"  same, minus containment overlaps  : {screened:,} "
          f"(the population select_for_review.py draws from)")
    print(f"    of those, all six agree         : {screened_unanimous:,} "
          f"({100 * screened_unanimous / screened:.1f}%)")
    print(f"  all six returned the same answer  : {len(unanimous):,} "
          f"({100 * len(unanimous) / unanswered:.1f}%)")
    print(f"  at least two answers differ       : {split_verdict:,}")
    print("\ndistinct answers among the six:")
    for k, v in sorted(sizes.items()):
        print(f"  {k} distinct : {v:5,} items")
    print("\nlargest agreeing cluster:")
    for k, v in sorted(largest.items(), reverse=True):
        print(f"  {k}/6 agree : {v:5,} items")
    print("\nby construction path:")
    print(f"  {'path':12s} {'eligible':>9s} {'no LLM ok':>10s} {'rate':>7s} "
          f"{'unanimous':>10s} {'share':>7s}")
    for name, v in sorted(by_origin.items()):
        rate = v["unanswered"] / v["eligible"] if v["eligible"] else 0
        share = v["unanimous"] / v["unanswered"] if v["unanswered"] else 0
        print(f"  {name:12s} {v['eligible']:9,} {v['unanswered']:10,} {rate:6.1%} "
              f"{v['unanimous']:10,} {share:6.1%}")
    print("\nby generating family (the model that wrote the item):")
    print(f"  {'family':9s} {'eligible':>9s} {'no LLM ok':>10s} {'rate':>7s} "
          f"{'unanimous':>10s} {'share':>7s}")
    for name, v in sorted(by_generator.items(), key=lambda kv: -kv[1]["eligible"]):
        rate = v["unanswered"] / v["eligible"] if v["eligible"] else 0
        share = v["unanimous"] / v["unanswered"] if v["unanswered"] else 0
        print(f"  {name:9s} {v['eligible']:9,} {v['unanswered']:10,} {rate:6.1%} "
              f"{v['unanimous']:10,} {share:6.1%}")

    print("\n  same, holding the construction path fixed:")
    print(f"  {'family':9s} {'transposed':>22s} {'induced':>22s}")
    for name in sorted({f for f, _ in by_generator_origin}):
        cells = []
        for org in ("transposed", "novel"):
            v = by_generator_origin.get((name, org))
            if not v or not v["eligible"]:
                cells.append(f"{'---':>22s}")
            else:
                cells.append(f"{v['unanswered']:6,}/{v['eligible']:<6,} "
                             f"{v['unanswered'] / v['eligible']:6.1%}")
        print(f"  {name:9s} " + " ".join(cells))
    print(f"\n[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
