#!/usr/bin/env python
"""Test whether a Lorepedia-adapted retriever generalises or merely memorises names.

Adaptation reads the Lorepedia, never the items. If its gain came from having seen an item's
names during adaptation, the gain would concentrate on items whose names the Lorepedia happens
to share and vanish on the rest. If instead the model learned the register and morphology of
the worlds, the gain survives on items whose names it has never seen.

That distinction matters twice over: it says what the adaptation learned, and it is the check
that the adaptation carries no item-specific signal. Items are therefore grouped by how much of
their gold-document naming the Lorepedia covers, and the paired per-item change in recall is
reported within each group.

    python scripts/adaptation_generalisation.py --tasks-dir <split>/tasks \\
        --base lore-dense-minilm --adapted lore-dense-minilm-emb-1e4 \\
        --lorepedia dist/lorepedia/lorepedia.jsonl --hotpot <hotpot.json> --output <json>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proper_nouns import nouns  # noqa: E402


def load_tasks(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record
    return out


def recall(record: dict) -> tuple[float, set[str]]:
    """Share of gold documents present in the retrieved context, and the gold titles."""
    gold = {title for title, _ in record["supporting_facts"]}
    retrieved = {title for title, _ in record["context"]}
    return (len(gold & retrieved) / len(gold) if gold else 0.0), gold


def bootstrap(deltas: list[float], rounds: int, seed: int) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(rounds):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--lorepedia", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    lore_names: set[str] = set()
    for line in args.lorepedia.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            lore_names |= nouns(record["title"] + ". " + " ".join(record["sentences"]))[0]

    # Names shared with real-world text are in the tokenizer already and say nothing about what
    # the adaptation saw, so only names absent from the reference corpus count.
    reference: set[str] = set()
    for record in json.loads(args.hotpot.read_text()):
        for text in [record["question"]] + [t + ". " + " ".join(s) for t, s in record["context"]]:
            reference |= nouns(text)[0]
    print(f"[names] Lorepedia {len(lore_names)}, real-world reference {len(reference)}", flush=True)

    base = load_tasks(args.tasks_dir / f"{args.base}.jsonl")
    adapted = load_tasks(args.tasks_dir / f"{args.adapted}.jsonl")
    shared = sorted(set(base) & set(adapted))
    print(f"[items] {len(shared)} paired", flush=True)

    buckets: dict[str, list[float]] = {"none": [], "partial": [], "most": []}
    base_scores: dict[str, list[float]] = {k: [] for k in buckets}
    for identifier in shared:
        base_recall, gold = recall(base[identifier])
        adapted_recall, _ = recall(adapted[identifier])
        # Naming of the documents the item requires, which is what the retriever has to match.
        names: set[str] = set()
        for title, sentences in base[identifier]["context"]:
            if title in gold:
                names |= nouns(title + ". " + " ".join(sentences))[0]
        invented = {n for n in names if n not in reference}
        covered = len([n for n in invented if n in lore_names]) / len(invented) if invented else 0.0
        key = "none" if covered == 0 else ("partial" if covered <= 0.5 else "most")
        buckets[key].append(adapted_recall - base_recall)
        base_scores[key].append(base_recall)

    report = {"base": args.base, "adapted": args.adapted, "items": len(shared), "groups": {}}
    print(f"\n{'group':<10}{'items':>7}{'base R':>9}{'delta':>9}{'95% CI':>22}")
    for key in ("none", "partial", "most"):
        deltas = buckets[key]
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        low, high = bootstrap(deltas, args.bootstrap, args.seed)
        base_mean = sum(base_scores[key]) / len(base_scores[key])
        report["groups"][key] = {
            "items": len(deltas), "base_recall": round(base_mean, 4),
            "mean_delta": round(mean, 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
            "excludes_zero": low > 0 or high < 0,
        }
        print(f"{key:<10}{len(deltas):>7}{base_mean:>9.3f}{mean:>+9.4f}"
              f"{f'[{low:+.4f}, {high:+.4f}]':>22}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
