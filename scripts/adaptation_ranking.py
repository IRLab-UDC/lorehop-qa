#!/usr/bin/env python3
"""Pair the unadapted and Lorepedia-adapted rankings for every embedder.

The paper reports first-stage quality under two retriever conditions, and the pairing has to be
exact: the same items, the same corpus, the same rerank depth, one embedder swapped. This reads
the per-embedder retrieval reports written by build_dense_tasks.py and emits a single artefact
holding both conditions, their difference, and the adaptation recipe that produced the second.

The HotpotQA rows are the control. The Lorepedia contains no HotpotQA text, so an adapted model
that moves there has changed in ways unrelated to the vocabulary it was adapted on, and the
LoreHopQA gain cannot be read as familiarity with the invented worlds.

    python scripts/adaptation_ranking.py --root <experiments>/e5_bm25_retrieval/test \
        --output <experiments>/results/test/adaptation_ranking.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Handle -> (unadapted system name, adapted system name). The adapted rankings carry the tag of
# the checkpoint that produced them; the paper reports the condition, not the tag.
PAIRS = {
    "jina": "jina-adapted2",
    "minilm": "minilm-adapted2",
    "simcse": "simcse-adapted2",
}

# The unadapted Jina systems predate the per-embedder naming and are recorded without a suffix.
BASE = {"jina": "", "minilm": "-minilm", "simcse": "-simcse"}

RECIPE = {
    "objective": "inverse cloze task, in-batch negatives",
    "corpus": "Lorepedia (released composition)",
    "trainable_parameters": "input embeddings only; encoder frozen",
    "learning_rate": 1e-4,
    "epochs": 1,
    "batch_size": 64,
    "note": (
        "One configuration. Batch size, objective, data volume and which parameters are "
        "unfrozen were not swept; a flat result is a result for this recipe."
    ),
}

FIELDS = ("recall@2", "recall@5", "recall@10", "recall@20", "all_gold_in_top_10")


def load(root: Path) -> dict[str, dict]:
    systems: dict[str, dict] = {}
    for report in sorted(root.glob("retrieval*.json")):
        body = json.loads(report.read_text())
        for section in ("systems", "collections"):
            for name, summary in body.get(section, {}).items():
                systems[name] = summary.get("recall", summary)
    return systems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="e5_bm25_retrieval/<split> tree.")
    parser.add_argument("--adapters", type=Path, default=None, help="Directory of checkpoints.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    systems = load(args.root)
    rows: dict[str, dict] = {}
    missing: list[str] = []

    for collection in ("lore", "hotpot"):
        for strategy in ("dense", "hybrid"):
            for handle, tag in PAIRS.items():
                base = f"{collection}-{strategy}{BASE[handle]}"
                adapted = f"{collection}-{strategy}-{tag}"
                if base not in systems or adapted not in systems:
                    missing.extend(n for n in (base, adapted) if n not in systems)
                    continue
                rows[adapted] = {
                    "collection": collection,
                    "strategy": strategy,
                    "embedder": handle,
                    "control": collection == "hotpot",
                    "unadapted": {f: systems[base][f] for f in FIELDS},
                    "adapted": {f: systems[adapted][f] for f in FIELDS},
                    "delta": {
                        f: round(systems[adapted][f] - systems[base][f], 4) for f in FIELDS
                    },
                }

    if missing:
        raise SystemExit("no ranking report for: " + ", ".join(sorted(set(missing))))

    # The quantity the argument turns on is not the gain but the cross-collection gap: how far
    # each embedder falls behind its own HotpotQA score on LoreHopQA, and how much of that
    # distance adaptation on background text recovers. An embedder whose deficit is unfamiliar
    # vocabulary should close part of it; one whose HotpotQA advantage rests on something
    # unsupervised text cannot supply should not.
    gaps: dict[str, dict] = {}
    for strategy in ("dense", "hybrid"):
        for handle, tag in PAIRS.items():
            lore, hotpot = f"lore-{strategy}-{tag}", f"hotpot-{strategy}-{tag}"
            if lore not in rows or hotpot not in rows:
                continue
            base = rows[hotpot]["unadapted"]["recall@10"] - rows[lore]["unadapted"]["recall@10"]
            after = rows[hotpot]["adapted"]["recall@10"] - rows[lore]["adapted"]["recall@10"]
            gaps[f"{strategy}-{handle}"] = {
                "gap_unadapted": round(base, 4),
                "gap_adapted": round(after, 4),
                "gap_closed": round(base - after, 4),
                "fraction_closed": round((base - after) / base, 4) if base else None,
            }

    # The lexical baseline is what the adapted rows have to be read against: the question is not
    # whether adaptation helps but whether it changes the ordering against BM25.
    baseline = {c: systems.get(f"{c}-bm25", {}) for c in ("lore", "hotpot")}

    # Recall@10 as a share of BM25 on the same collection. BM25 itself differs between the two
    # (0.766 against 0.784), so an absolute difference across collections carries that gap as
    # well as the embedder's; dividing by the baseline each system was measured against removes
    # it, and the reversal becomes the point where a system crosses 100%.
    relative: dict[str, dict] = {}
    for name, row in rows.items():
        anchor = baseline[row["collection"]].get("recall@10")
        if not anchor:
            continue
        before = row["unadapted"]["recall@10"] / anchor
        after = row["adapted"]["recall@10"] / anchor
        relative[name] = {
            "bm25_share_unadapted": round(before, 4),
            "bm25_share_adapted": round(after, 4),
            "delta_points": round((after - before) * 100, 2),
        }

    # The shortfall each embedder has to make up, in the same normalised terms: how far short of
    # its own HotpotQA attainment it falls on LoreHopQA, and how much of that adaptation closes.
    shortfall: dict[str, dict] = {}
    for strategy in ("dense", "hybrid"):
        for handle, tag in PAIRS.items():
            lore, hotpot = f"lore-{strategy}-{tag}", f"hotpot-{strategy}-{tag}"
            if lore not in relative or hotpot not in relative:
                continue
            before = (relative[hotpot]["bm25_share_unadapted"]
                      - relative[lore]["bm25_share_unadapted"]) * 100
            after = (relative[hotpot]["bm25_share_adapted"]
                     - relative[lore]["bm25_share_adapted"]) * 100
            shortfall[f"{strategy}-{handle}"] = {
                "shortfall_points_unadapted": round(before, 2),
                "shortfall_points_adapted": round(after, 2),
                "fraction_recovered": round((before - after) / before, 4) if before else None,
            }

    # The checkpoint that produced each ranking names itself in the retrieval report, so the
    # training record is read from that path rather than from a checkpoint of the same shape.
    # Two runs of the same recipe differing only in epochs are otherwise indistinguishable, and
    # attributing one run's trajectory to the other's ranking is a mistake nothing would catch.
    training: dict[str, dict] = {}
    if args.adapters:
        for report in sorted(args.root.glob("retrieval-dense-*-adapted2.json")):
            body = json.loads(report.read_text())
            checkpoint = Path(body["embedder"]).name
            record = args.adapters / checkpoint / "adaptation.json"
            if not record.is_file():
                raise SystemExit(f"{report.name} cites {checkpoint}, which has no adaptation.json")
            training[checkpoint] = json.loads(record.read_text())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "recipe": RECIPE,
                "bm25": baseline,
                "cross_collection_gap": gaps,
                "relative_to_bm25": relative,
                "normalised_shortfall": shortfall,
                "training": training,
                "systems": rows,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )

    for name in sorted(rows):
        row = rows[name]
        mark = " [control]" if row["control"] else ""
        print(
            f"{name:34s} R@10 {row['unadapted']['recall@10']:.3f} -> "
            f"{row['adapted']['recall@10']:.3f}  "
            f"({row['delta']['recall@10']:+.3f}){mark}"
        )
    print("--- Recall@10 as a share of BM25 on the same collection ---")
    for name in sorted(relative):
        r = relative[name]
        print(f"{name:34s} {r['bm25_share_unadapted']:6.1%} -> {r['bm25_share_adapted']:6.1%}"
              f"  ({r['delta_points']:+.1f} pp)")
    print("--- shortfall against own HotpotQA attainment, in points of BM25 ---")
    for name in sorted(shortfall):
        f = shortfall[name]
        share = "n/a" if f["fraction_recovered"] is None else f"{f['fraction_recovered']:.0%}"
        print(f"{name:20s} {f['shortfall_points_unadapted']:5.1f} -> "
              f"{f['shortfall_points_adapted']:5.1f} pp  recovered {share}")
    print("--- cross-collection gap (HotpotQA R@10 minus LoreHopQA R@10) ---")
    for name in sorted(gaps):
        g = gaps[name]
        share = "n/a" if g["fraction_closed"] is None else f"{g['fraction_closed']:.0%}"
        print(
            f"{name:20s} {g['gap_unadapted']:.3f} -> {g['gap_adapted']:.3f}  "
            f"closed {g['gap_closed']:+.3f} ({share})"
        )
    for collection, summary in baseline.items():
        if summary:
            print(f"{collection}-bm25{'':<24s} R@10 {summary['recall@10']:.3f}")
    print(f"[written] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
