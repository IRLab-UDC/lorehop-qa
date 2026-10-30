#!/usr/bin/env python
"""Measure how each embedder's tokenizer splits the collection's invented names.

An embedder that has never seen a name cannot hold a representation for it: the name arrives as
a run of subword fragments, and whatever the encoder composes from those fragments is what the
index stores. Fine-tuning the encoder cannot repair that, because the unit the model would need
to learn does not exist in its vocabulary. This measures the size of that gap by comparing the
pieces per name on this collection against the pieces per name on real-world text the same
tokenizers were built over.

    python scripts/tokenizer_fragmentation.py --dist-dir dist --hotpot <hotpot.json> \\
        --lorepedia dist/lorepedia/lorepedia.jsonl --output experiments/tokenizer_fragmentation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proper_nouns import nouns  # noqa: E402

MODELS = {
    "jina": "jinaai/jina-embeddings-v5-text-small-retrieval",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "simcse": "princeton-nlp/unsup-simcse-bert-base-uncased",
}


def collection_names(dist_dir: Path, lorepedia: Path | None) -> tuple[Counter, Counter]:
    """Names in the items, and names in the Lorepedia, counted separately.

    Kept apart because an adaptation may only read the Lorepedia: the share of item names that
    the Lorepedia also carries bounds what any vocabulary built from it could cover.
    """
    items: Counter[str] = Counter()
    for split in ("test", "validation"):
        for line in (dist_dir / "lorehopqa" / f"{split}.jsonl").open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                for text in [record["question"]] + [t + ". " + " ".join(s)
                                                    for t, s in record["context"]]:
                    items.update(nouns(text)[0])
    lore: Counter[str] = Counter()
    if lorepedia:
        for line in lorepedia.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                lore.update(nouns(record["title"] + ". " + " ".join(record["sentences"]))[0])
    return items, lore


def reference_names(paths: list[Path]) -> Counter:
    out: Counter[str] = Counter()
    for path in paths:
        for record in json.loads(path.read_text()):
            for text in [record["question"]] + [t + ". " + " ".join(s)
                                                for t, s in record["context"]]:
                out.update(nouns(text)[0])
    return out


def pieces(tokenizer, names: list[str]) -> list[int]:
    encoded = tokenizer(names, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in encoded]


def summarise(counts: list[int]) -> dict:
    n = len(counts)
    return {
        "names": n,
        "mean_pieces": round(sum(counts) / n, 3) if n else 0.0,
        "single_token_share": round(sum(c == 1 for c in counts) / n, 4) if n else 0.0,
        "three_or_more_share": round(sum(c >= 3 for c in counts) / n, 4) if n else 0.0,
        "max_pieces": max(counts) if n else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, action="append", required=True)
    parser.add_argument("--lorepedia", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    items, lore = collection_names(args.dist_dir, args.lorepedia)
    reference = reference_names(args.hotpot)
    # Names shared with real-world text are not what the claim is about: they are already in the
    # tokenizer for the ordinary reason. The comparison is between names each side owns alone.
    invented = sorted(n for n in items if n not in reference)
    real = sorted(n for n in reference if n not in items)
    covered = sum(1 for n in invented if n in lore)
    # Types understate what a Lorepedia-built vocabulary would reach: a name the items use once
    # matters less than one they use in every world. The occurrence-weighted share is the figure
    # that bounds a vocabulary adaptation, and the frequency bands show where the misses fall.
    occurrences = sum(items[n] for n in invented)
    covered_occurrences = sum(items[n] for n in invented if n in lore)
    bands = {}
    for label, low, high in (("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19), ("20+", 20, 10 ** 9)):
        band = [n for n in invented if low <= items[n] <= high]
        bands[label] = {
            "names": len(band),
            "in_lorepedia": sum(1 for n in band if n in lore),
            "share": round(sum(1 for n in band if n in lore) / max(1, len(band)), 4),
        }
    print(f"[names] {len(invented)} invented types, {len(real)} real-world, "
          f"{covered} invented types in the Lorepedia ({covered / max(1, len(invented)):.1%}); "
          f"occurrence-weighted {covered_occurrences / max(1, occurrences):.1%}", flush=True)
    for label, band in bands.items():
        print(f"[band] item frequency {label}: {band['in_lorepedia']}/{band['names']} "
              f"in the Lorepedia ({band['share']:.1%})", flush=True)

    report = {
        "invented_names": len(invented),
        "real_world_names": len(real),
        "invented_names_in_lorepedia": covered,
        "lorepedia_coverage_of_item_names": round(covered / max(1, len(invented)), 4),
        "lorepedia_coverage_occurrence_weighted": round(covered_occurrences / max(1, occurrences), 4),
        "coverage_by_item_frequency": bands,
        "models": {},
    }
    for tag, model in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model)
        report["models"][tag] = {
            "model": model,
            "vocab_size": tokenizer.vocab_size,
            "invented": summarise(pieces(tokenizer, invented)),
            "real_world": summarise(pieces(tokenizer, real)),
        }
        inv = report["models"][tag]["invented"]
        ref = report["models"][tag]["real_world"]
        print(f"[{tag}] invented {inv['mean_pieces']} pieces/name "
              f"({inv['single_token_share']:.1%} single-token) vs real-world "
              f"{ref['mean_pieces']} ({ref['single_token_share']:.1%})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
