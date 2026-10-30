#!/usr/bin/env python3
"""Derived corpus figures quoted in the paper, written to an artefact.

Statistics that appear in prose rather than in a results table are the ones most likely to drift:
they are computed once while writing and never recomputed when the data changes. Emitting them
here puts them under the same verification as every other figure, so
the paper's numbers can be matched against them instead of checked by eye.

    python3 scripts/corpus_statistics.py --dist-dir dist --hotpot <hotpot.json> \
        --output experiments/corpus_statistics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_es_tasks as bm  # noqa: E402


def restrict(items: list[dict], documents: int, seed: int) -> list[dict]:
    """HotpotQA cut to the transposed subset's layout: the gold pair plus distractors.

    Mirrors reader.restrict_context, which is what the matched-layout conditions use. A figure
    quoted for "matched layouts" has to be measured on the same contexts those conditions build,
    not on the full ten-document distractor setting.
    """
    import random

    rng = random.Random(seed)
    out = []
    for item in items:
        marked = {title for title, _ in item["supporting_facts"]}
        supporting = [d for d in item["context"] if d[0] in marked]
        others = [d for d in item["context"] if d[0] not in marked]
        rng.shuffle(others)
        keep = supporting + others[: max(0, documents - len(supporting))]
        out.append({**item, "context": keep})
    return out


def profile(items: list[dict]) -> dict:
    facts = sum(len(item["supporting_facts"]) for item in items)
    gold_documents = sum(len({title for title, _ in item["supporting_facts"]}) for item in items)
    sentences = sum(len(s) for item in items for _, s in item["context"])
    words = sum(len(sentence.split())
                for item in items for _, s in item["context"] for sentence in s)
    documents = sum(len(item["context"]) for item in items)
    return {
        "items": len(items),
        "supporting_facts": facts,
        "supporting_documents": gold_documents,
        "facts_per_supporting_document": round(facts / gold_documents, 4),
        "candidate_sentences_per_item": round(sentences / len(items), 4),
        "documents_per_item": round(documents / len(items), 4),
        "corpus_sentences": sentences,
        "corpus_words": words,
        "corpus_words_millions": round(words / 1e6, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lorepedia", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    lore = bm.load_lorehopqa(args.dist_dir, args.split)
    transposed = [i for i in lore if i["meta"].get("origin") == "transposed"]
    report = {
        "lore_all": profile(lore),
        "lore_transposed": profile(transposed),
    }
    if args.hotpot:
        source_ids = {i["meta"]["source_id"] for i in transposed if i["meta"].get("source_id")}
        hotpot = bm.load_hotpot(args.hotpot, source_ids)
        report["hotpot_paired"] = profile(hotpot)
        layout = round(sum(len(i["context"]) for i in transposed) / len(transposed))
        # The distractors kept are sampled, so one draw is not the quantity to quote. Averaging
        # over seeds gives the figure the matched-layout conditions see in expectation, and the
        # spread shows whether the choice of seed could matter.
        draws = [profile(restrict(hotpot, layout, args.seed + offset)) for offset in range(5)]
        sentences = [d["candidate_sentences_per_item"] for d in draws]
        report["hotpot_matched_layout"] = {
            **draws[0],
            "documents_kept": layout,
            "candidate_sentences_per_item": round(sum(sentences) / len(sentences), 4),
            "candidate_sentences_per_item_min": round(min(sentences), 4),
            "candidate_sentences_per_item_max": round(max(sentences), 4),
            "seeds": len(draws),
        }

    if args.lorepedia and args.lorepedia.exists():
        articles = [json.loads(line) for line in args.lorepedia.open(encoding="utf-8") if line.strip()]
        sentences = sum(len(a["sentences"]) for a in articles)
        words = sum(len(s.split()) for a in articles for s in a["sentences"])
        report["lorepedia"] = {
            "articles": len(articles),
            "worlds": len({a["world"] for a in articles}),
            "sentences": sentences,
            "words": words,
            "words_millions": round(words / 1e6, 4),
            "sentences_per_article": round(sentences / len(articles), 4),
            "words_per_article": round(words / len(articles), 4),
            "articles_per_world": round(len(articles) / len({a["world"] for a in articles}), 4),
        }

    args.output.write_text(json.dumps(report, indent=2))
    for name, body in report.items():
        if "facts_per_supporting_document" in body:
            print(f"{name:22s} facts/gold-doc={body['facts_per_supporting_document']:.4f}  "
                  f"cand-sent/item={body['candidate_sentences_per_item']:.2f}  "
                  f"words={body['corpus_words_millions']:.4f}M")
        else:
            print(f"{name:22s} " + "  ".join(f"{k}={v}" for k, v in body.items()))
    print(f"\n[report] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
