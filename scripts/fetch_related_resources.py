#!/usr/bin/env python
"""Sample PhantomWiki and NovelHopQA into the closed-book task format.

The reviewer question these support is whether the resources nearest to ours differ in the one
property ours is built for: that the answer cannot be produced without the retrieved context.
That is measurable on questions and answers alone, so only those fields are pulled.

Rows come through the Hugging Face datasets-server API rather than the parquet files, because
no host in this pipeline has pyarrow and the samples needed are small. Standard library only,
so this runs anywhere with network access.

    python3 scripts/fetch_related_resources.py --output-dir experiments --split test --sample 800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"


def fetch(dataset, config, split, offset, length, attempts=6):
    """One page of rows, retrying on the rate limit.

    The datasets-server throttles bursts, and a sampling run is exactly a burst: without backoff
    the job dies part-way and leaves a truncated sample that looks like a complete one.
    """
    url = (
        f"{API}?dataset={urllib.parse.quote(dataset, safe='')}"
        f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
        f"&offset={offset}&length={length}"
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == attempts - 1:
                raise
            wait = 5 * (2 ** attempt)
            print(f"[api] HTTP {exc.code}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit("unreachable")


def pull(dataset, config, splits, cap):
    """Rows from one dataset, in API page order, capped per split."""
    rows = []
    for split in splits:
        first = fetch(dataset, config, split, 0, 1)
        total = first.get("num_rows_total", 0)
        wanted = min(cap, total)
        offset = 0
        while offset < wanted:
            page = fetch(dataset, config, split, offset, min(100, wanted - offset))
            batch = [item["row"] for item in page.get("rows", [])]
            if not batch:
                break
            for row in batch:
                row["_split"] = split
            rows.extend(batch)
            offset += len(batch)
        print(f"[{dataset}] {split}: {min(offset, wanted)} of {total} rows", flush=True)
    return rows


def as_items(rows, source, question_key="question", answer_key="answer"):
    """Closed-book items: identifier, question, answer, and whatever hop information exists.

    PhantomWiki answers are lists of entities, since a relational query can have several correct
    fillers. They are kept as lists and scored against the best-matching gold, which is the
    convention for multi-answer question answering; collapsing them to one string would score a
    correct answer wrong whenever the model named a different valid entity.
    """
    items = []
    for index, row in enumerate(rows):
        answer = row.get(answer_key)
        answers = answer if isinstance(answer, list) else [answer]
        answers = [str(a) for a in answers if a is not None and str(a).strip()]
        if not answers or not row.get(question_key):
            continue
        items.append(
            {
                "id": f"{source}-{row.get('id', index)}",
                "question": str(row[question_key]),
                "answer": answers[0],
                "type": "multi-hop",
                "level": None,
                "context": [],
                "supporting_facts": [],
                "meta": {
                    "collection": source,
                    "origin": source,
                    "answers": answers,
                    "hops": row.get("number_of_hops") or row.get("difficulty"),
                    "subset": row.get("_split"),
                    "book": row.get("book"),
                },
            }
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample related multi-hop resources.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=800, help="Items per resource.")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tasks = args.output_dir / "e1_closed_book" / args.split / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)

    plans = [
        # Hops 2-4 only, so the comparison is against multi-hop questions rather than the
        # single-hop portion of the collection.
        ("novelhopqa", "abhaygupta1266/novelhopqa", "default", ["hop_2", "hop_3", "hop_4"]),
        ("phantomwiki", "kilian-group/phantom-wiki-v1", "question-answer",
         ["depth_20_size_500_seed_1", "depth_20_size_500_seed_2"]),
    ]

    for name, dataset, config, splits in plans:
        per_split = max(1, args.sample // len(splits))
        rows = pull(dataset, config, splits, per_split)
        items = as_items(rows, name)
        rng.shuffle(items)
        items = items[: args.sample]
        path = tasks / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")
        print(f"[tasks] {name}: {len(items)} items -> {path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
