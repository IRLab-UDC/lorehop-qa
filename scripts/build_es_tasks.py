#!/usr/bin/env python
"""Build the retrieval task lists for E5 using Elasticsearch.

Retrieval and reading are separated here for a practical reason and kept separate for a better
one. The practical reason is that the compute nodes may have no route to the Elasticsearch
host, where only the access node does, so ranking cannot happen inside a GPU job. The better
reason is that a ranking every reader shares has to be produced exactly once --- if each reader
retrieved for itself, the six panels would be answering six different collections and no
per-item comparison between them would mean anything.

So this script runs on the access node, writes the task lists into the shared tree, and the GPU
jobs consume them without touching the network.

DEPENDENCIES.  Standard library only. The access node carries Python 3.9 with no pydantic,
no orjson and no Elasticsearch client, and the pipeline's own package needs all three; rather
than install anything on a shared login node, the handful of things needed here --- reading the
released JSONL, hashing a document to an identifier, and speaking the Elasticsearch REST API ---
are written out directly.

    # on the access node, from inside the synced tree
    python3 scripts/build_es_tasks.py \\
        --dist-dir dist --hotpot /path/to/hotpot_dev_distractor_v1.json \\
        --output-dir experiments --split test --top-k 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Anserini's defaults, and the values quoted in the paper. Stated as constants rather than left
# to the server's own defaults (k1=1.2, b=0.75) so that the run is reproducible from the text.
DEFAULT_K1 = 0.9
DEFAULT_B = 0.4


# --------------------------------------------------------------------------------------
# Elasticsearch, over the REST API
# --------------------------------------------------------------------------------------


def request(method, url, payload=None, ndjson=False, timeout=120):
    data = None
    headers = {}
    if payload is not None:
        if ndjson:
            data = payload.encode("utf-8")
            headers["Content-Type"] = "application/x-ndjson"
        else:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"{method} {url} -> HTTP {exc.code}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"{method} {url} unreachable: {exc.reason}") from exc
    return json.loads(body) if body.strip() else {}


def create_index(es, index, k1, b, analyzer, recreate, expected=0):
    exists = True
    try:
        request("HEAD", f"{es}/{index}")
    except SystemExit:
        exists = False

    if exists and recreate:
        print(f"[es] deleting existing index {index}", flush=True)
        request("DELETE", f"{es}/{index}")
        exists = False
    if exists:
        # An index exists as soon as indexing starts, so its presence does not mean the corpus
        # is in it. A run interrupted partway would otherwise be resumed against a short index,
        # and the only visible effect would be a lower recall that looks like a result.
        request("POST", f"{es}/{index}/_refresh")
        held = request("GET", f"{es}/{index}/_count").get("count", 0)
        if held >= expected:
            print(f"[es] reusing existing index {index} ({held} documents)", flush=True)
            return False
        print(f"[es] {index} holds {held} of {expected} documents; rebuilding", flush=True)
        request("DELETE", f"{es}/{index}")
        exists = False

    # One shard, no replicas: this is a read-only evaluation index on a shared server, and BM25
    # scores are computed per shard, so a single shard also removes the score variation that
    # sharding would otherwise introduce between runs.
    request(
        "PUT",
        f"{es}/{index}",
        {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}},
            },
            "mappings": {
                "properties": {"content": {"type": "text", "analyzer": analyzer}}
            },
        },
    )
    print(f"[es] created {index} (BM25 k1={k1} b={b}, analyzer={analyzer})", flush=True)
    return True


def bulk_index(es, index, documents, batch=2000):
    total = len(documents)
    for start in range(0, total, batch):
        chunk = documents[start : start + batch]
        lines = []
        for identifier, content in chunk:
            lines.append(json.dumps({"index": {"_index": index, "_id": identifier}}))
            lines.append(json.dumps({"content": content}))
        result = request("POST", f"{es}/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if result.get("errors"):
            first = next(
                (item for item in result["items"] if item.get("index", {}).get("error")), None
            )
            raise SystemExit(f"bulk indexing failed: {json.dumps(first)[:500]}")
        print(f"[es] indexed {min(start + batch, total)}/{total}", flush=True)
    request("POST", f"{es}/{index}/_refresh")


TRANSIENT = ("es_rejected_execution_exception", "circuit_breaking_exception",
             "search_context_missing_exception")


def transient(error):
    """Whether a per-response error is load-related and worth retrying.

    Elasticsearch reports a full search thread pool inside the response body rather than as an
    HTTP status, so a rejected batch looks like a successful request holding a failed query. The
    server is shared, and a large corpus raises the cost of each search, so rejection under load
    is expected rather than exceptional and the batch is retried instead of ending the run.
    """
    blob = json.dumps(error)
    return any(kind in blob for kind in TRANSIENT)


def msearch(es, index, questions, size, batch=25, attempts=6):
    """Rank a list of questions, returning a list of [(doc_id, score), ...] in submission order."""
    rankings = []
    for start in range(0, len(questions), batch):
        chunk = questions[start : start + batch]
        lines = []
        for question in chunk:
            lines.append(json.dumps({"index": index}))
            lines.append(
                json.dumps(
                    {
                        "size": size,
                        "query": {"match": {"content": question}},
                        "_source": False,
                    }
                )
            )
        payload = "\n".join(lines) + "\n"
        for attempt in range(attempts):
            result = request("POST", f"{es}/_msearch", payload, ndjson=True)
            responses = result.get("responses", [])
            failed = [r["error"] for r in responses if "error" in r]
            if not failed:
                break
            if not all(transient(error) for error in failed) or attempt == attempts - 1:
                raise SystemExit(f"search failed: {json.dumps(failed[0])[:400]}")
            pause = 2 ** attempt
            print(f"[es] {len(failed)} of {len(responses)} rejected under load; "
                  f"retrying in {pause}s", flush=True)
            time.sleep(pause)
        for response in responses:
            rankings.append(
                [(hit["_id"], hit["_score"]) for hit in response["hits"]["hits"]]
            )
        if (start // batch) % 20 == 0:
            print(f"[es] searched {min(start + batch, len(questions))}/{len(questions)}", flush=True)
    return rankings


# --------------------------------------------------------------------------------------
# Items and corpus
# --------------------------------------------------------------------------------------


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def document_key(title, sentences):
    """Stable short identifier for a document.

    Title alone is not unique: two items in the same world may both write about an institution
    without writing the same article, and 6,018 titles in the test split occur on more than one
    document. Hashing title and text together keeps those distinct, so retrieval recall counts
    the document an item actually marked.
    """
    digest = hashlib.sha1(("\n".join([title] + list(sentences))).encode("utf-8")).hexdigest()
    return digest[:16]


def load_lorehopqa(dist_dir, split):
    items = []
    provenance = {}
    prov_path = Path(dist_dir) / "lorehopqa-private" / f"{split}.provenance.jsonl"
    if prov_path.exists():
        provenance = {row["_id"]: row for row in read_jsonl(prov_path)}
    for row in read_jsonl(Path(dist_dir) / "lorehopqa" / f"{split}.jsonl"):
        source = provenance.get(row["_id"], {})
        items.append(
            {
                "id": row["_id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row.get("level"),
                "context": row["context"],
                "supporting_facts": row["supporting_facts"],
                "meta": {
                    "collection": "lorehopqa",
                    "origin": source.get("origin"),
                    "source_id": source.get("source_id"),
                    "generator_model": source.get("generator_model"),
                },
            }
        )
    return items


def load_hotpot(path, keep_ids):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = []
    for row in payload:
        if row["_id"] not in keep_ids:
            continue
        items.append(
            {
                "id": row["_id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type", "bridge"),
                "level": row.get("level"),
                "context": row["context"],
                "supporting_facts": row["supporting_facts"],
                "meta": {"collection": "hotpotqa", "origin": "hotpotqa"},
            }
        )
    return items


def load_lorepedia(path):
    """The background articles shipped alongside the collection.

    These belong to no item and are relevant to no question. They exist so that retrieval runs
    against a corpus of world-consistent text rather than only against the documents the
    questions were built from, which is the situation a practitioner indexing an encyclopaedia
    faces and the one the item-only corpus does not represent.
    """
    articles = []
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            articles.append((record["title"], record["sentences"]))
    return articles


def build_corpus(items, background=()):
    """Every distinct document, plus the gold ids each item depends on."""
    documents = {}
    gold = {}
    for item in items:
        marked = {fact[0] for fact in item["supporting_facts"]}
        item_gold = []
        for title, sentences in item["context"]:
            key = document_key(title, sentences)
            documents[key] = (title, sentences)
            if title in marked:
                item_gold.append(key)
        gold[item["id"]] = item_gold
    for title, sentences in background:
        documents.setdefault(document_key(title, sentences), (title, sentences))
    return documents, gold


def recall_at(gold, ranking, k):
    if not gold:
        return 0.0
    top = {identifier for identifier, _ in ranking[:k]}
    return sum(1 for identifier in gold if identifier in top) / len(gold)


def build(items, documents, gold, rankings, top_k):
    """Rewrite each item so its context is what the retriever returned."""
    rows, stats = [], {2: [], 5: [], 10: [], 20: [], 100: []}
    complete = 0
    for item, ranking in zip(items, rankings):
        item_gold = gold[item["id"]]
        for k, values in stats.items():
            values.append(recall_at(item_gold, ranking, k))

        top = ranking[:top_k]
        retrieved = [identifier for identifier, _ in top]
        found = [identifier for identifier in item_gold if identifier in retrieved]
        complete += 1 if item_gold and len(found) == len(item_gold) else 0

        context = []
        for identifier in retrieved:
            title, sentences = documents[identifier]
            context.append([title, list(sentences)])

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "context": context,
                "supporting_facts": [list(fact) for fact in item["supporting_facts"]],
                "meta": dict(
                    item["meta"],
                    documents=len(context),
                    supporting_documents=len({fact[0] for fact in item["supporting_facts"]}),
                    supporting_facts=len(item["supporting_facts"]),
                    gold_documents=len(item_gold),
                    gold_retrieved=len(found),
                    full_support=bool(item_gold) and len(found) == len(item_gold),
                    gold_ranks=[retrieved.index(i) + 1 for i in found],
                    retrieval="bm25-elasticsearch",
                ),
            }
        )

    summary = {f"recall@{k}": sum(v) / len(v) for k, v in sorted(stats.items()) if v}
    summary[f"all_gold_in_top_{top_k}"] = complete / len(items) if items else 0.0
    summary["items"] = len(items)
    summary["documents"] = len(documents)
    return rows, summary


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.replace(path)


def run(name, items, args, index, background=()):
    print(f"\n=== {name}: {len(items)} items ===", flush=True)
    documents, gold = build_corpus(items, background)
    print(f"[corpus] {len(documents)} distinct documents", flush=True)

    fresh = create_index(args.es, index, args.k1, args.b, args.analyzer, args.recreate,
                         len(documents))
    if fresh:
        payload = [
            (key, title + "\n" + " ".join(sentences))
            for key, (title, sentences) in documents.items()
        ]
        started = time.time()
        bulk_index(args.es, index, payload)
        print(f"[es] indexing took {time.time() - started:.0f}s", flush=True)

    started = time.time()
    rankings = msearch(args.es, index, [item["question"] for item in items], max(args.top_k, 100))
    print(f"[es] {len(items)} queries in {time.time() - started:.0f}s", flush=True)

    rows, summary = build(items, documents, gold, rankings, args.top_k)
    tasks = Path(args.output_dir) / "e5_bm25_retrieval" / args.split / "tasks" / f"{name}.jsonl"
    write_jsonl(tasks, rows)
    print(f"[tasks] {name}: {len(rows)} items -> {tasks}", flush=True)
    print("[retrieval] " + "  ".join(f"{k}={v:.3f}" for k, v in summary.items() if k.startswith(("recall", "all_gold"))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build E5 retrieval task lists with Elasticsearch.")
    parser.add_argument("--dist-dir", required=True)
    parser.add_argument("--hotpot", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--es", default="http://localhost:9200")
    parser.add_argument("--index-prefix", default="lorehopqa")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--analyzer", default="english")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the indices.")
    parser.add_argument("--lorepedia", type=Path, default=None,
                        help="Background articles to add to the LoreHopQA corpus.")
    args = parser.parse_args()

    info = request("GET", args.es)
    print(f"[es] {args.es} -> {info.get('version', {}).get('number', '?')}", flush=True)

    lore = load_lorehopqa(args.dist_dir, args.split)
    # With background articles the corpus is a different collection, so it gets its own index
    # and its own system name rather than overwriting the item-only ranking it is compared to.
    background = load_lorepedia(args.lorepedia) if args.lorepedia else ()
    if background:
        print(f"[lorepedia] {len(background)} background articles", flush=True)
    suffix = "-lorepedia" if background else ""
    summaries = {
        f"lore-bm25{suffix}": run(
            f"lore-bm25{suffix}", lore, args,
            f"{args.index_prefix}-{args.split}-lore{suffix}", background
        )
    }

    if args.hotpot and not background:
        source_ids = {
            item["meta"]["source_id"]
            for item in lore
            if item["meta"].get("origin") == "transposed" and item["meta"].get("source_id")
        }
        hotpot = load_hotpot(args.hotpot, source_ids)
        summaries["hotpot-bm25"] = run(
            "hotpot-bm25", hotpot, args, f"{args.index_prefix}-{args.split}-hotpot"
        )

    report = (Path(args.output_dir) / "e5_bm25_retrieval" / args.split
              / f"retrieval{suffix}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "retriever": "elasticsearch-bm25",
                "elasticsearch": info.get("version", {}).get("number"),
                "k1": args.k1,
                "b": args.b,
                "analyzer": args.analyzer,
                "top_k": args.top_k,
                "collections": summaries,
            },
            handle,
            indent=2,
        )
    print(f"\n[report] -> {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
