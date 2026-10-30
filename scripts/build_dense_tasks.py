#!/usr/bin/env python
"""Build dense and hybrid retrieval task lists with Elasticsearch.

Runs on the access node, the only host that can reach the search server, and consumes the
vectors written by scripts/embed_corpus.py on a GPU node. Standard library only: this machine
has neither numpy nor an Elasticsearch client.

Two systems are produced alongside the BM25 run:

  dense   approximate nearest neighbours over HNSW, the retrieval Elasticsearch performs itself
  hybrid  BM25 top-`rerank` candidates, rescored exactly by cosine against the same vectors

Both are computed server-side. Reranking in Python would mean a hundred 1024-dimensional dot
products per query across twenty thousand queries, which is hours in an interpreter and seconds
in Lucene, so the candidate set is sent back as an id filter and scored there.

    python3 scripts/build_dense_tasks.py --output-dir experiments --split test --top-k 10
"""

from __future__ import annotations

import argparse
import array
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_es_tasks as bm  # noqa: E402  (same directory, standard library only)


def request(method, url, payload=None, ndjson=False, timeout=180):
    data, headers = None, {}
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
        raise SystemExit(f"{method} {url} -> HTTP {exc.code}\n{exc.read().decode()[:600]}") from exc
    return json.loads(body) if body.strip() else {}


def load_vectors(stem: Path):
    """Ids and a flat float32 buffer.

    Deliberately not a list of lists. Materialising 82,856 vectors of 1,024 dimensions as Python
    floats costs about 2.7 GB and is killed on a shared access node; the flat `array` is 339 MB
    and rows are sliced out of it only when a request is being built.
    """
    meta = json.loads(stem.with_suffix(".meta.json").read_text())
    ids = stem.with_suffix(".ids.txt").read_text(encoding="utf-8").split("\n")
    flat = array.array("f")
    with open(stem.with_suffix(".vecs.f32"), "rb") as handle:
        flat.fromfile(handle, meta["n"] * meta["dim"])
    if len(ids) != meta["n"]:
        raise SystemExit(f"{stem}: {len(ids)} ids for {meta['n']} vectors")
    return ids, flat, meta


def row(flat, dim, index):
    """One vector as a list, built on demand and discarded by the caller."""
    return flat[index * dim : (index + 1) * dim].tolist()


def create_index(es, index, dim, recreate, field="vec", expected=0):
    exists = True
    try:
        request("HEAD", f"{es}/{index}")
    except SystemExit:
        exists = False
    if exists and recreate:
        request("DELETE", f"{es}/{index}")
        exists = False
    if exists:
        # Adding a field rather than rebuilding: each embedder gets its own vector field on the
        # same documents, so several can be compared without re-indexing the corpus or losing
        # the vectors already there.
        mapping = request("GET", f"{es}/{index}/_mapping")
        present = list(mapping.values())[0]["mappings"].get("properties", {})
        if field in present:
            # The mapping is added before any vector is written, so its presence only means a
            # run started, not that it finished. Skipping on that alone would leave part of the
            # corpus without vectors and quietly shrink what dense retrieval can return, so the
            # decision is made on how many documents actually carry the field.
            request("POST", f"{es}/{index}/_refresh")
            covered = request("POST", f"{es}/{index}/_count",
                              {"query": {"exists": {"field": field}}}).get("count", 0)
            if covered >= expected:
                print(f"[es] reusing {index}.{field} ({covered} documents)", flush=True)
                return False
            print(f"[es] {index}.{field} covers {covered} of {expected}; reindexing",
                  flush=True)
            return True
        request("PUT", f"{es}/{index}/_mapping", {"properties": {
            field: {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine"}}})
        print(f"[es] added field {field} to {index}", flush=True)
        return True
    request("PUT", f"{es}/{index}", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            field: {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine"}
        }},
    })
    print(f"[es] created {index} (dense_vector dims={dim}, cosine, HNSW)", flush=True)
    return True


def bulk_index(es, index, ids, flat, dim, field="vec", batch=200):
    total = len(ids)
    started = time.time()
    for start in range(0, total, batch):
        lines = []
        for offset, identifier in enumerate(ids[start : start + batch]):
            vector = row(flat, dim, start + offset)
            # retry_on_conflict lets two embedders write their own vector field to the same
            # document at once. Without it, concurrent updates to one document return 409 and
            # the run dies partway through the corpus.
            lines.append(json.dumps({"update": {"_index": index, "_id": identifier,
                                                "retry_on_conflict": 5}}))
            # Six significant digits: the vectors are unit-norm, so this is far below the
            # precision cosine ranking can resolve, and it roughly halves the payload sent
            # over the tunnel.
            body = '{"' + field + '":[' + ",".join(f"{v:.6g}" for v in vector) + "]}"
            lines.append('{"doc":' + body + ',"doc_as_upsert":true}')
        result = request("POST", f"{es}/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if result.get("errors"):
            first = next(i for i in result["items"] if (i.get("index") or i.get("update") or {}).get("error"))
            raise SystemExit(f"bulk failed: {json.dumps(first)[:400]}")
        if (start // batch) % 50 == 0:
            print(f"[es] indexed {min(start + batch, total)}/{total}", flush=True)
    request("POST", f"{es}/{index}/_refresh")
    print(f"[es] indexing took {time.time() - started:.0f}s", flush=True)



def msearch_batch(es, payload, label, attempts=6):
    """One _msearch, retried while every failure in it is load-related.

    Elasticsearch reports a saturated search thread pool inside the response body rather than as
    an HTTP status, so a rejected batch arrives as a successful request holding failed queries.
    The server is shared and a 1024-dimensional HNSW search is expensive, so rejection under load
    is expected rather than exceptional -- the lexical stage has retried it from the start, and
    without the same handling here a busy minute ends a run that has already done its indexing.

    A non-transient failure still stops immediately: retrying a malformed query would only turn
    a clear error into a slow one.
    """
    for attempt in range(attempts):
        result = request("POST", f"{es}/_msearch", payload, ndjson=True)
        responses = result.get("responses", [])
        failed = [r["error"] for r in responses if "error" in r]
        if not failed:
            return responses
        if not all(bm.transient(error) for error in failed) or attempt == attempts - 1:
            raise SystemExit(f"{label} failed: {json.dumps(failed[0])[:400]}")
        pause = 2 ** attempt
        print(f"[es] {label}: batch rejected under load, retrying in {pause}s", flush=True)
        time.sleep(pause)
    raise SystemExit(f"{label}: exhausted retries")


def knn_search(es, index, flat, dim, count, k, candidates, field="vec", batch=25):
    """Approximate nearest neighbours, Elasticsearch's own HNSW search."""
    rankings = []
    for start in range(0, count, batch):
        lines = []
        for position in range(start, min(start + batch, count)):
            vector = row(flat, dim, position)
            lines.append(json.dumps({"index": index}))
            lines.append(json.dumps({
                "knn": {"field": field, "query_vector": [round(v, 6) for v in vector],
                        "k": k, "num_candidates": candidates},
                "size": k, "_source": False,
            }))
        for response in msearch_batch(es, "\n".join(lines) + "\n", "knn"):
            rankings.append([(h["_id"], h["_score"]) for h in response["hits"]["hits"]])
        if (start // batch) % 40 == 0:
            print(f"[es] knn {min(start + batch, count)}/{count}", flush=True)
    return rankings


def rerank_search(es, index, flat, dim, candidate_sets, k, field="vec", batch=20):
    """Exact cosine rescoring of a fixed candidate set, scored inside Elasticsearch."""
    rankings = []
    count = len(candidate_sets)
    for start in range(0, count, batch):
        lines = []
        for position in range(start, min(start + batch, count)):
            vector = row(flat, dim, position)
            candidates = candidate_sets[position]
            lines.append(json.dumps({"index": index}))
            lines.append(json.dumps({
                "query": {"script_score": {
                    "query": {"ids": {"values": candidates}},
                    "script": {
                        "source": f"cosineSimilarity(params.q, '{field}') + 1.0",
                        "params": {"q": [round(v, 6) for v in vector]},
                    },
                }},
                "size": k, "_source": False,
            }))
        for response in msearch_batch(es, "\n".join(lines) + "\n", "rerank"):
            rankings.append([(h["_id"], h["_score"]) for h in response["hits"]["hits"]])
        if (start // batch) % 40 == 0:
            print(f"[es] rerank {min(start + batch, count)}/{count}", flush=True)
    return rankings


def emit(items, corpus_docs, gold, rankings, top_k, path, retriever, base_meta):
    """Write one task list, its context replaced by what this retriever returned."""
    stats = {2: [], 5: [], 10: [], 20: []}
    complete = 0
    rows = []
    for item, ranking in zip(items, rankings):
        item_gold = gold[item["id"]]
        retrieved = [identifier for identifier, _ in ranking[:top_k]]
        for k, values in stats.items():
            top = {identifier for identifier, _ in ranking[:k]}
            values.append(
                sum(1 for g in item_gold if g in top) / len(item_gold) if item_gold else 0.0
            )
        found = [g for g in item_gold if g in retrieved]
        complete += 1 if item_gold and len(found) == len(item_gold) else 0
        context = [[corpus_docs[i][0], list(corpus_docs[i][1])] for i in retrieved]
        rows.append({
            "id": item["id"], "question": item["question"], "answer": item["answer"],
            "type": item["type"], "level": item["level"], "context": context,
            "supporting_facts": [list(f) for f in item["supporting_facts"]],
            "meta": dict(item["meta"], **base_meta, documents=len(context),
                         supporting_documents=len({f[0] for f in item["supporting_facts"]}),
                         supporting_facts=len(item["supporting_facts"]),
                         gold_documents=len(item_gold), gold_retrieved=len(found),
                         full_support=bool(item_gold) and len(found) == len(item_gold),
                         retrieval=retriever),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {f"recall@{k}": sum(v) / len(v) for k, v in sorted(stats.items())}
    summary[f"all_gold_in_top_{top_k}"] = complete / len(rows) if rows else 0.0
    print(f"[tasks] {path.name}: {len(rows)} items  " +
          "  ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Dense and hybrid retrieval task lists.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", type=int, default=100, help="BM25 candidates to rescore.")
    parser.add_argument("--candidates", type=int, default=200, help="HNSW num_candidates.")
    parser.add_argument("--es", default="http://localhost:9200")
    parser.add_argument("--index-prefix", default="lorehopqa")
    parser.add_argument("--tag", default="jina", help="Which embedder's vectors to use.")
    parser.add_argument("--field", default="vec", help="Vector field; one per embedder.")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    root = args.output_dir / "e5_bm25_retrieval" / args.split
    info = request("GET", args.es)
    print(f"[es] {args.es} -> {info.get('version', {}).get('number')}", flush=True)

    summaries = {}
    meta = {}
    for name in ("lore", "hotpot"):
        stem = root / "vectors" / f"{name}-docs-{args.tag}"
        if not stem.with_suffix(".meta.json").exists():
            print(f"[skip] no vectors for {name}", flush=True)
            continue

        # The corpus is rebuilt from the released files with the same identifier function the
        # BM25 stage used, so document ids agree across all three retrievers and a dense hit
        # outside BM25's top-10 still resolves to its text.
        lore = bm.load_lorehopqa(args.dist_dir, args.split)
        if name == "lore":
            items = lore
        else:
            source_ids = {
                item["meta"]["source_id"] for item in lore
                if item["meta"].get("origin") == "transposed" and item["meta"].get("source_id")
            }
            items = bm.load_hotpot(args.hotpot, source_ids)
        corpus_docs, gold = bm.build_corpus(items)

        doc_ids, doc_flat, meta = load_vectors(stem)
        query_ids, query_flat, _ = load_vectors(root / "vectors" / f"{name}-queries-{args.tag}")
        dim = meta["dim"]
        order = {item["id"]: item for item in items}
        items = [order[qid] for qid in query_ids]
        print(f"[{name}] {len(doc_ids)} documents, {len(query_ids)} queries, dim {meta['dim']}",
              flush=True)

        index = f"{args.index_prefix}-{args.split}-{name}-dense"
        if create_index(args.es, index, dim, args.recreate, args.field, len(doc_ids)):
            bulk_index(args.es, index, doc_ids, doc_flat, dim, args.field)
        del doc_flat

        base_meta = {"embedder": meta.get("model")}
        # The first embedder was written before there were others, and its task lists are already
        # scored, so its names stay unsuffixed rather than invalidating those runs.
        def system_name(system: str) -> str:
            return f"{name}-{system}" if args.tag == "jina" else f"{name}-{system}-{args.tag}"

        def task_path(system: str) -> Path:
            return root / "tasks" / f"{system_name(system)}.jsonl"

        started = time.time()
        dense = knn_search(args.es, index, query_flat, dim, len(query_ids),
                           args.top_k, args.candidates, args.field)
        print(f"[{name}] dense search {time.time() - started:.0f}s", flush=True)
        summaries[system_name("dense")] = emit(
            items, corpus_docs, gold, dense, args.top_k, task_path("dense"),
            f"dense-{args.tag}", base_meta)

        # Candidates for reranking come from the BM25 index built by build_es_tasks.py.
        lexical_index = f"{args.index_prefix}-{args.split}-{name}"
        started = time.time()
        lexical = bm.msearch(args.es, lexical_index,
                             [item["question"] for item in items], args.rerank)
        candidate_sets = [[identifier for identifier, _ in ranking] for ranking in lexical]
        hybrid = rerank_search(args.es, index, query_flat, dim, candidate_sets, args.top_k, args.field)
        print(f"[{name}] hybrid rerank {time.time() - started:.0f}s", flush=True)
        summaries[system_name("hybrid")] = emit(
            items, corpus_docs, gold, hybrid, args.top_k, task_path("hybrid"),
            f"bm25+cosine-{args.tag}", base_meta)

    report = root / f"retrieval-dense-{args.tag}.json"
    report.write_text(json.dumps({"embedder": meta.get("model"), "top_k": args.top_k,
                                  "rerank_depth": args.rerank, "systems": summaries}, indent=2))
    print(f"\n[report] -> {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
