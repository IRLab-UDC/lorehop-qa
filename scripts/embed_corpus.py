#!/usr/bin/env python
"""Embed the retrieval corpora and queries for the dense and hybrid runs.

Runs on a GPU node and writes vectors to the shared tree. The access node, which is the only
host that can reach Elasticsearch, has no numpy and no GPU, so this stage cannot happen there;
equally the GPU nodes have no route to Elasticsearch, so indexing cannot happen here. Vectors
therefore travel between the two as a plain float32 file that the standard library can read.

Batching is the reason to do this offline rather than per query at search time: 135k documents
and 20k queries go through one loaded model in a single pass, which is minutes rather than the
hours a per-request encoder would cost.

The prompts and pooling are the model's own, taken from its sentence-transformers configuration:
queries are prefixed "Query: ", documents "Document: ", pooling is last-token, and vectors are
L2-normalised. Getting either prefix wrong costs retrieval quality silently, so the script
verifies the norms before writing anything.

    PYTHONPATH=src python scripts/embed_corpus.py \
        --dist-dir dist --hotpot <hotpot.json> --output-dir <tree> --split test
"""

from __future__ import annotations

import argparse
import array
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The corpus and its identifiers come from the BM25 stage, deliberately. The two stages index
# the same documents into two Elasticsearch indices and the hybrid run sends candidate ids from
# one to the other, so a second definition of "which document is this" silently breaks both the
# join and the recall counts. Sharing the function makes that impossible.
import build_es_tasks as bm  # noqa: E402

QUERY_PROMPT = "Query: "
DOCUMENT_PROMPT = "Document: "


def write_vectors(path: Path, ids: list[str], vectors: list[list[float]], model: str, prompt: str):
    """Vectors as raw float32, ids as text, shape in a sidecar.

    Chosen so the access node can read them with `array` from the standard library: it has no
    numpy, and a JSON encoding of 135k x 1024 floats would be gigabytes of text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dim = len(vectors[0])
    flat = array.array("f")
    for vector in vectors:
        flat.extend(vector)
    with open(path.with_suffix(".vecs.f32"), "wb") as handle:
        flat.tofile(handle)
    path.with_suffix(".ids.txt").write_text("\n".join(ids), encoding="utf-8")
    path.with_suffix(".meta.json").write_text(
        json.dumps({"n": len(ids), "dim": dim, "model": model, "prompt": prompt}, indent=2),
        encoding="utf-8",
    )
    size = len(ids) * dim * 4 / 1e6
    print(f"[write] {path.name}: {len(ids)} x {dim} ({size:.0f} MB)", flush=True)


def load_prior(stem: Path, model: str, prompt: str, label: str):
    """Vectors from an earlier run, indexed by id, or None if there are none to reuse.

    Refuses anything encoded differently. A vector is only interchangeable with one produced by
    the same model under the same prompt prefix: reusing a vector written by another embedder,
    or written without the "Document: " prefix this run applies, puts two incompatible
    geometries in one index, and the only visible effect is a lower recall that reads as a
    result rather than as a bug.
    """
    meta_path = stem.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("model") != model or meta.get("prompt") != prompt:
        print(f"[{label}] not reusing {stem.name}: written by {meta.get('model')!r} with "
              f"prompt {meta.get('prompt')!r}, this run uses {model!r} / {prompt!r}", flush=True)
        return None
    ids = stem.with_suffix(".ids.txt").read_text(encoding="utf-8").split("\n")
    if len(ids) != meta["n"]:
        raise SystemExit(f"{stem}: {len(ids)} ids for {meta['n']} vectors")
    flat = array.array("f")
    with open(stem.with_suffix(".vecs.f32"), "rb") as handle:
        flat.fromfile(handle, meta["n"] * meta["dim"])
    return {"index": {i: n for n, i in enumerate(ids)}, "flat": flat, "dim": meta["dim"]}


def encode_reusing(llm, ids, texts, reuse_from, stem_name, model, prompt, fit, label):
    """Encode only the ids an earlier run did not already cover.

    Safe for documents and for documents only, because a document's id is sha1 of its title and
    sentences: repairing a document gives it a new id, so an id found in the earlier run names
    text that has not changed. The overwhelming majority of a rebuilt corpus is in that position
    and does not need the GPU a second time.

    Queries are the opposite case and are never reused. A query is keyed by item id, which is
    stable across a repair even when the question was rewritten, so matching on it would pair a
    repaired question with the vector of the question it replaced -- and nothing downstream
    could show that had happened. Queries are also the cheap half: 14.5k against 83k documents.

    The vectors come back in the order `ids` gives, so the caller's ids.txt still lines up with
    the buffer; only the source of each row differs.
    """
    prior = None
    if reuse_from:
        # The first embedder ran before any others existed and wrote its vectors with no tag in
        # the name, so a run tagged "jina" has to look under both spellings or it would be the
        # one model that re-encodes the whole corpus for want of a suffix.
        candidates = [stem_name]
        if stem_name.endswith("-jina"):
            candidates.append(stem_name[: -len("-jina")])
        for candidate in candidates:
            prior = load_prior(reuse_from / candidate, model, prompt, label)
            if prior is not None:
                break
    if prior is None:
        return embed(llm, fit(texts, label), label)

    missing = [n for n, i in enumerate(ids) if i not in prior["index"]]
    print(f"[{label}] reusing {len(ids) - len(missing):,} vectors, encoding {len(missing):,}",
          flush=True)
    fresh = {}
    if missing:
        encoded = embed(llm, fit([texts[n] for n in missing], label), label)
        if len(encoded[0]) != prior["dim"]:
            raise SystemExit(f"{label}: new vectors are {len(encoded[0])}-dimensional, "
                             f"the reused ones are {prior['dim']}")
        fresh = dict(zip(missing, encoded))

    dim, flat = prior["dim"], prior["flat"]
    out = []
    for n, identifier in enumerate(ids):
        if n in fresh:
            out.append(fresh[n])
        else:
            row = prior["index"][identifier]
            out.append(flat[row * dim:(row + 1) * dim].tolist())
    return out


def embed(llm, texts: list[str], label: str) -> list[list[float]]:
    started = time.time()
    outputs = llm.embed(texts)
    vectors = [list(output.outputs.embedding) for output in outputs]

    # The model normalises; a norm far from 1 means the pooling or the prompt handling is not
    # what the model's configuration specifies, and every cosine downstream would be wrong.
    norms = [sum(value * value for value in vector) ** 0.5 for vector in vectors[:64]]
    worst = max(abs(norm - 1.0) for norm in norms)
    if worst > 0.05:
        raise SystemExit(f"{label}: vectors are not unit norm (worst deviation {worst:.3f})")
    print(
        f"[embed] {label}: {len(vectors)} vectors in {time.time() - started:.0f}s "
        f"(norm deviation {worst:.4f})",
        flush=True,
    )
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed corpora and queries for dense retrieval.")
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--hotpot", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--model", default="jinaai/jina-embeddings-v5-text-small-retrieval")
    parser.add_argument("--tag", default="jina", help="Names this embedder's vector files and index field.")
    parser.add_argument("--no-prompts", action="store_true",
                        help="Skip the Query:/Document: prefixes for models that define none.")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-from", type=Path, default=None,
                        help="Vectors directory from an earlier run. Ids present there are\n"
                             "copied instead of re-encoded, provided the same model and the\n"
                             "same prompt prefix produced them.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from vllm import LLM

    lore = bm.load_lorehopqa(args.dist_dir, args.split)
    if args.limit:
        lore = lore[: args.limit]

    collections = {"lore": lore}
    if args.hotpot:
        source_ids = {
            item["meta"]["source_id"]
            for item in lore
            if item["meta"].get("origin") == "transposed" and item["meta"].get("source_id")
        }
        collections["hotpot"] = bm.load_hotpot(args.hotpot, source_ids)

    print(f"[model] loading {args.model}", flush=True)
    llm = LLM(model=args.model, runner="pooling", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85)

    root = args.output_dir / "e5_bm25_retrieval" / args.split / "vectors"
    # Prefixes belong to the model that documents them. Applying Jina's to a model
    # trained without them would embed a literal word, not a role.
    qp, dp = ("", "") if args.no_prompts else (QUERY_PROMPT, DOCUMENT_PROMPT)

    # A few real-Wikipedia paragraphs exceed the context window, and vLLM rejects an over-long
    # input rather than truncating it, which fails the whole job. Truncation is what an embedder
    # does with a long passage anyway; doing it explicitly keeps the failure out of the run and
    # makes the bound visible.
    #
    # The cut has to be measured in tokens. Characters are not a safe proxy: a limit loose enough
    # to keep a 8k-token window busy overshoots a 256-token one, which is how this first failed.
    tokenizer = llm.get_tokenizer()
    try:
        reserve = tokenizer.num_special_tokens_to_add()
    except (AttributeError, NotImplementedError):
        reserve = 2
    budget = args.max_model_len - reserve

    def fit(texts: list[str], label: str) -> list[str]:
        # One token spans at least one character, so anything shorter than the budget in
        # characters is already under it in tokens. Only the long tail is worth tokenising.
        for _ in range(3):
            candidates = [i for i, text in enumerate(texts) if len(text) > budget]
            if not candidates:
                break
            encoded = tokenizer([texts[i] for i in candidates],
                                add_special_tokens=False)["input_ids"]
            over = [(i, ids) for i, ids in zip(candidates, encoded) if len(ids) > budget]
            if not over:
                break
            # Decoding and re-encoding is not exactly stable, so the loop re-measures rather
            # than trusting one pass to land under the limit.
            for i, ids in over:
                texts[i] = tokenizer.decode(ids[:budget - 2])
            print(f"[{label}] {len(over)} of {len(texts)} truncated to {budget} tokens",
                  flush=True)
        return texts

    for name, items in collections.items():
        done = (root / f"{name}-queries-{args.tag}").with_suffix(".meta.json")
        if done.exists() and not args.overwrite:
            print(f"[skip] {name}: vectors already written", flush=True)
            continue
        documents, _ = bm.build_corpus(items)
        print(f"[{name}] {len(items)} queries, {len(documents)} documents", flush=True)

        keys = list(documents)
        docs_text = [dp + documents[k][0] + "\n" + " ".join(documents[k][1]) for k in keys]
        write_vectors(root / f"{name}-docs-{args.tag}", keys,
                      encode_reusing(llm, keys, docs_text, args.reuse_from,
                                     f"{name}-docs-{args.tag}", args.model, dp,
                                     fit, f"{name} docs"),
                      args.model, dp)

        # Queries are always encoded fresh; see encode_reusing for why reuse is unsafe here.
        query_ids = [item["id"] for item in items]
        query_text = fit([qp + item["question"] for item in items], f"{name} queries")
        write_vectors(root / f"{name}-queries-{args.tag}", query_ids,
                      embed(llm, query_text, f"{name} queries"), args.model, qp)

    print(f"\n[done] vectors under {root}", flush=True)
    return 0


if __name__ == "__main__":
    status = main()
    # Every vector is on disk by now. vLLM's engine teardown sometimes aborts the interpreter
    # ("terminate called without an active exception", job 12130), which would fail the job
    # and, through kill_invalid_depend, cancel the whole retrieval stage behind it. Skip it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
