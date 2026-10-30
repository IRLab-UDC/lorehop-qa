#!/usr/bin/env python
"""E5 -- End-to-end RAG with a real retriever.

Experiments 1 to 4 hand the reader an oracle context: the gold documents plus a fixed set of
distractors, as HotpotQA's distractor setting does. That measures reading. This one measures the
thing the collection was built to let anyone measure --- whether end-to-end answer accuracy
tracks retrieval quality --- by ranking every document in the collection with BM25 and giving the
reader only the top of that ranking.

THE RANKING IS BUILT ELSEWHERE, AND THIS SCRIPT WILL NOT BUILD IT.  Retrieval runs against an
Elasticsearch instance reachable only from the Slurm access node, so `scripts/build_es_tasks.py`
produces the task lists there and this script consumes them. That separation is also what keeps
the experiment meaningful: one ranking, shared by every reader in the panel. If each reader
retrieved for itself the six would be answering six different collections, and the per-item
comparisons between them would say nothing. A missing task list is therefore an error with
instructions, never a quiet fallback to some other retriever.

    # first, on the access node
    python3 scripts/build_es_tasks.py --dist-dir dist --hotpot <hotpot.json> \\
        --output-dir experiments --split test --top-k 10

    # then, per reader, on a GPU node
    PYTHONPATH=src python scripts/experiment_5_bm25_retrieval.py \\
        --config configs/readers.yaml --model qwen \\
        --dist-dir dist --output-dir experiments --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lorehopqa.experiments import (  # noqa: E402
    OPEN_BOOK,
    Paths,
    TaskSet,
    add_common_arguments,
    main_guard,
    run_experiment,
)

EXPERIMENT = "e5_bm25_retrieval"

AXES = ("origin", "type", "level", "full_support", "gold_retrieved")


def require_tasks(paths: Paths, name: str) -> TaskSet:
    """Load a ranking built by `build_es_tasks.py`, or explain how to produce it.

    Deliberately not a fallback. Silently ranking with something else would produce numbers that
    look exactly like the reported ones and answer a different question.
    """
    path = paths.tasks(name)
    if not path.exists():
        raise SystemExit(
            f"No retrieval task list at {path}.\n"
            "E5 consumes a ranking built against Elasticsearch, which is reachable only from the\n"
            "Slurm access node. Build it there first:\n\n"
            "    python3 scripts/build_es_tasks.py --dist-dir dist \\\n"
            "        --hotpot <hotpot_dev_distractor_v1.json> \\\n"
            f"        --output-dir <output-dir> --split {paths.split} --top-k 10\n"
        )
    return TaskSet.materialise(name, path, lambda: [], rebuild=False)


def build_datasets(paths: Paths, args: argparse.Namespace) -> dict[str, tuple[TaskSet, str]]:
    """Every retrieval system whose ranking has been built, over both collections.

    The rankings come from build_es_tasks.py (lexical) and build_dense_tasks.py (dense and
    hybrid). Whichever exist are run; the BM25 pair is required, because a run with no lexical
    baseline has nothing to compare the others against.
    """
    datasets: dict[str, tuple[TaskSet, str]] = {}
    for collection in ("lore", "hotpot"):
        # Systems are discovered rather than listed, so a ranking from a newly added embedder
        # (lore-dense-simcse and the like) is swept as soon as its task list exists instead of
        # being dropped for not matching a hardcoded name.
        found = sorted(paths.tasks(f"{collection}-*").parent.glob(f"{collection}-*.jsonl"))
        for path in found:
            # --systems narrows the sweep to named systems (the part after "lore-"/"hotpot-"). The
            # adapted embedders' rankings are only ever scored for recall (the paper's Table 8),
            # so running every reader over them would be GPU time no table reads.
            system = path.stem.split("-", 1)[1]
            if args.systems and system not in args.systems:
                continue
            datasets[path.stem] = (require_tasks(paths, path.stem), OPEN_BOOK)
    # A narrowed run leaves the baseline to whichever job names bm25: a reader's sweep split into
    # one job per system would otherwise have every job answer lore-bm25, concurrently, into the
    # same append-only prediction file.
    if "lore-bm25" not in datasets and not args.systems:
        datasets["lore-bm25"] = (require_tasks(paths, "lore-bm25"), OPEN_BOOK)
    missing = [n for n in ("lore-bm25", "hotpot-bm25") if n not in datasets and not args.systems]
    if missing:
        print(f"[e5] no task list for {', '.join(missing)}", flush=True)
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="E5: end-to-end RAG with BM25 retrieval.")
    add_common_arguments(parser)
    parser.add_argument(
        "--systems", type=lambda value: set(filter(None, value.split(","))), default=set(),
        help="comma-separated retrieval systems to run (e.g. bm25,dense,hybrid-minilm); default all",
    )
    args = parser.parse_args()
    return run_experiment(args, experiment=EXPERIMENT, datasets=build_datasets, axes=AXES)


if __name__ == "__main__":
    main_guard(main)
