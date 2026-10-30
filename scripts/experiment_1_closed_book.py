#!/usr/bin/env python
"""E1 -- Closed-book answer quality by model family.

Each reader is shown the question and nothing else. The number that comes back is the guessing
floor for every other experiment: on an invented world a reader has no parametric route to the
answer, so anything above chance on \\ds means either the fiction leaked or the question gives
itself away, and the same measurement on the paired HotpotQA items estimates how much of the
source collection can be answered from memory alone.

Two datasets are run, and the pairing between them is the point. `lorehopqa` is the whole split;
`hotpot-paired` is exactly the HotpotQA dev items whose skeletons the transposed subset preserves,
one per transposed item. The per-item difference across that pairing is the leakage estimate:
same question family, same hop structure, same evidence layout, differing only in whether the
world is real.

    PYTHONPATH=src python scripts/experiment_1_closed_book.py \\
        --config configs/readers.yaml --model qwen \\
        --dist-dir dist --hotpot /path/hotpot_dev_distractor_v1.json \\
        --output-dir experiments --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lorehopqa.experiments import (  # noqa: E402
    CLOSED_BOOK,
    Paths,
    TaskSet,
    add_common_arguments,
    limited,
    main_guard,
    run_experiment,
)
from lorehopqa.reader import load_hotpot, load_lorehopqa  # noqa: E402

EXPERIMENT = "e1_closed_book"


def build_datasets(paths: Paths, args: argparse.Namespace) -> dict[str, tuple[TaskSet, str]]:
    items = limited(load_lorehopqa(paths.lorehopqa, paths.provenance), args.limit)

    datasets = {
        "lorehopqa": (
            TaskSet.materialise(
                "lorehopqa",
                paths.tasks("lorehopqa"),
                lambda: items,
                rebuild=args.rebuild_tasks,
            ),
            CLOSED_BOOK,
        )
    }

    if paths.hotpot is not None:
        # The paired subset, not the whole dev split: comparing our items against 7,405 arbitrary
        # HotpotQA items would confound the leakage estimate with whatever difference exists
        # between the transposed skeletons and the ones transposition passed over.
        source_ids = {
            item.meta["source_id"]
            for item in items
            if item.meta.get("origin") == "transposed" and item.meta.get("source_id")
        }
        datasets["hotpot-paired"] = (
            TaskSet.materialise(
                "hotpot-paired",
                paths.tasks("hotpot-paired"),
                lambda: load_hotpot(paths.hotpot, keep_ids=source_ids),
                rebuild=args.rebuild_tasks,
            ),
            CLOSED_BOOK,
        )

    # The nearest existing resources, sampled by scripts/fetch_related_resources.py. Closed-book
    # answerability is the property this paper is built around, so measuring it on PhantomWiki and
    # NovelHopQA under the identical prompt and scorer is what turns a descriptive comparison into
    # an empirical one.
    for related in ("phantomwiki", "novelhopqa"):
        if paths.tasks(related).exists():
            datasets[related] = (
                TaskSet.materialise(related, paths.tasks(related), lambda: [], rebuild=False),
                CLOSED_BOOK,
            )

    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="E1: closed-book answer quality.")
    add_common_arguments(parser)
    args = parser.parse_args()
    return run_experiment(args, experiment=EXPERIMENT, datasets=build_datasets)


if __name__ == "__main__":
    main_guard(main)
