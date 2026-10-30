#!/usr/bin/env python
"""E3 -- Full-context reading with the classic metrics.

Each reader receives the complete context and returns both an answer and a supporting-sentence
set, scored on all six official HotpotQA metrics. This is the baseline table later systems are
compared against, so it is reported for base models with no task-specific fine-tuning.

Supporting-fact F1 carries the weight here. Answer F1 alone cannot separate a system that found
the right evidence from one that reached the right string another way, which is the whole
difficulty the collection addresses; on a collection where the answer exists nowhere but the
supplied documents, the gap between answer F1 and joint F1 isolates readers that answer correctly
while pointing at the wrong sentences.

Two datasets run. `lore` is the split as released. `lore10` is the transposed subset padded to
ten documents, which is the layout HotpotQA distractor items ship in — reported so that the E3
numbers can be read next to a HotpotQA-shaped context rather than only next to our four.

    PYTHONPATH=src python scripts/experiment_3_full_context.py \\
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
    require_withheld,
    TaskSet,
    add_common_arguments,
    limited,
    main_guard,
    rng_for,
    run_experiment,
)
from lorehopqa.reader import (  # noqa: E402
    DistractorPool,
    assign_worlds,
    load_lorehopqa,
    pad_context,
)

EXPERIMENT = "e3_full_context"

# Beyond the axes every experiment reports, E3 is the table that has to answer "which shapes are
# hard", so it also breaks down by how many documents a reader must combine and by the layout.
AXES = (
    "origin",
    "type",
    "level",
    "generator_model",
    "supporting_documents",
    "documents",
)


def build_datasets(paths: Paths, args: argparse.Namespace) -> dict[str, tuple[TaskSet, str]]:
    items = limited(load_lorehopqa(paths.lorehopqa, paths.provenance), args.limit)

    datasets: dict[str, tuple[TaskSet, str]] = {
        "lore": (
            TaskSet.materialise("lore", paths.tasks("lore"), lambda: items, rebuild=args.rebuild_tasks),
            OPEN_BOOK,
        )
    }

    if args.skip_layout_control:
        return datasets

    require_withheld(paths, lore=True)
    worlds = assign_worlds(paths.provenance, paths.lore_dir)
    pool = DistractorPool.build(items, worlds)
    paired = [item for item in items if item.meta.get("origin") == "transposed"]

    def padded() -> list:
        return [
            pad_context(
                item,
                documents=10,
                pool=pool,
                world=worlds.get(item.id, "default"),
                rng=rng_for(args.seed, "e3", "pad", item.id),
            )
            for item in paired
        ]

    datasets["lore10"] = (
        TaskSet.materialise("lore10", paths.tasks("lore10"), padded, rebuild=args.rebuild_tasks),
        OPEN_BOOK,
    )
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="E3: full-context reading, official metrics.")
    add_common_arguments(parser)
    parser.add_argument(
        "--skip-layout-control",
        action="store_true",
        help="Run only the released layout, not the ten-document padding.",
    )
    args = parser.parse_args()
    return run_experiment(args, experiment=EXPERIMENT, datasets=build_datasets, axes=AXES)


if __name__ == "__main__":
    main_guard(main)
