#!/usr/bin/env python
"""E2 -- Answer quality under ablated evidence.

For every item, one variant per supporting document, each with a single supporting document
withheld. The withheld document is replaced by an additional distractor from the same invented
world rather than deleted: deletion shortens the context and drops the number of candidates by
one, so a fall in accuracy would confound the loss of evidence with the change of layout.

An item whose hops are genuinely required becomes unanswerable when a required fact is removed,
so leave-one-out performance on \\ds should fall to the closed-book floor E1 measured. Anything
appreciably above the floor marks items where the answer survives the loss of a fact the
annotation calls necessary, and that residual is a property of the collection, not of the reader.

The comparison with HotpotQA is what the experiment is for. There a withheld gold paragraph can
be replaced from the model's parameters, so the drop should be markedly shallower. Both directions
of the layout control run, because neither is neutral: `hotpot4` restricts HotpotQA to the gold
pair plus two distractors, which is our transposed layout and makes HotpotQA easier than it is;
`lore10` pads our paired items to ten documents, which is HotpotQA's layout and makes ours harder.
A conclusion that survives only one of them is not supported.

    PYTHONPATH=src python scripts/experiment_2_ablated_evidence.py \\
        --config configs/readers.yaml --model qwen \\
        --dist-dir dist --hotpot /path/hotpot_dev_distractor_v1.json \\
        --output-dir experiments --split test --sample 5000
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
    build_ablations,
    limited,
    main_guard,
    rng_for,
    run_experiment,
    stratified_sample,
)
from lorehopqa.reader import (  # noqa: E402
    DistractorPool,
    assign_worlds,
    load_hotpot,
    load_lorehopqa,
    pad_context,
    restrict_context,
)

EXPERIMENT = "e2_ablated_evidence"

# The composition the subsample must preserve. Origin and question family carry the breakdowns
# E2 reports; supporting_documents decides how many variants an item contributes, so a subsample
# that drifted on it would change the size of the run as well as its shape.
STRATA = ("origin", "type", "level", "supporting_documents")


def build_datasets(paths: Paths, args: argparse.Namespace) -> dict[str, tuple[TaskSet, str]]:
    every = limited(load_lorehopqa(paths.lorehopqa, paths.provenance), args.limit)
    require_withheld(paths, lore=True)
    worlds = assign_worlds(paths.provenance, paths.lore_dir)

    # The pool is drawn from the whole split, not from the subsample: a replacement paragraph is
    # only required to be same-world and answer-free, and restricting the supply to the sampled
    # items would make replacements repeat within a world.
    lore_pool = DistractorPool.build(every, worlds)

    sample = stratified_sample(
        every, size=args.sample, keys=STRATA, rng=rng_for(args.seed, "e2", "sample")
    )
    paired = [item for item in sample if item.meta.get("origin") == "transposed"]
    source_ids = {item.meta["source_id"] for item in paired if item.meta.get("source_id")}

    datasets: dict[str, tuple[TaskSet, str]] = {}

    def add(name: str, build) -> None:
        datasets[name] = (
            TaskSet.materialise(name, paths.tasks(name), build, rebuild=args.rebuild_tasks),
            OPEN_BOOK,
        )

    # --- \ds, as released ---------------------------------------------------------------
    add("lore-full", lambda: sample)
    add(
        "lore-loo",
        lambda: build_ablations(
            sample, pool=lore_pool, worlds=worlds, seed=args.seed, tag="lore-loo"
        ),
    )

    # --- \ds, padded to HotpotQA's ten documents (one direction of the layout control) ----
    def padded() -> list:
        return [
            pad_context(
                item,
                documents=10,
                pool=lore_pool,
                world=worlds.get(item.id, "default"),
                rng=rng_for(args.seed, "e2", "pad", item.id),
            )
            for item in paired
        ]

    add("lore10-full", padded)
    add(
        "lore10-loo",
        lambda: build_ablations(
            padded(), pool=lore_pool, worlds=worlds, seed=args.seed, tag="lore10-loo"
        ),
    )

    if paths.hotpot is None:
        return datasets

    hotpot = load_hotpot(paths.hotpot, keep_ids=source_ids)
    hotpot_pool = DistractorPool.build(hotpot)
    hotpot_worlds = {item.id: "hotpotqa" for item in hotpot}

    # --- HotpotQA, as distributed -------------------------------------------------------
    add("hotpot-full", lambda: hotpot)
    add(
        "hotpot-loo",
        lambda: build_ablations(
            hotpot, pool=hotpot_pool, worlds=hotpot_worlds, seed=args.seed, tag="hotpot-loo"
        ),
    )

    # --- HotpotQA, cut to our four documents (the other direction) ----------------------
    def restricted() -> list:
        return [
            restrict_context(
                item, distractors=2, rng=rng_for(args.seed, "e2", "restrict", item.id)
            )
            for item in hotpot
        ]

    add("hotpot4-full", restricted)
    add(
        "hotpot4-loo",
        lambda: build_ablations(
            restricted(),
            pool=hotpot_pool,
            worlds=hotpot_worlds,
            seed=args.seed,
            tag="hotpot4-loo",
        ),
    )

    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="E2: answer quality under ablated evidence.")
    add_common_arguments(parser)
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Stratified subsample of the split to ablate (default 5000; 0 or negative for all).",
    )
    args = parser.parse_args()
    if args.sample <= 0:
        args.sample = 10**9
    return run_experiment(args, experiment=EXPERIMENT, datasets=build_datasets)


if __name__ == "__main__":
    main_guard(main)
