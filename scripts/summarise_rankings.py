#!/usr/bin/env python3
"""Collect end-to-end results across retrieval strategies and embedders.

Reads the per-run metric files written by experiment_5 and the retrieval reports written by
build_es_tasks.py / build_dense_tasks.py, and prints one table per collection. Systems are
discovered from the metric filenames, so a new embedder appears here as soon as its runs land.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

MODELS = list(PANEL_NAMES)


def retrieval_quality(root: Path) -> dict[str, dict]:
    # The BM25 stage keys its report "collections" and the dense stage "systems"; both hold the
    # same recall fields, so either spelling is accepted rather than normalised after the fact.
    quality = {}
    for report in sorted(root.glob("retrieval*.json")):
        body = json.loads(report.read_text())
        for section in ("systems", "collections"):
            for name, summary in body.get(section, {}).items():
                quality[name] = summary.get("recall", summary)
    return quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    quality = retrieval_quality(args.root)
    print("=== retrieval quality ===")
    for name in sorted(quality):
        r = quality[name]
        recall = r.get("recall@10")
        allgold = r.get("all_gold_in_top_10")
        parts = [f"  {name:26s}"]
        parts.append(f"R@10={recall:.3f}" if isinstance(recall, (int, float)) else "R@10=   -")
        parts.append(f"all-gold={allgold:.3f}" if isinstance(allgold, (int, float)) else "")
        print(" ".join(parts))

    metrics = args.root / "metrics"
    systems: dict[str, set[str]] = {}
    for path in sorted(metrics.glob("*.json")):
        match = re.fullmatch(r"(lore|hotpot)-([a-z0-9+-]+)\.([a-z0-9]+)\.json", path.name)
        if match:
            systems.setdefault(match.group(1), set()).add(match.group(2))

    for collection in ("lore", "hotpot"):
        names = sorted(systems.get(collection, ()))
        if not names:
            continue
        print(f"\n=== {collection}: answer F1 / SP F1 ===")
        header = f"{'model':9s}" + "".join(f"{n:>18s}" for n in names)
        print(header)
        for model in MODELS:
            row = f"{model:9s}"
            for name in names:
                path = metrics / f"{collection}-{name}.{model}.json"
                if not path.exists():
                    row += f"{'-':>18s}"
                    continue
                overall = json.loads(path.read_text()).get("overall", {})
                row += f"{overall.get('answer_f1', 0):9.3f}{overall.get('sp_f1', 0):9.3f}"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
