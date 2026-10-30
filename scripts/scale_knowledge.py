#!/usr/bin/env python3
"""Does model size predict what an LLM answers without documents?

On a collection of real entities it should: parametric knowledge grows with scale. On a
collection of invented ones there is nothing for scale to have learned, so the relationship
should disappear. This fits closed-book answer F1 against parameter count for both collections
and writes the result where the paper's verification pass can reach it.

Six models is a small sample and they differ in training data and release date as well as in
size, so the fit bounds the observation rather than isolating scale.

    python3 scripts/scale_knowledge.py --experiments experiments --config configs/readers.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

# The reported panel, which must match PARTICIPATING + CONTROL in paper_tables.py.
MODELS = list(PANEL_NAMES)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # parameters_b lives in the reader panel, so the sizes come from the same place the runs did.
    sizes: dict[str, float] = {}
    name = None
    for line in args.config.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("parameters_b:") and name:
            sizes[name] = float(stripped.split(":", 1)[1])

    metrics = args.experiments / "e1_closed_book" / args.split / "metrics"
    rows = []
    for model in MODELS:
        lore = metrics / f"lorehopqa.{model}.json"
        hotpot = metrics / f"hotpot-paired.{model}.json"
        if not (lore.exists() and hotpot.exists() and model in sizes):
            print(f"[skip] {model}")
            continue
        def score(path: Path) -> float:
            body = json.loads(path.read_text())
            return (body.get("excluding_guessable") or body["overall"])["answer_f1"]
        rows.append((model, sizes[model], score(hotpot), score(lore)))

    if len(rows) < 3:
        raise SystemExit("too few models to fit")

    log_size = [math.log(r[1]) for r in rows]
    report = {
        "models": {r[0]: {"parameters_b": r[1], "hotpot_f1": round(r[2], 4),
                          "lore_f1": round(r[3], 4)} for r in rows},
        "n": len(rows),
        # Four decimals, not three. The paper quotes these to two, and a coefficient such as
        # -0.156236 stored as -0.155 no longer rounds to the figure the paper prints, so the
        # artefact and the manuscript disagree over a digit neither of them got wrong.
        "log_size_vs_hotpot_r": round(pearson(log_size, [r[2] for r in rows]), 4),
        "log_size_vs_lore_r": round(pearson(log_size, [r[3] for r in rows]), 4),
        "size_vs_hotpot_r": round(pearson([r[1] for r in rows], [r[2] for r in rows]), 4),
        "size_vs_lore_r": round(pearson([r[1] for r in rows], [r[3] for r in rows]), 4),
    }
    print(f"{'model':9s}{'params':>8s}{'HotpotQA':>10s}{'LoreHopQA':>11s}")
    for model, size, hot, lore in sorted(rows, key=lambda r: -r[1]):
        print(f"{model:9s}{size:8.1f}{hot:10.3f}{lore:11.3f}")
    print(f"\n  log-size vs HotpotQA  r = {report['log_size_vs_hotpot_r']:+.3f}")
    print(f"  log-size vs LoreHopQA r = {report['log_size_vs_lore_r']:+.3f}   (n={report['n']})")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\n[report] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
