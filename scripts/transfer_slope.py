#!/usr/bin/env python3
"""How much answer quality moves per unit of retrieval quality.

With one retriever, the sensitivity of answering to ranking can only be stated as a difference
between two points. Seven retrieval systems spanning a wide recall range allow it to be estimated
as a slope, per LLM and per collection, which is what an IR practitioner needs in order to judge
whether a ranking improvement is worth its cost on a given collection.
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


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and Pearson r."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / sxx, sxy / (sxx * syy) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None,
                        help="Write the fits to JSON so the paper's figures are checkable.")
    args = parser.parse_args()
    fits: dict[str, dict] = {}

    recall: dict[str, float] = {}
    for report in sorted(args.root.glob("retrieval*.json")):
        body = json.loads(report.read_text())
        for section in ("systems", "collections"):
            for name, summary in body.get(section, {}).items():
                value = summary.get("recall", summary).get("recall@10")
                if value is not None:
                    recall[name] = value

    metrics = args.root / "metrics"
    for collection, label in (("lore", "LoreHopQA"), ("hotpot", "HotpotQA")):
        systems = sorted(n for n in recall if n.startswith(f"{collection}-"))
        print(f"\n=== {label}: recall@10 {min(recall[s] for s in systems):.3f}"
              f"-{max(recall[s] for s in systems):.3f} over {len(systems)} systems ===")
        print(f"{'model':9s}{'dAns/dRec':>11s}{'r':>7s}{'dSP/dRec':>11s}{'r':>7s}")
        for model in MODELS:
            xs, ans, sup = [], [], []
            for system in systems:
                path = metrics / f"{system}.{model}.json"
                if not path.exists():
                    continue
                overall = json.loads(path.read_text()).get("overall", {})
                xs.append(recall[system])
                ans.append(overall.get("answer_f1", 0.0))
                sup.append(overall.get("sp_f1", 0.0))
            if len(xs) < 3:
                continue
            sa, ra = fit(xs, ans)
            ss, rs = fit(xs, sup)
            print(f"{model:9s}{sa:11.3f}{ra:7.3f}{ss:11.3f}{rs:7.3f}")
            fits.setdefault(collection, {})[model] = {
                "answer_slope": round(sa, 3), "answer_r": round(ra, 3),
                "sp_slope": round(ss, 3), "sp_r": round(rs, 3),
                "systems": len(xs), "recall_min": min(xs), "recall_max": max(xs),
            }
    if args.report:
        args.report.write_text(json.dumps(fits, indent=2))
        print(f"\n[report] -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
