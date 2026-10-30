#!/usr/bin/env python3
"""Emit the ablation table over one matched population, and the survival rates behind it.

The two sides of the table have to describe the same items. \\ds\\ generates variants for the
whole subsample, but only its transposed items have a HotpotQA counterpart, so a comparison that
uses all of them on one side and the paired subset on the other is not paired at all. Everything
here is restricted to the transposed items and their sources.

Variant counts are reported alongside item counts because the columns average over variants.

    python3 scripts/ablation_table.py --experiments experiments --split test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

MODELS = list(PANEL_NAMES)
NAMES = dict(DISPLAY)
POOL = set(POOL_NAMES)


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def survival(tasks: Path, name: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    path = tasks / f"{name}.jsonl"
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        answer = norm(row["answer"])
        if not answer:
            continue
        text = norm(" ".join(s for _, ss in row["context"] for s in ss))
        out[row["id"]] = answer in text
    return out


def rows(path: Path) -> list[dict]:
    return ([json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            if path.exists() else [])


def boot(diffs, resamples=2000, seed=20260901):
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--latex", action="store_true",
                        help="Emit the table body, so the paper is not transcribed by hand.")
    args = parser.parse_args()
    base = args.experiments / "e2_ablated_evidence" / args.split
    tasks = base / "tasks"

    # The transposed items of the subsample are the only ones with a counterpart.
    full_rows = rows(tasks / "lore-full.jsonl")
    transposed = {r["id"] for r in full_rows if r.get("meta", {}).get("origin") == "transposed"}
    print(f"[population] {len(transposed)} transposed items of {len(full_rows)} in the subsample")

    report: dict = {"transposed_items": len(transposed)}
    for side, loo_name, full_name, closed in (
            ("lore", "lore-loo", "lore-full", "lorehopqa"),
            ("hotpot", "hotpot4-loo", "hotpot4-full", "hotpot-paired")):
        keep = (lambda i: i in transposed) if side == "lore" else (lambda i: True)
        surv = survival(tasks, loo_name)
        variants = [i for i, s in surv.items() if keep(i.split("::")[0])]
        removed = [i for i in variants if surv[i] is False]
        print(f"[{side}] {len(variants)} variants, answer removed in {len(removed)} "
              f"({len(removed)/len(variants):.1%}), survives {1-len(removed)/len(variants):.1%}")
        report[side] = {"variants": len(variants), "answer_removed": len(removed),
                        "survival_rate": round(1 - len(removed) / len(variants), 4)}

        for model in MODELS:
            loo = rows(base / "predictions" / f"{loo_name}.{model}.jsonl")
            full = rows(base / "predictions" / f"{full_name}.{model}.jsonl")
            book = {r["id"]: r for r in rows(args.experiments / "e1_closed_book" / args.split /
                                             "predictions" / f"{closed}.{model}.jsonl")}
            if not loo or not full:
                continue
            loo = [r for r in loo if keep((r.get("ablation_of") or r["id"]).split("::")[0])]
            full = [r for r in full if keep(r["id"])]
            ablated = [r for r in loo
                       if surv.get(r["id"]) is False
                       and not (r.get("is_yes_no") or r.get("answer_in_question"))]
            paired = [(r, book[r["ablation_of"]]) for r in ablated
                      if r.get("ablation_of") in book]
            diffs = [a["answer_f1"] - b["answer_f1"] for a, b in paired]
            low, high = boot(diffs) if diffs else (0.0, 0.0)
            report.setdefault("cells", {}).setdefault(model, {})[side] = {
                "full": round(statistics.fmean(r["answer_f1"] for r in full), 4),
                "loo": round(statistics.fmean(r["answer_f1"] for r in loo), 4),
                "ablated": round(statistics.fmean(r["answer_f1"] for r in ablated), 4),
                "ablated_variants": len(ablated),
                "vs_closed_book": round(statistics.fmean(diffs), 4) if diffs else None,
                "ci_low": round(low, 4), "ci_high": round(high, 4),
                "null": low <= 0.0 <= high,
            }

    print()
    print(f"{'model':9s}" + "".join(f"{h:>10s}" for h in
          ("L-full", "L-loo", "L-abl", "L-vsCB", "H-full", "H-loo", "H-abl", "H-vsCB")))
    for model in MODELS:
        cell = report.get("cells", {}).get(model)
        if not cell:
            continue
        row = f"{NAMES[model]:9s}"
        for side in ("lore", "hotpot"):
            c = cell.get(side, {})
            row += f"{c.get('full', 0):10.3f}{c.get('loo', 0):10.3f}{c.get('ablated', 0):10.3f}"
            v = c.get("vs_closed_book")
            row += f"{v:+10.3f}" % () if False else f"{v:+10.3f}" if v is not None else f"{'--':>10s}"
        print(row)
    for model in MODELS:
        cell = report.get("cells", {}).get(model)
        if cell:
            print(f"  {NAMES[model]:8s} ablated variants: "
                  + ", ".join(f"{s}={cell[s]['ablated_variants']}" for s in cell))
            break

    if args.latex:
        cells = report.get("cells", {})
        # A model missing either side is a run still in flight, not a row: including it would
        # put a zero in the table and a wrong mean under it.
        complete = [m for m in MODELS
                    if m in cells and {"lore", "hotpot"} <= set(cells[m])]
        skipped = [m for m in MODELS if m in cells and m not in complete]
        if skipped:
            print(f"% incomplete, omitted: {', '.join(skipped)}")
        groups = [("Families participating in construction",
                   [m for m in complete if m in POOL]),
                  ("Control families",
                   [m for m in complete if m not in POOL])]
        print("\n% --- generated by scripts/ablation_table.py ---")
        for title, members in groups:
            # A full-width span carries the vertical rule past the last column;
            # splitting at the rule keeps the separator and leaves no trailing edge.
            print(f"\\multicolumn{{5}}{{l|}}{{\\emph{{{title}}}}} & "
                  f"\\multicolumn{{4}}{{l}}{{}}\\\\")
            for model in members:
                cell, out = cells[model], [f"{NAMES[model]:8s}"]
                for side in ("lore", "hotpot"):
                    c = cell[side]
                    out += [f"{c['full']:.3f}", f"{c['loo']:.3f}", f"{c['ablated']:.3f}"]
                    mark = "^{\\dag}" if c["null"] else ""
                    out.append(f"${c['vs_closed_book']:+.3f}{mark}$")
                print(" & ".join(out) + "\\\\")
            avg = []
            for side in ("lore", "hotpot"):
                for key in ("full", "loo", "ablated", "vs_closed_book"):
                    values = [cells[m][side][key] for m in members]
                    mean = sum(values) / len(values)
                    avg.append(f"${mean:+.3f}$" if key == "vs_closed_book" else f"{mean:.3f}")
            print("\\emph{mean} & " + " & ".join(avg) + "\\\\")
            if title.startswith("Families"):
                print("\\midrule")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\n[report] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
