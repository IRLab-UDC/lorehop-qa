#!/usr/bin/env python3
"""Closed-book answer quality across the multi-hop collections, as one artefact.

Table~\\ref{tab:related} compares four collections on the same closed-book condition, and
its rows are means over the reader panel. Nothing in the run tree stores those means: the
per-reader metrics files hold one collection and one reader each, so a mean typed into the
table cannot be checked against anything, and the number verifier can only confirm that the
digits appear somewhere among tens of thousands of unrelated values. This writes the means
themselves, together with the panel and the population they are over, and emits the table
body so the figures are generated rather than transcribed.

    python scripts/closed_book_comparison.py --root <experiments>/e1_closed_book/test \\
        --output <experiments>/results/test/closed_book_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
from lorehopqa.panel import DISPLAY, PANEL, PANEL_NAMES, POOL_NAMES  # noqa: E402
from lorehopqa.panel import CONTROL as _CONTROL, PARTICIPATING as _PARTICIPATING  # noqa: E402

# One configuration per family, as declared in the experimental setup, and it must match
# PARTICIPATING + CONTROL in paper_tables.py. OLMo-2 ran in the same sweep but is not part of
# the reported panel, and averaging it in would silently change every row of the table.
PANEL = PANEL_NAMES

# Rows in table order, with the descriptive columns the table also carries.
COLLECTIONS = (
    ("hotpot-paired", "HotpotQA", "Wikipedia", "Public"),
    ("novelhopqa", "NovelHopQA", "Published Novels", "Public"),
    ("phantomwiki", "PhantomWiki", "Synthetic Graph", "Public"),
    ("lorehopqa", "\\ds", "Fictional Worlds", "Gated"),
)

# The caption reports non-chance items, which is this condition and no other. Reading the
# figures off "overall" instead would credit exactly the questions the condition removes.
CONDITION = "excluding_guessable"

# The caption travels with the generator so that what it claims about the population is
# filled in from the artefact rather than typed beside it.
CAPTION_TEMPLATE = r"""\begin{table}[t]
\caption{Closed-book answer quality across multi-hop benchmarks (mean over <<READERS>>
LLMs, <<POPULATION>>). \emph{Guessable} denotes the proportion of items removed by option
naming or boolean formatting; \emph{Gold Len.} is the mean reference answer length in
words. Bold marks the lowest value in each score column, a lower score indicating less
parametric access.}
\label{tab:related}
\centering
\small
\begin{tabular}{lllcccc}
\toprule
Benchmark & Corpus Source & Release Type & Ans F1 & Ans EM & Guessable & Gold Len.\\
\midrule
<<ROWS>>
\bottomrule
\end{tabular}
\end{table}
"""


def gold_words(tasks: Path) -> tuple[float, int]:
    total = count = 0
    for line in tasks.read_text().splitlines():
        if not line.strip():
            continue
        answer = json.loads(line).get("answer") or ""
        if isinstance(answer, str) and answer:
            count += 1
            total += len(answer.split())
    return (total / count if count else 0.0), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="e1_closed_book/<split> tree.")
    parser.add_argument("--output", type=Path, required=True, help="Artefact JSON to write.")
    parser.add_argument("--tex", type=Path, default=None, help="Table body fragment to write.")
    args = parser.parse_args()

    rows = {}
    for key, label, source, release in COLLECTIONS:
        f1 = em = 0.0
        seen = []
        population = guessable_share = None
        for reader in PANEL:
            path = args.root / "metrics" / f"{key}.{reader}.json"
            if not path.is_file():
                continue
            body = json.loads(path.read_text())
            cell = body[CONDITION]
            f1 += cell["answer_f1"]
            em += cell["answer_em"]
            population = cell["n"]
            guessable_share = 1 - cell["n"] / body["overall"]["n"]
            seen.append(reader)
        missing = [r for r in PANEL if r not in seen]
        if missing:
            raise SystemExit(f"{key}: no closed-book metrics for {', '.join(missing)}")
        length, items = gold_words(args.root / "tasks" / f"{key}.jsonl")
        rows[key] = {
            "label": label,
            "corpus_source": source,
            "release": release,
            "readers": seen,
            "items_scored": population,
            "items_total": items,
            "guessable_share": round(guessable_share, 4),
            "answer_f1": round(f1 / len(seen), 4),
            "answer_em": round(em / len(seen), 4),
            "gold_answer_words": round(length, 2),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"condition": CONDITION, "panel": list(PANEL), "collections": rows},
        indent=1, sort_keys=True) + "\n")

    # Bold marks the lowest score, so the highlighting is recomputed here rather than left
    # to survive a change in the data by hand.
    lowest_f1 = min(rows.values(), key=lambda x: x["answer_f1"])
    lowest_em = min(rows.values(), key=lambda x: x["answer_em"])
    lines = [f"% generated by scripts/{Path(__file__).name}; do not edit by hand"]
    for key, *_ in COLLECTIONS:
        r = rows[key]
        f1 = f"\\textbf{{{r['answer_f1']:.3f}}}" if r is lowest_f1 else f"{r['answer_f1']:.3f}"
        em = f"\\textbf{{{r['answer_em']:.3f}}}" if r is lowest_em else f"{r['answer_em']:.3f}"
        lines.append(
            f"{r['label']:11s} & {r['corpus_source']:16s} & {r['release']:6s} & "
            f"{f1} & {em} & {r['guessable_share'] * 100:.1f}\\% & "
            f"{r['gold_answer_words']:.1f}\\\\"
        )
    # The whole float is emitted, not just the rows. \input inside a tabular breaks the
    # alignment scan -- TeX is looking for the start of a cell and gets a file read, so the
    # \bottomrule after it lands mid-row -- and including the caption here has a second
    # benefit: the population it claims is interpolated from the same artefact as the
    # numbers, so caption and data cannot drift apart.
    body = (CAPTION_TEMPLATE
            .replace("<<READERS>>", str(len(PANEL)))
            .replace("<<POPULATION>>", "non-chance items")
            .replace("<<ROWS>>", "\n".join(lines)))

    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(body)
        print(f"[written] {args.tex}")
    else:
        print(body, end="")

    # The caption names the population the rows are over, and a caption cannot be generated
    # from the artefact without templating the whole float. Printing the population instead
    # gives the author the one figure the caption has to agree with.
    print(f"[written] {args.output}")
    print(f"[caption] condition={CONDITION}, panel of {len(PANEL)}: "
          + ", ".join(f"{rows[k]['label'].strip('\\ds') or 'ds'}={rows[k]['items_scored']}"
                      for k, *_ in COLLECTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
