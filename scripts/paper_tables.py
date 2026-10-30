#!/usr/bin/env python3
"""Emit the paper's result tables, whole floats, from the analysis artefacts.

Nothing in the manuscript is typed by hand. Each function here reads a results artefact and
returns a complete `table` float -- caption, label, tabular, rows and highlighting -- which
the paper includes with \\input. Two properties follow that a verification pass cannot give:
a figure in the paper cannot disagree with a run, and a caption cannot drift away from the
population it describes, because both come from the same artefact in the same call.

The whole float is emitted rather than the rows alone because \\input inside a tabular breaks
the alignment scan: TeX is looking for the start of a cell, receives a file read, and the
following \\bottomrule lands mid-row.

    python3 scripts/paper_tables.py --experiments <experiments> --tex-dir paper/tables --all
    python3 scripts/paper_tables.py --experiments <experiments> --table ablation

Adding a table: write a function returning the float, register it in TABLES, and replace the
hand-written float in the manuscript with \\input{tables/<name>}.
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

# The reported panel, in table order, split into the groups the tables show as row blocks.
# Fixed here rather than discovered: the run tree holds every model ever swept, and a table
# that averages whatever it finds will silently report a five-model mean as a six-model one.
PARTICIPATING = _PARTICIPATING
CONTROL = _CONTROL


def require(container: dict, keys, where: str):
    """Fetch a nested path, naming the artefact and the path when it is absent.

    A missing model or condition has to stop the run. Falling back to a sibling that happens
    to exist is how a table comes to report one population under another's caption.
    """
    node = container
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise SystemExit(f"{where}: no '{'/'.join(map(str, keys))}' (missing '{key}')")
        node = node[key]
    return node


def thousands(value: int) -> str:
    """LaTeX thousands separator. Applied per number: a blanket comma replacement over a
    caption also rewrites the commas in its prose."""
    return f"{value:,}".replace(",", "{,}")


def mean(values) -> float:
    values = list(values)
    if not values:
        raise SystemExit("mean of nothing")
    return sum(values) / len(values)


def group_rows(render, group_label: str, group, span: int, indent: str = "  ") -> list[str]:
    """One \\multirow block: a spanning group label, then a row per model."""
    lines = [f"\\multirow{{{len(group)}}}{{*}}{{{group_label}}}"]
    for key, label in group:
        lines.append(f"{indent}& {label:8s} & " + render(key) + "\\\\")
    return lines


def float_wrap(caption: str, label: str, colspec: str, header: list[str],
               blocks: list[list[str]], extra: str = "") -> str:
    body = "\n\\midrule\n".join("\n".join(block) for block in blocks)
    return (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n\\small\n"
        f"{extra}"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        + "\n".join(header) + "\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# ----------------------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------------------

def closed(art: dict) -> str:
    readers = require(art["summary"], ("e1", "readers"), "summary.json")

    def render(key: str) -> str:
        entry = require(readers, (key,), "summary.json e1")
        cells = [f"{require(entry, (section, 'answer_f1'), f'e1/{key}'):.3f}"
                 for section in ("all", "all_no_guessable",
                                 "hotpot_paired", "hotpot_paired_no_guessable")]
        gap = require(entry, ("leakage_gap_f1_no_guessable",), f"e1/{key}")
        radius = (gap["ci_high"] - gap["ci_low"]) / 2
        # Leading zero dropped in the margin: it is a width the column does not need.
        cells.append(f"${gap['mean_difference']:.3f}_{{\\pm {radius:.3f}}}$"
                     .replace("\\pm 0.", "\\pm ."))
        cells.append(f"{gap['positive']}/{gap['negative']}")
        return " & ".join(cells)

    def mean_row(group) -> str:
        parts = []
        for section in ("all", "all_no_guessable", "hotpot_paired", "hotpot_paired_no_guessable"):
            parts.append(f"\\emph{{{mean(readers[k][section]['answer_f1'] for k, _ in group):.3f}}}")
        parts.append(f"\\emph{{{mean(readers[k]['leakage_gap_f1_no_guessable']['mean_difference'] for k, _ in group):.3f}}}")
        parts.append("---")
        return "  & \\emph{Mean} & " + " & ".join(parts) + "\\\\"

    blocks = []
    for label, group in (("Participating", PARTICIPATING), ("Control", CONTROL)):
        block = group_rows(render, label, group, span=8)
        block.insert(len(block), "  \\cmidrule(lr){2-8}")
        block.append(mean_row(group))
        # \multirow spans the model rows plus the mean row beneath them.
        block[0] = f"\\multirow{{{len(group) + 1}}}{{*}}{{{label}}}"
        blocks.append(block)

    pairs = readers[PANEL[0][0]]["leakage_gap_f1_no_guessable"]["pairs"]
    first = readers[PANEL[0][0]]

    def removed(section: str) -> str:  # share of a section the Excl. columns leave out
        return f"{1 - first[section + '_no_guessable']['n'] / first[section]['n']:.0%}".replace("%", "\\%")

    lore, hotpot = removed("all"), removed("hotpot_paired")
    share = (f"{lore} of both collections" if lore == hotpot
             else f"{lore} of \\ds, {hotpot} of HotpotQA")
    caption = (
        "Experiment 1: Closed-book answer F1 with zero context supplied (test split). "
        f"\\emph{{Excl.}} removes guessable boolean and option-naming questions ({share}). "
        "$\\Delta$ is the mean paired per-item difference over "
        f"{thousands(pairs)} pairs with 95\\% bootstrap CI radius as subscript. Discordant "
        "pairs are reported as \\emph{HotpotQA wins / \\ds\\ wins}.")
    header = [
        "& & \\multicolumn{2}{c}{\\ds} & \\multicolumn{2}{c}{HotpotQA} & & \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "Group & Model & All & Excl. & All & Excl. & $\\Delta$ ($\\pm$95\\% CI) & Discordant\\\\",
    ]
    return float_wrap(caption, "tab:closed", "llcccccc", header, blocks,
                      extra="\\setlength{\\tabcolsep}{3.5pt}\n")


# ----------------------------------------------------------------------------------
# Experiment 2
# ----------------------------------------------------------------------------------

def ablation(art: dict) -> str:
    e2 = require(art["summary"], ("e2", "readers"), "summary.json")
    paired = art["ablation_paired"]

    # lore10 and hotpot4 are the two sides of one pairing, over the same items. Pairing
    # either against a different sweep would compare two populations under one caption.
    SIDES = (("lore10", "\\ds"), ("hotpot4", "HotpotQA (paired)"))

    def render(key: str) -> str:
        cells = []
        for side, _ in SIDES:
            full = require(e2, (key, side, "full", "answer_f1"), "summary.json e2")
            loo = require(e2, (key, side, "leave_one_out", "answer_f1"), "summary.json e2")
            cell = require(paired, (side, key), "ablation_paired.json")
            dagger = "^{\\dag}" if cell["ci_low"] <= 0 <= cell["ci_high"] else ""
            cells += [f"{full:.3f}", f"{loo:.3f}", f"{cell['ablated_f1']:.3f}",
                      f"${cell['difference']:+.3f}{dagger}$"]
        return " & ".join(cells)

    def mean_row(group) -> str:
        parts = []
        for side, _ in SIDES:
            parts.append(f"\\emph{{{mean(e2[k][side]['full']['answer_f1'] for k, _ in group):.3f}}}")
            parts.append(f"\\emph{{{mean(e2[k][side]['leave_one_out']['answer_f1'] for k, _ in group):.3f}}}")
            parts.append(f"\\emph{{{mean(paired[side][k]['ablated_f1'] for k, _ in group):.3f}}}")
            parts.append(f"\\emph{{${mean(paired[side][k]['difference'] for k, _ in group):+.3f}$}}")
        return "  & \\emph{Mean} & " + " & ".join(parts) + "\\\\"

    blocks = []
    for label, group in (("Participating", PARTICIPATING), ("Control", CONTROL)):
        block = group_rows(render, label, group, span=10)
        block.append("  \\cmidrule(lr){2-10}")
        block.append(mean_row(group))
        block[0] = f"\\multirow{{{len(group) + 1}}}{{*}}{{{label}}}"
        blocks.append(block)

    lore_n = paired["lore10"][PANEL[0][0]]["items"]
    hot_n = paired["hotpot4"][PANEL[0][0]]["items"]
    variants = require(art["summary"], ("e2", "readers", PANEL[0][0], "lore10",
                                        "leave_one_out", "n"), "summary.json")
    caption = (
        "Experiment 2: Answer F1 under leave-one-out ablation, matched four-document layout. "
        "\\emph{Full} and \\emph{LOO} cover this experiment's population "
        "(Table~\\ref{tab:populations}), with LOO averaging one variant per relevant document "
        f"over {thousands(variants)} variants. \\emph{{Ablated}} restricts to non-chance "
        "variants where the answer string is absent from the reduced context, "
        f"{thousands(lore_n)} on \\ds and {thousands(hot_n)} on HotpotQA. "
        "\\emph{$-$CB} is ablated F1 minus closed-book F1 on those same items "
        "($\\dag$ indicates a 95\\% bootstrap CI containing zero).")
    header = [
        "& & \\multicolumn{4}{c}{\\ds} & \\multicolumn{4}{c}{HotpotQA (paired)}\\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
        "Group & Model & Full & LOO & Ablated & $-$CB & Full & LOO & Ablated & $-$CB\\\\",
    ]
    return float_wrap(caption, "tab:ablation", "llcccc|cccc", header, blocks)


# ----------------------------------------------------------------------------------
# Experiment 3
# ----------------------------------------------------------------------------------

def reading(art: dict) -> str:
    e3 = require(art["summary"], ("e3", "readers"), "summary.json")
    e2 = require(art["summary"], ("e2", "readers"), "summary.json")
    FIELDS = ("answer_f1", "sp_f1", "joint_f1")

    # Both halves of the matched block come from the same 1,982 pairs. The \ds side used to be
    # e3's `transposed` section, which is every transposed test item, 5,745 of them -- so the
    # block set 5,745 items beside the 1,982 HotpotQA items that are the sources of a subset of
    # them, and called the result matched. e2's lore10 is the \ds side of this pairing: the same
    # 1,982 items, at the same four-document layout, as hotpot4.
    def render(key: str) -> str:
        cells = [f"{require(e3, (key, 'all', f), 'e3'):.3f}" for f in FIELDS]
        cells += [f"{require(e2, (key, 'lore10', 'full', f), 'e2'):.3f}" for f in FIELDS]
        cells += [f"{require(e2, (key, 'hotpot4', 'full', f), 'e2'):.3f}" for f in FIELDS]
        return " & ".join(cells)

    blocks = [group_rows(render, label, group, span=11)
              for label, group in (("Participating", PARTICIPATING), ("Control", CONTROL))]
    caption = (
        "Experiment 3: Answering with the complete context. Metrics are answer F1 (Ans), "
        "supporting-fact F1 (Sup) and joint F1. Transposed and paired HotpotQA items use the "
        "standardised four-document layout.")
    header = [
        "& & \\multicolumn{3}{c|}{Full Benchmark} & \\multicolumn{6}{c}{Matched Paired Items (4-Doc Layout)}\\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-11}",
        "& & \\multicolumn{3}{c|}{\\ds} & \\multicolumn{3}{c}{\\ds} & \\multicolumn{3}{c}{HotpotQA}\\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "Group & Model & Ans & Sup & Joint & Ans & Sup & Joint & Ans & Sup & Joint\\\\",
    ]
    return float_wrap(caption, "tab:reading", "llccc|cccccc", header, blocks,
                      extra="\\setlength{\\tabcolsep}{2.5pt}\n")


# ----------------------------------------------------------------------------------
# Experiment 4
# ----------------------------------------------------------------------------------

def bm25(art: dict) -> str:
    e5 = require(art["summary"], ("e5", "readers"), "summary.json")
    e1 = require(art["summary"], ("e1", "readers"), "summary.json")
    SIDES = (("lore", "all"), ("hotpot", "hotpot_paired"))

    def values(key: str):
        out = []
        for side, closed_section in SIDES:
            cell = require(e5, (key, side, "bm25"), "summary.json e5")
            top = cell["overall"]["answer_f1"]
            complete = cell["full_support"]["answer_f1"]
            floor = require(e1, (key, closed_section, "answer_f1"), "summary.json e1")
            out.append((top, complete, floor / top))
        return out

    best = {}
    for column in range(2):
        for side in range(2):
            best[(side, column)] = max(values(k)[side][column] for k, _ in PANEL)

    def render(key: str) -> str:
        cells = []
        for side, (top, complete, share) in enumerate(values(key)):
            for column, value in enumerate((top, complete)):
                text = f"{value:.3f}"
                cells.append(f"\\textbf{{{text}}}" if value == best[(side, column)] else text)
            cells.append(f"{share * 100:.1f}\\%")
        return " & ".join(cells)

    blocks = [group_rows(render, label, group, span=8)
              for label, group in (("Participating", PARTICIPATING), ("Control", CONTROL))]
    caption = (
        "Experiment 4: End-to-end answer F1 with BM25 top-10 retrieval over the full corpora. "
        "\\emph{Complete} restricts to questions where all gold documents were retrieved. "
        "\\emph{Share} is the closed-book score of Table~\\ref{tab:closed} as a percentage of "
        "the top-10 score, the accuracy floor no ranking can move. Bold marks the best value "
        "per column.")
    header = [
        "& & \\multicolumn{3}{c}{\\ds} & \\multicolumn{3}{c}{HotpotQA (paired)}\\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Group & Model & Top-10 & Complete & Share & Top-10 & Complete & Share\\\\",
    ]
    return float_wrap(caption, "tab:bm25", "llcccccc", header, blocks)


# ----------------------------------------------------------------------------------
# Experiment 5
# ----------------------------------------------------------------------------------

def embedders(art: dict) -> str:
    quality, e5m = art["retrieval"], art["e5_metrics"]

    # (row label, dataset suffix). The unsuffixed dense/hybrid systems are Jina v5, named
    # before the per-embedder sweeps existed.
    ROWS = (
        ("Lexical", (("BM25", "bm25"),)),
        ("Dense", (("Jina v5", "dense"), ("MiniLM-L6", "dense-minilm"), ("SimCSE", "dense-simcse"))),
        ("Hybrid", (("Jina v5", "hybrid"), ("MiniLM-L6", "hybrid-minilm"), ("SimCSE", "hybrid-simcse"))),
    )

    def cells(suffix: str):
        out = []
        for collection in ("lore", "hotpot"):
            system = f"{collection}-{suffix}"
            recall = require(quality, (system, "recall@10"), "retrieval reports")
            answer = mean(require(e5m, (system, reader, "answer_f1"), "e5 metrics")
                          for reader, _ in PANEL)
            support = mean(require(e5m, (system, reader, "sp_f1"), "e5 metrics")
                           for reader, _ in PANEL)
            out += [recall, answer, support]
        return out

    table = {suffix: cells(suffix) for _, group in ROWS for _, suffix in group}
    best = [max(v[i] for v in table.values()) for i in range(6)]

    blocks = []
    for group_label, group in ROWS:
        rendered = []
        for label, suffix in group:
            marked = " & ".join(
                f"\\textbf{{{v:.3f}}}" if v == best[i] else f"{v:.3f}"
                for i, v in enumerate(table[suffix]))
            rendered.append(f"  & {label:9s} & {marked}\\\\")
        if len(group) == 1:
            blocks.append([f"{group_label} & {rendered[0].split('& ', 1)[1]}"])
        else:
            blocks.append([f"\\multirow{{{len(group)}}}{{*}}{{{group_label}}}"] + rendered)

    # The per-model consistency of the ordering is a finding, so it belongs in the prose
    # where it can be framed and verified, not in a caption that would carry it past the
    # number-checking pass and go stale when the table is regenerated.
    caption = (
        "Experiment 5: Retrieval and end-to-end generation quality across seven first-stage "
        f"configurations (means over {len(PANEL)} LLMs, top-10 context). R@10 is Recall@10 of the "
        "ranking; Ans F1 and SP F1 are end-to-end answer and supporting-fact F1. Bold marks "
        "the best value per column.")
    header = [
        "& & \\multicolumn{3}{c}{\\ds} & \\multicolumn{3}{c}{HotpotQA (paired)}\\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Type & Model & R@10 & Ans F1 & SP F1 & R@10 & Ans F1 & SP F1\\\\",
    ]
    return float_wrap(caption, "tab:embedders", "llccc|ccc", header, blocks)


def adapted(art: dict) -> str:
    report = art["adaptation"]
    rel = report["relative_to_bm25"]
    LABEL = (("jina", "Jina v5"), ("minilm", "MiniLM-L6"), ("simcse", "SimCSE"))

    def render(strategy: str, handle: str) -> str:
        cells = []
        for collection in ("lore", "hotpot"):
            row = require(rel, (f"{collection}-{strategy}-{handle}-adapted2",),
                          "adaptation_ranking.json")
            delta = row["delta_points"]
            cells += [f"{row['bm25_share_unadapted'] * 100:.1f}",
                      f"{row['bm25_share_adapted'] * 100:.1f}",
                      "0.0" if abs(delta) < 0.05 else f"${delta:+.1f}$"]
        return " & ".join(cells)

    blocks = []
    for strategy, group_label in (("dense", "Dense"), ("hybrid", "Hybrid")):
        block = [f"\\multirow{{{len(LABEL)}}}{{*}}{{{group_label}}}"]
        for handle, label in LABEL:
            block.append(f"  & {label:9s} & " + render(strategy, handle) + "\\\\")
        blocks.append(block)

    lore_bm25 = report["bm25"]["lore"]["recall@10"]
    hot_bm25 = report["bm25"]["hotpot"]["recall@10"]
    caption = (
        "Recall@10 as a percentage of BM25 on the same collection, before and after adaptation "
        "on the Lorepedia (inverse cloze task, input embeddings only, one epoch). BM25 is 100 "
        f"by construction, at {lore_bm25:.3f} Recall@10 on \\ds and {hot_bm25:.3f} on HotpotQA; "
        "values above 100 exceed it. $\\Delta$ is in percentage points. HotpotQA is the "
        "control: the Lorepedia carries none of its text.")
    header = [
        "& & \\multicolumn{3}{c}{\\ds} & \\multicolumn{3}{c}{HotpotQA (control)}\\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Type & Model & Base & Adapted & $\\Delta$ & Base & Adapted & $\\Delta$\\\\",
    ]
    return float_wrap(caption, "tab:adapted", "llccc|ccc", header, blocks)




# ----------------------------------------------------------------------------------
# Dataset description
# ----------------------------------------------------------------------------------

def composition(art: dict) -> str:
    stats = require(art["statistics"], ("sections",), "statistics.json")
    lore = art["lorepedia"]
    COLUMNS = (("transposed", "Transposed"), ("novel", "Induced"), ("all", "Overall"))

    def row(label: str, pick, fmt="{:,}") -> str:
        cells = []
        for section, _ in COLUMNS:
            value = pick(stats[section])
            cells.append("---" if value is None else fmt.format(value).replace(",", "{,}"))
        return f"{label} & " + " & ".join(cells) + "\\\\"

    def kind(*names):
        # bridge and chain occupy one row: transposition produces bridges, induction chains.
        return lambda s: sum(s["type"].get(n, 0) for n in names) or None

    lines = [
        row("Total Items", lambda s: s["items"]),
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Question Families}}\\\\",
        row("\\quad \\texttt{bridge} / \\texttt{chain}", kind("bridge", "chain")),
        row("\\quad \\texttt{comparison}", kind("comparison")),
        row("\\quad \\texttt{convergent}", kind("convergent")),
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Evidence Layout (means)}}\\\\",
        row("\\quad Candidate Documents", lambda s: s["documents_per_item"]["mean"], "{:.1f}"),
        row("\\quad Relevant Documents",
            lambda s: s["supporting_documents_per_item"]["mean"], "{:.1f}"),
        row("\\quad Marked Supporting Facts",
            lambda s: s["supporting_facts_per_item"]["mean"], "{:.1f}"),
        "\\midrule",
        "\\multicolumn{4}{l}{\\emph{Text Properties}}\\\\",
        row("\\quad Context Words per Item",
            lambda s: round(s["context_words_per_item"]["mean"]), "{:,}"),
        row("\\quad Yes/No Questions",
            lambda s: s["yes_no_answers"]["share"] * 100, "{:.1f}\\%"),
        "\\midrule",
        row("Document Corpus Size", lambda s: s["corpus"]["documents"]),
        row("Total Document Words", lambda s: s["corpus"]["words"] / 1e6, "{:.2f}M"),
        "\\midrule",
        ("\\multicolumn{3}{l}{Lorepedia Articles (Background Corpus)} & "
         + thousands(lore["released"]) + "\\\\"),
        ("\\multicolumn{3}{l}{Lorepedia Total Words} & "
         + f"{lore['words'] / 1e6:.2f}M" + "\\\\"),
    ]

    caption = (
        "Composition of \\ds. Layout and text metrics are per-item arithmetic means. The "
        f"Lorepedia is a standalone background corpus for index expansion, spanning the same "
        f"{lore['worlds']} worlds.")
    header = ["Metric & Transposed & Induced & Overall\\\\"]
    return float_wrap(caption, "tab:composition", "lrrr", header, [lines]).replace(
        "\\small\n", "")


def populations(art: dict) -> str:
    e1, e2, e3 = (require(art["summary"], (k, "readers"), "summary.json")
                  for k in ("e1", "e2", "e3"))
    paired, retrieval = art["ablation_paired"], art["retrieval_bm25"]
    first = PANEL[0][0]

    def both(lore, hotpot) -> str:
        return f"{thousands(lore)} & {thousands(hotpot)}"

    rows = [
        ("1 Closed book", "items", both(e1[first]["all"]["n"], e1[first]["hotpot_paired"]["n"])),
        (None, "\\quad excluding chance",
         both(e1[first]["all_no_guessable"]["n"], e1[first]["hotpot_paired_no_guessable"]["n"])),
        (None, "\\quad paired, both sides",
         both(e1[first]["leakage_gap_f1_no_guessable"]["pairs"],
              e1[first]["leakage_gap_f1_no_guessable"]["pairs"])),
        ("2 Ablated evidence", "items",
         both(e2[first]["lore10"]["full"]["n"], e2[first]["hotpot4"]["full"]["n"])),
        (None, "\\quad variants",
         both(e2[first]["lore10"]["leave_one_out"]["n"],
              e2[first]["hotpot4"]["leave_one_out"]["n"])),
        (None, "\\quad answer removed, not chance",
         both(paired["lore10"][first]["items"], paired["hotpot4"][first]["items"])),
        ("3 Complete context", "items",
         f"{thousands(e3[first]['all']['n'])} & ---"),
        # The matched block of tab:reading is 1,982 against 1,982, not the full transposed
        # subset against them. Reporting the subset here would describe a pairing the table
        # does not make.
        (None, "\\quad matched pairs, both sides",
         both(e2[first]["lore10"]["full"]["n"], e2[first]["hotpot4"]["full"]["n"])),
        ("4--5 Retrieval", "questions",
         both(retrieval["lore-bm25"]["items"], retrieval["hotpot-bm25"]["items"])),
        (None, "\\quad documents indexed",
         both(retrieval["lore-bm25"]["documents"], retrieval["hotpot-bm25"]["documents"])),
    ]

    blocks, current = [], []
    for experiment, measured, cells in rows:
        if experiment and current:
            blocks.append(current)
            current = []
        stub = experiment if experiment else "                  "
        current.append(f"{stub} & {measured} & {cells}\\\\")
    blocks.append(current)

    caption = (
        "Evaluation populations. All experiments draw on the test split. Chance-answerable "
        "items are yes/no questions and comparisons naming their own answer.")
    header = ["Experiment & Measured on & \\ds & HotpotQA\\\\"]
    return float_wrap(caption, "tab:populations", "llrr", header, blocks)


TABLES = {"composition": composition, "populations": populations, "closed": closed, "ablation": ablation, "reading": reading,
          "bm25": bm25, "embedders": embedders, "adapted": adapted}


def load(experiments: Path, dist: Path, split: str) -> dict:
    results = experiments / "results" / split
    e5 = experiments / "e5_bm25_retrieval" / split

    quality: dict[str, dict] = {}
    for report in sorted(e5.glob("retrieval*.json")):
        body = json.loads(report.read_text())
        for section in ("systems", "collections"):
            for name, summary in body.get(section, {}).items():
                quality[name] = summary.get("recall", summary)

    e5_metrics: dict[str, dict] = {}
    for path in sorted((e5 / "metrics").glob("*.json")):
        body = json.loads(path.read_text())
        e5_metrics.setdefault(body["dataset"], {})[body["reader"]] = body["overall"]

    bm25_report = json.loads((e5 / "retrieval.json").read_text())["collections"]

    return {
        "summary": json.loads((results / "summary.json").read_text()),
        "statistics": json.loads((dist / "lorehopqa" / "statistics.json").read_text()),
        "lorepedia": json.loads(
            (experiments / "lorepedia" / "provenance.json").read_text()),
        "retrieval_bm25": bm25_report,
        "ablation_paired": json.loads((experiments / "ablation_paired.json").read_text()),
        "adaptation": json.loads((results / "adaptation_ranking.json").read_text()),
        "retrieval": quality,
        "e5_metrics": e5_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True,
                        help="Release directory holding lorehopqa/statistics.json.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--table", choices=sorted(TABLES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tex-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.all and not args.table:
        raise SystemExit("pass --table <name> or --all")

    art = load(args.experiments, args.dist, args.split)
    wanted = sorted(TABLES) if args.all else [args.table]
    for name in wanted:
        text = f"% generated by scripts/paper_tables.py --table {name}; do not edit by hand\n"
        text += TABLES[name](art)
        if args.tex_dir:
            args.tex_dir.mkdir(parents=True, exist_ok=True)
            (args.tex_dir / f"tab_{name}.tex").write_text(text)
            print(f"[written] {args.tex_dir / f'tab_{name}.tex'}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
