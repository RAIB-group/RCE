#!/usr/bin/env python3
"""Build a LaTeX table from comprehensive FIQA CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASET_ORDER = ["adience", "agedb", "calfw", "cplfw", "lfw", "xqlfw"]
DEFAULT_FR_MODEL_ORDER = ["adaface", "arcface_o", "magface"]


def parse_list_arg(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x]


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def sort_alpha_columns(columns: Iterable[str]) -> list[str]:
    alphas: list[tuple[float, str]] = []
    for col in columns:
        if not col.startswith("alpha="):
            continue
        try:
            value = float(col.split("=", 1)[1])
        except ValueError:
            continue
        alphas.append((value, col))
    return [name for _, name in sorted(alphas, key=lambda x: x[0])]


def alpha_to_latex(alpha_name: str) -> str:
    # "alpha=0" -> "$\\alpha=0$"
    return rf"$\alpha={alpha_name.split('=', 1)[1]}$"


def compute_highlights(
    df: pd.DataFrame,
    fr_model_order: list[str],
    dataset_order: list[str],
    alpha_cols: list[str],
) -> dict[tuple[str, str, str, str], str]:
    """
    Highlight best and second-best per (fr_model, dataset, alpha) across FIQA rows.
    Returns:
        (fr_model, fiqa, dataset, alpha_col) -> "best" | "second"
    Ties are handled by value: all equal-best get bold; all equal-second get underline.
    """
    highlight: dict[tuple[str, str, str, str], str] = {}

    for fr_model in fr_model_order:
        subset_fr = df[df["fr_model"] == fr_model]
        if subset_fr.empty:
            continue

        for dataset in dataset_order:
            subset = subset_fr[subset_fr["dataset"] == dataset]
            if subset.empty:
                continue

            for alpha in alpha_cols:
                vals = []
                for _, row in subset.iterrows():
                    v = row[alpha]
                    if pd.notna(v):
                        vals.append((row["fiqa"], float(v)))

                if not vals:
                    continue

                unique_vals = sorted({v for _, v in vals}, reverse=True)
                best_val = unique_vals[0] if len(unique_vals) >= 1 else None
                second_val = unique_vals[1] if len(unique_vals) >= 2 else None

                for fiqa, v in vals:
                    key = (fr_model, fiqa, dataset, alpha)
                    if best_val is not None and v == best_val:
                        highlight[key] = "best"
                    elif second_val is not None and v == second_val:
                        highlight[key] = "second"

    return highlight


def format_value(
    value: float | None,
    mark: str | None,
    decimals: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    if mark == "best":
        return rf"\textbf{{{s}}}"
    if mark == "second":
        return rf"\underline{{{s}}}"
    return s


def build_latex_table(
    df: pd.DataFrame,
    dataset_order: list[str] | None,
    fr_model_order: list[str] | None,
    alpha_cols: list[str] | None,
    caption: str = "FIQA results across datasets and $\\alpha$ values.",
    label: str = "tab:fiqa_results",
    use_resizebox: bool = True,
) -> str:
    required = {"dataset", "fr_model", "fiqa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if alpha_cols is None:
        alpha_cols = sort_alpha_columns(df.columns)
    if not alpha_cols:
        raise ValueError("No alpha columns found. Expected columns like alpha=0, alpha=1, ...")

    missing_alpha = [a for a in alpha_cols if a not in df.columns]
    if missing_alpha:
        raise ValueError(f"Requested alpha columns not found in CSV: {missing_alpha}")

    if dataset_order is None:
        dataset_order = df["dataset"].drop_duplicates().tolist()
    else:
        existing = set(df["dataset"].dropna().astype(str).tolist())
        dataset_order = [d for d in dataset_order if d in existing]

    if fr_model_order is None:
        fr_model_order = df["fr_model"].drop_duplicates().tolist()

    fiqa_order = df["fiqa"].drop_duplicates().tolist()

    value_map: dict[tuple[str, str, str], list[float | None]] = {}
    for _, row in df.iterrows():
        value_map[(row["fr_model"], row["fiqa"], row["dataset"])] = [row[a] for a in alpha_cols]

    highlight = compute_highlights(df, fr_model_order, dataset_order, alpha_cols)

    printed_blocks: list[tuple[str, list[str]]] = []
    for fr_model in fr_model_order:
        mask_fr = df["fr_model"] == fr_model
        fr_fiqas = [f for f in fiqa_order if (mask_fr & (df["fiqa"] == f)).any()]
        if fr_fiqas:
            printed_blocks.append((fr_model, fr_fiqas))

    # tabular spec
    col_format = "ll|" + "|".join(["r" * len(alpha_cols)] * len(dataset_order))

    tabular_lines: list[str] = []
    tabular_lines.append(r"\begin{tabular}{" + col_format + "}")
    tabular_lines.append(r"\toprule")

    # Header row 1
    header1 = [r"\multirow{2}{*}{FR}", r"\multirow{2}{*}{FIQA}"]
    for i, dataset in enumerate(dataset_order):
        align = "c|" if i < len(dataset_order) - 1 else "c"
        header1.append(rf"\multicolumn{{{len(alpha_cols)}}}{{{align}}}{{{latex_escape(dataset)}}}")
    tabular_lines.append(" & ".join(header1) + r" \\")

    # Header row 2
    header2 = ["", ""]
    for _ in dataset_order:
        for a in alpha_cols:
            header2.append(alpha_to_latex(a))
    tabular_lines.append(" & ".join(header2) + r" \\")
    tabular_lines.append(r"\midrule")

    # Body
    for block_i, (fr_model, fr_fiqas) in enumerate(printed_blocks):
        n_fiqas = len(fr_fiqas)
        for j, fiqa in enumerate(fr_fiqas):
            row = []
            row.append(
                rf"\multirow{{{n_fiqas}}}{{*}}{{\rotatebox{{90}}{{{latex_escape(fr_model)}}}}}"
                if j == 0
                else ""
            )
            row.append(latex_escape(fiqa))

            for dataset in dataset_order:
                vals = value_map.get((fr_model, fiqa, dataset), [None] * len(alpha_cols))
                for alpha_name, v in zip(alpha_cols, vals):
                    mark = highlight.get((fr_model, fiqa, dataset, alpha_name))
                    row.append(format_value(v, mark))

            tabular_lines.append(" & ".join(row) + r" \\")

        if block_i < len(printed_blocks) - 1:
            tabular_lines.append(r"\midrule")

    tabular_lines.append(r"\bottomrule")
    tabular_lines.append(r"\end{tabular}")

    tabular = "\n".join(tabular_lines)

    outer: list[str] = []
    outer.append(r"\begin{table*}[t]")
    outer.append(r"\centering")
    outer.append(r"\scriptsize")
    outer.append(r"\setlength{\tabcolsep}{3pt}")
    outer.append(r"\renewcommand{\arraystretch}{1.08}")
    if use_resizebox:
        outer.append(r"\resizebox{\textwidth}{!}{%")
        outer.append(tabular)
        outer.append(r"}")
    else:
        outer.append(tabular)
    outer.append(rf"\caption{{{caption}}}")
    outer.append(rf"\label{{{label}}}")
    outer.append(r"\end{table*}")

    return "\n".join(outer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FIQA CSV results to a LaTeX table.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--out", help="Optional path to write .tex output. If omitted, prints to stdout.")
    parser.add_argument(
        "--dataset-order",
        help="Comma-separated dataset order. Default: adience,agedb,calfw,cplfw,lfw,xqlfw",
    )
    parser.add_argument(
        "--fr-model-order",
        help="Comma-separated FR model order. Default: adaface,arcface_o,magface",
    )
    parser.add_argument(
        "--alpha-cols",
        help="Comma-separated alpha columns (e.g., alpha=0,alpha=1,alpha=2,alpha=3). Default: detect+sort.",
    )
    parser.add_argument("--caption", default=r"FIQA results across datasets and $\alpha$ values.")
    parser.add_argument("--label", default="tab:fiqa_results")
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Disable resizebox wrapper.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    dataset_order = parse_list_arg(args.dataset_order) or DEFAULT_DATASET_ORDER
    fr_model_order = parse_list_arg(args.fr_model_order) or DEFAULT_FR_MODEL_ORDER
    alpha_cols = parse_list_arg(args.alpha_cols)

    latex = build_latex_table(
        df=df,
        dataset_order=dataset_order,
        fr_model_order=fr_model_order,
        alpha_cols=alpha_cols,
        caption=args.caption,
        label=args.label,
        use_resizebox=not args.no_resizebox,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex + "\n", encoding="utf-8")
    else:
        print(latex)


if __name__ == "__main__":
    main()
