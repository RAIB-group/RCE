#!/usr/bin/env python3
"""Build a LaTeX table from multiple FIQA result CSVs.

Row structure:
    FR | FIQA

Column structure:
    dataset -> {edc, fixthredc, balpairedc, fixthrbalpairedc, ranking}
    where:
        edc                 -> pAUC@0.3(fnmr)
        fixthredc           -> pAUC@0.3(cer)
        balpairedc          -> pAUC@0.3(fnmr)
        fixthrbalpairedc    -> pAUC@0.3(cer)
        ranking             -> alpha=0, alpha=10

Header design:
    dataset
      -> method
         -> merged metric row:
              first 4 cols  = pAUC@0.3
              last 2 cols   = Ranking
         -> subcols:
              FNMR | CER | FNMR | CER | alpha=0 | alpha=10

Run:
    python latex/comp_results_to_latex_table.py --out latex/table.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ORDER = ["adience", "agedb", "calfw", "cplfw", "lfw", "xqlfw"]
DEFAULT_FR_MODEL_ORDER = ["adaface", "arcface", "magface"]

DEFAULT_EDC_CSV = "/home/bw/FIQA/evd_ijcb/output/evd/default_methods/comprehensive_results.csv"
DEFAULT_FIXTHR_EDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol1_evd_fix_thr/default_methods/comprehensive_results.csv"
DEFAULT_BALPAIREDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol2_same_gen_imp_min/default_methods/comprehensive_results.csv"
DEFAULT_FIXTHR_BALPAIREDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol3_same_gen_imp_min_fix_thr/default_methods/comprehensive_results.csv"
DEFAULT_RANKING_CSV = "/home/bw/FIQA/evd_ijcb/output/sol4_rank_sample/comprehensive_results.csv"

METHOD_STRUCTURE = [
    ("edc", ["pAUC@0.3(fnmr)"]),
    ("fixthredc", ["pAUC@0.3(cer)"]),
    ("balpairedc", ["pAUC@0.3(fnmr)"]),
    ("fixthrbalpairedc", ["pAUC@0.3(cer)"]),
    ("ranking", ["alpha=0", "alpha=10"]),
]

METHOD_DISPLAY_MAP = {
    "balpairedc": "bpedc",
    "fixthrbalpairedc": "fthrbpedc",
}

# Highlight direction by method:
# pAUC methods -> lower is better, ranking -> higher is better.
METHOD_DIRECTION = {
    "edc": "min",
    "fixthredc": "min",
    "balpairedc": "min",
    "fixthrbalpairedc": "min",
    "ranking": "max",
}


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


def subcol_to_latex(name: str) -> str:
    if name.startswith("alpha="):
        return rf"$\alpha={name.split('=', 1)[1]}$"
    if name == "pAUC@0.3(fnmr)":
        return r"FNMR"
    if name == "pAUC@0.3(cer)":
        return r"CER"
    return latex_escape(name)


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_actual = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        found = norm_to_actual.get(c.strip().lower())
        if found is not None:
            return found
    return None


def prepare_single_value_method_df(
    df: pd.DataFrame,
    method_name: str,
    subcol_name: str,
) -> pd.DataFrame:
    required = {"dataset", "fr_model", "fiqa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{method_name}: missing required columns: {sorted(missing)}")

    value_col = first_existing_column(
        df,
        [
            "pAUC@0.3(fnmr)",
            "pAUC@0.3(cer)",
            "pauc@0.3(fnmr)",
            "pauc@0.3(cer)",
            "pAUC@0.3",
            "pauc@0.3",
            "pauc",
            "score",
            "value",
            "metric",
            "result",
        ],
    )
    if value_col is None:
        raise ValueError(
            f"{method_name}: could not find value column. "
            "Expected one of: pAUC@0.3, pauc@0.3, pauc, score, value, metric, result"
        )

    rank_col = first_existing_column(df, ["rank"])

    out = df[["dataset", "fr_model", "fiqa", value_col]].copy()
    out = out.rename(columns={value_col: "value"})
    out["rank"] = df[rank_col] if rank_col is not None else pd.NA
    out["method"] = method_name
    out["subcol"] = subcol_name
    return out[["dataset", "fr_model", "fiqa", "method", "subcol", "value", "rank"]]


def prepare_ranking_df(
    df: pd.DataFrame,
    alpha_cols: list[str] | None = None,
) -> pd.DataFrame:
    required = {"dataset", "fr_model", "fiqa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ranking: missing required columns: {sorted(missing)}")

    if alpha_cols is None:
        alpha_cols = ["alpha=0", "alpha=10"]

    missing_alpha = [c for c in alpha_cols if c not in df.columns]
    if missing_alpha:
        raise ValueError(f"ranking: missing required alpha columns: {missing_alpha}")

    parts = []
    for a in alpha_cols:
        rank_col = first_existing_column(
            df,
            [f"rank_{a}", f"rank{a}", f"{a}_rank", f"rank {a}"],
        )
        tmp = df[["dataset", "fr_model", "fiqa", a]].copy()
        tmp = tmp.rename(columns={a: "value"})
        tmp["rank"] = df[rank_col] if rank_col is not None else pd.NA
        tmp["method"] = "ranking"
        tmp["subcol"] = a
        parts.append(tmp[["dataset", "fr_model", "fiqa", "method", "subcol", "value", "rank"]])

    return pd.concat(parts, ignore_index=True)


def combine_sources(
    edc_csv: str | Path,
    fixthredc_csv: str | Path,
    balpairedc_csv: str | Path,
    fixthrbalpairedc_csv: str | Path,
    ranking_csv: str | Path,
) -> pd.DataFrame:
    df_edc = prepare_single_value_method_df(load_csv(edc_csv), "edc", "pAUC@0.3(fnmr)")
    df_fix = prepare_single_value_method_df(load_csv(fixthredc_csv), "fixthredc", "pAUC@0.3(cer)")
    df_bal = prepare_single_value_method_df(load_csv(balpairedc_csv), "balpairedc", "pAUC@0.3(fnmr)")
    df_fixbal = prepare_single_value_method_df(
        load_csv(fixthrbalpairedc_csv),
        "fixthrbalpairedc",
        "pAUC@0.3(cer)",
    )
    df_rank = prepare_ranking_df(load_csv(ranking_csv), ["alpha=0", "alpha=10"])

    df = pd.concat([df_edc, df_fix, df_bal, df_fixbal, df_rank], ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fr_model"] = df["fr_model"].astype(str).str.strip()
    df["fiqa"] = df["fiqa"].astype(str).str.strip()
    return df


def compute_highlights(
    df: pd.DataFrame,
    fr_model_order: list[str],
    dataset_order: list[str],
) -> dict[tuple[str, str, str, str, str], str]:
    """
    Highlight best and second-best per (fr_model, dataset, method, subcol) across FIQA rows.
    Returns:
        (fr_model, fiqa, dataset, method, subcol) -> "best" | "second"
    """
    highlight: dict[tuple[str, str, str, str, str], str] = {}

    method_subcols = [(m, s) for m, subcols in METHOD_STRUCTURE for s in subcols]

    df2 = df.copy()
    df2["dataset_key"] = df2["dataset"].astype(str).str.strip().str.lower()
    df2["fr_model_key"] = df2["fr_model"].astype(str).str.strip().str.lower()

    for fr_model in fr_model_order:
        fr_model_key = fr_model.strip().lower()
        subset_fr = df2[df2["fr_model_key"] == fr_model_key]
        if subset_fr.empty:
            continue

        for dataset in dataset_order:
            dataset_key = dataset.strip().lower()
            subset_ds = subset_fr[subset_fr["dataset_key"] == dataset_key]
            if subset_ds.empty:
                continue

            for method, subcol in method_subcols:
                subset = subset_ds[
                    (subset_ds["method"] == method) &
                    (subset_ds["subcol"] == subcol)
                ]
                if subset.empty:
                    continue

                vals = []
                for _, row in subset.iterrows():
                    v = row["value"]
                    if pd.notna(v):
                        vals.append((row["fiqa"], float(v)))

                if not vals:
                    continue

                direction = METHOD_DIRECTION.get(method, "max")
                reverse = direction == "max"
                unique_vals = sorted({v for _, v in vals}, reverse=reverse)
                best_val = unique_vals[0] if len(unique_vals) >= 1 else None
                second_val = unique_vals[1] if len(unique_vals) >= 2 else None

                for fiqa, v in vals:
                    key = (fr_model_key, fiqa, dataset_key, method, subcol)
                    if best_val is not None and v == best_val:
                        highlight[key] = "best"
                    elif second_val is not None and v == second_val:
                        highlight[key] = "second"

    return highlight


def _format_rank(rank: object) -> str | None:
    if rank is None or pd.isna(rank):
        return None
    try:
        f_rank = float(rank)
        return str(int(f_rank)) if f_rank.is_integer() else f"{f_rank:g}"
    except (TypeError, ValueError):
        s = str(rank).strip()
        if not s or s.lower() == "nan":
            return None
        return s


def format_value(
    value: float | None,
    mark: str | None,
    rank: object = None,
    decimals: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    if mark == "best":
        s = rf"\textbf{{{s}}}"
    elif mark == "second":
        s = rf"\underline{{{s}}}"

    rank_text = _format_rank(rank)
    if rank_text is None:
        return s
    return f"({latex_escape(rank_text)}) {s}"


def build_latex_table(
    df: pd.DataFrame,
    dataset_order: list[str] | None,
    fr_model_order: list[str] | None,
    caption: str = "FIQA results across datasets.",
    label: str = "tab:fiqa_results",
    use_resizebox: bool = True,
) -> str:
    required = {"dataset", "fr_model", "fiqa", "method", "subcol", "value", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Combined dataframe is missing required columns: {sorted(missing)}")

    df2 = df.copy()
    df2["dataset"] = df2["dataset"].astype(str).str.strip()
    df2["fr_model"] = df2["fr_model"].astype(str).str.strip()
    df2["fiqa"] = df2["fiqa"].astype(str).str.strip()
    df2["dataset_key"] = df2["dataset"].str.lower()
    df2["fr_model_key"] = df2["fr_model"].str.lower()

    if dataset_order is None:
        dataset_order = df2["dataset"].drop_duplicates().tolist()
    else:
        existing = set(df2["dataset_key"].dropna().astype(str).tolist())
        dataset_order = [d for d in dataset_order if d.strip().lower() in existing]

    if fr_model_order is None:
        fr_model_order = df2["fr_model"].drop_duplicates().tolist()

    fiqa_order = df2["fiqa"].drop_duplicates().tolist()

    dataset_display_map: dict[str, str] = (
        df2.drop_duplicates("dataset_key")
        .set_index("dataset_key")["dataset"]
        .to_dict()
    )
    fr_display_map: dict[str, str] = (
        df2.drop_duplicates("fr_model_key")
        .set_index("fr_model_key")["fr_model"]
        .to_dict()
    )

    dataset_keys = [d.strip().lower() for d in dataset_order]
    fr_model_keys = [m.strip().lower() for m in fr_model_order]

    value_map: dict[tuple[str, str, str, str, str], float | None] = {}
    rank_map: dict[tuple[str, str, str, str, str], object] = {}
    for _, row in df2.iterrows():
        key = (row["fr_model_key"], row["fiqa"], row["dataset_key"], row["method"], row["subcol"])
        value_map[key] = row["value"]
        rank_map[key] = row["rank"]

    highlight = compute_highlights(df2, fr_model_keys, dataset_keys)

    printed_blocks: list[tuple[str, list[str]]] = []
    for fr_model in fr_model_keys:
        mask_fr = df2["fr_model_key"] == fr_model
        fr_fiqas = [f for f in fiqa_order if (mask_fr & (df2["fiqa"] == f)).any()]
        if fr_fiqas:
            printed_blocks.append((fr_model, fr_fiqas))

    cols_per_dataset = sum(len(subcols) for _, subcols in METHOD_STRUCTURE)  # 6
    total_data_cols = cols_per_dataset * len(dataset_keys)
    col_format = "ll|" + "|".join(["r"] * total_data_cols)

    tabular_lines: list[str] = []
    tabular_lines.append(r"\begin{tabular}{" + col_format + "}")
    tabular_lines.append(r"\toprule")

    # Header row 1: datasets
    header1 = [r"\multirow{4}{*}{FR}", r"\multirow{4}{*}{FIQA}"]
    for i, dataset in enumerate(dataset_keys):
        align = "c|" if i < len(dataset_keys) - 1 else "c"
        header1.append(
            rf"\multicolumn{{{cols_per_dataset}}}{{{align}}}{{{latex_escape(dataset_display_map.get(dataset, dataset))}}}"
        )
    tabular_lines.append(" & ".join(header1) + r" \\")

    # Header row 2: methods
    header2 = ["", ""]
    for ds_i, _dataset in enumerate(dataset_keys):
        for method_i, (method, subcols) in enumerate(METHOD_STRUCTURE):
            is_last_block = (ds_i == len(dataset_keys) - 1) and (method_i == len(METHOD_STRUCTURE) - 1)
            align = "c|" if not is_last_block else "c"
            method_label = METHOD_DISPLAY_MAP.get(method, method)
            header2.append(rf"\multicolumn{{{len(subcols)}}}{{{align}}}{{{latex_escape(method_label)}}}")
    tabular_lines.append(" & ".join(header2) + r" \\")

    # Header row 3: merged metric row
    header3 = ["", ""]
    for ds_i, _dataset in enumerate(dataset_keys):
        last_dataset = ds_i == len(dataset_keys) - 1
        if last_dataset:
            header3.extend([
                r"\multicolumn{4}{c}{pAUC@0.3}",
                r"\multicolumn{2}{c}{Ranking}",
            ])
        else:
            header3.extend([
                r"\multicolumn{4}{c|}{pAUC@0.3}",
                r"\multicolumn{2}{c|}{Ranking}",
            ])
    tabular_lines.append(" & ".join(header3) + r" \\")

    # Header row 4: subcolumns
    header4 = ["", ""]
    for _dataset in dataset_keys:
        header4.extend([
            subcol_to_latex("pAUC@0.3(fnmr)"),
            subcol_to_latex("pAUC@0.3(cer)"),
            subcol_to_latex("pAUC@0.3(fnmr)"),
            subcol_to_latex("pAUC@0.3(cer)"),
            subcol_to_latex("alpha=0"),
            subcol_to_latex("alpha=10"),
        ])
    tabular_lines.append(" & ".join(header4) + r" \\")
    tabular_lines.append(r"\midrule")

    # Body
    for block_i, (fr_model, fr_fiqas) in enumerate(printed_blocks):
        n_fiqas = len(fr_fiqas)
        for j, fiqa in enumerate(fr_fiqas):
            row = []
            row.append(
                rf"\multirow{{{n_fiqas}}}{{*}}{{\rotatebox{{90}}{{{latex_escape(fr_display_map.get(fr_model, fr_model))}}}}}"
                if j == 0 else ""
            )
            row.append(latex_escape(fiqa))

            for dataset in dataset_keys:
                for method, subcols in METHOD_STRUCTURE:
                    for subcol in subcols:
                        v = value_map.get((fr_model, fiqa, dataset, method, subcol))
                        r = rank_map.get((fr_model, fiqa, dataset, method, subcol))
                        mark = highlight.get((fr_model, fiqa, dataset, method, subcol))
                        row.append(format_value(v, mark, r))

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
    parser = argparse.ArgumentParser(description="Convert multiple FIQA result CSVs to a nested LaTeX table.")
    parser.add_argument("--edc-csv", default=DEFAULT_EDC_CSV, help="Path to EDC CSV.")
    parser.add_argument("--fixthredc-csv", default=DEFAULT_FIXTHR_EDC_CSV, help="Path to fixthredc CSV.")
    parser.add_argument("--balpairedc-csv", default=DEFAULT_BALPAIREDC_CSV, help="Path to balpairedc CSV.")
    parser.add_argument("--fixthrbalpairedc-csv", default=DEFAULT_FIXTHR_BALPAIREDC_CSV, help="Path to fixthrbalpairedc CSV.")
    parser.add_argument("--ranking-csv", default=DEFAULT_RANKING_CSV, help="Path to ranking CSV.")
    parser.add_argument("--out", help="Optional path to write .tex output. If omitted, prints to stdout.")
    parser.add_argument(
        "--dataset-order",
        help="Comma-separated dataset order. Default: adience,agedb,calfw,cplfw,lfw,xqlfw",
    )
    parser.add_argument(
        "--fr-model-order",
        help="Comma-separated FR model order. Default: adaface,arcface,magface",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Comparison of FIQA methods across datasets. "
            "For EDC and balanced-pair EDC variants, pAUC@0.3 is computed on FNMR; "
            "for fixed-threshold variants, pAUC@0.3 is computed on CER; "
            "for ranking, results are reported for $\\alpha=0$ and $\\alpha=10$."
        ),
    )
    parser.add_argument("--label", default="tab:fiqa_results")
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Disable resizebox wrapper.",
    )
    args = parser.parse_args()

    dataset_order = parse_list_arg(args.dataset_order) or DEFAULT_DATASET_ORDER
    fr_model_order = parse_list_arg(args.fr_model_order) or DEFAULT_FR_MODEL_ORDER

    df = combine_sources(
        edc_csv=args.edc_csv,
        fixthredc_csv=args.fixthredc_csv,
        balpairedc_csv=args.balpairedc_csv,
        fixthrbalpairedc_csv=args.fixthrbalpairedc_csv,
        ranking_csv=args.ranking_csv,
    )

    latex = build_latex_table(
        df=df,
        dataset_order=dataset_order,
        fr_model_order=fr_model_order,
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
