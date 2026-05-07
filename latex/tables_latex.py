#!/usr/bin/env python3
"""Build 2 LaTeX tables from multiple FIQA result CSVs.

Tables:
    1) pAUC table + Meta block
    2) ranking table + Meta block

Method names:
    edc                 -> EDC
    fixthredc           -> FT
    balpairedc          -> PPD
    fixthrbalpairedc    -> FT-PPD
    ranking             -> EBR

Meta block:
    - In pAUC table:
        Meta = average rank across datasets for
               EDC | FT-EDC | PPD-EDC | FT-PPD-EDC
    - In ranking table:
        Meta = average rank across datasets for
               alpha=0 | alpha=3

Formatting:
    - Main cells: "(rank) value"
    - Meta cells: "(rank) avg_rank"
    - Best values are bold, second-best are underlined
    - For pAUC/CER: lower is better
    - For EBR: higher is better
    - For Meta average rank: lower is better

Run:
    python tables_latex.py --outdir latex/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ORDER = ["adience", "calfw", "cplfw", "lfw", "xqlfw"]
DEFAULT_FR_MODEL_ORDER = ["adaface", "arcface", "magface", "swinface"]

DEFAULT_EDC_CSV = "/home/bw/FIQA/evd_ijcb/output/evd/default_methods/comprehensive_results.csv"
DEFAULT_FIXTHR_EDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol1_evd_fix_thr/default_methods/comprehensive_results.csv"
DEFAULT_BALPAIREDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol2_same_gen_imp_min/default_methods/comprehensive_results.csv"
DEFAULT_FIXTHR_BALPAIREDC_CSV = "/home/bw/FIQA/evd_ijcb/output/sol3_same_gen_imp_min_fix_thr/default_methods/comprehensive_results.csv"
DEFAULT_RANKING_CSV = "/home/bw/FIQA/evd_ijcb/output/sol4_rank_sample/comprehensive_results.csv"

PAUC_METHOD_STRUCTURE = [
    ("edc", ["pAUC@0.3(fnmr)"]),
    ("fixthredc", ["pAUC@0.3(cer)"]),
    ("balpairedc", ["pAUC@0.3(fnmr)"]),
    ("fixthrbalpairedc", ["pAUC@0.3(cer)"]),
]

RANKING_METHOD_STRUCTURE = [
    ("ranking", ["alpha=0", "alpha=3"]),
]

METHOD_DISPLAY_MAP = {
    "edc": "EDC",
    "fixthredc": "FT",
    "balpairedc": "PPD",
    "fixthrbalpairedc": "FT-PPD",
    "ranking": "EBR",
}

METHOD_DIRECTION = {
    "edc": "min",
    "fixthredc": "min",
    "balpairedc": "min",
    "fixthrbalpairedc": "min",
    "ranking": "max",
}

FR_MODEL_DISPLAY_NAMES = {
    "arcface": r"ArcFace~\cite{Deng2018ArcFaceAA}",
    "magface": r"MagFace~\cite{Meng2021MagFaceAU}",
    "adaface": r"AdaFace~\cite{Kim2022AdaFaceQA}",
    "swinface": r"SwinFace~\cite{qin2023swinface}",
}

METHOD_DISPLAY_NAMES = {
    "faceqnet": r"FaceQnet~\cite{hernandez2019faceqnet}",
    "ser-fiq": r"SER-FIQA~\cite{Terhorst2020SERFIQUE}",
    "pfe": r"PFE~\cite{chen2021fast}",
    "magface": r"MagFace~\cite{Meng2021MagFaceAU}",
    "sdd-fiqa": r"SDD-FIQA~\cite{Ou2021SDDFIQAUF}",
    "lightqnet": r"LightQNet~\cite{9528058}",
    "faceqgen": r"FaceQGen~\cite{HernandezOrtega2021FaceQgenSD}",
    "faceqan": r"FaceQAN~\cite{Babnik2022FaceQANFI}",
    "cr-fiqa(L)": r"CR-FIQA (L)~\cite{Boutros2021CRFIQAFI}",
    "diffiqa": r"DifFIQA~\cite{babnik2023diffiqa}",
    "diffiqa(R)": r"DifFIQA (R)~\cite{babnik2023diffiqa}",
    "clib-fiqa": r"CLIB-FIQA~\cite{Ou_2024_CVPR}",
    "grafiqs": r"GraFIQs~\cite{kolf2024grafiqs}",
    "ediffiqa(S)": r"eDifFIQA (S)~\cite{10468647}",
    "ediffiqa(M)": r"eDifFIQA (M)~\cite{10468647}",
    "ediffiqa(L)": r"eDifFIQA (L)~\cite{10468647}",
    "froqAda": r"FROQ~\cite{Babnik2025FROQ1OF}",
    "vit-fiqa": r"ViT-FIQA~\cite{atzori2025vit}",
}

def _fiqa_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


METHOD_DISPLAY_NAMES_BY_KEY = {_fiqa_key(k): v for k, v in METHOD_DISPLAY_NAMES.items()}

# Common aliases seen in input CSVs (already-formatted names, minor variants).
FIQA_ALIASES_TO_KEY = {
    "ser-fiqa": "ser-fiq",
    "froq": "froqada",
}


def fiqa_to_display(name: str) -> str:
    k = _fiqa_key(name)
    canonical = FIQA_ALIASES_TO_KEY.get(k, k)
    return METHOD_DISPLAY_NAMES_BY_KEY.get(canonical, name)

FIQA_KEY_ORDER = [
    "faceqnet",
    "ser-fiq",
    "pfe",
    "magface",
    "sdd-fiqa",
    "lightqnet",
    "faceqgen",
    "faceqan",
    "cr-fiqa(l)",
    "diffiqa",
    "diffiqa(r)",
    "clib-fiqa",
    "grafiqs",
    "ediffiqa(s)",
    "ediffiqa(m)",
    "ediffiqa(l)",
    "froqada",
    "vit-fiqa",
]

FIQA_DISPLAY_ORDER = [METHOD_DISPLAY_NAMES_BY_KEY[k] for k in FIQA_KEY_ORDER]


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


def fiqa_display_name(name: str) -> str:
    return fiqa_to_display(name)


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
        alpha_cols = ["alpha=0", "alpha=3"]

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
    df_rank = prepare_ranking_df(load_csv(ranking_csv), ["alpha=0", "alpha=3"])

    df = pd.concat([df_edc, df_fix, df_bal, df_fixbal, df_rank], ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fr_model"] = df["fr_model"].astype(str).str.strip()
    df["fiqa"] = df["fiqa"].astype(str).str.strip()
    fiqa_keys = df["fiqa"].map(_fiqa_key).map(lambda k: FIQA_ALIASES_TO_KEY.get(k, k))
    df = df[fiqa_keys != "pcnet"].copy()
    df["fiqa"] = df["fiqa"].map(fiqa_to_display)
    return df


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
    show_rank: bool = True,
) -> str:
    if value is None or pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    if mark == "best":
        s = rf"\textbf{{{s}}}"
    elif mark == "second":
        s = rf"\underline{{{s}}}"

    if not show_rank:
        return s

    rank_text = _format_rank(rank)
    if rank_text is None:
        return s
    return f"({latex_escape(rank_text)}) {s}"


def format_meta_value(
    value: float | None,
    mark: str | None,
    rank_num: int | None,
    decimals: int = 2,
) -> str:
    if value is None or pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    if mark == "best":
        s = rf"\textbf{{{s}}}"
    elif mark == "second":
        s = rf"\underline{{{s}}}"
    if rank_num is None:
        return s
    return f"({rank_num}) {s}"


def compute_main_highlights(
    df: pd.DataFrame,
    fr_model_order: list[str],
    dataset_order: list[str],
    method_structure: list[tuple[str, list[str]]],
) -> dict[tuple[str, str, str, str, str], str]:
    highlight: dict[tuple[str, str, str, str, str], str] = {}

    method_subcols = [(m, s) for m, subcols in method_structure for s in subcols]

    df2 = df.copy()
    df2["dataset_key"] = df2["dataset"].astype(str).str.strip().str.lower()
    df2["fr_model_key"] = df2["fr_model"].astype(str).str.strip().str.lower()

    for fr_model in fr_model_order:
        subset_fr = df2[df2["fr_model_key"] == fr_model]
        if subset_fr.empty:
            continue

        for dataset in dataset_order:
            subset_ds = subset_fr[subset_fr["dataset_key"] == dataset]
            if subset_ds.empty:
                continue

            for method, subcol in method_subcols:
                subset = subset_ds[
                    (subset_ds["method"] == method) &
                    (subset_ds["subcol"] == subcol)
                ]
                if subset.empty:
                    continue

                vals: list[tuple[str, float]] = []
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
                    key = (fr_model, fiqa, dataset, method, subcol)
                    if best_val is not None and v == best_val:
                        highlight[key] = "best"
                    elif second_val is not None and v == second_val:
                        highlight[key] = "second"

    return highlight


def compute_meta_average_ranks(
    df: pd.DataFrame,
    dataset_keys: list[str],
    fr_model_keys: list[str],
    fiqa_order: list[str],
    meta_structure: list[tuple[str, list[str], str]],
) -> tuple[
    dict[tuple[str, str, str], float | None],
    dict[tuple[str, str, str], str],
    dict[tuple[str, str, str], int],
]:
    avg_rank_map: dict[tuple[str, str, str], float | None] = {}
    highlight_map: dict[tuple[str, str, str], str] = {}
    display_rank_map: dict[tuple[str, str, str], int] = {}

    df2 = df.copy()
    df2["dataset_key"] = df2["dataset"].astype(str).str.strip().str.lower()
    df2["fr_model_key"] = df2["fr_model"].astype(str).str.strip().str.lower()

    for fr_model in fr_model_keys:
        subset_fr = df2[df2["fr_model_key"] == fr_model]
        if subset_fr.empty:
            continue

        for method, subcols, display_name in meta_structure:
            vals_for_highlight: list[tuple[str, float]] = []

            for fiqa in fiqa_order:
                subset_fiqa = subset_fr[subset_fr["fiqa"] == fiqa]
                if subset_fiqa.empty:
                    continue

                ranks: list[float] = []
                for dataset in dataset_keys:
                    subset_ds = subset_fiqa[subset_fiqa["dataset_key"] == dataset]
                    if subset_ds.empty:
                        continue

                    for subcol in subcols:
                        if subcol == "avg":
                            # "avg" is a meta placeholder: aggregate ranks over all
                            # available subcolumns for this method in the dataset.
                            sub = subset_ds[subset_ds["method"] == method]
                        else:
                            sub = subset_ds[
                                (subset_ds["method"] == method) &
                                (subset_ds["subcol"] == subcol)
                            ]
                        if sub.empty:
                            continue
                        for rv in sub["rank"].tolist():
                            if pd.isna(rv):
                                continue
                            try:
                                ranks.append(float(rv))
                            except Exception:
                                continue

                key = (fr_model, fiqa, display_name)
                if ranks:
                    avg_val = sum(ranks) / len(ranks)
                    avg_rank_map[key] = avg_val
                    vals_for_highlight.append((fiqa, avg_val))
                else:
                    avg_rank_map[key] = None

            if vals_for_highlight:
                unique_vals = sorted({v for _, v in vals_for_highlight})  # lower is better
                best_val = unique_vals[0] if len(unique_vals) >= 1 else None
                second_val = unique_vals[1] if len(unique_vals) >= 2 else None
                value_to_display_rank = {v: i + 1 for i, v in enumerate(unique_vals)}

                for fiqa, v in vals_for_highlight:
                    key = (fr_model, fiqa, display_name)
                    display_rank_map[key] = value_to_display_rank[v]
                    if best_val is not None and v == best_val:
                        highlight_map[key] = "best"
                    elif second_val is not None and v == second_val:
                        highlight_map[key] = "second"

    return avg_rank_map, highlight_map, display_rank_map


def build_metric_table(
    df: pd.DataFrame,
    dataset_order: list[str] | None,
    fr_model_order: list[str] | None,
    method_structure: list[tuple[str, list[str]]],
    meta_structure: list[tuple[str, list[str], str]],
    caption: str,
    label: str,
    use_resizebox: bool = False,
    font_size: str = r"\footnotesize",
    tabcolsep: int = 4,
    show_rank: bool = True,
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

    method_names = {m for m, _ in method_structure}
    df2 = df2[df2["method"].isin(method_names)].copy()

    if dataset_order is None:
        dataset_order = df2["dataset"].drop_duplicates().tolist()
    else:
        existing = set(df2["dataset_key"].dropna().astype(str).tolist())
        dataset_order = [d for d in dataset_order if d.strip().lower() in existing]

    if fr_model_order is None:
        fr_model_order = df2["fr_model"].drop_duplicates().tolist()

    available_fiqas = set(df2["fiqa"].astype(str))
    fiqa_order = [f for f in FIQA_DISPLAY_ORDER if f in available_fiqas]
    remaining_fiqas = [f for f in df2["fiqa"].drop_duplicates().tolist() if f not in fiqa_order]
    fiqa_order.extend(remaining_fiqas)

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

    highlight_main = compute_main_highlights(df2, fr_model_keys, dataset_keys, method_structure)
    meta_avg_map, meta_highlight_map, meta_display_rank_map = compute_meta_average_ranks(
        df=df,
        dataset_keys=dataset_keys,
        fr_model_keys=fr_model_keys,
        fiqa_order=fiqa_order,
        meta_structure=meta_structure,
    )

    printed_blocks: list[tuple[str, list[str]]] = []
    for fr_model in fr_model_keys:
        mask_fr = df2["fr_model_key"] == fr_model
        fr_fiqas = [f for f in fiqa_order if (mask_fr & (df2["fiqa"] == f)).any()]
        if fr_fiqas:
            printed_blocks.append((fr_model, fr_fiqas))

    cols_per_dataset = sum(len(subcols) for _, subcols in method_structure)
    meta_cols = sum(len(subcols) for _, subcols, _ in meta_structure)
    total_data_cols = cols_per_dataset * len(dataset_keys) + meta_cols
    col_format = "ll|" + "|".join(["r"] * total_data_cols)

    tabular_lines: list[str] = []
    tabular_lines.append(r"\begin{tabular}{" + col_format + "}")
    tabular_lines.append(r"\toprule")

    header1 = [r"\multirow{3}{*}{FR}", r"\multirow{3}{*}{FIQA}"]
    for dataset in dataset_keys:
        header1.append(
            rf"\multicolumn{{{cols_per_dataset}}}{{c|}}{{{latex_escape(dataset_display_map.get(dataset, dataset))}}}"
        )
    header1.append(rf"\multicolumn{{{meta_cols}}}{{c}}{{Meta}}")
    tabular_lines.append(" & ".join(header1) + r" \\")

    header2 = ["", ""]
    for _dataset in dataset_keys:
        for method, subcols in method_structure:
            header2.append(rf"\multicolumn{{{len(subcols)}}}{{c|}}{{{latex_escape(METHOD_DISPLAY_MAP.get(method, method))}}}")
    for i, (_method, subcols, display_name) in enumerate(meta_structure):
        align = "c|" if i < len(meta_structure) - 1 else "c"
        header2.append(rf"\multicolumn{{{len(subcols)}}}{{{align}}}{{{display_name}}}")
    tabular_lines.append(" & ".join(header2) + r" \\")

    header3 = ["", ""]
    for _dataset in dataset_keys:
        for _method, subcols in method_structure:
            for subcol in subcols:
                header3.append(subcol_to_latex(subcol))
    for _method, subcols, _display_name in meta_structure:
        for subcol in subcols:
            if subcol.startswith("alpha="):
                header3.append(subcol_to_latex(subcol))
            else:
                header3.append(r"Avg.\ Rank")
    tabular_lines.append(" & ".join(header3) + r" \\")
    tabular_lines.append(r"\midrule")

    for block_i, (fr_model, fr_fiqas) in enumerate(printed_blocks):
        n_fiqas = len(fr_fiqas)
        for j, fiqa in enumerate(fr_fiqas):
            row = []
            row.append(
                rf"\multirow{{{n_fiqas}}}{{*}}{{\rotatebox{{90}}{{{FR_MODEL_DISPLAY_NAMES.get(fr_model, fr_display_map.get(fr_model, fr_model))}}}}}"
                if j == 0 else ""
            )
            row.append(fiqa)

            for dataset in dataset_keys:
                for method, subcols in method_structure:
                    for subcol in subcols:
                        v = value_map.get((fr_model, fiqa, dataset, method, subcol))
                        r = rank_map.get((fr_model, fiqa, dataset, method, subcol))
                        mark = highlight_main.get((fr_model, fiqa, dataset, method, subcol))
                        row.append(format_value(v, mark, r, show_rank=show_rank))

            for _method, _subcols, display_name in meta_structure:
                avg_v = meta_avg_map.get((fr_model, fiqa, display_name))
                avg_mark = meta_highlight_map.get((fr_model, fiqa, display_name))
                avg_r = meta_display_rank_map.get((fr_model, fiqa, display_name))
                row.append(format_meta_value(avg_v, avg_mark, avg_r))

            tabular_lines.append(" & ".join(row) + r" \\")

        if block_i < len(printed_blocks) - 1:
            tabular_lines.append(r"\midrule")

    tabular_lines.append(r"\bottomrule")
    tabular_lines.append(r"\end{tabular}")
    tabular = "\n".join(tabular_lines)

    outer: list[str] = []
    outer.append(r"\begin{table*}[t]")
    outer.append(r"\centering")
    outer.append(font_size)
    outer.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}")
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
    parser = argparse.ArgumentParser(description="Convert FIQA result CSVs to 2 LaTeX tables.")
    parser.add_argument("--edc-csv", default=DEFAULT_EDC_CSV, help="Path to EDC CSV.")
    parser.add_argument("--fixthredc-csv", default=DEFAULT_FIXTHR_EDC_CSV, help="Path to FT-EDC CSV.")
    parser.add_argument("--balpairedc-csv", default=DEFAULT_BALPAIREDC_CSV, help="Path to PPD-EDC CSV.")
    parser.add_argument("--fixthrbalpairedc-csv", default=DEFAULT_FIXTHR_BALPAIREDC_CSV, help="Path to FT-PPD-EDC CSV.")
    parser.add_argument("--ranking-csv", default=DEFAULT_RANKING_CSV, help="Path to EBR CSV.")
    parser.add_argument("--outdir", required=True, help="Directory to write the .tex files.")
    parser.add_argument(
        "--dataset-order",
        help="Comma-separated dataset order. Default: adience,calfw,cplfw,lfw,xqlfw",
    )
    parser.add_argument(
        "--fr-model-order",
        help="Comma-separated FR model order. Default: adaface,arcface,magface,swinface",
    )
    parser.add_argument(
        "--pauc-no-rank-prefix",
        action="store_true",
        help="Do not show (rank) prefix in pAUC table cells.",
    )
    parser.add_argument(
        "--ranking-no-rank-prefix",
        action="store_true",
        help="Do not show (rank) prefix in ranking table cells.",
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pauc_tex = build_metric_table(
        df=df,
        dataset_order=dataset_order,
        fr_model_order=fr_model_order,
        method_structure=PAUC_METHOD_STRUCTURE,
        meta_structure=[
            ("edc", ["avg"], "EDC"),
            ("fixthredc", ["avg"], "FT-EDC"),
            ("balpairedc", ["avg"], "PPD-EDC"),
            ("fixthrbalpairedc", ["avg"], "FT-PPD-EDC"),
        ],
        caption=(
            "Comparison of FIQA methods across datasets in terms of pAUC@0.3. "
            "For EDC and PPD-EDC, pAUC@0.3 is computed on FNMR; "
            "for FT-EDC and FT-PPD-EDC, pAUC@0.3 is computed on CER. "
            "The Meta block reports the average rank across datasets for each evaluation protocol. "
            "Best and second-best results are highlighted."
        ),
        label="tab:fiqa_pauc",
        use_resizebox=False,
        font_size=r"\scriptsize",
        tabcolsep=3,
        show_rank=not args.pauc_no_rank_prefix,
    )

    ranking_tex = build_metric_table(
        df=df,
        dataset_order=dataset_order,
        fr_model_order=fr_model_order,
        method_structure=RANKING_METHOD_STRUCTURE,
        meta_structure=[
            ("ranking", ["alpha=0"], r"$\alpha=0$"),
            ("ranking", ["alpha=3"], r"$\alpha=3$"),
        ],
        caption=(
            "Comparison of FIQA methods across datasets in terms of EBR ranking scores. "
            "Results are reported for $\\alpha=0$ and $\\alpha=3$. "
            "The Meta block reports the average rank across datasets for each setting. "
            "Best and second-best results are highlighted."
        ),
        label="tab:fiqa_ranking",
        use_resizebox=True,
        font_size=r"\footnotesize",
        tabcolsep=4,
        show_rank=not args.ranking_no_rank_prefix,
    )

    (outdir / "table_pauc.tex").write_text(pauc_tex + "\n", encoding="utf-8")
    (outdir / "table_ranking.tex").write_text(ranking_tex + "\n", encoding="utf-8")

    print(f"Wrote: {outdir / 'table_pauc.tex'}")
    print(f"Wrote: {outdir / 'table_ranking.tex'}")


if __name__ == "__main__":
    main()
