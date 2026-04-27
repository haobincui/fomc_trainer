import argparse
from log_config import *
import glob
import json
import logging
import os
import re

import pandas as pd


from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import get_cached_embedding_model
from open_r1.validator.shapley.shapley_calc import shapley_value_calc
from utils import bootstrap_stats, compute_t_statistics, save_output


import pandas as pd
import numpy as np


DEFAULT_EMBEDDING_MODEL_PATH = "output/training/main/merged/analysis_sft"


def get_embedding_model(model_path: str = DEFAULT_EMBEDDING_MODEL_PATH):
    return get_cached_embedding_model(model_path)


def significance_stars(p_value: float) -> str:
    """Add significance stars based on p-value."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "$^{***}$"
    elif p_value < 0.01:
        return "$^{**}$"
    elif p_value < 0.05:
        return "$^{*}$"
    elif p_value < 0.1:
        return "$^{\\cdot}$"
    else:
        return ""


def generate_latex_shapley_table(input_file: str, output_file: str):
    """
    Generate a LaTeX table showing mean (top line) and t-stat with significance (bottom line)
    for each section and indicator.
    """
    # 1️⃣ 读取数据
    df = pd.read_excel(input_file)

    # 2️⃣ 保留必要列
    df = df[["indicator", "section_name", "mean", "t_stat", "p_value"]]

    # 3️⃣ 设置列顺序
    section_order = [
        "Participants' Views on Current Conditions and the Economic Outlook",
        "Staff Review of the Economic Situation",
        "Staff Review of the Financial Situation"
    ]

    # 4️⃣ 创建透视表（mean 与 t-stat 分别透视）
    mean_df = df.pivot(index="indicator", columns="section_name", values="mean")
    t_df = df.pivot(index="indicator", columns="section_name", values="t_stat")
    p_df = df.pivot(index="indicator", columns="section_name", values="p_value")

    mean_df = mean_df.reindex(columns=section_order)
    t_df = t_df.reindex(columns=section_order)
    p_df = p_df.reindex(columns=section_order)

    # 5️⃣ 生成 LaTeX 表格
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l" + "c" * len(section_order) + "}")
    latex_lines.append("\\hline")
    header = "Indicator & " + " & ".join(section_order) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")

    for indicator in mean_df.index:
        # 第一行：mean
        means = [f"{mean_df.loc[indicator, c]:.4f}" if pd.notna(mean_df.loc[indicator, c]) else "" for c in section_order]
        line1 = f"{indicator} & " + " & ".join(means) + " \\\\"

        # 第二行：t_stat + significance
        tstats = []
        for c in section_order:
            t = t_df.loc[indicator, c]
            p = p_df.loc[indicator, c]
            if pd.isna(t):
                tstats.append("")
            else:
                stars = significance_stars(p)
                tstats.append(f"({t:.2f}{stars})")
        line2 = " & " + " & ".join(tstats) + " \\\\"

        latex_lines.append(line1)
        latex_lines.append(line2)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")

    # 6️⃣ 保存为 .tex 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"✅ LaTeX table saved to: {output_file}")



from scipy import stats


def calc_shapley_diff(input_file: str, output_file: str, base_indicator: str = "None") -> pd.DataFrame:
    """
    Calculate Δmean (% change) and significance (t-stat, p-value)
    for each indicator relative to the base indicator ("None").

    Parameters
    ----------
    input_file : str
        Excel file path containing columns:
        ["indicator", "section_name", "mean", "std", "n"]
    output_file : str
        Output Excel path to save results.
    base_indicator : str, default="None"
        The baseline indicator used for comparison.

    Returns
    -------
    pd.DataFrame
        DataFrame with Δmean, t-statistic, p-value for each indicator-section pair.
    """

    df = pd.read_excel(input_file)

    required_cols = {"indicator", "section_name", "mean", "std", "n"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Input file missing required columns: {required_cols - set(df.columns)}")

    base_df = df[df["indicator"] == base_indicator].copy()
    others_df = df[df["indicator"] != base_indicator].copy()

    if base_df.empty:
        raise ValueError(f"❌ No baseline found for indicator '{base_indicator}'.")

    output_rows = []

    for indicator in others_df["indicator"].unique():
        ind_df = others_df[others_df["indicator"] == indicator]

        for section in ind_df["section_name"].unique():
            base_row = base_df.loc[base_df["section_name"] == section]
            ind_row = ind_df.loc[ind_df["section_name"] == section]

            if base_row.empty or ind_row.empty:
                continue

            mean_base = base_row["mean"].values[0]
            mean_ind = ind_row["mean"].values[0]
            std_base = base_row["std"].values[0]
            std_ind = ind_row["std"].values[0]
            n_base = base_row["n"].values[0]
            n_ind = ind_row["n"].values[0]

            # Δmean (%)
            delta_mean = ((mean_ind - mean_base) / mean_base) * 100

            # Welch's t-test
            se_diff = np.sqrt((std_base ** 2 / n_base) + (std_ind ** 2 / n_ind))
            if se_diff == 0:
                t_stat, p_value = np.nan, np.nan
            else:
                t_stat = (mean_ind - mean_base) / se_diff
                df_denom = ((std_base ** 2 / n_base) + (std_ind ** 2 / n_ind)) ** 2
                df_num = ((std_base ** 4) / (n_base ** 2 * (n_base - 1))) + ((std_ind ** 4) / (n_ind ** 2 * (n_ind - 1)))
                df_eff = df_denom / df_num if df_num > 0 else np.nan
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_eff)) if not np.isnan(t_stat) else np.nan

            output_rows.append({
                "indicator": indicator,
                "section_name": section,
                # "mean_base": mean_base,
                # "mean_indicator": mean_ind,
                "mean": delta_mean,
                "t_stat": t_stat,
                "p_value": p_value
            })

    diff_df = pd.DataFrame(output_rows)

    diff_df.to_excel(output_file, index=False)
    print(f"✅ Shapley Δmean + t-test results saved to: {output_file}")

    return diff_df

def generate_latex_shapley_table_with_diff(input_file: str, output_file: str):
    """
    Generate a LaTeX table showing mean (top line) and t-stat with significance (bottom line)
    for each section and indicator.
    """
    # 1️⃣ 读取数据
    df = pd.read_excel(input_file)

    # 2️⃣ 保留必要列
    df = df[df["n"] >= 100]
    df = df[["indicator", "section_name", "delta_mean(%)", "t_stat", "p_value"]]

    # 3️⃣ 设置列顺序
    section_order = [
        "Participants' Views on Current Conditions and the Economic Outlook",
        "Staff Review of the Economic Situation",
        "Staff Review of the Financial Situation"
    ]

    # 4️⃣ 创建透视表（mean 与 t-stat 分别透视）
    mean_df = df.pivot(index="indicator", columns="section_name", values="delta_mean(%)")
    t_df = df.pivot(index="indicator", columns="section_name", values="t_stat")
    p_df = df.pivot(index="indicator", columns="section_name", values="p_value")

    mean_df = mean_df.reindex(columns=section_order)
    t_df = t_df.reindex(columns=section_order)
    p_df = p_df.reindex(columns=section_order)

    # 5️⃣ 生成 LaTeX 表格
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l" + "c" * len(section_order) + "}")
    latex_lines.append("\\hline")
    header = "Indicator & " + " & ".join(section_order) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")

    for indicator in mean_df.index:
        # 第一行：mean
        means = [f"{mean_df.loc[indicator, c]:.4f}" if pd.notna(mean_df.loc[indicator, c]) else "" for c in section_order]
        line1 = f"{indicator} & " + " & ".join(means) + " \\\\"

        # 第二行：t_stat + significance
        tstats = []
        for c in section_order:
            t = t_df.loc[indicator, c]
            p = p_df.loc[indicator, c]
            if pd.isna(t):
                tstats.append("")
            else:
                stars = significance_stars(p)
                tstats.append(f"({t:.2f}{stars})")
        line2 = " & " + " & ".join(tstats) + " \\\\"

        latex_lines.append(line1)
        latex_lines.append(line2)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")

    # 6️⃣ 保存为 .tex 文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"✅ LaTeX table saved to: {output_file}")







def _parse_answer(text: str) -> str:
    """
    Extract the content appearing after the <answer> tag from a given text.

    Parameters
    ----------
    text : str
        The full input text containing <answer> and other tags.

    Returns
    -------
    str
        The substring that appears after <answer>, with leading/trailing whitespace removed.
        If no <answer> tag is found, returns the raw text.
    """
    if not isinstance(text, str):
        return ""
    
    match = re.search(r"<answer>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return text





def assemble_section(input_dicts: list[dict]):
    """
    assemble inputs by section
    """
    grouped: dict[str, list[dict]] = {}
    for row in input_dicts:
        section_name = row.get("section_name", "Unknown")
        grouped.setdefault(section_name, []).append(row)
    return grouped




import json
import logging
from tqdm import tqdm

def calc_shapley_values(indicator: str, input_file: str, base_file: str) -> list:
    """
    Calculate line-by-line Shapley values for an indicator file compared to the baseline.
    """

    # ✅ 安全读取文件
    try:
        with open(input_file, "r", encoding="utf-8") as f_in, open(base_file, "r", encoding="utf-8") as f_base:
            input_lines = f_in.readlines()
            base_lines = f_base.readlines()
    except Exception as e:
        raise IOError(f"❌ Failed to read input or base file: {e}")

    # ✅ 如果行数不同，取最短长度
    if len(input_lines) != len(base_lines):
        min_len = min(len(input_lines), len(base_lines))
        logging.warning(
            f"⚠️ Line count mismatch between files:\n"
            f"  {input_file}: {len(input_lines)} lines\n"
            f"  {base_file}: {len(base_lines)} lines\n"
            f"  → Using first {min_len} lines for comparison."
        )
    else:
        min_len = len(input_lines)

    output_lines = []
    embedding_model = get_embedding_model()

    for i in tqdm(range(min_len), desc=f"Computing Shapley for {indicator}"):
        try:
            input_dict = json.loads(input_lines[i])
            base_dict = json.loads(base_lines[i])
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON decode error at line {i+1}: {e}")
            continue

        if input_dict.get("index") != base_dict.get("index"):
            logging.warning(
                f"⚠️ Mismatched index at line {i+1}: "
                f"input={input_dict.get('index')} vs base={base_dict.get('index')} "
                f"({input_file} vs {base_file})"
            )
            continue

        index = input_dict.get("index", i)
        section_name = input_dict.get("section_name", f"Section_{index}")

        input_generated = input_dict.get("generated", "")
        base_generated = base_dict.get("generated", "")

        # ✅ 提取回答内容
        input_answer = _parse_answer(input_generated)
        base_answer = _parse_answer(base_generated)

        # ✅ 计算 Shapley 值（基于语义相似度）
        try:
            shapley = shapley_value_calc(
                subset_with_p=base_answer,   # with indicator
                subset=input_answer,           # baseline
                utility_function=cosine_similarity_calc,
                kwargs={"generated": base_answer, "model_wrapper": embedding_model}
            )
            logging.info(f"✅ Shapley calculation success at line {i+1}")
        except Exception as e:
            logging.error(f"❌ Shapley calculation failed at line {i+1}: {e}")
            shapley = None

        output_lines.append({
            "indicator": indicator,
            "index": index,
            "section_name": section_name,
            "masked_generated": input_generated,
            "base_generated": base_generated,
            "shapley": shapley
        })

        logging.debug(f"✅ Finished line {index} in indicator [{indicator}]")

    return output_lines

        

def run_mask_eval(input_folder: str):
    """
    Calculate Shapley Values for each indicator file within the given folder.

    Each indicator JSONL file (e.g., 'GDP_masked_after_2009.jsonl') is compared
    with the baseline file (where indicator == 'None'), to compute its contribution.

    The results for each indicator are written to `output_file`.

    Parameters
    ----------
    input_folder : str
        Directory containing input .jsonl files for masked indicators.
        Example: "output/valiation/.../mask_prompt/20250602/ft_model/after_2009/"

    output_file : str
        Path to save the aggregated evaluation results.

    Notes
    -----
    - Expected input file format: "<indicator>_masked_after_XXXX.jsonl"
    - Baseline file must include 'None' in its name (e.g., "None_masked_after_2009.jsonl").
    - Requires `calc_shapley_values()` and `save_output()` to be defined elsewhere.
    """
    # ✅ 统一路径格式
    input_folder = input_folder.rstrip("/") + "/"

    # ✅ 读取所有 .jsonl 文件
    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not input_files:
        logging.warning(f"⚠️ No .jsonl files found in: {input_folder}")
        return

    # ✅ 构建 indicator map
    indicator_map = {}
    for input_file in input_files:
        indicator = os.path.basename(input_file).split("_")[0]
        indicator_map[indicator] = input_file

    # ✅ 检查 baseline
    if "None" not in indicator_map:
        raise ValueError(f"❌ No baseline file found in folder {input_folder}. Expected a file with 'None' prefix.")
    base_file = indicator_map.pop("None")

    total = len(indicator_map)
    s, f, n = 0, 0, 0

    logging.info(f"🚀 Starting Shapley evaluation for {total} indicators...")
    logging.info(f"📘 Baseline file: {base_file}")

    # ✅ 循环计算每个 indicator 的 shapley 值
    for indicator, input_file in indicator_map.items():
        n += 1
        try:
            output_lines = calc_shapley_values(indicator, input_file, base_file)
            output_file = input_file.replace("mask_prompt", "shapley")
            save_output(output_lines, output_file)
            s += 1
            logging.info(f"✅ Finished [{indicator}] ({n}/{total}) → Success {s}, Failed {f}. Saved to [{output_file}]")
        except Exception as e:
            f += 1
            logging.warning(
                f"❌ Failed [{indicator}] ({n}/{total}) → Success {s}, Failed {f}. Error: {type(e).__name__}: {e}"
            )

    logging.info("=" * 80)
    logging.info(f"🏁 Completed all indicators: Success {s}, Failed {f}, Total {n}/{total}")
    logging.info("=" * 80)

def calc_shapley_values_by_section(
    indicator: str,
    input_file: str,
    base_df: pd.DataFrame,
    embedding_model=None,
) -> list:
    """
    Calculate section-level Shapley values for a given indicator file
    compared to the baseline DataFrame (base_df).

    Each record in the masked indicator file is matched with the corresponding
    (section_name, meeting_date) entry in base_df, and their generated text is compared
    via semantic cosine similarity.

    Output format:
        {
            "indicator": str,
            "index": int,
            "section_name": str,
            "meeting_date": str,
            "masked_generated": str,
            "base_generated": str,
            "shapley": float
        }

    Parameters
    ----------
    indicator : str
        The indicator name (e.g. "Bank-Capital").
    input_file : str
        Path to the masked indicator .jsonl file.
    base_df : pd.DataFrame
        Baseline DataFrame with columns ["section_name", "meeting_date", "response"].
    embedding_model : EmbeddingModel
        Pre-loaded embedding model used for cosine similarity computation.

    Returns
    -------
    list[dict]
        List of Shapley results by section and meeting_date.
    """
    output_lines = []
    embedding_model = embedding_model or get_embedding_model()

    # ✅ 读取 masked 文件
    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            input_lines = f_in.readlines()
    except Exception as e:
        raise IOError(f"❌ Failed to read input file {input_file}: {e}")

    for i, input_line in enumerate(input_lines):
        try:
            input_dict = json.loads(input_line)
        except json.JSONDecodeError as e:
            logging.error(f"❌ JSON decode error at line {i+1}: {e}")
            continue

        section_name = input_dict.get("section_name", "Unknown")
        meeting_date = input_dict.get("meeting_date", "Unknown")
        index = input_dict.get("index", i + 1)

        # 获取 masked 和 base 的文本内容
        input_generated = input_dict.get("generated", "")
        base_match = base_df[
            (base_df["section_name"] == section_name)
            & (base_df["meeting_date"] == meeting_date)
        ]

        if base_match.empty:
            logging.warning(f"⚠️ No baseline match for [{section_name}] - {meeting_date}")
            continue

        base_generated = base_match["response"].iloc[0]

        # ✅ 提取 <answer> 后的文本
        input_answer = _parse_answer(input_generated)
        base_answer = _parse_answer(base_generated)

        # ✅ 计算 Shapley 值
        try:
            shapley = shapley_value_calc(
                subset_with_p=base_answer,   # with indicator
                subset=input_answer,           # baseline
                utility_function=cosine_similarity_calc,
                kwargs={"generated": base_answer, "model_wrapper": embedding_model}
            )

            logging.debug(f"✅ Success: [{indicator}] line {i+1} ({section_name})")
        except Exception as e:
            logging.error(f"❌ Shapley calc failed at line {i+1}: {e}")
            shapley = None

        # ✅ 记录结果
        output_lines.append({
            "indicator": indicator,
            "index": index,
            "section_name": section_name,
            "meeting_date": meeting_date,
            "masked_generated": input_generated,
            "base_generated": base_generated,
            "shapley": shapley
        })

    logging.info(f"🎯 Finished indicator [{indicator}] — total {len(output_lines)} records.")
    return output_lines





def run_eval_mask_by_section(input_folder: str, raw_fomc_file: str):
    """
    meeting_date
    section_name
    response
    """
        # ✅ 读取所有 .jsonl 文件
    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not input_files:
        logging.warning(f"⚠️ No .jsonl files found in: {input_folder}")
        return

    # ✅ 构建 indicator map
    indicator_map = {}
    for input_file in input_files:
        indicator = os.path.basename(input_file).split("_")[0]
        indicator_map[indicator] = input_file

    # ✅ 检查 baseline
    if "None" not in indicator_map:
        raise ValueError(f"❌ No baseline file found in folder {input_folder}. Expected a file with 'None' prefix.")
    raw_fomc_df = pd.read_json(raw_fomc_file, lines=True)


    total = len(indicator_map)
    s, f, n = 0, 0, 0

    logging.info(f"🚀 Starting Shapley evaluation for {total} indicators...")
    logging.info(f"📘 Baseline file: {raw_fomc_file}")

    # ✅ 循环计算每个 indicator 的 shapley 值
    for indicator, input_file in indicator_map.items():
        n += 1
        try:
            output_lines = calc_shapley_values_by_section(indicator, input_file, raw_fomc_df)
            output_file = input_file.replace("mask_prompt", "shapley_by_raw")
            save_output(output_lines, output_file)
            s += 1
            logging.info(f"✅ Finished [{indicator}] ({n}/{total}) → Success {s}, Failed {f}. Saved to [{output_file}]")
        except Exception as e:
            f += 1
            logging.warning(
                f"❌ Failed [{indicator}] ({n}/{total}) → Success {s}, Failed {f}. Error: {type(e).__name__}: {e}"
            )

    logging.info("=" * 80)
    logging.info(f"🏁 Completed all indicators: Success {s}, Failed {f}, Total {n}/{total}")
    logging.info("=" * 80)




def generate_shapley_table(result_folder: str, output_file: str):
    """
    Generate bootstrap summary statistics for Shapley results by section.

    Each input JSONL file should contain entries of the form:
        {
            "indicator": str,
            "index": int,
            "section_name": str,
            "masked_generated": str,
            "base_generated": str,
            "shapley": float
        }

    Output fields:
        indicator, section_name, bootstrap_mean, ci_lower, ci_upper, t_stat, p_value, t_critical

    Parameters
    ----------
    result_folder : str
        Path to folder containing shapley result JSONL files.
    output_file : str
        Path to output summary file (.jsonl or .xlsx).
    """
    result_files = sorted(glob.glob(os.path.join(result_folder, "*.jsonl")))
    if not result_files:
        logging.warning(f"⚠️ No result files found in {result_folder}")
        return

    output_list = []
    logging.info(f"📁 Found {len(result_files)} shapley result files in {result_folder}")

    for file_idx, result_file in enumerate(result_files, 1):
        try:
            result_df = pd.read_json(result_file, lines=True)
        except Exception as e:
            logging.error(f"❌ Failed to read {result_file}: {e}")
            continue

        if "indicator" not in result_df.columns or "shapley" not in result_df.columns:
            logging.warning(f"⚠️ Missing required columns in {result_file}, skipped.")
            continue

        indicator = result_df["indicator"].iloc[0]
        logging.info(f"\n📘 [{file_idx}/{len(result_files)}] Processing indicator: {indicator}")

        # 分组计算
        for section_name, sub_df in result_df.groupby("section_name"):
            try:
                shapley_values = sub_df["shapley"].dropna().tolist()
                if not shapley_values:
                    logging.warning(f"⚠️ Empty shapley values in section [{section_name}] of {indicator}")
                    continue

                stats_result = compute_t_statistics(shapley_values)
                cur_section_output = {
                    "indicator": indicator,
                    "section_name": section_name,
                    **stats_result
                }
                output_list.append(cur_section_output)
                logging.info(f"   ✅ Finished Section [{section_name}] in Indicator [{indicator}]")

            except Exception as e:
                logging.error(f"❌ Error in section [{section_name}] of indicator [{indicator}]: {e}")
                continue

        logging.info(f"🎉 Finished Indicator [{indicator}] with {len(result_df)} rows.")

    # ✅ 保存结果
    if output_list:
        save_output(output_list, output_file)
        logging.info(f"✅ Finished ALL ({len(output_list)} total sections). Results saved in [{output_file}].")
    else:
        logging.warning("⚠️ No valid results to save.") 
    

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import logging



def generate_shapley_table_with_diff(result_folder: str, output_file: str, base_indicator: str = "None"):
    """
    Generate summary statistics for Shapley values by section, and compute the
    difference (Δmean, t-stat, p-value) relative to the baseline indicator ("None").

    If the baseline ("None") is missing, mean_base defaults to 1.0.
    """

    result_files = sorted(glob.glob(os.path.join(result_folder, "*.jsonl")))
    if not result_files:
        logging.warning(f"⚠️ No result files found in {result_folder}")
        return

    all_results = []
    logging.info(f"📁 Found {len(result_files)} shapley result files in {result_folder}")

    # Step 1️⃣: Compute section-level statistics for each indicator
    for file_idx, result_file in enumerate(result_files, 1):
        try:
            result_df = pd.read_json(result_file, lines=True)
        except Exception as e:
            logging.error(f"❌ Failed to read {result_file}: {e}")
            continue

        if "indicator" not in result_df.columns or "shapley" not in result_df.columns:
            logging.warning(f"⚠️ Missing required columns in {result_file}, skipped.")
            continue

        indicator = result_df["indicator"].iloc[0]
        logging.info(f"\n📘 [{file_idx}/{len(result_files)}] Processing indicator: {indicator}")

        for section_name, sub_df in result_df.groupby("section_name"):
            shapley_values = sub_df["shapley"].dropna().to_numpy()
            n = len(shapley_values)
            if n == 0:
                continue

            # ✅ 使用你自定义的 compute_t_statistics 函数
            stats_result = compute_t_statistics(shapley_values)
            stats_result.update({
                "indicator": indicator,
                "section_name": section_name
            })
            all_results.append(stats_result)

    df = pd.DataFrame(all_results)
    if df.empty:
        logging.warning("⚠️ No valid Shapley results to process.")
        return

    # Step 2️⃣: Compute Δmean relative to baseline ("None")
    base_df = df[df["indicator"] == base_indicator].copy()
    base_exists = not base_df.empty

    if base_exists:
        logging.info(f"✅ Found baseline indicator '{base_indicator}' with {len(base_df)} sections.")
    else:
        logging.warning(f"⚠️ Baseline indicator '{base_indicator}' not found. Using mean_base = 1.0 for diff computation.")

    diff_rows = []

    for indicator in df["indicator"].unique():
        if indicator == base_indicator:
            continue

        ind_df = df[df["indicator"] == indicator]

        for section in ind_df["section_name"].unique():
            ind_row = ind_df[ind_df["section_name"] == section]
            if ind_row.empty:
                continue

            mean_ind = ind_row["mean"].values[0]
            std_ind = ind_row["std"].values[0]
            n_ind = ind_row["n"].values[0]

            if base_exists:
                base_row = base_df[base_df["section_name"] == section]
                if base_row.empty:
                    mean_base = 1.0
                    logging.debug(f"⚠️ Section [{section}] missing in baseline, using mean_base=1.0")
                else:
                    mean_base = base_row["mean"].values[0]
            else:
                mean_base = 1.0

            # ✅ 计算 Δmean (%)
            delta_mean = ((mean_ind - mean_base) / mean_base) * 100 if mean_base != 0 else np.nan

            # ✅ 使用 compute_t_statistics 计算差值的 t-test
            diff_data = np.array([mean_ind - mean_base])
            diff_stats = compute_t_statistics(diff_data, mu_0=0.0)

            diff_rows.append({
                "indicator": indicator,
                "section_name": section,
                "mean_base": mean_base,
                "mean_indicator": mean_ind,
                "delta_mean(%)": delta_mean,
                "diff_t_stat": diff_stats["t_stat"],
                "diff_p_value": diff_stats["p_value"]
            })

    diff_df = pd.DataFrame(diff_rows)

    # Step 3️⃣: 合并原统计结果与差值结果
    merged = df.merge(
        diff_df[["indicator", "section_name", "mean_base", "delta_mean(%)", "diff_t_stat", "diff_p_value"]],
        on=["indicator", "section_name"],
        how="left"
    )

    # Step 4️⃣: 保存结果
    save_output(merged, output_file)
    logging.info(f"✅ Finished ALL ({len(merged)} rows). Results saved in [{output_file}].")

    return merged

def reg_sv(result_folder: str, output_file: str):
    """
    Reg of Shapley Values,
    Reg 1: SV = \alpha + \beta1 D_indicator + \beta2 D_section + \beta3 SV_unmask
    Reg 2: SV ~ 1 + SV_unmask + EntityEffects + SectionEffects, negative beta for SV_unmask

    input: [{"n":6,"mean":0.1894173125,"std":0.0712461061,"stderr":0.029086101,"t_stat":6.5122964511,"p_value":0.0012756467,"ci_lower":0.1146491095,"ci_upper":0.2641855155,"indicator":"Bank-Capital","section_name":",","mean_base":1.0,"delta_mean(%)":-81.0582687483,"diff_t_stat":null,"diff_p_value":null}]

    

    """
    target_secions = [
            "Participants' Views on Current Conditions and the Economic Outlook",
            "Staff Review of the Economic Situation",
            "Staff Review of the Financial Situation"
        ]
    

    result_files = sorted(glob.glob(os.path.join(result_folder, "*.jsonl")))
    if not result_files:
        logging.warning(f"⚠️ No result files found in {result_folder}")
        return

    all_results = []
    logging.info(f"📁 Found {len(result_files)} shapley result files in {result_folder}")

    # Step 1️⃣: Compute section-level statistics for each indicator
    for file_idx, result_file in enumerate(result_files, 1):
        try:
            result_df = pd.read_json(result_file, lines=True)
        except Exception as e:
            logging.error(f"❌ Failed to read {result_file}: {e}")
            continue

        if "indicator" not in result_df.columns or "shapley" not in result_df.columns:
            logging.warning(f"⚠️ Missing required columns in {result_file}, skipped.")
            continue

        indicator = result_df["indicator"].iloc[0]
        logging.info(f"\n📘 [{file_idx}/{len(result_files)}] Processing indicator: {indicator}")

        for section_name, sub_df in result_df.groupby("section_name"):
            shapley_values = sub_df["shapley"].dropna().to_numpy()
            n = len(shapley_values)
            if n == 0:
                continue

            # ✅ 使用你自定义的 compute_t_statistics 函数
            stats_result = compute_t_statistics(shapley_values)
            stats_result.update({
                "indicator": indicator,
                "section_name": section_name
            })
            all_results.append(stats_result)

    
    # input_df = pd.read_json(shapley_result_file, lines=True)
    # input_df = input_df[input_df["section_name"] in target_secions]

    



    from linearmodels import PanelOLS

    model = PanelOLS.from_formula(
        "SV ~ 1 + SV_unmask + EntityEffects + SectionEffects",
        data=panel_df
    )
    res = model.fit(cov_type="robust")
    print(res.summary)

    pass






def run_test1(input_folder: str, output_folder: str):
    """
    test 1: calculated shapley with None mask generation 
    """
    print("Running test 1: calculated shapley with None mask generation ")
    run_mask_eval(input_folder)

    shapley_output_file = output_folder + "/shapley_result/shapley_result.jsonl"

    shapley_input_folder = input_folder.replace("mask_prompt", "shapley")
    generate_shapley_table(shapley_input_folder, shapley_output_file)

    shapley_output_xl = output_folder + "/shapley_result/test1_shapley_result.xlsx"
    save_output(pd.read_json(shapley_output_file, lines = True), shapley_output_xl)
    generate_latex_shapley_table(shapley_output_xl, shapley_output_xl.replace(".xlsx", ".tex"))

    print(f"Finished Test 1, result saved in [{shapley_output_xl}]")

def run_test2(input_folder: str, raw_fomc_file: str, output_folder: str):
    """
    test 2: calculated shapley with raw fomc minutes
    """
    print("Running test 2: calculated shapley with raw fomc minutes")
    run_eval_mask_by_section(input_folder, raw_fomc_file)

    shapley_input_folder = input_folder.replace("mask_prompt", "shapley_by_raw")
    shapley_output_file = output_folder + "/shapley_result/test2_shapley_by_raw_result.jsonl"
    generate_shapley_table(shapley_input_folder, shapley_output_file)

    shapley_output_xl = output_folder + "/shapley_result/test2_shapley_by_raw_result.xlsx"
    save_output(pd.read_json(shapley_output_file, lines = True), shapley_output_xl)

    generate_latex_shapley_table(shapley_output_xl, shapley_output_xl.replace(".xlsx", ".tex"))
    print(f"Finished Test 2, result saved in [{shapley_output_xl}]")

def run_test3(input_folder: str, output_file: str):
    """
    test 3: percentage diff of shapley with synthetic (None as base)
    """
    print("test 3: percentage diff of shapley with synthetic ")
    generate_shapley_table_with_diff(input_folder, output_file)
    shapley_output_xl = output_file.replace(".jsonl", ".xlsx")
    save_output(pd.read_json(output_file, lines = True), shapley_output_xl)

    generate_latex_shapley_table_with_diff(shapley_output_xl, shapley_output_xl.replace(".xlsx", ".tex"))
    print(f"Finished Test 3, result saved in [{shapley_output_xl}]")


def run_test4(input_folder: str, output_file: str):
    """
    test4: percentage diff of shapley with raw minutes (None as base)
    """
    generate_shapley_table_with_diff(input_folder, output_file)
    shapley_output_xl = output_file.replace(".jsonl", ".xlsx")
    save_output(pd.read_json(output_file, lines = True), shapley_output_xl)
    generate_latex_shapley_table_with_diff(shapley_output_xl, shapley_output_xl.replace(".xlsx", ".tex"))
    print(f"Finished Test 4, result saved in [{shapley_output_xl}]")



def run_test5():
    raise NotImplementedError("run_test5() has not been stabilized and is intentionally disabled.")







   



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leave-one-out masking evaluation utilities.")
    parser.add_argument(
        "command",
        choices=["test1", "test2", "test3", "test4"],
        help="Named masking evaluation workflow to run.",
    )
    parser.add_argument("--input-folder", required=True, help="Input folder for the selected masking workflow.")
    parser.add_argument("--output-file", help="Output .jsonl file for test3/test4.")
    parser.add_argument("--output-folder", help="Output folder for test1/test2.")
    parser.add_argument(
        "--raw-fomc-file",
        default="dataset/processed/main/input_sources/synthetic_text/source_20250520.jsonl",
        help="Reference raw FOMC minutes file used by test2.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.command == "test1":
        if not args.output_folder:
            raise ValueError("--output-folder is required for test1")
        run_test1(args.input_folder, args.output_folder)
    elif args.command == "test2":
        if not args.output_folder:
            raise ValueError("--output-folder is required for test2")
        run_test2(args.input_folder, args.raw_fomc_file, args.output_folder)
    elif args.command == "test3":
        if not args.output_file:
            raise ValueError("--output-file is required for test3")
        run_test3(args.input_folder, args.output_file)
    elif args.command == "test4":
        if not args.output_file:
            raise ValueError("--output-file is required for test4")
        run_test4(args.input_folder, args.output_file)



    









    



    





    
        


        




