import pandas as pd
import numpy as np


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



if __name__ == '__main__':
    # input_file = "./shapley_by_raw_filter.xlsx"
    # output_file = "./shapley_by_raw_table.tex"
    # generate_latex_shapley_table(input_file, output_file)

    # %% calc shapley diff
    # input_file = "./shapley_by_raw_filter.xlsx"
    # diff_output_file = "./shapley_diff.xlsx"
    # latex_output_file = "./shapley_diff_table.tex"

    # # Step 1️⃣: 计算差值表
    # diff_df = calc_shapley_diff(input_file, diff_output_file, base_indicator="None")
    #
    # # Step 2️⃣: 生成 LaTeX 输出
    # generate_latex_shapley_table(diff_output_file, latex_output_file)

    # %% generate shapley with diff
    input_file = "./shapley_result_with_diff.xlsx"
    output_file = "./shapley_result_with_diff_table.tex"
    generate_latex_shapley_table_with_diff(input_file, output_file)



