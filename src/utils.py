import json
import os

import numpy as np
from scipy import stats

import pandas as pd


def save_output(output_list: list | pd.DataFrame, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if isinstance(output_list, pd.DataFrame):
        df = output_list
    else:
        df = pd.DataFrame(output_list)
    if output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=False)
    elif output_file.endswith(".jsonl"):
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported output file format: {output_file}")


def jsonl_to_xlsx(jsonl_file, xlsx_file):
    jsonl_df = pd.read_json(jsonl_file, lines=True)
    jsonl_df.to_excel(xlsx_file, index=False)
    jsonl_df.to_csv(xlsx_file.replace(".xlsx", ".csv"), index=False)
    print(f"Finished convert {jsonl_file}")




def compute_t_statistics(data, mu_0: float = 0.0, ci: float = 0.95) -> dict:
    """
    Compute one-sample t-test statistics for a given dataset.

    Parameters
    ----------
    data : list[float] | np.ndarray
        The sample data for which statistics are computed.
    mu_0 : float, default = 0.0
        The null hypothesis mean.
    ci : float, default = 0.95
        Confidence level for confidence interval.

    Returns
    -------
    dict
        Dictionary containing key statistical metrics:

        {
            "n": int,
                Number of valid (non-NaN) observations.

            "mean": float,
                Sample mean of the data.

            "std": float,
                Sample standard deviation (using ddof=1).

            "stderr": float,
                Standard error of the mean = std / sqrt(n).

            "t_stat": float,
                One-sample t-statistic computed as (mean - mu_0) / stderr.

            "p_value": float,
                Two-tailed p-value corresponding to the t-statistic.

            "ci_lower": float,
                Lower bound of the (1 - α) confidence interval for the mean.

            "ci_upper": float,
                Upper bound of the (1 - α) confidence interval for the mean.
        }
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)

    if n == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "stderr": np.nan,
            "t_stat": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan
        }

    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1) if n > 1 else 0.0
    stderr = std_val / np.sqrt(n) if n > 1 else np.nan

    if stderr > 0 and n > 1:
        t_stat = (mean_val - mu_0) / stderr
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
        ci_low, ci_high = stats.t.interval(ci, df=n - 1, loc=mean_val, scale=stderr)
    else:
        t_stat, p_value, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan

    return {
        "n": n,
        "mean": float(mean_val),
        "std": float(std_val),
        "stderr": float(stderr),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high)
    }


def bootstrap_stats(
    data_list, sampling_size: int = 20, sampling_step: int = 1000,
    ci: float = 0.95, mu_0: float = 0.0, output_file: str = None
) -> dict:
    """
    Bootstrap-based estimation of confidence intervals and t-statistics.

    Uses repeated random sampling (with replacement) to estimate the distribution
    of the mean, and then applies one-sample t-test statistics using `compute_t_statistics()`.

    Parameters
    ----------
    data_list : list or np.ndarray
        Input numeric data.
    sampling_size : int, default = 20
        Size of each bootstrap sample.
    sampling_step : int, default = 1000
        Number of bootstrap iterations.
    ci : float, default = 0.95
        Confidence level.
    mu_0 : float, default = 0.0
        Null hypothesis mean.
    output_file : str, optional
        If provided, saves results to this file (.json or .jsonl).

    Returns
    -------
    dict
        {
            "bootstrap_mean": float,
            "ci_lower": float,
            "ci_upper": float,
            "t_stat": float,
            "p_value": float
        }
    """
    data_array = np.array(data_list, dtype=float)
    data_array = data_array[~np.isnan(data_array)]

    if len(data_array) == 0:
        raise ValueError("Input data_list contains no valid numeric entries.")

    # ✅ Bootstrap resampling
    bootstrap_means = [
        np.mean(np.random.choice(data_array, size=sampling_size, replace=True))
        for _ in range(sampling_step)
    ]

    # ✅ Compute t-statistics from bootstrap means
    stats_result = compute_t_statistics(bootstrap_means, mu_0=mu_0, ci=ci)

    # ✅ Rename for clarity
    result = {
        "bootstrap_mean": stats_result["mean"],
        "ci_lower": stats_result["ci_lower"],
        "ci_upper": stats_result["ci_upper"],
        "t_stat": stats_result["t_stat"],
        "p_value": stats_result["p_value"],
        "n": stats_result["n"],
        "std": stats_result["std"],
        "stderr": stats_result["stderr"]
    }

    # ✅ Optional: Save output
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
