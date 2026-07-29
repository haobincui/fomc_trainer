from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import (
    COARSE_LABELS,
    compute_classification_metrics,
    coarse_vote,
    extract_meeting_date,
    load_jsonl,
    write_json,
)
from open_r1.utils.fomc import parse_boxed_vote

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:  # pragma: no cover - handled via CLI error
    ColumnTransformer = None
    SimpleImputer = None
    LogisticRegression = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None


DECISION_DATASET_ROOT = ROOT / "dataset" / "processed" / "main" / "datasets" / "decision_grpo"
MARKET_BASELINE_DEFAULT = ROOT / "dataset" / "external" / "market_baselines" / "market_implied_baseline.csv"
INPUT_DATA_ROOT = ROOT / "src" / "process_fomc_report" / "generate_prompt_and_response" / "input_data"


@dataclass
class SeriesStore:
    fedfunds: pd.DataFrame
    treasury_2y: pd.DataFrame
    treasury_10y: pd.DataFrame
    cpi: pd.DataFrame
    unemployment: pd.DataFrame
    sp500: pd.DataFrame


def _normalize_month_series(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "observation_date" in normalized.columns:
        normalized["observation_date"] = pd.to_datetime(normalized["observation_date"], errors="coerce")
        return normalized.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)

    if {"Year", "Period", "Value"}.issubset(normalized.columns):
        normalized["Period"] = normalized["Period"].astype(str)
        normalized = normalized[normalized["Period"].str.match(r"M\d{2}")]
        normalized["observation_date"] = pd.to_datetime(
            normalized["Year"].astype(str) + "-" + normalized["Period"].str[1:] + "-01",
            errors="coerce",
        )
        normalized["Value"] = pd.to_numeric(normalized["Value"], errors="coerce")
        return normalized.dropna(subset=["observation_date", "Value"]).sort_values("observation_date").reset_index(drop=True)

    raise ValueError(f"Unsupported monthly series format: columns={normalized.columns.tolist()}")


def _normalize_sp500(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["observation_date"] = pd.to_datetime(normalized["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    normalized["Close"] = (
        normalized["Close"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return normalized.dropna(subset=["observation_date", "Close"]).sort_values("observation_date").reset_index(drop=True)


def load_series_store() -> SeriesStore:
    treasury_root = INPUT_DATA_ROOT / "us_data" / "Treasury Yields"
    cpi_root = INPUT_DATA_ROOT / "us_data" / "Consumer Price Index (CPI)"
    unemployment_root = INPUT_DATA_ROOT / "us_data" / "Unemployment Rate"
    equity_root = INPUT_DATA_ROOT / "us_data" / "Equity Market Indices"
    ffr_root = INPUT_DATA_ROOT / "ffr"

    fedfunds = pd.read_csv(ffr_root / "FEDFUNDS.csv")
    fedfunds["observation_date"] = pd.to_datetime(fedfunds["Effective Date"], format="%Y/%m/%d", errors="coerce")
    fedfunds = fedfunds.dropna(subset=["observation_date"]).sort_values("observation_date").reset_index(drop=True)

    treasury_2y = pd.read_csv(
        treasury_root / "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity(Percent, Not Seasonally Adjusted).csv"
    )
    treasury_10y = pd.read_csv(
        treasury_root / "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity(Percent, Not Seasonally Adjusted).csv"
    )
    treasury_2y["observation_date"] = pd.to_datetime(treasury_2y["observation_date"], errors="coerce")
    treasury_10y["observation_date"] = pd.to_datetime(treasury_10y["observation_date"], errors="coerce")

    cpi = _normalize_month_series(
        pd.read_csv(cpi_root / "Consumer Price Index for All Urban Consumers (CPI-U, seasonal adjusted).csv")
    )
    unemployment = _normalize_month_series(
        pd.read_csv(unemployment_root / "Unemployment Rate (Seasonal adjusted).csv")
    )
    sp500 = _normalize_sp500(pd.read_csv(equity_root / "S&P 500 Index.csv"))

    return SeriesStore(
        fedfunds=fedfunds,
        treasury_2y=treasury_2y,
        treasury_10y=treasury_10y,
        cpi=cpi,
        unemployment=unemployment,
        sp500=sp500,
    )


def _latest_row(df: pd.DataFrame, meeting_date: pd.Timestamp) -> pd.Series:
    eligible = df.loc[df["observation_date"] <= meeting_date]
    if eligible.empty:
        raise ValueError(f"No data available on or before {meeting_date.date()} in series.")
    return eligible.iloc[-1]


def _lagged_row(df: pd.DataFrame, meeting_date: pd.Timestamp, months: int) -> pd.Series:
    cutoff = (meeting_date.to_period("M") - months).to_timestamp()
    eligible = df.loc[df["observation_date"] <= cutoff]
    if eligible.empty:
        raise ValueError(f"No lagged data available for {meeting_date.date()} ({months} months)")
    return eligible.iloc[-1]


def build_feature_table(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "eval", "test"):
        for row in load_jsonl(dataset_root / f"{split}.jsonl"):
            rows.append(
                {
                    "meeting_date": str(row["meeting_date"])[:10],
                    "split": split,
                    "rate_change": row["rate_change"],
                    "current_rate": float(row["current_rate"]),
                }
            )

    meetings = pd.DataFrame(rows).sort_values("meeting_date").drop_duplicates(subset=["meeting_date"], keep="first").reset_index(drop=True)
    meetings["lag1_action"] = meetings["rate_change"].shift(1).fillna("No change")

    series = load_series_store()
    records = []
    for row in meetings.to_dict(orient="records"):
        meeting_ts = pd.Timestamp(row["meeting_date"])
        fedfunds_now = _latest_row(series.fedfunds, meeting_ts)
        fedfunds_lag = _lagged_row(series.fedfunds, meeting_ts, 3)
        treasury_2y = _latest_row(series.treasury_2y, meeting_ts)
        treasury_10y = _latest_row(series.treasury_10y, meeting_ts)
        cpi_now = _latest_row(series.cpi, meeting_ts)
        cpi_lag = _lagged_row(series.cpi, meeting_ts, 12)
        unemployment = _latest_row(series.unemployment, meeting_ts)
        sp500_now = _latest_row(series.sp500, meeting_ts)
        sp500_lag = _lagged_row(series.sp500, meeting_ts, 1)

        records.append(
            {
                **row,
                "current_rate": float(row["current_rate"]),
                "EFFR_3m_change": float(fedfunds_now["Rate (%)"]) - float(fedfunds_lag["Rate (%)"]),
                "two_year_yield": float(treasury_2y["DGS2"]),
                "ten_two_spread": float(treasury_10y["DGS10"]) - float(treasury_2y["DGS2"]),
                "CPI_yoy": (float(cpi_now["Value"]) / float(cpi_lag["Value"]) - 1.0) * 100.0,
                "unemployment_rate": float(unemployment["Value"]),
                "SP500_1m_return": (float(sp500_now["Close"]) / float(sp500_lag["Close"]) - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(records)


def _exact_match_metrics(df: pd.DataFrame, prediction_col: str) -> dict:
    predictions = df[prediction_col].fillna("").astype(str)
    invalid_mask = predictions.eq("")
    return {
        "n_meetings": int(len(df)),
        "exact_match": float((df["rate_change"].astype(str) == predictions).mean()),
        "invalid_predictions": int(invalid_mask.sum()),
        "target_vote_counts": {str(k): int(v) for k, v in df["rate_change"].value_counts(dropna=False).items()},
        "predicted_vote_counts": {str(k): int(v) for k, v in predictions.value_counts(dropna=False).items()},
    }


def _coarse_metrics(df: pd.DataFrame, prediction_col: str) -> dict:
    coarse_true = df["rate_change"].map(coarse_vote).tolist()
    coarse_pred = df[prediction_col].map(coarse_vote).tolist()
    return compute_classification_metrics(coarse_true, coarse_pred, COARSE_LABELS)


def _load_model_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported prediction file: {path}")

    if "meeting_date" not in df.columns:
        df["meeting_date"] = df.apply(lambda row: extract_meeting_date(str(row.get("prompt", ""))), axis=1)
    else:
        df["meeting_date"] = df["meeting_date"].astype(str).str[:10]

    if "generated_vote" not in df.columns:
        if "generated" in df.columns:
            df["generated_vote"] = df["generated"].fillna("").map(parse_boxed_vote)
        elif "prediction" in df.columns:
            df["generated_vote"] = df["prediction"].fillna("").astype(str)
        else:
            raise ValueError(f"{path} is missing generated_vote/generated/prediction columns")

    if "index" not in df.columns:
        df["index"] = range(len(df))

    df["has_vote"] = df["generated_vote"].fillna("").astype(str).ne("")
    df = df.sort_values(["meeting_date", "has_vote", "index"], ascending=[True, False, True])
    return df.drop_duplicates(subset=["meeting_date"], keep="first").reset_index(drop=True)


def _maybe_load_market_baseline(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "meeting_date" not in df.columns:
        raise ValueError(f"Market baseline file {path} must contain meeting_date")
    if "predicted_label" not in df.columns:
        raise ValueError(f"Market baseline file {path} must contain predicted_label")
    df["meeting_date"] = df["meeting_date"].astype(str).str[:10]
    return df.drop_duplicates(subset=["meeting_date"], keep="first").reset_index(drop=True)


def _fit_multinomial_logit(feature_table: pd.DataFrame) -> pd.DataFrame:
    if Pipeline is None:
        raise ImportError("scikit-learn is required for the multinomial logit baseline. Please install the project requirements.")

    numeric_features = [
        "current_rate",
        "EFFR_3m_change",
        "two_year_yield",
        "ten_two_spread",
        "CPI_yoy",
        "unemployment_rate",
        "SP500_1m_return",
    ]
    categorical_features = ["lag1_action"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    classifier = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    train_df = feature_table.loc[feature_table["split"] == "train"].copy()
    test_df = feature_table.loc[feature_table["split"] == "test"].copy()
    classifier.fit(train_df[numeric_features + categorical_features], train_df["rate_change"])
    test_df["predicted_label"] = classifier.predict(test_df[numeric_features + categorical_features])
    return test_df


def evaluate_baselines(
    *,
    dataset_root: Path,
    market_baseline_path: Path | None,
    prediction_files: list[Path],
    output_json: Path,
) -> dict:
    feature_table = build_feature_table(dataset_root)
    test_df = feature_table.loc[feature_table["split"] == "test"].copy().reset_index(drop=True)

    majority_label = feature_table.loc[feature_table["split"] == "train", "rate_change"].mode().iloc[0]
    test_df["majority_pred"] = majority_label
    test_df["lag1_pred"] = test_df["lag1_action"]

    logit_df = _fit_multinomial_logit(feature_table)
    market_df = _maybe_load_market_baseline(market_baseline_path)

    results = {
        "reference_dataset": str(dataset_root),
        "test_meetings": test_df["meeting_date"].tolist(),
        "baselines": {
            "majority_class": {
                "exact_match": _exact_match_metrics(test_df, "majority_pred"),
                "coarse_metrics": _coarse_metrics(test_df, "majority_pred"),
            },
            "lag1_action": {
                "exact_match": _exact_match_metrics(test_df, "lag1_pred"),
                "coarse_metrics": _coarse_metrics(test_df, "lag1_pred"),
            },
            "multinomial_logit": {
                "exact_match": _exact_match_metrics(logit_df, "predicted_label"),
                "coarse_metrics": _coarse_metrics(logit_df, "predicted_label"),
                "features": [
                    "current_rate",
                    "lag1_action",
                    "EFFR_3m_change",
                    "two_year_yield",
                    "ten_two_spread",
                    "CPI_yoy",
                    "unemployment_rate",
                    "SP500_1m_return",
                ],
            },
        },
        "model_predictions": {},
    }

    if market_df is not None:
        merged_market = test_df.merge(market_df, on="meeting_date", how="left", validate="one_to_one")
        results["baselines"]["market_implied"] = {
            "coverage": {
                "matched_meetings": int(merged_market["predicted_label"].notna().sum()),
                "total_test_meetings": int(len(merged_market)),
            },
            "exact_match": _exact_match_metrics(merged_market.fillna({"predicted_label": ""}), "predicted_label"),
            "coarse_metrics": _coarse_metrics(merged_market.fillna({"predicted_label": ""}), "predicted_label"),
        }

    for prediction_file in prediction_files:
        label = prediction_file.stem
        model_df = _load_model_predictions(prediction_file)
        merged = test_df.merge(
            model_df[["meeting_date", "generated_vote"]],
            on="meeting_date",
            how="left",
            validate="one_to_one",
        )
        merged["generated_vote"] = merged["generated_vote"].fillna("")
        results["model_predictions"][label] = {
            "source_file": str(prediction_file),
            "exact_match": _exact_match_metrics(merged, "generated_vote"),
            "coarse_metrics": _coarse_metrics(merged, "generated_vote"),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate decision baselines on the canonical held-out meetings.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DECISION_DATASET_ROOT,
    )
    parser.add_argument(
        "--market-baseline",
        type=Path,
        default=MARKET_BASELINE_DEFAULT,
    )
    parser.add_argument(
        "--prediction-file",
        action="append",
        type=Path,
        default=[],
        help="Optional model prediction files (.jsonl or .xlsx). Repeat for multiple models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "output" / "evaluation" / "main" / "decision_baselines.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate_baselines(
        dataset_root=args.dataset_root,
        market_baseline_path=args.market_baseline,
        prediction_files=args.prediction_file,
        output_json=args.output_json,
    )
    print("✅ Decision baseline evaluation finished")
    for label, payload in results["baselines"].items():
        exact = payload["exact_match"]["exact_match"]
        accuracy = payload["coarse_metrics"]["accuracy"]
        print(f"- {label}: exact_match={exact:.4f}, coarse_accuracy={accuracy:.4f}")
    for label, payload in results["model_predictions"].items():
        exact = payload["exact_match"]["exact_match"]
        accuracy = payload["coarse_metrics"]["accuracy"]
        print(f"- {label}: exact_match={exact:.4f}, coarse_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
