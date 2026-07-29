from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INDICATOR_FILE = BASE_DIR / "input_data" / "indicators.xlsx"


def _load_indicator_lookup() -> dict[str, str]:
    indicator_df = pd.read_excel(INDICATOR_FILE)
    return dict(zip(indicator_df["indicators"], indicator_df["sections"]))


LABEL_TO_SECTION = _load_indicator_lookup()
NORMALIZED_INDICATOR_NAMES = {re.sub(r"\s+", "", key): key for key in LABEL_TO_SECTION}


RELABEL_MAP = {
    "Capital Adequacy Ratio": "Bank Capital",
    "Crude Oil Prices": "Commodity Prices",
    "PCE Price Index": "Personal Consumption Expenditures (PCE)",
    "PCE Price Index [LABEL]Consumer Price Index (CPI)": "Personal Consumption Expenditures (PCE), Consumer Price Index (CPI)",
    "Total Assets of Federal Reserve": "Federal Reserve Balance Sheet",
    "Total Liabilities of Federal Reserve": "Federal Reserve Balance Sheet",
    "Reserve Balances with Federal": "Federal Reserve Balance Sheet",
    "Reserve Balances with Fed": "Federal Reserve Balance Sheet",
    "Nonfarm Payrolls": "Labour Market",
    "Job Openings (JOLTS)": "Labour Market",
    "Labor Force Participation Rate": "Labour Market",
    "Holdings of Mortgage-Backed Securities (MBS)": "Federal Reserve Balance Sheet",
    "Mortgage-Backed Securities": "Federal Reserve Balance Sheet",
    "Non-Performing Loans (NPLs)": "Bank Credit to Private Sector",
    "Mortgage Applications": "Bank Credit to Private Sector",
    "Consumer Credit": "Bank Credit to Private Sector",
    "Consumer Spending": "Personal Consumption Expenditures (PCE)",
    "Money Supply (M1,M2)": "Money Supply",
    "Money Supply (M1, M2)": "Money Supply",
    "Money Supply (M2)": "Money Supply",
    "UnemploymentRate": "Unemployment Rate",
    "BusinessInvestment": "Business Investment",
    "ConsumerPriceIndex(CPI)": "Consumer Price Index (CPI)",
    "FederalFundsRate": "Federal Funds Rate",
    "BankCredittoPrivateSection": "Bank Credit to Private Sector",
    "BankCreditToPrivateSector": "Bank Credit to Private Sector",
    "BankCredittoprivateSector": "Bank Credit to Private Sector",
    "BankCredittothePrivateSector": "Bank Credit to Private Sector",
    "MoneySupply": "Money Supply",
    "InternationalEquity Markets": "Equity Market Indices",
    "CPI": "Consumer Price Index (CPI)",
    "Government Spending": "Government Purchases",
    "Current Account Balance": "Trade Balance",
    "MarketVolatility": "Market Volatility (VIX)",
    "PersonalConsumptionExpenditures": "Personal Consumption Expenditures (PCE)",
}


def remove_label_bracket(text: str) -> str:
    match = re.search(r"\s*(.*?)\[/LABEL\]", text)
    return match.group(1).strip() if match else text.strip()


def remove_quotes(text: str) -> str:
    return text.replace('"', "").replace("'", "")


def remove_multi_label_items(text: str) -> str:
    matches = re.findall(r"\[LABEL\](.+)", text)
    cleaned = [match.strip() for match in matches if match.strip()]
    if not cleaned:
        return text
    return ", ".join(cleaned)


def remove_brackets(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")


def remove_label_and_quotes(text: str) -> str:
    match = re.search(r'labeled_file\s*:\s*([^"]+)', text, flags=re.IGNORECASE)
    return match.group(1) if match else text


def replace_old_label(text: str) -> str:
    for old_label, new_label in NORMALIZED_INDICATOR_NAMES.items():
        if old_label in text:
            text = text.replace(old_label, new_label)
    for old_label, new_label in RELABEL_MAP.items():
        if old_label in text:
            text = text.replace(old_label, new_label)
    return text


def process_label(text: str) -> str:
    text = remove_label_and_quotes(text)
    text = remove_brackets(text)
    text = remove_multi_label_items(text)
    text = remove_quotes(text)
    text = remove_label_bracket(text)
    return replace_old_label(text)


def extract_date_from_filename(filename: str) -> date | None:
    match = re.search(r"\d{8}", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d").date()
    except ValueError:
        return None


def extract_labels_from_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"Input must be a string: {text!r}")

    pattern = r"\[LABEL\](.*?)\[/LABEL\]|\[LABEL\]\{(.*?)\}|\[LABEL\]\s*(.*?)(?=\[|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    labels = [group for match in matches for group in match if group.strip()]
    labels = [process_label(label.strip()) for label in labels if label.strip()]
    labels = list(dict.fromkeys(labels))

    if not labels:
        return "Not-Found"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels)


def process_len(text: str) -> bool:
    return len(text.replace(" ", "")) <= 200
