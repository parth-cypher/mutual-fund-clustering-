"""
Loading the "Mutual Funds India - Detailed" dataset.

Original source (Kaggle, by ravibarnawal):
    https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed
The project ships a copy in data/. If it is missing, get_dataset_path():
  1. looks for the CSV elsewhere in the project and the VS Code workspace folder above it,
  2. otherwise downloads the public copy from GitHub (pinned version, checked with SHA-256),
  3. otherwise raises DatasetError with manual download steps.

The original CSV is never modified. In the file, missing risk values are written as "-";
they are read as empty (NaN) values in memory.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_NAME = "comprehensive_mutual_funds_data.csv"
DATA_PATH = DATA_DIR / CSV_NAME

DATASET_NAME = "Mutual Funds India - Detailed"
KAGGLE_URL = "https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed"
GITHUB_COPY = "https://github.com/nileshiq/Mutual-Funds-India"
GITHUB_COMMIT = "88cd08b8c451844a2aaad3528ad8600218a5f491"
DOWNLOAD_URL = f"https://raw.githubusercontent.com/nileshiq/Mutual-Funds-India/{GITHUB_COMMIT}/{CSV_NAME}"
EXPECTED_SHA256 = "2672b241c0715e4e099d81771b09da2108f5c2415693c0df2982cfc9ec155b5e"

ID_COLUMNS = ["scheme_name", "amc_name", "category", "sub_category"]
NUMERIC_COLUMNS = [
    "min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr", "fund_age_yr",
    "sortino", "alpha", "sd", "beta", "sharpe", "risk_level", "rating",
    "returns_1yr", "returns_3yr", "returns_5yr",
]

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "scheme_name": "Name of the mutual fund scheme",
    "min_sip": "Minimum monthly SIP amount (₹)",
    "min_lumpsum": "Minimum one-time investment (₹)",
    "expense_ratio": "Yearly fee charged by the fund (% of assets)",
    "fund_size_cr": "Assets managed by the fund (₹ crore)",
    "fund_age_yr": "Age of the fund (years)",
    "fund_manager": "Fund manager",
    "sortino": "Sortino ratio: return per unit of downside volatility",
    "alpha": "Alpha: return above what the fund's market exposure predicts",
    "sd": "Standard deviation of returns (%): how much returns swing (volatility)",
    "beta": "Beta: how strongly the fund moves with its market benchmark",
    "sharpe": "Sharpe ratio: extra return per unit of volatility",
    "risk_level": "Risk level 1-6 (appears to follow SEBI's riskometer: 1 = Low ... 6 = Very High)",
    "amc_name": "Asset management company",
    "rating": "Fund rating (0-5)",
    "category": "Fund category (Equity, Debt, Hybrid, ...)",
    "sub_category": "Fund sub-category (e.g. Small Cap Mutual Funds)",
    "returns_1yr": "Return over the last 1 year (%)",
    "returns_3yr": "Return over the last 3 years (% per year, as given by the source)",
    "returns_5yr": "Return over the last 5 years (% per year, as given by the source)",
}


class DatasetError(Exception):
    """The dataset is missing or unusable. The message explains what to do."""


def manual_download_help() -> str:
    return (
        "How to get the dataset manually:\n"
        f"  Option A (Kaggle, free account): open {KAGGLE_URL},\n"
        f"            click 'Download', unzip it and copy {CSV_NAME} into the data/ folder.\n"
        f"  Option B (no account): open {GITHUB_COPY},\n"
        f"            open {CSV_NAME}, click the download button and put the file in data/.\n"
        "Then run the command again."
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_existing_dataset(search_roots: Optional[List[Path]] = None) -> Optional[Path]:
    """Look for the CSV in data/, then a few folder levels around the project."""
    if DATA_PATH.is_file():
        return DATA_PATH
    roots = search_roots if search_roots is not None else [PROJECT_ROOT, PROJECT_ROOT.parent]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*", "*/*", "*/*/*"):
            for candidate in root.glob(pattern):
                if ".venv" in candidate.parts or "site-packages" in candidate.parts:
                    continue
                if candidate.is_file() and candidate.name == CSV_NAME:
                    return candidate
    return None


def download_dataset(destination: Path = DATA_PATH, timeout: int = 30) -> Path:
    """Download the pinned public copy and check that it is exactly the expected file."""
    print(f"Downloading the dataset:\n  {DOWNLOAD_URL}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=timeout) as response:
            partial.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            hint = ("macOS tip: if Python came from python.org, run 'Install Certificates.command' "
                    "in Applications > Python 3.x and try again.\n")
        raise DatasetError(f"Download failed: {error}\n{hint}\n" + manual_download_help()) from error
    if _sha256(partial) != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise DatasetError("The downloaded file is not the expected dataset (checksum mismatch).\n\n"
                           + manual_download_help())
    partial.replace(destination)
    print(f"Saved data/{destination.name}")
    return destination


def get_dataset_path(allow_download: bool = True) -> Path:
    """Path of the CSV in data/, finding or downloading it first if needed."""
    found = find_existing_dataset()
    if found == DATA_PATH:
        return DATA_PATH
    if found is not None:
        print(f"Found the dataset at {found}; copying it into data/")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found, DATA_PATH)
        return DATA_PATH
    if allow_download:
        return download_dataset()
    raise DatasetError(f"Dataset not found at data/{CSV_NAME}.\n\n" + manual_download_help())


def load_dataset(path: Optional[Path] = None, required_numeric: Optional[List[str]] = None) -> pd.DataFrame:
    """Read the CSV into memory and check it.

    * "-" (used by the source for "not available") becomes NaN.
    * Numeric columns are converted to numbers; any other text in them is an error.
    * A ``fund_id`` column (the row number in the file) is added, because a few
      scheme names appear more than once (different plans of the same scheme).
    """
    path = Path(path) if path is not None else DATA_PATH
    if not path.is_file():
        raise DatasetError(f"Dataset not found at {path}.\n\n" + manual_download_help())
    try:
        frame = pd.read_csv(path, na_values=["-", "", " "])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise DatasetError(f"Could not read {path.name} as a CSV file: {error}") from error

    frame.columns = [str(column).strip() for column in frame.columns]
    required = ["scheme_name", "category"] + list(required_numeric or [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path.name} is missing the column(s) {missing}. "
                           f"Found: {list(frame.columns)}. Is it the right dataset?")
    if frame.empty:
        raise DatasetError(f"{path.name} has no rows.")

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = converted.isna() & frame[column].notna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).unique()[:3].tolist()
            raise DatasetError(f"Column '{column}' has non-numeric values such as {examples}.")
        frame[column] = converted
    for column in ID_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()

    frame.insert(0, "fund_id", range(1, len(frame) + 1))
    return frame


def missing_value_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column (only columns that have any)."""
    counts = frame.isna().sum()
    counts = counts[counts > 0]
    return pd.DataFrame({"Column": counts.index, "Missing": counts.values,
                         "Percent": (100 * counts.values / len(frame)).round(1)})
