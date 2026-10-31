"""
Group mutual funds by risk and return with hierarchical clustering.

Run from the project folder:
    python train.py                              # default: 4 clusters, Ward linkage
    python train.py --clusters 5 --linkage complete
    python train.py --features returns_1yr returns_3yr sd sharpe
    python train.py --help

Prints a summary and saves:
    models/fund_clusters.joblib     preprocessing + cluster centres (to place new funds)
    reports/fund_clusters.csv       every fund with its cluster
    reports/cluster_summary.csv     one row per cluster (median figures)
    reports/k_selection.csv         quality scores for k = 2..8
    reports/cluster_metrics.txt     readable summary of the results
    reports/*.png                   charts (dendrogram, scatter plots, profiles, ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src import charts
from src.clustering import (
    CANDIDATE_FEATURES, DEFAULT_CLUSTERS, DEFAULT_FEATURES, DEFAULT_LINKAGE, FEATURE_LABELS, LINKAGE_METHODS,
    k_selection_table, model_bundle, run_clustering,
)
from src.data import DATASET_NAME, KAGGLE_URL, PROJECT_ROOT, DatasetError, get_dataset_path, load_dataset

MODEL_PATH = PROJECT_ROOT / "models" / "fund_clusters.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

METRIC_TEXT = {
    "silhouette": "Silhouette score (-1 to 1, higher = better separated)",
    "davies_bouldin": "Davies–Bouldin index (lower = better)",
    "calinski_harabasz": "Calinski–Harabasz score (higher = better)",
    "cophenetic_correlation": "Cophenetic correlation (how well the tree keeps the real distances, 0 to 1)",
    "ari_vs_category": "Adjusted Rand index vs fund category (agreement, 0 = random, 1 = identical)",
    "ari_vs_risk_level": "Adjusted Rand index vs risk level (agreement, 0 = random, 1 = identical)",
}


def write_report(result, k_table: pd.DataFrame, path: Path) -> None:
    lines = [
        "Grouping Mutual Funds by Risk and Return Using Hierarchical Clustering",
        "=" * 70,
        f"Dataset           : {DATASET_NAME} ({KAGGLE_URL})",
        f"Funds in file     : {result.rows_total}",
        f"Funds clustered   : {len(result.data)} ({result.rows_dropped} skipped: a chosen feature was missing)",
        f"Features          : {', '.join(FEATURE_LABELS.get(f, f) for f in result.features)}",
        "Preprocessing     : values capped at the 1st/99th percentiles, then standardised",
        f"Method            : agglomerative clustering, {result.linkage} linkage, k = {result.n_clusters}",
        "",
        "Cluster quality:",
        *[f"  {METRIC_TEXT.get(k, k)}: {v:.3f}" for k, v in result.metrics.items()],
        "",
        "Clusters (median values):",
        result.summary.to_string(index=False),
        "",
        "Choosing the number of clusters:",
        k_table.round(3).to_string(index=False),
        "",
        "Clusters vs fund category (number of funds):",
        pd.crosstab(result.data["cluster"], result.data["category"]).to_string(),
        "",
        "This is an educational analysis of past data, not investment advice.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hierarchical clustering of mutual funds by risk and return.")
    parser.add_argument("--clusters", "-k", type=int, default=DEFAULT_CLUSTERS, help="Number of clusters (2-8)")
    parser.add_argument("--linkage", choices=LINKAGE_METHODS, default=DEFAULT_LINKAGE)
    parser.add_argument("--features", nargs="+", choices=CANDIDATE_FEATURES, default=DEFAULT_FEATURES)
    parser.add_argument("--no-download", action="store_true", help="Never try to download the dataset")
    parser.add_argument("--models-dir", type=Path, default=MODEL_PATH.parent)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)
    if not 2 <= args.clusters <= 8:
        parser.error("--clusters must be between 2 and 8")

    try:
        frame = load_dataset(get_dataset_path(allow_download=not args.no_download), required_numeric=args.features)
    except DatasetError as error:
        print(f"\nDATASET PROBLEM\n{error}", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(frame)} funds, {frame.shape[1] - 1} columns from {DATASET_NAME}")
    missing = frame[args.features].isna().sum()
    print("Missing values in the chosen features: " + ", ".join(f"{c}={int(n)}" for c, n in missing.items()))

    try:
        result = run_clustering(frame, args.features, args.clusters, args.linkage)
    except ValueError as error:
        print(f"\nSETTINGS PROBLEM: {error}", file=sys.stderr)
        return 2
    k_table = k_selection_table(frame, args.features, args.linkage)

    print(f"Clustered {len(result.data)} funds ({result.rows_dropped} skipped because a chosen feature was missing)")
    print(f"Method: {args.linkage} linkage, {args.clusters} clusters, features: {', '.join(args.features)}\n")
    columns = ["Cluster", "Name", "Funds"] + [FEATURE_LABELS[f] for f in args.features if f in FEATURE_LABELS]
    print(result.summary[[c for c in columns if c in result.summary.columns] + ["Main category"]].to_string(index=False))
    print("\nCluster quality:")
    for key, value in result.metrics.items():
        print(f"  {METRIC_TEXT.get(key, key)}: {value:.3f}")
    print("\nScores for other numbers of clusters:")
    print(k_table.round(3).to_string(index=False))

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle(result), args.models_dir / MODEL_PATH.name)
    output_columns = ["fund_id", "scheme_name", "amc_name", "category", "sub_category", "risk_level",
                      *result.features, "cluster", "cluster_name"]
    output_columns = list(dict.fromkeys(c for c in output_columns if c in result.data.columns))
    result.data[output_columns].to_csv(args.reports_dir / "fund_clusters.csv", index=False)
    result.summary.to_csv(args.reports_dir / "cluster_summary.csv", index=False)
    k_table.to_csv(args.reports_dir / "k_selection.csv", index=False)
    write_report(result, k_table, args.reports_dir / "cluster_metrics.txt")
    saved = charts.save_all(frame, result, k_table, args.reports_dir)

    print(f"\nSaved {args.models_dir.name}/{MODEL_PATH.name}, 4 CSV/TXT reports and {len(saved)} charts in "
          f"{args.reports_dir.name}/")
    print("Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
