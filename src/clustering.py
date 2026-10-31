"""
Hierarchical (agglomerative) clustering of mutual funds by risk and return.

Steps
-----
1. Keep the funds that have a value for every chosen feature.
2. Cap extreme values at the 1st and 99th percentiles ("winsorizing"). Without
   this, a single extreme fund forms a cluster of its own.
3. Standardise every feature (mean 0, standard deviation 1) so that no feature
   dominates just because of its units.
4. Agglomerative clustering: start with every fund as its own cluster and
   repeatedly merge the two closest clusters. Ward linkage merges the pair that
   increases the within-cluster variance the least.
5. Number the clusters 1..k from lowest to highest risk (median standard
   deviation) and give each a readable name such as "Low risk · Low return".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import cophenet, linkage as scipy_linkage
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_FEATURES = ["returns_3yr", "sd", "sharpe"]
CANDIDATE_FEATURES = ["returns_1yr", "returns_3yr", "returns_5yr", "sd", "beta", "sharpe", "sortino", "alpha"]
FEATURE_LABELS = {
    "returns_1yr": "1-year return (%)",
    "returns_3yr": "3-year return (%)",
    "returns_5yr": "5-year return (%)",
    "sd": "Standard deviation (%)",
    "beta": "Beta",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "alpha": "Alpha",
    "expense_ratio": "Expense ratio (%)",
    "risk_level": "Risk level (1-6)",
}
DEFAULT_CLUSTERS = 4
DEFAULT_LINKAGE = "ward"
LINKAGE_METHODS = ["ward", "complete", "average", "single"]
CLIP_QUANTILES = (0.01, 0.99)
K_RANGE = range(2, 9)

RISK_COLUMN = "sd"            # used to order and name clusters
RETURN_COLUMN = "returns_3yr"  # used to name clusters
LEVEL_WORDS = ["Very low", "Low", "Moderate", "High", "Very high"]
PROFILE_COLUMNS = ["returns_1yr", "returns_3yr", "returns_5yr", "sd", "beta", "sharpe", "sortino", "alpha",
                   "expense_ratio"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap each column at quantiles learned from the data it is fitted on."""

    def __init__(self, lower: float = CLIP_QUANTILES[0], upper: float = CLIP_QUANTILES[1]):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


def build_preprocessor() -> Pipeline:
    return Pipeline([("clip", Winsorizer()), ("scale", StandardScaler())])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ClusteringResult:
    features: List[str]
    n_clusters: int
    linkage: str
    data: pd.DataFrame                 # funds used, with 'cluster' (1..k) and 'cluster_name'
    scaled: np.ndarray                 # preprocessed feature matrix (same row order as data)
    preprocessor: Pipeline
    linkage_matrix: np.ndarray         # SciPy linkage matrix, used to draw the dendrogram
    centroids: np.ndarray              # mean of each cluster in the preprocessed space (row c-1 = cluster c)
    names: Dict[int, str]
    summary: pd.DataFrame
    metrics: Dict[str, float]
    rows_total: int
    rows_dropped: int
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _validate(frame: pd.DataFrame, features: Sequence[str], n_clusters: int, linkage: str) -> None:
    if len(features) < 2:
        raise ValueError("Choose at least two features.")
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(f"Unknown feature(s): {missing}")
    if linkage not in LINKAGE_METHODS:
        raise ValueError(f"linkage must be one of {LINKAGE_METHODS}")
    if n_clusters < 2:
        raise ValueError("Use at least 2 clusters.")


def cluster_labels(scaled: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    """scikit-learn agglomerative clustering (raw labels 0..k-1)."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(scaled)


def _order_by_risk(data: pd.DataFrame, raw: np.ndarray, scaled: np.ndarray) -> np.ndarray:
    """Renumber clusters 1..k from the lowest to the highest median standard deviation."""
    keys = []
    for label in np.unique(raw):
        members = raw == label
        if RISK_COLUMN in data.columns and data.loc[members, RISK_COLUMN].notna().any():
            keys.append((float(data.loc[members, RISK_COLUMN].median()), label))
        else:
            keys.append((float(scaled[members, 0].mean()), label))
    mapping = {old: new for new, (_, old) in enumerate(sorted(keys), start=1)}
    return np.array([mapping[label] for label in raw])


def _level(value: float, reference: pd.Series) -> str:
    """Describe a value with one of five words based on quintiles of all funds."""
    if pd.isna(value) or reference.dropna().empty:
        return "Unknown"
    edges = reference.dropna().quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    return LEVEL_WORDS[int(np.searchsorted(edges, value, side="right"))]


def name_clusters(data: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """Readable names like 'Low risk · Moderate return' (unique within one result)."""
    names: Dict[int, str] = {}
    for cluster in sorted(np.unique(labels)):
        members = data[labels == cluster]
        risk = _level(members[RISK_COLUMN].median(), data[RISK_COLUMN]) if RISK_COLUMN in data else "Unknown"
        ret = _level(members[RETURN_COLUMN].median(), data[RETURN_COLUMN]) if RETURN_COLUMN in data else "Unknown"
        names[int(cluster)] = f"{risk} risk · {ret.lower()} return"
    # Make duplicates unique by adding the median Sharpe ratio
    for base in {n for n in names.values() if list(names.values()).count(n) > 1}:
        for cluster in [c for c, n in names.items() if n == base]:
            sharpe = data.loc[labels == cluster, "sharpe"].median() if "sharpe" in data else np.nan
            suffix = f"Sharpe {sharpe:.2f}" if pd.notna(sharpe) else f"group {cluster}"
            names[cluster] = f"{base} ({suffix})"
    return names


def cluster_summary(data: pd.DataFrame, labels: np.ndarray, names: Mapping[int, str]) -> pd.DataFrame:
    """One row per cluster with its size and median risk/return figures."""
    rows = []
    for cluster in sorted(names):
        members = data[labels == cluster]
        row: Dict[str, object] = {"Cluster": cluster, "Name": names[cluster], "Funds": int(len(members))}
        for column in PROFILE_COLUMNS:
            if column in members:
                row[FEATURE_LABELS.get(column, column)] = round(float(members[column].median()), 2)
        if "risk_level" in members:
            row["Median risk level"] = float(members["risk_level"].median())
        if "category" in members and len(members):
            top = members["category"].value_counts()
            row["Main category"] = f"{top.index[0]} ({100 * top.iloc[0] / len(members):.0f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_clusters(scaled: np.ndarray, labels: np.ndarray, data: pd.DataFrame,
                      linkage_matrix: np.ndarray) -> Dict[str, float]:
    """Internal quality scores plus agreement with the dataset's own labels."""
    metrics: Dict[str, float] = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
        "cophenetic_correlation": float(cophenet(linkage_matrix, pdist(scaled))[0]),
    }
    if "category" in data:
        metrics["ari_vs_category"] = float(adjusted_rand_score(data["category"], labels))
    if "risk_level" in data and data["risk_level"].notna().all():
        metrics["ari_vs_risk_level"] = float(adjusted_rand_score(data["risk_level"], labels))
    return metrics


def run_clustering(frame: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
                   n_clusters: int = DEFAULT_CLUSTERS, linkage: str = DEFAULT_LINKAGE) -> ClusteringResult:
    """Cluster the funds and return everything needed for reports and the app."""
    features = list(features)
    _validate(frame, features, n_clusters, linkage)
    data = frame.dropna(subset=features).reset_index(drop=True)
    if len(data) <= n_clusters:
        raise ValueError(f"Only {len(data)} funds have all the chosen features; need more than {n_clusters}.")

    preprocessor = build_preprocessor()
    scaled = preprocessor.fit_transform(data[features])
    labels = _order_by_risk(data, cluster_labels(scaled, n_clusters, linkage), scaled)
    names = name_clusters(data, labels)

    data = data.copy()
    data["cluster"] = labels
    data["cluster_name"] = data["cluster"].map(names)
    centroids = np.vstack([scaled[labels == c].mean(axis=0) for c in sorted(names)])
    linkage_matrix = scipy_linkage(scaled, method=linkage)

    return ClusteringResult(
        features=features,
        n_clusters=n_clusters,
        linkage=linkage,
        data=data,
        scaled=scaled,
        preprocessor=preprocessor,
        linkage_matrix=linkage_matrix,
        centroids=centroids,
        names=names,
        summary=cluster_summary(data, labels, names),
        metrics=evaluate_clusters(scaled, labels, data, linkage_matrix),
        rows_total=int(len(frame)),
        rows_dropped=int(len(frame) - len(data)),
    )


def k_selection_table(frame: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
                      linkage: str = DEFAULT_LINKAGE, k_values=K_RANGE) -> pd.DataFrame:
    """Quality scores for several numbers of clusters, to help choose k."""
    data = frame.dropna(subset=list(features))
    scaled = build_preprocessor().fit_transform(data[list(features)])
    rows = []
    for k in k_values:
        if k >= len(data):
            break
        labels = cluster_labels(scaled, k, linkage)
        sizes = np.bincount(labels)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(scaled, labels),
            "davies_bouldin": davies_bouldin_score(scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(scaled, labels),
            "smallest_cluster": int(sizes.min()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Using the clusters
# ---------------------------------------------------------------------------
def _to_vector(values: Mapping[str, object], features: Sequence[str]) -> np.ndarray:
    vector = []
    for feature in features:
        if feature not in values:
            raise ValueError(f"Missing value for {FEATURE_LABELS.get(feature, feature)}.")
        try:
            number = float(values[feature])
        except (TypeError, ValueError):
            raise ValueError(f"{FEATURE_LABELS.get(feature, feature)} must be a number.") from None
        if not np.isfinite(number):
            raise ValueError(f"{FEATURE_LABELS.get(feature, feature)} must be a finite number.")
        vector.append(number)
    return np.array([vector])


def assign_cluster(preprocessor: Pipeline, centroids: np.ndarray, features: Sequence[str],
                   values: Mapping[str, object]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Place a fund (new or existing) in the cluster whose centre is closest.

    Hierarchical clustering itself cannot label new points, so the nearest
    cluster centre (in the same scaled space) is used. Returns
    (cluster number, distance to each cluster centre, scaled point).
    """
    point = preprocessor.transform(_to_vector(values, features))
    distances = np.linalg.norm(centroids - point, axis=1)
    return int(np.argmin(distances)) + 1, distances, point[0]


def similar_funds(result: ClusteringResult, point: np.ndarray, n: int = 5,
                  exclude_fund_id: Optional[int] = None) -> pd.DataFrame:
    """The n funds closest to a point in the scaled risk/return space."""
    distances = np.linalg.norm(result.scaled - point, axis=1)
    table = result.data.assign(distance=distances)
    if exclude_fund_id is not None:
        table = table[table["fund_id"] != exclude_fund_id]
    return table.nsmallest(n, "distance")


def fund_label(row: pd.Series) -> str:
    """Unique, readable label for a fund (some scheme names repeat)."""
    return f"{row['scheme_name']} — {row.get('sub_category', row.get('category', ''))} (#{int(row['fund_id'])})"


def model_bundle(result: ClusteringResult) -> Dict[str, object]:
    """What train.py saves with joblib to assign new funds later without re-clustering."""
    return {
        "features": result.features,
        "n_clusters": result.n_clusters,
        "linkage": result.linkage,
        "preprocessor": result.preprocessor,
        "centroids": result.centroids,
        "names": result.names,
        "summary": result.summary,
        "metrics": result.metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "scikit_learn_version": sklearn.__version__,
    }
