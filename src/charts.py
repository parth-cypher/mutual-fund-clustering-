"""Matplotlib charts shared by train.py (saved as PNG files) and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # draw without opening windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from src.clustering import FEATURE_LABELS, PROFILE_COLUMNS, ClusteringResult  # noqa: E402

# Colour-blind-checked categorical palette (fixed order) + a marker per group as a second cue
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID, AXIS = "#1f1f1f", "#6b6a66", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f4f8fe", "#9ec5f4", "#3987e5", "#184f95"])
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
CATEGORY_ORDER = ["Debt", "Hybrid", "Equity", "Solution Oriented", "Other"]


def color_of(cluster: int) -> str:
    return PALETTE[(cluster - 1) % len(PALETTE)]


def marker_of(cluster: int) -> str:
    return MARKERS[(cluster - 1) % len(MARKERS)]


def _style(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor="#3d3c39")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, fontsize=11.5, color=INK, loc="left")


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def feature_histograms(frame: pd.DataFrame, features: List[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.2))
    for ax, feature in zip(np.atleast_1d(axes), features):
        values = frame[feature].dropna()
        upper = values.quantile(0.99)
        ax.hist(values.clip(upper=upper), bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
        ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
        ax.set_ylabel("Funds")
        _style(ax, "y")
    fig.suptitle("Distribution of the clustering features (values above the 99th percentile shown in the last bar)",
                 fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def risk_return_by_category(frame: pd.DataFrame) -> plt.Figure:
    data = frame.dropna(subset=["sd", "returns_3yr"])
    categories = [c for c in CATEGORY_ORDER if c in set(data["category"])] + \
                 sorted(set(data["category"]) - set(CATEGORY_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, category in enumerate(categories):
        part = data[data["category"] == category]
        ax.scatter(part["sd"], part["returns_3yr"], s=26, alpha=0.75, color=PALETTE[i % 8],
                   marker=MARKERS[i % 8], edgecolors="white", linewidths=0.4, label=f"{category} ({len(part)})")
    ax.set_xscale("log")
    ax.set_xlabel("Risk: standard deviation of returns (%, log scale)")
    ax.set_ylabel("3-year return (%)")
    _title(ax, "Risk vs return, coloured by the dataset's fund category")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Clustering results
# ---------------------------------------------------------------------------
def risk_return_clusters(result: ClusteringResult, highlight: Optional[pd.Series] = None) -> plt.Figure:
    """Scatter of risk (x) vs return (y), coloured and shaped by cluster."""
    data = result.data
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for cluster, name in result.names.items():
        part = data[data["cluster"] == cluster]
        ax.scatter(part["sd"], part["returns_3yr"], s=28, alpha=0.8, color=color_of(cluster),
                   marker=marker_of(cluster), edgecolors="white", linewidths=0.4,
                   label=f"{cluster}. {name} ({len(part)})")
    if highlight is not None:
        ax.scatter([highlight["sd"]], [highlight["returns_3yr"]], s=260, facecolors="none", edgecolors=INK,
                   linewidths=2, label="Selected fund", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Risk: standard deviation of returns (%, log scale)")
    ax.set_ylabel("3-year return (%)")
    _title(ax, f"Risk vs return, coloured by cluster ({result.linkage} linkage, k = {result.n_clusters})")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def _leaf_cluster_colors(result: ClusteringResult) -> Dict[int, str]:
    """Colour of every node in the linkage tree: the cluster colour if all its funds share one cluster."""
    n = len(result.data)
    labels = result.data["cluster"].to_numpy()
    node_cluster: Dict[int, Optional[int]] = {i: int(labels[i]) for i in range(n)}
    for i, (left, right, _, _) in enumerate(result.linkage_matrix):
        a, b = node_cluster[int(left)], node_cluster[int(right)]
        node_cluster[n + i] = a if a is not None and a == b else None
    return {node: (color_of(c) if c is not None else "#9a9993") for node, c in node_cluster.items()}


def dendrogram_chart(result: ClusteringResult, last_merges: int = 40) -> plt.Figure:
    colors = _leaf_cluster_colors(result)
    heights = result.linkage_matrix[:, 2]
    k = result.n_clusters
    cut = (heights[-k] + heights[-k + 1]) / 2 if k > 1 else heights[-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(result.linkage_matrix, truncate_mode="lastp", p=last_merges, show_leaf_counts=True,
               leaf_rotation=90, leaf_font_size=8, link_color_func=lambda node: colors[node], ax=ax)
    ax.axhline(cut, color=INK, linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], cut, f"  cut → {k} clusters", va="bottom", ha="right", fontsize=9, color=INK)
    ax.set_ylabel(f"Merge distance ({result.linkage} linkage)")
    ax.set_xlabel("Groups of funds (number of funds in brackets)")
    _title(ax, f"Dendrogram – last {last_merges} merges of {len(result.data)} funds")
    _style(ax, "y")
    handles = [plt.Line2D([0], [0], color=color_of(c), linewidth=3) for c in result.names]
    ax.legend(handles, [f"{c}. {n}" for c, n in result.names.items()], frameon=False, fontsize=8.5,
              loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    return fig


def k_selection_chart(table: pd.DataFrame, chosen_k: int) -> plt.Figure:
    """Silhouette (higher is better) and Davies-Bouldin (lower is better) for each k."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, column, label, better in [(axes[0], "silhouette", "Silhouette score", "higher is better"),
                                      (axes[1], "davies_bouldin", "Davies–Bouldin index", "lower is better")]:
        ax.plot(table["k"], table[column], color=PALETTE[0], linewidth=2, marker="o", markersize=7)
        chosen = table[table["k"] == chosen_k]
        if not chosen.empty:
            ax.scatter(chosen["k"], chosen[column], s=160, facecolors="none", edgecolors=INK, linewidths=2,
                       zorder=5, label=f"chosen k = {chosen_k}")
            ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(table["k"])
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(label)
        _title(ax, f"{label} ({better})")
        _style(ax)
    fig.tight_layout()
    return fig


def cluster_profile_heatmap(result: ClusteringResult) -> plt.Figure:
    """Cluster medians; colour shows how far each is above/below the all-fund median."""
    columns = [c for c in PROFILE_COLUMNS if c in result.data.columns]
    data = result.data
    medians = data.groupby("cluster")[columns].median()
    spread = (data[columns].quantile(0.75) - data[columns].quantile(0.25)).replace(0, np.nan)
    relative = ((medians - data[columns].median()) / spread).clip(-2, 2).fillna(0)
    fig, ax = plt.subplots(figsize=(1.15 * len(columns) + 2.5, 0.65 * len(medians) + 1.6))
    ax.imshow(relative.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    for i in range(len(medians)):
        for j in range(len(columns)):
            ax.text(j, i, f"{medians.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([FEATURE_LABELS.get(c, c).replace(" (", "\n(") for c in columns], fontsize=8.5)
    ax.set_yticks(range(len(medians)))
    ax.set_yticklabels([f"{c}. {result.names[c]}" for c in medians.index], fontsize=9)
    _title(ax, "Cluster profiles: median values (red = above all-fund median, blue = below)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def cluster_vs_category(result: ClusteringResult) -> plt.Figure:
    table = pd.crosstab(result.data["cluster"], result.data["category"])
    ordered = [c for c in CATEGORY_ORDER if c in table.columns] + [c for c in table.columns if c not in CATEGORY_ORDER]
    table = table[ordered]
    fig, ax = plt.subplots(figsize=(1.3 * len(ordered) + 3, 0.6 * len(table) + 1.6))
    ax.imshow(table.to_numpy(), cmap=SEQUENTIAL, aspect="auto")
    peak = table.to_numpy().max()
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
                    color="white" if value > 0.6 * peak else INK)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, fontsize=9)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{c}. {result.names[c]}" for c in table.index], fontsize=9)
    ax.set_xlabel("Fund category in the dataset (not used for clustering)")
    _title(ax, "How the clusters line up with fund categories (number of funds)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def save_all(frame: pd.DataFrame, result: ClusteringResult, k_table: pd.DataFrame, folder: Path) -> List[Path]:
    """Save every chart as a PNG and return the paths."""
    folder.mkdir(parents=True, exist_ok=True)
    charts = {
        "feature_distributions.png": feature_histograms(frame, result.features),
        "risk_return_by_category.png": risk_return_by_category(frame),
        "dendrogram.png": dendrogram_chart(result),
        "k_selection.png": k_selection_chart(k_table, result.n_clusters),
        "risk_return_clusters.png": risk_return_clusters(result),
        "cluster_profiles.png": cluster_profile_heatmap(result),
        "cluster_vs_category.png": cluster_vs_category(result),
    }
    paths = []
    for name, fig in charts.items():
        path = folder / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
