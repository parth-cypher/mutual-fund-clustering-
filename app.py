"""
Streamlit dashboard: Grouping Mutual Funds by Risk and Return Using Hierarchical Clustering.

Run from the project folder:
    streamlit run app.py

The app clusters the funds itself (it takes well under a second), so you can
change the number of clusters, the linkage method and the features in the
sidebar and see the result immediately. train.py saves the same results as
files for your report.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import charts
from src.clustering import (
    CANDIDATE_FEATURES, DEFAULT_CLUSTERS, DEFAULT_FEATURES, DEFAULT_LINKAGE, FEATURE_LABELS, LINKAGE_METHODS,
    assign_cluster, fund_label, k_selection_table, run_clustering, similar_funds,
)
from src.data import (
    COLUMN_DESCRIPTIONS, CSV_NAME, DATASET_NAME, GITHUB_COPY, KAGGLE_URL, DatasetError, get_dataset_path,
    load_dataset, missing_value_table,
)

st.set_page_config(page_title="Mutual Fund Clustering", page_icon="📊", layout="wide")

PAGES = ["Overview", "Data Exploration", "Clusters", "Explore Funds", "Find a Fund's Group", "About"]
DISCLAIMER = ("Educational project based on past data. Past returns do not guarantee future returns. "
              "This is not investment advice.")
TABLE_COLUMNS = ["fund_id", "scheme_name", "amc_name", "category", "sub_category", "returns_1yr", "returns_3yr",
                 "returns_5yr", "sd", "beta", "sharpe", "risk_level", "expense_ratio"]


# ---------------------------------------------------------------------------
# Cached data and clustering
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str, modified: float) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource
def cluster(path: str, modified: float, features: tuple, n_clusters: int, linkage: str):
    return run_clustering(load_data(path, modified), list(features), n_clusters, linkage)


@st.cache_data
def k_table(path: str, modified: float, features: tuple, linkage: str) -> pd.DataFrame:
    return k_selection_table(load_data(path, modified), list(features), linkage)


def keep_valid(key: str, options: list) -> None:
    """If sidebar settings changed the choices, reset a stored selection that no longer exists."""
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = options[0]


def show(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


def nice_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: FEATURE_LABELS.get(c, c) for c in frame.columns}
    renamed.update({"fund_id": "ID", "scheme_name": "Fund", "amc_name": "Fund house", "category": "Category",
                    "sub_category": "Sub-category", "cluster": "Cluster", "cluster_name": "Cluster name",
                    "distance": "Distance"})
    return frame.rename(columns=renamed)


def dataset_missing_page() -> None:
    st.title("📊 Grouping Mutual Funds by Risk and Return")
    st.warning(f"The dataset was not found. It should be at data/{CSV_NAME}.")
    st.markdown(
        f"""
1. Open the Kaggle dataset **{DATASET_NAME}**: [{KAGGLE_URL}]({KAGGLE_URL}) (free account), click **Download** and
   unzip it – or download `{CSV_NAME}` from the public copy at [{GITHUB_COPY}]({GITHUB_COPY}).
2. Put `{CSV_NAME}` in this project's `data` folder (or run `python train.py`, which downloads it).
3. Refresh this page.
"""
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(frame: pd.DataFrame, result) -> None:
    st.title("📊 Grouping Mutual Funds by Risk and Return")
    st.subheader("Using Hierarchical Clustering")
    st.markdown(
        "**Objective:** group Indian mutual funds that behave alike in terms of **risk** (how much their returns "
        "swing) and **return**, without using their official category, and check whether the groups make sense."
    )
    st.markdown(f"**Dataset:** [{DATASET_NAME}]({KAGGLE_URL}) (Kaggle) – {len(frame)} funds, 20 columns.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Funds in dataset", len(frame))
    c2.metric("Funds clustered", len(result.data))
    c3.metric("Fund houses", frame["amc_name"].nunique())
    c4.metric("Clusters", result.n_clusters)
    st.caption(f"{result.rows_dropped} funds were skipped because a chosen feature was missing. "
               f"Features: {', '.join(FEATURE_LABELS.get(f, f) for f in result.features)} · "
               f"{result.linkage} linkage (change these in the sidebar).")

    st.subheader("The groups")
    show(charts.risk_return_clusters(result))
    st.dataframe(result.summary, hide_index=True)
    st.caption("Values are medians of the funds in each cluster. Clusters are numbered from lowest to highest risk.")
    st.info(DISCLAIMER)


def page_data(frame: pd.DataFrame, result) -> None:
    st.title("🔍 Data Exploration")
    st.write(f"{len(frame)} funds and {frame.shape[1] - 1} columns. Each row is one mutual fund scheme.")
    tab_preview, tab_columns, tab_missing, tab_charts = st.tabs(
        ["Preview", "Column meanings", "Missing values", "Charts"])
    with tab_preview:
        st.dataframe(frame.head(50), hide_index=True)
        with st.expander("Summary statistics of the numeric columns"):
            st.dataframe(frame.drop(columns=["fund_id"]).describe().T.round(2))
    with tab_columns:
        st.dataframe(pd.DataFrame({"Column": list(COLUMN_DESCRIPTIONS), "Meaning": list(COLUMN_DESCRIPTIONS.values())}),
                     hide_index=True)
    with tab_missing:
        st.write("In the original file, values that are not available are written as `-`. They are read as "
                 "missing values (the file itself is not changed).")
        table = missing_value_table(frame)
        if table.empty:
            st.success("No missing values.")
        else:
            st.dataframe(table, hide_index=True)
        st.caption("Funds missing any of the chosen clustering features are left out of the clustering.")
    with tab_charts:
        show(charts.feature_histograms(frame, result.features))
        show(charts.risk_return_by_category(frame))
        st.caption("The x-axis uses a log scale so that the many low-risk debt funds are not squashed together.")


def page_clusters(frame: pd.DataFrame, result, scores: pd.DataFrame) -> None:
    st.title("🌳 Clusters")
    m = result.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette score", f"{m['silhouette']:.3f}", help="-1 to 1; higher means better separated groups")
    c2.metric("Davies–Bouldin index", f"{m['davies_bouldin']:.3f}", help="Lower is better")
    c3.metric("Calinski–Harabasz score", f"{m['calinski_harabasz']:.0f}", help="Higher is better")
    c4.metric("Cophenetic correlation", f"{m['cophenetic_correlation']:.3f}",
              help="How well the tree keeps the real distances between funds (0 to 1)")

    tabs = st.tabs(["Dendrogram", "Risk vs return", "Cluster profiles", "Clusters vs category", "Choosing k"])
    with tabs[0]:
        show(charts.dendrogram_chart(result))
        st.markdown("Each fund starts as its own group. Going **up**, the two closest groups are merged; the height "
                    "of a merge shows how different they were. Cutting the tree at the dashed line gives "
                    f"**{result.n_clusters} clusters**. Only the last 40 merges are drawn; numbers in brackets are "
                    "how many funds sit below each branch.")
    with tabs[1]:
        show(charts.risk_return_clusters(result))
    with tabs[2]:
        show(charts.cluster_profile_heatmap(result))
        st.dataframe(result.summary, hide_index=True)
    with tabs[3]:
        show(charts.cluster_vs_category(result))
        line = f"Adjusted Rand index vs category: **{m['ari_vs_category']:.3f}**"
        if "ari_vs_risk_level" in m:
            line += f" · vs risk level: **{m['ari_vs_risk_level']:.3f}**"
        st.markdown(line + " (0 = no more agreement than chance, 1 = identical grouping).")
        st.caption("The category was NOT used for clustering, so this is an independent check: debt and equity "
                   "funds should mostly land in different clusters.")
    with tabs[4]:
        show(charts.k_selection_chart(scores, result.n_clusters))
        st.dataframe(scores.round(3), hide_index=True)
        best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
        st.markdown(
            f"The highest silhouette score is at **k = {best}**. Fewer clusters are usually easier to separate, so "
            "k is also chosen by how useful the groups are: with more clusters, funds with similar risk but "
            "different returns (or Sharpe ratios) get their own group. Use the slider in the sidebar to compare.")


def page_explore(result) -> None:
    st.title("📋 Explore Funds")
    options = ["All clusters"] + [f"{c}. {n}" for c, n in result.names.items()]
    keep_valid("explore_cluster", options)
    choice = st.selectbox("Cluster", options, key="explore_cluster")
    search = st.text_input("Search by fund or fund house name", key="explore_search")
    table = result.data
    if choice != "All clusters":
        table = table[table["cluster"] == int(choice.split(".")[0])]
    if search.strip():
        text = search.strip().lower()
        table = table[table["scheme_name"].str.lower().str.contains(text, regex=False)
                      | table["amc_name"].str.lower().str.contains(text, regex=False)]
    columns = [c for c in TABLE_COLUMNS if c in table.columns] + ["cluster", "cluster_name"]
    st.write(f"{len(table)} funds")
    st.dataframe(nice_columns(table[columns].sort_values("returns_3yr", ascending=False)), hide_index=True)
    st.download_button("Download this table (CSV)", table[columns].to_csv(index=False).encode("utf-8"),
                       file_name="fund_clusters.csv", mime="text/csv")


def _show_assignment(result, cluster_id: int, distances, point, exclude_fund_id=None) -> None:
    st.success(f"Group: **{cluster_id}. {result.names[cluster_id]}**")
    st.dataframe(pd.DataFrame({"Cluster": [f"{c}. {n}" for c, n in result.names.items()],
                               "Distance to cluster centre": distances.round(3)}), hide_index=True)
    st.markdown("**Most similar funds** (closest in the scaled risk/return space)")
    nearest = similar_funds(result, point, n=5, exclude_fund_id=exclude_fund_id)
    columns = ["scheme_name", "category", *result.features, "cluster_name", "distance"]
    st.dataframe(nice_columns(nearest[list(dict.fromkeys(columns))].round(3)), hide_index=True)


def page_find(result) -> None:
    st.title("🎯 Find a Fund's Group")
    st.write("Pick a fund from the dataset, or type in the numbers for any fund, to see which group it belongs to.")
    mode = st.radio("Input", ["Pick a fund from the dataset", "Enter your own numbers"], horizontal=True,
                    key="find_mode")
    if mode == "Pick a fund from the dataset":
        labels = result.data.apply(fund_label, axis=1)
        options = sorted(labels)
        keep_valid("find_fund", options)
        chosen = st.selectbox("Fund", options, key="find_fund")
        fund = result.data[labels == chosen].iloc[0]
        figures = {f: fund[f] for f in result.features}
        st.write(", ".join(f"{FEATURE_LABELS.get(f, f)}: **{v:.2f}**" for f, v in figures.items()))
        cluster_id, distances, point = assign_cluster(result.preprocessor, result.centroids, result.features, figures)
        if cluster_id != int(fund["cluster"]):
            st.caption(f"Note: the hierarchical clustering put this fund in cluster {int(fund['cluster'])}; "
                       f"its nearest cluster centre is {cluster_id}.")
            cluster_id = int(fund["cluster"])
        _show_assignment(result, cluster_id, distances, point, exclude_fund_id=int(fund["fund_id"]))
        show(charts.risk_return_clusters(result, highlight=fund))
    else:
        medians = result.data[result.features].median()
        with st.form("own_numbers"):
            columns = st.columns(len(result.features))
            values = {}
            for column, feature in zip(columns, result.features):
                values[feature] = column.number_input(FEATURE_LABELS.get(feature, feature),
                                                      value=round(float(medians[feature]), 2), step=0.1,
                                                      format="%.2f", key=f"own_{feature}")
            submitted = st.form_submit_button("Find group", type="primary")
        if submitted:
            try:
                cluster_id, distances, point = assign_cluster(result.preprocessor, result.centroids,
                                                              result.features, values)
            except ValueError as error:
                st.error(str(error))
                return
            _show_assignment(result, cluster_id, distances, point)
        st.caption("Hierarchical clustering cannot place new funds by itself, so a new fund is put in the group "
                   "whose centre is closest, after the same capping and scaling used for clustering.")
    st.info(DISCLAIMER)


def page_about() -> None:
    st.title("ℹ️ About the Project")
    st.markdown(
        f"""
**Problem.** India has hundreds of mutual fund schemes. Their official categories (equity, debt, hybrid …) say what
they invest in, but not how they actually behaved. This project groups funds by their **measured risk and return**.

**Dataset.** [{DATASET_NAME}]({KAGGLE_URL}) on Kaggle (web-scraped data on 814 Indian mutual funds, 20 columns:
returns over 1, 3 and 5 years, standard deviation, beta, Sharpe, Sortino, alpha, expense ratio, fund size, category …).

**Workflow**
1. Load the CSV; treat `-` as missing; check the columns.
2. Choose the features (default: 3-year return, standard deviation, Sharpe ratio) and skip funds missing any of them.
3. Cap extreme values at the 1st/99th percentiles, then standardise every feature.
4. **Agglomerative hierarchical clustering**: every fund starts alone; the two closest groups are merged again and
   again until one group remains. The record of merges is the **dendrogram**; cutting it gives *k* clusters.
5. Evaluate with the silhouette score, Davies–Bouldin index, Calinski–Harabasz score and cophenetic correlation,
   and compare the clusters with the funds' real categories (adjusted Rand index).

**Linkage methods** decide what "closest groups" means:
- **Ward** (default): merge the pair that increases the within-group variance the least → compact, similar-sized groups.
- **Complete**: distance between the two farthest funds of the groups.
- **Average**: average distance between all pairs of funds.
- **Single**: distance between the two closest funds – tends to form one long chain (try it in the sidebar).

**Limitations**
- The data is a one-time web scrape; returns and risk figures change every day.
- The time window used for standard deviation, Sharpe and the other ratios is not stated in the dataset.
- A few funds have unusual values (e.g. very high returns with very low volatility); capping limits their effect.
- Clusters describe past behaviour only; the choice of features and *k* changes the groups.

**Future scope:** add more features (expense ratio, fund size), compare with K-Means and DBSCAN, update the data
automatically, and track how funds move between groups over time.

*{DISCLAIMER}*
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        path = get_dataset_path(allow_download=False)
        modified = path.stat().st_mtime
        frame = load_data(str(path), modified)
    except DatasetError as error:
        if "not found" in str(error).lower():
            dataset_missing_page()
        else:
            st.error(str(error))
        return

    st.sidebar.title("📊 Fund Clustering")
    page = st.sidebar.radio("Go to", PAGES, key="page")
    st.sidebar.divider()
    st.sidebar.markdown("**Clustering settings**")
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 8, DEFAULT_CLUSTERS, key="k")
    linkage = st.sidebar.selectbox("Linkage method", LINKAGE_METHODS, index=LINKAGE_METHODS.index(DEFAULT_LINKAGE),
                                   key="linkage")
    features = st.sidebar.multiselect("Features", CANDIDATE_FEATURES, default=DEFAULT_FEATURES,
                                      format_func=lambda f: FEATURE_LABELS.get(f, f), key="features")
    if len(features) < 2:
        st.sidebar.error("Choose at least two features.")
        features = DEFAULT_FEATURES
    st.sidebar.caption(DISCLAIMER)

    try:
        result = cluster(str(path), modified, tuple(features), n_clusters, linkage)
    except ValueError as error:
        st.error(f"Clustering failed with these settings: {error}")
        return

    if page == "Overview":
        page_overview(frame, result)
    elif page == "Data Exploration":
        page_data(frame, result)
    elif page == "Clusters":
        page_clusters(frame, result, k_table(str(path), modified, tuple(features), linkage))
    elif page == "Explore Funds":
        page_explore(result)
    elif page == "Find a Fund's Group":
        page_find(result)
    else:
        page_about()


main()
