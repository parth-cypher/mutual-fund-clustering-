# Grouping Mutual Funds by Risk and Return Using Hierarchical Clustering

This project groups Indian mutual funds that behaved alike in **risk** and **return**. It uses
**agglomerative hierarchical clustering**. It does not tell the model each fund's official category; it then checks
whether the groups it finds make sense. A **Streamlit** dashboard shows the dendrogram and the groups, and lets you
change the settings and look up any fund.

> Educational project based on past data. Past returns do not guarantee future returns. This is not investment advice.

## Project structure

```
mutual-fund-clustering/
├── app.py               Streamlit dashboard
├── train.py             Runs the clustering and saves results, charts and the model
├── requirements.txt
├── src/
│   ├── data.py          Finds, downloads and checks the dataset
│   ├── clustering.py    Preprocessing, hierarchical clustering, naming, evaluation
│   └── charts.py        Matplotlib charts
├── data/                comprehensive_mutual_funds_data.csv (included) + README.md
├── models/              fund_clusters.joblib (created by train.py)
├── reports/             CSV/TXT results and PNG charts (created by train.py)
└── tests/test_project.py
```

## Dataset

| | |
|---|---|
| **Name** | Mutual Funds India – Detailed |
| **Source** | Kaggle, uploaded by ravibarnawal: <https://www.kaggle.com/datasets/ravibarnawal/mutual-funds-india-detailed> |
| **Copy included** | `data/comprehensive_mutual_funds_data.csv`, taken from <https://github.com/nileshiq/Mutual-Funds-India> (that repository's README credits the Kaggle dataset). |
| **Size** | 814 mutual fund schemes, 20 columns |

**Columns used for clustering (default)**

| Column | Meaning |
|---|---|
| `returns_3yr` | Return over the last 3 years (%) – the **return** measure |
| `sd` | Standard deviation of returns (%) – how much returns swing, the **risk** measure |
| `sharpe` | Sharpe ratio – extra return per unit of risk, the **risk-adjusted return** measure |

Other columns include:
- returns: `returns_1yr`, `returns_5yr`
- risk ratios: `beta`, `sortino`, `alpha`
- fund details: `expense_ratio`, `fund_size_cr`, `fund_age_yr`, `min_sip`, `min_lumpsum`, `fund_manager`, `amc_name`
- labels: `rating`, `risk_level` (1–6, which appears to follow SEBI's riskometer), `category` (Equity, Debt, Hybrid,
  Solution Oriented, Other) and `sub_category`

In the file, values that are not available are written as `-`; the code reads them as missing values. 25 funds miss
at least one default feature and are left out of the clustering.

**Why these three features?** They describe return, risk and risk-adjusted return:
- `returns_1yr` is short-term and contains an extreme one-off value (one fund shows 130.8 %).
- `returns_5yr` is missing for 167 funds.
- `beta` and `alpha` are missing for 42 funds and depend on each fund's benchmark.

You can still add any of these in the dashboard or with `train.py --features`.

## Method

1. **Load and check** the CSV. Make sure the required columns exist, read `-` as missing, convert numbers, and
   report problems clearly. The file itself is never changed.
2. **Select funds** that have every chosen feature.
3. **Cap extreme values** at the 1st and 99th percentiles (winsorizing). Without this, a single unusual fund (for
   example one with a 77.7 % standard deviation) ends up as a cluster of its own.
4. **Standardise** each feature to mean 0 and standard deviation 1, so that no feature dominates because of its
   units.
5. **Agglomerative hierarchical clustering** (scikit-learn `AgglomerativeClustering`). Every fund starts as its own
   cluster, and the two closest clusters are merged again and again. The record of merges is the **dendrogram**
   (drawn with SciPy); cutting it at a chosen height gives *k* clusters.
   - **Ward linkage** (default) merges the pair of clusters whose merge increases the within-cluster variance the
     least. This gives compact, similar-sized groups.
   - `complete`, `average` and `single` linkage can be compared in the dashboard. Single linkage tends to "chain"
     almost all funds into one cluster.
6. **Number and name the clusters.**
   - Clusters are numbered 1…k from the lowest to the highest median standard deviation.
   - Each gets a name such as "Low risk · very low return". The words come from where the cluster's median standard
     deviation and 3-year return sit among all funds (fifths: very low, low, moderate, high, very high).
7. **Evaluate.**
   - **Silhouette score**, from −1 to 1 (higher is better): how much closer funds are to their own cluster than to
     the next one.
   - **Davies–Bouldin index** (lower is better) and **Calinski–Harabasz score** (higher is better).
   - **Cophenetic correlation**: how well the dendrogram preserves the real distances between funds.
   - **Adjusted Rand index** against the funds' `category` and `risk_level`. These labels were *not* used for
     clustering, so this is an independent check.
8. **Choose k.** Scores are calculated for k = 2…8 (`reports/k_selection.csv`, the "Choosing k" tab).
   - k = 2 usually scores highest, because it simply splits debt-like from equity-like funds.
   - The default k = 4 still separates well and is more useful: it splits the low-risk funds by risk-adjusted return,
     and singles out the highest-risk, highest-return group.
   - You can change k in the sidebar or with `train.py --clusters`.
9. **Placing a new fund.** Hierarchical clustering has no `predict` step, so a new fund goes to the cluster whose
   centre is closest after the same capping and scaling. On this dataset the nearest centre agrees with the tree's
   own cluster for about 95 % of funds.

## Installation (macOS, VS Code terminal)

Open the terminal in VS Code (**Terminal → New Terminal**):

```bash
cd mutual-fund-clustering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the clustering

```bash
python train.py
```

This prints the clusters, their median figures and the quality scores, and saves:

- `reports/fund_clusters.csv` – every fund with its cluster
- `reports/cluster_summary.csv` – one row per cluster
- `reports/k_selection.csv` – scores for k = 2…8
- `reports/cluster_metrics.txt` – readable summary
- `reports/*.png` – the dendrogram and the other charts (risk vs return, cluster profiles, clusters vs category,
  choosing k, feature distributions)
- `models/fund_clusters.joblib` – scaler and cluster centres, used to place new funds

Options:

```bash
python train.py --clusters 5 --linkage complete
python train.py --features returns_1yr returns_3yr sd sharpe
python train.py --help
```

## Launch the dashboard

```bash
streamlit run app.py
```

It opens at <http://localhost:8501> (press **Enter** if Streamlit asks for an email; stop it with **Ctrl+C**). Use
these terminal commands, not VS Code's ▶ Run button.

| Page | What it shows |
|---|---|
| Overview | Objective, dataset, number of funds, the groups on a risk-vs-return chart, cluster summary |
| Data Exploration | Data preview, column meanings, missing values, feature distributions, risk vs return by category |
| Clusters | Quality scores, dendrogram, risk vs return by cluster, cluster profiles, clusters vs category, choosing k |
| Explore Funds | Every fund in a chosen cluster, search by name, download as CSV |
| Find a Fund's Group | Pick a fund (or type in its numbers) to see its group and the 5 most similar funds |
| About | Problem, workflow, linkage methods, limitations, future scope |

The sidebar changes the number of clusters, the linkage method and the features. The app re-clusters instantly, so
`train.py` does not need to run first.

## Run the tests

```bash
python -m unittest -v
```

The tests check:
- that the included file is the expected dataset (checksum)
- that `-` is read as a missing value
- that clusters are ordered by risk and have unique names
- that the scikit-learn clustering matches the SciPy dendrogram
- that every linkage method runs
- that new funds are placed correctly and similar-fund search works
- the full `train.py` run and reloading the saved model
- error handling: missing file or columns, bad values, failed download

## Limitations

- **A snapshot:** the data was scraped once; real fund figures change daily, and the scrape date is not stated.
- **Undocumented time window:** the dataset does not say which period `sd`, `sharpe` and the other ratios cover.
- **Unusual values:** a few funds look odd.
  - Four unrelated debt funds (overnight, ultra-short, corporate bond and dynamic bond) all show exactly 32.4 %
    3-year return. This is very likely a scraping error.
  - One fund has a beta of −31.9.
  - Capping limits the effect of these values, but they are not removed, because the original file is kept unchanged.
- **Duplicate names:** some scheme names appear more than once (different plans of the same scheme). Each row is
  kept and given a `fund_id`.
- **Choices change the groups:** the features, k and the linkage method all affect the result. Clusters describe
  past behaviour, not future performance.

## Future scope

- Compare with K-Means, DBSCAN and Gaussian Mixture Models.
- Add cost and size features (expense ratio, fund size) or more return periods.
- Refresh the data automatically and track how funds move between groups over time.
- Use the groups to suggest diversified sets of funds (with proper financial guidance).

---
Academic project for a B.Tech CSE Machine Learning course. The dataset belongs to its original authors on Kaggle.
