"""
Automated checks. Run from the project folder with:

    python -m unittest -v

Most tests use the real dataset in data/. A few use small made-up tables only
to test error handling; those numbers mean nothing about real funds.
"""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

import train
from src import charts, clustering, data

REAL_DATA = data.DATA_PATH.is_file()


@unittest.skipUnless(REAL_DATA, "data/comprehensive_mutual_funds_data.csv not found")
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = data.load_dataset(required_numeric=clustering.DEFAULT_FEATURES)
        cls.result = clustering.run_clustering(cls.frame)

    def test_file_is_the_expected_dataset(self):
        self.assertEqual(data._sha256(data.DATA_PATH), data.EXPECTED_SHA256)
        self.assertEqual(self.frame.shape, (814, 21))  # 20 original columns + fund_id

    def test_dash_values_become_missing_numbers(self):
        for column in ["sd", "sharpe", "beta", "alpha", "sortino"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.frame[column]), column)
        self.assertEqual(int(self.frame["sd"].isna().sum()), 24)
        self.assertEqual(int(self.frame["returns_5yr"].isna().sum()), 167)

    def test_funds_with_missing_features_are_skipped(self):
        complete = self.frame.dropna(subset=clustering.DEFAULT_FEATURES)
        self.assertEqual(len(self.result.data), len(complete))
        self.assertEqual(self.result.rows_dropped, len(self.frame) - len(complete))

    def test_clusters_are_numbered_from_lowest_to_highest_risk(self):
        medians = self.result.data.groupby("cluster")["sd"].median()
        self.assertEqual(list(medians.index), [1, 2, 3, 4])
        self.assertTrue(medians.is_monotonic_increasing)
        self.assertEqual(int(self.result.summary["Funds"].sum()), len(self.result.data))

    def test_names_are_unique(self):
        names = list(self.result.names.values())
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all("risk" in n and "return" in n for n in names))

    def test_sklearn_and_scipy_trees_give_the_same_groups(self):
        scipy_labels = fcluster(self.result.linkage_matrix, self.result.n_clusters, criterion="maxclust")
        self.assertAlmostEqual(adjusted_rand_score(self.result.data["cluster"], scipy_labels), 1.0)

    def test_metrics_are_in_valid_ranges(self):
        m = self.result.metrics
        self.assertTrue(-1 <= m["silhouette"] <= 1)
        self.assertGreater(m["davies_bouldin"], 0)
        self.assertTrue(0 <= m["cophenetic_correlation"] <= 1)
        self.assertTrue(-1 <= m["ari_vs_category"] <= 1)

    def test_k_selection_table(self):
        table = clustering.k_selection_table(self.frame)
        self.assertEqual(list(table["k"]), list(clustering.K_RANGE))
        self.assertTrue(table["silhouette"].between(-1, 1).all())

    def test_every_linkage_method_runs(self):
        for method in clustering.LINKAGE_METHODS:
            result = clustering.run_clustering(self.frame, n_clusters=3, linkage=method)
            self.assertEqual(sorted(result.names), [1, 2, 3])

    def test_capping_prevents_a_one_fund_cluster(self):
        sizes = self.result.data["cluster"].value_counts()
        self.assertGreater(int(sizes.min()), 1)

    def test_assign_existing_fund_mostly_matches_its_cluster(self):
        matches = 0
        for _, fund in self.result.data.iterrows():
            values = {f: fund[f] for f in self.result.features}
            cluster_id, distances, _ = clustering.assign_cluster(
                self.result.preprocessor, self.result.centroids, self.result.features, values)
            self.assertEqual(len(distances), self.result.n_clusters)
            matches += cluster_id == fund["cluster"]
        self.assertGreater(matches / len(self.result.data), 0.9)

    def test_similar_funds_excludes_the_fund_itself(self):
        fund = self.result.data.iloc[0]
        _, _, point = clustering.assign_cluster(self.result.preprocessor, self.result.centroids,
                                                self.result.features, {f: fund[f] for f in self.result.features})
        nearest = clustering.similar_funds(self.result, point, n=5, exclude_fund_id=int(fund["fund_id"]))
        self.assertEqual(len(nearest), 5)
        self.assertNotIn(int(fund["fund_id"]), set(nearest["fund_id"]))
        self.assertTrue(nearest["distance"].is_monotonic_increasing)

    def test_fund_labels_are_unique(self):
        labels = self.result.data.apply(clustering.fund_label, axis=1)
        self.assertEqual(labels.nunique(), len(labels))

    def test_training_script_writes_all_outputs_and_model_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code = train.main(["--no-download", "--models-dir", str(tmp / "models"), "--reports-dir", str(tmp / "rep")])
            self.assertEqual(code, 0)
            for name in ["fund_clusters.csv", "cluster_summary.csv", "k_selection.csv", "cluster_metrics.txt"]:
                self.assertTrue((tmp / "rep" / name).is_file(), name)
            self.assertEqual(len(list((tmp / "rep").glob("*.png"))), 7)
            bundle = joblib.load(tmp / "models" / "fund_clusters.joblib")
            cluster_id, _, _ = clustering.assign_cluster(bundle["preprocessor"], bundle["centroids"],
                                                         bundle["features"],
                                                         {"returns_3yr": 40, "sd": 20, "sharpe": 1.9})
            self.assertEqual(bundle["names"][cluster_id], self.result.names[cluster_id])
            exported = pd.read_csv(tmp / "rep" / "fund_clusters.csv")
            self.assertEqual(len(exported), len(self.result.data))

    def test_training_script_rejects_bad_settings(self):
        with self.assertRaises(SystemExit):
            train.main(["--no-download", "--clusters", "1"])


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def small_table(n: int = 40) -> pd.DataFrame:
        """Made-up funds (test data only)."""
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "scheme_name": [f"Fund {i}" for i in range(n)],
            "amc_name": "Test AMC",
            "category": rng.choice(["Debt", "Equity"], n),
            "sub_category": "Test",
            "returns_3yr": rng.uniform(4, 40, n).round(1).astype(object),
            "sd": rng.uniform(0.5, 20, n).round(2).astype(object),
            "sharpe": rng.uniform(-0.5, 2, n).round(2).astype(object),
            "risk_level": rng.integers(1, 7, n),
        })

    def test_missing_file_explains_manual_download(self):
        with self.assertRaises(data.DatasetError) as error:
            data.load_dataset(self.tmp / "nothing.csv")
        self.assertIn("kaggle.com", str(error.exception))

    def test_missing_column_is_reported(self):
        path = self.tmp / "funds.csv"
        self.small_table().drop(columns=["sd"]).to_csv(path, index=False)
        with self.assertRaisesRegex(data.DatasetError, "sd"):
            data.load_dataset(path, required_numeric=["sd"])

    def test_text_in_numeric_column_is_reported_but_dash_is_allowed(self):
        table = self.small_table()
        table.loc[0, "sd"] = "-"
        path = self.tmp / "ok.csv"
        table.to_csv(path, index=False)
        self.assertTrue(np.isnan(data.load_dataset(path).loc[0, "sd"]))
        table.loc[1, "sharpe"] = "high"
        table.to_csv(path, index=False)
        with self.assertRaisesRegex(data.DatasetError, "sharpe"):
            data.load_dataset(path)

    def test_invalid_settings_are_rejected(self):
        frame = data.load_dataset(self._write(self.small_table()))
        with self.assertRaises(ValueError):
            clustering.run_clustering(frame, features=["sd"])
        with self.assertRaises(ValueError):
            clustering.run_clustering(frame, linkage="centroid")
        with self.assertRaises(ValueError):
            clustering.run_clustering(frame, features=["sd", "not_a_column"])
        with self.assertRaises(ValueError):
            clustering.run_clustering(frame.head(3), n_clusters=4)

    def test_invalid_new_fund_values_are_rejected(self):
        frame = data.load_dataset(self._write(self.small_table()))
        result = clustering.run_clustering(frame, n_clusters=3)
        args = (result.preprocessor, result.centroids, result.features)
        for bad in [{"returns_3yr": "abc", "sd": 1, "sharpe": 1}, {"returns_3yr": None, "sd": 1, "sharpe": 1},
                    {"returns_3yr": float("nan"), "sd": 1, "sharpe": 1}, {"sd": 1, "sharpe": 1}]:
            with self.assertRaises(ValueError):
                clustering.assign_cluster(*args, bad)
        cluster_id, _, _ = clustering.assign_cluster(*args, {"returns_3yr": "12.5", "sd": 3, "sharpe": 0.5})
        self.assertIn(cluster_id, result.names)

    def test_winsorizer_learns_bounds_from_fit_data_only(self):
        clipper = clustering.Winsorizer(0.1, 0.9).fit(np.arange(11, dtype=float).reshape(-1, 1))
        self.assertEqual(clipper.transform([[100.0], [-5.0]]).ravel().tolist(), [9.0, 1.0])

    def test_failed_or_wrong_download_gives_instructions(self):
        target = self.tmp / "data" / data.CSV_NAME
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(data.DatasetError, "manually"):
                data.download_dataset(target)
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"not,the,dataset\n"
        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(data.DatasetError, "checksum"):
                data.download_dataset(target)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.glob("*")), [])

    def test_charts_are_saved(self):
        frame = data.load_dataset(self._write(self.small_table(60)))
        result = clustering.run_clustering(frame, n_clusters=3)
        table = clustering.k_selection_table(frame, k_values=range(2, 5))
        paths = charts.save_all(frame, result, table, self.tmp / "charts")
        self.assertEqual(len(paths), 7)
        self.assertTrue(all(p.stat().st_size > 1000 for p in paths))

    def _write(self, table: pd.DataFrame) -> Path:
        path = self.tmp / "funds.csv"
        table.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
