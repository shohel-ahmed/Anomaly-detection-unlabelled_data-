# load required libraries
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from csv import writer
from collections import Counter

import duckdb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from colorama import Fore, Back, Style

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from constants import CONTAMINATION_RATIO
from utils.utility import Utility


# =========================
# Data Preprocessor
# =========================
class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Handles comprehensive data cleaning and preprocessing with datetime handling."""

    def __init__(self, max_cardinality=50):
        self.max_cardinality = max_cardinality
        self.num_cols = []
        self.cat_cols = []
        self.id_cols = []
        self.dt_cols = []              # NEW: track datetime columns
        self.feature_names_ = []
        self.scaler = StandardScaler()
        self.encoder = None

    def _to_epoch_seconds(self, s: pd.Series) -> pd.Series:
        """Vectorized datetime -> epoch seconds; NaT -> NaN."""
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        # (ts - epoch).dt.total_seconds() preserves NaT -> NaN
        return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # --- Detect and convert datetime columns up front ---
        self.dt_cols = []
        for c in X.columns:
            if np.issubdtype(X[c].dtype, np.datetime64):
                self.dt_cols.append(c)
        for c in self.dt_cols:
            X[c] = self._to_epoch_seconds(X[c])

        # Save feature names
        self.feature_names_ = X.columns.tolist()

        # Identify numeric and categorical columns (after datetime conversion)
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Identify potential ID columns (80%+ unique, non-numeric)
        unique_threshold = 0.8
        self.id_cols = [
            col
            for col in X.columns
            if (len(X) > 0 and (X[col].nunique() / len(X) > unique_threshold))
            and (col not in self.num_cols)
        ]

        print(f"Identified Categorical Columns: {self.cat_cols}")
        print(f"Identified Numeric Columns: {self.num_cols}")
        print(f"Identified ID Columns: {self.id_cols}")

        # Remove ID columns from lists
        self.cat_cols = [col for col in self.cat_cols if col not in self.id_cols]
        self.num_cols = [col for col in self.num_cols if col not in self.id_cols]

        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame

        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Drop ID columns
        X = X.drop(columns=self.id_cols, errors="ignore")

        # Apply the same datetime conversion
        for c in self.dt_cols:
            if c in X.columns:
                X[c] = self._to_epoch_seconds(X[c])

        # Refresh lists for columns that still exist
        existing_num_cols = [col for col in self.num_cols if col in X.columns]
        existing_cat_cols = [col for col in self.cat_cols if col in X.columns]

        # Numeric imputation (iterative)
        if existing_num_cols:
            num_imputer = IterativeImputer(max_iter=10, random_state=0)
            X[existing_num_cols] = num_imputer.fit_transform(X[existing_num_cols])

        # Categorical imputation
        if existing_cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X[existing_cat_cols] = cat_imputer.fit_transform(X[existing_cat_cols])

        # Remove duplicates
        X = X.drop_duplicates().reset_index(drop=True)

        # Label-encode only object/category columns (datetime already numeric)
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            X_transformed = X.copy()
            for col in categorical_cols:
                X_transformed[col] = LabelEncoder().fit_transform(X_transformed[col])
            X = pd.DataFrame(X_transformed, columns=X_transformed.columns, index=X.index)

        # Scale numerical columns to [0,1]
        if existing_num_cols:
            scaler = MinMaxScaler()
            X[existing_num_cols] = scaler.fit_transform(X[existing_num_cols])

        # Final safeguard — keep only numeric columns for sklearn
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric):
            X = X.drop(columns=list(non_numeric))

        return X


# (Optional) Kept for compatibility, not used in the current pipeline
class RobustEncoder(BaseEstimator, TransformerMixin):
    """Smart encoding for categorical features (kept as-is, but not used)."""

    def __init__(self, max_cardinality=50):
        self.max_cardinality = max_cardinality
        self.encoder = None
        self.high_card_cols = []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.encoder = ColumnTransformer(
            [("label", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.cat_cols)],
            remainder="passthrough",
        )
        self.encoder.fit(X)
        if self.num_cols:
            self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X = X.copy()
        transformed_X = self.encoder.transform(X)
        feature_names = self.cat_cols + self.num_cols
        transformed_X = pd.DataFrame(transformed_X, columns=feature_names, index=X.index)
        if self.num_cols:
            transformed_X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return transformed_X


# =========================
# ML Anomaly Detection
# =========================
class MLAnomalyDetection:
    def __init__(self, target_path, file_name, iforest_max_samples: int = 10000):
        self.target_path = target_path
        self.file_name = file_name
        self.newpath = f"{target_path}/Anomaly_Output"
        self.contamination_ratio = CONTAMINATION_RATIO
        self.iforest_max_samples = int(iforest_max_samples)  # NEW: cap for stability

        self.preprocessor = Pipeline(
            steps=[
                ("cleaner", DataPreprocessor()),
                # ('encoder', RobustEncoder())  # optional, not needed after DataPreprocessor
            ]
        )

    # ---------------- Utilities & Viz ----------------
    def decision_function(self, X, model):
        return model.decision_function(X)

    def show_scatter_2(self, X, outlier_labels):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=outlier_labels, cmap="coolwarm", edgecolors="k")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Outlier Visualization using PCA")
        plt.colorbar(label="Outlier (1) / Normal (0)")
        # plt.show()

    def shap_plot(self, X, model, outlier_labels):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_importance = np.abs(shap_values).mean(axis=0)
        important_features = np.argsort(feature_importance)[-2:]  # Top 2
        plt.scatter(
            X.iloc[:, important_features[0]],
            X.iloc[:, important_features[1]],
            c=outlier_labels,
            cmap="coolwarm",
            edgecolors="k",
        )
        plt.xlabel(X.columns[important_features[0]])
        plt.ylabel(X.columns[important_features[1]])
        plt.title("Outlier Visualization using Top SHAP Features")
        plt.colorbar(label="Outlier (1) / Normal (0)")
        # plt.show()

    # ---------------- Model Selection ----------------
    def _iforest_internal_score(self, estimator, X, y=None):
        """
        Unsupervised scorer: maximize average score_samples(X).
        Higher is better (more inlier-like density on validation folds).
        """
        s = estimator.score_samples(X)
        return float(np.mean(s))

    def iso_optimal_model(self, data_frame, contamination_ratio):
        """Enhanced model training with optimized search (unsupervised scorer + max_samples cap)."""
        X = data_frame

        # Cap max_samples for stability on big samples
        max_samples_actual = int(min(self.iforest_max_samples, len(X)))

        param_grid = {
            "n_estimators": [100, 200],
            "max_features": [0.5, 0.8],   # fraction of features
            "bootstrap": [True, False],
            # You can expose max_samples to search too if desired:
            # "max_samples": [256, 1024, max_samples_actual]
        }

        iso_forest = IsolationForest(
            contamination=contamination_ratio,
            random_state=42,
            max_samples=max_samples_actual,   # NEW: enforce cap
            n_jobs=-1,
        )

        search = RandomizedSearchCV(
            estimator=iso_forest,
            param_distributions=param_grid,
            n_iter=8,
            scoring=self._iforest_internal_score,  # custom unsupervised scorer
            cv=3,
            n_jobs=-1,
            error_score="raise",  # surface the first real error
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X)

        return search.best_estimator_

    # ---------------- Helpers ----------------
    def check_numeral_ordinal_nominal(self, data_frame, col):
        dtype = data_frame[col].dtype
        uniques = data_frame[col].unique()
        if np.issubdtype(dtype, np.number):
            return "numeral"
        elif dtype == "object":
            sorted_uniques = sorted(uniques)
            if np.array_equal(uniques, sorted_uniques):
                return "ordinal"
            else:
                return "nominal"
        else:
            return "unknown"

    def tabular_encoding_categorical_data(self, data_frame):
        encoded_data = data_frame
        for col in data_frame.columns:
            if len(encoded_data[col].unique()) == len(encoded_data[col]):
                encoded_data = encoded_data.drop([col], axis=1)
            elif len(encoded_data[col]) == encoded_data[col].isnull().sum():
                encoded_data = encoded_data.drop([col], axis=1)
        categorical_cols = encoded_data.select_dtypes(include=["object"]).columns.tolist()
        for clm in categorical_cols:
            _ = self.check_numeral_ordinal_nominal(encoded_data, clm)
            encoded_data[clm] = LabelEncoder().fit_transform(encoded_data[clm])
        return encoded_data

    def iso_predictor(self, data_frame, model):
        iso_predictions = model.predict(data_frame)
        iso_outlier_indices = [i for i, e in enumerate(iso_predictions) if e == -1]
        iso_inlier_indices = [i for i, e in enumerate(iso_predictions) if e == 1]
        return iso_outlier_indices, iso_predictions

    def column_selection_for_model(self, attribute_df_location):
        csv_path = Path(f"{attribute_df_location}.csv")
        parquet_path = Path(f"{attribute_df_location}.parquet")
        if csv_path.exists():
            query = f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}')"
        elif parquet_path.exists():
            query = f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        else:
            raise FileNotFoundError("Profile file not found.")
        profile_df = duckdb.query(query).to_df()
        cat_col_name = profile_df.loc[
            (profile_df["Has_Distinct"] == 0)
            & (profile_df["Missing"] == 0)
            & (profile_df["Cat_variable_Count"] < 100)
            & (profile_df["Feature_Type"] == "Text"),
            "Attribute",
        ]
        num_col_name = profile_df.loc[
            (profile_df["Has_Distinct"] == 0)
            & (profile_df["Missing"] == 0)
            & (profile_df["Feature_Type"] == "Numeric"),
            "Attribute",
        ]
        col_name = pd.concat([cat_col_name, num_col_name])
        return col_name.to_list()

    def visualize_potential_anomaly(self, X, predicted_y, important_features):
        cmap = mcolors.ListedColormap(["red", "blue"])
        plt.figure(figsize=(12, 5))
        plt.scatter(
            X.iloc[:, important_features[0]],
            X.iloc[:, important_features[1]],
            c=predicted_y,
            cmap=cmap,
            marker="o",
            edgecolors="k",
        )
        plt.xlabel(X.columns[important_features[0]])
        plt.ylabel(X.columns[important_features[1]])
        plt.title("Potential Anomaly Detected")
        plt.tight_layout()
        x_lim = plt.xlim()
        y_lim = plt.ylim()
        fig = plt.gcf()
        fig.savefig(f"{self.newpath}\\normal_vs_potential_anomaly_plot.png", bbox_inches="tight")
        return x_lim, y_lim

    def visualize_data_without_anomaly(self, XX, x_lim, y_lim, features):
        plt.figure(figsize=(12, 5))
        plt.ylim(y_lim[0], y_lim[1])
        plt.xlim(x_lim[0], x_lim[1])
        plt.scatter(XX[features[0]], XX[features[1]], cmap="viridis", marker="o")
        plt.title("Normal Data (Without potential anomaly)", fontsize=12, fontfamily="sans-serif")
        plt.xlabel(f"{features[0]}", fontsize=12, fontfamily="sans-serif")
        plt.ylabel(f"{features[1]}", fontsize=12, fontfamily="sans-serif")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(f"{self.newpath}\\normal_data_plot.png", bbox_inches="tight")
        # plt.show()

    def data_loading(self, target_loc, table_name, sample_size=1_000_000):
        """
        Load data using DuckDB, but only bring a sample into memory for ML.
        """
        target_path = Path(target_loc)
        csv_path = target_path / f"{table_name}.csv"
        parquet_path = target_path / f"{table_name}.parquet"

        if csv_path.exists():
            rel = duckdb.read_csv(csv_path.as_posix())
        elif parquet_path.exists():
            rel = duckdb.read_parquet(parquet_path.as_posix())
        else:
            raise FileNotFoundError("Neither CSV nor Parquet file found.")

        # Return a limited sample for sklearn instead of full dataset
        df = rel.limit(sample_size).to_df()
        return df

    def anomaly_explainer(self, model, all_data):
        potential_anomaly_data = all_data[all_data["CLASS"] == -1]
        anomaly_data_without_class = potential_anomaly_data.drop("CLASS", axis=1)
        feature_cols = anomaly_data_without_class.columns

        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(anomaly_data_without_class)
        feature_importance = np.abs(shap_values).mean(axis=0)
        most_significant_features = np.argsort(feature_importance)[-2:]  # Top 2

        # Sort anomalies by most important features
        important_feature_names = [feature_cols[i] for i in most_significant_features]
        sorted_anomaly_data = potential_anomaly_data.sort_values(
            by=important_feature_names, ascending=[False, False]
        )
        merged_data = pd.concat([sorted_anomaly_data, all_data]).drop_duplicates()

        # SHAP summary plot
        plt.clf()
        fig = plt.gcf()
        shap.summary_plot(shap_values, anomaly_data_without_class, show=False)
        plt.gcf().axes[-1].set_ylabel("Variable Impact", fontsize=12, fontfamily="sans-serif")
        plt.title("SHAP Summary Plot", fontsize=12, fontfamily="sans-serif")
        plt.xlabel("SHAP Value (impact on model output)", fontsize=12, fontfamily="sans-serif")
        plt.xticks(fontsize=12, fontfamily="sans-serif")
        plt.yticks(fontsize=12, fontfamily="sans-serif")
        fig.savefig(f"{self.newpath}/shap_summary_plot.png", bbox_inches="tight")

        potential_anomaly_data["most_contributing_col"] = pd.Series(dtype="object")
        potential_anomaly_data["most_contributing_col_values"] = pd.Series(dtype="object")
        return potential_anomaly_data, most_significant_features, merged_data

    def append_csv(self, csv_file, List):
        with open(f"{csv_file}.csv", "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
        f_object.close()

    def count_rows(self, file_name: str, target_path: str) -> int:
        """
        Count rows in a large CSV or Parquet file without loading into memory.
        Uses DuckDB for efficient metadata-based counting.
        """
        target_path = Path(target_path)
        csv_path = target_path / f"{file_name}.csv"
        parquet_path = target_path / f"{file_name}.parquet"
        con = duckdb.connect()
        if csv_path.exists():
            query = f"SELECT COUNT(*) FROM read_csv_auto('{csv_path.as_posix()}')"
        elif parquet_path.exists():
            query = f"SELECT COUNT(*) FROM read_parquet('{parquet_path.as_posix()}')"
        else:
            raise ValueError("Unsupported file format. Only CSV or Parquet is supported.")
        samples_count = con.execute(query).fetchone()[0]
        con.close()
        return samples_count

    # ---------------- Orchestration ----------------
    def execute(self):
        start_time = Utility.log_start_time()
        Utility.ensure_directory_exists(self.newpath)

        # Determine a safe sample size for Isolation Forest
        row_count = self.count_rows(self.file_name, self.target_path)
        print(f"Total samples in the dataset:{row_count}")
        SAMPLE_SIZE = 500_000 if row_count > 500_000 else int(row_count)

        # Load a manageable sample
        X = self.data_loading(self.target_path, self.file_name, sample_size=SAMPLE_SIZE)

        # Preprocess (ensures numeric-only; datetimes converted)
        X = self.preprocessor.fit_transform(X)

        # Extra safety: ensure numeric & finite
        X = X.replace([np.inf, -np.inf], np.nan)
        # Drop rows with any NaN (IsolationForest cannot handle NaNs)
        X = X.dropna(axis=0, how="any").reset_index(drop=True)

        # Train Isolation Forest on sample (with max_samples cap)
        model = self.iso_optimal_model(X, self.contamination_ratio)

        # Predict
        anomaly_indices, predicted_y = self.iso_predictor(X, model)
        X["CLASS"] = predicted_y

        # Explain & visualize
        anomaly_data_explainer, significant_features, sorted_data = self.anomaly_explainer(model, X)
        x_lim, y_lim = self.visualize_potential_anomaly(X, predicted_y, significant_features)

        # Save outputs
        Utility.save_table(self.newpath, f"anomaly_with_exp.csv", anomaly_data_explainer)
        Utility.save_table(self.newpath, f"ML_anomaly_prediction_table.csv", sorted_data)

        elapsed_time = Utility.log_end_time(start_time)
        self.append_csv(f"{self.target_path}/task_time_{self.file_name}", ["ML_Anomaly_Detection", elapsed_time])
        print("Data validation ISO module runs successfully!")
