"""Shared helpers: metrics, permutation importance, result files, and MLflow runs.

Every run writes its leaderboard with the same columns, so compare.py can
rank FLAML, H2O, and the baseline in one table:
model, algorithm, cv_rmse, cv_mae, fit_seconds, predict_ms_per_row.
"""

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RESULTS, SEED

LEADERBOARD_COLUMNS = ["model", "algorithm", "cv_rmse", "cv_mae", "fit_seconds",
                       "predict_ms_per_row"]


def leaderboard_metrics(row: pd.Series) -> dict:
    """The best model's leaderboard columns as MLflow metrics, under the same names in every run.

    Values a tool does not report (FLAML has no CV MAE) are left out.
    """
    names = ["cv_rmse", "cv_mae", "fit_seconds", "predict_ms_per_row"]
    return {name: float(row[name]) for name in names if pd.notna(row[name])}


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def permutation_importance_rmse(predict, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5,
                                seed: int = SEED) -> pd.DataFrame:
    """Mean RMSE increase when each feature is shuffled, sorted from most to least important.

    Written by hand because sklearn's permutation_importance validates the
    estimator type, which FLAML's AutoML object does not declare.
    """
    rng = np.random.default_rng(seed)
    base = regression_metrics(y, predict(X))["rmse"]
    rows = []
    for col in X.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            # .values keeps a Categorical for category columns, so the dtype survives.
            shuffled[col] = X[col].iloc[rng.permutation(len(X))].values
            increases.append(regression_metrics(y, predict(shuffled))["rmse"] - base)
        rows.append({"feature": col, "rmse_increase": float(np.mean(increases)),
                     "rmse_increase_std": float(np.std(increases))})
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False, ignore_index=True)


def write_table(df: pd.DataFrame, name: str, log: bool = True) -> None:
    """Write results/{name}.csv and .md, and log both to the active MLflow run."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = RESULTS / f"{name}.csv", RESULTS / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False, floatfmt=".4g") + "\n")
    if log:
        mlflow.log_artifact(str(csv_path), artifact_path="tables")
        mlflow.log_artifact(str(md_path), artifact_path="tables")


def write_json(obj: dict, name: str, log: bool = True) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    if log:
        mlflow.log_artifact(str(path))


def read_json(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def mlflow_run(run_name: str, tags: dict):
    """Start an MLflow run in the project experiment. Use as a context manager."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=run_name, tags=tags)
