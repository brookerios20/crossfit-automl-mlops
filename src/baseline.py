"""Rescore the Assignment 2 model under the same CV folds and test set as the AutoML runs.

The model is the Assignment 2 run v2_hp_a: HistGradientBoostingRegressor on the
v2 features (the all-features set here) with hyperparameter set hp_a. Its
Assignment 2 test RMSE was 141.3, which this run should reproduce.

Outputs in results/: baseline_all_leaderboard.csv/.md and baseline_all_summary.json.
"""

import time

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import BASELINE_PARAMS, N_FOLDS, SEED
from src.dataset import all_features, load_table, split
from src.reporting import LEADERBOARD_COLUMNS, mlflow_run, regression_metrics, write_json, write_table

RUN = "baseline_all"


def build_model(X: pd.DataFrame, params: dict) -> Pipeline:
    """Ordinal-encode the categorical columns and pass them to HGB as native categoricals.

    Copied from Assignment 2 src/train.py. Unseen categories at predict time
    become -1, which HGB treats as missing.
    """
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    encoder = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # After the transformer, categorical columns come first.
    is_categorical = [True] * len(categorical) + [False] * len(numeric)
    model = HistGradientBoostingRegressor(categorical_features=is_categorical,
                                          random_state=SEED, **params)
    return Pipeline([("encode", encoder), ("model", model)])


def run_baseline() -> dict:
    table = load_table()
    features = all_features(table)
    X, y, folds, X_test, y_test = split(table, features)
    pipe = build_model(X, BASELINE_PARAMS)

    with mlflow_run(RUN, {"tool": "baseline", "feature_set": "all",
                          "algorithm": "HistGradientBoostingRegressor"}):
        mlflow.log_params({**BASELINE_PARAMS, "n_features": len(features), "n_train": len(X),
                           "n_test": len(X_test), "n_folds": N_FOLDS})

        # One fold per group value, so these are the same folds FLAML and H2O use.
        cv = cross_validate(pipe, X, y, groups=folds, cv=GroupKFold(n_splits=N_FOLDS),
                            scoring={"rmse": "neg_root_mean_squared_error",
                                     "mae": "neg_mean_absolute_error"})

        start = time.perf_counter()
        pipe.fit(X, y)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pipe.predict(X)
        predict_ms_per_row = (time.perf_counter() - start) * 1000 / len(X)

        test = regression_metrics(y_test, pipe.predict(X_test))
        row = {"model": "assignment2_v2_hp_a", "algorithm": "HistGradientBoostingRegressor",
               "cv_rmse": float(-cv["test_rmse"].mean()), "cv_mae": float(-cv["test_mae"].mean()),
               "fit_seconds": fit_seconds, "predict_ms_per_row": predict_ms_per_row}
        mlflow.log_metrics({"cv_rmse": row["cv_rmse"], "cv_mae": row["cv_mae"],
                            "fit_seconds": fit_seconds, "predict_ms_per_row": predict_ms_per_row,
                            **{f"test_{k}": v for k, v in test.items()}})

        write_table(pd.DataFrame([row], columns=LEADERBOARD_COLUMNS), f"{RUN}_leaderboard")
        summary = {"run": RUN, "best_model": row["model"], "best_cv_rmse": row["cv_rmse"],
                   "test": test, "n_features": len(features), "n_train": len(X),
                   "n_test": len(X_test)}
        write_json(summary, f"{RUN}_summary")
    return summary


if __name__ == "__main__":
    print(run_baseline())
