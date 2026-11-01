"""Run FLAML AutoML on one feature set and log it to MLflow.

FLAML tunes one configuration per learner (lgbm, xgboost, rf, extra_tree, ...),
so its leaderboard has one row per learner: the best configuration found and
its 5-fold CV RMSE.

Outputs in results/, where run is flaml_all or flaml_top3:
- {run}_leaderboard.csv/.md: learners ranked by CV RMSE, with timed refits
- {run}_importance.csv/.md: permutation importance of the best model; the
  all-features ranking selects the top-3 features
- {run}_summary.json: best learner and configuration, search statistics, test metrics
- {run}_search.log: FLAML's trial log
- models/{run}.pkl: the fitted AutoML object
"""

import json
import pickle
import time

import mlflow
import numpy as np
import pandas as pd
from flaml import AutoML

from src.config import (EXPLAIN_SAMPLE, FEATURE_SETS, FLAML_TIME_BUDGET, MODELS, N_FOLDS,
                        RESULTS, SEED)
from src.dataset import feature_list, load_table, split
from src.reporting import (LEADERBOARD_COLUMNS, leaderboard_metrics, mlflow_run,
                           permutation_importance_rmse, regression_metrics, write_json,
                           write_table)

TOOL = "flaml"


def _transform(automl: AutoML, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the data transform FLAML runs before its learners see the data.

    The transformer is private API, so a FLAML version that renames it falls
    back to the raw frame, which is already numeric plus category columns.
    """
    transformer = getattr(automl, "_transformer", None)
    return transformer.transform(X) if transformer is not None else X


def timed_refit(automl: AutoML, learner: str, X: pd.DataFrame, y: pd.Series):
    """Refit one learner's best configuration on the full training split and time it.

    FLAML records no comparable per-learner fit time, so every learner is
    refit here the same way. best_model_for_estimator returns the learner built
    with its best configuration. Returns (fit_seconds, predict_ms_per_row).
    """
    model = automl.best_model_for_estimator(learner)
    if model is None:
        return None, None
    Xt = _transform(automl, X)
    start = time.perf_counter()
    model.fit(Xt, y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model.predict(Xt)
    predict_ms_per_row = (time.perf_counter() - start) * 1000 / len(Xt)
    return fit_seconds, predict_ms_per_row


def log_search_history(search_log, time_budget: int) -> None:
    """Log each improving trial's CV RMSE as MLflow step metrics, so the UI charts the search."""
    try:
        from flaml.automl.data import get_output_from_log
        times, best_losses, losses, _, _ = get_output_from_log(str(search_log), time_budget)
    except Exception as exc:  # the log reader has moved between FLAML versions
        print(f"Search history not logged: {exc}")
        return
    for step, (seconds, loss, best) in enumerate(zip(times, losses, best_losses)):
        mlflow.log_metrics({"trial_cv_rmse": float(loss), "best_cv_rmse_so_far": float(best),
                            "trial_wall_seconds": float(seconds)}, step=step)


def run_flaml(feature_set: str = "all") -> dict:
    run = f"{TOOL}_{feature_set}"
    table = load_table()
    features = feature_list(table, TOOL, feature_set)
    X, y, folds, X_test, y_test = split(table, features)
    RESULTS.mkdir(parents=True, exist_ok=True)
    search_log = RESULTS / f"{run}_search.log"

    automl = AutoML()
    settings = {
        "task": "regression",
        "metric": "rmse",
        "time_budget": FLAML_TIME_BUDGET,
        "eval_method": "cv",
        # GroupKFold over the fold ids: each fold holds exactly one group,
        # so FLAML validates on the same folds as the baseline and H2O.
        "split_type": "group",
        "n_splits": N_FOLDS,
        "seed": SEED,
        "log_file_name": str(search_log),
        "verbose": 1,
        # FLAML's own MLflow logging saves its pipeline with mlflow.sklearn, whose
        # default skops format cannot serialize it. This module logs the run itself.
        "mlflow_logging": False,
    }

    with mlflow_run(run, {"tool": TOOL, "feature_set": feature_set}):
        mlflow.log_params({**{k: v for k, v in settings.items() if k != "log_file_name"},
                           "n_features": len(features), "features": ",".join(features)[:500],
                           "n_train": len(X), "n_test": len(X_test)})

        start = time.perf_counter()
        # FLAML modifies its inputs in place (it resets y's index), so it gets copies
        # and X, y keep the row labels used below.
        automl.fit(X_train=X.copy(), y_train=y.copy(), groups=folds.to_numpy(), **settings)
        search_seconds = time.perf_counter() - start

        test = regression_metrics(y_test, automl.predict(X_test))

        # Importance is measured on training rows so the test set never guides
        # the choice of top features.
        sample = X.sample(n=min(EXPLAIN_SAMPLE, len(X)), random_state=SEED)
        importance = permutation_importance_rmse(automl.predict, sample, y.loc[sample.index])
        write_table(importance, f"{run}_importance")

        rows = []
        for learner, loss in automl.best_loss_per_estimator.items():
            if loss is None or not np.isfinite(loss):
                continue  # learner never completed a trial within the budget
            fit_seconds, predict_ms_per_row = timed_refit(automl, learner, X, y)
            rows.append({"model": learner, "algorithm": learner, "cv_rmse": float(loss),
                         "cv_mae": None, "fit_seconds": fit_seconds,
                         "predict_ms_per_row": predict_ms_per_row,
                         "config": json.dumps(automl.best_config_per_estimator[learner])})
        leaderboard = (pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS + ["config"])
                       .sort_values("cv_rmse", ignore_index=True))
        write_table(leaderboard, f"{run}_leaderboard")

        best_row = leaderboard[leaderboard["model"] == automl.best_estimator].iloc[0]
        mlflow.log_metrics({**leaderboard_metrics(best_row),
                            "search_seconds": search_seconds,
                            "time_to_find_best_model": float(automl.time_to_find_best_model),
                            **{f"test_{k}": v for k, v in test.items()}})
        mlflow.set_tag("best_learner", automl.best_estimator)
        mlflow.log_artifact(str(search_log))
        log_search_history(search_log, FLAML_TIME_BUDGET)

        MODELS.mkdir(parents=True, exist_ok=True)
        model_path = MODELS / f"{run}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(automl, f)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        summary = {
            "run": run,
            "best_model": automl.best_estimator,
            "best_config": automl.best_config,
            "best_cv_rmse": float(automl.best_loss),
            "best_iteration": automl.best_iteration,
            "time_to_find_best_model": float(automl.time_to_find_best_model),
            "search_seconds": search_seconds,
            "time_budget": FLAML_TIME_BUDGET,
            "test": test,
            "features": features,
            "top_features": importance["feature"].head(5).tolist(),
        }
        write_json(summary, f"{run}_summary")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    args = parser.parse_args()
    print(run_flaml(args.features))
