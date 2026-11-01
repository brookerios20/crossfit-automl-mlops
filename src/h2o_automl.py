"""Run H2O AutoML on one feature set and log it to MLflow.

H2O trains GLM, GBM, DRF, XGBoost, and deep learning models plus stacked
ensembles. The leaderboard metrics are 5-fold CV metrics on the training split,
using the same folds as FLAML and the baseline through fold_column.

Outputs in results/, where run is h2o_all or h2o_top3:
- {run}_leaderboard.csv/.md: every model ranked by CV RMSE, with H2O's training and prediction times
- {run}_importance.csv/.md: variable importance of the best non-ensemble model; the
  all-features ranking selects the top-3 features
- {run}_summary.json: leader, test metrics, and the model the importance came from
- {run}_*.png: explain plots (variable importance heatmap, model correlation, SHAP, partial dependence)
- models/{run}/: the leader as a MOJO, or as an H2O binary model where MOJO export is blocked
"""

import time

import matplotlib

matplotlib.use("Agg")  # write plots to files without a display

import h2o  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from h2o.automl import H2OAutoML, get_leaderboard  # noqa: E402

from src.config import (EXPLAIN_SAMPLE, FEATURE_SETS, FOLD_COL, H2O_MAX_MEM,  # noqa: E402
                        H2O_MAX_MODELS, MODELS, RESULTS, SEED, TARGET)
from src.dataset import CATEGORICAL, feature_list, load_table, split  # noqa: E402
from src.reporting import (LEADERBOARD_COLUMNS, leaderboard_metrics,  # noqa: E402
                           mlflow_run, regression_metrics, write_json, write_table)

TOOL = "h2o"


def _patch_numpy_str_names() -> None:
    """Make H2O serialize numpy string column names as plain strings.

    H2O writes each argument into its server expressions with repr(). Its explain
    code passes column names through np.array, and under NumPy 2 repr(np.str_("gender"))
    is "np.str_('gender')", which the server rejects. The SHAP and partial dependence
    plots for categorical columns fail without this.
    """
    from h2o.expr import ExprNode

    original = ExprNode._arg_to_expr
    if getattr(original, "_str_patched", False):
        return

    def arg_to_expr(arg):
        if isinstance(arg, np.str_):
            arg = str(arg)
        elif isinstance(arg, (list, tuple)):
            arg = type(arg)(str(x) if isinstance(x, np.str_) else x for x in arg)
        return original(arg)

    arg_to_expr._str_patched = True
    ExprNode._arg_to_expr = staticmethod(arg_to_expr)


_patch_numpy_str_names()


def to_h2o(X: pd.DataFrame, y: pd.Series, folds: pd.Series | None = None) -> h2o.H2OFrame:
    """Convert to an H2OFrame with the categorical columns as enums."""
    df = X.copy()
    df[TARGET] = y
    if folds is not None:
        df[FOLD_COL] = folds
    cats = [c for c in CATEGORICAL if c in df.columns]
    df[cats] = df[cats].astype(str)
    return h2o.H2OFrame(df, column_types={c: "enum" for c in cats})


def importance_by_feature(model, features: list[str]) -> pd.DataFrame | None:
    """Variable importance summed back onto the input columns.

    Some H2O algorithms one-hot encode enums and report levels such as
    "region.Europe", so each level is credited to its column.
    """
    varimp = model.varimp(use_pandas=True)
    if varimp is None or varimp.empty:
        return None

    def column(name: str) -> str:
        if name in features:
            return name
        prefix = name.split(".", 1)[0]
        return prefix if prefix in features else name

    varimp["feature"] = varimp["variable"].map(column)
    return (varimp.groupby("feature", as_index=False)["percentage"].sum()
            .rename(columns={"percentage": "importance_share"})
            .sort_values("importance_share", ascending=False, ignore_index=True))


def save_plot(make_plot, name: str) -> None:
    """Draw one H2O explain plot and save it, skipping plots the model type does not support."""
    path = RESULTS / f"{name}.png"
    try:
        result = make_plot()
        fig = result.figure() if hasattr(result, "figure") else plt.gcf()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(path), artifact_path="plots")
    except Exception as exc:  # explain plots vary by H2O version and model type
        print(f"Skipped {path.name}: {exc}")
    finally:
        plt.close("all")


def run_h2o(feature_set: str = "all") -> dict:
    run = f"{TOOL}_{feature_set}"
    table = load_table()
    features = feature_list(table, TOOL, feature_set)
    X, y, folds, X_test, y_test = split(table, features)

    h2o.init(max_mem_size=H2O_MAX_MEM, nthreads=-1)
    # A rerun with the same project name would add to the old leaderboard, so start clean.
    h2o.remove_all()
    train = to_h2o(X, y, folds)
    test = to_h2o(X_test, y_test)

    settings = {"max_models": H2O_MAX_MODELS, "seed": SEED, "sort_metric": "RMSE"}

    with mlflow_run(run, {"tool": TOOL, "feature_set": feature_set}):
        mlflow.log_params({**settings, "fold_column": FOLD_COL, "n_features": len(features),
                           "features": ",".join(features)[:500], "n_train": len(X),
                           "n_test": len(X_test)})

        aml = H2OAutoML(project_name=run, verbosity="warn", **settings)
        start = time.perf_counter()
        aml.train(x=features, y=TARGET, training_frame=train, fold_column=FOLD_COL)
        search_seconds = time.perf_counter() - start

        lb = get_leaderboard(aml, extra_columns="ALL").as_data_frame()
        leaderboard = pd.DataFrame({
            "model": lb["model_id"],
            "algorithm": lb["algo"],
            "cv_rmse": lb["rmse"],
            "cv_mae": lb["mae"],
            # A stacked ensemble's time covers only its metalearner, not its base models.
            "fit_seconds": lb["training_time_ms"] / 1000,
            "predict_ms_per_row": lb["predict_time_per_row_ms"],
        }, columns=LEADERBOARD_COLUMNS)
        write_table(leaderboard, f"{run}_leaderboard")

        leader = aml.leader
        perf = leader.model_performance(test)
        test_metrics = regression_metrics(y_test, leader.predict(test).as_data_frame()["predict"])
        # Cross-check against H2O's own scoring of the same frame.
        test_metrics["h2o_rmse"] = float(perf.rmse())

        # Stacked ensembles have no variable importance (H2O lab, section 14),
        # so the ranking comes from the best model that has one.
        importance, importance_model = None, None
        for model_id, algo in zip(lb["model_id"], lb["algo"]):
            if algo == "StackedEnsemble":
                continue
            model = h2o.get_model(model_id)
            importance = importance_by_feature(model, features)
            if importance is not None:
                importance_model = model
                break
        if importance is not None:
            write_table(importance, f"{run}_importance")

        # A seeded random sample: athletes.csv is ordered by gender, so the first rows
        # are all men and would hide gender from the SHAP and partial dependence plots.
        sample = X.sample(n=min(EXPLAIN_SAMPLE, len(X)), random_state=SEED)
        explain_frame = to_h2o(sample, y.loc[sample.index])
        save_plot(lambda: aml.varimp_heatmap(), f"{run}_varimp_heatmap")
        save_plot(lambda: aml.model_correlation_heatmap(explain_frame), f"{run}_model_correlation")
        if importance_model is not None:
            save_plot(lambda: importance_model.shap_summary_plot(explain_frame), f"{run}_shap_summary")
            for feature in importance["feature"].head(3).astype(str):
                save_plot(lambda f=str(feature): importance_model.pd_plot(explain_frame, column=f),
                          f"{run}_pdp_{feature}")

        mlflow.log_metrics({**leaderboard_metrics(leaderboard.iloc[0]),
                            "search_seconds": search_seconds,
                            **{f"test_{k}": v for k, v in test_metrics.items()}})
        mlflow.set_tag("leader", leader.model_id)

        # H2O-3 OSS from 3.46.0.12 blocks MOJO export, so fall back to H2O's binary
        # model format, which reloads with h2o.load_model on the same H2O version.
        model_dir = MODELS / run
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_path = leader.download_mojo(path=str(model_dir))
        except Exception as exc:
            print(f"MOJO export unavailable ({exc}); saving the binary model.")
            try:
                model_path = h2o.save_model(leader, path=str(model_dir), force=True)
            except Exception as exc2:
                print(f"Model export skipped: {exc2}")
                model_path = None
        if model_path:
            mlflow.log_artifact(model_path, artifact_path="model")

        summary = {
            "run": run,
            "best_model": leader.model_id,
            "best_algorithm": lb["algo"].iloc[0],
            "best_cv_rmse": float(leaderboard["cv_rmse"].iloc[0]),
            "search_seconds": search_seconds,
            "max_models": H2O_MAX_MODELS,
            "n_models": len(lb),
            "test": test_metrics,
            "features": features,
            "importance_model": importance_model.model_id if importance_model else None,
            "top_features": importance["feature"].head(5).tolist() if importance is not None else [],
        }
        write_json(summary, f"{run}_summary")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    args = parser.parse_args()
    print(run_h2o(args.features))
