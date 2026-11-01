"""Combine the run outputs into the tables the brief asks for.

Writes results/summary.md with one section per question, and a CSV for each table:
- top5_features: top 5 features per tool (all-features runs)
- top3_by_score: top 3 models by CV RMSE per tool and feature set
- top3_by_speed: top 3 models by fit time per tool and feature set
- vs_baseline: best FLAML and H2O models against the Assignment 2 model
"""

import pandas as pd

from src.config import FEATURE_SETS, N_FOLDS, RESULTS, TOP_K
from src.reporting import read_json, write_table

TOOLS = ("flaml", "h2o")
RANK_COLUMNS = ["tool", "feature_set", "rank", "model", "algorithm", "cv_rmse", "cv_mae",
                "fit_seconds", "predict_ms_per_row"]


def _leaderboard(run: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{run}_leaderboard.csv")


def _top3(by: str) -> pd.DataFrame:
    frames = []
    for tool in TOOLS:
        for feature_set in FEATURE_SETS:
            lb = _leaderboard(f"{tool}_{feature_set}").dropna(subset=[by])
            top = lb.sort_values(by, ignore_index=True).head(3)
            top.insert(0, "rank", range(1, len(top) + 1))
            top.insert(0, "feature_set", feature_set)
            top.insert(0, "tool", tool)
            frames.append(top[RANK_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _top5_features() -> pd.DataFrame:
    frames = []
    for tool in TOOLS:
        imp = pd.read_csv(RESULTS / f"{tool}_all_importance.csv").head(5)
        imp.insert(0, "rank", range(1, len(imp) + 1))
        imp.insert(0, "tool", tool)
        frames.append(imp)
    return pd.concat(frames, ignore_index=True)


def _best_row(run: str) -> dict:
    """The run's best model with its CV score, speed, and test metrics."""
    summary = read_json(f"{run}_summary")
    lb = _leaderboard(run)
    best = lb[lb["model"] == summary["best_model"]].iloc[0]
    return {"run": run, "model": best["model"], "algorithm": best["algorithm"],
            "cv_rmse": best["cv_rmse"], "test_rmse": summary["test"]["rmse"],
            "test_mae": summary["test"]["mae"], "test_r2": summary["test"]["r2"],
            "fit_seconds": best["fit_seconds"], "predict_ms_per_row": best["predict_ms_per_row"],
            "search_seconds": summary.get("search_seconds")}


def _vs_baseline() -> pd.DataFrame:
    runs = ["baseline_all"] + [f"{tool}_{fs}" for tool in TOOLS for fs in FEATURE_SETS]
    return pd.DataFrame([_best_row(run) for run in runs])


def compare() -> pd.DataFrame:
    tables = {
        "top5_features": ("Top 5 features (all-features runs)", _top5_features()),
        "top3_by_score": ("Top 3 models by CV RMSE", _top3("cv_rmse")),
        "top3_by_speed": ("Top 3 models by fit time", _top3("fit_seconds")),
        "vs_baseline": ("Best models against the Assignment 2 model", _vs_baseline()),
    }
    sections = ["# Results summary\n",
                f"CV metrics use the same {N_FOLDS}-fold split of the training set. "
                f"Test metrics use the held-out 20% test split. The top-{TOP_K} runs use the "
                "features ranked highest by the same tool's all-features run.\n"]
    for name, (title, df) in tables.items():
        write_table(df, name, log=False)
        sections.append(f"## {title}\n\n{df.to_markdown(index=False, floatfmt='.4g')}\n")
    (RESULTS / "summary.md").write_text("\n".join(sections))
    return tables["vs_baseline"][1]


if __name__ == "__main__":
    print(compare().to_string())
