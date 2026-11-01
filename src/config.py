"""Paths, budgets, and settings shared by every pipeline step."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = Path(os.environ.get("ATHLETES_CSV", ROOT / "data" / "athletes.csv"))
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = "athletes_automl"

ID_COL = "athlete_id"
TARGET = "total_lift"
SPLIT_COL = "split"
FOLD_COL = "fold"
SEED = 42

# Every model (baseline, FLAML, H2O) is validated on the same 5 folds of the training split.
N_FOLDS = 5

# The brief: "only the top features (if you have to choose a number put in 3)".
TOP_K = 3
FEATURE_SETS = ("all", f"top{TOP_K}")

# FLAML stops on a time budget. H2O stops on a model count, which the H2O lab
# recommends for reproducibility.
FLAML_TIME_BUDGET = int(os.environ.get("FLAML_TIME_BUDGET", 300))
H2O_MAX_MODELS = int(os.environ.get("H2O_MAX_MODELS", 20))
H2O_MAX_MEM = os.environ.get("H2O_MAX_MEM", "4G")

# Rows used for permutation importance and H2O explain plots.
EXPLAIN_SAMPLE = 5000

# The Assignment 2 model with the best test RMSE: v2 features, hyperparameter set hp_a.
BASELINE_PARAMS = {"max_iter": 100, "max_depth": 3, "learning_rate": 0.1, "early_stopping": False}
