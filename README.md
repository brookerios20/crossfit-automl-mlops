# CrossFit AutoML MLOps pipeline

This repository runs AutoML on the CrossFit Games athletes dataset (`athletes.csv`) to predict each athlete's `total_lift`, the sum of their clean and jerk, snatch, deadlift, and back squat maxes. It runs FLAML inside a Prefect and MLflow pipeline, repeats the search with H2O AutoML, and compares both against the hand-built model from Assignment 2 ([crossfit-feature-store-mlops-pipeline](https://github.com/brookerios20/crossfit-feature-store-mlops-pipeline)).

Assignment 3 for ADSP 32021 Machine Learning Operations. The answers to each question are in [REPORT.md](REPORT.md).

## Goals

The brief asks for the following:

1. Load the athletes dataset and use the platform's AutoML to find the best model.
2. Report data insights from the AutoML run and the top 5 features.
3. Report the top 3 models by validation score and the top 3 by speed, with all features and with the top 3 features.
4. Compare the top models with the earlier model on validation score and speed.
5. Classify the platform's AutoML as no-code, low-code, or full-code.
6. Repeat the AutoML steps with H2O AutoML.

## Stack

| Role | Tool |
|---|---|
| Orchestration | Prefect flow (`pipeline.py`) |
| Experiment tracking | MLflow, with a SQLite backend (`mlflow.db`) |
| AutoML on the platform | FLAML, logging to MLflow |
| Second AutoML tool | H2O AutoML |
| Baseline | scikit-learn `HistGradientBoostingRegressor`, the Assignment 2 run `v2_hp_a` |

## Pipeline

```
athletes.csv
   │  src/dataset.py    clean (Assignment 2 rules), add target, hash test split and 5 CV folds
   ▼
modeling table ──► src/baseline.py       Assignment 2 model, same folds and test set
              ├──► src/flaml_automl.py   all features ─► feature ranking ─► top 3 features
              └──► src/h2o_automl.py     all features ─► feature ranking ─► top 3 features
                                   │
                                   ▼  every run logged to MLflow experiment "athletes_automl"
                         src/compare.py ──► results/summary.md
```

### Data

The cleaning, target, and test split match Assignment 2, so the test set holds the same athletes. The all-features set is the Assignment 2 v2 feature set of 34 columns: age, height, weight, gender, region, BMI, and one 0/1 column per survey answer. The four lift columns are never features, because their sum is the target.

### Validation

Every model is validated with 5-fold cross-validation on the training split, using the same folds. The folds come from the same `athlete_id` hash as the test split. FLAML receives them as groups (`split_type="group"`), H2O as a `fold_column`, and the baseline through `GroupKFold`. The metric is RMSE, as in Assignment 2. FLAML and the baseline average the RMSE of the five folds. H2O computes RMSE over the pooled out-of-fold predictions, which can give a slightly different value for the same model.

The 20% test split is scored once per best model and never used for model selection or feature ranking.

### Top-3 feature runs

Each tool's all-features run writes a feature ranking, and its top-3 run trains on the 3 highest-ranked features. FLAML ranks by permutation importance (RMSE increase when a feature is shuffled) on 5,000 training rows. H2O ranks by the variable importance of its best non-ensemble model, because stacked ensembles report none.

### Speed

Speed is training time on the full training split, with prediction time per row reported alongside. H2O reports both for every model. FLAML records no comparable time per learner, so `src/flaml_automl.py` refits each learner's best configuration and times it. A stacked ensemble's H2O training time covers only its metalearner.

### Budgets

| Tool | Budget | Setting |
|---|---|---|
| FLAML | 300 s per run | `FLAML_TIME_BUDGET` |
| H2O | 20 models per run, plus stacked ensembles | `H2O_MAX_MODELS` |

Both settings live in `src/config.py` and can be overridden with environment variables of the same name.

## How to run

The project needs Python 3.10 or later, a dedicated environment, and a Java runtime for H2O, which `java -version` confirms. On macOS, XGBoost and LightGBM also need the OpenMP runtime from `brew install libomp`. On Apple Silicon, install it with the native Homebrew in `/opt/homebrew`, because arm64 Python cannot load the x86_64 build from an Intel Homebrew. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the course-provided `athletes.csv` at `data/athletes.csv`, or point to it with `export ATHLETES_CSV=/path/to/athletes.csv`. The file is about 70 MB and is excluded from git.

Run the whole pipeline:

```bash
python pipeline.py
```

Open the MLflow UI at http://127.0.0.1:5001 to browse the runs. The command uses port 5001 because macOS AirPlay Receiver occupies port 5000.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

The steps can also run one at a time. Each top-3 run needs its all-features run first.

```bash
python -m src.baseline
python -m src.flaml_automl --features all
python -m src.flaml_automl --features top3
python -m src.h2o_automl --features all
python -m src.h2o_automl --features top3
python -m src.compare
```

## Results

The pipeline writes these files to `results/`:

| File | Contents |
|---|---|
| `summary.md` | all comparison tables in one page |
| `top5_features.csv` | top 5 features per tool |
| `top3_by_score.csv` | top 3 models by CV RMSE per tool and feature set |
| `top3_by_speed.csv` | top 3 models by fit time per tool and feature set |
| `vs_baseline.csv` | best model of each run against the Assignment 2 model, with CV and test metrics |
| `{run}_leaderboard.csv` | full leaderboard of one run |
| `{run}_importance.csv` | feature ranking of one run |
| `{run}_summary.json` | best model, configuration, search time, and test metrics of one run |
| `h2o_*.png` | H2O explain plots: variable importance heatmap, model correlation, SHAP summary, partial dependence |
| `screenshots/` | MLflow UI screenshots |

Runs are named `baseline_all`, `flaml_all`, `flaml_top3`, `h2o_all`, and `h2o_top3`.

## Repository layout

```
├── pipeline.py          Prefect flow
├── requirements.txt
├── REPORT.md            answers to the assignment questions
├── data/                athletes.csv goes here
├── results/             generated tables, plots, and screenshots
└── src/
    ├── config.py        paths, folds, budgets, baseline hyperparameters
    ├── data_prep.py     cleaning and feature engineering from Assignment 2
    ├── dataset.py       modeling table, feature sets, split
    ├── reporting.py     metrics, permutation importance, result files, MLflow runs
    ├── baseline.py      Assignment 2 model on the shared folds
    ├── flaml_automl.py  FLAML runs
    ├── h2o_automl.py    H2O runs
    └── compare.py       comparison tables
```

## Assisting sources

Claude Opus 5.5 (Anthropic).
