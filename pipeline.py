"""End-to-end Prefect flow: baseline -> FLAML (all, top 3) -> H2O (all, top 3) -> compare.

Run from the repo root:
    python pipeline.py
"""

from prefect import flow, task

from src.baseline import run_baseline
from src.compare import compare
from src.config import FEATURE_SETS
from src.flaml_automl import run_flaml
from src.h2o_automl import run_h2o


@task
def baseline_task() -> dict:
    """Rescore the Assignment 2 model on the shared folds and test set."""
    return run_baseline()


@task
def flaml_task(feature_set: str) -> dict:
    return run_flaml(feature_set)


@task
def h2o_task(feature_set: str) -> dict:
    return run_h2o(feature_set)


@task
def compare_task():
    return compare()


@flow(name="crossfit-automl-pipeline", log_prints=True)
def pipeline() -> None:
    print(baseline_task())
    # Runs go one at a time: each top-3 run reads the ranking its all-features
    # run wrote, and the timing columns assume an otherwise idle machine.
    for feature_set in FEATURE_SETS:
        print(flaml_task(feature_set))
    for feature_set in FEATURE_SETS:
        print(h2o_task(feature_set))
    print(compare_task().to_string())


if __name__ == "__main__":
    pipeline()
