"""Build the modeling table: features, target, test split, and CV folds.

The all-features set is the Assignment 2 v2 feature set: age, height, weight,
gender, region, BMI, and one 0/1 column per survey answer. The top-k set keeps
the k features ranked highest by the same tool's all-features run.

The test split (20%) and the folds both come from hashing athlete_id, so they
match Assignment 2 and stay the same across reruns and tools.
"""

import pandas as pd

from src.config import FOLD_COL, ID_COL, N_FOLDS, RAW_CSV, RESULTS, SPLIT_COL, TARGET, TOP_K
from src.data_prep import (BASE_FEATURES, add_split, add_total_lift, clean,
                           engineered_features, load_raw)

CATEGORICAL = ["gender", "region"]
NON_FEATURES = [ID_COL, TARGET, SPLIT_COL, FOLD_COL]


def load_table(raw_csv=RAW_CSV) -> pd.DataFrame:
    """Clean the raw CSV and return one row per athlete with features, target, split, and fold."""
    df = add_split(add_total_lift(clean(load_raw(raw_csv))))

    features = pd.concat([df[BASE_FEATURES], engineered_features(df)], axis=1)
    # Survey strings end with "|", so splitting them produces an empty answer column ("eat__",
    # "background__", ...) with one value for every athlete. They carry no signal, and
    # dropping them leaves the 34 features Assignment 2 trained on.
    features = features.loc[:, features.nunique() > 1]
    features[["age", "height", "weight"]] = features[["age", "height", "weight"]].astype("float64")
    # FLAML and the baseline read the category dtype as categorical; H2O gets enum columns.
    features[CATEGORICAL] = features[CATEGORICAL].astype("category")

    table = pd.concat([df[[ID_COL, TARGET, SPLIT_COL]], features], axis=1)
    # Test rows hold hash buckets 0-19 and training rows 20-99, so bucket % 5
    # gives five training folds of 16 buckets each.
    bucket = pd.util.hash_pandas_object(df[ID_COL], index=False).to_numpy() % 100
    table[FOLD_COL] = bucket % N_FOLDS
    return table.reset_index(drop=True)


def all_features(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in NON_FEATURES]


def top_features(tool: str, k: int = TOP_K) -> list[str]:
    """Read the feature ranking written by the tool's all-features run."""
    path = RESULTS / f"{tool}_all_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the {tool} all-features run first.")
    return pd.read_csv(path)["feature"].head(k).tolist()


def feature_list(table: pd.DataFrame, tool: str, feature_set: str) -> list[str]:
    return all_features(table) if feature_set == "all" else top_features(tool)


def split(table: pd.DataFrame, features: list[str]):
    """Return X_train, y_train, folds, X_test, y_test."""
    train = table[table[SPLIT_COL] == "train"]
    test = table[table[SPLIT_COL] == "test"]
    return train[features], train[TARGET], train[FOLD_COL], test[features], test[TARGET]
