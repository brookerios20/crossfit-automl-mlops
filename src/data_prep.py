"""Clean athletes.csv and engineer features (copied from Assignment 2).

The target is total_lift = candj + snatch + deadlift + backsq. The four lifts
are never model features because they sum to the target.
"""

import re

import numpy as np
import pandas as pd

from src.config import ID_COL, SPLIT_COL, TARGET

LIFT_COLS = ["candj", "snatch", "deadlift", "backsq"]
SURVEY_COLS = ["eat", "background", "experience", "schedule", "howlong"]
BASE_FEATURES = ["age", "height", "weight", "gender", "region"]
TEST_FRACTION = 0.20


def load_raw(path) -> pd.DataFrame:
    """Read the raw CrossFit athletes CSV."""
    return pd.read_csv(path)


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete rows and implausible values, following the course cleaning code.

    The female deadlift cap is applied per gender, so the 636 lb limit takes effect.
    """
    data = data.dropna(subset=["region", "age", "weight", "height", "howlong", "gender",
                               "eat", "train", "background", "experience", "schedule",
                               "deadlift", "candj", "snatch", "backsq"])
    data = data.drop(columns=["affiliate", "team", "name", "fran", "helen", "grace",
                              "filthy50", "fgonebad", "run400", "run5k", "pullups", "train"])

    # Remove outliers
    data = data[data["weight"] < 1500]
    data = data[data["gender"] != "--"]
    data = data[data["age"] >= 18]
    data = data[(data["height"] < 96) & (data["height"] > 48)]

    male_ok = (data["gender"] == "Male") & (data["deadlift"] <= 1105)
    female_ok = (data["gender"] == "Female") & (data["deadlift"] <= 636)
    data = data[(data["deadlift"] > 0) & (male_ok | female_ok)]
    data = data[(data["candj"] > 0) & (data["candj"] <= 395)]
    data = data[(data["snatch"] > 0) & (data["snatch"] <= 496)]
    data = data[(data["backsq"] > 0) & (data["backsq"] <= 1069)]

    # Clean survey data
    data = data.replace({"Decline to answer|": np.nan})
    data = data.dropna(subset=SURVEY_COLS)

    # The split and CV folds are assigned per athlete, so each athlete must appear once.
    data = data.dropna(subset=[ID_COL]).drop_duplicates(subset=[ID_COL]).copy()
    data[ID_COL] = data[ID_COL].astype("int64")
    return data


def add_total_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Add the regression target."""
    df = df.copy()
    df[TARGET] = df[LIFT_COLS].sum(axis=1)
    return df


def add_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> pd.DataFrame:
    """Assign train or test by hashing athlete_id, so the split is stable across reruns."""
    df = df.copy()
    bucket = pd.util.hash_pandas_object(df[ID_COL], index=False).to_numpy() % 100
    df[SPLIT_COL] = np.where(bucket < int(test_fraction * 100), "test", "train")
    return df


def _slug(text: str) -> str:
    """Make a column-name-safe slug."""
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return bmi plus one 0/1 column per survey answer, indexed like df.

    Survey answers are pipe-delimited multi-select strings, so each answer
    becomes its own column. "Decline to answer" columns are dropped.
    """
    out = pd.DataFrame(index=df.index)
    out["bmi"] = 703 * df["weight"] / df["height"] ** 2
    for col in SURVEY_COLS:
        dummies = df[col].str.get_dummies(sep="|")
        dummies.columns = [f"{col}__{_slug(c)}" for c in dummies.columns]
        dummies = dummies.T.groupby(level=0).max().T  # merge answers that slug to the same name
        dummies = dummies.drop(columns=[f"{col}__decline_to_answer"], errors="ignore")
        out = pd.concat([out, dummies.astype("int32")], axis=1)
    return out
