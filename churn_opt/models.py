# churn_opt/models.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def _dense_onehot_encoder() -> OneHotEncoder:
    """
    Create OneHotEncoder that outputs dense arrays.
    HistGradientBoosting requires dense input in most practical setups.

    Compatible across sklearn versions:
    - new: sparse_output=False
    - old: sparse=False
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (numeric_cols, categorical_cols) based on dtypes.
    Assumes df includes 'Churn' column, which will be excluded automatically.
    """
    X = df.drop(columns=["Churn"])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(df: pd.DataFrame, model_type: str, random_state: int = 42) -> Pipeline:
    """
    Build preprocessing + model pipeline.

    model_type:
      - 'lr'  : Logistic Regression (class_weight='balanced')
      - 'hgb' : HistGradientBoostingClassifier
    """
    numeric_cols, categorical_cols = get_feature_lists(df)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", _dense_onehot_encoder(), categorical_cols),
        ],
        remainder="drop",
    )

    if model_type == "lr":
        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=0.0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lr' or 'hgb'.")

    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
