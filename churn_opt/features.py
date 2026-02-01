# churn_opt/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Telco churn project.

    Creates:
    - is_new_customer: tenure < 6
    - is_month_to_month: Contract == 'Month-to-month'
    - num_services: count of subscribed services (binary aggregation)
    - has_security_bundle: OnlineSecurity OR TechSupport

    Notes:
    - Does NOT drop columns. (Dropping TotalCharges is handled in the main pipeline.)
    - Keeps the logic consistent with your working script.
    """
    df = df.copy()

    # is_new_customer
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] < 6).astype(int)

    # is_month_to_month
    if "Contract" in df.columns:
        df["is_month_to_month"] = (
            df["Contract"].astype(str).str.strip().eq("Month-to-month")
        ).astype(int)

    # Helper: map various service columns into binary indicators
    def to_yes_no_bin(series: pd.Series, col: str) -> pd.Series:
        s = series.astype(str).str.strip()
        if col == "InternetService":
            # DSL/Fiber optic -> 1, No -> 0
            return (s != "No").astype(int)
        if col == "MultipleLines":
            # 'No phone service' treated as 0
            return (s == "Yes").astype(int)
        # Most service columns are Yes/No
        return (s == "Yes").astype(int)

    # num_services: count Yes among typical service columns if present
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present_service_cols = [c for c in service_cols if c in df.columns]

    if present_service_cols:
        bins = [to_yes_no_bin(df[c], c) for c in present_service_cols]
        df["num_services"] = np.sum(np.vstack(bins), axis=0).astype(int)

    # has_security_bundle: OnlineSecurity OR TechSupport
    if ("OnlineSecurity" in df.columns) or ("TechSupport" in df.columns):
        sec = to_yes_no_bin(df["OnlineSecurity"], "OnlineSecurity") if "OnlineSecurity" in df.columns else 0
        tech = to_yes_no_bin(df["TechSupport"], "TechSupport") if "TechSupport" in df.columns else 0
        df["has_security_bundle"] = ((sec + tech) > 0).astype(int)

    return df
