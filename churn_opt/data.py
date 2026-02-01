# churn_opt/data.py
from __future__ import annotations

import pandas as pd


def load_telco_csv(path: str) -> pd.DataFrame:
    """
    Load and clean the Telco Customer Churn dataset.

    Steps:
    - Read CSV
    - Fix TotalCharges (convert to numeric, coerce errors)
    - Convert Churn to binary target (Yes/No -> 1/0)
    - Drop customerID
    - Fix missing TotalCharges when tenure == 0
    """

    df = pd.read_csv(path)

    # Fix TotalCharges: in this dataset it is often string with blanks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target: Churn Yes/No -> 1/0
    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain 'Churn' column")

    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    # Drop customerID (identifier)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle missing TotalCharges (usually tenure == 0)
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
        df.loc[mask, "TotalCharges"] = 0.0

    return df
