# scripts/run_pipeline.py
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_opt.data import load_telco_csv
from churn_opt.features import add_features
from churn_opt.experiment import run_one_model


def prepare_splits(csv_path: str, random_state: int = 42):
    df = load_telco_csv(csv_path)
    df = add_features(df)

    # Keep consistent with pipeline decision
    if "TotalCharges" in df.columns:
        df = df.drop(columns=["TotalCharges"])

    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["Churn"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, stratify=train_df["Churn"], random_state=random_state
    )
    return train_df, val_df, test_df


def save_outputs(res_lr: pd.DataFrame, res_hgb: pd.DataFrame) -> None:
    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)

    res_lr.to_csv(out_tables / "results_lr.csv", index=False)
    res_hgb.to_csv(out_tables / "results_hgb.csv", index=False)

    def best_policy_per_scenario(res: pd.DataFrame) -> pd.DataFrame:
        best = (
            res.sort_values(["scenario", "test_profit"], ascending=[True, False])
            .groupby("scenario")
            .head(1)
            .reset_index(drop=True)
        )
        return best[["scenario", "policy", "val_choice", "test_profit", "target_rate_test"]]

    best_lr = best_policy_per_scenario(res_lr).assign(model="LR")
    best_hgb = best_policy_per_scenario(res_hgb).assign(model="HGB")
    best_compare = pd.concat([best_lr, best_hgb], ignore_index=True)
    best_compare.to_csv(out_tables / "best_policy_compare.csv", index=False)

    # Final recommendation artifact: default to LR best policies
    final_reco = best_lr.drop(columns=["model"]).rename(columns={
        "policy": "recommended_policy",
        "val_choice": "decision_parameter",
        "test_profit": "expected_test_profit",
        "target_rate_test": "expected_target_rate",
    })
    final_reco.to_csv(out_tables / "final_recommendation.csv", index=False)

    print("\nSaved outputs to:")
    print(f"  {out_tables / 'results_lr.csv'}")
    print(f"  {out_tables / 'results_hgb.csv'}")
    print(f"  {out_tables / 'best_policy_compare.csv'}")
    print(f"  {out_tables / 'final_recommendation.csv'}")


def main():
    csv_path = "C:/Users/Burakk/Desktop/churn-optimization/data/raw/Telco-Customer-Churn.csv"
    random_state = 42

    print("\nLoading data and preparing splits...")
    train_df, val_df, test_df = prepare_splits(csv_path, random_state=random_state)

    # Run models
    res_lr = run_one_model(
        model_type="lr",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        do_calibration=True,
        calibration_method="sigmoid",
        random_state=random_state,
    )

    res_hgb = run_one_model(
        model_type="hgb",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        do_calibration=True,
        calibration_method="sigmoid",
        random_state=random_state,
    )

    save_outputs(res_lr, res_hgb)


if __name__ == "__main__":
    main()
