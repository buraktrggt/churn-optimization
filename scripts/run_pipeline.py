# scripts/run_pipeline.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from churn_opt.data import load_telco_csv
from churn_opt.features import add_features
from churn_opt.experiment import run_one_model


def prepare_splits(csv_path: str, random_state: int = 42):
    raw = pd.read_csv(csv_path)
    id_col = None
    for c in ["customerID", "customerId", "CustomerID", "CustomerId", "customer_id"]:
        if c in raw.columns:
            id_col = c
            break

    # 2) Asıl loader ile işlenmiş df'yi al
    df = load_telco_csv(csv_path)

    # 3) ID load_telco_csv içinde düşmüşse geri ekle
    if id_col is not None and "customerID" not in df.columns:
        # Aynı satır sırasını koruduğunu varsayıyoruz (loader satır silmiyorsa)
        if len(raw) != len(df):
            raise ValueError(
                f"Row count mismatch: raw={len(raw)} vs loaded={len(df)}. "
                "load_telco_csv may be filtering rows; need a merge key."
            )
        df.insert(0, "customerID", raw[id_col].values)
    elif "customerID" in df.columns:
        # varsa normalize et
        pass
    else:
        # raw'da da yoksa: dosyan seninkinde gerçekten ID içermiyor demektir
        # yine de pipeline çalışsın diye sentinel id üretelim
        df.insert(0, "customerID", np.arange(len(df)).astype(str))

    # Feature engineering
    df = add_features(df)

    # add_features tekrar düşürdüyse geri ekle (koruma)
    if "customerID" not in df.columns:
        # raw ile aynı uzunluk garanti ise yeniden eklenebilir
        # burada en güvenlisi: df baştan ID ile geldiği için bu noktaya düşmemeli
        df.insert(0, "customerID", raw[id_col].values if id_col is not None else np.arange(len(df)).astype(str))

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



def save_outputs(res_lr: pd.DataFrame, res_hgb: pd.DataFrame, targeting_lr: pd.DataFrame | None = None) -> None:
    # Always save to repo-root/outputs/tables regardless of where script is executed from
    project_root = Path(__file__).resolve().parents[1]
    out_tables = project_root / "outputs" / "tables"
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

    # Customer-level targeting list
    if targeting_lr is not None and len(targeting_lr) > 0:
        targeting_lr.to_csv(out_tables / "targeting_list_lr.csv", index=False)

    print("\nSaved outputs to:")
    print(f"  {out_tables / 'results_lr.csv'}")
    print(f"  {out_tables / 'results_hgb.csv'}")
    print(f"  {out_tables / 'best_policy_compare.csv'}")
    print(f"  {out_tables / 'final_recommendation.csv'}")
    if targeting_lr is not None and len(targeting_lr) > 0:
        print(f"  {out_tables / 'targeting_list_lr.csv'}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(Path("C:/Users/Burakk/Desktop/churn-optimization/data/raw/Telco-Customer-Churn.csv")),
        help="Path to Telco-Customer-Churn.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = args.csv
    random_state = args.seed

    print("\nLoading data and preparing splits...")
    train_df, val_df, test_df = prepare_splits(csv_path, random_state=random_state)

    # Run LR (and get targeting list)
    res_lr, targeting_lr = run_one_model(
        model_type="lr",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        do_calibration=True,
        calibration_method="sigmoid",
        random_state=random_state,
        return_scored=True,
    )

    # Run HGB (aggregate only)
    res_hgb = run_one_model(
        model_type="hgb",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        do_calibration=True,
        calibration_method="sigmoid",
        random_state=random_state,
        return_scored=False,
    )

    save_outputs(res_lr, res_hgb, targeting_lr=targeting_lr)


if __name__ == "__main__":
    main()
