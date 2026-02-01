# churn_opt/experiment.py
from __future__ import annotations

from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from .models import build_pipeline
from .scenarios import SCENARIOS
from .profit import (
    compute_expected_profit_per_customer,
    profit_from_targeting,
    select_score_threshold_by_profit,
    evaluate_score_topk_grid,
)


def report_metrics(split: str, y: np.ndarray, p: np.ndarray) -> None:
    roc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    print(f"\n[{split}] ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | Brier={brier:.4f}")


def _get_probs_with_optional_calibration(
    pipe,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    do_calibration: bool,
    calibration_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit base pipeline on train. Optionally fit calibrator on validation.
    Return calibrated (or raw) probabilities: (p_val, p_test).
    """
    pipe.fit(X_train, y_train)

    model_for_pred = pipe
    if do_calibration:
        calibrator = CalibratedClassifierCV(
            estimator=pipe,
            method=calibration_method,
            cv=5,
        )
        calibrator.fit(X_val, y_val)
        model_for_pred = calibrator

    p_val = model_for_pred.predict_proba(X_val)[:, 1]
    p_test = model_for_pred.predict_proba(X_test)[:, 1]
    return p_val, p_test


def _build_targeting_list_for_threshold_policy(
    test_df: pd.DataFrame,
    score_test: np.ndarray,
    threshold: float,
    scenario_name: str,
    model_tag: str,
) -> pd.DataFrame:
    """
    Build customer-level targeting list using score_threshold policy.
    score_test is the expected-profit score per customer for the given scenario.
    """
    out = pd.DataFrame({
        "scenario": scenario_name,
        "model": model_tag,
        "policy": "score_threshold",
        "decision_parameter": float(threshold),
        "score": score_test.astype(float),
        "decision": (score_test >= threshold),
    })

    # Attach customerID if present (from original test_df, not X_test)
    if "customerID" in test_df.columns:
        out.insert(0, "customerID", test_df["customerID"].values)

    # Sort: most actionable first
    out = out.sort_values(["decision", "score"], ascending=[False, False]).reset_index(drop=True)
    return out


def run_one_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    do_calibration: bool = True,
    calibration_method: str = "sigmoid",
    random_state: int = 42,
    ks_grid: Optional[List[float]] = None,
    return_scored: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Run end-to-end experiment for a given model type:
    - Train (+ optional calibration)
    - Print technical metrics
    - For each scenario:
        - print Top-K grids on VAL and TEST
        - compute policies: call_none, call_all, score_threshold, score_topk
    - Print profit tables and best policy per scenario
    Returns a results DataFrame.

    If return_scored=True, also returns a TEST customer-level targeting list for score_threshold
    (one block per scenario).
    """
    if ks_grid is None:
        ks_grid = [0.01, 0.03, 0.05, 0.10]

    print("\n" + "=" * 80)
    print(f"MODEL: {model_type.upper()} | calibration={do_calibration} ({calibration_method})")
    print("=" * 80)

    # IMPORTANT:
    # build_pipeline likely infers columns from the df you pass.
    # If customerID exists, we must remove it from the schema to prevent ColumnTransformer errors.
    schema_df = train_df.drop(columns=["customerID"], errors="ignore")
    pipe = build_pipeline(schema_df, model_type=model_type, random_state=random_state)

    # Split X/y (never feed customerID to the model)
    X_train = train_df.drop(columns=["Churn", "customerID"], errors="ignore")
    y_train = train_df["Churn"].values

    X_val = val_df.drop(columns=["Churn", "customerID"], errors="ignore")
    y_val = val_df["Churn"].values

    X_test = test_df.drop(columns=["Churn", "customerID"], errors="ignore")
    y_test = test_df["Churn"].values

    # Fit + predict probabilities
    p_val, p_test = _get_probs_with_optional_calibration(
        pipe=pipe,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test,
        do_calibration=do_calibration,
        calibration_method=calibration_method,
    )

    # Report metrics
    report_metrics("VAL", y_val, p_val)
    report_metrics("TEST", y_test, p_test)

    # Needed for profit
    if "MonthlyCharges" not in X_val.columns or "MonthlyCharges" not in X_test.columns:
        raise ValueError("MonthlyCharges column is required for profit computation.")

    mc_val = X_val["MonthlyCharges"].to_numpy()
    mc_test = X_test["MonthlyCharges"].to_numpy()

    results = []
    targeting_frames: List[pd.DataFrame] = []

    for _, scen in SCENARIOS.items():
        # Profit scores
        score_val = compute_expected_profit_per_customer(p_val, mc_val, scen)
        score_test = compute_expected_profit_per_customer(p_test, mc_test, scen)

        # Baselines on TEST (incremental profit vs do nothing)
        targeted_none = np.zeros_like(y_test, dtype=bool)
        targeted_all = np.ones_like(y_test, dtype=bool)

        test_profit_none = profit_from_targeting(y_test, targeted_none, mc_test, scen)  # should be 0
        test_profit_all = profit_from_targeting(y_test, targeted_all, mc_test, scen)

        # 1) Score-threshold selection on VAL
        best_s, val_profit_s = select_score_threshold_by_profit(score_val, y_val, mc_val, scen)
        targeted_test_s = (score_test >= best_s)
        test_profit_s = profit_from_targeting(y_test, targeted_test_s, mc_test, scen)

        # 2) Top-K grids (VAL/TEST)
        val_grid = evaluate_score_topk_grid(score_val, y_val, mc_val, scen, ks_grid)
        test_grid = evaluate_score_topk_grid(score_test, y_test, mc_test, scen, ks_grid)

        print(f"\n--- {scen.name.upper()} | VAL Top-K grid (by SCORE) ---")
        print(val_grid.to_string(index=False))
        print(f"\n--- {scen.name.upper()} | TEST Top-K grid (by SCORE) ---")
        print(test_grid.to_string(index=False))

        # Choose best K on VAL and apply on TEST
        best_k = float(val_grid.iloc[0]["k"])
        val_profit_k = float(val_grid.iloc[0]["profit"])

        n_test = len(score_test)
        order = np.argsort(-score_test)
        m = max(1, int(round(n_test * best_k)))
        targeted_test_k = np.zeros(n_test, dtype=bool)
        targeted_test_k[order[:m]] = True
        test_profit_k = profit_from_targeting(y_test, targeted_test_k, mc_test, scen)

        # Collect results
        results.append({
            "scenario": scen.name,
            "policy": "score_threshold",
            "val_choice": float(best_s),
            "val_profit": float(val_profit_s),
            "test_profit": float(test_profit_s),
            "target_rate_test": float(targeted_test_s.mean()),
        })
        results.append({
            "scenario": scen.name,
            "policy": "score_topk",
            "val_choice": float(best_k),
            "val_profit": float(val_profit_k),
            "test_profit": float(test_profit_k),
            "target_rate_test": float(targeted_test_k.mean()),
        })
        results.append({
            "scenario": scen.name,
            "policy": "call_none",
            "val_choice": np.nan,
            "val_profit": np.nan,
            "test_profit": float(test_profit_none),
            "target_rate_test": float(targeted_none.mean()),
        })
        results.append({
            "scenario": scen.name,
            "policy": "call_all",
            "val_choice": np.nan,
            "val_profit": np.nan,
            "test_profit": float(test_profit_all),
            "target_rate_test": float(targeted_all.mean()),
        })

        # Optional: customer-level targeting list for threshold policy
        if return_scored:
            targeting_frames.append(
                _build_targeting_list_for_threshold_policy(
                    test_df=test_df,  # <-- must be the original test_df that still has customerID
                    score_test=score_test,
                    threshold=float(best_s),
                    scenario_name=scen.name,
                    model_tag=model_type.upper(),
                )
            )

    res_df = pd.DataFrame(results).sort_values(["scenario", "test_profit"], ascending=[True, False])

    print("\n=== PROFIT RESULTS (chosen on VAL, reported on TEST) ===")
    print(res_df.to_string(index=False))

    print("\n=== BEST POLICY PER SCENARIO (by TEST profit) ===")
    best = (
        res_df.sort_values(["scenario", "test_profit"], ascending=[True, False])
        .groupby("scenario")
        .head(1)
    )
    print(best[["scenario", "policy", "val_choice", "test_profit", "target_rate_test"]].to_string(index=False))

    if not return_scored:
        return res_df

    targeting_df = pd.concat(targeting_frames, ignore_index=True) if targeting_frames else pd.DataFrame()
    return res_df, targeting_df
