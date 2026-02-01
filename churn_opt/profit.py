# churn_opt/profit.py
from __future__ import annotations

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from .scenarios import Scenario


def compute_expected_profit_per_customer(
    p_churn: np.ndarray,
    monthly_charges: np.ndarray,
    scenario: Scenario,
) -> np.ndarray:
    """
    Expected incremental profit IF we target the customer:
      EP = P(churn) * save_rate * CLV - cost
      CLV = MonthlyCharges * margin * expected_months
      cost = MonthlyCharges * (discount_rate + contact_rate)
    """
    clv = monthly_charges * scenario.margin * scenario.expected_months
    cost = monthly_charges * (scenario.discount_rate + scenario.contact_rate)
    return p_churn * scenario.save_rate * clv - cost


def profit_from_targeting(
    y_true: np.ndarray,
    targeted: np.ndarray,
    monthly_charges: np.ndarray,
    scenario: Scenario,
) -> float:
    """
    Incremental profit vs 'do nothing' baseline.

    - If targeted:
        pay cost always
        if true churner (y_true==1), expected saved value = save_rate * CLV
    - If not targeted:
        incremental profit = 0
    """
    clv = monthly_charges * scenario.margin * scenario.expected_months
    cost = monthly_charges * (scenario.discount_rate + scenario.contact_rate)

    targeted = targeted.astype(bool)
    y_true = y_true.astype(int)

    inc = 0.0
    inc += (-cost[targeted]).sum()
    inc += (scenario.save_rate * clv[targeted & (y_true == 1)]).sum()
    return float(inc)


def select_score_threshold_by_profit(
    score: np.ndarray,
    y_true: np.ndarray,
    monthly_charges: np.ndarray,
    scenario: Scenario,
    quantiles: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Select a score threshold s to target customers where score >= s,
    maximizing incremental profit on the given split (typically validation).

    Threshold candidates are generated from score quantiles for speed/stability.
    Also includes 0.0 to represent "only target positive expected profit".
    """
    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, 91)  # 5%..95%

    candidates = np.quantile(score, quantiles)
    candidates = np.unique(np.concatenate([candidates, np.array([0.0])]))

    best_s = 0.0
    best_profit = -np.inf

    for s in candidates:
        targeted = (score >= s)
        prof = profit_from_targeting(y_true, targeted, monthly_charges, scenario)
        if prof > best_profit:
            best_profit = prof
            best_s = float(s)

    return best_s, float(best_profit)


def evaluate_score_topk_grid(
    score: np.ndarray,
    y_true: np.ndarray,
    monthly_charges: np.ndarray,
    scenario: Scenario,
    ks: List[float],
) -> pd.DataFrame:
    """
    Evaluate incremental profit for Top-K targeting based on the profit score.
    Returns a DataFrame sorted by profit descending.
    """
    n = len(score)
    order = np.argsort(-score)  # descending
    rows = []

    for k in ks:
        m = max(1, int(round(n * k)))
        targeted = np.zeros(n, dtype=bool)
        targeted[order[:m]] = True

        prof = profit_from_targeting(y_true, targeted, monthly_charges, scenario)
        rows.append(
            {"k": float(k), "profit": float(prof), "target_rate": float(targeted.mean())}
        )

    return pd.DataFrame(rows).sort_values("profit", ascending=False)
