# churn_opt/scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Scenario:
    name: str
    margin: float
    expected_months: int
    discount_rate: float
    save_rate: float
    contact_rate: float


SCENARIOS: Dict[str, Scenario] = {
    "worst": Scenario("worst", margin=0.30, expected_months=6,  discount_rate=0.23, save_rate=0.05, contact_rate=0.02),
    "base":  Scenario("base",  margin=0.40, expected_months=12, discount_rate=0.18, save_rate=0.10, contact_rate=0.02),
    "best":  Scenario("best",  margin=0.60, expected_months=18, discount_rate=0.10, save_rate=0.20, contact_rate=0.02),
}
