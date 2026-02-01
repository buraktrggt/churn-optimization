# churn_opt/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------
# Paths (repo-root based)
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

FINAL_RECO_PATH = OUT_TABLES / "final_recommendation.csv"
TARGETING_FULL_PATH = OUT_TABLES / "targeting_list_lr.csv"

# If True, API will refuse to start unless required CSVs exist.
FAIL_FAST_ON_MISSING_FILES = False

app = FastAPI(title="Churn Profit Decision API", version="1.0")


# -----------------------
# Schemas
# -----------------------
class ScoreCustomerRequest(BaseModel):
    scenario: str
    features: Dict[str, Any]
    # features kept for realistic contract; this service uses precomputed scores.


class ScoreCustomerResponse(BaseModel):
    customerID: str
    scenario: str
    recommended_policy: str
    decision_parameter: float
    score: float
    decision: bool


class PoliciesResponse(BaseModel):
    policies: List[Dict[str, Any]]


class ScenariosResponse(BaseModel):
    scenarios: List[str]


class CustomerLookupResponse(BaseModel):
    scenario: str
    customerID: str
    score: float
    decision: bool


# -----------------------
# Utilities
# -----------------------
def _required_files() -> List[Path]:
    return [FINAL_RECO_PATH, TARGETING_FULL_PATH]


def _load_final_reco() -> pd.DataFrame:
    if not FINAL_RECO_PATH.exists():
        raise FileNotFoundError(
            f"{FINAL_RECO_PATH} not found. Ensure outputs are generated under repo_root/outputs/tables "
            f"(run scripts/run_pipeline.py from the repo root)."
        )
    return pd.read_csv(FINAL_RECO_PATH)


def _load_targeting_full() -> pd.DataFrame:
    if not TARGETING_FULL_PATH.exists():
        raise FileNotFoundError(
            f"{TARGETING_FULL_PATH} not found. Ensure outputs are generated under repo_root/outputs/tables "
            f"(run scripts/run_pipeline.py from the repo root)."
        )
    return pd.read_csv(TARGETING_FULL_PATH)


def _ensure_columns(df: pd.DataFrame, required: List[str], filename: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"{filename} missing columns: {missing}"
        )


# -----------------------
# Startup checks
# -----------------------
@app.on_event("startup")
def _startup_checks() -> None:
    missing = [p for p in _required_files() if not p.exists()]
    if missing:
        msg = f"Missing required files: {', '.join(str(p) for p in missing)}"
        if FAIL_FAST_ON_MISSING_FILES:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")


# -----------------------
# Endpoints
# -----------------------
@app.get("/")
def root():
    return {
        "service": "Churn Profit Decision API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "recommend_policy": "/recommend_policy",
            "scenarios": "/scenarios",
            "customer_lookup": "/customer_lookup?scenario=...&customerID=...",
            "score_customer": "POST /score_customer",
        },
        "required_files": {
            "final_recommendation": str(FINAL_RECO_PATH),
            "targeting_list": str(TARGETING_FULL_PATH),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend_policy", response_model=PoliciesResponse)
def recommend_policy():
    """
    Returns the policy table (final_recommendation.csv).
    """
    try:
        df = _load_final_reco()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"policies": df.to_dict(orient="records")}


@app.get("/scenarios", response_model=ScenariosResponse)
def list_scenarios():
    """
    Lists available scenarios from final_recommendation.csv.
    """
    try:
        df = _load_final_reco()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _ensure_columns(df, ["scenario"], "final_recommendation.csv")
    scenarios = sorted(df["scenario"].dropna().astype(str).unique().tolist())
    return {"scenarios": scenarios}


@app.get("/customer_lookup", response_model=CustomerLookupResponse)
def customer_lookup(scenario: str, customerID: str):
    """
    Convenience GET helper:
    /customer_lookup?scenario=BASE&customerID=0002-ORFBO
    """
    try:
        df_targets = _load_targeting_full()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _ensure_columns(df_targets, ["scenario", "customerID", "score", "decision"], "targeting_list_lr.csv")

    rows = df_targets[
        (df_targets["scenario"].astype(str) == str(scenario)) &
        (df_targets["customerID"].astype(str) == str(customerID))
    ]

    if len(rows) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customerID}' not found in targeting list for scenario '{scenario}'"
        )

    r = rows.iloc[0]
    return {
        "scenario": str(scenario),
        "customerID": str(customerID),
        "score": float(r["score"]),
        "decision": bool(r["decision"]),
    }


@app.post("/score_customer", response_model=ScoreCustomerResponse)
def score_customer(req: ScoreCustomerRequest):
    """
    Given a scenario and a customerID inside features, returns:
    - the recommended policy (threshold)
    - the precomputed score
    - the decision (True/False)

    NOTE:
    This API uses the precomputed targeting_list_lr.csv.
    It demonstrates a production-like decision service without retraining.
    """
    try:
        df_targets = _load_targeting_full()
        df_policy = _load_final_reco()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate columns
    _ensure_columns(df_policy, ["scenario", "recommended_policy", "decision_parameter"], "final_recommendation.csv")
    _ensure_columns(df_targets, ["scenario", "customerID", "score", "decision"], "targeting_list_lr.csv")

    scenario = str(req.scenario)

    if "customerID" not in req.features:
        raise HTTPException(status_code=400, detail="features must include customerID")

    customer_id = str(req.features["customerID"])

    # 1) Policy for scenario
    pol = df_policy[df_policy["scenario"].astype(str) == scenario]
    if len(pol) == 0:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found")

    pol_row = pol.iloc[0]
    recommended_policy = str(pol_row["recommended_policy"])
    decision_parameter = float(pol_row["decision_parameter"])

    # 2) Customer row for scenario
    rows = df_targets[
        (df_targets["scenario"].astype(str) == scenario) &
        (df_targets["customerID"].astype(str) == customer_id)
    ]

    if len(rows) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' not found in targeting list for scenario '{scenario}'"
        )

    r = rows.iloc[0]

    return ScoreCustomerResponse(
        customerID=customer_id,
        scenario=scenario,
        recommended_policy=recommended_policy,
        decision_parameter=decision_parameter,
        score=float(r["score"]),
        decision=bool(r["decision"]),
    )
