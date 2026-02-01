# Profit-Optimized Customer Churn Targeting

A **business-driven churn modeling project** that optimizes **who to target and how many customers to contact** based on **expected incremental profit**, not just prediction accuracy.

This project reframes churn prediction from a classification task into a **decision-making problem under economic constraints**.

---

## Problem Definition

Predicting churn alone does not create business value.

Retention actions (discounts, incentives, outreach) have **real costs**, and budgets are limited.  
Targeting too many customers can reduce overall profit, even with a highly accurate model.

This project answers:

- Which customers should be targeted?
- How many customers should be contacted?
- Under which assumptions does retention become profitable?

---

## Decision Framework

For each customer, the model computes an **expected profit score**:

- **Churn probability**: `P(churn)`
- **Customer lifetime value (CLV proxy)**  
  `CLV = MonthlyCharges × margin × expected_lifetime`
- **Intervention cost**  
  `Cost = MonthlyCharges × (discount_rate + contact_cost_rate)`
- **Expected incremental profit**  
  `ExpectedProfit = P(churn) × save_rate × CLV − Cost`

Customers are ranked by expected profit, and only **economically viable segments** are targeted.

This enables:
- Optimal threshold selection
- Top‑K targeting based on value, not risk alone
- Scenario-based decision analysis

---

## Dataset

**IBM Telco Customer Churn Dataset**

- ~7,000 customers
- Subscription, contract, usage, and billing features
- Binary churn label

Expected location:
```
data/raw/Telco-Customer-Churn.csv
```

### Download dataset

```bash
python scripts/download_dataset.py
```

---

## Modeling Approach

1. Data cleaning and feature engineering  
2. Train / validation / test split  
3. Models:
   - Logistic Regression (probability calibrated)
   - Histogram Gradient Boosting
4. Probability calibration
5. Profit-based evaluation across multiple business scenarios

Evaluation focuses on:
- Expected profit
- Profit lift vs. no-action baseline
- Optimal targeting size

Traditional metrics (ROC‑AUC, PR‑AUC) are tracked for model sanity, not as final objectives.

---

## Project Structure

```
churn-optimization/
├── churn_opt/          # core logic: data, features, models, profit evaluation
├── scripts/
│   ├── download_dataset.py
│   └── run_pipeline.py
├── data/
│   └── raw/
├── outputs/
│   └── tables/
├── tests/
└── README.md
```

---

## Running the Pipeline

```bash
python scripts/run_pipeline.py --csv data/raw/Telco-Customer-Churn.csv
```

The pipeline:
- trains models
- evaluates profit under multiple scenarios
- exports decision tables and targeting recommendations as CSV

---

## Output Artifacts

Generated outputs include:
- Model comparison tables
- Scenario-based profit results
- Final recommended targeting policy
- Exportable customer targeting lists
---

## How This Would Be Used in Practice

1. Score active customers on a recurring schedule
2. Compute expected profit using current business assumptions
3. Select the optimal targeting threshold
4. Export selected customers to CRM or campaign tools
5. Monitor realized profit and recalibrate assumptions

---

## Scope and Assumptions

Business parameters (margin, save rate, costs) are **scenario assumptions**, not fixed truths.

In a real organization, these values would be calibrated with finance, marketing, and retention teams.

---

## Why This Project Matters

Most churn projects stop at predicting churn.

This project demonstrates how to:
- translate predictions into decisions
- align modeling with business objectives
- design ML systems for real-world deployment
