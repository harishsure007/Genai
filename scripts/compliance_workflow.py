"""
Step 3: Compliance Workflow — Fraud Detection Integration
==========================================================
Loads the trained XGBoost model and runs it against a transaction batch to:

  1. Score every transaction (fraud probability 0–1)
  2. Assign a risk tier  HIGH / MEDIUM / REVIEW based on configurable thresholds
  3. Flag transactions that breach the HIGH or MEDIUM threshold
  4. Produce compliance outputs:
       reports/flagged_transactions.csv   — full detail on flagged rows
       reports/compliance_summary.txt     — narrative summary for compliance team
       reports/risk_score_distribution.png
       reports/flagged_by_category.png
       reports/flagged_by_country.png

Threshold rationale
-------------------
  HIGH   ≥ 0.70  → immediate review / block candidate
  MEDIUM ≥ 0.40  → watch list / enhanced due diligence
  REVIEW  < 0.40  → no immediate action; retained for audit trail

These are starting points. A compliance officer should calibrate them against
the business's cost matrix (false-positive cost vs. missed-fraud cost).
"""

import os
import textwrap
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_PATH   = "data/transactions.csv"
MODEL_PATH  = "models/fraud_model.pkl"
REPORT_DIR  = "reports"

THRESHOLD_HIGH   = 0.70   # flag for immediate review
THRESHOLD_MEDIUM = 0.40   # flag for watchlist / enhanced due diligence

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 65)
print("STEP 3: COMPLIANCE WORKFLOW — FRAUD DETECTION")
print(f"Run timestamp : {RUN_TIMESTAMP}")
print("=" * 65)

# --------------------------------------------------------------------------- #
# 1. Load Model Artifact
# --------------------------------------------------------------------------- #

print("\n[1] Loading model artifact …")
artifact        = joblib.load(MODEL_PATH)
model           = artifact["model"]
label_encoders  = artifact["label_encoders"]
feature_cols    = artifact["feature_cols"]

print(f"    Model     : {type(model).__name__}")
print(f"    Features  : {feature_cols}")
print(f"    Train AUC : {artifact['roc_auc']:.4f}")

# --------------------------------------------------------------------------- #
# 2. Load & Pre-process Transactions
# --------------------------------------------------------------------------- #

print("\n[2] Loading transaction batch …")
df_raw = pd.read_csv(DATA_PATH)
print(f"    Loaded {len(df_raw):,} transactions")

df = df_raw.copy()

# Apply the same label encoding used during training
categorical_cols = ["merchant_category", "card_type", "country"]
for col in categorical_cols:
    le: LabelEncoder = label_encoders[col]
    # Handle unseen categories gracefully by mapping to the most-frequent class
    known = set(le.classes_)
    df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
    df[col] = le.transform(df[col])

# --------------------------------------------------------------------------- #
# 3. Score Every Transaction
# --------------------------------------------------------------------------- #

print("\n[3] Scoring transactions …")
X = df[feature_cols].values
fraud_prob = model.predict_proba(X)[:, 1]

df_raw = df_raw.copy()
df_raw["fraud_probability"] = fraud_prob.round(4)

# --------------------------------------------------------------------------- #
# 4. Assign Risk Tier & Flag
# --------------------------------------------------------------------------- #

print("\n[4] Assigning risk tiers …")

def assign_tier(p: float) -> str:
    if p >= THRESHOLD_HIGH:
        return "HIGH"
    if p >= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "REVIEW"

df_raw["risk_tier"] = df_raw["fraud_probability"].apply(assign_tier)
df_raw["flagged"]   = df_raw["risk_tier"].isin(["HIGH", "MEDIUM"]).astype(int)

tier_counts = df_raw["risk_tier"].value_counts()
print(f"    HIGH   (≥{THRESHOLD_HIGH:.0%}) : {tier_counts.get('HIGH',   0):>6,}")
print(f"    MEDIUM (≥{THRESHOLD_MEDIUM:.0%}) : {tier_counts.get('MEDIUM', 0):>6,}")
print(f"    REVIEW  (<{THRESHOLD_MEDIUM:.0%}) : {tier_counts.get('REVIEW',  0):>6,}")

# --------------------------------------------------------------------------- #
# 5. Compliance Output A — Flagged Transaction Detail CSV
# --------------------------------------------------------------------------- #

flagged_df = (
    df_raw[df_raw["flagged"] == 1]
    .sort_values("fraud_probability", ascending=False)
    .reset_index(drop=True)
)

flag_path = f"{REPORT_DIR}/flagged_transactions.csv"
flagged_df.to_csv(flag_path, index=False)
print(f"\n[5] Flagged transactions saved → {flag_path}")
print(f"    {len(flagged_df):,} transactions flagged "
      f"({len(flagged_df)/len(df_raw):.1%} of batch)")

# --------------------------------------------------------------------------- #
# 6. Compliance Output B — Narrative Summary
# --------------------------------------------------------------------------- #

total        = len(df_raw)
n_flagged    = flagged_df["flagged"].sum()
n_high       = tier_counts.get("HIGH",   0)
n_medium     = tier_counts.get("MEDIUM", 0)
n_true_fraud = int(df_raw["is_fraud"].sum()) if "is_fraud" in df_raw.columns else "N/A"

# Breakdown helpers
def top_breakdown(col: str, n: int = 5) -> str:
    counts = flagged_df[col].value_counts().head(n)
    lines  = [f"      {v:<25} {c:>5,}" for v, c in counts.items()]
    return "\n".join(lines)

avg_amount_flagged = flagged_df["amount_usd"].mean() if "amount_usd" in flagged_df else "N/A"
avg_amount_all     = df_raw["amount_usd"].mean()     if "amount_usd" in df_raw     else "N/A"

summary_text = textwrap.dedent(f"""
FRAUD DETECTION COMPLIANCE SUMMARY
====================================
Generated   : {RUN_TIMESTAMP}
Thresholds  : HIGH ≥ {THRESHOLD_HIGH:.0%}  |  MEDIUM ≥ {THRESHOLD_MEDIUM:.0%}

BATCH OVERVIEW
--------------
  Total transactions reviewed   : {total:>10,}
  Known fraud labels (ground truth): {n_true_fraud!s:>8}

FLAGGED ACTIVITY
-----------------
  Total flagged                 : {n_flagged:>10,}  ({n_flagged/total:.2%} of batch)
    └─ HIGH risk                : {n_high:>10,}  ({n_high/total:.2%})
    └─ MEDIUM risk              : {n_medium:>10,}  ({n_medium/total:.2%})

FINANCIAL EXPOSURE
-------------------
  Avg transaction amount (all)     : ${avg_amount_all:>10.2f}
  Avg transaction amount (flagged) : ${avg_amount_flagged:>10.2f}
  Total value at risk (flagged)    : ${flagged_df['amount_usd'].sum():>12,.2f}
  Total value at risk (HIGH only)  : ${flagged_df[flagged_df['risk_tier']=='HIGH']['amount_usd'].sum():>12,.2f}

TOP FLAGGED MERCHANT CATEGORIES
---------------------------------
{top_breakdown('merchant_category')}

TOP FLAGGED COUNTRIES
----------------------
{top_breakdown('country')}

TOP FLAGGED CARD TYPES
-----------------------
{top_breakdown('card_type')}

RECOMMENDED ACTIONS
--------------------
  1. HIGH-risk transactions ({n_high:,}): Escalate for immediate manual review
     or automated block. Priority queue for compliance officer sign-off.

  2. MEDIUM-risk transactions ({n_medium:,}): Add to enhanced due-diligence
     watch list. Apply step-up authentication or hold for 24 h review.

  3. Re-evaluate thresholds quarterly against confirmed fraud case outcomes
     to re-calibrate precision/recall balance.

  4. For transactions flagged as HIGH but cleared post-review, retain a
     labelled feedback dataset for model retraining.

OUTPUT FILES
-------------
  Flagged detail CSV  : {flag_path}
  Risk distribution   : {REPORT_DIR}/risk_score_distribution.png
  By merchant         : {REPORT_DIR}/flagged_by_category.png
  By country          : {REPORT_DIR}/flagged_by_country.png
""").strip()

summary_path = f"{REPORT_DIR}/compliance_summary.txt"
with open(summary_path, "w") as f:
    f.write(summary_text)

print(f"\n[6] Compliance summary saved → {summary_path}")
print(summary_text)

# --------------------------------------------------------------------------- #
# 7. Compliance Output C — Visualisations
# --------------------------------------------------------------------------- #

print(f"\n[7] Generating compliance charts …")

# -- Risk Score Distribution -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_raw["fraud_probability"], bins=50, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(THRESHOLD_MEDIUM, color="orange", lw=1.5, linestyle="--", label=f"MEDIUM ≥ {THRESHOLD_MEDIUM}")
ax.axvline(THRESHOLD_HIGH,   color="red",    lw=1.5, linestyle="--", label=f"HIGH ≥ {THRESHOLD_HIGH}")
ax.set_xlabel("Fraud Probability Score")
ax.set_ylabel("Number of Transactions")
ax.set_title("Risk Score Distribution — Batch")
ax.legend()
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/risk_score_distribution.png", dpi=150)
plt.close()
print(f"    Saved risk_score_distribution.png")

# -- Flagged Count by Merchant Category --------------------------------------
cat_counts = flagged_df.groupby(["merchant_category", "risk_tier"]).size().unstack(fill_value=0)
for col in ["HIGH", "MEDIUM"]:
    if col not in cat_counts.columns:
        cat_counts[col] = 0
cat_counts = cat_counts[["HIGH", "MEDIUM"]].sort_values("HIGH", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
cat_counts.plot(kind="barh", ax=ax, color=["crimson", "orange"], edgecolor="white")
ax.set_xlabel("Number of Flagged Transactions")
ax.set_title("Flagged Transactions by Merchant Category")
ax.legend(title="Risk Tier")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/flagged_by_category.png", dpi=150)
plt.close()
print(f"    Saved flagged_by_category.png")

# -- Flagged Count by Country ------------------------------------------------
country_counts = flagged_df.groupby(["country", "risk_tier"]).size().unstack(fill_value=0)
for col in ["HIGH", "MEDIUM"]:
    if col not in country_counts.columns:
        country_counts[col] = 0
country_counts = country_counts[["HIGH", "MEDIUM"]].sort_values("HIGH", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
country_counts.plot(kind="barh", ax=ax, color=["crimson", "orange"], edgecolor="white")
ax.set_xlabel("Number of Flagged Transactions")
ax.set_title("Flagged Transactions by Country")
ax.legend(title="Risk Tier")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/flagged_by_country.png", dpi=150)
plt.close()
print(f"    Saved flagged_by_country.png")

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #

print(f"\n{'=' * 65}")
print("COMPLIANCE WORKFLOW COMPLETE")
print(f"  Batch size     : {total:,}")
print(f"  Flagged        : {n_flagged:,}  ({n_flagged/total:.1%})")
print(f"  HIGH risk      : {n_high:,}")
print(f"  MEDIUM risk    : {n_medium:,}")
print(f"  Reports in     : {REPORT_DIR}/")
print("=" * 65)
