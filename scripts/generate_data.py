"""
Step 1: Generate Synthetic Financial Transaction Data

Produces a CSV with realistic transaction attributes and a fraud label.
Fraud rate is kept low (~2%) to mirror real-world class imbalance.
"""

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NUM_TRANSACTIONS = 10_000
FRAUD_RATE = 0.02          # 2 % of transactions are fraudulent
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "online_retail", "electronics",
    "travel", "entertainment", "healthcare", "fuel", "atm_withdrawal",
]

COUNTRIES = ["US", "UK", "CA", "DE", "FR", "AU", "IN", "NG", "BR", "CN"]
HIGH_RISK_COUNTRIES = {"NG", "BR", "CN"}

CARD_TYPES = ["credit", "debit", "prepaid"]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def random_datetime(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_transaction(is_fraud: bool) -> dict:
    txn_id = str(uuid.uuid4())
    timestamp = random_datetime(START_DATE, END_DATE)
    hour = timestamp.hour

    # Fraudulent transactions tend to be larger, happen at odd hours,
    # and originate from high-risk locations more often.
    if is_fraud:
        amount = round(np.random.lognormal(mean=5.5, sigma=1.2), 2)   # heavier tail
        country = random.choices(
            COUNTRIES,
            weights=[1, 1, 1, 1, 1, 1, 1, 5, 4, 4],   # higher weight on risky countries
        )[0]
        hour = random.choices(range(24), weights=[
            3, 3, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1,        # night hours more likely
            1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3
        ])[0]
        card_type = random.choices(CARD_TYPES, weights=[2, 2, 6])[0]  # prepaid more common
        is_international = country != "US"
        is_high_risk_country = country in HIGH_RISK_COUNTRIES
        num_prev_txns_1h = random.randint(3, 15)        # velocity spike
        distance_from_home_km = round(random.uniform(500, 15_000), 1)
    else:
        amount = round(np.random.lognormal(mean=3.8, sigma=0.9), 2)
        country = random.choices(COUNTRIES, weights=[6, 2, 2, 1, 1, 1, 1, 1, 1, 1])[0]
        card_type = random.choices(CARD_TYPES, weights=[5, 4, 1])[0]
        is_international = country != "US"
        is_high_risk_country = country in HIGH_RISK_COUNTRIES
        num_prev_txns_1h = random.randint(0, 3)
        distance_from_home_km = round(random.uniform(0, 200), 1)

    merchant_category = random.choice(MERCHANT_CATEGORIES)

    # Card-present vs card-not-present (online)
    is_online = merchant_category == "online_retail" or (is_fraud and random.random() < 0.6)

    return {
        "transaction_id": txn_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "hour_of_day": hour,
        "day_of_week": timestamp.weekday(),          # 0=Mon … 6=Sun
        "amount_usd": amount,
        "merchant_category": merchant_category,
        "card_type": card_type,
        "country": country,
        "is_international": int(is_international),
        "is_high_risk_country": int(is_high_risk_country),
        "is_online": int(is_online),
        "num_prev_txns_1h": num_prev_txns_1h,
        "distance_from_home_km": distance_from_home_km,
        "is_fraud": int(is_fraud),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    n_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)
    n_legit = NUM_TRANSACTIONS - n_fraud

    records = (
        [generate_transaction(is_fraud=True)  for _ in range(n_fraud)]
        + [generate_transaction(is_fraud=False) for _ in range(n_legit)]
    )

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)

    # Summary
    print(f"Dataset saved to {out_path}")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Fraud rows   : {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.1%})")
    print(f"  Legit rows   : {(~df['is_fraud'].astype(bool)).sum():,}")
    print(f"\nSample (5 rows):\n{df.head().to_string()}")
    print(f"\nAmount stats:\n{df.groupby('is_fraud')['amount_usd'].describe().round(2)}")


if __name__ == "__main__":
    main()
