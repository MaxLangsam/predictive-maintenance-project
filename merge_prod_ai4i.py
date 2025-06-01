# merge_prod_ai4i.py
import pandas as pd

def merge_production_ai4i(ai_path: str, prod_path: str, tolerance_min: int = 3) -> pd.DataFrame:
    """
    Merge AI4I and production-floor DataFrames on nearest timestamp (backward match).
    - ai_path: path to ai4i_with_timestamps.csv
    - prod_path: path to production_floor_flattened.csv
    - tolerance_min: max minutes difference for a match
    Returns merged DataFrame indexed by timestamp.
    """
    # 1) Load AI4I
    ai = pd.read_csv(ai_path, parse_dates=["timestamp"]).set_index("timestamp")

    # 2) Load Production-Floor ops
    prod = pd.read_csv(prod_path, parse_dates=["timestamp"]).set_index("timestamp")

    # 3) Reset indexes for merge_asof
    ai_reset = ai.reset_index().rename(columns={"timestamp": "ts"})
    prod_reset = prod.reset_index().rename(columns={"timestamp": "ts"})

    ai_reset = ai_reset.sort_values("ts")
    prod_reset = prod_reset.sort_values("ts")

    # 4) merge_asof (backward = last prod_event ≤ AI4I ts)
    merged = pd.merge_asof(
        ai_reset,
        prod_reset,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_min)
    )

    # 5) Drop rows where no production match was found
    merged = merged.dropna(subset=["job_id", "task_id"])

    # 6) Set timestamp back as index
    return merged.set_index("ts")

if __name__ == "__main__":
    merged_df = merge_production_ai4i(
        ai_path="data/ai4i_with_timestamps.csv",
        prod_path="data/production_floor_flattened.csv",
        tolerance_min=3
    )
    print(merged_df.head())
    merged_df.to_csv("data/merged_production_ai4i.csv")
