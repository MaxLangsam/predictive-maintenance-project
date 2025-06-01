# save_to_sqlite.py
import sqlite3
import pandas as pd

def save_to_sqlite(df: pd.DataFrame, db: str = "factory_merged.db", table: str = "factory_merged"):
    """
    Write merged DataFrame to SQLite (overwriting if table exists).
    """
    conn = sqlite3.connect(db)
    df_reset = df.reset_index().rename(columns={"ts": "timestamp"})
    df_reset.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df_reset)} rows to '{db}' table '{table}'.")

if __name__ == "__main__":
    df_merged = pd.read_csv("data/merged_production_ai4i.csv", parse_dates=["ts"]).set_index("ts")
    save_to_sqlite(df_merged)
