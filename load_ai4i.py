# load_ai4i.py
import pandas as pd
from datetime import timedelta

def load_ai4i_with_timestamps(csv_path: str) -> pd.DataFrame:
    """
    Load AI4I CSV and assign a timestamp every 10 seconds.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("UDI").reset_index(drop=True)

    START_TIME = pd.to_datetime("2020-01-01 00:00:00")
    INTERVAL = timedelta(seconds=10)

    df["timestamp"] = [START_TIME + i * INTERVAL for i in range(len(df))]

    df = df.rename(columns={
        "Air temperature [K]": "air_temp_K",
        "Process temperature [K]": "proc_temp_K",
        "Rotational speed [rpm]": "rot_speed_rpm",
        "Torque [Nm]": "torque_Nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure"
    })

    return df.set_index("timestamp")

if __name__ == "__main__":
    ai4i_df = load_ai4i_with_timestamps("data/ai4i2020.csv")
    print(ai4i_df.head())
    ai4i_df.to_csv("data/ai4i_with_timestamps.csv")
