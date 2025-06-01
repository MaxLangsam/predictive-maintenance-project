import pandas as pd
from datetime import timedelta

def load_production_floor(
    routes_path: str,
    process_times_path: str,
    setup_times_path: str
) -> pd.DataFrame:
    """
    Reads the three ‘1000_200_*.csv’ files and returns a single DataFrame:
      ['timestamp', 'job_id', 'task_id', 'machine_id', 'setup_time', 'processing_time'].
    The timestamp is computed by assuming each job starts at 2020-01-01 00:00:00
    and that each operation takes (setup_time + processing_time) minutes in sequence.
    """

    # 1) Load the 1000×200 tables
    proc_df = pd.read_csv(process_times_path, header=None, engine="python")
    setup_df = pd.read_csv(setup_times_path, header=None, engine="python")

    # 2) Read routes.csv: each line is a comma-separated list of machine IDs
    routes_list = []
    with open(routes_path, "r") as f:
        for line in f:
            # Split on commas, convert to int if digit, ignore empties
            route = [int(x) for x in line.strip().split(",") if x.strip().isdigit()]
            routes_list.append(route)

    # Sanity check: we expect exactly as many routes as rows in proc_df
    if len(routes_list) != proc_df.shape[0]:
        raise ValueError(
            f"Expected {proc_df.shape[0]} lines in routes.csv but found {len(routes_list)}."
        )

    # 3) Build a list of “events” (one per job‐machine pair)
    records = []
    START_TIME = pd.to_datetime("2020-01-01 00:00:00")

    for job_idx, route in enumerate(routes_list):
        job_id = job_idx + 1
        cum_minutes = 0  # track elapsed minutes for this job

        for task_idx, machine_id in enumerate(route, start=1):
            # Look up processing_time and setup_time in the DataFrames
            processing_time = proc_df.iat[job_idx, machine_id - 1]
            setup_time = setup_df.iat[job_idx, machine_id - 1]

            # Compute this operation’s timestamp = START_TIME + cum_minutes (cast to int)
            ts = START_TIME + timedelta(minutes=int(cum_minutes))

            # Append the event
            records.append({
                "timestamp": ts,
                "job_id": job_id,
                "task_id": task_idx,
                "machine_id": machine_id,
                "setup_time": int(setup_time),
                "processing_time": int(processing_time)
            })

            # Increment elapsed time by setup + processing (both in minutes)
            cum_minutes = int(cum_minutes) + int(setup_time) + int(processing_time)

    # 4) Convert to DataFrame, index by timestamp
    df_ops = pd.DataFrame(records)
    df_ops = df_ops.set_index("timestamp")

    return df_ops

# If you save this as flatten_production_floor.py, you can run it directly
if __name__ == "__main__":
    routes_path = "data/1000_200_routes.csv"
    proc_path = "data/1000_200_process_times.csv"
    setup_path = "data/1000_200_set-up_times.csv"

    df_prod = load_production_floor(routes_path, proc_path, setup_path)
    print(df_prod.head(10))

    # Optionally write out for inspection
    df_prod.to_csv("data/production_floor_flattened.csv")
