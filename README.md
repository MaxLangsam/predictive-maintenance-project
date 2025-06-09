
# Predictive Maintenance Project

This repository demonstrates a complete end-to-end workflow for building a predictive maintenance model by fusing CNC sensor data (AI4I) with shop-floor job scheduling data. The goal is to predict machine failures using combined features from two domains, all managed in a local, self-contained environment (SQLite + Python).

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Prerequisites](#prerequisites)  
4. [Data Sources](#data-sources)  
5. [Setup & Installation](#setup--installation)  
6. [Step 1: Flatten Shop-Floor (“Production-Floor”) Data](#step-1-flatten-shop-floor-production-floor-data)  
7. [Step 2: Load AI4I Data & Assign Timestamps](#step-2-load-ai4i-data--assign-timestamps)  
8. [Step 3: Merge CNC & Shop-Floor Data](#step-3-merge-cnc--shop-floor-data)  
9. [Step 4: Persist Merged Data to SQLite](#step-4-persist-merged-data-to-sqlite)  
10. [Step 5: Train Predictive Model](#step-5-train-predictive-model)  
11. [Step 6: Detailed Evaluation & Visualization](#step-6-detailed-evaluation--visualization)  
12. [Outputs & Artifacts](#outputs--artifacts)  
13. [Deployment & Inference](#deployment--inference)  
14. [Next Steps & Extensions](#next-steps--extensions)  
15. [References](#references)

---

## Project Overview

This project fuses two complementary datasets:

- **AI4I 2020 CNC Sensor Data**: Real-world dataset capturing CNC machine sensor readings (temperatures, rotational speed, torque, tool wear, categorical failure flags, etc.) sampled at a high frequency (simulated as every 10 seconds).  
- **Shop-Floor Job Scheduling Data (“Production-Floor”)**: Three CSV files defining 1,000 jobs and their routes through 200 machines, with per-job processing and setup times. Each job’s operations are timestamped based on cumulative time (setup + processing).

By harmonizing these sources on a common timestamp axis, we create a “digital twin” that provides both equipment sensor states and the shop-floor context for each cycle. A Random Forest classifier is then trained to predict “machine failure” (binary label) using features from both domains.

Key steps:

1. **Flatten production-floor CSVs** into a timestamped “operations” table.  
2. **Assign pseudo-timestamps** to AI4I rows (10-second intervals).  
3. **Merge** on nearest timestamp using `pandas.merge_asof`, within a 3-minute tolerance.  
4. **Store** the fused data in a local SQLite database (`factory_merged.db`).  
5. **Train** a scalable ML pipeline (preprocessing + RandomForest) on the merged features.  
6. **Evaluate** with detailed metrics, ROC/PR curves, feature importance, and SHAP explanations.  
7. **Visualize & Deploy** results via saved images (ROC/PR/SHAP) and optional API/Dashboard.

---

## Repository Structure

```

├── data/                                   
│   ├── 1000_200_process_times.csv
│   ├── 1000_200_routes.csv
│   ├── 1000_200_set-up_times.csv
│   ├── ai4i2020.csv
│   ├── ai4i_with_timestamps.csv
│   ├── factory_merged.db
│   ├── merged_production_ai4i.csv
│   └── production_floor_flattened.csv
├── images/                                 # Evaluation plots (generated at runtime)
│   ├── evaluation_pr_curve.png
│   └── evaluation_roc_curve.png
├── output/                                 # Model artifacts & metrics (generated at runtime)
│   ├── learning_curve.png
│   ├── metrics.txt
│   ├── rf_pipeline.joblib
│   ├── rf_pipeline_full.joblib
│   └── rf_pipeline_production_merged.joblib
├── README.md                               # ← This file
├── flatten_production_floor.py             # Flatten shop-floor CSVs
├── load_ai4i.py                            # Load AI4I & assign timestamps
├── merge_prod_ai4i.py                      # Merge AI4I & production-floor
├── save_to_sqlite.py                       # Save merged data into SQLite
├── train.py                                # Basic training script on AI4I
├── train_fe.py                             # Training with feature engineering
├── train_production_merged.py              # Train RF on fused data
├── detailed_evaluation.py                  # Metrics, ROC/PR, SHAP plots
├── inference_check.py                      # Spot-check inference on saved model
├── eda.py                                  # Exploratory Data Analysis script
├── dashboard_app.py                        # Streamlit dashboard (optional)
├── serve_api.py                            # FastAPI endpoint (optional)
└── requirements.txt                        # Python dependencies

```

---

## Prerequisites

- **Operating System**: Windows, macOS, or Linux  
- **Python 3.8+**  
- **Disk Space**: ~100 MB for intermediate CSVs, SQLite, and plots  
- **Memory**: ≥ 4 GB RAM for training  
- **Python Packages** (install via `pip install -r requirements.txt`):
```

pandas
numpy
scikit-learn
matplotlib
shap
joblib
fastapi       # Optional, if deploying an API
uvicorn       # Optional, if deploying an API
streamlit     # Optional, if deploying a dashboard

````

Ensure you activate the virtual environment before running any scripts:

```bash
python -m venv .venv
# on Windows PowerShell:
.\.venv\Scripts\Activate
# on macOS/Linux:
source .venv/bin/activate

pip install pandas numpy scikit-learn matplotlib shap joblib fastapi uvicorn streamlit
````

---

## Data Sources

### 1. AI4I 2020 Dataset (`ai4i2020.csv`)

* Contains CNC sensor readings and machine failure labels.
* Key columns:

  * `UDI`, `Product ID`, `Type`: Identifiers
  * `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
  * `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: Failure flags
* We simulate timestamps at 10-second intervals to mimic high-frequency sampling.

### 2. Production-Floor Data (`1000_200_*.csv`)

* **`1000_200_routes.csv`**: Each of the 1,000 lines corresponds to a job, listing machine IDs (1–200) visited, in sequence (comma-separated).
* **`1000_200_process_times.csv`**: A 1,000 × 200 table (no header): row *i*, column *j* = processing minutes for job *i* on machine *j* (0 = not visited).
* **`1000_200_set-up_times.csv`**: A 1,000 × 200 table (no header): row *i*, column *j* = setup minutes for job *i* on machine *j* (0 = none).
* Each job’s operation is timestamped as `2020-01-01 00:00:00 + cumulative_minutes` where `cumulative_minutes = sum(setup + processing)` for that job.

---

## Setup & Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/predictive_maintenance.git
   cd predictive_maintenance
   ```

2. **Create & Activate Virtual Environment**

   ```bash
   python -m venv .venv
   # on Windows PowerShell:
   .\.venv\Scripts\Activate
   # on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install pandas numpy scikit-learn matplotlib shap joblib fastapi uvicorn streamlit
   ```

4. **Verify Data Files Are Present**

   * `data/ai4i2020.csv`
   * `data/1000_200_routes.csv`
   * `data/1000_200_process_times.csv`
   * `data/1000_200_set-up_times.csv`

---

## Step 1: Flatten Shop-Floor (“Production-Floor”) Data

**Script**: `flatten_production_floor.py`

This script reads the three CSVs (`routes`, `process_times`, `setup_times`), flattens them into a timestamped operations table, and writes `production_floor_flattened.csv`.

```bash
python flatten_production_floor.py
```

* **Input**:

  * `data/1000_200_routes.csv`
  * `data/1000_200_process_times.csv`
  * `data/1000_200_set-up_times.csv`

* **Output**:

  * `data/production_floor_flattened.csv`

    ```csv
    timestamp,job_id,task_id,machine_id,setup_time,processing_time
    2020-01-01 00:00:00,1,1,17,2,10
    2020-01-01 00:12:00,1,2,42,3,8
    …
    ```

**Key Logic**:

* Each job (line in routes) is a sequence of machine IDs.
* For each job:

  1. Initialize `cum_minutes = 0`, `timestamp = START_TIME`.
  2. For each task:

     * `processing_time = proc_df.iat[job_idx, machine_id - 1]`
     * `setup_time = setup_df.iat[job_idx, machine_id - 1]`
     * Append record with `timestamp = START_TIME + timedelta(minutes=cum_minutes)`.
     * `cum_minutes += setup_time + processing_time`.
* Result: a DataFrame of shop-floor events at minute resolution.

---

## Step 2: Load AI4I Data & Assign Timestamps

**Script**: `load_ai4i.py`

This script reads `data/ai4i2020.csv`, sorts by `UDI`, and assigns a timestamp every 10 seconds starting from `2020-01-01 00:00:00`. The processed data is saved as `ai4i_with_timestamps.csv`.

```bash
python load_ai4i.py
```

* **Input**:

  * `data/ai4i2020.csv`

* **Output**:

  * `data/ai4i_with_timestamps.csv`

    ```csv
    timestamp,UDI,Product ID,Type,air_temp_K,proc_temp_K,rot_speed_rpm,torque_Nm,tool_wear_min,machine_failure,TWF,HDF,PWF,OSF,RNF
    2020-01-01 00:00:00,1234,1,L,300.15,310.22,1500.0,40.0,0.0,0,0,0,0,0,0
    2020-01-01 00:00:10,1235,1,M,299.25,309.18,1560.0,42.0,1.0,0,0,0,0,0,0
    …
    ```

**Key Logic**:

* `START_TIME = pd.to_datetime("2020-01-01 00:00:00")`
* `INTERVAL = timedelta(seconds=10)`
* `df["timestamp"] = [START_TIME + i * INTERVAL for i in range(len(df))]`
* Rename columns:

  * `"Air temperature [K]" → "air_temp_K"`
  * `"Process temperature [K]" → "proc_temp_K"`
  * `"Rotational speed [rpm]" → "rot_speed_rpm"`
  * `"Torque [Nm]" → "torque_Nm"`
  * `"Tool wear [min]" → "tool_wear_min"`
  * `"Machine failure" → "machine_failure"`

---

## Step 3: Merge CNC & Shop-Floor Data

**Script**: `merge_prod_ai4i.py`

This script uses `pd.merge_asof` to align each AI4I cycle with the most recent shop-floor operation (within a 3-minute tolerance). Output is saved to `data/merged_production_ai4i.csv`.

```bash
python merge_prod_ai4i.py
```

* **Inputs**:

  * `data/ai4i_with_timestamps.csv`
  * `production_floor_flattened.csv`

* **Output**:

  * `data/merged_production_ai4i.csv`

    ```csv
    ts,UDI,Product ID,Type,air_temp_K,proc_temp_K,rot_speed_rpm,torque_Nm,tool_wear_min,machine_failure,TWF,HDF,PWF,OSF,RNF,job_id,task_id,machine_id,setup_time,processing_time
    2020-01-01 00:00:00,1234,1,L,300.15,310.22,1500,40,0,0,0,0,0,0,, , , , , 
    2020-01-01 00:10:00,1235,1,M,299.25,309.18,1560,42,1,0,0,0,0,0,1,2,17,2,10
    …
    ```

**Key Logic**:

1. Load AI4I (parse `timestamp` as datetime, set index).
2. Load production-floor (parse `timestamp` as datetime, set index).
3. Reset indexes to `"ts"` for merging.
4. Sort by `"ts"` and call:

   ```python
   merged = pd.merge_asof(
       ai_reset.sort_values("ts"),
       prod_reset.sort_values("ts"),
       on="ts",
       direction="backward",
       tolerance=pd.Timedelta(minutes=3)
   )
   merged = merged.dropna(subset=["job_id", "task_id"])
   merged = merged.set_index("ts")
   ```
5. Save result to `data/merged_production_ai4i.csv`.

---

## Step 4: Persist Merged Data to SQLite

**Script**: `save_to_sqlite.py`

This script reads `data/merged_production_ai4i.csv` and writes it into a local SQLite database `data/factory_merged.db` (table: `factory_merged`).

```bash
python save_to_sqlite.py
```

* **Input**:

  * `data/merged_production_ai4i.csv`

* **Output**:

  * `data/factory_merged.db`

    * Table `factory_merged` with columns:

      ```
      timestamp,UDI,Product ID,Type,air_temp_K,proc_temp_K,rot_speed_rpm,torque_Nm,tool_wear_min,machine_failure,TWF,HDF,PWF,OSF,RNF,job_id,task_id,machine_id,setup_time,processing_time
      ```

**Key Logic**:

1. Open SQLite connection:

   ```python
   conn = sqlite3.connect("data/factory_merged.db")
   ```
2. Read CSV into pandas with `parse_dates=["ts"]`, set index, rename `"ts"` to `"timestamp"`.
3. `df_reset.to_sql("factory_merged", conn, if_exists="replace", index=False)`.
4. Close connection.

---

## Step 5: Train Predictive Model

**Script**: `train_production_merged.py`

This script loads merged data from SQLite, builds a preprocessing + RandomForest pipeline, runs 5-fold cross-validation, prints metrics, fits on all data, and saves the final pipeline to `output/rf_pipeline_production_merged.joblib`.

```bash
python train_production_merged.py
```

* **Input**:

  * `data/factory_merged.db` (table: `factory_merged`)

* **Output**:

  * `outputs/rf_pipeline_production_merged.joblib`

**Key Logic**:

1. **Load merged data**:

   ```python
   conn = sqlite3.connect("factory_merged.db")
   df = pd.read_sql_query("SELECT * FROM factory_merged", conn, parse_dates=["timestamp"])
   conn.close()
   df = df.set_index("timestamp")
   ```
2. **Select features & target**:

   ```python
   feature_cols = [
       "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
       "setup_time", "processing_time"
   ]
   categorical_cols = ["machine_id"]
   X = df[feature_cols + categorical_cols]
   y = df["machine_failure"]
   ```
3. **Build preprocessing pipeline**:

   ```python
   preprocessor = ColumnTransformer(
       transformers=[
           ("num", StandardScaler(), feature_cols),
           ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
       ],
       remainder="drop"
   )
   pipeline = Pipeline([
       ("preprocessor", preprocessor),
       ("classifier", RandomForestClassifier(
           n_estimators=100, random_state=42, class_weight="balanced"
       ))
   ])
   ```
4. **5-fold stratified CV**:

   ```python
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   scoring = ["accuracy", "roc_auc", "f1"]
   results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
   ```

   Print `mean ± std` for each metric.
5. **Fit on all data**:

   ```python
   pipeline.fit(X, y)
   os.makedirs("outputs", exist_ok=True)
   joblib.dump(pipeline, "outputs/rf_pipeline_production_merged.joblib")
   ```

---

## Step 6: Detailed Evaluation & Visualization

**Script**: `detailed_evaluation.py`

This script performs an intensive evaluation:

1. **Load a holdout slice** from `data/factory_merged.db` (by date range or fallback to last 1,000 rows).
2. **Compute classification metrics** (precision, recall, F1, ROC-AUC, confusion matrix).
3. **Plot & save**:

   * ROC curve → `images/evaluation_roc_curve.png`
   * Precision-Recall curve → `images/evaluation_pr_curve.png`
4. **Inspect false negatives/false positives** (first 10 each).
5. **Print top 10 feature importances** from the RandomForest.
6. **Generate SHAP summary**:

   * Transform a sample (≤ 500 rows) through the pipeline’s preprocessor.
   * Retrieve output feature names via `prep.get_feature_names_out()`.
   * Compute `shap_values(..., check_additivity=False)` and plot summary into `images/shap_summary.png`.

```bash
python detailed_evaluation.py
```

* **Inputs**:

  * `data/factory_merged.db`
  * `output/rf_pipeline_production_merged.joblib`

* **Outputs** (in `images/`):

  * `images/evaluation_roc_curve.png`
  * `images/evaluation_pr_curve.png`
  * `images/shap_summary.png`

---

## Outputs & Artifacts

After running all scripts, you’ll have:

1. **CSV Files** (in `data/`):

   * `production_floor_flattened.csv`
   * `ai4i_with_timestamps.csv`
   * `merged_production_ai4i.csv`

2. **SQLite Database** (in ]`data/`):

   * `factory_merged.db` (table: `factory_merged` containing all merged features)

3. **Trained Model**:

   * `outputs/rf_pipeline_production_merged.joblib` (Scikit-Learn Pipeline)

4. **Evaluation Plots** (in `images/`):

   * `evaluation_roc_curve.png`
   * `evaluation_pr_curve.png`
   * `shap_summary.png`

5. **Console Output**:

   * Classification reports, feature importances, false-positive/false-negative examples.

---

## Deployment & Inference

### 1. Spot-Check Inference

Use `inference_check.py` to load the saved model and run a quick prediction on the last 20 rows of `data/factory_merged.db`:

```bash
python inference_check.py
```

* Prints `Predictions: [...]` and `Failure probabilities: [...]`.
* Verifies the model pipeline runs without error on real data.

### 2. FastAPI Endpoint (Optional)

To serve online predictions, create `serve_api.py` (example):

```python
# serve_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
clf = joblib.load("output/rf_pipeline_production_merged.joblib")

class InputRow(BaseModel):
    air_temp_K: float
    proc_temp_K: float
    rot_speed_rpm: float
    torque_Nm: float
    tool_wear_min: float
    setup_time: float
    processing_time: float
    machine_id: int

@app.post("/predict")
def predict(row: InputRow):
    df = pd.DataFrame([row.dict()])
    pred = clf.predict(df)[0]
    prob = float(clf.predict_proba(df)[:, 1])
    return {"prediction": int(pred), "failure_probability": prob}
```

Run with:

```bash
uvicorn serve_api:app --reload
```

Send a POST request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"air_temp_K":300,"proc_temp_K":310,"rot_speed_rpm":1500,"torque_Nm":40,"tool_wear_min":5,"setup_time":2,"processing_time":10,"machine_id":17}'
```

### 3. Streamlit Dashboard (Optional)

For an interactive UI, create `dashboard_app.py` (example):

```python
# dashboard_app.py
import streamlit as st
import pandas as pd
import joblib

st.title("Production + AI4I Failure Predictor")

clf = joblib.load("outputs/rf_pipeline_production_merged.joblib")

uploaded = st.file_uploader("Upload merged CSV (factory_merged format)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    X_new = df[[
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "setup_time", "processing_time", "machine_id"
    ]]
    preds = clf.predict(X_new)
    probs = clf.predict_proba(X_new)[:, 1]
    df["predicted_failure"] = preds
    df["failure_prob"] = probs
    st.dataframe(df.head(10))
    st.download_button("Download predictions CSV", df.to_csv(index=False), "predictions.csv")
```

Run:

```bash
streamlit run dashboard_app.py
```

Users can upload merged data slices and obtain predictions in real time.

---

## Next Steps & Extensions

1. **Time-Based Cross-Validation**

   * Implement walk-forward (sliding window) validation to mimic production, where the model only sees past data.

2. **Drift Monitoring & Automated Retraining**

   * Write a scheduled script to compare recent feature distributions (Population Stability Index).
   * If performance drops below a threshold, trigger `train_production_merged.py` to retrain and redeploy.

3. **Hyperparameter Tuning**

   * Use `GridSearchCV` or `RandomizedSearchCV` to optimize RandomForest parameters (`max_depth`, `n_estimators`, `min_samples_leaf`, etc.).

4. **Feature Engineering**

   * Add interaction terms, rolling averages, or nonlinear transforms.
   * Explore temporal features (time since last failure, moving windows).

5. **Alternative Models**

   * Experiment with XGBoost/LightGBM or Neural Networks for potentially better performance.
   * Address class imbalance using SMOTE or cost-sensitive learning.

6. **Explainability & Reporting**

   * Extend SHAP analyses to generate interactive HTML reports.
   * Add LIME or Partial Dependence Plots for local interpretability.

7. **Full CI/CD Pipeline**

   * Automate linting, unit tests, and retraining in GitHub Actions.
   * On new data commits, trigger end-to-end pipeline and report metrics.

8. **Cloud Deployment**

   * Dockerize the API and deploy to AWS Elastic Beanstalk, Azure App Service, or GCP Cloud Run.
   * Migrate from SQLite to a managed PostgreSQL or MySQL instance for multi-user access.

---

## References

1. **AI4I 2020 Dataset**: CNC machine sensor data used for predictive maintenance tasks.
2. **Job-Shop Scheduling (JS\_DS\_01)**: Synthetic job scheduling dataset (converted here to production-floor CSVs).
3. **Pandas Documentation**: [merge\_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html)
4. **Scikit-Learn Documentation**:

   * [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
   * [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
   * [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. **SHAP Documentation**: [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/NB%20-%20XGBoost%20Regression.html)

---

**Congratulations!** You now have a fully reproducible predictive maintenance pipeline that:

* Fuses multi-domain data (sensors + shop-floor operations).
* Stores and queries features via SQLite.
* Trains and evaluates a robust ML model.
* Generates explainable visualizations (ROC/PR/SHAP).
* Can be deployed as an API or dashboard.

Feel free to iterate on any step—feature engineering, model architecture, or deployment—to tailor it to your real-world use case!

```
```
