# train_production_merged.py
import sqlite3
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_merged_from_db(db_path: str = "data/factory_merged.db", table: str = "factory_merged") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn, parse_dates=["timestamp"])
    conn.close()
    return df.set_index("timestamp")

def train_and_evaluate(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Builds a pipeline combining CNC sensor features and production-floor features.
    Runs 5-fold CV, fits on all data, and saves the model.
    """
    feature_cols = [
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "setup_time", "processing_time"
    ]
    categorical_cols = ["machine_id"]  # if you want to one-hot encode this

    X = df[feature_cols + categorical_cols]
    y = df["machine_failure"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "f1"]

    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
    )

    print("=== 5-Fold CV Results ===")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

    pipeline.fit(X, y)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "rf_pipeline_production_merged.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")

if __name__ == "__main__":
    df = load_merged_from_db()
    train_and_evaluate(df)
