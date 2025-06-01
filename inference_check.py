# detailed_evaluation.py

import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import OneHotEncoder

def load_holdout_data(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Attempt to load rows from 'factory_merged' between start_date and end_date.
    If no rows are found, fall back to the last 1000 rows in the table.
    Returns a DataFrame indexed by timestamp.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT *
          FROM factory_merged
         WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
         ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, parse_dates=["timestamp"]).set_index("timestamp")

    if df.shape[0] == 0:
        print(f"No data found between {start_date} and {end_date}. Falling back to last 1000 rows.")
        fallback = pd.read_sql_query(
            "SELECT * FROM factory_merged ORDER BY timestamp DESC LIMIT 1000",
            conn,
            parse_dates=["timestamp"]
        ).set_index("timestamp")
        df = fallback.iloc[::-1]  # reverse to chronological order

    conn.close()
    return df

def compute_classification_metrics(df: pd.DataFrame, pipeline_path: str):
    """
    Given a DataFrame and a trained pipeline, compute and print metrics.
    Returns feature matrix X, true labels, predicted labels, predicted probabilities, and the pipeline.
    """
    if df.shape[0] == 0:
        raise ValueError("No data provided for evaluation.")

    numeric_cols = [
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "setup_time", "processing_time"
    ]
    categorical_cols = ["machine_id"]

    for col in numeric_cols + categorical_cols + ["machine_failure"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    X = df[numeric_cols + categorical_cols]
    y_true = df["machine_failure"]

    clf = joblib.load(pipeline_path)

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return X, y_true, y_pred, y_proba, clf

def plot_roc_pr_curves(y_true, y_proba, output_prefix: str = "evaluation"):
    """
    Plot ROC and Precision-Recall curves, saving images under 'images/'.
    """
    os.makedirs("images", exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"images/{output_prefix}_roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(f"images/{output_prefix}_pr_curve.png")
    plt.close()

def print_false_cases(df: pd.DataFrame, X, y_true, y_pred, clf, num_cases: int = 10):
    """
    Print first few false negatives and false positives with probabilities.
    """
    df2 = df.copy()
    df2["y_true"] = y_true
    df2["y_pred"] = y_pred
    df2["y_proba"] = clf.predict_proba(X)[:, 1]

    fn = df2[(df2["y_true"] == 1) & (df2["y_pred"] == 0)].head(num_cases)
    fp = df2[(df2["y_true"] == 0) & (df2["y_pred"] == 1)].head(num_cases)

    print(f"\n=== First {num_cases} False Negatives ===")
    if not fn.empty:
        print(fn[[
            "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm",
            "tool_wear_min", "setup_time", "processing_time", "machine_id", "y_proba"
        ]])
    else:
        print("None")

    print(f"\n=== First {num_cases} False Positives ===")
    if not fp.empty:
        print(fp[[
            "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm",
            "tool_wear_min", "setup_time", "processing_time", "machine_id", "y_proba"
        ]])
    else:
        print("None")

def show_feature_importance(clf):
    """
    Print the top 10 feature importances from the RandomForest inside the pipeline.
    """
    numeric_cols = [
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "setup_time", "processing_time"
    ]
    ohe: OneHotEncoder = clf.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_cols = list(ohe.get_feature_names_out(["machine_id"]))
    all_features = numeric_cols + ohe_cols

    importances = clf.named_steps["classifier"].feature_importances_
    idx_sorted = np.argsort(importances)[::-1]

    print("\n=== Top 10 Feature Importances ===")
    for i in idx_sorted[:10]:
        print(f"{all_features[i]}: {importances[i]:.4f}")

def plot_shap_summary(X, clf, output_path: str = "shap_summary.png"):
    """
    Generate and save a SHAP summary plot on a sample of X, saving to 'images/'.
    Disables additivity checking during shap_values call, and ensures feature names match.
    """
    if X.shape[0] == 0:
        print("No samples for SHAP plot.")
        return

    os.makedirs("images", exist_ok=True)

    # Sample up to 500 rows for speed
    sample = X.sample(n=min(500, len(X)), random_state=42)
    prep = clf.named_steps["preprocessor"]
    model = clf.named_steps["classifier"]

    # Transform the sample (make sure it’s dense)
    X_transformed = prep.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Build correct feature names to match columns of X_transformed:
    numeric_cols = [
        "air_temp_K", "proc_temp_K", "rot_speed_rpm", "torque_Nm", "tool_wear_min",
        "setup_time", "processing_time"
    ]
    ohe: OneHotEncoder = prep.named_transformers_["cat"]
    ohe_cols = list(ohe.get_feature_names_out(["machine_id"]))
    feature_names = numeric_cols + ohe_cols

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    # Disable additivity check here
    shap_values = explainer.shap_values(X_transformed, check_additivity=False)

    # Plot summary, using our feature_names
    shap.summary_plot(
        shap_values[1],  # SHAP values for the positive class
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"images/{output_path}")
    plt.close()

if __name__ == "__main__":
    # 1) Load hold-out data (use your own dates or let fallback load last 1000 rows)
    df_hold = load_holdout_data(
        db_path="data/factory_merged.db",
        start_date="2024-07-01",
        end_date="2024-07-15"
    )

    if df_hold.empty:
        print("No hold-out data available for the specified dates or fallback.")
        exit()

    # 2) Compute metrics
    X_hold, y_hold, y_pred, y_proba, pipeline = compute_classification_metrics(
        df_hold,
        pipeline_path="output/rf_pipeline_production_merged.joblib"
    )

    # 3) Plot ROC & Precision-Recall curves
    plot_roc_pr_curves(y_hold, y_proba, output_prefix="evaluation")

    # 4) Inspect false positives/negatives
    print_false_cases(df_hold, X_hold, y_hold, y_pred, pipeline, num_cases=10)

    # 5) Feature importances
    show_feature_importance(pipeline)

    # 6) SHAP summary plot
    plot_shap_summary(X_hold, pipeline, output_path="shap_summary.png")

    print("\nEvaluation complete. Plots saved to the 'images/' folder.")
