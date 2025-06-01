import argparse
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def detect_columns(df):
    """
    The AI4I 2020 CSV typically has headers like:
      - 'UDI' (ignored)
      - 'Product ID' or 'product_id'
      - 'Type' or 'product_type'
      - 'Air temperature [K]'
      - 'Process temperature [K]'
      - 'Rotational speed [rpm]'
      - 'Torque [Nm]'
      - 'Tool wear [min]'
      - 'Machine failure' (target)
      - ... possibly other failure flags

    This function finds the actual column names by looking for substrings.
    Returns a dict mapping:
      speed_col, torque_col, air_col, process_col, tool_wear_col, target_col.
    """

    col_lower = {c.lower(): c for c in df.columns}

    def find_col(substring):
        """Return the first column name containing `substring` (case-insensitive)."""
        for low, orig in col_lower.items():
            if substring in low:
                return orig
        raise KeyError(f"No column containing '{substring}' found in CSV.")

    # Find columns by substring
    air_col = find_col("air temperature")
    process_col = find_col("process temperature")
    speed_col = find_col("rotational speed")
    torque_col = find_col("torque")
    tool_wear_col = find_col("tool wear")
    target_col = find_col("machine failure")

    return {
        "air": air_col,
        "process": process_col,
        "speed": speed_col,
        "torque": torque_col,
        "wear": tool_wear_col,
        "target": target_col,
    }


def add_feature_engineering(df, cols):
    """
    Add interaction and polynomial features based on detected column names.
    cols: dictionary from detect_columns()
    """
    df = df.copy()

    speed = cols["speed"]
    torque = cols["torque"]
    air = cols["air"]
    process = cols["process"]
    wear = cols["wear"]

    # 1. Interaction terms
    df["speed_x_torque"] = df[speed] * df[torque]
    df["temp_diff"] = df[air] - df[process]

    # 2. Wear-rate feature (avoid divide by zero)
    eps = 1e-8
    df["wear_rate"] = df[wear] / (df[speed] + eps)

    # 3. Polynomial features for the core numeric columns
    poly_cols = [air, process, speed, torque, wear]
    for col in poly_cols:
        df[f"{col}_squared"] = df[col] ** 2
        df[f"{col}_cubed"] = df[col] ** 3

    return df


def build_pipeline(numeric_features, categorical_features):
    """
    Build a scikit‐learn Pipeline that:
     - Scales numeric features with StandardScaler
     - One‐hot encodes categorical features
     - Feeds into a RandomForestClassifier
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    return pipeline


def evaluate_with_cv(X, y, pipeline, cv_splits=5):
    """
    Perform Stratified K-Fold cross-validation and print mean ± std for
    accuracy, ROC-AUC, and F1 scores.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "f1"]

    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
    )

    print("=== Cross‐Validation Results (5-Fold) ===")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")
    print()

    return results


def plot_learning_curve(pipeline, X, y, output_path=None):
    """
    Plot training vs. cross‐validation learning curve (accuracy vs. training size).
    Saves the plot if output_path is provided; otherwise, displays it.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(
        train_sizes,
        train_mean,
        "o-",
        color="blue",
        label="Training score",
    )
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    plt.plot(
        train_sizes,
        valid_mean,
        "o-",
        color="orange",
        label="Cross‐validation score",
    )
    plt.fill_between(
        train_sizes,
        valid_mean - valid_std,
        valid_mean + valid_std,
        alpha=0.1,
        color="orange",
    )

    plt.title("Learning Curve (Accuracy vs. Training Size)")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved learning curve to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate AI4I model with feature engineering + CV"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to AI4I CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model & plots",
    )
    args = parser.parse_args()

    # 1. Load raw data
    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    print("Data loaded successfully.\n")

    # 2. Detect the actual column names in your CSV
    print("Detecting column names in dataset ...")
    try:
        cols = detect_columns(df)
        print("Detected columns:\n", cols, "\n")
    except KeyError as e:
        print(str(e))
        print(
            "Please ensure your CSV has those columns or adjust detect_columns() accordingly."
        )
        return

    # 3. Feature engineering
    print("Adding feature engineering ...")
    df_fe = add_feature_engineering(df, cols)

    # 4. Separate features & target
    target_col = cols["target"]
    X = df_fe.drop(columns=[target_col])
    y = df_fe[target_col]

    # 5. Identify numeric & categorical features after feature engineering
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numeric features ({}):".format(len(numeric_cols)), numeric_cols)
    print("Categorical features ({}):".format(len(categorical_cols)), categorical_cols, "\n")

    # 6. Build the pipeline
    pipeline = build_pipeline(numeric_cols, categorical_cols)

    # 7. Cross‐validation evaluation
    _ = evaluate_with_cv(X, y, pipeline, cv_splits=5)

    # 8. Fit final pipeline on all data and save
    print("Fitting final pipeline on entire dataset ...")
    pipeline.fit(X, y)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "rf_pipeline_full.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Final pipeline saved to {model_path}\n")

    # 9. Plot & save learning curve
    lc_path = os.path.join(args.output_dir, "learning_curve.png")
    print("Plotting learning curve ...")
    plot_learning_curve(pipeline, X, y, output_path=lc_path)
    print("All done.")


if __name__ == "__main__":
    main()
