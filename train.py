import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def load_data(filepath):
    """
    Load the AI4I dataset from a CSV file.
    Expected columns: product_type, air_temperature, process_temperature, rotational_speed,
    torque, tool_wear, and the failure flags including 'Machine failure'.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_and_train(df, model_output_dir):
    """
    Preprocess the dataframe and train a RandomForestClassifier to predict 'Machine failure'.
    Saves both the trained pipeline and a metrics report.
    """
    # Separate features and target
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing pipeline:
    # - Scale numeric features
    # - One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create a pipeline that first preprocesses, then fits a RandomForest
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Print metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save pipeline and metrics
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, 'rf_pipeline.joblib')
    joblib.dump(clf, model_path)
    print(f"Model pipeline saved to {model_path}")

    metrics_path = os.path.join(model_output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"Metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model on AI4I dataset")
    parser.add_argument('--data', type=str, required=True, help="Path to AI4I CSV file")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save model and metrics")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    df = load_data(args.data)

    print("Starting training...")
    preprocess_and_train(df, args.output_dir)

if __name__ == "__main__":
    main()
