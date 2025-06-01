import argparse
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def eda_summary(df, output_dir):
    """
    Perform exploratory data analysis:
    - Print basic summary statistics
    - Print missing value counts
    - Generate distribution plots for numerical features
    - Generate correlation heatmap
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Basic summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    df.describe().to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")

    # 2. Missing value counts
    missing_counts = df.isnull().sum()
    missing_path = os.path.join(output_dir, "missing_values.csv")
    missing_counts.to_csv(missing_path, header=["missing_count"])
    print(f"Saved missing value counts to {missing_path}")
    print("\nMissing value counts per column:")
    print(missing_counts[missing_counts > 0])

    # 3. Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        dist_path = os.path.join(output_dir, f"dist_{col}.png")
        plt.savefig(dist_path)
        plt.close()
        print(f"Saved distribution plot for {col} to {dist_path}")

    # 4. Correlation heatmap (numeric features only)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numeric Features)")
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved correlation heatmap to {heatmap_path}")

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis on AI4I dataset")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to AI4I CSV file for EDA"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eda",
        help="Directory to save EDA outputs (plots, CSVs)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    print("Data loaded successfully. Beginning EDA...\n")

    eda_summary(df, args.output_dir)
    print("\nEDA complete. Check the output directory for results.")

if __name__ == "__main__":
    main()
