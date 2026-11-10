import os
import sys
import time
from pathlib import Path

# Prevent matplotlib from opening GUI popup windows that freeze terminal execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from your project files
from src.preprocess import load_data, clean_data, prepare_features, AUDIO_FEATURES
from src.evaluation import evaluate_predictions, plot_actual_vs_predicted, plot_residuals, save_metrics_to_csv
from sklearn.linear_model import LinearRegression


def main():
    start_time = time.time()
    print("=" * 65)
    print("AI101 Team 13: Music Popularity Analysis & Prediction")
    print("=" * 65)

    dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
    if not dataset_path.exists():
        print(f"[Error] Dataset not found at: {dataset_path}")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocessing
    print("\n[Step 1/4] Running Data Cleaning and Preprocessing...")
    df_raw = load_data(str(dataset_path))
    df, _ = clean_data(df_raw)
    print(f"  Cleaned dataset size: {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # Step 2: Correlation Heatmap
    print("\n[Step 2/4] Generating Correlation Heatmap...")
    available_features = [f for f in AUDIO_FEATURES if f in df.columns]
    corr = df[["popularity"] + available_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Audio Features & Popularity")
    plt.tight_layout()
    corr_path = out_dir / "correlation_matrix.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"  [Saved] Heatmap -> {corr_path}")

    # Step 3: Train-Test Split & Linear Regression
    print("\n[Step 3/4] Training Linear Regression Model...")
    X_train, X_test, y_train, y_test, _ = prepare_features(df)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"  Model trained on {len(X_train):,} samples using {len(available_features)} audio features.")

    # Step 4: Evaluate and Save Results
    print("\n[Step 4/4] Evaluating Model and Generating Reports...")
    metrics = evaluate_predictions(y_test, y_pred)

    print("\n" + "-" * 35)
    print("        MODEL EVALUATION")
    print("-" * 35)
    print(f"  MAE  : {metrics['MAE']}")
    print(f"  RMSE : {metrics['RMSE']}")
    print(f"  R²   : {metrics['R2']}")
    print("-" * 35)

    plot_actual_vs_predicted(y_test, y_pred, output_path=str(out_dir / "actual_vs_predicted.png"))
    plot_residuals(y_test, y_pred, output_path=str(out_dir / "residuals_plot.png"))
    save_metrics_to_csv(metrics, output_path=str(out_dir / "metrics_summary.csv"))

    elapsed_time = time.time() - start_time
    print(f"\n[Done] Pipeline successfully finished in {elapsed_time:.2f} seconds.")
    print("=" * 65)


if __name__ == "__main__":
    main()