

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, output_path='outputs/actual_vs_predicted.png'):
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.15, color='#1DB954', edgecolors='none', label='Test Tracks')
    
    # Reference 45-degree line for perfect prediction
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#E91429', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    
    plt.title('Actual vs. Predicted Song Popularity', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Popularity (0 - 100)', fontsize=12)
    plt.ylabel('Predicted Popularity (0 - 100)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Actual vs Predicted plot -> {output_path}")


def plot_residuals(y_true, y_pred, output_path='outputs/residuals_plot.png'):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = np.array(y_true) - np.array(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.15, color='#4A90E2', edgecolors='none')
    plt.axhline(0, color='#E91429', linestyle='--', linewidth=2)
    
    plt.title('Residuals vs. Predicted Popularity', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Popularity', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] Residual plot -> {output_path}")


def save_metrics_to_csv(metrics_dict, output_path='outputs/metrics_summary.csv'):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([metrics_dict])
    df.to_csv(output_path, index=False)
    print(f"[Saved] Evaluation metrics -> {output_path}")