"""
Data Cleaning and Preprocessing Module
AI101 Mini-Project, Group 13: Hit Song Popularity Prediction
Author: Hemanshu Samadhan Borse (2026ESB1477, GitHub: xlent-h)

Loads the Spotify Tracks dataset, cleans invalid/duplicate records,
and outputs data/cleaned_dataset.csv and outputs/cleaning_summary.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(filepath='data/dataset.csv'):
    """Loads dataset from file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

# Audio features used for regression
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

UNIT_RANGE = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence"
]


def get_paths():
    """Resolves project paths cleanly regardless of working directory."""
    # Check if run from repo root or src/
    root = Path("..") if Path("../data").exists() else Path(".")
    data_path = root / "data" / "dataset.csv"
    clean_path = root / "data" / "cleaned_dataset.csv"
    out_dir = root / "outputs"
    return data_path, clean_path, out_dir


def invalid_mask(df):
    """
    Identifies invalid records:
    - Popularity outside 0-100
    - Unit range audio features outside 0-1
    - Non-positive duration or tempo
    """
    bad = ~df["popularity"].between(0, 100)
    for col in UNIT_RANGE:
        if col in df.columns:
            bad |= ~df[col].between(0, 1)
    if "duration_ms" in df.columns:
        bad |= df["duration_ms"] <= 0
    if "tempo" in df.columns:
        bad |= df["tempo"] <= 0
    return bad


def clean_data(df_raw):
    """
    Executes the exact cleaning pipeline from data_preprocessing.ipynb.
    Returns cleaned DataFrame and step-by-step summary DataFrame.
    """
    df = df_raw.copy()
    log = [("Original dataset", len(df), df.shape[1])]

    # a) Drop unnamed index columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    log.append(("Drop unnamed index column(s)", len(df), df.shape[1]))

    # b) Drop rows with missing values
    df = df.dropna()
    log.append(("Drop rows with missing values", len(df), df.shape[1]))

    # c) Drop fully identical rows
    if "track_genre" in df.columns:
        df["track_genre"] = df["track_genre"].str.strip()
    df = df.drop_duplicates()
    log.append(("Drop fully identical rows", len(df), df.shape[1]))

    # d) Drop repeats of same track within same genre
    if "track_id" in df.columns and "track_genre" in df.columns:
        df = df.drop_duplicates(subset=["track_id", "track_genre"], keep="first")
    log.append(("Drop repeated track_id within the same genre", len(df), df.shape[1]))

    # e) Drop invalid records
    df = df[~invalid_mask(df)]
    log.append(("Drop invalid records", len(df), df.shape[1]))

    # f) Make 'explicit' numeric (0/1)
    if "explicit" in df.columns and df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)

    df = df.reset_index(drop=True)

    summary = pd.DataFrame(log, columns=["step", "rows_after", "columns_after"])
    summary["rows_removed"] = (-summary["rows_after"].diff()).fillna(0).astype(int)

    return df, summary


def prepare_features(df, test_size=0.2, random_state=42):
    """
    Extracts features and target, performs 80/20 train-test split,
    and applies standard scaling as taught in AI101.
    """
    available_features = [f for f in AUDIO_FEATURES if f in df.columns]
    X = df[available_features].copy()
    y = df["popularity"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=available_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=available_features, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, available_features


def run_preprocessing():
    """Standalone runner that saves cleaned dataset and summary CSV."""
    data_path, clean_path, out_dir = get_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {data_path}")
    df_raw = pd.read_csv(data_path)
    rows_before, cols_before = df_raw.shape
    print(f"Raw shape: {rows_before:,} rows, {cols_before} columns")

    cleaned_df, summary = clean_data(df_raw)
    rows_after, cols_after = cleaned_df.shape
    print(f"Cleaned shape: {rows_after:,} rows, {cols_after} columns ({rows_before - rows_after:,} removed)")

    # Save to disk
    cleaned_df.to_csv(clean_path, index=False)
    summary.to_csv(out_dir / "cleaning_summary.csv", index=False)
    print(f"Saved cleaned data to: {clean_path}")
    print(f"Saved cleaning summary to: {out_dir / 'cleaning_summary.csv'}")


if __name__ == "__main__":
    run_preprocessing()