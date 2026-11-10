
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


print("Libraries imported successfully!")
data_path = Path("data/dataset.csv") if Path("data/dataset.csv").exists() else Path("../data/dataset.csv")
df = pd.read_csv(data_path)

out_dir = Path("outputs") if Path("outputs").exists() else Path("../outputs")
out_dir.mkdir(exist_ok=True)

original_shape = df.shape
print("Dataset shape (rows, columns):", original_shape)
print(df.head())
print("Dataset dimensions:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types and missing-value overview:")
df.info()

print("\nSummary of numerical columns:")
print(df.describe())
print("Missing values in each column:")
print(df.isnull().sum())

print("\nNumber of completely repeated rows:", df.duplicated().sum())
print("Repeated track_id rows after the first occurrence:", df["track_id"].duplicated().sum())

print("\nRows with zero duration:", (df["duration_ms"] == 0).sum())

print("\nLast ten rows:")
print(df.tail(10))
print("Mean popularity:", df["popularity"].mean())
print("Median popularity:", df["popularity"].median())
print("Minimum popularity:", df["popularity"].min())
print("Maximum popularity:", df["popularity"].max())

print("\nPopularity summary:")
print(df["popularity"].describe())
plt.figure(figsize=(8, 4))
plt.hist(df["popularity"], bins=20, edgecolor="black")
plt.title("Distribution of Spotify Track Popularity")
plt.xlabel("Popularity score")
plt.ylabel("Number of tracks")
plt.grid(alpha=0.25)
plt.show()
genre_popularity = (
    df.groupby("track_genre")["popularity"]
      .mean()
      .sort_values(ascending=False)
)

top_10_genres = genre_popularity.head(10)

print("Top 10 genres by average popularity:")
print(top_10_genres.reset_index(name="average_popularity"))

plt.figure(figsize=(10, 5))
top_10_genres.sort_values().plot(kind="barh", edgecolor="black")
plt.title("Top 10 Genres by Average Track Popularity")
plt.xlabel("Average popularity")
plt.ylabel("Genre")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
plot_sample = df.sample(n=min(5000, len(df)), random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(plot_sample["danceability"], plot_sample["popularity"], alpha=0.3, s=12)
plt.title("Danceability vs Track Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.grid(alpha=0.25)
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(plot_sample["energy"], plot_sample["popularity"], alpha=0.3, s=12)
plt.title("Energy vs Track Popularity")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.grid(alpha=0.25)
plt.show()
numeric_features = [
    "popularity", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms"
]

correlation_with_popularity = (
    df[numeric_features]
      .corr()["popularity"]
      .drop("popularity")
      .sort_values(ascending=False)
)

print("Correlation of numerical features with popularity:")
print(correlation_with_popularity.to_frame(name="correlation"))
plt.figure(figsize=(9, 5))
correlation_with_popularity.sort_values().plot(kind="barh", edgecolor="black")
plt.title("Correlation of Audio Features with Popularity")
plt.xlabel("Correlation coefficient")
plt.ylabel("Audio feature")
plt.axvline(0, linestyle="--", linewidth=1)
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(out_dir / "correlation_matrix.png", dpi=300)
plt.show()
explicit_popularity = df.groupby("explicit")["popularity"].mean()
print("Average popularity by explicit label:")
print(explicit_popularity)

plt.figure(figsize=(6, 4))
explicit_popularity.plot(kind="bar", edgecolor="black")
plt.title("Average Popularity: Explicit vs Non-Explicit Tracks")
plt.xlabel("Explicit")
plt.ylabel("Average popularity")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
plt.show()
df["duration_min"] = df["duration_ms"] / 60000

print("Average duration (minutes):", df["duration_min"].mean())
print("Median duration (minutes):", df["duration_min"].median())

print(df[["track_name", "artists", "duration_ms", "duration_min"]].head())

print("DATASET OVERVIEW")
print("Rows:", original_shape[0])
print("Original columns:", original_shape[1])
print("Number of genres:", df["track_genre"].nunique())
print("Mean popularity:", round(df["popularity"].mean(), 2))
print("Median popularity:", round(df["popularity"].median(), 2))

print("\nTOP 5 GENRES BY AVERAGE POPULARITY")
print(genre_popularity.head(5).round(2))

print("\nFEATURES MOST POSITIVELY CORRELATED WITH POPULARITY")
print(correlation_with_popularity.head(3).round(3))

print("\nFEATURES MOST NEGATIVELY CORRELATED WITH POPULARITY")
print(correlation_with_popularity.tail(3).round(3))

