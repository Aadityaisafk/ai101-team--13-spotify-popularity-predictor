# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path




data_path = Path("data/dataset.csv") if Path("data/dataset.csv").exists() else Path("../data/dataset.csv")
df = pd.read_csv(data_path)


# 3. Display Dataset

print("First 5 rows:")
print(df.head())

print("Dataset Shape:")
print(df.shape)

print("Dataset Information:")
df.info()

print("Missing Values:")
print(df.isnull().sum())


# 4. Data Cleaning

df = df.dropna()

df = df.drop_duplicates(subset=["track_id"] if "track_id" in df.columns else None)

df = df.drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)


# 5. Select Input Features (X) and Target (y)

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

X = df[features]

y = df["popularity"]


# 6. Split Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Create Linear Regression Model

model = LinearRegression()


# 8. Train the Model

model.fit(X_train, y_train)


# 9. Make Predictions

y_pred = model.predict(X_test)


# 10. Evaluate the Model

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)


# 11. Visualize Actual vs Predicted Values

plt.scatter(y_test, y_pred, alpha=0.2, color="#1DB954", label="Test Tracks")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Spotify Popularity")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()

# Create outputs folder if missing and save
out_dir = Path("outputs") if Path("outputs").exists() else Path("../outputs")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "actual_vs_predicted.png", dpi=300)

plt.show()


# 12. Display Regression Coefficients

print("Intercept:", model.intercept_)

print("Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)
