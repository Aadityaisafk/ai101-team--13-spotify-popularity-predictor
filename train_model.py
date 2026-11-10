# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 2. Upload Dataset

uploaded = files.upload()

file_name = next(iter(uploaded))

df = pd.read_csv(r"C:\Users\Msi pc\Downloads\archive\spotify-tracks-dataset.csv")


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

df = df.drop_duplicates()

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

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Popularity")

plt.ylabel("Predicted Popularity")

plt.title("Actual vs Predicted Spotify Popularity")

plt.show()


# 12. Display Regression Coefficients

print("Intercept:", model.intercept_)

print("Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", coefficient)
