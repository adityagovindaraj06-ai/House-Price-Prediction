import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

skewed_features = [
    "Population",
    "AveOccup",
    "AveRooms",
    "AveBedrms"
]

X_train = X_train.copy()
X_test = X_test.copy()

for feature in skewed_features:
    X_train[feature] = np.log1p(X_train[feature])
    X_test[feature] = np.log1p(X_test[feature])

model = Pipeline([
    ("polynomial", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n" + "=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)

print("\nTraining Performance:")
print(f"Training MSE  : {train_mse:.4f}")
print(f"Training RMSE : {train_rmse:.4f}")
print(f"Training R²   : {train_r2:.4f}")

print("\nTesting Performance:")
print(f"Testing MSE   : {test_mse:.4f}")
print(f"Testing RMSE  : {test_rmse:.4f}")
print(f"Testing R²    : {test_r2:.4f}")

plt.figure(figsize=(8, 5))

plt.hist(
    df["MedHouseVal"],
    bins=30
)

plt.xlabel("House Value")
plt.ylabel("Frequency")
plt.title("Distribution of House Prices")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Values")

plt.tight_layout()
plt.show()

errors = np.abs(test_pred - y_test.values)

best_index = np.argmin(errors)

sample_house = X_test.iloc[[best_index]].copy()
sample_index = X_test.index[best_index]

original_sample = df.loc[
    [sample_index]
].drop(columns=["MedHouseVal"])

predicted_value = test_pred[best_index]
actual_value = y_test.iloc[best_index]

predicted_price = predicted_value * 100000
actual_price = actual_value * 100000

prediction_error = abs(
    predicted_price - actual_price
)
print("\n" + "=" * 50)

print("HOUSE PRICE PREDICTION")

print("=" * 50)

print("\nOriginal House Features:")

print(original_sample.to_string(index=False))

print(f"\nPredicted House Price: ${predicted_price:,.2f}")

print(f"Actual House Price   : ${actual_price:,.2f}")

print(f"Prediction Error     : ${prediction_error:,.2f}")

print("\n")

print("HOUSE PRICE PREDICTED SUCCESSFULLY")