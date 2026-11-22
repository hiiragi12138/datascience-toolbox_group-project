import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

df = pd.read_csv("instagram_analytics_cleaned.csv")

feature_columns = ["media_type", "content_category", "has_call_to_action", "caption_length",
    "hashtags_count",
    "post_hour",]

X = df[feature_columns]
y = df["engagement_rate"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(X_train.dtypes)

preprocessor = ColumnTransformer(
    transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["media_type", "content_category"]),],
    remainder = "passthrough")

linear_model_pipeline = Pipeline([("preprocess", preprocessor), ("model", LinearRegression()),])

linear_model_pipeline.fit(X_train, y_train)
print("Model trained successfully.")

y_pred = linear_model_pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

forest_model = Pipeline([("preprocess", clone(preprocessor)), ("model", RandomForestRegressor(n_estimators=100, max_depth = 10, random_state=42)),])
forest_model.fit(X_train, y_train)
print("Random Forest Model trained successfully.")

forest_pred = forest_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, forest_pred))
print("Random Forest RMSE:", root_mean_squared_error(y_test, forest_pred))
print("Random Forest R²:", r2_score(y_test, forest_pred))

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("Baseline MAE:", mean_absolute_error(y_test, baseline_pred))
print("Baseline RMSE:", root_mean_squared_error(y_test, baseline_pred))
print("Baseline R²:", r2_score(y_test, baseline_pred))  

results = []
for name, predictions in [
    ("Baseline", baseline_pred),
    ("Linear Regression", y_pred),
    ("Random Forest", forest_pred),
]:
    results.append({
        "model": name,
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": root_mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    })

pd.DataFrame(results).to_csv("model_metrics.csv", index=False)