import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate():
    print("[-] Loading raw food delivery data...")
    data_path = os.path.join("data", "raw", "food_delivery_orders.csv")
    df = pd.read_csv(data_path)

    # 1. Feature Engineering
    df["load_distance_ratio"] = np.round(df["items_count"] / df["distance_km"], 2)
    df["is_peak_hour"] = df["time_of_day"].apply(lambda x: 1 if "Peak" in x else 0)

    # 2. Features and Target Split
    target = "delivery_time_min"
    drop_cols = ["order_id", target]
    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = ["weather", "traffic_level", "time_of_day", "vehicle_type"]
    numeric_features = ["distance_km", "prep_time_min", "items_count", "rider_experience_yrs", "load_distance_ratio", "is_peak_hour"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )

    # 4. Model Candidates
    models = {
        "Baseline Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42)
    }

    best_model = None
    best_mae = float("inf")
    best_model_name = ""

    print("\n" + "="*65)
    print(f"{'Model':<28} | {'MAE (mins)':<10} | {'RMSE':<8} | {'R2 Score':<8}")
    print("="*65)

    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"{name:<28} | {mae:<10.2f} | {rmse:<8.2f} | {r2:<8.3f}")

        if mae < best_mae:
            best_mae = mae
            best_model = pipe
            best_model_name = name

    print("="*65)
    print(f"\n[✓] Optimal Model: {best_model_name} (MAE: {best_mae:.2f} mins)")

    # 5. Save the trained pipeline
    os.makedirs("models", exist_ok=True)
    model_output_path = os.path.join("models", "delivery_eta_pipeline.pkl")
    joblib.dump(best_model, model_output_path)
    print(f"[✓] Serialized production pipeline saved to: {model_output_path}")

if __name__ == "__main__":
    train_and_evaluate()