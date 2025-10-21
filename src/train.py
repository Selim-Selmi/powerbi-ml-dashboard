# src/train.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==============================
# Load Data
# ==============================
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_data():
    ratings = pd.read_excel(DATA_DIR / "ratings.xlsx")
    consumers = pd.read_excel(DATA_DIR / "consumers.xlsx")
    restaurants = pd.read_excel(DATA_DIR / "restaurants.xlsx")
    cuisines = pd.read_excel(DATA_DIR / "restaurant_cuisines.xlsx")
    preferences = pd.read_excel(DATA_DIR / "consumer_preferences.xlsx")
    return ratings, consumers, restaurants, cuisines, preferences


# ==============================
# Prepare Dataset
# ==============================
def make_dataset():
    ratings, consumers, restaurants, cuisines, preferences = load_data()

    print("✅ Columns loaded from Excel files:")
    print("ratings:", ratings.columns.tolist())

    # Ensure target column exists
    if "Overall_Rating" not in ratings.columns:
        raise ValueError("❌ Column 'Overall_Rating' not found in ratings.xlsx")

    # Primary cuisine per restaurant
    cuisines_primary = (
        cuisines.sort_values(["Restaurant_ID", "Cuisine"])
        .drop_duplicates(subset=["Restaurant_ID"], keep="first")
        .rename(columns={"Cuisine": "Primary_Cuisine"})
    )

    # Merge datasets
    df = (
        ratings
        .merge(consumers, on="Consumer_ID", how="left")
        .merge(restaurants, on="Restaurant_ID", how="left")
        .merge(cuisines_primary, on="Restaurant_ID", how="left")
        .merge(preferences, on="Consumer_ID", how="left")
    )

    # Derived feature: cuisine match
    df["Cuisine_Match"] = (
        df["Primary_Cuisine"] == df["Preferred_Cuisine"]
    ).astype(int)

    # Select relevant features
    candidate_features = [
        "Age", "Budget", "Drink_Level", "Transportation_Method",
        "Price", "Alcohol_Service", "Smoking_Allowed", "Franchise",
        "Area", "Parking", "Primary_Cuisine", "Preferred_Cuisine",
        "Cuisine_Match"
    ]
    features = [f for f in candidate_features if f in df.columns]

    # Keep only existing features and target
    df = df[features + ["Overall_Rating"]].copy()

    # Separate numeric and categorical
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "Overall_Rating" in num_cols:
        num_cols.remove("Overall_Rating")

    cat_cols = [c for c in df.columns if c not in num_cols + ["Overall_Rating"]]

    # Handle missing values
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            most_freq = df[c].mode()[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(most_freq)

    X = df.drop("Overall_Rating", axis=1)
    y = df["Overall_Rating"]

    print(f"✅ Final dataset prepared: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    return X, y, num_cols, cat_cols


# ==============================
# Train XGBoost Model
# ==============================
def train():
    X, y, num_cols, cat_cols = make_dataset()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    # XGBoost model
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        eval_metric="rmse"
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb)
    ])

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [3, 5, 7, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    # Randomized search for best hyperparameters
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=10,
        scoring="r2",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        error_score="raise"
    )

    print("🔍 Starting XGBoost hyperparameter tuning...")
    search.fit(X_train, y_train)

    # Evaluate
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"✅ Best R²: {r2:.4f} | RMSE: {rmse:.4f}")
    print("Best parameters:", search.best_params_)

    # Save model
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "xgb_rating_model.joblib")

    print("💾 Model saved to models/xgb_rating_model.joblib")


if __name__ == "__main__":
    train()
