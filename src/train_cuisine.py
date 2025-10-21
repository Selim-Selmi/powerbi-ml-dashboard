# src/train_cuisine.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_data():
    consumers = pd.read_excel(DATA_DIR / "consumers.xlsx")
    preferences = pd.read_excel(DATA_DIR / "consumer_preferences.xlsx")
    df = consumers.merge(preferences, on="Consumer_ID", how="inner")
    return df
def make_dataset():
    df = load_data()

    # Drop missing
    df = df[~df["Preferred_Cuisine"].isna()].copy()

    # Group similar cuisines
    group_map = {
        "Mexican": "Latin", "Latin American": "Latin",
        "Japanese": "Asian", "Chinese": "Asian", "Sushi": "Asian",
        "Italian": "European", "Spanish": "European", "French": "European",
        "American": "American", "Fast Food": "American", "Burgers": "American", "Barbecue": "American",
        "Coffee Shop": "Cafe", "Bakery": "Cafe", "Juice": "Cafe"
    }
    df["Preferred_Cuisine"] = df["Preferred_Cuisine"].map(group_map)
    df = df[df["Preferred_Cuisine"].notna()].copy()

    # Features
    features = [
        "Age", "Budget", "Drink_Level", "Transportation_Method",
        "Marital_Status", "Children", "Occupation", "Smoker"
    ]
    target = "Preferred_Cuisine"

    # Fill missing
    for c in df.columns:
        if df[c].dtype == "object" or df[c].dtype.name == "category":
            most_freq = df[c].mode()[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(most_freq)
        else:
            df[c] = df[c].fillna(df[c].median())

    # ✅ Balance dataset manually
    min_samples = df["Preferred_Cuisine"].value_counts().min()
    df = (
        df.groupby("Preferred_Cuisine", group_keys=False)
          .apply(lambda x: x.sample(min_samples, random_state=42))
          .reset_index(drop=True)
    )

    X = df[features]
    y = df[target]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols



def train():
    X, y, num_cols, cat_cols = make_dataset()
    # Remove rare classes with fewer than 2 samples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    # Now safely stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("🔍 Training Cuisine Classifier...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "cuisine_classifier.joblib")
    print("💾 Model saved to models/cuisine_classifier.joblib")

if __name__ == "__main__":
    train()
