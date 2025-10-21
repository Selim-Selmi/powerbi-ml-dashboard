# src/data_prep.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_raw():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    consumers = pd.read_csv(DATA_DIR / "consumers.csv")
    restaurants = pd.read_csv(DATA_DIR / "restaurants.csv")
    rest_cuisines = pd.read_csv(DATA_DIR / "restaurant_cuisines.csv")
    return ratings, consumers, restaurants, rest_cuisines

def build_primary_cuisine(rest_cuisines: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the first cuisine per restaurant as 'Primary_Cuisine' for simplicity.
    """
    primary = (
        rest_cuisines
        .sort_values(["Restaurant_ID", "Cuisine"])
        .drop_duplicates(subset=["Restaurant_ID"], keep="first")
        .rename(columns={"Cuisine": "Primary_Cuisine"})
        [["Restaurant_ID", "Primary_Cuisine"]]
    )
    return primary


def make_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ratings, consumers, restaurants, rest_cuisines = load_raw()

    # primary cuisine
    pc = build_primary_cuisine(rest_cuisines)

    # join ratings with user features
    df = ratings.merge(consumers, on="Consumer_ID", how="left", suffixes=("", "_cons"))

    # join restaurant features
    df = df.merge(restaurants, on="Restaurant_ID", how="left", suffixes=("", "_rest"))

    # join primary cuisine
    df = df.merge(pc, on="Restaurant_ID", how="left")

    # target
    y = df["Overall_Rating"]

    # remove rows with missing target
    df = df[~y.isna()].copy()
    y = y.loc[df.index]

    # rename restaurant geo cols to avoid clashes (if not already)
    if "Latitude_rest" not in df.columns or "Longitude_rest" not in df.columns:
        df = df.rename(columns={
            "Latitude_y": "Latitude_rest",
            "Longitude_y": "Longitude_rest",
            "Latitude_x": "Latitude",
            "Longitude_x": "Longitude",
        })

    # =====================
    # --- Feature Engineering ---
    # =====================

    # Distance between consumer and restaurant (Haversine formula)
    R = 6371  # Earth radius in km
    lat1, lon1 = np.radians(df["Latitude"]), np.radians(df["Longitude"])
    lat2, lon2 = np.radians(df["Latitude_rest"]), np.radians(df["Longitude_rest"])

    df["Distance_km"] = 2 * R * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2 +
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )

    # Age group bins
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[0, 25, 40, 60, 100],
        labels=["Young", "Adult", "Middle", "Senior"]
    )

    # Franchise binary
    df["Is_Franchise"] = df["Franchise"].map({"Yes": 1, "No": 0}).fillna(0)

    # Cuisine match: merge consumer preferences
    # Cuisine match: merge consumer preferences (ensure one preference per consumer)
    pref = pd.read_csv(DATA_DIR / "consumer_preferences.csv")
    # if multiple preferences per consumer, keep the most frequent or first
    pref = (
        pref.groupby("Consumer_ID")["Preferred_Cuisine"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    df = df.merge(pref, on="Consumer_ID", how="left")

    df["Cuisine_Match"] = (
            df["Primary_Cuisine"] == df["Preferred_Cuisine"]
        ).astype(int)     

    # Budget / Price match
    df["Budget_Price_Match"] = (
        df["Budget"].str.lower() == df["Price"].str.lower()
    ).astype(int)

    # =====================
    # --- Feature Selection ---
    # =====================
    feature_cols = [
        # Consumer features
        "City", "State", "Country", "Smoker", "Drink_Level", "Transportation_Method",
        "Marital_Status", "Children", "Age", "Age_Group", "Occupation", "Budget",
        "Latitude", "Longitude",

        # Restaurant features
        "City_rest", "State_rest", "Country_rest", "Alcohol_Service",
        "Smoking_Allowed", "Price", "Franchise", "Is_Franchise",
        "Area", "Parking", "Latitude_rest", "Longitude_rest",

        # Derived
        "Primary_Cuisine", "Distance_km", "Cuisine_Match", "Budget_Price_Match"
    ]

    existing_features = [c for c in feature_cols if c in df.columns]
    X = df[existing_features].copy()

    # =====================
    # --- Cleaning ---
    # =====================
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill missing numerics with median
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    # Fill missing categoricals with most frequent (mode)
    for c in cat_cols:
        if X[c].isna().any():
            most_freq = X[c].mode()[0] if not X[c].mode().empty else "Unknown"
            X[c] = X[c].fillna(most_freq)

    return X, y, df[["Consumer_ID", "Restaurant_ID"]]


if __name__ == "__main__":
    X, y, ids = make_dataset()
    print("✅ Data prepared successfully!")
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Sample features:\n", X.head(3))
