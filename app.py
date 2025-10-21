from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)
app.secret_key = "change-this-in-production"

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# -------- Load data (once at startup) --------
def load_data():
    consumers = pd.read_excel(DATA_DIR / "consumers.xlsx")
    restaurants = pd.read_excel(DATA_DIR / "restaurants.xlsx")
    cuisines = pd.read_excel(DATA_DIR / "restaurant_cuisines.xlsx")
    preferences = pd.read_excel(DATA_DIR / "consumer_preferences.xlsx")

    cuisines_primary = (
        cuisines.sort_values(["Restaurant_ID", "Cuisine"])
        .drop_duplicates(subset=["Restaurant_ID"], keep="first")
        .rename(columns={"Cuisine": "Primary_Cuisine"})
    )
    restaurants = restaurants.merge(cuisines_primary, on="Restaurant_ID", how="left")

    return consumers, restaurants, preferences

consumers_df, restaurants_df, preferences_df = load_data()

# -------- Load trained models --------
rating_model = joblib.load(MODELS_DIR / "xgb_rating_model.joblib")
cuisine_model = joblib.load(MODELS_DIR / "cuisine_classifier.joblib")

# -------- Helpers --------
def prepare_rating_features(consumer_id: int, restaurant_id: int) -> pd.DataFrame:
    c_row = consumers_df.loc[consumers_df["Consumer_ID"] == consumer_id]
    r_row = restaurants_df.loc[restaurants_df["Restaurant_ID"] == restaurant_id]

    if c_row.empty or r_row.empty:
        raise ValueError("Invalid Consumer_ID or Restaurant_ID.")

    pref = preferences_df.loc[
        preferences_df["Consumer_ID"] == consumer_id, "Preferred_Cuisine"
    ]
    preferred_cuisine = pref.iloc[0] if not pref.empty else "Unknown"

    cuisine_match = int(r_row["Primary_Cuisine"].values[0] == preferred_cuisine)

    data = {
        "Age": c_row["Age"].values[0],
        "Budget": c_row["Budget"].values[0],
        "Drink_Level": c_row["Drink_Level"].values[0],
        "Transportation_Method": c_row["Transportation_Method"].values[0],
        "Price": r_row["Price"].values[0],
        "Alcohol_Service": r_row["Alcohol_Service"].values[0],
        "Smoking_Allowed": r_row["Smoking_Allowed"].values[0],
        "Franchise": r_row["Franchise"].values[0],
        "Area": r_row["Area"].values[0],
        "Parking": r_row["Parking"].values[0],
        "Primary_Cuisine": r_row["Primary_Cuisine"].values[0],
        "Preferred_Cuisine": preferred_cuisine,
        "Cuisine_Match": cuisine_match,
    }

    X = pd.DataFrame([data])
    # Ensure numeric types where needed
    for col in ["Age", "Cuisine_Match"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Cast others to string to align with OneHotEncoder expectations
    for col in X.columns:
        if col not in ["Age", "Cuisine_Match"]:
            X[col] = X[col].astype(str).replace("nan", "Unknown").fillna("Unknown")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

# ==================== Routes ====================

@app.route("/")
def home():
    return render_template(
        "index.html",
        consumer_ids=sorted(consumers_df["Consumer_ID"].unique().tolist()),
        restaurant_ids=sorted(restaurants_df["Restaurant_ID"].unique().tolist()),
    )

@app.route("/rating", methods=["GET", "POST"])
def rating():
    consumer_ids = sorted(consumers_df["Consumer_ID"].unique().tolist())
    restaurant_ids = sorted(restaurants_df["Restaurant_ID"].unique().tolist())

    prediction = None
    features_preview = None

    if request.method == "POST":
        try:
            consumer_id = int(request.form.get("consumer_id"))
            restaurant_id = int(request.form.get("restaurant_id"))

            X = prepare_rating_features(consumer_id, restaurant_id)
            y_pred = rating_model.predict(X)[0]
            prediction = round(float(y_pred), 2)
            features_preview = X.to_dict(orient="records")[0]
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template(
        "rating.html",
        consumer_ids=consumer_ids,
        restaurant_ids=restaurant_ids,
        prediction=prediction,
        features_preview=features_preview,
    )

@app.route("/cuisine", methods=["GET", "POST"])
def cuisine():
    top3 = None
    if request.method == "POST":
        try:
            payload = {
                "Age": int(request.form.get("age")),
                "Budget": request.form.get("budget"),
                "Drink_Level": request.form.get("drink_level"),
                "Transportation_Method": request.form.get("transportation"),
                "Marital_Status": request.form.get("marital_status"),
                "Children": request.form.get("children"),
                "Occupation": request.form.get("occupation"),
                "Smoker": request.form.get("smoker"),
            }
            X = pd.DataFrame([payload])
            probs = cuisine_model.predict_proba(X)[0]
            classes = cuisine_model.classes_
            top_idx = np.argsort(probs)[::-1][:3]
            top3 = [(classes[i], round(probs[i] * 100, 1)) for i in top_idx]
        except Exception as e:
            flash(f"Cuisine prediction error: {e}", "danger")

    return render_template("cuisine.html", top3=top3)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
