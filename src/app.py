# app.py
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ===============================
# Paths
# ===============================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    consumers = pd.read_excel(DATA_DIR / "consumers.xlsx")
    restaurants = pd.read_excel(DATA_DIR / "restaurants.xlsx")
    cuisines = pd.read_excel(DATA_DIR / "restaurant_cuisines.xlsx")
    preferences = pd.read_excel(DATA_DIR / "consumer_preferences.xlsx")

    # Primary cuisine per restaurant
    cuisines_primary = (
        cuisines.sort_values(["Restaurant_ID", "Cuisine"])
        .drop_duplicates(subset=["Restaurant_ID"], keep="first")
        .rename(columns={"Cuisine": "Primary_Cuisine"})
    )

    # Merge restaurant cuisines
    restaurants = restaurants.merge(cuisines_primary, on="Restaurant_ID", how="left")

    return consumers, restaurants, preferences


# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    rating_model = joblib.load(MODELS_DIR / "xgb_rating_model.joblib")
    cuisine_model = joblib.load(MODELS_DIR / "cuisine_classifier.joblib")
    return rating_model, cuisine_model


# ===============================
# Prepare Features for Rating Prediction
# ===============================
def prepare_features(consumer_row, restaurant_row, preferences):
    pref = preferences.loc[
        preferences["Consumer_ID"] == consumer_row["Consumer_ID"].values[0],
        "Preferred_Cuisine",
    ]
    preferred_cuisine = pref.iloc[0] if not pref.empty else "Unknown"

    cuisine_match = int(
        restaurant_row["Primary_Cuisine"].values[0] == preferred_cuisine
    )

    data = {
        "Age": consumer_row["Age"].values[0],
        "Budget": consumer_row["Budget"].values[0],
        "Drink_Level": consumer_row["Drink_Level"].values[0],
        "Transportation_Method": consumer_row["Transportation_Method"].values[0],
        "Price": restaurant_row["Price"].values[0],
        "Alcohol_Service": restaurant_row["Alcohol_Service"].values[0],
        "Smoking_Allowed": restaurant_row["Smoking_Allowed"].values[0],
        "Franchise": restaurant_row["Franchise"].values[0],
        "Area": restaurant_row["Area"].values[0],
        "Parking": restaurant_row["Parking"].values[0],
        "Primary_Cuisine": restaurant_row["Primary_Cuisine"].values[0],
        "Preferred_Cuisine": preferred_cuisine,
        "Cuisine_Match": cuisine_match,
    }

    features = pd.DataFrame([data])

    # Clean datatypes
    num_like = ["Age", "Cuisine_Match"]
    for c in num_like:
        if c in features.columns:
            features[c] = pd.to_numeric(features[c], errors="coerce")
            features[c] = features[c].fillna(features[c].median())

    for c in features.columns:
        if c not in num_like:
            features[c] = features[c].astype(str).replace("nan", "Unknown").fillna("Unknown")

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="Restaurant ML Predictor", page_icon="🍽️", layout="centered")
    st.title("🍽️ Restaurant Intelligence Dashboard")
    st.caption("Predict restaurant ratings and recommend cuisines with Machine Learning")

    consumers, restaurants, preferences = load_data()
    rating_model, cuisine_model = load_models()

    # Create two tabs
    tab1, tab2 = st.tabs(["⭐ Rating Prediction", "🍜 Cuisine Recommendation"])

    # =======================
    # TAB 1: Rating Prediction
    # =======================
    with tab1:
        st.subheader("⭐ Predict Consumer’s Restaurant Rating")

        consumer_ids = consumers["Consumer_ID"].sort_values().unique().tolist()
        restaurant_ids = restaurants["Restaurant_ID"].sort_values().unique().tolist()

        selected_consumer = st.selectbox("Select Consumer ID", consumer_ids)
        selected_restaurant = st.selectbox("Select Restaurant ID", restaurant_ids)

        if st.button("Predict Rating"):
            try:
                consumer_row = consumers.loc[consumers["Consumer_ID"] == selected_consumer]
                restaurant_row = restaurants.loc[restaurants["Restaurant_ID"] == selected_restaurant]
                features = prepare_features(consumer_row, restaurant_row, preferences)

                pred = rating_model.predict(features)[0]

                st.success(f"Predicted Overall Rating: **{pred:.2f} / 2.00**")

                with st.expander("Show Feature Details"):
                    st.dataframe(features)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # =======================
    # TAB 2: Cuisine Recommendation
    # =======================
    with tab2:
        st.subheader("🍜 Recommend Cuisine Preferences")

        st.write("Predict what cuisine type a consumer is most likely to prefer based on their lifestyle and habits.")

        c = {}
        c["Age"] = st.number_input("Age", 18, 80, 25)
        c["Budget"] = st.selectbox("Budget", ["Low", "Medium", "High"])
        c["Drink_Level"] = st.selectbox("Drink Level", ["Abstemious", "Social Drinker", "Casual Drinker"])
        c["Transportation_Method"] = st.selectbox("Transportation", ["Public", "Car Owner", "Walk"])
        c["Marital_Status"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        c["Children"] = st.selectbox("Children", ["None", "Yes"])
        c["Occupation"] = st.selectbox("Occupation", ["Student", "Professional", "Other"])
        c["Smoker"] = st.selectbox("Smoker", ["Yes", "No"])

        if st.button("Predict Preferred Cuisine"):
            input_df = pd.DataFrame([c])
            try:
                probs = cuisine_model.predict_proba(input_df)[0]
                classes = cuisine_model.classes_

                top_idx = np.argsort(probs)[::-1][:3]
                top_cuisines = [(classes[i], probs[i]) for i in top_idx]

                st.subheader("🍽️ Top-3 Recommended Cuisines")
                for cuisine, p in top_cuisines:
                    st.write(f"- **{cuisine}** → {p*100:.1f}% confidence")

                best_cuisine = top_cuisines[0][0]
                st.success(f"✅ Most likely cuisine: **{best_cuisine}**")

            except Exception as e:
                st.error(f"Error during cuisine prediction: {e}")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit + XGBoost + RandomForest")


if __name__ == "__main__":
    main()
