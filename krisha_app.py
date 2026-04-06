import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import os
import pickle

feature_impact_data = {
    "Feature": [
        "condition",
        "construction_year",
        "complex_class",
        "ceiling_height",
        "owner",
        "area",
        "house_type",
        "parking",
        "floor_count",
        "room_count",
        "bathroom_info",
        "elevator",
        "floor",
        "last_floor",
        "first_floor"
    ],
    "Monetary Impact (KZT per m²)": [
        31321.96,
        27628.11,
        26595.28,
        26059.63,
        14829.47,
        13294.54,
        11608.35,
        11568.23,
        9024.09,
        8811.56,
        5848.01,
        3220.63,
        2837.62,
        2609.03,
        431.88
    ]
}

pricing_proportions_data = {
    "Pricing Label": ["Fair price", "Overpriced", "Underpriced", "Strongly overpriced"],
    "Proportion (%)": [84.51, 10.42, 3.21, 1.87]
}

# Owner vs Realtor median overpricing percentage
owner_vs_realtor_data = {
    "Type": ["owner", "agent"],
    "Median Overpricing %": [-0.000819, -0.002661]
}

# Convert to DataFrames for easy display in Streamlit
feature_df = pd.DataFrame(feature_impact_data)
pricing_df = pd.DataFrame(pricing_proportions_data)
owner_realtor_df = pd.DataFrame(owner_vs_realtor_data)


# ================== CONFIG ==================
st.set_page_config(
    page_title="Apartment Price Analyzer | Astana",
    layout="wide"
)

MODEL_PATH = "/Users/allikhankoshamet/Desktop/ml/catboost_model.pkl"
DATA_PATH = "/Users/allikhankoshamet/Desktop/ml/merged_df.csv"

# ================== FEATURES ==================
CATEGORICAL_FEATURES = [
    "condition",
    "house_type",
    "complex_class",
    "bathroom_info",
    "owner",
    "parking",
    "elevator",
    "in_pledge"
]

BASE_FEATURES = [
    "area",
    "room_count",
    "floor",
    "floor_count",
    "construction_year",
    "ceiling_height",
    "first_floor",
    "last_floor",
] + CATEGORICAL_FEATURES

LOCATION_FEATURES = [
    "distance_to_center",
    "distance_to_botanical_garden",
    "distance_to_triathlon_park",
    "distance_to_astana_park",
    "distance_to_treatment_facility",
    "distance_to_railway_station_1",
    "distance_to_railway_station_2",
    "distance_to_industrial_zone"
]

ENHANCED_FEATURES = BASE_FEATURES + LOCATION_FEATURES

# ================== DATA + MODEL ==================
@st.cache_resource
def get_model_and_data():
    df = pd.read_csv(DATA_PATH)

    # ---- Basic cleaning ----
    df = df[(df["price"] > 0) & (df["area"] > 10)]
    df["price_per_m2"] = df["price"] / df["area"]
    df["log_price_per_m2"] = np.log(df["price_per_m2"])

    # ---- Outliers ----
    q_low = df["price_per_m2"].quantile(0.01)
    q_high = df["price_per_m2"].quantile(0.99)
    df = df[(df["price_per_m2"] > q_low) & (df["price_per_m2"] < q_high)]

    # ---- Categoricals ----
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("missing").astype(str)

    # ---- Drop NA ----
    df = df.dropna(subset=ENHANCED_FEATURES + ["log_price_per_m2"])

    # ---- Load or Train ----
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        X = df[ENHANCED_FEATURES]
        y = df["log_price_per_m2"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES
        )

        model = CatBoostRegressor(
            iterations=1000,
            depth=8,
            learning_rate=0.03,
            loss_function="MAE",
            random_seed=42,
            verbose=False
        )

        model.fit(train_pool)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    # ---- Fair price for dataset ----
    full_pool = Pool(
        df[ENHANCED_FEATURES],
        cat_features=CATEGORICAL_FEATURES
    )

    df["pred_log_price"] = model.predict(full_pool)
    df["fair_price_per_m2"] = np.exp(df["pred_log_price"])
    df["fair_price"] = df["fair_price_per_m2"] * df["area"]
    df["overpricing_pct"] = (
        (df["price"] - df["fair_price"]) / df["fair_price"] * 100
    )

    return model, df


model, df = get_model_and_data()

# ================== UI ==================
st.title("🏠 Apartment Price Analyzer — Astana")

st.sidebar.header("🔮 Predict Fair Price")

# ---- Inputs ----
area = st.sidebar.number_input("Area (m²)", 10.0, 500.0, 50.0)
room_count = st.sidebar.selectbox("Rooms", [1, 2, 3, 4, 5, 6])
floor = st.sidebar.number_input("Floor", 1, 50, 5)
floor_count = st.sidebar.number_input("Total floors", 1, 50, 10)
construction_year = st.sidebar.number_input("Construction year", 1950, 2026, 2010)
ceiling_height = st.sidebar.number_input("Ceiling height (m)", 2.0, 5.0, 2.7)

first_floor = st.sidebar.checkbox("First floor")
last_floor = st.sidebar.checkbox("Last floor")

condition = st.sidebar.selectbox("Condition", df["condition"].unique())
house_type = st.sidebar.selectbox("House type", df["house_type"].unique())
complex_class = st.sidebar.selectbox("Complex class", df["complex_class"].unique())
bathroom_info = st.sidebar.selectbox("Bathroom", df["bathroom_info"].unique())
owner = st.sidebar.selectbox("Owner", df["owner"].unique())
parking = st.sidebar.selectbox("Parking", df["parking"].unique())
elevator = st.sidebar.selectbox("Elevator", df["elevator"].unique())
in_pledge = st.sidebar.selectbox("In pledge", ["0", "1"])

# ---- Location ----
st.sidebar.subheader("📍 Location distances (km)")
dist_center = st.sidebar.number_input("Center", 0.0, 30.0, 5.0)
dist_botanical = st.sidebar.number_input("Botanical garden", 0.0, 30.0, 5.0)
dist_triathlon = st.sidebar.number_input("Triathlon park", 0.0, 30.0, 5.0)
dist_astana_park = st.sidebar.number_input("Astana park", 0.0, 30.0, 5.0)
dist_treatment = st.sidebar.number_input("Treatment facility", 0.0, 30.0, 5.0)
dist_rail1 = st.sidebar.number_input("Railway station 1", 0.0, 30.0, 5.0)
dist_rail2 = st.sidebar.number_input("Railway station 2", 0.0, 30.0, 5.0)
dist_industrial = st.sidebar.number_input("Industrial zone", 0.0, 30.0, 5.0)

# ---- Input DF ----
input_df = pd.DataFrame([{
    "area": area,
    "room_count": room_count,
    "floor": floor,
    "floor_count": floor_count,
    "construction_year": construction_year,
    "ceiling_height": ceiling_height,
    "first_floor": int(first_floor),
    "last_floor": int(last_floor),
    "condition": condition,
    "house_type": house_type,
    "complex_class": complex_class,
    "bathroom_info": bathroom_info,
    "owner": owner,
    "parking": parking,
    "elevator": elevator,
    "in_pledge": in_pledge,
    "distance_to_center": dist_center,
    "distance_to_botanical_garden": dist_botanical,
    "distance_to_triathlon_park": dist_triathlon,
    "distance_to_astana_park": dist_astana_park,
    "distance_to_treatment_facility": dist_treatment,
    "distance_to_railway_station_1": dist_rail1,
    "distance_to_railway_station_2": dist_rail2,
    "distance_to_industrial_zone": dist_industrial
}])

# ---- Predict ----
if st.sidebar.button("💰 Predict price"):
    pool = Pool(input_df, cat_features=CATEGORICAL_FEATURES)
    log_pred = model.predict(pool)[0]

    price_m2 = np.exp(log_pred)
    total_price = price_m2 * area

    st.success("### Fair Market Value")
    col1, col2 = st.columns(2)
    col1.metric("Price per m²", f"{price_m2:,.0f} KZT")
    col2.metric("Total price", f"{total_price:,.0f} KZT")

# ================== TABS ==================
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Overpriced",
    "🔥 Underpriced",
    "📊 District stats",
    "📈 Analysis Dashboard"
])

with tab1:
    st.subheader("Top overpriced apartments")
    st.dataframe(
        df.sort_values("overpricing_pct", ascending=False)
        .head(15)[[
            "name", "address", "district", "area", "room_count",
            "price", "fair_price", "overpricing_pct"
        ]]
    )

with tab2:
    st.subheader("Top underpriced apartments")
    st.dataframe(
        df.sort_values("overpricing_pct")
        .head(15)[[
            "district", "area", "room_count",
            "price", "fair_price", "overpricing_pct"
        ]]
    )

with tab3:
    st.subheader("Average price per m² by district")
    avg_price = (
        df.groupby("district")["price_per_m2"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(avg_price)

with tab4:
    st.header("Feature Monetary Impacts")
    st.dataframe(feature_df.style.format({"Monetary Impact (KZT per m²)": "{:.2f}"}))

    st.header("Pricing Label Proportions")
    st.dataframe(pricing_df.style.format({"Proportion (%)": "{:.2f}"}))

    st.header("Owner vs Realtor Median Overpricing")
    st.dataframe(owner_realtor_df.style.format({"Median Overpricing %": "{:.6f}"}))

    st.subheader("Example Code Snippet for Owner vs Realtor Analysis")
    st.code("""
    owner_vs_realtor = (
    data.groupby("owner")["overpricing_pct"]
    .median()
    .sort_values(ascending=False)
    )
    print(owner_vs_realtor)
    """, language="python")