import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Krisha ML", layout="centered")
st.title("🏠 Оценка стоимости недвижимости")

# ===== FEATURES FROM NOTEBOOK =====
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
    *CATEGORICAL_FEATURES
]

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv("merged_df.csv")

    df = df[(df["price"] > 0) & (df["area"] > 10)]
    df["price_per_m2"] = df["price"] / df["area"]
    df["log_price_per_m2"] = np.log(df["price_per_m2"])

    df = df.dropna(subset=BASE_FEATURES + ["log_price_per_m2"])
    return df

df = load_data()

# ===== TRAIN MODEL =====
@st.cache_resource
def train_model(df):
    X = df[BASE_FEATURES]
    y = df["log_price_per_m2"]

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        feature_names=X_train.columns.tolist()
    )

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(train_pool)
    return model

model = train_model(df)

# ===== UI INPUTS =====
st.sidebar.header("Параметры квартиры")

user_input = {}

user_input["area"] = st.sidebar.number_input("Площадь (м²)", 20.0, 500.0, 50.0)
user_input["room_count"] = st.sidebar.number_input("Кол-во комнат", 1, 10, 2)
user_input["floor"] = st.sidebar.number_input("Этаж", 1, 50, 3)
user_input["floor_count"] = st.sidebar.number_input("Этажность дома", 1, 50, 9)
user_input["construction_year"] = st.sidebar.number_input("Год постройки", 1950, 2025, 2015)
user_input["ceiling_height"] = st.sidebar.number_input("Высота потолков", 2.0, 5.0, 2.7)
user_input["first_floor"] = st.sidebar.selectbox("Первый этаж", [0, 1])
user_input["last_floor"] = st.sidebar.selectbox("Последний этаж", [0, 1])

for col in CATEGORICAL_FEATURES:
    user_input[col] = st.sidebar.selectbox(
        col,
        df[col].astype(str).unique().tolist()
    )

# ===== PREDICTION =====
if st.button("🔮 Рассчитать цену"):
    input_df = pd.DataFrame([user_input])

    pool = Pool(
        input_df,
        cat_features=CATEGORICAL_FEATURES,
        feature_names=input_df.columns.tolist()
    )

    log_price_m2 = model.predict(pool)[0]
    price_per_m2 = np.exp(log_price_m2)
    total_price = price_per_m2 * user_input["area"]

    st.success("💰 Результат оценки")
    st.metric("Цена за м²", f"{price_per_m2:,.0f} ₸")
    st.metric("Итоговая цена", f"{total_price:,.0f} ₸")