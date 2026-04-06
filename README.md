# 🏠 Apartment Price Analyzer — Astana

A powerful **Streamlit** web application that uses a **CatBoost** machine learning model to predict fair market prices of apartments in **Astana (Nur-Sultan)**, Kazakhstan.

The app not only predicts the fair price but also analyzes the market by identifying overpriced and underpriced listings, provides district statistics, and includes an analytical dashboard with feature impact insights.

## ✨ Key Features

- **Accurate Price Prediction** — Predicts fair price per m² and total price using 24+ features (including location distances)
- **Market Analysis** — Identifies overpriced and underpriced apartments in real time
- **District Statistics** — Average price per m² by district with interactive bar chart
- **Feature Impact Analysis** — Shows monetary impact of each feature on price
- **Owner vs Agent Insights** — Compares median overpricing between private owners and realtors
- **Modern UI** — Clean layout with tabs and sidebar controls
- **Location-Aware** — Takes into account distances to key landmarks in Astana

## 🛠 Tech Stack

- **Python**
- **Streamlit** — Interactive web interface
- **CatBoost** — High-performance gradient boosting model
- **Pandas & NumPy** — Data processing
- **scikit-learn** — Data splitting
- **Pickle** — Model serialization

## 📁 Project Structure
apartment-price-analyzer/
├── krisha_app.py             
├── catboost_model.pkl       
├── merged_df.csv             
├── README.md
└── requirements.txt
