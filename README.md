# 🏠 Apartment Price Analyzer – Astana

A powerful **Streamlit** web application designed to predict fair market prices for apartments in **Astana (Nur-Sultan)**, Kazakhstan, using a **CatBoost** machine learning model.

This app not only predicts fair prices but also analyzes the market by identifying overpriced and underpriced listings, providing district statistics, and featuring an analytical dashboard with insights on feature impact.

## ✨ Key Features

- **Accurate Price Prediction**: Predicts fair price per square meter and total price using over 24 features (including distances to key locations).
- **Market Analysis**: Identifies overpriced and underpriced apartments in real time.
- **District Statistics**: Displays average price per square meter by district with an interactive bar chart.
- **Feature Impact Analysis**: Shows the monetary impact of each feature on the price.
- **Owner vs. Agent Insights**: Compares median overpricing between private owners and realtors.
- **Modern User Interface**: Features a clean layout with tabs and sidebar controls.
- **Location Awareness**: Considers distances to key landmarks in Astana.

## 🛠 Tech Stack

- **Python**
- **Streamlit**: Interactive web interface
- **CatBoost**: High-performance gradient boosting model
- **Pandas & NumPy**: Data processing
- **scikit-learn**: Data splitting
- **Pickle**: Model serialization

## 📁 Project Structure

```
apartment-price-analyzer/
├── krisha_app.py
├── catboost_model.pkl
├── merged_df.csv
├── README.md
└── requirements.txt
```


## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/apartment-price-analyzer.git
cd apartment-price-analyzer

pip install -r requirements.txt
streamlit run krisha_app.py
