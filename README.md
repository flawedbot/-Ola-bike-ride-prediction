# 🚲 Ola Bike Demand Prediction AI

This is a modular Machine Learning application designed to predict bike-sharing demand based on temporal and environmental features. The project features a custom **Black and Purple** web interface for real-time predictions.

## 🌟 Project Overview
The goal of this project is to optimize fleet distribution for Ola bike ride requests. By analyzing historical data, weather patterns, and temporal features like hour and day of the week, the model provides an automated prediction of demand.

## 🛠️ Tech Stack
*   **Language:** Python 3.14
*   **Framework:** Flask (Backend)
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Random Forest Regressor)
*   **Frontend:** HTML5/CSS3 with a Black & Purple aesthetic
*   **OS:** Windows

## 📁 Project Structure
```text
ola_forecasting/
├── app.py                     # Flask application (The "Bridge")
├── src/                       # Source code directory
│   ├── components/            # Pipeline components
│   │   ├── data_ingestion.py  # Splits raw data into train/test
│   │   ├── data_transformation.py # Preprocesses features
│   │   └── model_trainer.py   # Trains the ML brain
├── artifacts/                 # Generated models and CSVs (Automatic)
├── notebook/
│   └── data/
│       └── ola_raw.csv        # Your cleaned dataset
└── templates/
    └── index.html             # Black & Purple dashboard

🚀 How to Run
1. Clone the repository
Bash
git clone https://github.com/flawedbot/-Ola-bike-ride-prediction.git
cd -Ola-bike-ride-prediction
2. Install dependencies
Bash
pip install -r requirements.txt
3. Run the notebook
Bash
jupyter notebook
💡 Future Improvements
Add XGBoost / advanced models
Build a Streamlit web app for real-time predictions
Deploy the model on cloud (AWS / Render)
Integrate real-time weather data
Improve feature engineering
