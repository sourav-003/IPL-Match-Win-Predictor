# 🏏 IPL Match Win Predictor — Live Win Probability (T20 · IPL)

[**▶ Live Demo on Railway**](https://web-production-9dbd2.up.railway.app/)

Predict the probability that the **chasing team** wins an IPL match in real-time using features derived from ball-by-ball data.  
This project includes a complete workflow: **data preprocessing, feature engineering, model training (scikit-learn), pipeline serialization, and a Streamlit frontend** for live predictions.

---

## ✨ Features
- Predicts **chasing team win probability** based on current match context.  
- Interactive **Streamlit web app** with simple user inputs.  
- Serialized **sklearn Pipeline** (`pipe.pkl`) for easy inference.  
- Includes **data cleaning, feature engineering, and evaluation** in the Jupyter notebook.  

---

## 🧠 How it Works

### 1. Data
- **`matches.csv`** → match metadata (`id`, `city`, `team1`, `team2`, `winner`, …)  
- **`deliveries.csv`** → ball-by-ball events (`match_id`, `over`, `ball`, `batting_team`, `bowling_team`, `total_runs`, `player_dismissed`, …)  

**Filtering:**
- Matches with **`dl_applied == 1`** removed (rain-affected).  
- Only modern franchises kept:  
- Sunrisers Hyderabad  
- Mumbai Indians  
- Royal Challengers Bangalore  
- Kolkata Knight Riders  
- Kings XI Punjab  
- Chennai Super Kings

---


### 2. Label (Target Variable)
Binary outcome:  
- `1` → Batting (chasing) team won.  
- `0` → Otherwise.  

---

### 3. Feature Engineering (2nd Innings Only)
For each ball of the chase:
- `runs_left` = target − current_score  
- `balls_left` = 126 − (over × 6 + ball)  
- `wickets` = cumulative dismissals  
- `total_runs_x` = target score  
- `current_run_rate` = (current_score × 6) ÷ balls_bowled  
- `req_run_rate` = (runs_left × 6) ÷ balls_left  

Categorical: `batting_team`, `bowling_team`, `city`  

Final features:  
- batting_team  
- bowling_team  
- city  
- runs_left  
- balls_left  
- wickets  
- total_runs_x  
- current_run_rate  
- req_run_rate  


---

### 4. Model
- **Pipeline:**  
  - `ColumnTransformer` with `OneHotEncoder` on categorical features.  
  - Logistic Regression (`liblinear`) as primary model.  
  - Random Forest also tested.  
- **Train/Test Split:** 80/20 stratified.  
- **Evaluation:** Accuracy + Precision/Recall/F1 report.  

Serialized to: **`pipe.pkl`**

---

## 🗂️ Project Structure
├── IPL_Match_Win_Predictor.ipynb # Data prep + model training
├── matches.csv # Match data
├── deliveries.csv # Ball-by-ball data
├── pipe.pkl # Trained pipeline
├── app.py # Streamlit app (frontend)
├── requirements.txt # Dependencies
└── README.md

---

## 🚀 Quickstart

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Place Data
Ensure **matches.csv** and **deliveries.csv** are in the root folder.

## 3. (Optional) Retrain
Run the Jupyter notebook to regenerate the trained pipeline:
```bash
jupyter notebook IPL_Match_Win_Predictor.ipynb
```

---

## 4. Run Streamlit App

To launch the web application locally, run the following command:

```bash
streamlit run app.py
```

---

## 🧪 Example Usage (Programmatic Inference)

You can also use the trained pipeline (`pipe.pkl`) directly in Python:

```python
import pickle
import pandas as pd

# Load trained pipeline
pipe = pickle.load(open("pipe.pkl", "rb"))

# Sample match scenario
row = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Mumbai Indians",
    "city": "Mumbai",
    "runs_left": 83,
    "balls_left": 57,
    "wickets": 3,
    "total_runs_x": 168,
    "current_run_rate": 8.2,
    "req_run_rate": 8.74
}

# Convert to DataFrame
X = pd.DataFrame([row])

# Predict win probability for chasing team
print("Win probability (chasing):", pipe.predict_proba(X)[0, 1])
```

---

## 🌐 Deployment

This project can be deployed on **Railway, Heroku, Render**, or similar platforms.

**Entrypoint command:**
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
⚠️ **Important:**  
Ensure that **scikit-learn versions match** between training and deployment environments, otherwise the pickle model (`pipe.pkl`) may not load correctly.

---

## 📦 Requirements
- pandas  
- numpy  
- scikit-learn  
- streamlit  
- matplotlib  
- seaborn  

---

## 📊 Evaluation
- Dataset class balance: ~46% vs 54%  
- Logistic Regression baseline performed well  
- Random Forest also explored  

---

## ⚠️ Limitations
- Trained only on historical IPL data  
- Ignores toss, pitch conditions, player form  
- Venue approximated by city  
- Pickle model not portable across sklearn versions  

---

## 🛣️ Roadmap
- ✅ Probability calibration (Platt/Isotonic)  
- ✅ Include toss, venue, and season features  
- ✅ Try boosting models (XGBoost/LightGBM)  
- ✅ Expose REST API endpoint  
- ✅ Add unit tests for edge cases  

---

## 🙌 Acknowledgements
- IPL open datasets (`matches.csv`, `deliveries.csv`)  
- Libraries: scikit-learn, pandas, Streamlit, Matplotlib, Seaborn  

---

## 🔗 Links
- **Live App**: Railway Demo  
- **Notebook**: `IPL_Match_Win_Predictor.ipynb`  
- **Model**: `pipe.pkl`  




- Rajasthan Royals  
- Delhi Capitals  
