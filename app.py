import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle

# --- Constants ---
TEAMS = [
    'Royal Challengers Bangalore', 'Mumbai Indians', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Sunrisers Hyderabad', 'Rajasthan Royals',
    'Chennai Super Kings', 'Delhi Capitals'
]

CITIES = [
    'Chandigarh', 'Chennai', 'Bangalore', 'Mumbai', 'Dharamsala',
    'Hyderabad', 'Cuttack', 'Jaipur', 'Raipur', 'Delhi', 'Nagpur',
    'Kolkata', 'Indore', 'Centurion', 'Ahmedabad', 'Abu Dhabi',
    'East London', 'Durban', 'Pune', 'Visakhapatnam', 'Mohali',
    'Johannesburg', 'Cape Town', 'Bengaluru', 'Sharjah',
    'Port Elizabeth', 'Kimberley', 'Ranchi', 'Bloemfontein'
]

# --- Load model safely ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("pipe.joblib")  # recommended format
    except FileNotFoundError:
        with open("pipe.pkl", "rb") as f:  # fallback
            return pickle.load(f)

try:
    pipe = load_model()
except Exception as e:
    st.error("⚠️ Could not load model. Make sure pipe.joblib/pipe.pkl exists and matches your Python & scikit-learn versions.")
    st.exception(e)
    st.stop()

# --- UI ---
st.title("🏏 IPL Win Predictor")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select batting team", sorted(TEAMS))
with col2:
    bowling_team = st.selectbox("Select bowling team", sorted(TEAMS))

city = st.selectbox("Select city", sorted(CITIES))
target = st.number_input("Target", min_value=1, max_value=500, value=160)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Score", min_value=0, max_value=500, value=80)
with col4:
    overs = st.number_input("Overs completed (full overs only)", min_value=0, max_value=20, value=10, step=1)
with col5:
    wickets_fallen = st.number_input("Wickets fallen", min_value=0, max_value=10, value=2)

# --- Prediction ---
if st.button("Predict Probability"):
    if batting_team == bowling_team:
        st.error("Batting and bowling team cannot be the same.")
        st.stop()

    runs_left = max(0, target - score)
    balls_bowled = overs * 6
    balls_left = max(0, 120 - balls_bowled)
    wickets_left = 10 - wickets_fallen

    # Handle CRR & RRR
    crr = score / overs if overs > 0 else 0.0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else (0.0 if runs_left == 0 else float("inf"))

    # Prepare input row
    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets": wickets_left,
        "total_runs_x": target,
        "current_run_rate": crr,
        "req_run_rate": 0.0 if not np.isfinite(rrr) else rrr,
    }])

    try:
        result = pipe.predict_proba(input_df)[0]
        lossprob, winprob = float(result[0]), float(result[1])

        st.subheader("📊 Prediction Result")
        st.metric(label=f"{batting_team} Win Probability", value=f"{round(winprob*100)}%")
        st.metric(label=f"{bowling_team} Win Probability", value=f"{round(lossprob*100)}%")
        st.caption(f"Required Run Rate: {round(rrr,2) if np.isfinite(rrr) else '∞'}   •   Balls Left: {balls_left}")

        st.progress(min(1.0, max(0.0, winprob)))
    except Exception as e:
        st.error("Model prediction failed. Check that the training pipeline and inputs match.")
        st.exception(e)
