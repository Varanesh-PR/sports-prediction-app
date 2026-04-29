import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🏏 IPL Match Winner Predictor")

# Load dataset
df = pd.read_csv("matches.csv")
df = df[['team1', 'team2', 'toss_winner', 'winner']].dropna()

# Encode data
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Train model
X = df[['team1', 'team2', 'toss_winner']]
y = df['winner']

model = RandomForestClassifier()
model.fit(X, y)

# UI
teams = list(encoders['team1'].classes_)

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
toss_winner = st.selectbox("Toss Winner", teams)

if st.button("Predict Winner"):
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner]
    })

    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)
    winner = encoders['winner'].inverse_transform(prediction)

    st.success(f"🏆 Predicted Winner: {winner[0]}")