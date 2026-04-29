import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="IPL Predictor", page_icon="🏏")

# Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏏 IPL Match Winner Predictor</h1>", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("matches.csv")

# Keep required columns
df = df[['team1', 'team2', 'toss_winner', 'batsman', 'runs', 'winner']].dropna()

# Encode categorical data
encoders = {}
for col in ['team1', 'team2', 'toss_winner', 'batsman', 'winner']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df[['team1', 'team2', 'toss_winner', 'batsman', 'runs']]
y = df['winner']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")

# UI inputs
teams = list(encoders['team1'].classes_)
players = list(encoders['batsman'].classes_)

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)
    team2 = st.selectbox("Select Team 2", teams)

with col2:
    toss_winner = st.selectbox("Toss Winner", teams)
    batsman = st.selectbox("Select Batsman", players)

runs = st.slider("Runs Scored", 0, 120, 30)

# Prediction
if st.button("🔮 Predict Winner"):
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'batsman': [batsman],
        'runs': [runs]
    })

    for col in ['team1', 'team2', 'toss_winner', 'batsman']:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)
    winner = encoders['winner'].inverse_transform(prediction)

    st.success(f"🏆 Predicted Winner: {winner[0]}")

# ---------------- GRAPH SECTION ---------------- #

st.subheader("📊 Data Insights")

df_original = pd.read_csv("matches.csv")

# Wins per team
fig1, ax1 = plt.subplots()
df_original['winner'].value_counts().plot(kind='bar', ax=ax1)
plt.title("Matches Won by Each Team")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Runs distribution
fig2, ax2 = plt.subplots()
df_original['runs'].plot(kind='hist', bins=15, ax=ax2)
plt.title("Runs Distribution")
st.pyplot(fig2)

# Toss vs Winner
fig3, ax3 = plt.subplots()
pd.crosstab(df_original['toss_winner'], df_original['winner']).plot(ax=ax3)
plt.title("Toss Winner vs Match Winner")
st.pyplot(fig3)
