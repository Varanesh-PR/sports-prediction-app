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

# Load data
df = pd.read_csv("matches.csv")
df = df[['team1', 'team2', 'toss_winner', 'winner']].dropna()

# Encode data
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df[['team1', 'team2', 'toss_winner']]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")

# UI inputs
teams = list(encoders['team1'].classes_)

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)
    team2 = st.selectbox("Select Team 2", teams)

with col2:
    toss_winner = st.selectbox("Toss Winner", teams)

# Prediction
if st.button("🔮 Predict Winner"):
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

# ---------------- GRAPH SECTION ---------------- #

st.subheader("📊 Match Insights")

# Decode back for graph
df_graph = pd.read_csv("matches.csv")

# Wins count graph
fig, ax = plt.subplots()
df_graph['winner'].value_counts().plot(kind='bar', ax=ax)
plt.title("Matches Won by Each Team")
plt.xticks(rotation=45)

st.pyplot(fig)

# Toss impact
fig2, ax2 = plt.subplots()
pd.crosstab(df_graph['toss_winner'], df_graph['winner']).plot(ax=ax2)
plt.title("Toss Winner vs Match Winner")

st.pyplot(fig2)
