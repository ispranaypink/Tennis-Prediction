import streamlit as st
import pandas as pd


import pickle
with open('./notebooks/rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

# Load your matches data for player codes and stats
matches = pd.read_csv("C:/Users/prana/Working directory/Projects repository/data-science/Tennis-prediction/data/t-1_cleaned_data.csv")

# Prepare player lists
players = sorted(set(matches['Player_1']).union(set(matches['Player_2'])))

st.title("Tennis Match Outcome Predictor")

# User selects players
player1 = st.selectbox("Select Player 1", players)
player2 = st.selectbox("Select Player 2", [p for p in players if p != player1])

# Get player codes and stats
def get_player_info(player, col_name):
    row = matches[matches[col_name] == player].iloc[0]
    return row

player1_info = get_player_info(player1, "Player_1")
player2_info = get_player_info(player2, "Player_2")

# User can override or input stats
rank1 = st.number_input("Player 1 Rank", value=int(player1_info["Rank_1"]))
rank2 = st.number_input("Player 2 Rank", value=int(player2_info["Rank_2"]))
pts1 = st.number_input("Player 1 Points", value=int(player1_info["Pts_1"]))
pts2 = st.number_input("Player 2 Points", value=int(player2_info["Pts_2"]))
odd1 = st.number_input("Player 1 Odds", value=float(player1_info["Odd_1"]))
odd2 = st.number_input("Player 2 Odds", value=float(player2_info["Odd_2"]))

# Get codes
player1_code = int(player1_info["playerCode"])
player2_code = int(player2_info["opponentCode"])

# Prepare input for prediction
predictors = ["playerCode", "opponentCode", "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]
input_data = pd.DataFrame([{
    "playerCode": player1_code,
    "opponentCode": player2_code,
    "Rank_1": rank1,
    "Rank_2": rank2,
    "Pts_1": pts1,
    "Pts_2": pts2,
    "Odd_1": odd1,
    "Odd_2": odd2
}])

if st.button("Predict Winner"):
    prediction = rf.predict(input_data)[0]
    proba = rf.predict_proba(input_data)[0]
    winner = player1 if prediction == 1 else player2
    st.success(f"Predicted Winner: {winner}")
    st.write(f"Probability {player1}: {proba[1]:.2f}")
    st.write(f"Probability {player2}: {proba[0]:.2f}")