import pandas as pd
import numpy as np


data = pd.read_csv("C:/Users/prana/Working directory/Projects repository/data-science/Tennis-prediction/data/atp_tennis.csv")


def parse_simple_score(score):
    
    p1_sets, p2_sets = 0, 0
    
    # Split into individual sets (e.g., ['6-4', '7-6'])
    sets = score.split()
    
    for set_score in sets:
        p1, p2 = map(int, set_score.split('-'))
        p1_sets += 1 if p1 > p2 else 0
        p2_sets += 1 if p2 > p1 else 0
    
    return p1_sets, p2_sets

def clean_data(raw_df):
    df = raw_df.copy()
    
    #Replace '-1' with NaN
    df['Odd_1'] = df['Odd_1'].replace(-1, np.nan)
    df['Odd_2'] = df['Odd_2'].replace(-1, np.nan)
    df['Pts_1'] = df['Pts_1'].replace(-1, np.nan)
    df['Pts_2'] = df['Pts_2'].replace(-1, np.nan)

    #Change Date format from Object to date-time
    df['Date'] = pd.to_datetime(df['Date'])  

    #Change these columns from  Object to category for better ML predictions
    categorical_cols = ['Series', 'Court', 'Surface', 'Round']
    df[categorical_cols] = df[categorical_cols].astype('category')

    #make a winner encoded column for better ML Accuracy
    df['Winner_encoded'] = (df['Winner'] == df['Player_1']).astype(int)

    #Split score column into two columns of who won how many sets
    df[['Player1_SetsWon', 'Player2_SetsWon']] = df['Score'].apply(
    lambda x: pd.Series(parse_simple_score(x))
    )

    #downcast rank, winner encoded dtypes
    df[['Rank_1', 'Rank_2']] = df[['Rank_1', 'Rank_2']].astype('int32')
    df['Winner_encoded'] = df['Winner_encoded'].astype('int8')

    #change player dtype from object to category    
    df[['Player_1', 'Player_2']] = df[['Player_1', 'Player_2']].astype('category')

    #replace -1 in ranks to NaN
    df['Rank_1'] = df['Rank_1'].replace(-1, np.nan)
    df['Rank_2'] = df['Rank_2'].replace(-1, np.nan)


    return df

fresh_data = clean_data(data)
fresh_data.to_csv("./data/cleaned-atp-tennis.csv")