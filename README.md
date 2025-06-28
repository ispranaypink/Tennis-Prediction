# WhatIf-Tennis: Tennis Match Predictor
Ever wanted to pit prime Federer against peak Djokovic? Wondered how Nadal would fare against Sampras on hard courts? This app lets you settle legendary "what ifs" using real ATP data and machine learning.
#### Try it out: [tennis-iknowwhowins](https://tennis-iknowwhowins.streamlit.app/)
## How it works
- Pick any two players - from any era
- The app analyzes each player's stats and playing patterns based on their selected ranking point in their career.
- The model predicts the likely winner

### Model currently used: RandomForestClassifier trained on ATP match data
### Accuracy: 68.2%

## Still working on:
- Neural Networks - (hopefully gives a better accuracy)
- Elo based ratings - (might give better accuracy, according to sports prediction articles)
- Tournament mode - predict an entire Grand Slam
- Clean UI by removing noisy metrics like odds/points
- Backend automation for live data updates

## Run Locally
```
git clone https://github.com/ispranaypink/WhatIf-Tennis.git
cd WhatIf-Tennis
pip install -r requirements.txt
streamlit run src/app.py
```
