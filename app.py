from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load models
match_model = joblib.load("match_winner_model.pkl")
live_model = joblib.load("live_model.pkl")

matches = pd.read_csv("matches.csv")

# ---------- PRE MATCH ----------
def team_win_pct(df, team):
    p = df[(df.team1==team)|(df.team2==team)]
    if len(p)==0: return 0.5
    return (p.winner==team).mean()

def recent_form(df, team, n=10):
    r = df[(df.team1==team)|(df.team2==team)].tail(n)
    if len(r)==0: return 0.5
    return (r.winner==team).mean()

def h2h(df, t1, t2):
    h = df[((df.team1==t1)&(df.team2==t2))|((df.team1==t2)&(df.team2==t1))]
    if len(h)==0: return 0.5
    return (h.winner==t1).mean()

@app.get("/")
def home():
    return {"status":"IPL AI Live Predictor Running"}

@app.get("/predict")
def predict(team1:str, team2:str):
    t1_win = team_win_pct(matches,team1)
    t2_win = team_win_pct(matches,team2)

    t1_recent = recent_form(matches,team1)
    t2_recent = recent_form(matches,team2)

    h2h_val = h2h(matches,team1,team2)

    X = [[t1_win-t2_win, t1_recent-t2_recent, h2h_val-0.5]]
    prob = match_model.predict_proba(X)[0][1]

    return {
        team1: round(prob*100,2),
        team2: round((1-prob)*100,2)
    }

# ---------- LIVE PREDICTION ----------
@app.get("/live_predict")
def live_predict(runs:int, overs:float, wickets:int, target:int):

    balls = int(overs * 6)
    balls_left = 120 - balls
    runs_left = target - runs
    wickets_left = 10 - wickets

    current_rr = runs / (balls/6)
    required_rr = runs_left / (balls_left/6)

    X = [[runs_left, balls_left, wickets_left, current_rr, required_rr]]

    prob = live_model.predict_proba(X)[0][1]

    return {
        "Batting team win %": round(prob*100,2),
        "Bowling team win %": round((1-prob)*100,2)
    }

