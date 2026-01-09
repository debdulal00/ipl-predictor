from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

# 🔓 Allow website to access API (CORS fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & data
model = joblib.load("match_winner_model.pkl")
matches = pd.read_csv("matches.csv")

# Feature functions
def team_win_pct(df, team):
    p = df[(df.team1==team)|(df.team2==team)]
    if len(p)==0: 
        return 0.5
    return (p.winner==team).mean()

def recent_form(df, team, n=10):
    r = df[(df.team1==team)|(df.team2==team)].tail(n)
    if len(r)==0: 
        return 0.5
    return (r.winner==team).mean()

def h2h(df, t1, t2):
    h = df[((df.team1==t1)&(df.team2==t2))|((df.team1==t2)&(df.team2==t1))]
    if len(h)==0: 
        return 0.5
    return (h.winner==t1).mean()

@app.get("/")
def home():
    return {"status":"IPL Predictor API running"}

@app.get("/predict")
def predict(team1: str, team2: str):
    hist = matches

    t1_win = team_win_pct(hist, team1)
    t2_win = team_win_pct(hist, team2)

    t1_recent = recent_form(hist, team1)
    t2_recent = recent_form(hist, team2)

    h2h_val = h2h(hist, team1, team2)

    X = [[
        t1_win - t2_win,
        t1_recent - t2_recent,
        h2h_val - 0.5
    ]]

    prob = model.predict_proba(X)[0][1]

    return {
        team1: round(prob * 100, 2),
        team2: round((1 - prob) * 100, 2)
    }

