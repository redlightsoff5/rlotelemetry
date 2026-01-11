import os
import time
import json
from typing import Optional, List, Dict, Any

import fastf1
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

# -----------------------------
# FastF1 cache (persistent disk recommended on Render)
# -----------------------------
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

ALLOWED_YEARS = [2025, 2026]

app = FastAPI(title="RLO Telemetry API", version="1.0.0")

# CORS for Vercel/Netlify frontend
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _latest_completed_event(year: int):
    sched = fastf1.get_event_schedule(year, include_testing=False)
    # use EventDate when available; keep events <= today
    today = pd.Timestamp.utcnow().normalize()
    if "EventDate" in sched.columns:
        past = sched[sched["EventDate"] <= today]
    else:
        # fallback: if column names differ
        date_col = [c for c in sched.columns if "Date" in c][0]
        past = sched[sched[date_col] <= today]
    if past.empty:
        return None
    return past.iloc[-1]

def _get_session(year: int, event: str, session: str):
    try:
        s = fastf1.get_session(year, event, session)
        # minimize loads; telemetry only when needed
        s.load(telemetry=False, weather=False, messages=False)
        return s
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load session: {e}")

def _driver_team_map(session) -> Dict[str, str]:
    # Use session.laps which includes Team per driver for that session
    laps = session.laps
    if laps is None or len(laps) == 0:
        return {}
    df = laps[["Driver", "Team"]].dropna().drop_duplicates()
    return dict(zip(df["Driver"], df["Team"]))

# Canonical team color palette (broadcast-style)
TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "McLaren": "#FF8000",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "Sauber": "#52E252",
    "Haas": "#B6BABD",
}

TEAM_ALIASES = {
    # 2024–2026 naming variations (FastF1 strings vary)
    "RB": "Racing Bulls",
    "Visa Cash App RB": "Racing Bulls",
    "Visa Cash App RB F1 Team": "Racing Bulls",
    "Racing Bulls": "Racing Bulls",
    "AlphaTauri": "Racing Bulls",
    "Kick Sauber": "Sauber",
    "Sauber": "Sauber",
    "Stake F1 Team": "Sauber",
    "Stake": "Sauber",
    "Alfa Romeo": "Sauber",
}

def team_color(team: str) -> str:
    if not team:
        return "#111111"
    canonical = TEAM_ALIASES.get(team, team)
    return TEAM_COLORS.get(canonical, "#111111")

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/years")
def years():
    return {"years": ALLOWED_YEARS}

@app.get("/events")
def events(year: int = Query(..., description="Season year, 2025 or 2026")):
    if year not in ALLOWED_YEARS:
        raise HTTPException(status_code=400, detail="Year not allowed")
    sched = fastf1.get_event_schedule(year, include_testing=False)
    # Return only past events so UI always works
    today = pd.Timestamp.utcnow().normalize()
    if "EventDate" in sched.columns:
        sched = sched[sched["EventDate"] <= today]
    # Minimal fields to populate dropdowns
    cols = [c for c in ["EventName", "RoundNumber", "EventDate", "Location", "Country"] if c in sched.columns]
    out = sched[cols].copy()
    out = out.sort_values(cols[1] if "RoundNumber" in cols else cols[0])
    return {"events": out.to_dict(orient="records")}

@app.get("/default")
def defaults():
    # Pick latest year that has at least one past event
    for y in reversed(ALLOWED_YEARS):
        ev = _latest_completed_event(y)
        if ev is not None:
            return {"year": y, "event": ev["EventName"], "session": "Race"}
    # fallback
    return {"year": 2025, "event": "Bahrain Grand Prix", "session": "Race"}

@app.get("/session/meta")
def session_meta(
    year: int,
    event: str,
    session: str = Query("Race"),
):
    if year not in ALLOWED_YEARS:
        raise HTTPException(status_code=400, detail="Year not allowed")
    s = _get_session(year, event, session)
    d2t = _driver_team_map(s)
    drivers = sorted(list(d2t.keys()))
    return {
        "year": year,
        "event": event,
        "session": session,
        "drivers": drivers,
        "driverTeams": d2t,
        "teamColors": {drv: team_color(team) for drv, team in d2t.items()},
        "watermark": "@redlightsoff5",
    }

@app.get("/charts/pace")
def chart_pace(
    year: int,
    event: str,
    session: str = Query("Race"),
    drivers: Optional[str] = Query(None, description="Comma-separated driver codes, optional"),
    max_laps: int = Query(80, ge=10, le=200),
):
    if year not in ALLOWED_YEARS:
        raise HTTPException(status_code=400, detail="Year not allowed")

    s = fastf1.get_session(year, event, session)
    # This endpoint needs lap times; telemetry False keeps it lighter
    s.load(telemetry=False, weather=False, messages=False)

    laps = s.laps
    if laps is None or len(laps) == 0:
        return {"series": [], "x": []}

    # Clean + derive lap time seconds
    df = laps[["Driver", "LapNumber", "LapTime", "Team"]].dropna()
    df = df[df["LapTime"].notna()]
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Optional driver filter
    if drivers:
        want = [d.strip() for d in drivers.split(",") if d.strip()]
        df = df[df["Driver"].isin(want)]

    # Limit laps for payload size
    df = df[df["LapNumber"] <= max_laps]

    # x axis = lap numbers present
    x = sorted(df["LapNumber"].unique().tolist())

    # Build series
    series = []
    for drv, g in df.sort_values(["Driver", "LapNumber"]).groupby("Driver"):
        team = g["Team"].dropna().iloc[0] if "Team" in g.columns and not g["Team"].dropna().empty else ""
        y_map = dict(zip(g["LapNumber"], g["LapTimeSeconds"]))
        y = [y_map.get(l, None) for l in x]
        series.append({
            "name": drv,
            "color": team_color(team),
            "data": y,
        })

    # Sort series by best lap (fastest)
    def best(s):
        vals = [v for v in s["data"] if isinstance(v, (int, float))]
        return min(vals) if vals else 1e9
    series = sorted(series, key=best)

    return {"x": x, "series": series, "watermark": "@redlightsoff5"}
