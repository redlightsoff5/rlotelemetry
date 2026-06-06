import os, warnings, logging, traceback, glob, json
from types import SimpleNamespace
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# Default template (rlo_dark) is registered below, once the palette is defined.

from dash import Dash, dcc, html, Input, Output, State, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# Pre-computed parquet data (built by precompute.py / GitHub Action)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(APP_DIR, "data"))

# Years supported in the UI (keep this list tight on purpose)
YEARS_ALLOWED = [2025, 2026]

def _utc_today_token() -> str:
    # used to refresh cached schedules daily without redeploys
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

def default_year_value() -> int:
    """Pick the newest year that has at least one completed event; fallback to first allowed year."""
    today = pd.Timestamp.utcnow().tz_localize(None)
    for y in sorted(YEARS_ALLOWED, reverse=True):
        try:
            df = get_schedule_df(y, _utc_today_token())
            if not df.empty and (df['EventDate'] <= today).any():
                return y
        except Exception:
            continue
    return YEARS_ALLOWED[0]

# Brand
SITE_TITLE = "Telemetry by RedLightsOff"
WATERMARK  = "@redlightsoff5"

IG_URL  = os.getenv("IG_URL", "https://instagram.com/redlightsoff5")
BMC_URL = os.getenv("BMC_URL", "https://buymeacoffee.com/redlightsoff5")

# ---- Dark F1 broadcast palette ----
COL_BG      = "#0b0b0f"
COL_PANEL   = "rgba(0,0,0,0)"          # transparent: let the dark card show through
COL_CARD    = "#16161f"
COL_RED     = "#e10600"                # F1 red
COL_TEXT    = "#f4f4f8"
COL_MUTED   = "#9aa0ad"
COL_GRID    = "rgba(255,255,255,0.08)"
COL_AXIS    = "rgba(255,255,255,0.22)"
FONT_FAMILY = "Titillium Web, Segoe UI, Arial, sans-serif"

# Dark Plotly template applied to every figure
pio.templates["rlo_dark"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COL_TEXT, family=FONT_FAMILY, size=13),
    title=dict(font=dict(color="#ffffff", size=16)),
    colorway=["#e10600", "#3671C6", "#27F4D2", "#FF8000", "#64C4FF",
              "#52E252", "#229971", "#6692FF", "#B6BABD", "#f4f4f8"],
    xaxis=dict(gridcolor=COL_GRID, linecolor=COL_AXIS, zeroline=False,
               tickfont=dict(color=COL_MUTED), title=dict(font=dict(color=COL_MUTED))),
    yaxis=dict(gridcolor=COL_GRID, linecolor=COL_AXIS, zeroline=False,
               tickfont=dict(color=COL_MUTED), title=dict(font=dict(color=COL_MUTED))),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                font=dict(color=COL_TEXT), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#1d1d28", bordercolor="rgba(255,255,255,0.16)",
                    font=dict(color="#f4f4f8", family=FONT_FAMILY)),
))
pio.templates.default = "rlo_dark"

TEAM_COLORS = {
    # Broadcast-style team colors (used on F1 graphics; same for both drivers of a team)
    'Red Bull':      '#3671C6',
    'McLaren':       '#FF8000',
    'Ferrari':       '#E80020',
    'Mercedes':      '#27F4D2',
    'Aston Martin':  '#229971',
    'Alpine':        '#0093CC',
    'Williams':      '#64C4FF',
    'Racing Bulls':  '#6692FF',
    'Sauber':        '#52E252',
    'Haas':          '#B6BABD',
    'Cadillac':      '#000000'

}

# Map many possible FastF1 team strings to a canonical key in TEAM_COLORS
TEAM_ALIASES = {
    'red bull': 'Red Bull',
    'oracle red bull': 'Red Bull',
    'red bull racing': 'Red Bull',

    # VCARB / RB / Racing Bulls naming
    'rb f1': 'Racing Bulls',
    'racing bulls': 'Racing Bulls',
    'visa cash app rb': 'Racing Bulls',
    'vcarb': 'Racing Bulls',
    'rb': 'Racing Bulls',   # keep last as a catch-all

    'ferrari': 'Ferrari',
    'scuderia ferrari': 'Ferrari',

    'mercedes': 'Mercedes',
    'mercedes-amg': 'Mercedes',

    'mclaren': 'McLaren',

    'aston martin': 'Aston Martin',

    'alpine': 'Alpine',
    'bwt alpine': 'Alpine',

    'williams': 'Williams',
    'williams racing': 'Williams',

    # Sauber / Kick / Stake / Audi (bucketed for 2025–2026)
    'sauber': 'Sauber',
    'kick sauber': 'Sauber',
    'stake': 'Sauber',
    'stake f1': 'Sauber',
    'audi': 'Sauber',

    'haas': 'Haas',
    'haas f1': 'Haas',

    # Cadillac / GM (2026+)
    'cadillac': 'Cadillac',
    'gm cadillac': 'Cadillac',
    'cadillac f1': 'Cadillac',
    'cadillac f1 team': 'Cadillac',
    'andretti': 'Cadillac',          # por si FastF1 lo etiqueta así en algún punto

}

DRIVER_TEAM_OVERRIDE = {
    # Force team colour if Team string is missing/odd in some sessions
    (2026, 'PER'): 'Cadillac',
    (2026, 'BOT'): 'Cadillac',
}

def canonical_team(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    for key, canon in TEAM_ALIASES.items():
        if key in s:
            return canon
    return name  # fallback

COMMON_LAYOUT = dict(
    paper_bgcolor=COL_PANEL,
    plot_bgcolor=COL_PANEL,
    font=dict(color=COL_TEXT, family=FONT_FAMILY),
    margin=dict(l=14, r=14, t=54, b=46)
)

def brand(fig):
    """Apply common styling + centered watermark to a Plotly figure."""
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=COL_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COL_GRID, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=40, color="rgba(255,255,255,0.05)", family=FONT_FAMILY),
        xanchor="center", yanchor="middle",
    )
    return fig

def fig_empty(title):
    f = go.Figure()
    f.update_layout(title=title, **COMMON_LAYOUT)
    return brand(f)

def polish(fig, grid=False):
    fig.update_layout(**COMMON_LAYOUT)
    if not grid:
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
    return brand(fig)

def s_to_mssmmm(x):
    if pd.isna(x): return ""
    x = float(x)
    m = int(x // 60); s = int(x % 60); ms = int(round((x - int(x))*1000))
    return f"{m}:{s:02d}.{ms:03d}"

def is_race(ses):
    t = (getattr(ses, 'session_type', '') or '').upper()
    n = (getattr(ses, 'name', '') or '').upper()
    return t == 'R' or 'RACE' in n

# ---------- Live Schedule & Options ----------
@lru_cache(maxsize=8)
def get_schedule_df(year:int, date_token:str) -> pd.DataFrame:
    # IMPORTANT: include testing in schedule
    df = ff1.get_event_schedule(year, include_testing=True).copy()
    df['EventDate'] = pd.to_datetime(df['EventDate'])
    df['EventFormat'] = df['EventFormat'].astype(str).str.lower()
    df = df[['RoundNumber','EventName','EventFormat','EventDate']].sort_values(['EventDate','RoundNumber']).reset_index(drop=True)
    return df

def build_gp_options(year:int):
    df = get_schedule_df(year, _utc_today_token())

    # determine test_number by chronological order among testing events
    testing_df = df[df['EventFormat'] == 'testing'].sort_values('EventDate')
    test_dates = testing_df['EventDate'].dt.date.astype(str).tolist()

    opts = []
    for _, r in df.sort_values(['EventDate','RoundNumber']).iterrows():
        fmt = str(r.EventFormat).lower()
        date = r.EventDate.date()
        name = str(r.EventName)

        if fmt == "testing":
            test_number = test_dates.index(str(date)) + 1
            opts.append({
                "label": f"Test de pretemporada #{test_number} ({date})",
                "value": f"TEST|{test_number}"
            })
        else:
            opts.append({
                "label": f"R{int(r.RoundNumber)} — {name} ({date})",
                "value": f"GP|{name}"
            })
    return opts

def default_event_value(year:int):
    df = get_schedule_df(year, _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df['EventDate'] <= today].sort_values('EventDate')
    if past.empty:
        return None

    # Prefer the most recent event we have pre-computed → instant first paint.
    gp_past = past[past['EventFormat'].astype(str).str.lower() != 'testing']
    for _, ev in gp_past[::-1].iterrows():
        if read_cached_session(int(year), f"GP|{ev['EventName']}", 'R') is not None:
            return f"GP|{str(ev['EventName'])}"

    last = past.iloc[-1]
    if str(last['EventFormat']).lower() == 'testing':
        testing_df = df[df['EventFormat'] == 'testing'].sort_values('EventDate')
        test_dates = testing_df['EventDate'].dt.date.astype(str).tolist()
        test_number = test_dates.index(str(last['EventDate'].date())) + 1
        return f"TEST|{test_number}"

    return f"GP|{str(last['EventName'])}"

SESSION_OPTIONS = [
    {"label": "Libres 1 (FP1)",       "value": "FP1"},
    {"label": "Libres 2 (FP2)",       "value": "FP2"},
    {"label": "Libres 3 (FP3)",       "value": "FP3"},
    {"label": "Clasificación Sprint", "value": "SQ"},
    {"label": "Clasificación",        "value": "Q"},
    {"label": "Sprint",               "value": "SR"},
    {"label": "Carrera",              "value": "R"},
]

TEST_SESSION_OPTIONS = [
    {"label": "Día 1", "value": "T1"},
    {"label": "Día 2", "value": "T2"},
    {"label": "Día 3", "value": "T3"},
]

# ---------- Pre-computed data (served instantly; falls back to live FastF1) ----------
def _slug(name: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in str(name))
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")

class CachedSession:
    """Minimal stand-in for a FastF1 Session built from pre-computed parquet.
    Exposes only what the chart builders use: .laps, .results, .name,
    .session_type and .event.year."""
    def __init__(self, laps, results, session_type, name, year):
        self.laps = laps
        self.results = results if results is not None else pd.DataFrame()
        self.session_type = session_type
        self.name = name
        self.event = SimpleNamespace(year=year)

@lru_cache(maxsize=16)
def read_cached_session(year:int, event_value:str, sess_code:str):
    """Build a CachedSession from data/<year>/<slug>/<SESS>.* parquet, or None."""
    try:
        kind, payload = str(event_value).split("|", 1)
    except ValueError:
        return None
    if kind != "GP":
        return None  # testing stays live
    out_dir = os.path.join(DATA_DIR, str(int(year)), _slug(payload))
    laps_path = os.path.join(out_dir, f"{sess_code}.laps.parquet")
    if not os.path.exists(laps_path):
        return None
    try:
        laps = pd.read_parquet(laps_path)
        res_path = os.path.join(out_dir, f"{sess_code}.results.parquet")
        results = pd.read_parquet(res_path) if os.path.exists(res_path) else pd.DataFrame()
        meta = {}
        meta_path = os.path.join(out_dir, f"{sess_code}.meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        return CachedSession(laps, results, meta.get("session_type", ""),
                             meta.get("name", sess_code), int(year))
    except Exception:
        traceback.print_exc()
        return None

# ---------- Loaders ----------
@lru_cache(maxsize=8)
def load_session_laps(year:int, event_value:str, sess_code:str):
    """
    event_value:
      - 'GP|<EventName>'
      - 'TEST|<test_number>'  (1..)
    sess_code:
      - GP: FP1/FP2/FP3/SQ/Q/SR/R
      - TEST: T1/T2/T3
    """
    event_value = str(event_value)
    sess_code = str(sess_code).upper()

    cached = read_cached_session(int(year), event_value, sess_code)
    if cached is not None:
        return cached

    kind, payload = event_value.split("|", 1)

    if kind == "TEST":
        test_number = int(payload)
        day_number = int(sess_code.replace("T", ""))  # T1->1
        ses = ff1.get_testing_session(int(year), test_number, day_number)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        return ses

    # GP normal
    event_name = payload
    try:
        ses = ff1.get_session(int(year), event_name, sess_code)
    except Exception:
        if sess_code == "SQ":
            ses = ff1.get_session(int(year), event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses


@lru_cache(maxsize=8)
def load_session_results_only(year:int, event_value:str, sess_code:str):
    """Lighter loader (no laps) used to sum championship points across a season."""
    kind, payload = str(event_value).split("|", 1)
    if kind == "TEST":
        raise ValueError("testing sessions have no results")
    ses = ff1.get_session(int(year), payload, str(sess_code).upper())
    ses.load(laps=False, telemetry=False, weather=False, messages=False)
    return ses


@lru_cache(maxsize=8)
def season_standings(year:int, date_token:str):
    """Official driver & constructor standings from Ergast/jolpica — ONE fast,
    authoritative call (correct points, no per-race loading). Falls back to
    summing cached results only if the API is unavailable."""
    try:
        from fastf1.ergast import Ergast
        erg = Ergast(result_type='pandas', auto_cast=True)
        ds = erg.get_driver_standings(season=int(year)).content[0]
        cs = erg.get_constructor_standings(season=int(year)).content[0]
        drv_rows = []
        for _, r in ds.iterrows():
            ab = str(r.get('driverCode') or '').strip()
            name = (str(r.get('givenName') or '') + ' ' + str(r.get('familyName') or '')).strip()
            teams = r.get('constructorNames')
            team = (teams[-1] if isinstance(teams, (list, tuple)) and len(teams) else
                    (str(teams) if teams is not None and not isinstance(teams, (list, tuple)) else ''))
            drv_rows.append((ab or name, name or ab, canonical_team(team), float(r.get('points') or 0)))
        team_rows = [(canonical_team(str(r.get('constructorName') or '')), float(r.get('points') or 0))
                     for _, r in cs.iterrows()]
        if drv_rows:
            return drv_rows, team_rows
    except Exception:
        traceback.print_exc()
    return _season_standings_sum(int(year), date_token)


def _season_standings_sum(year:int, date_token:str):
    """Fallback: sum points from cached/live Race + Sprint results."""
    sched = get_schedule_df(year, date_token)
    today = pd.Timestamp.utcnow().tz_localize(None)
    done = sched[(sched['EventFormat'] != 'testing') & (sched['EventDate'] <= today)].sort_values('EventDate')

    drv_pts, drv_team, drv_name, team_pts = {}, {}, {}, {}
    for _, ev in done.iterrows():
        name = str(ev['EventName'])
        for code in ('R', 'SR'):                       # race + sprint
            ses = read_cached_session(int(year), f"GP|{name}", code)
            if ses is None:
                try:
                    ses = load_session_results_only(int(year), f"GP|{name}", code)
                except Exception:
                    continue
            res = getattr(ses, 'results', None)
            if res is None or res.empty or 'Points' not in res.columns:
                continue
            for _, r in res.iterrows():
                ab = r.get('Abbreviation')
                if not isinstance(ab, str) or not ab:
                    continue
                pts = r.get('Points')
                pts = 0.0 if pd.isna(pts) else float(pts)
                drv_pts[ab] = drv_pts.get(ab, 0.0) + pts
                tm = canonical_team(str(r.get('TeamName') or ''))
                if tm:
                    drv_team[ab] = tm
                    team_pts[tm] = team_pts.get(tm, 0.0) + pts
                fn = r.get('FullName')
                if isinstance(fn, str) and fn:
                    drv_name[ab] = fn

    drv_rows = [(ab, drv_name.get(ab, ab), drv_team.get(ab, ''), p)
                for ab, p in sorted(drv_pts.items(), key=lambda kv: kv[1], reverse=True)]
    team_rows = [(tm, p) for tm, p in sorted(team_pts.items(), key=lambda kv: kv[1], reverse=True)]
    return drv_rows, team_rows


# ---------- Schedule helpers (Home & Live tabs) ----------
SESSION_ES = {
    "Practice 1": "Libres 1", "Practice 2": "Libres 2", "Practice 3": "Libres 3",
    "Qualifying": "Clasificación", "Sprint": "Sprint", "Sprint Race": "Sprint",
    "Sprint Qualifying": "Clasificación Sprint", "Sprint Shootout": "Clasificación Sprint",
    "Race": "Carrera",
}

@lru_cache(maxsize=4)
def get_full_schedule(year:int, date_token:str):
    return ff1.get_event_schedule(int(year), include_testing=False)

def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _sessions_of(ev):
    out = []
    for i in range(1, 6):
        name = ev.get(f"Session{i}")
        dt = ev.get(f"Session{i}DateUtc")
        if name is None or (isinstance(name, float) and pd.isna(name)) or pd.isna(dt):
            continue
        out.append((str(name), _naive(dt)))
    return out

def next_session_info(year:int, token:str):
    """Dict for the next upcoming session of the season, or None."""
    try:
        df = get_full_schedule(year, token)
    except Exception:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    best = None
    for _, ev in df.iterrows():
        for name, dt in _sessions_of(ev):
            if dt > now and (best is None or dt < best["utc"]):
                best = {"event": str(ev["EventName"]),
                        "location": str(ev.get("Location", "") or ""),
                        "country": str(ev.get("Country", "") or ""),
                        "session": SESSION_ES.get(name, name), "utc": dt}
    return best

def live_session_info(year:int, token:str):
    """Dict if a session is happening right now (start..start+3h), else None."""
    try:
        df = get_full_schedule(year, token)
    except Exception:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    for _, ev in df.iterrows():
        for name, dt in _sessions_of(ev):
            if dt <= now <= dt + pd.Timedelta(hours=3):
                return {"event": str(ev["EventName"]),
                        "session": SESSION_ES.get(name, name), "started": dt}
    return None

def last_race_podium(year:int, token:str):
    """(event_name, [(pos, abbr, full, team), ...]) for the most recent completed race."""
    try:
        sched = get_schedule_df(year, token)
    except Exception:
        return None
    now = pd.Timestamp.utcnow().tz_localize(None)
    done = sched[(sched["EventFormat"] != "testing") & (sched["EventDate"] <= now)].sort_values("EventDate")
    for _, ev in done[::-1].iterrows():
        name = str(ev["EventName"])
        ses = read_cached_session(int(year), f"GP|{name}", "R")
        if ses is None:
            try:
                ses = load_session_results_only(int(year), f"GP|{name}", "R")
            except Exception:
                continue
        res = getattr(ses, "results", None)
        if res is None or res.empty or "Position" not in res.columns:
            continue
        top = res.sort_values("Position").head(3)
        podium = [(int(r["Position"]) if pd.notna(r["Position"]) else 0,
                   str(r.get("Abbreviation") or ""), str(r.get("FullName") or ""),
                   str(r.get("TeamName") or "")) for _, r in top.iterrows()]
        return name, podium
    return None


@lru_cache(maxsize=1)
def load_session_telemetry(year:int, event_value:str, sess_code:str):
    """Heavy loader (telemetry=True). maxsize=1 keeps only ONE session in RAM so
    the free Render instance (512 MB) does not run out of memory."""
    event_value = str(event_value)
    sess_code = str(sess_code).upper()
    kind, payload = event_value.split("|", 1)
    if kind == "TEST":
        ses = ff1.get_testing_session(int(year), int(payload), int(sess_code.replace("T", "")))
    else:
        try:
            ses = ff1.get_session(int(year), payload, sess_code)
        except Exception:
            if sess_code == "SQ":
                ses = ff1.get_session(int(year), payload, "SS")
            else:
                raise
    ses.load(laps=True, telemetry=True, weather=False, messages=False)
    return ses


def fastest_lap_telemetry(ses, driver):
    """Telemetry (Distance/Speed/X/Y/...) of a driver's fastest lap, or None."""
    try:
        lap = ses.laps.pick_drivers(driver).pick_fastest()
    except Exception:
        return None
    if lap is None:
        return None
    try:
        return lap.get_telemetry().add_distance()
    except Exception:
        return None

def circuit_corners(ses):
    """Corner markers (Number, Distance, X, Y) for the circuit, or None."""
    try:
        return ses.get_circuit_info().corners
    except Exception:
        return None

# ---------- Builders ----------
def driver_team_color_map(ses):
    laps = ses.laps[['Driver','Team']].copy() if hasattr(ses, "laps") else pd.DataFrame()
    if laps.empty or 'Driver' not in laps.columns:
        return {}

    year = None
    try:
        year = int(getattr(getattr(ses, "event", None), "year", None) or 0) or None
    except Exception:
        year = None

    # Normalize team per driver if present
    laps = laps.dropna(subset=['Driver'])
    team_series = None
    if 'Team' in laps.columns and laps['Team'].notna().any():
        team_series = laps.dropna(subset=['Team']).groupby('Driver')['Team'].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
        ).apply(canonical_team)

    # Build final map with overrides
    out = {}
    drivers = laps['Driver'].dropna().unique().tolist()
    for drv in drivers:
        forced = DRIVER_TEAM_OVERRIDE.get((year, drv)) if year else None
        team = forced or (team_series.get(drv) if team_series is not None and drv in team_series.index else None)
        team = canonical_team(team) if isinstance(team, str) else (forced or "")
        out[drv] = TEAM_COLORS.get(team, '#cccccc')

    return out

def gap_to_leader_df(ses):
    laps = ses.laps.copy().dropna(subset=['LapTime'])
    if laps.empty: return pd.DataFrame()
    if is_race(ses):
        laps['LapSeconds'] = laps['LapTime'].dt.total_seconds()
        laps['Cum'] = laps.groupby('Driver', dropna=False)['LapSeconds'].cumsum()
        lead = laps.groupby('LapNumber', dropna=False)['Cum'].min().rename('Lead').reset_index()
        d = laps.merge(lead, on='LapNumber', how='left')
        d['Gap_s'] = d['Cum'] - d['Lead']
        d['GapStr'] = d['Gap_s'].apply(s_to_mssmmm)
        return d[['Driver','LapNumber','Gap_s','GapStr']]
    best = laps.groupby('Driver', dropna=False)['LapTime'].min().rename('Best').reset_index()
    gbest = best['Best'].min()
    best['Gap_s'] = (best['Best'] - gbest).dt.total_seconds()
    best['GapStr'] = best['Gap_s'].apply(s_to_mssmmm)
    best['LapNumber'] = 1
    return best[['Driver','LapNumber','Gap_s','GapStr']]

def positions_gained_df(ses):
    res = ses.results
    if res is None or res.empty or not {'Abbreviation','GridPosition','Position'}.issubset(res.columns):
        return pd.DataFrame()
    df = res[['Abbreviation','GridPosition','Position']].copy()
    df['Driver'] = df['Abbreviation']
    df['PositionsGained'] = df['GridPosition'] - df['Position']
    return df.sort_values('PositionsGained', ascending=False)

def tyre_stints_df(ses):
    laps = ses.laps.copy()
    if laps.empty or 'Compound' not in laps.columns: return pd.DataFrame()
    laps['Compound'] = laps['Compound'].astype(str).str.upper()
    agg = (laps.groupby(['Driver','Stint','Compound'], dropna=False)
               .agg(LapStart=('LapNumber','min'), LapEnd=('LapNumber','max')))
    agg['Laps'] = agg['LapEnd'] - agg['LapStart'] + 1
    return agg.reset_index().sort_values(['Driver','Stint'])

def pace_df(ses):
    laps = ses.laps.copy().dropna(subset=['LapTime'])
    if laps.empty: return pd.DataFrame()
    laps['LapSeconds'] = laps['LapTime'].dt.total_seconds().astype(float)
    laps['LapStr'] = laps['LapSeconds'].apply(s_to_mssmmm)
    return laps[['Driver','LapNumber','LapSeconds','LapStr']]

def sector_records_df(ses):
    laps = ses.laps.copy(); out=[]
    for i,c in enumerate(['Sector1Time','Sector2Time','Sector3Time'], start=1):
        if c not in laps.columns: continue
        idx = laps[c].idxmin()
        if pd.isna(idx): continue
        row = laps.loc[idx]
        out.append({'Sector':f'S{i}','Driver':row['Driver'],'Time (s)':round(row[c].total_seconds(),3)})
    return pd.DataFrame(out)

def speed_records_df(ses):
    laps = ses.laps.copy()
    cols = [c for c in ['SpeedI1','SpeedI2','SpeedFL','SpeedST'] if c in laps.columns]
    if cols:
        grp = laps.groupby('Driver', dropna=False)[cols].max().reset_index()
        return grp.rename(columns={'SpeedI1':'I1 (km/h)','SpeedI2':'I2 (km/h)','SpeedFL':'Finish (km/h)','SpeedST':'Trap (km/h)'})
    best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].dropna(subset=['LapTime'])
    rows=[]
    for _, r in best.iterrows():
        try:
            vmax = float(r.get_car_data().add_distance()['Speed'].max())
        except Exception:
            vmax = np.nan
        rows.append({'Driver': r['Driver'], 'Trap (km/h)': vmax})
    return pd.DataFrame(rows)

def export_df_for_chart(chart_key: str, ses, selected_drivers):
    """
    Returns a dataframe representing what the chart plots.
    Keep it deterministic: same filters as the chart.
    """
    selected_drivers = selected_drivers or []

    if chart_key == "gap":
        df = gap_to_leader_df(ses)
        if selected_drivers:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

    if chart_key == "lc":
        laps = ses.laps[['Driver','LapNumber','Position']].dropna()
        if selected_drivers:
            laps = laps[laps["Driver"].isin(selected_drivers)]
        return laps

    if chart_key == "ep":
        df = pace_df(ses)
        if selected_drivers:
            df = df[df["Driver"].isin(selected_drivers)]
        if not df.empty:
            df = df.sort_values(['Driver','LapNumber'])
            df["MA3"] = df.groupby("Driver", dropna=False)["LapSeconds"].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
        return df

    if chart_key == "pos":
        df = positions_gained_df(ses)
        if selected_drivers and not df.empty:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

    if chart_key == "ty":
        df = tyre_stints_df(ses)
        if selected_drivers:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

    if chart_key == "pace":
        df = pace_df(ses)
        if selected_drivers:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

    if chart_key == "best":
        laps = ses.laps.dropna(subset=['LapTime'])
        if selected_drivers:
            laps = laps[laps["Driver"].isin(selected_drivers)]
        if laps.empty:
            return pd.DataFrame()
        best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
        best['Best_s'] = best['LapTime'].dt.total_seconds()
        best['BestStr'] = best['Best_s'].apply(s_to_mssmmm)
        return best[['Driver','LapNumber','Best_s','BestStr']]

    if chart_key == "sec":
        df = sector_records_df(ses)
        if selected_drivers and not df.empty:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

    if chart_key == "spd":
        df = speed_records_df(ses)
        if selected_drivers and not df.empty:
            df = df[df["Driver"].isin(selected_drivers)]
        return df

# ================= Dash =================
external_stylesheets=[dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE  # browser tab title

app.index_string = f"""<!DOCTYPE html>
<html lang="es">
<head>
  {{%metas%}}
  <meta name="description" content="RLO Telemetry — análisis y telemetría de Fórmula 1 por @redlightsoff5: ritmo, neumáticos, telemetría, clasificación y resultados.">
  <title>{SITE_TITLE}</title>
  {{%favicon%}}
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;800;900&family=Roboto+Mono:wght@500;700&display=swap" rel="stylesheet">
  {{%css%}}
</head>
<body>
  <div class="rlo-navbar">
    <div class="logo-wrap">
      <img src="/assets/logo.png" alt="logo"/>
    </div>
    <div class="rlo-brand">
      <div class="rlo-title"><span>RLO</span> Telemetry</div>
      <div class="rlo-subtitle">by @redlightsoff5</div>
    </div>
    <div class="rlo-actions">
      <a class="rlo-action rlo-ig" href="{IG_URL}" target="_blank" rel="noopener noreferrer">Instagram</a>
      <a class="rlo-action rlo-bmc" href="{BMC_URL}" target="_blank" rel="noopener noreferrer">Apoyar</a>
    </div>
  </div>

  {{%app_entry%}}

  <footer>
    {{%config%}}
    {{%scripts%}}
    {{%renderer%}}
  </footer>
</body>
</html>
"""

def header_controls():
    y0 = default_year_value()
    return html.Div(className="rlo-filter", children=dbc.Row([
        dbc.Col([
            dbc.Label("Año"),
            dcc.Dropdown(
                id='year-dd',
                options=[{'label': str(y), 'value': y} for y in YEARS_ALLOWED],
                value=y0,
                clearable=False
            ),
            html.Div(id='year-warning', className="mt-1", style={'fontSize':'0.85rem','opacity':0.85})
        ], md=3),

        dbc.Col([
            dbc.Label("Gran Premio"),
            dcc.Dropdown(
                id='event-dd',
                options=build_gp_options(y0),
                value=default_event_value(y0),
                clearable=False,
                placeholder="Selecciona evento..."
            )
        ], md=6),

        dbc.Col([
            dbc.Label("Sesión"),
            dcc.Dropdown(
                id='session-dd',
                options=SESSION_OPTIONS,
                value='R',
                clearable=False
            )
        ], md=3),
    ], className="g-2 align-items-end"))

def graph_box(graph_id: str, title: str, chart_key: str):
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H5(title, className="m-0", style={"display":"inline-block", "marginRight":"10px"}),
                    dbc.Button(
                        "CSV",
                        id={"role": "csv", "chart": chart_key},
                        n_clicks=0,
                        size="sm",
                        outline=True,
                        color="secondary",
                        style={"padding":"2px 10px"}
                    ),
                    dcc.Download(id={"role":"csv-dl", "chart": chart_key}),
                ], style={"display":"flex", "alignItems":"center", "gap":"8px"}),
                md=6
            ),
            dbc.Col([
                dcc.Dropdown(
                    id={"role": "drv", "chart": chart_key},
                    multi=True,
                    placeholder="Filtrar pilotos (opcional)",
                    options=[],
                    value=[]
                )
            ], md=6),
        ], className="g-2 align-items-center"),

        dcc.Loading(
            dcc.Graph(
                id=graph_id,
                figure=fig_empty(title),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": "420px"}
            ),
            type="default"
        )
    ])

def tab_evolution():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box('gap','Diferencia (Gap)','gap'), md=6),
            dbc.Col(graph_box('lapchart','Posición por vuelta','lc'), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box('evo-pace','Evolución del ritmo','ep'), md=6),
            dbc.Col(graph_box('pos','Posiciones ganadas','pos'), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_tyres():
    return html.Div([ graph_box('tyre-strategy','Estrategia de neumáticos','ty') ])

def tab_pace():
    return html.Div([ graph_box('pace','Ritmo de carrera','pace') ])

def tab_records():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box('best-laps','Mejores vueltas','best'), md=6),
            dbc.Col(graph_box('sectors','Récords por sector','sec'), md=6),
        ], className="g-2")
    ])

def tab_speeds():
    return html.Div([ graph_box('speeds','Velocidades','spd') ])

# ---------- Table helpers (results & standings) ----------
def _fmt_pts(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return str(int(p)) if p.is_integer() else f"{p:g}"

def _td_str(td, with_hours=False):
    if td is None or pd.isna(td):
        return ""
    s = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    if with_hours and s >= 3600:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        return f"{h}:{m:02d}:{sec:06.3f}"
    return s_to_mssmmm(s)

def _team_color(team):
    return TEAM_COLORS.get(canonical_team(team or ""), "#cccccc")

def _driver_cell(full, abbr, team):
    kids = [html.Span(className="rlo-team-bar", style={"background": _team_color(team)})]
    if full:
        kids.append(html.Span(full, style={"fontWeight": 600}))
        if abbr:
            kids.append(html.Span(abbr, className="rlo-mono",
                                  style={"color": "var(--muted)", "marginLeft": "8px", "fontSize": "12px"}))
    else:
        kids.append(html.Span(abbr or "—", style={"fontWeight": 600}))
    return html.Span(kids, style={"display": "flex", "alignItems": "center"})

def _table(header, rows):
    thead = html.Thead(html.Tr([html.Th(h) for h in header]))
    tbody = html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows])
    return dbc.Table([thead, tbody], className="rlo-table", borderless=True, hover=True, responsive=True)

def build_results_table(ses, sess_code):
    code = str(sess_code).upper()
    res = getattr(ses, 'results', None)
    has_res = (res is not None and not res.empty
               and 'Abbreviation' in res.columns and 'Position' in res.columns)

    if has_res and code in ('R', 'SR'):
        rows = []
        for _, r in res.sort_values('Position').iterrows():
            pos = r.get('Position'); pos = "" if pd.isna(pos) else str(int(pos))
            t = r.get('Time'); status = str(r.get('Status') or "")
            if pd.notna(t):
                disp = _td_str(t, with_hours=True) if pos == "1" else "+" + _td_str(t)
            else:
                disp = status
            rows.append([
                html.Span(pos, className="rlo-pos"),
                _driver_cell(r.get('FullName'), r.get('Abbreviation'), str(r.get('TeamName') or "")),
                str(r.get('TeamName') or ""),
                html.Span(disp, className="rlo-mono"),
                html.Span(_fmt_pts(r.get('Points')), className="rlo-pts"),
            ])
        return _table(['Pos', 'Piloto', 'Equipo', 'Tiempo / Estado', 'Pts'], rows)

    if has_res and code in ('Q', 'SQ'):
        rows = []
        for _, r in res.sort_values('Position').iterrows():
            pos = r.get('Position'); pos = "" if pd.isna(pos) else str(int(pos))
            rows.append([
                html.Span(pos, className="rlo-pos"),
                _driver_cell(r.get('FullName'), r.get('Abbreviation'), str(r.get('TeamName') or "")),
                str(r.get('TeamName') or ""),
                html.Span(_td_str(r.get('Q1')), className="rlo-mono"),
                html.Span(_td_str(r.get('Q2')), className="rlo-mono"),
                html.Span(_td_str(r.get('Q3')), className="rlo-mono"),
            ])
        return _table(['Pos', 'Piloto', 'Equipo', 'Q1', 'Q2', 'Q3'], rows)

    return _fastest_lap_table(ses)   # practice / testing / no official results

def _fastest_lap_table(ses):
    laps = ses.laps.dropna(subset=['LapTime']) if hasattr(ses, 'laps') else pd.DataFrame()
    if laps.empty:
        return html.Div("Sin datos de esta sesión todavía.", className="rlo-note")
    best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
    best['s'] = best['LapTime'].dt.total_seconds()
    best = best.sort_values('s')
    counts = laps.groupby('Driver')['LapNumber'].count()
    rows = []
    for i, (_, r) in enumerate(best.iterrows(), start=1):
        drv = r['Driver']; team = str(r.get('Team') or "")
        rows.append([
            html.Span(str(i), className="rlo-pos"),
            _driver_cell(None, drv, team),
            canonical_team(team) or team,
            html.Span(s_to_mssmmm(r['s']), className="rlo-mono"),
            html.Span(str(int(counts.get(drv, 0)))),
        ])
    return _table(['Pos', 'Piloto', 'Equipo', 'Mejor vuelta', 'Vueltas'], rows)

def build_standings_drivers(drv_rows):
    if not drv_rows:
        return html.Div("Aún no hay puntos esta temporada.", className="rlo-note")
    rows = [[
        html.Span(str(i), className="rlo-pos"),
        _driver_cell(name, ab, team),
        team,
        html.Span(_fmt_pts(pts), className="rlo-pts"),
    ] for i, (ab, name, team, pts) in enumerate(drv_rows, start=1)]
    return _table(['Pos', 'Piloto', 'Equipo', 'Pts'], rows)

def build_standings_teams(team_rows):
    if not team_rows:
        return html.Div("Aún no hay puntos esta temporada.", className="rlo-note")
    rows = []
    for i, (tm, pts) in enumerate(team_rows, start=1):
        rows.append([
            html.Span(str(i), className="rlo-pos"),
            html.Span([html.Span(className="rlo-team-bar", style={"background": _team_color(tm)}),
                       html.Span(tm, style={"fontWeight": 600})],
                      style={"display": "flex", "alignItems": "center"}),
            html.Span(_fmt_pts(pts), className="rlo-pts"),
        ])
    return _table(['Pos', 'Equipo', 'Pts'], rows)

# ---------- New tab layouts ----------
def tab_telemetry():
    return html.Div([
        html.Div(
            "📡 Telemetría de la vuelta rápida. Para la VELOCIDAD elige 1–3 pilotos; "
            "para el DELTA y el MAPA DE DOMINANCIA elige exactamente 2. La primera carga "
            "descarga datos (unos segundos).",
            className="rlo-note"),
        graph_box('tel-speed', 'Velocidad vs distancia — vuelta rápida (con curvas)', 'tel_spd'),
        dbc.Row([
            dbc.Col(graph_box('tel-delta', 'Delta de tiempo (2 pilotos)', 'tel_delta'), md=6),
            dbc.Col(graph_box('tel-map', 'Mapa de dominancia / velocidad', 'tel_map'), md=6),
        ], className="g-2 mt-2"),
    ])

def tab_results():
    return html.Div([
        html.Div(className="box", children=[
            html.H5("Resultado de la sesión", className="m-0"),
            dcc.Loading(html.Div(id="results-table", className="mt-2"), type="default"),
        ]),
        html.Div("Clasificación del campeonato", className="rlo-section-title mt-3"),
        dbc.Row([
            dbc.Col(html.Div(className="box", children=[
                html.H5("Pilotos", className="m-0"),
                dcc.Loading(html.Div(id="standings-drivers", className="mt-2"), type="default"),
            ]), md=6),
            dbc.Col(html.Div(className="box", children=[
                html.H5("Constructores", className="m-0"),
                dcc.Loading(html.Div(id="standings-teams", className="mt-2"), type="default"),
            ]), md=6),
        ], className="g-2"),
    ])

# ---------- Home & Live tabs ----------
def _leader_row(rank, name, sub, pts, color):
    return html.Div(className="rlo-leader", children=[
        html.Span(str(rank), className="rlo-leader-rank"),
        html.Span(className="rlo-team-bar", style={"background": color, "height": "26px"}),
        html.Div([html.Div(name, className="rlo-leader-name"),
                  html.Div(sub, className="rlo-leader-sub")], style={"flex": "1", "minWidth": "0"}),
        html.Span(pts, className="rlo-leader-pts"),
    ])

def tab_home():
    return html.Div([
        html.Div(className="rlo-hero", children=[
            html.Div("RLO TELEMETRY", className="rlo-hero-title"),
            html.Div("Análisis y telemetría de Fórmula 1 — datos al instante, en español.",
                     className="rlo-hero-sub"),
        ]),
        dbc.Row([
            dbc.Col(html.Div(className="box rlo-next", children=[
                html.Div("PRÓXIMA SESIÓN", className="rlo-kicker"),
                dcc.Loading(html.Div(id="home-next"), type="default"),
            ]), md=5),
            dbc.Col(html.Div(className="box", children=[
                html.Div("ÚLTIMO PODIO", className="rlo-kicker"),
                dcc.Loading(html.Div(id="home-podium"), type="default"),
            ]), md=7),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(html.Div(className="box", children=[
                html.Div("LÍDERES — PILOTOS", className="rlo-kicker"),
                dcc.Loading(html.Div(id="home-drv"), type="default"),
            ]), md=6),
            dbc.Col(html.Div(className="box", children=[
                html.Div("LÍDERES — CONSTRUCTORES", className="rlo-kicker"),
                dcc.Loading(html.Div(id="home-team"), type="default"),
            ]), md=6),
        ], className="g-2 mt-1"),
        dcc.Store(id="home-target"),
        dcc.Interval(id="home-cd-int", interval=1000, n_intervals=0),
    ])

def tab_live():
    return html.Div([
        html.Div(id="live-status", className="mb-2"),
        html.Div(className="box", children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                            "flexWrap": "wrap", "marginBottom": "10px"}, children=[
                html.H5("Timing en vivo", className="m-0"),
                html.A("Abrir F1-Dash ↗", href="https://f1-dash.com", target="_blank",
                       rel="noopener noreferrer", className="rlo-action rlo-bmc"),
                html.A("Race Telemetry ↗", href="https://www.racetelemetry.com", target="_blank",
                       rel="noopener noreferrer", className="rlo-action"),
                html.A("F1 oficial ↗", href="https://www.formula1.com/en/timing/f1-live",
                       target="_blank", rel="noopener noreferrer", className="rlo-action"),
            ]),
            html.Iframe(src="https://www.f1telemetry.xyz/",
                        style={"width": "100%", "height": "70vh", "border": "0",
                               "borderRadius": "12px", "background": "#0c0c12"}),
            html.Div("El timing en vivo se sirve desde proveedores externos gratuitos. "
                     "Si no se ve aquí (algunos bloquean la incrustación), usa los botones de "
                     "arriba — F1-Dash es el más completo.",
                     className="rlo-note", style={"marginTop": "10px"}),
        ]),
    ])

app.layout = dbc.Container([
    header_controls(),
    dcc.Tabs(
        id="tabs",
        value="inicio",
        parent_className="rlo-tabs-parent",
        className="rlo-tabs",
        children=[
            dcc.Tab(label="Inicio", value="inicio", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="En vivo", value="live", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Evolución", value="evo", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Ritmo", value="pace", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Neumáticos", value="tyres", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Telemetría", value="tele", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Récords", value="records", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Velocidades", value="speeds", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Resultados", value="results", className="rlo-tab", selected_className="rlo-tab--selected"),
        ],
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_home()),
    dcc.Store(id='store'),
    dcc.Store(id='drivers-store'),
    dcc.Store(id='team-color-store')
], fluid=True, className='rlo-page')

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"inicio":tab_home, "live":tab_live, "evo":tab_evolution, "pace":tab_pace,
            "tyres":tab_tyres, "tele":tab_telemetry, "records":tab_records,
            "speeds":tab_speeds, "results":tab_results}.get(val, tab_home)()

# NEW: Session dropdown changes depending on whether event is TEST or GP
@app.callback(
    Output('session-dd','options'),
    Output('session-dd','value'),
    Input('event-dd','value'),
    State('session-dd','value')
)
def _event_changed_set_sessions(event_val, current):
    if not event_val:
        return SESSION_OPTIONS, 'R'

    kind = str(event_val).split("|", 1)[0]
    if kind == "TEST":
        valid = {o["value"] for o in TEST_SESSION_OPTIONS}
        new_val = current if current in valid else "T1"
        return TEST_SESSION_OPTIONS, new_val

    valid = {o["value"] for o in SESSION_OPTIONS}
    new_val = current if current in valid else "R"
    return SESSION_OPTIONS, new_val

# ===== Load session once =====
@app.callback(
    Output('store','data'),
    Output('drivers-store','data'),
    Output('team-color-store','data'),
    Input('year-dd','value'),
    Input('event-dd','value'),
    Input('session-dd','value')
)
def load_session_meta(year, event_value, sess_code):
    if not year or not event_value or not sess_code:
        return no_update, [], {}
    try:
        ses = load_session_laps(int(year), str(event_value), str(sess_code))
        laps = ses.laps.dropna(subset=['LapTime'])
        drivers = sorted(laps['Driver'].dropna().unique().tolist())
        colors = driver_team_color_map(ses)
        return {'year': int(year), 'event': str(event_value), 'sess': str(sess_code)}, drivers, colors
    except Exception:
        traceback.print_exc()
        return no_update, [], {}

# ===== Populate ONLY mounted dropdowns (incl. when switching tabs) =====
@app.callback(
    Output({'role':'drv','chart':ALL}, 'options'),
    Output({'role':'drv','chart':ALL}, 'value'),
    Input('drivers-store','data'),
    Input('tab-body','children'),
    State({'role':'drv','chart':ALL}, 'id')
)
def fill_dropdowns(drivers, _children, ids):
    drivers = drivers or []
    opts = [{'label': d, 'value': d} for d in drivers]
    out_opts, out_val = [], []
    for _id in ids:
        out_opts.append(opts)
        # Telemetry is heavy: start empty so nothing downloads until the user picks.
        out_val.append([] if _id.get('chart') in ('tel_spd', 'tel_map', 'tel_delta') else drivers)
    return out_opts, out_val

# ===== Year-driven GP list + guard against future-only seasons =====
@app.callback(
    Output('event-dd','options'),
    Output('event-dd','value'),
    Output('year-warning','children'),
    Input('year-dd','value'),
    State('event-dd','value')
)
def _year_changed(year, current_event):
    if year is None:
        return [], None, ""
    year = int(year)
    try:
        opts = build_gp_options(year)
        default_ev = default_event_value(year)
        valid = {o['value'] for o in (opts or [])}
        value = current_event if current_event in valid else default_ev
        warn = ""
        if default_ev is None:
            warn = f"Aún no hay eventos disputados en {year}. Selecciona 2025 para ver datos."
            value = None
        return opts, value, warn
    except Exception:
        return [], None, f"Calendario no disponible para {year}."

# ---------- helper to color by team ----------
def set_trace_color(fig, name_to_color):
    for tr in fig.data:
        c = (name_to_color or {}).get(tr.name)
        if c:
            tr.update(line=dict(color=c), marker=dict(color=c))
    return fig

# ---------- Evolution charts ----------
@app.callback(
    Output('gap','figure'),
    Input('store','data'), Input({'role':'drv','chart':'gap'}, 'value'),
    State('team-color-store','data')
)
def chart_gap(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    df = gap_to_leader_df(ses)
    if selected: df = df[df['Driver'].isin(selected)]
    if df.empty: return fig_empty("Diferencia — sin datos")
    if is_race(ses):
        f = px.line(df, x='LapNumber', y='Gap_s', color='Driver', line_shape='spline', custom_data=['GapStr'], title='Diferencia con el líder (MM:SS.mmm)')
        f.update_traces(hovertemplate="%{fullData.name} — Vuelta %{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="s", tickformat=".3f")
    else:
        gg = df.sort_values('Gap_s')
        f = px.bar(gg, x='Driver', y='Gap_s', custom_data=['GapStr'], title='Diferencia con la mejor (MM:SS.mmm)')
        f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="s", tickformat=".3f")
    return set_trace_color(polish(f), color_map)

@app.callback(
    Output('lapchart','figure'),
    Input('store','data'), Input({'role':'drv','chart':'lc'}, 'value'),
    State('team-color-store','data')
)
def chart_lapchart(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    laps = ses.laps[['Driver','LapNumber','Position']].dropna()
    if selected: laps = laps[laps['Driver'].isin(selected)]
    if laps.empty: return fig_empty("Posición por vuelta — sin datos")
    f = px.line(laps, x='LapNumber', y='Position', color='Driver', title="Posición por vuelta (menor = mejor)")
    f.update_yaxes(autorange="reversed", dtick=1)
    return set_trace_color(polish(f), color_map)

@app.callback(
    Output('evo-pace','figure'),
    Input('store','data'), Input({'role':'drv','chart':'ep'}, 'value'),
    State('team-color-store','data')
)
def chart_evo(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    pdf = pace_df(ses)
    if selected: pdf = pdf[pdf['Driver'].isin(selected)]
    if pdf.empty: return fig_empty("Evolución — sin datos de vueltas")
    pdf = pdf.sort_values(['Driver','LapNumber'])
    pdf['MA3'] = pdf.groupby('Driver', dropna=False)['LapSeconds'].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f = go.Figure()
    for drv, d in pdf.groupby('Driver'):
        f.add_trace(go.Scatter(x=d['LapNumber'], y=d['MA3'], mode='lines', name=str(drv), line=dict(shape='spline', width=2.2),
                               hovertemplate=f"{drv} — Vuelta %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_yaxes(title="s (media 3 vueltas)", tickformat=".3f")
    f.update_layout(title="Evolución del ritmo (media 3 vueltas)")
    return set_trace_color(polish(f), color_map)

@app.callback(
    Output('pos','figure'),
    Input('store','data'), Input({'role':'drv','chart':'pos'}, 'value')
)
def chart_pos(data, selected):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    df = positions_gained_df(ses)
    if selected and not df.empty: df = df[df['Driver'].isin(selected)]
    if df.empty: return fig_empty("Posiciones ganadas — (solo carrera / sin datos)")
    f = px.bar(df, x='Abbreviation', y='PositionsGained', title='Posiciones ganadas', text='PositionsGained')
    f.update_traces(marker_line_width=0)
    return polish(f)

# ---------- Tyres ----------
@app.callback(
    Output('tyre-strategy','figure'),
    Input('store','data'), Input({'role':'drv','chart':'ty'}, 'value')
)
def chart_tyres(data, selected):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    st = tyre_stints_df(ses)
    if selected: st = st[st['Driver'].isin(selected)]
    if st.empty: return fig_empty("Estrategia de neumáticos — sin datos")
    f = go.Figure()
    order = st['Driver'].unique().tolist()[::-1]
    cmap = {'SOFT':'#DA291C','MEDIUM':'#FFD12E','HARD':'#F0F0F0','INTERMEDIATE':'#43B02A','WET':'#00A3E0'}
    for _, r in st.iterrows():
        f.add_trace(go.Bar(x=[int(r['Laps'])], y=[r['Driver']], base=[int(r['LapStart'])-1],
                           orientation='h', marker_color=cmap.get(str(r['Compound']).upper(), '#888'),
                           showlegend=False,
                           hovertemplate=f"{r['Driver']} — {r['Compound']}<br>Lap {int(r['LapStart'])}–{int(r['LapEnd'])}<extra></extra>"))
    for n,c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))
    f.update_layout(title='Estrategia de neumáticos', barmode='stack',
                    yaxis=dict(categoryorder='array', categoryarray=order, title='Piloto'),
                    xaxis_title='Vuelta')
    return polish(f, grid=True)

# ---------- Pace ----------
@app.callback(
    Output('pace','figure'),
    Input('store','data'), Input({'role':'drv','chart':'pace'}, 'value'),
    State('team-color-store','data')
)
def chart_pace(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    pdf = pace_df(ses)
    if selected: pdf = pdf[pdf['Driver'].isin(selected)]
    if pdf.empty: return fig_empty("Ritmo — sin datos de vueltas")
    f = go.Figure()
    for drv, d in pdf.groupby('Driver'):
        f.add_trace(go.Scatter(x=d['LapNumber'], y=d['LapSeconds'], mode='lines+markers', line=dict(width=1.6), marker=dict(size=5),
                               name=str(drv),
                               hovertemplate=f"{drv} — Vuelta %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_yaxes(title="s", tickformat=".3f")
    f.update_layout(title="Ritmo vuelta a vuelta")
    return set_trace_color(polish(f), color_map)

# ---------- Records ----------
@app.callback(
    Output('best-laps','figure'),
    Input('store','data'), Input({'role':'drv','chart':'best'}, 'value'),
    State('team-color-store','data')
)
def chart_best(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    laps = ses.laps.dropna(subset=['LapTime'])
    if selected: laps = laps[laps['Driver'].isin(selected)]
    if laps.empty: return fig_empty("Mejores vueltas — sin datos")
    best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
    best['Best_s'] = best['LapTime'].dt.total_seconds()
    best['BestStr'] = best['Best_s'].apply(s_to_mssmmm)
    f = px.bar(best.sort_values('Best_s'), x='Driver', y='Best_s', custom_data=['BestStr'], title="Mejores vueltas")
    f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
    f.update_yaxes(title="s", tickformat=".3f")
    return set_trace_color(polish(f), color_map)

@app.callback(
    Output('sectors','figure'),
    Input('store','data'), Input({'role':'drv','chart':'sec'}, 'value')
)
def chart_sectors(data, selected):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    df = sector_records_df(ses)
    if selected and not df.empty: df = df[df['Driver'].isin(selected)]
    if df.empty: return fig_empty("Récords por sector — sin datos")
    f = go.Figure(data=[go.Table(
        header=dict(values=['Sector', 'Piloto', 'Tiempo (s)'],
                    fill_color='#1d1d28', line_color='rgba(255,255,255,0.12)',
                    align='left', height=30, font=dict(color='#f4f4f8', size=13)),
        cells=dict(values=[df['Sector'], df['Driver'], df['Time (s)']],
                   fill_color='rgba(255,255,255,0.02)', line_color='rgba(255,255,255,0.08)',
                   align='left', height=28, font=dict(color='#f4f4f8'))
    )])
    f.update_layout(title="Récords por sector", paper_bgcolor=COL_PANEL)
    return brand(f)

# ---------- Speeds ----------
@app.callback(
    Output('speeds','figure'),
    Input('store','data'), Input({'role':'drv','chart':'spd'}, 'value')
)
def chart_speeds(data, selected):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(int(data.get('year', 2025)), data['event'], data['sess'])
    spd = speed_records_df(ses)
    if selected and not spd.empty: spd = spd[spd['Driver'].isin(selected)]
    if spd.empty: return fig_empty("Velocidades — sin datos")
    if spd.shape[1] > 2:
        dm = spd.melt(id_vars='Driver', var_name='Metric', value_name='km/h')
        f = px.bar(dm, x='Driver', y='km/h', color='Metric', barmode='group', title='Velocidades')
    else:
        ycol = spd.columns[-1]
        f = px.bar(spd, x='Driver', y=ycol, title='Velocidades')
    f.update_traces(marker_line_width=0)
    f.update_layout(yaxis_title="km/h")
    return polish(f)

# ---------- Telemetry ----------
@app.callback(
    Output('tel-speed', 'figure'),
    Input('store', 'data'),
    Input({'role': 'drv', 'chart': 'tel_spd'}, 'value'),
    Input('tabs', 'value'),
    State('team-color-store', 'data'),
)
def chart_tel_speed(data, selected, tab, color_map):
    if tab != 'tele' or not data:
        raise PreventUpdate
    if not selected:
        return fig_empty("Elige 1–3 pilotos para comparar")
    selected = selected[:3]
    try:
        ses = load_session_telemetry(int(data.get('year', 2025)), data['event'], data['sess'])
    except Exception:
        traceback.print_exc()
        return fig_empty("No se pudo cargar la telemetría")
    f = go.Figure()
    drawn = False
    for drv in selected:
        tel = fastest_lap_telemetry(ses, drv)
        if tel is None or 'Speed' not in tel.columns or 'Distance' not in tel.columns:
            continue
        c = (color_map or {}).get(drv)
        f.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Speed'], mode='lines', name=str(drv),
            line=dict(color=c, width=2) if c else dict(width=2),
            hovertemplate=f"{drv} — %{{x:.0f}} m<br>%{{y:.0f}} km/h<extra></extra>"))
        drawn = True
    if not drawn:
        return fig_empty("Sin telemetría disponible para esos pilotos")
    corners = circuit_corners(ses)
    if corners is not None and 'Distance' in getattr(corners, 'columns', []):
        for _, cc in corners.iterrows():
            try:
                xd = float(cc['Distance'])
            except Exception:
                continue
            f.add_vline(x=xd, line=dict(color='rgba(255,255,255,0.10)', width=1),
                        annotation_text=f"T{int(cc['Number'])}", annotation_position="top",
                        annotation_font=dict(size=9, color=COL_MUTED))
    f.update_layout(title="Velocidad vs distancia — vuelta rápida")
    f.update_xaxes(title="Distancia (m)")
    f.update_yaxes(title="km/h")
    return polish(f)

@app.callback(
    Output('tel-map', 'figure'),
    Input('store', 'data'),
    Input({'role': 'drv', 'chart': 'tel_map'}, 'value'),
    Input('tabs', 'value'),
    State('team-color-store', 'data'),
)
def chart_tel_map(data, selected, tab, color_map):
    if tab != 'tele' or not data:
        raise PreventUpdate
    sel = (selected or [])[:2]
    if not sel:
        return fig_empty("Elige pilotos (2 = mapa de dominancia)")
    try:
        ses = load_session_telemetry(int(data.get('year', 2025)), data['event'], data['sess'])
    except Exception:
        traceback.print_exc()
        return fig_empty("No se pudo cargar la telemetría")
    color_map = color_map or {}
    try:
        if len(sel) >= 2:
            a, b = sel[0], sel[1]
            ta = fastest_lap_telemetry(ses, a)
            tb = fastest_lap_telemetry(ses, b)
            if ta is None or tb is None or not {'X', 'Y', 'Speed', 'Distance'}.issubset(ta.columns):
                return fig_empty("Sin datos de posición")
            sb = np.interp(ta['Distance'].values, tb['Distance'].values, tb['Speed'].values)
            a_faster = ta['Speed'].values >= sb
            ca = color_map.get(a) or '#e10600'
            cb = color_map.get(b) or '#27F4D2'
            cols = np.where(a_faster, ca, cb)
            f = go.Figure(go.Scatter(x=ta['X'], y=ta['Y'], mode='markers',
                          marker=dict(size=5, color=cols), showlegend=False, hoverinfo='skip'))
            f.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                   marker=dict(color=ca, size=9), name=f"{a} +rápido"))
            f.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                   marker=dict(color=cb, size=9), name=f"{b} +rápido"))
            f.update_layout(title=f"Dominancia — {a} vs {b}")
        else:
            drv = sel[0]
            tel = fastest_lap_telemetry(ses, drv)
            if tel is None or not {'X', 'Y', 'Speed'}.issubset(set(tel.columns)):
                return fig_empty("Sin datos de posición")
            f = go.Figure(go.Scatter(
                x=tel['X'], y=tel['Y'], mode='markers',
                marker=dict(size=6, color=tel['Speed'], colorscale='Turbo', showscale=True,
                            colorbar=dict(title="km/h", thickness=12, outlinewidth=0)),
                hovertemplate="%{marker.color:.0f} km/h<extra></extra>", name=str(drv)))
            f.update_layout(title=f"Mapa de pista — {drv}")
    except Exception:
        traceback.print_exc()
        return fig_empty("No se pudo dibujar el mapa")
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return polish(f)

@app.callback(
    Output('tel-delta', 'figure'),
    Input('store', 'data'),
    Input({'role': 'drv', 'chart': 'tel_delta'}, 'value'),
    Input('tabs', 'value'),
    State('team-color-store', 'data'),
)
def chart_tel_delta(data, selected, tab, color_map):
    if tab != 'tele' or not data:
        raise PreventUpdate
    sel = (selected or [])[:2]
    if len(sel) < 2:
        return fig_empty("Elige 2 pilotos para el delta")
    a, b = sel[0], sel[1]
    try:
        from fastf1.utils import delta_time
        ses = load_session_telemetry(int(data.get('year', 2025)), data['event'], data['sess'])
        lap_a = ses.laps.pick_drivers(a).pick_fastest()
        lap_b = ses.laps.pick_drivers(b).pick_fastest()
        delta, ref_tel, _ = delta_time(lap_a, lap_b)
    except Exception:
        traceback.print_exc()
        return fig_empty("No se pudo calcular el delta")
    try:
        x = ref_tel['Distance']
    except Exception:
        x = list(range(len(delta)))
    cb = (color_map or {}).get(b) or '#27F4D2'
    f = go.Figure(go.Scatter(x=x, y=delta, mode='lines',
                  line=dict(color=cb, width=2), fill='tozeroy',
                  fillcolor='rgba(255,255,255,0.05)',
                  hovertemplate="%{x:.0f} m<br>Δ %{y:+.3f}s<extra></extra>", name=f"{b} vs {a}"))
    f.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1))
    f.update_layout(title=f"Delta de tiempo — {b} respecto a {a}  (↑ {b} más lento)")
    f.update_xaxes(title="Distancia (m)")
    f.update_yaxes(title="Δ s")
    return polish(f)

# ---------- Results & championship standings ----------
@app.callback(
    Output("results-table", "children"),
    Output("standings-drivers", "children"),
    Output("standings-teams", "children"),
    Input('store', 'data'),
    Input('tabs', 'value'),
)
def render_results(data, tab):
    if tab != 'results':
        raise PreventUpdate
    if not data:
        msg = html.Div("Selecciona un evento arriba.", className="rlo-note")
        return msg, msg, msg
    year = int(data.get('year', 2025))
    try:
        ses = load_session_laps(year, data['event'], data['sess'])
        results_tbl = build_results_table(ses, data['sess'])
    except Exception:
        traceback.print_exc()
        results_tbl = html.Div("No se pudo cargar el resultado de esta sesión.", className="rlo-note")
    try:
        drv_rows, team_rows = season_standings(year, _utc_today_token())
        drivers_tbl = build_standings_drivers(drv_rows)
        teams_tbl = build_standings_teams(team_rows)
    except Exception:
        traceback.print_exc()
        drivers_tbl = teams_tbl = html.Div("Clasificación no disponible ahora mismo.", className="rlo-note")
    return results_tbl, drivers_tbl, teams_tbl

# ---------- Home & Live content ----------
@app.callback(
    Output('home-next', 'children'),
    Output('home-target', 'data'),
    Output('home-podium', 'children'),
    Output('home-drv', 'children'),
    Output('home-team', 'children'),
    Input('year-dd', 'value'),
    Input('tabs', 'value'),
)
def render_home(year, tab):
    if tab != 'inicio':
        raise PreventUpdate
    token = _utc_today_token()
    try:
        year = int(year)
    except Exception:
        year = default_year_value()

    target_ms = None
    nx = next_session_info(year, token)
    if nx:
        target_ms = int(pd.Timestamp(nx['utc']).tz_localize('UTC').timestamp() * 1000)
        loc = nx['location'] or nx['country']
        next_div = html.Div([
            html.Div(nx['event'], className="rlo-next-event"),
            html.Div(nx['session'] + (f" · {loc}" if loc else ""), className="rlo-next-sub"),
            html.Div("—:—:—", id="home-cd", className="rlo-cd"),
            html.Div(nx['utc'].strftime('%d/%m/%Y · %H:%M UTC'), className="rlo-next-when"),
        ])
    else:
        next_div = html.Div("Sin sesiones próximas en el calendario.", className="rlo-note")

    pod = last_race_podium(year, token)
    if pod:
        name, podium = pod
        medals = ['🥇', '🥈', '🥉']
        podium_div = html.Div(
            [html.Div(name, className="rlo-next-sub", style={"marginBottom": "8px"})] +
            [_leader_row(medals[i] if i < 3 else str(i + 1), full or ab, ab, '', _team_color(team))
             for i, (pos, ab, full, team) in enumerate(podium)])
    else:
        podium_div = html.Div("Sin resultados de carrera todavía.", className="rlo-note")

    try:
        drv_rows, team_rows = season_standings(year, token)
    except Exception:
        drv_rows, team_rows = [], []
    drv_div = (html.Div([_leader_row(i + 1, name, ab, _fmt_pts(pts), _team_color(team))
                         for i, (ab, name, team, pts) in enumerate(drv_rows[:5])])
               if drv_rows else html.Div("Aún sin puntos esta temporada.", className="rlo-note"))
    team_div = (html.Div([_leader_row(i + 1, tm, '', _fmt_pts(pts), _team_color(tm))
                          for i, (tm, pts) in enumerate(team_rows[:5])])
                if team_rows else html.Div("Aún sin puntos esta temporada.", className="rlo-note"))

    return next_div, target_ms, podium_div, drv_div, team_div

app.clientside_callback(
    """
    function(n, target) {
        if (!target) { return '—:—:—'; }
        var diff = Math.floor((target - Date.now())/1000);
        if (diff <= 0) { return '¡EN PISTA!'; }
        var d = Math.floor(diff/86400); diff -= d*86400;
        var h = Math.floor(diff/3600); diff -= h*3600;
        var m = Math.floor(diff/60); var s = diff - m*60;
        function p(x){ return ('0'+x).slice(-2); }
        return (d>0 ? d+'d ' : '') + p(h)+':'+p(m)+':'+p(s);
    }
    """,
    Output('home-cd', 'children'),
    Input('home-cd-int', 'n_intervals'),
    State('home-target', 'data'),
)

@app.callback(
    Output('live-status', 'children'),
    Input('year-dd', 'value'),
    Input('tabs', 'value'),
)
def render_live(year, tab):
    if tab != 'live':
        raise PreventUpdate
    token = _utc_today_token()
    try:
        year = int(year)
    except Exception:
        year = default_year_value()
    live = live_session_info(year, token)
    if live:
        return html.Div(className="rlo-live-banner live", children=[
            html.Span("● EN VIVO", className="rlo-live-dot"),
            html.Span(f"{live['session']} — {live['event']}"),
        ])
    nx = next_session_info(year, token)
    if nx:
        return html.Div(className="rlo-live-banner", children=[
            html.Span("○ Sin sesión ahora", className="rlo-live-dot off"),
            html.Span(f"Próxima: {nx['session']} — {nx['event']} · {nx['utc'].strftime('%d/%m %H:%M UTC')}"),
        ])
    return html.Div("Sin información de sesiones.", className="rlo-note")

# ================= CSV download (pattern-matching) =================
@app.callback(
    Output({"role": "csv-dl", "chart": MATCH}, "data"),
    Input({"role": "csv", "chart": MATCH}, "n_clicks"),
    State({"role": "csv", "chart": MATCH}, "id"),
    State("store", "data"),
    State({"role": "drv", "chart": MATCH}, "value"),
    prevent_initial_call=True,
)
def download_chart_csv(n_clicks, btn_id, store_data, selected_drivers):
    if not n_clicks or not store_data:
        return no_update

    chart_key = btn_id.get("chart")
    try:
        # Carga sesión (laps-only) para exportar datos rápidos
        ses = load_session_laps(
            int(store_data.get("year", 2025)),
            str(store_data["event"]),
            str(store_data["sess"]),
        )

        df = export_df_for_chart(chart_key, ses, selected_drivers)

        if df is None:
            return no_update

        # Si está vacío, aún así devolvemos CSV con headers (mejor UX)
        safe_event = str(store_data["event"]).replace("|", "_").replace(" ", "_")
        safe_sess = str(store_data["sess"]).replace(" ", "_")
        fname = f"{chart_key}_{store_data.get('year', 0)}_{safe_event}_{safe_sess}.csv"

        return dcc.send_data_frame(df.to_csv, fname, index=False)

    except Exception:
        traceback.print_exc()
        return no_update

# ================= Run =================
server = app.server
from flask import jsonify

@server.route("/health", methods=["GET"])
def health():
    n = 0
    try:
        for _root, _dirs, files in os.walk(DATA_DIR):
            n += sum(1 for f in files if f.endswith(".laps.parquet"))
    except Exception:
        n = -1
    return jsonify(status="ok", precomputed_sessions=n)

@server.route("/warmup", methods=["GET"])
def warmup():
    # Warmup schedule + try loading last past session/test for allowed years
    try:
        now = pd.Timestamp.utcnow().tz_localize(None)
        for y in YEARS_ALLOWED:
            try:
                df = get_schedule_df(y, _utc_today_token())
                past = df[df["EventDate"] <= now].sort_values("EventDate")
                if past.empty:
                    continue
                last = past.iloc[-1]

                if str(last["EventFormat"]).lower() == "testing":
                    # pick test_number by date order
                    testing_df = df[df["EventFormat"] == "testing"].sort_values("EventDate")
                    test_dates = testing_df["EventDate"].dt.date.astype(str).tolist()
                    test_number = test_dates.index(str(last["EventDate"].date())) + 1
                    try:
                        s = ff1.get_testing_session(y, test_number, 1)
                        s.load(telemetry=False, weather=False, messages=False)
                    except Exception:
                        pass
                else:
                    gp = str(last["EventName"])
                    for sess_name in ("R", "Q"):
                        try:
                            s = ff1.get_session(y, gp, sess_name)
                            s.load(telemetry=False, weather=False, messages=False)
                            break
                        except Exception:
                            continue
            except Exception:
                continue
        return jsonify(status="warmed")
    except Exception as e:
        return jsonify(status="error", detail=str(e)), 500

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
