import os, warnings, logging, traceback
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

from dash import Dash, dcc, html, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

YEARS_ALLOWED = [2025, 2026]

def _utc_today_token() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

# Brand
SITE_TITLE = "Telemetría by RedLightsOff"
WATERMARK  = "@redlightsoff5"

IG_URL  = os.getenv("IG_URL", "https://instagram.com/redlightsoff5")
BMC_URL = os.getenv("BMC_URL", "https://buymeacoffee.com/redlightsoff5")

COL_BG    = "#9f9797"
COL_PANEL = "#ffffff"
COL_RED   = "#e11d2e"
COL_TEXT  = "#000000"

TEAM_COLORS = {
    "Red Bull":      "#3671C6",
    "McLaren":       "#FF8000",
    "Ferrari":       "#E80020",
    "Mercedes":      "#27F4D2",
    "Aston Martin":  "#229971",
    "Alpine":        "#0093CC",
    "Williams":      "#64C4FF",
    "Racing Bulls":  "#6692FF",
    "Sauber":        "#52E252",
    "Haas":          "#B6BABD",
    # Custom
    "Cadillac":      "#6B7280",  # gris
    "Audi":          "#1F2937",  # grafito oscuro
}

TEAM_ALIASES = {
    "red bull": "Red Bull",
    "oracle red bull": "Red Bull",
    "red bull racing": "Red Bull",

    "rb f1": "Racing Bulls",
    "racing bulls": "Racing Bulls",
    "visa cash app rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "rb": "Racing Bulls",

    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",

    "mercedes": "Mercedes",
    "mercedes-amg": "Mercedes",

    "mclaren": "McLaren",
    "aston martin": "Aston Martin",

    "alpine": "Alpine",
    "bwt alpine": "Alpine",

    "williams": "Williams",
    "williams racing": "Williams",

    # Sauber/Kick/Stake/Audi buckets
    "sauber": "Sauber",
    "kick sauber": "Sauber",
    "stake": "Sauber",
    "stake f1": "Sauber",
    "audi": "Audi",

    "haas": "Haas",
    "cadillac": "Cadillac",
}

def canonical_team(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    for key, canon in TEAM_ALIASES.items():
        if key in s:
            return canon
    return name

def s_to_mssmmm(x):
    if pd.isna(x): return ""
    x = float(x)
    m = int(x // 60); s = int(x % 60); ms = int(round((x - int(x))*1000))
    return f"{m}:{s:02d}.{ms:03d}"

def is_race(ses):
    t = (getattr(ses, "session_type", "") or "").upper()
    n = (getattr(ses, "name", "") or "").upper()
    return t == "R" or "RACE" in n

COMMON_LAYOUT = dict(
    paper_bgcolor=COL_PANEL,
    plot_bgcolor=COL_PANEL,
    font=dict(color=COL_TEXT),
    margin=dict(l=12, r=12, t=56, b=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

def brand(fig: go.Figure):
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=42, color="rgba(255,255,255,0.18)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.25
    )
    return fig

def fig_empty(title: str, subtitle: str = "Sin datos"):
    f = go.Figure()
    f.update_layout(title=f"{title}<br><sup>{subtitle}</sup>")
    return brand(f)

def set_trace_color(fig, name_to_color):
    for tr in fig.data:
        c = (name_to_color or {}).get(tr.name)
        if c:
            if hasattr(tr, "line") and tr.line is not None:
                tr.update(line=dict(color=c))
            if hasattr(tr, "marker") and tr.marker is not None:
                tr.update(marker=dict(color=c))
    return fig

# ---------- Schedule ----------
@lru_cache(maxsize=8)
def get_schedule_df(year: int, date_token: str) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=True).copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    if "EventFormat" not in df.columns:
        df["EventFormat"] = ""
    df["EventFormat"] = df["EventFormat"].astype(str).str.lower()
    keep = ["RoundNumber","EventName","EventDate","EventFormat"]
    df = df[keep].sort_values(["EventDate","RoundNumber"]).reset_index(drop=True)
    return df

def build_event_options(year: int):
    df = get_schedule_df(year, _utc_today_token())
    testing_df = df[df["EventFormat"].str.contains("test", na=False)].sort_values("EventDate").reset_index(drop=True)
    test_keys = {(r.EventDate.date(), str(r.EventName)): i+1 for i, r in testing_df.iterrows()}

    opts=[]
    for _, r in df.iterrows():
        fmt = str(r.EventFormat).lower()
        date = r.EventDate.date()
        name = str(r.EventName)
        if "test" in fmt:
            n = test_keys.get((date, name), 1)
            opts.append({"label": f"Test de Pretemporada #{n} ({date})", "value": f"TEST|{n}"})
        else:
            opts.append({"label": f"R{int(r.RoundNumber)} — {name} ({date})",
                         "value": f"GP|round={int(r.RoundNumber)}|name={name}"})
    return opts

def default_event_value(year: int):
    df = get_schedule_df(year, _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df["EventDate"] <= today].sort_values("EventDate")
    if past.empty:
        return None
    last = past.iloc[-1]
    if "test" in str(last["EventFormat"]).lower():
        testing_df = df[df["EventFormat"].str.contains("test", na=False)].sort_values("EventDate").reset_index(drop=True)
        test_keys = {(r.EventDate.date(), str(r.EventName)): i+1 for i, r in testing_df.iterrows()}
        n = test_keys.get((last["EventDate"].date(), str(last["EventName"])), 1)
        return f"TEST|{n}"
    return f"GP|round={int(last['RoundNumber'])}|name={str(last['EventName'])}"

def default_year_value() -> int:
    today = pd.Timestamp.utcnow().tz_localize(None)
    for y in sorted(YEARS_ALLOWED, reverse=True):
        try:
            df = get_schedule_df(y, _utc_today_token())
            if not df.empty and (df["EventDate"] <= today).any():
                return y
        except Exception:
            continue
    return YEARS_ALLOWED[0]

def parse_event_value(val: str):
    if not val:
        return None
    s = str(val)
    if s.startswith("TEST|"):
        try: return {"type":"test","test": int(s.split("|",1)[1])}
        except Exception: return {"type":"test","test": 1}
    if s.startswith("GP|"):
        out={"type":"gp"}
        for part in s.split("|")[1:]:
            if part.startswith("round="):
                try: out["round"]=int(part.split("=",1)[1])
                except Exception: pass
            if part.startswith("name="):
                out["name"]=part.split("=",1)[1]
        return out
    return {"type":"gp","name": s}

SESSION_OPTIONS = [
    {"label": "FP1", "value": "FP1"},
    {"label": "FP2", "value": "FP2"},
    {"label": "FP3", "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},
    {"label": "Clasificación", "value": "Q"},
    {"label": "Sprint", "value": "SR"},
    {"label": "Carrera", "value": "R"},
]
TEST_SESSION_OPTIONS = [
    {"label": "Día 1", "value": 1},
    {"label": "Día 2", "value": 2},
    {"label": "Día 3", "value": 3},
]

# ---------- Loaders ----------
@lru_cache(maxsize=64)
def load_session(year: int, event_value: str, sess_code):
    info = parse_event_value(event_value)
    if not info:
        raise ValueError("Evento no seleccionado")

    if info.get("type") == "test":
        test_number = int(info.get("test", 1))
        day = int(sess_code) if sess_code is not None else 1
        ses = ff1.get_testing_session(int(year), test_number, day)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        return ses

    # GP
    event_name = info.get("name") or str(event_value)
    code = str(sess_code)
    try:
        ses = ff1.get_session(int(year), event_name, code)
    except Exception:
        if str(code).upper() == "SQ":
            ses = ff1.get_session(int(year), event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

def driver_team_color_map(ses):
    laps = ses.laps[["Driver","Team"]].dropna()
    if laps.empty: return {}
    team = laps.groupby("Driver")["Team"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
    team = team.apply(canonical_team)
    return {drv: TEAM_COLORS.get(t, "#cccccc") for drv, t in team.items()}

# ---------- Cached derived dataframes ----------
@lru_cache(maxsize=64)
def get_laps_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    laps = ses.laps.copy()
    laps = laps.dropna(subset=["LapTime"])
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds().astype(float)
    laps["LapStr"] = laps["LapSeconds"].apply(s_to_mssmmm)
    if "TyreLife" in laps.columns:
        # keep
        pass
    return laps

@lru_cache(maxsize=64)
def get_gap_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty:
        return pd.DataFrame()
    if is_race(ses):
        laps2 = laps[["Driver","LapNumber","LapSeconds"]].copy()
        laps2["Cum"] = laps2.groupby("Driver")["LapSeconds"].cumsum()
        lead = laps2.groupby("LapNumber")["Cum"].min().rename("Lead").reset_index()
        d = laps2.merge(lead, on="LapNumber", how="left")
        d["Gap_s"] = d["Cum"] - d["Lead"]
        d["GapStr"] = d["Gap_s"].apply(s_to_mssmmm)
        return d
    # non-race: best lap gap
    best = laps.groupby("Driver")["LapSeconds"].min().rename("Best_s").reset_index()
    gbest = best["Best_s"].min()
    best["Gap_s"] = best["Best_s"] - gbest
    best["GapStr"] = best["Gap_s"].apply(s_to_mssmmm)
    best["LapNumber"] = 1
    best["Cum"] = best["Best_s"]
    best["Lead"] = gbest
    return best

@lru_cache(maxsize=64)
def get_interval_ahead_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    if not is_race(ses):
        return pd.DataFrame()
    g = get_gap_df(year, event_value, sess_code)
    if g.empty:
        return pd.DataFrame()
    # For each lap, sort by cumulative time, compute interval to car ahead.
    out=[]
    for lap, grp in g.groupby("LapNumber"):
        gg = grp.sort_values("Cum")
        gg["AheadCum"] = gg["Cum"].shift(1)
        gg["IntervalAhead_s"] = gg["Cum"] - gg["AheadCum"]
        gg.loc[gg["AheadCum"].isna(), "IntervalAhead_s"] = np.nan
        gg["IntervalStr"] = gg["IntervalAhead_s"].apply(s_to_mssmmm)
        out.append(gg[["Driver","LapNumber","IntervalAhead_s","IntervalStr"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

@lru_cache(maxsize=64)
def get_best_lap_df(year:int, event_value:str, sess_code):
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty:
        return pd.DataFrame()
    best = laps.loc[laps.groupby("Driver")["LapSeconds"].idxmin()].copy()
    best["BestStr"] = best["LapSeconds"].apply(s_to_mssmmm)
    return best[["Driver","LapSeconds","BestStr","LapNumber","Team"]].sort_values("LapSeconds")

@lru_cache(maxsize=64)
def get_positions_gained_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    res = ses.results
    if res is None or res.empty:
        return pd.DataFrame()
    need = {"Abbreviation","GridPosition","Position"}
    if not need.issubset(res.columns):
        return pd.DataFrame()
    df = res[["Abbreviation","GridPosition","Position"]].copy()
    df["Driver"] = df["Abbreviation"]
    df["PositionsGained"] = df["GridPosition"] - df["Position"]
    return df.sort_values("PositionsGained", ascending=False)

@lru_cache(maxsize=64)
def get_lapchart_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    laps = ses.laps.copy()
    if laps.empty or "Position" not in laps.columns:
        return pd.DataFrame()
    df = laps[["Driver","LapNumber","Position"]].dropna()
    return df

@lru_cache(maxsize=64)
def get_delta_to_best_lap_df(year:int, event_value:str, sess_code):
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty:
        return pd.DataFrame()
    best_overall = laps["LapSeconds"].min()
    d = laps[["Driver","LapNumber","LapSeconds","LapStr"]].copy()
    d["DeltaBest_s"] = d["LapSeconds"] - best_overall
    d["DeltaBestStr"] = d["DeltaBest_s"].apply(s_to_mssmmm)
    return d

@lru_cache(maxsize=64)
def get_stints_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    laps = ses.laps.copy()
    if laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    laps["Compound"] = laps["Compound"].astype(str).str.upper()
    if "Stint" not in laps.columns:
        return pd.DataFrame()
    agg = (laps.groupby(["Driver","Stint","Compound"], dropna=False)
               .agg(LapStart=("LapNumber","min"), LapEnd=("LapNumber","max")))
    agg["Laps"] = agg["LapEnd"] - agg["LapStart"] + 1
    return agg.reset_index().sort_values(["Driver","Stint"])

@lru_cache(maxsize=64)
def get_compound_usage_df(year:int, event_value:str, sess_code):
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    df = laps.dropna(subset=["Compound"]).copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    counts = df["Compound"].value_counts().rename_axis("Compound").reset_index(name="Laps")
    return counts

@lru_cache(maxsize=64)
def get_pit_laps_df(year:int, event_value:str, sess_code):
    ses = load_session(year, event_value, sess_code)
    laps = ses.laps.copy()
    if laps.empty:
        return pd.DataFrame()
    # Prefer PitInTime if exists
    if "PitInTime" in laps.columns:
        p = laps.dropna(subset=["PitInTime"])[["Driver","LapNumber"]].copy()
        p["Type"] = "PitIn"
        return p
    # Fallback: stint change (approx)
    if "Stint" in laps.columns:
        d = laps[["Driver","LapNumber","Stint"]].dropna().sort_values(["Driver","LapNumber"]).copy()
        d["PrevStint"] = d.groupby("Driver")["Stint"].shift(1)
        p = d[(d["PrevStint"].notna()) & (d["Stint"] != d["PrevStint"])][["Driver","LapNumber"]].copy()
        p["Type"] = "StintChange"
        return p
    return pd.DataFrame()

@lru_cache(maxsize=64)
def get_pace_by_compound_df(year:int, event_value:str, sess_code):
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    df = laps.dropna(subset=["Compound"]).copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    return df[["Driver","Compound","LapSeconds","LapStr"]]

@lru_cache(maxsize=64)
def get_degradation_df(year:int, event_value:str, sess_code):
    laps = get_laps_df(year, event_value, sess_code)
    if laps.empty:
        return pd.DataFrame()
    if "TyreLife" in laps.columns and laps["TyreLife"].notna().any():
        df = laps.dropna(subset=["TyreLife"]).copy()
        df["TyreAge"] = df["TyreLife"].astype(float)
    else:
        # fallback: lap index within stint
        if "Stint" not in laps.columns:
            return pd.DataFrame()
        df = laps.dropna(subset=["Stint"]).copy()
        df = df.sort_values(["Driver","LapNumber"])
        df["TyreAge"] = df.groupby(["Driver","Stint"]).cumcount().astype(float)
    return df[["Driver","LapNumber","LapSeconds","LapStr","TyreAge","Compound"]].copy()

@lru_cache(maxsize=32)
def get_q_results_df(year:int, event_value:str):
    # Qualifying always uses Q session (GP only; testing not supported here)
    info = parse_event_value(event_value)
    if not info or info.get("type") != "gp":
        return pd.DataFrame()
    event_name = info.get("name") or str(event_value)
    try:
        ses = ff1.get_session(int(year), event_name, "Q")
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception:
        return pd.DataFrame()
    res = ses.results
    if res is None or res.empty:
        return pd.DataFrame()
    cols = []
    for c in ["Abbreviation","Position","Q1","Q2","Q3","GridPosition"]:
        if c in res.columns: cols.append(c)
    df = res[cols].copy()
    df.rename(columns={"Abbreviation":"Driver"}, inplace=True)
    # seconds columns
    for q in ["Q1","Q2","Q3"]:
        if q in df.columns:
            df[q+"_s"] = pd.to_timedelta(df[q]).dt.total_seconds()
            df[q+"_str"] = df[q+"_s"].apply(s_to_mssmmm)
    return df

# ================= Dash =================
external_stylesheets=[dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>{SITE_TITLE}</title>
  {{%favicon%}}
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
    return dbc.Row([
        dbc.Col([
            dbc.Label("Año"),
            dcc.Dropdown(
                id="year-dd",
                options=[{"label": str(y), "value": y} for y in YEARS_ALLOWED],
                value=default_year_value(),
                clearable=False
            ),
            html.Div(id="year-warning", className="mt-1", style={"fontSize":"0.85rem","opacity":0.85})
        ], md=3),
        dbc.Col([
            dbc.Label("Gran Premio / Test"),
            dcc.Dropdown(
                id="event-dd",
                options=build_event_options(default_year_value()),
                value=default_event_value(default_year_value()),
                placeholder="Selecciona evento...",
                clearable=False
            )
        ], md=6),
        dbc.Col([
            dbc.Label("Sesión"),
            dcc.Dropdown(
                id="session-dd",
                options=SESSION_OPTIONS,
                value="R",
                clearable=False
            )
        ], md=3),
    ], className="g-2")

def graph_box(graph_id: str, title: str, chart_key: str, height: int = 420):
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H5(title, className="m-0"), md=6, style={"display":"flex","alignItems":"center"}),
            dbc.Col([
                dcc.Dropdown(
                    id={"role":"drv", "chart": chart_key},
                    multi=True,
                    placeholder="Selecciona pilotos (opcional)",
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
                style={"height": f"{height}px"}
            ),
            type="default"
        )
    ])

def tab_rendimiento():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("pace","Ritmo: tiempo por vuelta", "pace"), md=6),
            dbc.Col(graph_box("rolling","Ritmo: media móvil (3v / 5v)", "roll"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("gap","Diferencia al líder (por vuelta)", "gap"), md=6),
            dbc.Col(graph_box("delta_best","Delta vs mejor vuelta de la sesión", "deltab"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("consistency","Consistencia: scatter (tiempo vs vuelta)", "cons"), md=6),
            dbc.Col(graph_box("best_rank","Ranking mejor vuelta", "best"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("lapchart","Lapchart: posición vs vuelta", "lc"), md=6),
            dbc.Col(graph_box("pos_gain","Posiciones ganadas/perdidas", "pg"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("cum_vs_leader","Tiempo acumulado vs líder", "cum"), md=6),
            dbc.Col(graph_box("interval_ahead","Intervalo al coche delante", "ia"), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_clasificacion():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("q_best_parts","Mejores tiempos Q1 / Q2 / Q3", "qparts"), md=6),
            dbc.Col(graph_box("q_progress","Mejora por sesión (Q1→Q2→Q3)", "qimp"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("knockout","Knockout: fase alcanzada", "ko"), md=6),
            dbc.Col(graph_box("grid_bestlap","Parrilla vs mejor vuelta (carrera)", "gbl"), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_neumaticos():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("tyre_timeline","Estrategia de neumáticos (stints)", "ty"), md=12, height=520),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("compound_pie","Uso de compuestos (pie)", "cp"), md=6),
            dbc.Col(graph_box("stint_len","Distribución longitud de stint", "sl"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("pit_hist","Histograma vueltas de pit-stop", "ph"), md=6),
            dbc.Col(graph_box("pace_compound","Ritmo por compuesto", "pc"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("degradation","Degradación: tiempo vs edad de neumático", "deg"), md=6),
            dbc.Col(graph_box("pace_drop","Caída de ritmo vs vida neumático", "pd"), md=6),
        ], className="g-2 mt-1"),
    ])

app.layout = dbc.Container([
    header_controls(),
    dcc.Tabs(
        id="tabs",
        value="rend",
        parent_className="rlo-tabs-parent",
        className="rlo-tabs",
        children=[
            dcc.Tab(label="Rendimiento", value="rend", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Clasificación", value="qual", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Neumáticos", value="tyres", className="rlo-tab", selected_className="rlo-tab--selected"),
        ],
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_rendimiento()),
    dcc.Store(id="store"),
    dcc.Store(id="drivers-store"),
    dcc.Store(id="team-color-store"),
], fluid=True, className="rlo-page")

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"rend": tab_rendimiento, "qual": tab_clasificacion, "tyres": tab_neumaticos}.get(val, tab_rendimiento)()

# Switch session dropdown options based on GP vs Test
@app.callback(
    Output("session-dd","options"),
    Output("session-dd","value"),
    Input("event-dd","value"),
    State("session-dd","value")
)
def _event_changed_session_options(event_val, current):
    info = parse_event_value(event_val)
    if not info:
        return SESSION_OPTIONS, "R"
    if info.get("type") == "test":
        val = current if current in (1,2,3) else 1
        return TEST_SESSION_OPTIONS, val
    val = current if current in [o["value"] for o in SESSION_OPTIONS] else "R"
    return SESSION_OPTIONS, val

# Year-driven event list
@app.callback(
    Output("event-dd","options"),
    Output("event-dd","value"),
    Output("year-warning","children"),
    Input("year-dd","value"),
    State("event-dd","value")
)
def _year_changed(year, current_event):
    if year is None:
        return [], None, ""
    year = int(year)
    try:
        opts = build_event_options(year)
        default_ev = default_event_value(year)
        valid = {o["value"] for o in (opts or [])}
        value = current_event if current_event in valid else default_ev
        warn = ""
        if default_ev is None:
            warn = f"No hay eventos completados aún en {year}. Selecciona 2025 para ver datos."
            value = None
        return opts, value, warn
    except Exception:
        return [], None, f"Calendario no disponible para {year}."

# Load meta once
@app.callback(
    Output("store","data"),
    Output("drivers-store","data"),
    Output("team-color-store","data"),
    Input("year-dd","value"),
    Input("event-dd","value"),
    Input("session-dd","value")
)
def load_session_meta(year, event_value, sess_code):
    if not year or not event_value or sess_code is None:
        return no_update, [], {}
    try:
        ses = load_session(int(year), str(event_value), sess_code)
        laps = ses.laps.dropna(subset=["LapTime"]) if hasattr(ses, "laps") else pd.DataFrame()
        drivers = sorted(laps["Driver"].dropna().unique().tolist()) if not laps.empty else []
        colors = driver_team_color_map(ses)
        return {"year": int(year), "event_value": str(event_value), "sess": sess_code}, drivers, colors
    except Exception:
        traceback.print_exc()
        return no_update, [], {}

# Populate dropdowns for visible graphs
@app.callback(
    Output({"role":"drv","chart":ALL}, "options"),
    Output({"role":"drv","chart":ALL}, "value"),
    Input("drivers-store","data"),
    Input("tab-body","children"),
    State({"role":"drv","chart":ALL}, "id")
)
def fill_dropdowns(drivers, _children, ids):
    opts = [{"label": d, "value": d} for d in (drivers or [])]
    # default: all drivers selected (fast path for “overview”)
    val = drivers or []
    n = len(ids)
    return [opts]*n, [val]*n

# ================= Charts: Rendimiento =================
@app.callback(Output("pace","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"pace"}, "value"),
              State("team-color-store","data"))
def chart_pace(data, selected, color_map):
    if not data: return fig_empty("Ritmo: tiempo por vuelta")
    laps = get_laps_df(data["year"], data["event_value"], data["sess"])
    if selected: laps = laps[laps["Driver"].isin(selected)]
    if laps.empty: return fig_empty("Ritmo: tiempo por vuelta")
    f = go.Figure()
    for drv, d in laps.groupby("Driver"):
        d = d.sort_values("LapNumber")
        f.add_trace(go.Scatter(x=d["LapNumber"], y=d["LapSeconds"], mode="lines+markers", name=str(drv),
                               customdata=np.stack([d["LapStr"]], axis=-1),
                               hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>"))
    f.update_layout(title="Ritmo: tiempo por vuelta")
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("rolling","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"roll"}, "value"),
              State("team-color-store","data"))
def chart_rolling(data, selected, color_map):
    if not data: return fig_empty("Ritmo: media móvil (3v / 5v)")
    laps = get_laps_df(data["year"], data["event_value"], data["sess"])
    if selected: laps = laps[laps["Driver"].isin(selected)]
    if laps.empty: return fig_empty("Ritmo: media móvil (3v / 5v)")
    df = laps.sort_values(["Driver","LapNumber"]).copy()
    df["MA3"] = df.groupby("Driver")["LapSeconds"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["MA5"] = df.groupby("Driver")["LapSeconds"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    f = go.Figure()
    for drv, d in df.groupby("Driver"):
        f.add_trace(go.Scatter(x=d["LapNumber"], y=d["MA3"], mode="lines", name=str(drv),
                               hovertemplate=f"{drv} — MA3 V%{{x}}<br>%{{y:.3f}}s<extra></extra>"))
        f.add_trace(go.Scatter(x=d["LapNumber"], y=d["MA5"], mode="lines", name=str(drv),
                               line=dict(dash="dot"),
                               showlegend=False,
                               hovertemplate=f"{drv} — MA5 V%{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_layout(title="Ritmo: media móvil (3v sólido / 5v punteado)")
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("best_rank","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"best"}, "value"),
              State("team-color-store","data"))
def chart_best_rank(data, selected, color_map):
    if not data: return fig_empty("Ranking mejor vuelta")
    best = get_best_lap_df(data["year"], data["event_value"], data["sess"])
    if selected: best = best[best["Driver"].isin(selected)]
    if best.empty: return fig_empty("Ranking mejor vuelta")
    f = px.bar(best, x="Driver", y="LapSeconds", custom_data=["BestStr"], title="Ranking mejor vuelta (MM:SS.mmm)")
    f.update_traces(hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>", marker_line_width=0)
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("gap","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"gap"}, "value"),
              State("team-color-store","data"))
def chart_gap(data, selected, color_map):
    if not data: return fig_empty("Diferencia al líder (por vuelta)")
    ses = load_session(data["year"], data["event_value"], data["sess"])
    df = get_gap_df(data["year"], data["event_value"], data["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty: return fig_empty("Diferencia al líder (por vuelta)")
    if is_race(ses):
        f = px.line(df, x="LapNumber", y="Gap_s", color="Driver", custom_data=["GapStr"],
                    title="Diferencia al líder (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>")
        f.update_yaxes(title="gap (s)", tickformat=".3f")
    else:
        dd = df.sort_values("Gap_s")
        f = px.bar(dd, x="Driver", y="Gap_s", custom_data=["GapStr"], title="Diferencia a la mejor vuelta (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>", marker_line_width=0)
        f.update_yaxes(title="gap (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("delta_best","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"deltab"}, "value"),
              State("team-color-store","data"))
def chart_delta_best(data, selected, color_map):
    if not data: return fig_empty("Delta vs mejor vuelta de la sesión")
    d = get_delta_to_best_lap_df(data["year"], data["event_value"], data["sess"])
    if selected: d = d[d["Driver"].isin(selected)]
    if d.empty: return fig_empty("Delta vs mejor vuelta de la sesión")
    f = px.line(d, x="LapNumber", y="DeltaBest_s", color="Driver", custom_data=["DeltaBestStr"],
                title="Delta vs mejor vuelta de la sesión (MM:SS.mmm)")
    f.update_traces(hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>")
    f.update_yaxes(title="delta (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("consistency","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"cons"}, "value"),
              State("team-color-store","data"))
def chart_consistency(data, selected, color_map):
    if not data: return fig_empty("Consistencia: scatter (tiempo vs vuelta)")
    laps = get_laps_df(data["year"], data["event_value"], data["sess"])
    if selected: laps = laps[laps["Driver"].isin(selected)]
    if laps.empty: return fig_empty("Consistencia: scatter (tiempo vs vuelta)")
    f = px.scatter(laps, x="LapNumber", y="LapSeconds", color="Driver", custom_data=["LapStr"],
                   title="Consistencia: tiempo vs vuelta")
    f.update_traces(hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>", mode="markers", marker=dict(size=6))
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("lapchart","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"lc"}, "value"),
              State("team-color-store","data"))
def chart_lapchart(data, selected, color_map):
    if not data: return fig_empty("Lapchart: posición vs vuelta")
    df = get_lapchart_df(data["year"], data["event_value"], data["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty: return fig_empty("Lapchart: posición vs vuelta", "Disponible solo si existe columna Position")
    f = px.line(df, x="LapNumber", y="Position", color="Driver", title="Lapchart: posición vs vuelta")
    f.update_yaxes(autorange="reversed", dtick=1, title="posición")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("pos_gain","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"pg"}, "value"))
def chart_positions_gained(data, selected):
    if not data: return fig_empty("Posiciones ganadas/perdidas")
    ses = load_session(data["year"], data["event_value"], data["sess"])
    if not is_race(ses):
        return fig_empty("Posiciones ganadas/perdidas", "Solo carrera")
    df = get_positions_gained_df(data["year"], data["event_value"], data["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty: return fig_empty("Posiciones ganadas/perdidas", "Sin resultados")
    f = px.bar(df, x="Driver", y="PositionsGained", title="Posiciones ganadas/perdidas", text="PositionsGained")
    f.update_traces(marker_line_width=0)
    f.update_yaxes(title="posiciones")
    return brand(f)

@app.callback(Output("cum_vs_leader","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"cum"}, "value"),
              State("team-color-store","data"))
def chart_cum_vs_leader(data, selected, color_map):
    if not data: return fig_empty("Tiempo acumulado vs líder")
    ses = load_session(data["year"], data["event_value"], data["sess"])
    if not is_race(ses):
        return fig_empty("Tiempo acumulado vs líder", "Solo carrera")
    d = get_gap_df(data["year"], data["event_value"], data["sess"])
    if selected: d = d[d["Driver"].isin(selected)]
    if d.empty: return fig_empty("Tiempo acumulado vs líder")
    f = px.line(d, x="LapNumber", y="Gap_s", color="Driver", custom_data=["GapStr"],
                title="Tiempo acumulado vs líder (mismo que gap acumulado)")
    f.update_traces(hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>")
    f.update_yaxes(title="diferencia acumulada (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("interval_ahead","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"ia"}, "value"),
              State("team-color-store","data"))
def chart_interval_ahead(data, selected, color_map):
    if not data: return fig_empty("Intervalo al coche delante")
    ses = load_session(data["year"], data["event_value"], data["sess"])
    if not is_race(ses):
        return fig_empty("Intervalo al coche delante", "Solo carrera")
    d = get_interval_ahead_df(data["year"], data["event_value"], data["sess"])
    if selected and not d.empty:
        d = d[d["Driver"].isin(selected)]
    if d.empty: return fig_empty("Intervalo al coche delante")
    f = px.line(d, x="LapNumber", y="IntervalAhead_s", color="Driver", custom_data=["IntervalStr"],
                title="Intervalo al coche delante (por vuelta)")
    f.update_traces(hovertemplate="%{fullData.name} — V%{x}<br>%{customdata[0]}<extra></extra>")
    f.update_yaxes(title="intervalo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

# ================= Charts: Clasificación =================
@app.callback(Output("q_best_parts","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"qparts"}, "value"))
def chart_q_best_parts(data, selected):
    if not data: return fig_empty("Mejores tiempos Q1 / Q2 / Q3")
    q = get_q_results_df(data["year"], data["event_value"])
    if selected and not q.empty:
        q = q[q["Driver"].isin(selected)]
    if q.empty:
        return fig_empty("Mejores tiempos Q1 / Q2 / Q3", "Solo GP con sesión Q disponible")
    # build bar grouped
    rows=[]
    for part in ["Q1","Q2","Q3"]:
        if part+"_s" in q.columns:
            tmp = q[["Driver", part+"_s", part+"_str"]].rename(columns={part+"_s":"Seconds", part+"_str":"Str"})
            tmp["Parte"] = part
            rows.append(tmp)
    if not rows:
        return fig_empty("Mejores tiempos Q1 / Q2 / Q3", "Sin columnas Q1/Q2/Q3")
    df = pd.concat(rows, ignore_index=True)
    f = px.bar(df, x="Driver", y="Seconds", color="Parte", barmode="group", custom_data=["Str"],
               title="Mejores tiempos Q1/Q2/Q3")
    f.update_traces(hovertemplate="%{x} — %{customdata[0]}<extra></extra>", marker_line_width=0)
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return brand(f)

@app.callback(Output("q_progress","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"qimp"}, "value"))
def chart_q_progress(data, selected):
    if not data: return fig_empty("Mejora por sesión (Q1→Q2→Q3)")
    q = get_q_results_df(data["year"], data["event_value"])
    if selected and not q.empty:
        q = q[q["Driver"].isin(selected)]
    if q.empty:
        return fig_empty("Mejora por sesión (Q1→Q2→Q3)")
    # improvement: Q1 - best_available (Q2/Q3)
    q = q.copy()
    if "Q1_s" in q.columns:
        base = q["Q1_s"]
    else:
        return fig_empty("Mejora por sesión (Q1→Q2→Q3)", "Sin Q1")
    best = base.copy()
    for col in ["Q2_s","Q3_s"]:
        if col in q.columns:
            best = np.fmin(best, q[col].fillna(np.inf))
    q["Mejora_s"] = base - best
    q["Mejora_str"] = q["Mejora_s"].apply(lambda x: s_to_mssmmm(x) if pd.notna(x) else "")
    f = px.bar(q.sort_values("Mejora_s", ascending=False), x="Driver", y="Mejora_s", custom_data=["Mejora_str"],
               title="Mejora por sesión (Q1 → mejor de Q2/Q3)")
    f.update_traces(hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>", marker_line_width=0)
    f.update_yaxes(title="mejora (s)", tickformat=".3f")
    return brand(f)

@app.callback(Output("knockout","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"ko"}, "value"))
def chart_knockout(data, selected):
    if not data: return fig_empty("Knockout: fase alcanzada")
    q = get_q_results_df(data["year"], data["event_value"])
    if selected and not q.empty:
        q = q[q["Driver"].isin(selected)]
    if q.empty:
        return fig_empty("Knockout: fase alcanzada")
    def stage(row):
        if "Q3_s" in row and pd.notna(row["Q3_s"]): return "Q3"
        if "Q2_s" in row and pd.notna(row["Q2_s"]): return "Q2"
        if "Q1_s" in row and pd.notna(row["Q1_s"]): return "Q1"
        return "NA"
    q = q.copy()
    q["Fase"] = q.apply(stage, axis=1)
    order = ["Q1","Q2","Q3"]
    q["Fase"] = pd.Categorical(q["Fase"], categories=order, ordered=True)
    f = px.strip(q, x="Fase", y="Driver", title="Knockout: fase alcanzada", orientation="h")
    f.update_traces(marker=dict(size=10), hovertemplate="%{y} — %{x}<extra></extra>")
    return brand(f)

@app.callback(Output("grid_bestlap","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"gbl"}, "value"))
def chart_grid_vs_bestlap(data, selected):
    if not data: return fig_empty("Parrilla vs mejor vuelta (carrera)")
    info = parse_event_value(data["event_value"])
    if not info or info.get("type") != "gp":
        return fig_empty("Parrilla vs mejor vuelta (carrera)", "Solo GP")
    year = int(data["year"])
    event_value = str(data["event_value"])
    # Prefer race session for grid positions; fallback to selected session if race not available
    try:
        sesR = load_session(year, event_value, "R")
    except Exception:
        sesR = None
    if sesR is None or sesR.results is None or sesR.results.empty:
        return fig_empty("Parrilla vs mejor vuelta (carrera)", "Sin resultados de carrera")
    res = sesR.results
    if "Abbreviation" not in res.columns or "GridPosition" not in res.columns:
        return fig_empty("Parrilla vs mejor vuelta (carrera)", "Columnas faltantes")
    grid = res[["Abbreviation","GridPosition"]].copy().rename(columns={"Abbreviation":"Driver"})
    # best laps from race laps
    try:
        best = get_best_lap_df(year, event_value, "R").rename(columns={"LapSeconds":"Best_s"})
    except Exception:
        return fig_empty("Parrilla vs mejor vuelta (carrera)", "No se pudo cargar laps")
    df = grid.merge(best[["Driver","Best_s","BestStr"]], on="Driver", how="inner")
    if selected: df = df[df["Driver"].isin(selected)]
    if df.empty: return fig_empty("Parrilla vs mejor vuelta (carrera)")
    f = px.scatter(df, x="GridPosition", y="Best_s", text="Driver", custom_data=["BestStr"],
                   title="Parrilla vs mejor vuelta (carrera)")
    f.update_traces(hovertemplate="%{text}<br>Grid %{x}<br>%{customdata[0]}<extra></extra>", textposition="top center")
    f.update_xaxes(title="posición salida (grid)", dtick=1)
    f.update_yaxes(title="mejor vuelta (s)", tickformat=".3f")
    return brand(f)

# ================= Charts: Neumáticos =================
@app.callback(Output("tyre_timeline","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"ty"}, "value"))
def chart_tyres_timeline(data, selected):
    if not data: return fig_empty("Estrategia de neumáticos (stints)")
    st = get_stints_df(data["year"], data["event_value"], data["sess"])
    if selected and not st.empty:
        st = st[st["Driver"].isin(selected)]
    if st.empty: return fig_empty("Estrategia de neumáticos (stints)")
    cmap = {"SOFT":"#DA291C","MEDIUM":"#FFD12E","HARD":"#F0F0F0","INTERMEDIATE":"#43B02A","WET":"#00A3E0"}
    f = go.Figure()
    order = st["Driver"].unique().tolist()[::-1]
    for _, r in st.iterrows():
        f.add_trace(go.Bar(
            x=[int(r["Laps"])], y=[r["Driver"]],
            base=[int(r["LapStart"]) - 1],
            orientation="h",
            marker_color=cmap.get(str(r["Compound"]).upper(), "#888"),
            showlegend=False,
            hovertemplate=f"{r['Driver']} — {r['Compound']}<br>V{int(r['LapStart'])}–V{int(r['LapEnd'])}<extra></extra>"
        ))
    for n,c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))
    f.update_layout(title="Estrategia de neumáticos (stints)", barmode="stack",
                    yaxis=dict(categoryorder="array", categoryarray=order, title="piloto"),
                    xaxis_title="vuelta")
    f.update_xaxes(showgrid=True, zeroline=False)
    f.update_yaxes(showgrid=False)
    return brand(f)

@app.callback(Output("compound_pie","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"cp"}, "value"))
def chart_compound_pie(data, selected):
    if not data: return fig_empty("Uso de compuestos (pie)")
    laps = get_laps_df(data["year"], data["event_value"], data["sess"])
    if "Compound" not in laps.columns:
        return fig_empty("Uso de compuestos (pie)", "Sin columna Compound")
    if selected:
        laps = laps[laps["Driver"].isin(selected)]
    if laps.empty:
        return fig_empty("Uso de compuestos (pie)")
    df = laps.dropna(subset=["Compound"]).copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    counts = df["Compound"].value_counts().rename_axis("Compound").reset_index(name="Laps")
    f = px.pie(counts, names="Compound", values="Laps", title="Uso de compuestos (vueltas)")
    return brand(f)

@app.callback(Output("stint_len","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"sl"}, "value"))
def chart_stint_length(data, selected):
    if not data: return fig_empty("Distribución longitud de stint")
    st = get_stints_df(data["year"], data["event_value"], data["sess"])
    if selected and not st.empty:
        st = st[st["Driver"].isin(selected)]
    if st.empty:
        return fig_empty("Distribución longitud de stint")
    f = px.histogram(st, x="Laps", nbins=20, title="Distribución longitud de stint (nº vueltas)")
    f.update_yaxes(title="conteo")
    return brand(f)

@app.callback(Output("pit_hist","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"ph"}, "value"))
def chart_pit_hist(data, selected):
    if not data: return fig_empty("Histograma vueltas de pit-stop")
    p = get_pit_laps_df(data["year"], data["event_value"], data["sess"])
    if selected and not p.empty:
        p = p[p["Driver"].isin(selected)]
    if p.empty:
        return fig_empty("Histograma vueltas de pit-stop", "Sin datos de pits (o no carrera)")
    f = px.histogram(p, x="LapNumber", color="Type", nbins=30, title="Histograma vueltas de pit-stop")
    f.update_xaxes(title="vuelta")
    f.update_yaxes(title="conteo")
    return brand(f)

@app.callback(Output("pace_compound","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"pc"}, "value"))
def chart_pace_by_compound(data, selected):
    if not data: return fig_empty("Ritmo por compuesto")
    df = get_pace_by_compound_df(data["year"], data["event_value"], data["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Ritmo por compuesto")
    f = px.box(df, x="Compound", y="LapSeconds", points="outliers", title="Ritmo por compuesto")
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return brand(f)

@app.callback(Output("degradation","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"deg"}, "value"),
              State("team-color-store","data"))
def chart_degradation(data, selected, color_map):
    if not data: return fig_empty("Degradación: tiempo vs edad de neumático")
    df = get_degradation_df(data["year"], data["event_value"], data["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Degradación: tiempo vs edad de neumático", "Sin TyreLife/Stint")
    f = px.scatter(df, x="TyreAge", y="LapSeconds", color="Driver", custom_data=["LapStr","Compound"],
                   title="Degradación: tiempo vs edad de neumático")
    f.update_traces(hovertemplate="%{fullData.name}<br>Edad %{x}<br>%{customdata[0]} (%{customdata[1]})<extra></extra>", marker=dict(size=6))
    f.update_xaxes(title="edad neumático (v)")
    f.update_yaxes(title="tiempo (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

@app.callback(Output("pace_drop","figure"),
              Input("store","data"),
              Input({"role":"drv","chart":"pd"}, "value"),
              State("team-color-store","data"))
def chart_pace_drop(data, selected, color_map):
    if not data: return fig_empty("Caída de ritmo vs vida neumático")
    df = get_degradation_df(data["year"], data["event_value"], data["sess"])
    if df.empty:
        return fig_empty("Caída de ritmo vs vida neumático", "Sin TyreLife/Stint")
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Caída de ritmo vs vida neumático")
    # pace drop relative to each driver's best lap in this session
    best = df.groupby("Driver")["LapSeconds"].transform("min")
    df2 = df.copy()
    df2["Drop_s"] = df2["LapSeconds"] - best
    df2["DropStr"] = df2["Drop_s"].apply(s_to_mssmmm)
    f = px.scatter(df2, x="TyreAge", y="Drop_s", color="Driver", custom_data=["DropStr","Compound"],
                   title="Caída de ritmo vs vida neumático (delta vs mejor vuelta)")
    f.update_traces(hovertemplate="%{fullData.name}<br>Edad %{x}<br>+%{customdata[0]} (%{customdata[1]})<extra></extra>", marker=dict(size=6))
    f.update_xaxes(title="edad neumático (v)")
    f.update_yaxes(title="caída (s)", tickformat=".3f")
    return set_trace_color(brand(f), color_map)

# ================= Run =================
server = app.server

from flask import jsonify

@server.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
