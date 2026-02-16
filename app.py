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
SITE_TITLE = "RLO Telemetría"
WATERMARK  = "@redlightsoff5"

IG_URL  = os.getenv("IG_URL",  "https://instagram.com/redlightsoff5")
BMC_URL = os.getenv("BMC_URL", "https://buymeacoffee.com/redlightsoff5")

COL_BG    = "#9f9797"
COL_PANEL = "#ffffff"
COL_RED   = "#e11d2e"
COL_TEXT  = "#000000"

# ===== Team colors =====
TEAM_COLORS_2025 = {
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

TEAM_COLORS_2026 = {
    **TEAM_COLORS_2025,
    # Your requested tweaks
    "Audi": "#1F2937",       # dark graphite
    "Cadillac": "#6B7280",   # grey
}

TEAM_ALIASES_2025 = {
    "oracle red bull": "Red Bull",
    "red bull": "Red Bull",
    "red bull racing": "Red Bull",

    "mclaren": "McLaren",
    "ferrari": "Ferrari",
    "mercedes": "Mercedes",

    "aston martin": "Aston Martin",
    "alpine": "Alpine",
    "williams": "Williams",

    "rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "visa cash app": "Racing Bulls",
    "racing bulls": "Racing Bulls",

    "stake": "Sauber",
    "kick": "Sauber",
    "sauber": "Sauber",
    "haas": "Haas",
}

TEAM_ALIASES_2026 = {
    **TEAM_ALIASES_2025,
    "audi": "Audi",
    # if FastF1 still reports "Sauber" in 2026, you want it bucketed as Audi
    "sauber": "Audi",
    "kick": "Audi",
    "stake": "Audi",
    "cadillac": "Cadillac",
}

def canonical_team(name: str, year: int) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    aliases = TEAM_ALIASES_2026 if int(year) == 2026 else TEAM_ALIASES_2025
    for key, canon in aliases.items():
        if key in s:
            return canon
    return name

def team_palette(year: int):
    return TEAM_COLORS_2026 if int(year) == 2026 else TEAM_COLORS_2025

# ================= Helpers =================
COMMON_LAYOUT = dict(
    paper_bgcolor=COL_PANEL,
    plot_bgcolor=COL_PANEL,
    font=dict(color=COL_TEXT),
    margin=dict(l=12, r=12, t=48, b=12)
)

def brand(fig):
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=42, color="rgba(0,0,0,0.08)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.25
    )
    return fig

def fig_empty(title):
    f = go.Figure()
    f.update_layout(title=title, **COMMON_LAYOUT)
    return brand(f)

def s_to_mssmmm(x):
    if pd.isna(x): return ""
    x = float(x)
    m = int(x // 60); s = int(x % 60); ms = int(round((x - int(x))*1000))
    return f"{m}:{s:02d}.{ms:03d}"

def set_trace_color(fig, name_to_color):
    for tr in fig.data:
        c = (name_to_color or {}).get(tr.name)
        if c:
            tr.update(line=dict(color=c), marker=dict(color=c))
    return fig

def parse_event_value(val: str):
    if not val:
        return None
    s = str(val)
    if s.startswith("TEST|"):
        try:
            return {"type": "test", "test": int(s.split("|", 1)[1])}
        except Exception:
            return {"type": "test", "test": 1}
    if s.startswith("GP|"):
        out = {"type": "gp"}
        # GP|round=4|name=Bahrain Grand Prix
        parts = s.split("|")[1:]
        for p in parts:
            if p.startswith("round="):
                try: out["round"] = int(p.split("=",1)[1])
                except Exception: pass
            if p.startswith("name="):
                out["name"] = p.split("=",1)[1]
        return out
    return {"type": "gp", "name": s}

def is_race(ses):
    t = (getattr(ses, "session_type", "") or "").upper()
    n = (getattr(ses, "name", "") or "").upper()
    return t == "R" or "RACE" in n

# ================= Schedule (incl. testing) =================
@lru_cache(maxsize=8)
def get_schedule_df(year: int, date_token: str) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=True).copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    for col in ["RoundNumber", "EventName", "EventDate", "EventFormat"]:
        if col not in df.columns:
            df[col] = None
    df["EventFormat"] = df["EventFormat"].astype(str).str.lower()
    df = df[["RoundNumber","EventName","EventDate","EventFormat"]].sort_values(["EventDate","RoundNumber"]).reset_index(drop=True)
    return df

def build_event_options(year: int):
    df = get_schedule_df(year, _utc_today_token())
    testing_df = df[df["EventFormat"].str.contains("test", na=False)].sort_values("EventDate").reset_index(drop=True)
    test_keys = {(r.EventDate.date(), str(r.EventName)): i+1 for i, r in testing_df.iterrows()}

    opts = []
    for _, r in df.iterrows():
        fmt = str(r.EventFormat).lower()
        date = r.EventDate.date()
        name = str(r.EventName)
        if "test" in fmt:
            n = test_keys.get((date, name), 1)
            opts.append({"label": f"Test de Pretemporada #{n} ({date})", "value": f"TEST|{n}"})
        else:
            # some rows can have RoundNumber NaN; guard
            rn = r.RoundNumber
            try:
                rn_int = int(rn)
            except Exception:
                rn_int = 0
            if rn_int > 0:
                opts.append({"label": f"R{rn_int} — {name} ({date})", "value": f"GP|round={rn_int}|name={name}"})
            else:
                opts.append({"label": f"{name} ({date})", "value": f"GP|name={name}"})
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
    rn = last.get("RoundNumber", None)
    try:
        rn_int = int(rn)
        return f"GP|round={rn_int}|name={str(last['EventName'])}"
    except Exception:
        return f"GP|name={str(last['EventName'])}"

def default_year_value() -> int:
    # newest year with at least one completed item
    today = pd.Timestamp.utcnow().tz_localize(None)
    for y in sorted(YEARS_ALLOWED, reverse=True):
        try:
            df = get_schedule_df(y, _utc_today_token())
            if not df.empty and (df["EventDate"] <= today).any():
                return y
        except Exception:
            continue
    return YEARS_ALLOWED[0]

# ================= Sessions =================
SESSION_OPTIONS_GP = [
    {"label": "FP1", "value": "FP1"},
    {"label": "FP2", "value": "FP2"},
    {"label": "FP3", "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},
    {"label": "Qualifying", "value": "Q"},
    {"label": "Sprint", "value": "SR"},
    {"label": "Carrera", "value": "R"},
]

SESSION_OPTIONS_TEST = [
    {"label": "Día 1", "value": "T1"},
    {"label": "Día 2", "value": "T2"},
    {"label": "Día 3", "value": "T3"},
]

@lru_cache(maxsize=64)
def load_session(year: int, event_value: str, sess_code: str):
    info = parse_event_value(event_value)
    if not info:
        raise ValueError("No event selected")

    year = int(year)
    sess_code = str(sess_code).upper()

    # Testing
    if info["type"] == "test":
        test_number = int(info.get("test", 1))
        day_number = int(sess_code.replace("T", "")) if sess_code.startswith("T") else int(sess_code)
        # Primary API
        try:
            ses = ff1.get_testing_session(year, test_number, day_number)
            ses.load(laps=True, telemetry=False, weather=False, messages=False)
            if hasattr(ses, "laps") and not ses.laps.empty:
                return ses
        except Exception:
            ses = None

        # Fallback: sometimes exposed as a normal session name like "Pre-Season Testing" + "Day 1"
        # We try multiple common variants.
        candidates = [
            ("Pre-Season Testing", f"Day {day_number}"),
            ("Pre-Season Testing", f"Día {day_number}"),
            ("Pre-Season Testing", f"Day{day_number}"),
            ("Pre-Season Testing", str(day_number)),
        ]
        for ev, sc in candidates:
            try:
                s2 = ff1.get_session(year, ev, sc)
                s2.load(laps=True, telemetry=False, weather=False, messages=False)
                if hasattr(s2, "laps") and not s2.laps.empty:
                    return s2
            except Exception:
                continue

        # Return the primary session even if empty (UI will show "no data")
        return ses

    # GP
    name = info.get("name") or str(event_value)
    try:
        ses = ff1.get_session(year, name, sess_code)
    except Exception:
        if sess_code == "SQ":
            ses = ff1.get_session(year, name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

def driver_color_map(ses):
    laps = getattr(ses, "laps", pd.DataFrame())
    if laps is None or laps.empty or not {"Driver","Team"}.issubset(laps.columns):
        return {}
    year = int(getattr(getattr(ses, "event", None), "year", 0) or 0) or 2025
    palette = team_palette(year)

    team = (laps[["Driver","Team"]]
            .dropna(subset=["Driver"])
            .groupby("Driver")["Team"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
            .apply(lambda t: canonical_team(t, year)))

    return {drv: palette.get(tm, "#999999") for drv, tm in team.items()}

# ================= Data builders (cached) =================
@lru_cache(maxsize=64)
def _pace_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    laps = laps.dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds().astype(float)
    laps["LapStr"] = laps["LapSeconds"].apply(s_to_mssmmm)
    return laps[["Driver","LapNumber","LapSeconds","LapStr"]]

@lru_cache(maxsize=64)
def _gap_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    laps = laps.dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()

    if is_race(ses):
        laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
        laps["Cum"] = laps.groupby("Driver")["LapSeconds"].cumsum()
        lead = laps.groupby("LapNumber")["Cum"].min().rename("Lead").reset_index()
        d = laps.merge(lead, on="LapNumber", how="left")
        d["Gap_s"] = d["Cum"] - d["Lead"]
        d["GapStr"] = d["Gap_s"].apply(s_to_mssmmm)
        return d[["Driver","LapNumber","Gap_s","GapStr"]]

    best = laps.groupby("Driver")["LapTime"].min().rename("Best").reset_index()
    gbest = best["Best"].min()
    best["Gap_s"] = (best["Best"] - gbest).dt.total_seconds()
    best["GapStr"] = best["Gap_s"].apply(s_to_mssmmm)
    best["LapNumber"] = 1
    return best[["Driver","LapNumber","Gap_s","GapStr"]]

@lru_cache(maxsize=64)
def _lapchart_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    if laps.empty or not {"Driver","LapNumber","Position"}.issubset(laps.columns):
        return pd.DataFrame()
    df = laps[["Driver","LapNumber","Position"]].dropna()
    return df

@lru_cache(maxsize=64)
def _tyre_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    if laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    laps["Compound"] = laps["Compound"].astype(str).str.upper()
    agg = (laps.groupby(["Driver","Stint","Compound"])
               .agg(LapStart=("LapNumber","min"), LapEnd=("LapNumber","max")))
    agg["Laps"] = agg["LapEnd"] - agg["LapStart"] + 1
    return agg.reset_index().sort_values(["Driver","Stint"])

@lru_cache(maxsize=64)
def _best_laps_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    laps = laps.dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    best = laps.loc[laps.groupby("Driver")["LapTime"].idxmin()].copy()
    best["Best_s"] = best["LapTime"].dt.total_seconds()
    best["BestStr"] = best["Best_s"].apply(s_to_mssmmm)
    return best[["Driver","Best_s","BestStr"]].sort_values("Best_s")

@lru_cache(maxsize=64)
def _sector_records_df(year:int, event_value:str, sess_code:str):
    ses = load_session(year, event_value, sess_code)
    laps = getattr(ses, "laps", pd.DataFrame()).copy()
    if laps.empty:
        return pd.DataFrame()
    out=[]
    for i,c in enumerate(["Sector1Time","Sector2Time","Sector3Time"], start=1):
        if c not in laps.columns:
            continue
        # dropna then idxmin
        s = laps.dropna(subset=[c])
        if s.empty:
            continue
        idx = s[c].idxmin()
        row = s.loc[idx]
        out.append({"Sector": f"S{i}", "Driver": row["Driver"], "Time_s": round(row[c].total_seconds(), 3)})
    return pd.DataFrame(out)

# ================= Dash app =================
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
      <div class="rlo-title"><span>RLO</span> Telemetría</div>
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
    return dbc.Row([
        dbc.Col([
            dbc.Label("Año"),
            dcc.Dropdown(
                id="year-dd",
                options=[{"label": str(y), "value": y} for y in YEARS_ALLOWED],
                value=y0,
                clearable=False
            ),
        ], md=2),
        dbc.Col([
            dbc.Label("Gran Premio / Test"),
            dcc.Dropdown(
                id="event-dd",
                options=build_event_options(y0),
                value=default_event_value(y0),
                clearable=False,
                placeholder="Selecciona evento..."
            ),
        ], md=7),
        dbc.Col([
            dbc.Label("Sesión"),
            dcc.Dropdown(
                id="session-dd",
                options=SESSION_OPTIONS_GP,
                value="R",
                clearable=False
            ),
        ], md=3),
    ], className="g-2")

def graph_box(graph_id: str, title: str, chart_key: str):
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H5(title, className="m-0"), md=6, style={"display":"flex","alignItems":"center"}),
            dbc.Col([
                dcc.Dropdown(
                    id={"role": "drv", "chart": chart_key},
                    multi=True,
                    placeholder="Selecciona pilotos",
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

def tab_rendimiento():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("gap", "Diferencia", "gap"), md=6),
            dbc.Col(graph_box("lapchart", "Lapchart", "lc"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("pace", "Ritmo vuelta a vuelta", "pace"), md=12),
        ], className="g-2 mt-1"),
    ])

def tab_estrategia():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("tyre-strategy", "Estrategia de neumáticos", "ty"), md=12),
        ], className="g-2"),
    ])

def tab_records():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("best-laps", "Mejores vueltas", "best"), md=6),
            dbc.Col(graph_box("sectors", "Récords por sector", "sec"), md=6),
        ], className="g-2"),
    ])

app.layout = dbc.Container([
    header_controls(),
    dcc.Tabs(
        id="tabs",
        value="perf",
        parent_className="rlo-tabs-parent",
        className="rlo-tabs",
        children=[
            dcc.Tab(label="Rendimiento", value="perf", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Estrategia", value="strat", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Récords", value="rec", className="rlo-tab", selected_className="rlo-tab--selected"),
        ],
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_rendimiento()),
    dcc.Store(id="store"),
    dcc.Store(id="drivers-store"),
    dcc.Store(id="color-store"),
], fluid=True, className="rlo-page")

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"perf": tab_rendimiento, "strat": tab_estrategia, "rec": tab_records}.get(val, tab_rendimiento)()

# year change -> event options/value
@app.callback(
    Output("event-dd","options"),
    Output("event-dd","value"),
    Input("year-dd","value"),
    State("event-dd","value"),
)
def _year_changed(year, current):
    if year is None:
        return [], None
    year = int(year)
    opts = build_event_options(year)
    valid = {o["value"] for o in opts}
    default_ev = default_event_value(year)
    val = current if current in valid else default_ev
    return opts, val

# event change -> session options (GP vs Test)
@app.callback(
    Output("session-dd","options"),
    Output("session-dd","value"),
    Input("event-dd","value"),
    State("session-dd","value")
)
def _event_changed(event_val, current):
    info = parse_event_value(event_val)
    if not info:
        return SESSION_OPTIONS_GP, "R"
    if info["type"] == "test":
        val = current if current in {"T1","T2","T3"} else "T1"
        return SESSION_OPTIONS_TEST, val
    # GP
    valid = {o["value"] for o in SESSION_OPTIONS_GP}
    val = current if current in valid else "R"
    return SESSION_OPTIONS_GP, val

# load session meta once for current selection
@app.callback(
    Output("store","data"),
    Output("drivers-store","data"),
    Output("color-store","data"),
    Input("year-dd","value"),
    Input("event-dd","value"),
    Input("session-dd","value"),
)
def load_meta(year, event_value, sess_code):
    if not year or not event_value or not sess_code:
        return no_update, [], {}
    try:
        ses = load_session(int(year), str(event_value), str(sess_code))
        laps = getattr(ses, "laps", pd.DataFrame()).dropna(subset=["LapTime"])
        drivers = sorted(laps["Driver"].dropna().unique().tolist()) if not laps.empty else []
        colors = driver_color_map(ses)
        return {"year": int(year), "event": str(event_value), "sess": str(sess_code)}, drivers, colors
    except Exception:
        traceback.print_exc()
        return no_update, [], {}

# populate driver dropdowns only for mounted controls
@app.callback(
    Output({"role":"drv","chart":ALL}, "options"),
    Output({"role":"drv","chart":ALL}, "value"),
    Input("drivers-store","data"),
    Input("tab-body","children"),
    State({"role":"drv","chart":ALL}, "id"),
)
def fill_dropdowns(drivers, _children, ids):
    opts = [{"label": d, "value": d} for d in (drivers or [])]
    # default: all drivers selected for non-testing sessions, none for testing (faster + cleaner)
    # We'll detect testing by existence of "TEST|" in store in each chart callback, so here keep empty.
    n = len(ids)
    return [opts]*n, [[]]*n

# ================= Charts =================
@app.callback(
    Output("gap","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"gap"}, "value"),
    State("color-store","data")
)
def chart_gap(store, selected, colors):
    if not store:
        return fig_empty("Diferencia — (sin datos)")
    df = _gap_df(store["year"], store["event"], store["sess"])
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Diferencia — (sin datos)")
    # race -> line; non-race -> bar
    ses = load_session(store["year"], store["event"], store["sess"])
    if is_race(ses):
        f = px.line(df, x="LapNumber", y="Gap_s", color="Driver", custom_data=["GapStr"],
                    title="Diferencia al líder (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{fullData.name} — Vuelta %{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="s", tickformat=".3f")
    else:
        gg = df.sort_values("Gap_s")
        f = px.bar(gg, x="Driver", y="Gap_s", custom_data=["GapStr"],
                   title="Diferencia a la mejor vuelta de la sesión (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="s", tickformat=".3f")
    return set_trace_color(brand(f), colors)

@app.callback(
    Output("lapchart","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"lc"}, "value"),
    State("color-store","data")
)
def chart_lapchart(store, selected, colors):
    if not store:
        return fig_empty("Lapchart — (sin datos)")
    df = _lapchart_df(store["year"], store["event"], store["sess"])
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Lapchart — (sin datos)")
    f = px.line(df, x="LapNumber", y="Position", color="Driver", title="Posiciones por vuelta (más bajo = mejor)")
    f.update_yaxes(autorange="reversed", dtick=1)
    return set_trace_color(brand(f), colors)

@app.callback(
    Output("pace","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"pace"}, "value"),
    State("color-store","data")
)
def chart_pace(store, selected, colors):
    if not store:
        return fig_empty("Ritmo — (sin datos)")
    df = _pace_df(store["year"], store["event"], store["sess"])
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Ritmo — (sin datos)")
    f = go.Figure()
    for drv, d in df.groupby("Driver"):
        f.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapSeconds"], mode="lines+markers", name=str(drv),
            hovertemplate=f"{drv} — Vuelta %{{x}}<br>%{{y:.3f}}s (%{{text}})<extra></extra>",
            text=d["LapStr"]
        ))
    f.update_layout(title="Ritmo vuelta a vuelta")
    f.update_yaxes(title="s", tickformat=".3f")
    return set_trace_color(brand(f), colors)

@app.callback(
    Output("tyre-strategy","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"ty"}, "value"),
)
def chart_tyres(store, selected):
    if not store:
        return fig_empty("Estrategia de neumáticos — (sin datos)")
    st = _tyre_df(store["year"], store["event"], store["sess"])
    if selected:
        st = st[st["Driver"].isin(selected)]
    if st.empty:
        return fig_empty("Estrategia de neumáticos — (sin datos)")
    f = go.Figure()
    order = st["Driver"].unique().tolist()[::-1]
    cmap = {"SOFT":"#DA291C","MEDIUM":"#FFD12E","HARD":"#F0F0F0","INTERMEDIATE":"#43B02A","WET":"#00A3E0"}
    for _, r in st.iterrows():
        f.add_trace(go.Bar(
            x=[int(r["Laps"])], y=[r["Driver"]], base=[int(r["LapStart"])-1],
            orientation="h", marker_color=cmap.get(str(r["Compound"]).upper(), "#888"),
            showlegend=False,
            hovertemplate=f"{r['Driver']} — {r['Compound']}<br>Vuelta {int(r['LapStart'])}–{int(r['LapEnd'])}<extra></extra>"
        ))
    for n,c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))
    f.update_layout(title="Estrategia de neumáticos", barmode="stack",
                    yaxis=dict(categoryorder="array", categoryarray=order, title="Piloto"),
                    xaxis_title="Vuelta")
    f.update_xaxes(showgrid=True)
    f.update_yaxes(showgrid=False)
    return brand(f)

@app.callback(
    Output("best-laps","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"best"}, "value"),
    State("color-store","data")
)
def chart_best(store, selected, colors):
    if not store:
        return fig_empty("Mejores vueltas — (sin datos)")
    df = _best_laps_df(store["year"], store["event"], store["sess"])
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Mejores vueltas — (sin datos)")
    f = px.bar(df, x="Driver", y="Best_s", custom_data=["BestStr"], title="Mejores vueltas (MM:SS.mmm)")
    f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
    f.update_yaxes(title="s", tickformat=".3f")
    return set_trace_color(brand(f), colors)

@app.callback(
    Output("sectors","figure"),
    Input("store","data"),
    Input({"role":"drv","chart":"sec"}, "value"),
)
def chart_sectors(store, selected):
    if not store:
        return fig_empty("Récords por sector — (sin datos)")
    df = _sector_records_df(store["year"], store["event"], store["sess"])
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Récords por sector — (sin datos)")
    f = go.Figure(data=[go.Table(
        header=dict(values=["Sector","Piloto","Tiempo (s)"],
                    fill_color="#FFFFFF", font=dict(color="#000000", size=12)),
        cells=dict(values=[df["Sector"], df["Driver"], df["Time_s"]],
                   fill_color="#FFFFFF", font=dict(color="#000000"))
    )])
    f.update_layout(title="Récords por sector", paper_bgcolor=COL_PANEL)
    return brand(f)

# ================= Run =================
server = app.server

from flask import jsonify

@server.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
