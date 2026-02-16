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
    "Cadillac":      "#7a7a7a",
    "Audi":          "#9b9b9b",
}

TEAM_ALIASES = {
    "red bull": "Red Bull",
    "oracle red bull": "Red Bull",
    "red bull racing": "Red Bull",

    "mclaren": "McLaren",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",

    "mercedes": "Mercedes",
    "mercedes-amg": "Mercedes",

    "aston martin": "Aston Martin",

    "alpine": "Alpine",

    "williams": "Williams",

    # VCARB / RB / Racing Bulls
    "racing bulls": "Racing Bulls",
    "visa cash app rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "rb f1": "Racing Bulls",
    "rb": "Racing Bulls",

    # Sauber/Kick/Stake/Audi bucket
    "sauber": "Sauber",
    "kick sauber": "Sauber",
    "stake": "Sauber",
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

COMMON_LAYOUT = dict(
    paper_bgcolor=COL_PANEL,
    plot_bgcolor=COL_PANEL,
    font=dict(color=COL_TEXT),
    margin=dict(l=12, r=12, t=52, b=12),
)

def brand(fig: go.Figure) -> go.Figure:
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=44, color="rgba(0,0,0,0.08)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
    )
    return fig

def fig_empty(title: str) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, **COMMON_LAYOUT)
    return brand(f)

def polish(fig: go.Figure, grid: bool = False) -> go.Figure:
    fig.update_layout(**COMMON_LAYOUT)
    if not grid:
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
    return brand(fig)

def is_race(ses) -> bool:
    t = (getattr(ses, "session_type", "") or "").upper()
    n = (getattr(ses, "name", "") or "").upper()
    return t == "R" or "RACE" in n

# ---------- Schedule / options ----------
@lru_cache(maxsize=8)
def get_schedule_df(year: int, date_token: str) -> pd.DataFrame:
    # include_testing=True so we can show Bahrain pre-season testing
    df = ff1.get_event_schedule(year, include_testing=True).copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    cols = ["RoundNumber", "EventName", "EventFormat", "EventDate"]
    df = df[cols].sort_values(["EventDate", "RoundNumber"]).reset_index(drop=True)
    return df

def build_event_options(year: int):
    df = get_schedule_df(year, _utc_today_token())
    # Determine testing numbering by chronological order of testing days
    testing_df = df[df["EventFormat"] == "testing"].sort_values("EventDate")
    test_dates = testing_df["EventDate"].dt.date.astype(str).tolist()

    opts = []
    for _, r in df.iterrows():
        fmt = str(r.EventFormat).lower()
        date = r.EventDate.date()
        name = str(r.EventName)

        if fmt == "testing":
            # test_number based on chronological date among testing events
            test_number = test_dates.index(str(date)) + 1
            opts.append({
                "label": f"Test de Pretemporada #{test_number} ({date})",
                "value": f"TEST|{test_number}"
            })
        else:
            opts.append({
                "label": f"R{int(r.RoundNumber)} — {name} ({date})",
                "value": f"GP|{name}"
            })
    return opts

def default_event_value(year: int):
    df = get_schedule_df(year, _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df["EventDate"] <= today]
    if not past.empty:
        last = past.iloc[-1]
        fmt = str(last.EventFormat).lower()
        if fmt == "testing":
            # compute test_number similarly
            testing_df = df[df["EventFormat"] == "testing"].sort_values("EventDate")
            test_dates = testing_df["EventDate"].dt.date.astype(str).tolist()
            test_number = test_dates.index(str(last.EventDate.date())) + 1
            return f"TEST|{test_number}"
        return f"GP|{str(last.EventName)}"
    # If nothing in past, pick first
    if not df.empty:
        first = df.iloc[0]
        fmt = str(first.EventFormat).lower()
        if fmt == "testing":
            return "TEST|1"
        return f"GP|{str(first.EventName)}"
    return None

SESSION_OPTIONS_GP = [
    {"label": "FP1", "value": "FP1"},
    {"label": "FP2", "value": "FP2"},
    {"label": "FP3", "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},
    {"label": "Clasificación", "value": "Q"},
    {"label": "Sprint", "value": "SR"},
    {"label": "Carrera", "value": "R"},
]
SESSION_OPTIONS_TEST = [
    {"label": "Día 1", "value": "D1"},
    {"label": "Día 2", "value": "D2"},
    {"label": "Día 3", "value": "D3"},
]

# ---------- Loaders ----------
@lru_cache(maxsize=64)
def load_session_laps(year: int, event_value: str, sess_code: str):
    """
    event_value:
      - "GP|<EventName>"
      - "TEST|<test_number>"
    sess_code:
      - GP: FP1/FP2/FP3/SQ/Q/SR/R
      - TEST: D1/D2/D3
    """
    year = int(year)
    event_value = str(event_value or "")
    sess_code = str(sess_code or "")

    if event_value.startswith("TEST|"):
        test_number = int(event_value.split("|", 1)[1])
        day = int(sess_code.replace("D", "")) if sess_code.startswith("D") else 1
        ses = ff1.get_testing_session(year, test_number, day)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        return ses

    # GP weekend
    event_name = event_value.split("|", 1)[1] if "|" in event_value else event_value
    try:
        ses = ff1.get_session(year, event_name, sess_code)
    except Exception:
        # Back-compat for Sprint Shootout naming
        if sess_code.upper() == "SQ":
            ses = ff1.get_session(year, event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

def driver_team_color_map(ses):
    laps = ses.laps[["Driver", "Team"]].dropna()
    if laps.empty:
        return {}
    team = laps.groupby("Driver")["Team"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
    ).apply(canonical_team)
    return {drv: TEAM_COLORS.get(t, "#888888") for drv, t in team.items()}

def set_trace_color(fig, name_to_color):
    for tr in fig.data:
        c = (name_to_color or {}).get(tr.name)
        if c:
            tr.update(line=dict(color=c), marker=dict(color=c))
    return fig

# ---------- Basic graphs (kept from your style) ----------
def pace_df(ses):
    laps = ses.laps.copy().dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds().astype(float)
    return laps[["Driver", "LapNumber", "LapSeconds"]]

def gap_to_leader_df(ses):
    laps = ses.laps.copy().dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Cum"] = laps.groupby("Driver")["LapSeconds"].cumsum()
    lead = laps.groupby("LapNumber")["Cum"].min().rename("Lead").reset_index()
    d = laps.merge(lead, on="LapNumber", how="left")
    d["Gap_s"] = d["Cum"] - d["Lead"]
    return d[["Driver", "LapNumber", "Gap_s"]]

# ---------- Telemetry helpers ----------
@lru_cache(maxsize=128)
def fastest_lap_for_driver(year: int, event_value: str, sess_code: str, driver: str):
    ses = load_session_laps(year, event_value, sess_code)
    laps = ses.laps.pick_driver(driver).dropna(subset=["LapTime"])
    if laps.empty:
        return None
    return laps.pick_fastest()

@lru_cache(maxsize=128)
def telemetry_for_fastest_lap(year: int, event_value: str, sess_code: str, driver: str):
    lap = fastest_lap_for_driver(year, event_value, sess_code, driver)
    if lap is None:
        return pd.DataFrame()
    try:
        car = lap.get_car_data().add_distance()
    except Exception:
        return pd.DataFrame()

    df = car.copy()
    # Normalize columns we’ll use
    keep = [c for c in ["Distance", "Speed", "Throttle", "Brake", "nGear", "RPM", "DRS", "Time"] if c in df.columns]
    df = df[keep].dropna(subset=["Distance"])
    # Convert Time to seconds from lap start if present
    if "Time" in df.columns:
        try:
            df["t"] = df["Time"].dt.total_seconds()
        except Exception:
            pass
    return df

def line_telemetry(df, x, y, title, ytitle):
    if df.empty or x not in df.columns or y not in df.columns:
        return fig_empty(f"{title} — (sin datos)")
    f = px.line(df, x=x, y=y, title=title)
    f.update_layout(xaxis_title="Distancia (m)", yaxis_title=ytitle)
    return polish(f)

def step_drs(df):
    if df.empty or "Distance" not in df.columns or "DRS" not in df.columns:
        return fig_empty("DRS — (sin datos)")
    d = df[["Distance", "DRS"]].copy()
    # DRS can be {0,1,8,10,12} depending on feed; treat >0 as active
    d["DRS_active"] = (pd.to_numeric(d["DRS"], errors="coerce").fillna(0) > 0).astype(int)
    f = px.line(d, x="Distance", y="DRS_active", title="DRS (activo=1) vs distancia")
    f.update_yaxes(dtick=1, range=[-0.05, 1.05])
    f.update_layout(xaxis_title="Distancia (m)", yaxis_title="DRS")
    return polish(f)

def delta_time_trace(dfA, dfB, nameA, nameB):
    if dfA.empty or dfB.empty or "Distance" not in dfA.columns or "Distance" not in dfB.columns:
        return fig_empty("Delta tiempo — (sin datos)")
    if "t" not in dfA.columns or "t" not in dfB.columns:
        return fig_empty("Delta tiempo — (sin Time)")
    # common distance grid
    dmin = max(dfA["Distance"].min(), dfB["Distance"].min())
    dmax = min(dfA["Distance"].max(), dfB["Distance"].max())
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        return fig_empty("Delta tiempo — (sin solape)")
    grid = np.linspace(dmin, dmax, 1500)

    tA = np.interp(grid, dfA["Distance"].to_numpy(), dfA["t"].to_numpy())
    tB = np.interp(grid, dfB["Distance"].to_numpy(), dfB["t"].to_numpy())
    delta = tB - tA

    f = go.Figure()
    f.add_trace(go.Scatter(x=grid, y=delta, mode="lines", name=f"{nameB} − {nameA}"))
    f.update_layout(title="Delta tiempo vs distancia", xaxis_title="Distancia (m)", yaxis_title="Δt (s)")
    return polish(f)

# ================= Dash =================
external_stylesheets = [dbc.themes.FLATLY]
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
    <div class="logo-wrap"><img src="/assets/logo.png" alt="logo"/></div>
    <div class="rlo-brand">
      <div class="rlo-title">RLO</div>
      <div class="rlo-subtitle">by {WATERMARK}</div>
    </div>
    <div class="rlo-actions">
      <a class="rlo-action rlo-ig" href="{IG_URL}" target="_blank" rel="noopener noreferrer">Instagram</a>
      <a class="rlo-action rlo-bmc" href="{BMC_URL}" target="_blank" rel="noopener noreferrer">Support</a>
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
    y0 = YEARS_ALLOWED[-1]
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
            dbc.Label("Evento"),
            dcc.Dropdown(
                id="event-dd",
                options=build_event_options(y0),
                value=default_event_value(y0),
                clearable=False
            )
        ], md=7),
        dbc.Col([
            dbc.Label("Sesión"),
            dcc.Dropdown(
                id="session-dd",
                options=SESSION_OPTIONS_GP,
                value="R",
                clearable=False
            )
        ], md=3),
    ], className="g-2")

def graph_box(graph_id: str, title: str, chart_key: str, kind: str="multi"):
    is_multi = (kind == "multi")
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H5(title, className="m-0"), md=6, style={"display":"flex","alignItems":"center"}),
            dbc.Col([
                dcc.Dropdown(
                    id={"role":"drv","kind":kind,"chart":chart_key},
                    multi=is_multi,
                    placeholder=("Filtra pilotos (opcional)" if is_multi else "Selecciona piloto"),
                    options=[],
                    value=([] if is_multi else None),
                    clearable=True
                )
            ], md=6),
        ], className="g-2 align-items-center"),
        dcc.Loading(
            dcc.Graph(
                id=graph_id,
                figure=fig_empty(title),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": "420px"}
            )
        )
    ])

def tab_evolution():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("gap","Gap","gap","multi"), md=6),
            dbc.Col(graph_box("lapchart","Lapchart","lc","multi"), md=6),
        ], className="g-2"),
    ])

def tab_telemetry():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("tel-speed","Velocidad vs distancia","tele","single"), md=6),
            dbc.Col(graph_box("tel-throttle","Acelerador vs distancia","tele","single"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("tel-brake","Freno vs distancia","tele","single"), md=6),
            dbc.Col(graph_box("tel-gear","Marcha vs distancia","tele","single"), md=6),
        ], className="g-2 mt-1"),
        dbc.Row([
            dbc.Col(graph_box("tel-rpm","RPM vs distancia","tele","single"), md=6),
            dbc.Col(graph_box("tel-drs","DRS vs distancia","tele","single"), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_delta():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(className="box", children=[
                    dbc.Row([
                        dbc.Col(html.H5("Delta tiempo (B − A)", className="m-0"), md=6),
                        dbc.Col(dcc.Dropdown(
                            id={"role":"drv","kind":"single","chart":"delta_a"},
                            options=[], value=None, clearable=True,
                            placeholder="Piloto A"
                        ), md=3),
                        dbc.Col(dcc.Dropdown(
                            id={"role":"drv","kind":"single","chart":"delta_b"},
                            options=[], value=None, clearable=True,
                            placeholder="Piloto B"
                        ), md=3),
                    ], className="g-2 align-items-center"),
                    dcc.Loading(dcc.Graph(
                        id="delta-fig",
                        figure=fig_empty("Delta tiempo"),
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height":"520px"}
                    ))
                ])
            ], md=12)
        ])
    ])

app.layout = dbc.Container([
    header_controls(),
    dcc.Tabs(
        id="tabs",
        value="evo",
        parent_className="rlo-tabs-parent",
        className="rlo-tabs",
        children=[
            dcc.Tab(label="Evolution", value="evo", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Telemetry", value="tele", className="rlo-tab", selected_className="rlo-tab--selected"),
            dcc.Tab(label="Delta", value="delta", className="rlo-tab", selected_className="rlo-tab--selected"),
        ],
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_evolution()),
    dcc.Store(id="store"),
    dcc.Store(id="drivers-store"),
    dcc.Store(id="team-color-store"),
], fluid=True, className="rlo-page")

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"evo": tab_evolution, "tele": tab_telemetry, "delta": tab_delta}.get(val, tab_evolution)()

# ---- session options switch based on event type ----
@app.callback(
    Output("session-dd","options"),
    Output("session-dd","value"),
    Input("event-dd","value"),
    State("session-dd","value")
)
def _switch_session_options(event_value, current):
    if not event_value:
        return SESSION_OPTIONS_GP, "R"
    if str(event_value).startswith("TEST|"):
        # default day 1
        val = current if current in {"D1","D2","D3"} else "D1"
        return SESSION_OPTIONS_TEST, val
    val = current if current in {o["value"] for o in SESSION_OPTIONS_GP} else "R"
    return SESSION_OPTIONS_GP, val

# ---- year -> event list ----
@app.callback(
    Output("event-dd","options"),
    Output("event-dd","value"),
    Input("year-dd","value"),
    State("event-dd","value")
)
def _year_changed(year, current_event):
    if year is None:
        return [], None
    year = int(year)
    opts = build_event_options(year)
    valid = {o["value"] for o in opts}
    value = current_event if current_event in valid else default_event_value(year)
    return opts, value

# ---- load session meta once ----
@app.callback(
    Output("store","data"),
    Output("drivers-store","data"),
    Output("team-color-store","data"),
    Input("year-dd","value"),
    Input("event-dd","value"),
    Input("session-dd","value")
)
def load_session_meta(year, event_value, sess_code):
    if not year or not event_value or not sess_code:
        return no_update, [], {}
    try:
        ses = load_session_laps(int(year), str(event_value), str(sess_code))
        laps = ses.laps.dropna(subset=["LapTime"])
        drivers = sorted(laps["Driver"].dropna().unique().tolist())
        colors = driver_team_color_map(ses)
        return {"year": int(year), "event": str(event_value), "sess": str(sess_code)}, drivers, colors
    except Exception:
        traceback.print_exc()
        return no_update, [], {}

# ---- populate MULTI driver dropdowns (filters) ----
@app.callback(
    Output({"role":"drv","kind":"multi","chart":ALL}, "options"),
    Output({"role":"drv","kind":"multi","chart":ALL}, "value"),
    Input("drivers-store","data"),
    Input("tab-body","children"),
    State({"role":"drv","kind":"multi","chart":ALL}, "id"),
)
def fill_multi_dropdowns(drivers, _children, ids):
    opts = [{"label": d, "value": d} for d in (drivers or [])]
    n = len(ids)
    return [opts]*n, [[]]*n

# ---- populate SINGLE driver dropdowns (telemetry + delta) ----
@app.callback(
    Output({"role":"drv","kind":"single","chart":ALL}, "options"),
    Output({"role":"drv","kind":"single","chart":ALL}, "value"),
    Input("drivers-store","data"),
    Input("tab-body","children"),
    State({"role":"drv","kind":"single","chart":ALL}, "id"),
)
def fill_single_dropdowns(drivers, _children, ids):
    opts = [{"label": d, "value": d} for d in (drivers or [])]
    n = len(ids)
    return [opts]*n, [None]*n

# ---------- Evolution charts ----------
@app.callback(
    Output("gap","figure"),
    Input("store","data"),
    Input({"role":"drv","kind":"multi","chart":"gap"}, "value"),
    State("team-color-store","data"),
)
def chart_gap(data, selected, color_map):
    if not data:
        return fig_empty("(sin datos)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = gap_to_leader_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Gap — (sin datos)")
    f = px.line(df, x="LapNumber", y="Gap_s", color="Driver", title="Gap al líder (s)")
    f.update_yaxes(title="s")
    return set_trace_color(polish(f), color_map)

@app.callback(
    Output("lapchart","figure"),
    Input("store","data"),
    Input({"role":"drv","kind":"multi","chart":"lc"}, "value"),
    State("team-color-store","data"),
)
def chart_lapchart(data, selected, color_map):
    if not data:
        return fig_empty("(sin datos)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    laps = ses.laps[["Driver","LapNumber","Position"]].dropna()
    if selected:
        laps = laps[laps["Driver"].isin(selected)]
    if laps.empty:
        return fig_empty("Lapchart — (sin datos)")
    f = px.line(laps, x="LapNumber", y="Position", color="Driver", title="Posición vs vuelta")
    f.update_yaxes(autorange="reversed", dtick=1)
    return set_trace_color(polish(f), color_map)

# ---------- Telemetry charts (single driver) ----------
@app.callback(
    Output("tel-speed","figure"),
    Output("tel-throttle","figure"),
    Output("tel-brake","figure"),
    Output("tel-gear","figure"),
    Output("tel-rpm","figure"),
    Output("tel-drs","figure"),
    Input("store","data"),
    Input({"role":"drv","kind":"single","chart":"tele"}, "value"),
)
def telemetry_charts(data, driver):
    if not data or not driver:
        empty = fig_empty("Selecciona un piloto")
        return empty, empty, empty, empty, empty, empty
    df = telemetry_for_fastest_lap(int(data["year"]), data["event"], data["sess"], str(driver))
    f1 = line_telemetry(df, "Distance", "Speed", "Velocidad vs distancia", "km/h")
    f2 = line_telemetry(df, "Distance", "Throttle", "Acelerador vs distancia", "%")
    f3 = line_telemetry(df, "Distance", "Brake", "Freno vs distancia", "0/1")
    f4 = line_telemetry(df, "Distance", "nGear", "Marcha vs distancia", "gear")
    f5 = line_telemetry(df, "Distance", "RPM", "RPM vs distancia", "rpm")
    f6 = step_drs(df)
    return f1, f2, f3, f4, f5, f6

# ---------- Delta (two drivers) ----------
@app.callback(
    Output("delta-fig","figure"),
    Input("store","data"),
    Input({"role":"drv","kind":"single","chart":"delta_a"}, "value"),
    Input({"role":"drv","kind":"single","chart":"delta_b"}, "value"),
)
def delta_chart(data, drv_a, drv_b):
    if not data or not drv_a or not drv_b or drv_a == drv_b:
        return fig_empty("Selecciona 2 pilotos distintos")
    dfA = telemetry_for_fastest_lap(int(data["year"]), data["event"], data["sess"], str(drv_a))
    dfB = telemetry_for_fastest_lap(int(data["year"]), data["event"], data["sess"], str(drv_b))
    return delta_time_trace(dfA, dfB, str(drv_a), str(drv_b))

# ================= Run =================
server = app.server

from flask import jsonify

@server.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
