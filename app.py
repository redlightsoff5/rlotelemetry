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
from flask import jsonify

APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

YEARS_ALLOWED = [2025, 2026]

def _utc_today_token() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

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

SITE_TITLE = "Telemetría by RedLightsOff"
WATERMARK  = "@redlightsoff5"

IG_URL  = os.getenv("IG_URL", "https://instagram.com/redlightsoff5")
BMC_URL = os.getenv("BMC_URL", "https://buymeacoffee.com/redlightsoff5")

COL_BG    = "#9f9797"
COL_PANEL = "#ffffff"
COL_TEXT  = "#000000"
COL_RED   = "#e11d2e"

TEAM_COLORS_2025 = {
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
    "Cadillac":      "#6B7280",
}
TEAM_COLORS_2026 = {
    "Red Bull":      "#3671C6",
    "McLaren":       "#FF8000",
    "Ferrari":       "#E80020",
    "Mercedes":      "#27F4D2",
    "Aston Martin":  "#229971",
    "Alpine":        "#0093CC",
    "Williams":      "#64C4FF",
    "Racing Bulls":  "#6692FF",
    "Audi":          "#1F2937",
    "Haas":          "#B6BABD",
    "Cadillac":      "#6B7280",
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

    "sauber": "Sauber",
    "kick sauber": "Sauber",
    "stake": "Sauber",
    "stake f1": "Sauber",

    "audi": "Audi",

    "haas": "Haas",
    "haas f1": "Haas",

    "cadillac": "Cadillac",
    "gm cadillac": "Cadillac",
    "cadillac f1": "Cadillac",
    "cadillac f1 team": "Cadillac",
}

DRIVER_TEAM_OVERRIDE = {(2026, "PER"): "Cadillac", (2026, "BOT"): "Cadillac"}

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
    margin=dict(l=12, r=12, t=58, b=12),
)

def s_to_mssmmm(x):
    if pd.isna(x):
        return ""
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    m = int(x // 60)
    s = int(x % 60)
    ms = int(round((x - int(x)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    if s == 60:
        s = 0
        m += 1
    return f"{sign}{m}:{s:02d}.{ms:03d}"

def brand(fig):
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=42, color="rgba(0,0,0,0.10)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.25,
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

def is_race(ses):
    t = (getattr(ses, "session_type", "") or "").upper()
    n = (getattr(ses, "name", "") or "").upper()
    return t == "R" or "RACE" in n

@lru_cache(maxsize=64)
def load_session_laps(year:int, event_value:str, sess_code):
    """Load either a GP session or a testing day session.
    - GP: event_value is encoded 'GP|...' or raw EventName
    - TEST: event_value is 'TEST|<n>' and sess_code is day number (1..3)
    """
    info = parse_event_value(event_value)
    if not info:
        raise ValueError("No event selected")

    if info.get("type") == "test":
        test_number = int(info.get("test", 1))
        day = int(sess_code) if sess_code is not None else 1
        ses = ff1.get_testing_session(int(year), test_number, day)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        return ses

    # GP branch
    event_name = info.get("name") or str(event_value)
    code = str(sess_code)

    try:
        ses = ff1.get_session(int(year), event_name, code)
    except Exception:
        # Back-compat: Sprint Shootout naming
        if str(code).upper() == "SQ":
            ses = ff1.get_session(int(year), event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses.laps.loc[lap.name]
    except Exception:
        return None

@lru_cache(maxsize=128)
def _telemetry_cached(year: int, event_value: str, sess_code: str, driver: str, pick: str):
    ses = load_session(year, event_value, sess_code, telemetry=True)
    lap = _lap_for_driver(ses, driver, pick=pick)
    if lap is None:
        return None
    try:
        car = lap.get_car_data().add_distance()
    except Exception:
        return None
    keep = ["Distance", "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS"]
    for k in keep:
        if k not in car.columns:
            car[k] = np.nan
    car = car[keep].copy()
    car["Driver"] = driver
    return car.to_dict("list")

def _telemetry_df(year: int, event_value: str, sess_code: str, driver: str, pick: str = "best"):
    d = _telemetry_cached(year, event_value, sess_code, driver, pick)
    return pd.DataFrame(d) if d else pd.DataFrame()

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
            dcc.Dropdown(id="year-dd",
                         options=[{"label": str(y), "value": y} for y in YEARS_ALLOWED],
                         value=y0,
                         clearable=False),
            html.Div(id="year-warning", className="mt-1", style={"fontSize": "0.85rem", "opacity": 0.85})
        ], md=3),
        dbc.Col([
            dbc.Label("Gran Premio / Test"),
            dcc.Dropdown(id="event-dd",
                         options=build_gp_options(y0),
                         value=default_event_value(y0),
                         clearable=False,
                         placeholder="Selecciona evento...")
        ], md=6),
        dbc.Col([
            dbc.Label("Sesión"),
            dcc.Dropdown(id="session-dd",
                         options=SESSION_OPTIONS,
                         value="R",
                         clearable=False)
        ], md=3),
    ], className="g-2")

def graph_box(graph_id: str, title: str, chart_key: str, height=420):
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H5(title, className="m-0"), md=6, style={"display": "flex", "alignItems": "center"}),
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
                style={"height": f"{height}px"},
            ),
            type="default",
        ),
    ])

def telemetry_compare_box(graph_id: str, title: str, chart_key: str, height=460):
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H5(title, className="m-0"), md=4, style={"display": "flex", "alignItems": "center"}),
            dbc.Col([dcc.Dropdown(id={"role": "cmpA", "chart": chart_key}, options=[], value=None, clearable=True, placeholder="Piloto A")], md=3),
            dbc.Col([dcc.Dropdown(id={"role": "cmpB", "chart": chart_key}, options=[], value=None, clearable=True, placeholder="Piloto B")], md=3),
            dbc.Col([dcc.Dropdown(id={"role": "pick", "chart": chart_key},
                                  options=[{"label": "Mejor vuelta", "value": "best"}, {"label": "Última vuelta", "value": "last"}],
                                  value="best", clearable=False)], md=2),
        ], className="g-2 align-items-center"),
        dcc.Loading(
            dcc.Graph(
                id=graph_id,
                figure=fig_empty(title),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": f"{height}px"},
            ),
            type="default",
        ),
    ])

def tab_performance():
    return html.Div([
        dbc.Row([dbc.Col(graph_box("gap", "Gap", "gap"), md=6),
                 dbc.Col(graph_box("lapchart", "Lapchart", "lc"), md=6)], className="g-2"),
        dbc.Row([dbc.Col(graph_box("evo-pace", "Evolution Pace (MA3)", "ep"), md=6),
                 dbc.Col(graph_box("consistency", "Consistency (scatter)", "cons"), md=6)], className="g-2 mt-1"),
        dbc.Row([dbc.Col(graph_box("sectorsplit", "Sector split (box)", "secbox"), md=6),
                 dbc.Col(graph_box("best-laps", "Best Laps", "best"), md=6)], className="g-2 mt-1"),
        dbc.Row([dbc.Col(graph_box("speeds", "Speed Metrics", "spd"), md=12)], className="g-2 mt-1"),
    ])

def tab_strategy():
    return html.Div([
        dbc.Row([dbc.Col(graph_box("tyre-strategy", "Tyre Estrategia", "ty"), md=12)], className="g-2"),
        dbc.Row([dbc.Col(graph_box("compound-usage", "Uso de compuestos (vueltas)", "cu"), md=6),
                 dbc.Col(graph_box("pit-hist", "Pit stop laps", "pith"), md=6)], className="g-2 mt-1"),
        dbc.Row([dbc.Col(graph_box("degradation", "Degradation (LapTime vs TyreLife)", "deg"), md=12)], className="g-2 mt-1"),
    ])

def tab_telemetry():
    return html.Div([
        dbc.Row([dbc.Col(telemetry_compare_box("tel-speed", "Telemetría: Speed vs Distance", "telspeed"), md=12)], className="g-2"),
        dbc.Row([dbc.Col(telemetry_compare_box("tel-thrbrk", "Telemetría: Throttle / Brake", "telthr"), md=12)], className="g-2 mt-1"),
        dbc.Row([dbc.Col(telemetry_compare_box("tel-gear", "Telemetría: Gear / RPM", "telgear"), md=12)], className="g-2 mt-1"),
        dbc.Row([dbc.Col(telemetry_compare_box("tel-delta", "Telemetría: Delta de tiempo (aprox.)", "teldelta"), md=12)], className="g-2 mt-1"),
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
            dcc.Tab(label="Telemetría", value="tele", className="rlo-tab", selected_className="rlo-tab--selected"),
        ],
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_performance()),
    dcc.Store(id="store"),
    dcc.Store(id="drivers-store"),
    dcc.Store(id="team-color-store"),
], fluid=True, className="rlo-page")

@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def _render_tabs(val):
    return {"perf": tab_performance, "strat": tab_strategy, "tele": tab_telemetry}.get(val, tab_performance)()

@app.callback(
    Output("session-dd", "options"),
    Output("session-dd", "value"),
    Input("event-dd", "value"),
    State("session-dd", "value"),
)
def _event_changed_set_sessions(event_val, current):
    if not event_val:
        return SESSION_OPTIONS, "R"
    kind = str(event_val).split("|", 1)[0]
    if kind == "TEST":
        valid = {o["value"] for o in TEST_SESSION_OPTIONS}
        return TEST_SESSION_OPTIONS, (current if current in valid else "T1")
    valid = {o["value"] for o in SESSION_OPTIONS}
    return SESSION_OPTIONS, (current if current in valid else "R")

@app.callback(
    Output("event-dd", "options"),
    Output("event-dd", "value"),
    Output("year-warning", "children"),
    Input("year-dd", "value"),
    State("event-dd", "value"),
)
def _year_changed(year, current_event):
    if year is None:
        return [], None, ""
    year = int(year)
    try:
        opts = build_gp_options(year)
        default_ev = default_event_value(year)
        valid = {o["value"] for o in (opts or [])}
        value = current_event if current_event in valid else default_ev
        warn = "" if default_ev is not None else f"No completed events yet for {year}. Select 2025 to view data."
        return opts, value, warn
    except Exception:
        return [], None, f"Schedule unavailable for {year}."

@app.callback(
    Output("store", "data"),
    Output("drivers-store", "data"),
    Output("team-color-store", "data"),
    Input("year-dd", "value"),
    Input("event-dd", "value"),
    Input("session-dd", "value"),
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

@app.callback(
    Output({"role": "drv", "chart": ALL}, "options"),
    Output({"role": "drv", "chart": ALL}, "value"),
    Output({"role": "cmpA", "chart": ALL}, "options"),
    Output({"role": "cmpB", "chart": ALL}, "options"),
    Input("drivers-store", "data"),
    Input("tab-body", "children"),
    State({"role": "drv", "chart": ALL}, "id"),
    State({"role": "cmpA", "chart": ALL}, "id"),
    State({"role": "cmpB", "chart": ALL}, "id"),
)
def fill_dropdowns(drivers, _children, drv_ids, a_ids, b_ids):
    drivers = drivers or []
    opts = [{"label": d, "value": d} for d in drivers]
    drv_vals = [[] for _ in range(len(drv_ids))]
    return [opts] * len(drv_ids), drv_vals, [opts] * len(a_ids), [opts] * len(b_ids)

# ===== Rendimiento charts =====
@app.callback(Output("gap", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "gap"}, "value"),
              State("team-color-store", "data"))
def chart_gap(data, selected, color_map):
    if not data:
        return fig_empty("Gap — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = gap_to_leader_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Gap — no data")
    if is_race(ses):
        f = px.line(df, x="LapNumber", y="Gap_s", color="Driver", custom_data=["GapStr"], title="Diferencia al líder (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{fullData.name} — Lap %{x}<br>%{customdata[0]}<extra></extra>")
        f.update_yaxes(title="time", tickvals=[])
    else:
        gg = df.sort_values("Gap_s")
        f = px.bar(gg, x="Driver", y="Gap_s", custom_data=["GapStr"], title="Gap to Sesión Best (MM:SS.mmm)")
        f.update_traces(hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>")
        f.update_yaxes(title="time", tickvals=[])
    return set_trace_color(polish(f), color_map)

@app.callback(Output("lapchart", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "lc"}, "value"),
              State("team-color-store", "data"))
def chart_lapchart(data, selected, color_map):
    if not data:
        return fig_empty("Lapchart — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    laps = laps_df(ses)[["Driver", "LapNumber", "Position"]].dropna()
    if selected:
        laps = laps[laps["Driver"].isin(selected)]
    if laps.empty:
        return fig_empty("Lapchart — no data")
    f = px.line(laps, x="LapNumber", y="Position", color="Driver", title="Posiciones por vuelta (más bajo = mejor)")
    f.update_yaxes(autorange="reversed", dtick=1)
    return set_trace_color(polish(f), color_map)

@app.callback(Output("evo-pace", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "ep"}, "value"),
              State("team-color-store", "data"))
def chart_evolution(data, selected, color_map):
    if not data:
        return fig_empty("Evolution — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = pace_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Evolution — no lap data")
    df = df.sort_values(["Driver", "LapNumber"])
    df["MA3"] = df.groupby("Driver")["LapSeconds"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f = px.line(df, x="LapNumber", y="MA3", color="Driver", title="Evolución del ritmo (media móvil 3 vueltas)")
    f.update_traces(hovertemplate="%{fullData.name} — Lap %{x}<br>%{y:.3f}s<extra></extra>")
    f.update_yaxes(title="sec (MA3)")
    return set_trace_color(polish(f), color_map)

@app.callback(Output("consistency", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "cons"}, "value"),
              State("team-color-store", "data"))
def chart_consistency(data, selected, color_map):
    if not data:
        return fig_empty("Consistency — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = pace_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Consistency — no lap data")
    f = px.scatter(df, x="LapNumber", y="LapSeconds", color="Driver", title="Dispersión de consistencia")
    f.update_traces(hovertemplate="%{fullData.name} — Lap %{x}<br>%{y:.3f}s<extra></extra>")
    f.update_yaxes(title="sec")
    return set_trace_color(polish(f), color_map)

@app.callback(Output("sectorsplit", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "secbox"}, "value"),
              State("team-color-store", "data"))
def chart_sector_box(data, selected, color_map):
    if not data:
        return fig_empty("Sector split — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = sector_times_long_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Sector split — no data")
    f = px.box(df, x="Sector", y="Time_s", color="Driver", title="Distribución de tiempos por sector (box)")
    f.update_traces(hovertemplate="%{fullData.name} — %{x}<br>%{y:.3f}s<extra></extra>")
    f.update_yaxes(title="sec")
    return set_trace_color(polish(f), color_map)

@app.callback(Output("best-laps", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "best"}, "value"),
              State("team-color-store", "data"))
def chart_bestlaps(data, selected, color_map):
    if not data:
        return fig_empty("Best Laps — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    best = best_laps_df(ses)
    if selected:
        best = best[best["Driver"].isin(selected)]
    if best.empty:
        return fig_empty("Best Laps — no data")
    f = px.bar(best, x="Driver", y="Best_s", custom_data=["BestStr"], title="Mejores vueltas (MM:SS.mmm)")
    f.update_traces(hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>")
    f.update_yaxes(title="time", tickvals=[])
    return set_trace_color(polish(f), color_map)

@app.callback(Output("speeds", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "spd"}, "value"),
              State("team-color-store", "data"))
def chart_speeds(data, selected, color_map):
    if not data:
        return fig_empty("Speed Metrics — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    spd = speed_metrics_df(ses)
    if spd.empty:
        return fig_empty("Speed Metrics — no data")
    if selected:
        spd = spd[spd["Driver"].isin(selected)]
    if spd.empty:
        return fig_empty("Speed Metrics — no data")
    dm = spd.melt(id_vars="Driver", var_name="Metric", value_name="km/h")
    f = px.bar(dm, x="Driver", y="km/h", color="Metric", barmode="group", title="Velocidades máximas (por piloto)")
    f.update_yaxes(title="km/h")
    f.update_traces(marker_line_width=0)
    return set_trace_color(polish(f), color_map)

# ===== Estrategia charts =====
@app.callback(Output("tyre-strategy", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "ty"}, "value"))
def chart_tyres(data, selected):
    if not data:
        return fig_empty("Tyre Estrategia — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    st = tyre_stints_df(ses)
    if selected:
        st = st[st["Driver"].isin(selected)]
    if st.empty:
        return fig_empty("Tyre Estrategia — no data")
    cmap = {"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0F0", "INTERMEDIATE": "#43B02A", "WET": "#00A3E0"}
    f = go.Figure()
    order = st["Driver"].unique().tolist()[::-1]
    for _, r in st.iterrows():
        f.add_trace(go.Bar(
            x=[int(r["Laps"])], y=[r["Driver"]], base=[int(r["LapStart"]) - 1],
            orientation="h", marker_color=cmap.get(str(r["Compound"]).upper(), "#888"),
            showlegend=False,
            hovertemplate=f"{r['Driver']} — {r['Compound']}<br>Lap {int(r['LapStart'])}–{int(r['LapEnd'])}<extra></extra>",
        ))
    for n, c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))
    f.update_layout(title="Tyre Estrategia", barmode="stack",
                    yaxis=dict(categoryorder="array", categoryarray=order, title="Driver"),
                    xaxis_title="Lap")
    return polish(f, grid=True)

@app.callback(Output("compound-usage", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "cu"}, "value"))
def chart_compound_usage(data, selected):
    if not data:
        return fig_empty("Compound usage — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = compound_usage_df(ses)
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Compound usage — no data")
    f = px.bar(df, x="Driver", y="Laps", color="Compound", barmode="stack", title="Uso de compuestos (vueltas)")
    f.update_traces(marker_line_width=0)
    return polish(f)

@app.callback(Output("pit-hist", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "pith"}, "value"))
def chart_pits(data, selected):
    if not data:
        return fig_empty("Pit stop laps — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = pit_laps_df(ses)
    if selected and not df.empty:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Pit stop laps — no data")
    f = px.histogram(df, x="LapNumber", color="Driver", nbins=25, title="Distribución de paradas (aprox.)")
    f.update_layout(barmode="overlay")
    return polish(f)

@app.callback(Output("degradation", "figure"),
              Input("store", "data"),
              Input({"role": "drv", "chart": "deg"}, "value"),
              State("team-color-store", "data"))
def chart_degradation(data, selected, color_map):
    if not data:
        return fig_empty("Degradation — (no data)")
    ses = load_session_laps(int(data["year"]), data["event"], data["sess"])
    df = pace_df(ses)
    if "TyreLife" not in df.columns:
        return fig_empty("Degradation — TyreLife not available")
    df = df.dropna(subset=["TyreLife", "LapSeconds"])
    if selected:
        df = df[df["Driver"].isin(selected)]
    if df.empty:
        return fig_empty("Degradation — no data")
    f = px.scatter(df, x="TyreLife", y="LapSeconds", color="Driver", title="Degradación: tiempo de vuelta vs vida del neumático")
    f.update_traces(hovertemplate="%{fullData.name}<br>TyreLife %{x:.0f}<br>%{y:.3f}s<extra></extra>")
    f.update_yaxes(title="sec")
    return set_trace_color(polish(f), color_map)

# ===== Telemetría charts =====
@app.callback(Output("tel-speed", "figure"),
              Input("store", "data"),
              Input({"role": "cmpA", "chart": "telspeed"}, "value"),
              Input({"role": "cmpB", "chart": "telspeed"}, "value"),
              Input({"role": "pick", "chart": "telspeed"}, "value"),
              State("team-color-store", "data"))
def tel_speed(data, a, b, pick, color_map):
    if not data:
        return fig_empty("Telemetría speed — (no data)")
    if not a and not b:
        return fig_empty("Telemetría speed — select Piloto A and/or B")
    year = int(data["year"]); ev = data["event"]; sc = data["sess"]
    dfA = _telemetry_df(year, ev, sc, a, pick) if a else pd.DataFrame()
    dfB = _telemetry_df(year, ev, sc, b, pick) if b else pd.DataFrame()
    if dfA.empty and dfB.empty:
        return fig_empty("Telemetría speed — no telemetry available")
    f = go.Figure()
    if not dfA.empty:
        f.add_trace(go.Scatter(x=dfA["Distance"], y=dfA["Speed"], mode="lines", name=dfA["Driver"].iloc[0]))
    if not dfB.empty:
        f.add_trace(go.Scatter(x=dfB["Distance"], y=dfB["Speed"], mode="lines", name=dfB["Driver"].iloc[0], line=dict(dash="dash")))
    f.update_layout(title="Telemetría: Speed vs Distance")
    f.update_xaxes(title="Distance (m)")
    f.update_yaxes(title="Speed (km/h)")
    return set_trace_color(polish(f), color_map)

@app.callback(Output("tel-thrbrk", "figure"),
              Input("store", "data"),
              Input({"role": "cmpA", "chart": "telthr"}, "value"),
              Input({"role": "cmpB", "chart": "telthr"}, "value"),
              Input({"role": "pick", "chart": "telthr"}, "value"),
              State("team-color-store", "data"))
def tel_thrbrk(data, a, b, pick, color_map):
    if not data:
        return fig_empty("Telemetría throttle/brake — (no data)")
    if not a and not b:
        return fig_empty("Telemetría throttle/brake — select Piloto A and/or B")
    year = int(data["year"]); ev = data["event"]; sc = data["sess"]
    dfA = _telemetry_df(year, ev, sc, a, pick) if a else pd.DataFrame()
    dfB = _telemetry_df(year, ev, sc, b, pick) if b else pd.DataFrame()
    if dfA.empty and dfB.empty:
        return fig_empty("Telemetría throttle/brake — no telemetry available")

    f = go.Figure()
    def add_traces(df, dash_th="solid", dash_br="dot"):
        name = df["Driver"].iloc[0]
        f.add_trace(go.Scatter(x=df["Distance"], y=df["Throttle"], mode="lines", name=f"{name} Throttle", line=dict(dash=dash_th)))
        f.add_trace(go.Scatter(x=df["Distance"], y=df["Brake"], mode="lines", name=f"{name} Brake", line=dict(dash=dash_br)))
    if not dfA.empty:
        add_traces(dfA, "solid", "dot")
    if not dfB.empty:
        add_traces(dfB, "dash", "dashdot")

    f.update_layout(title="Telemetría: Throttle / Brake")
    f.update_xaxes(title="Distance (m)")
    f.update_yaxes(title="%", range=[0, 105])

    cmap = color_map or {}
    for tr in f.data:
        base = str(tr.name).split(" ")[0]
        c = cmap.get(base)
        if c:
            tr.update(line=dict(color=c, dash=tr.line.dash))
    return polish(f)

@app.callback(Output("tel-gear", "figure"),
              Input("store", "data"),
              Input({"role": "cmpA", "chart": "telgear"}, "value"),
              Input({"role": "cmpB", "chart": "telgear"}, "value"),
              Input({"role": "pick", "chart": "telgear"}, "value"),
              State("team-color-store", "data"))
def tel_gear(data, a, b, pick, color_map):
    if not data:
        return fig_empty("Telemetría gear/rpm — (no data)")
    if not a and not b:
        return fig_empty("Telemetría gear/rpm — select Piloto A and/or B")
    year = int(data["year"]); ev = data["event"]; sc = data["sess"]
    dfA = _telemetry_df(year, ev, sc, a, pick) if a else pd.DataFrame()
    dfB = _telemetry_df(year, ev, sc, b, pick) if b else pd.DataFrame()
    if dfA.empty and dfB.empty:
        return fig_empty("Telemetría gear/rpm — no telemetry available")

    f = go.Figure()
    def add(df, dash_g="solid", dash_r="dot"):
        name = df["Driver"].iloc[0]
        f.add_trace(go.Scatter(x=df["Distance"], y=df["nGear"], mode="lines", name=f"{name} Gear", line=dict(dash=dash_g)))
        f.add_trace(go.Scatter(x=df["Distance"], y=df["RPM"], mode="lines", name=f"{name} RPM", line=dict(dash=dash_r)))
    if not dfA.empty:
        add(dfA, "solid", "dot")
    if not dfB.empty:
        add(dfB, "dash", "dashdot")

    f.update_layout(title="Telemetría: Gear / RPM")
    f.update_xaxes(title="Distance (m)")
    f.update_yaxes(title="Gear / RPM")

    cmap = color_map or {}
    for tr in f.data:
        base = str(tr.name).split(" ")[0]
        c = cmap.get(base)
        if c:
            tr.update(line=dict(color=c, dash=tr.line.dash))
    return polish(f)

@app.callback(Output("tel-delta", "figure"),
              Input("store", "data"),
              Input({"role": "cmpA", "chart": "teldelta"}, "value"),
              Input({"role": "cmpB", "chart": "teldelta"}, "value"),
              Input({"role": "pick", "chart": "teldelta"}, "value"))
def tel_delta(data, a, b, pick):
    if not data:
        return fig_empty("Telemetría delta — (no data)")
    if not a or not b:
        return fig_empty("Telemetría delta — select Piloto A and Piloto B")
    year = int(data["year"]); ev = data["event"]; sc = data["sess"]
    dfA = _telemetry_df(year, ev, sc, a, pick)
    dfB = _telemetry_df(year, ev, sc, b, pick)
    if dfA.empty or dfB.empty:
        return fig_empty("Telemetría delta — telemetry not available")
    grid = np.linspace(0, min(dfA["Distance"].max(), dfB["Distance"].max()), 2000)
    vA = np.interp(grid, dfA["Distance"].to_numpy(), dfA["Speed"].to_numpy()) / 3.6
    vB = np.interp(grid, dfB["Distance"].to_numpy(), dfB["Speed"].to_numpy()) / 3.6
    vA = np.clip(vA, 1e-3, None)
    vB = np.clip(vB, 1e-3, None)
    ds = np.diff(grid, prepend=grid[0])
    tA = np.cumsum(ds / vA)
    tB = np.cumsum(ds / vB)
    out = pd.DataFrame({"Distance": grid, "Delta_s": (tB - tA)})
    f = px.line(out, x="Distance", y="Delta_s", title=f"Delta de tiempo (aprox.): {b} vs {a}")
    f.update_traces(hovertemplate="Dist %{x:.0f}m<br>Δ %{y:.3f}s<extra></extra>")
    f.update_yaxes(title="Δ seconds (B - A)")
    return polish(f)

server = app.server

@server.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))@app.callback(
    Output('session-dd','options'),
    Output('session-dd','value'),
    Input('event-dd','value'),
    State('session-dd','value')
)
def _event_changed_session_options(event_val, current):
    info = parse_event_value(event_val)
    if not info:
        return SESSION_OPTIONS, 'R'
    if info.get('type') == 'test':
        # Testing days: 1..3
        val = current if current in (1,2,3) else 1
        return TEST_SESSION_OPTIONS, val
    # GP sessions
    val = current if current in [o['value'] for o in SESSION_OPTIONS] else 'R'
    return SESSION_OPTIONS, val


