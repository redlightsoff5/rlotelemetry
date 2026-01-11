
import os, warnings, logging, traceback
from functools import lru_cache
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc
import dash_echarts

from flask_caching import Cache
from flask_compress import Compress

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# Years supported in the UI
YEARS_ALLOWED = [2025, 2026]

# Optional: donations button (set in Render env vars)
BMC_URL = os.environ.get("BMC_URL", "").strip()

SITE_TITLE = "RLO Telemetry"
TAGLINE = "Race pace, tyres, and evolution — updated as soon as sessions are available."
WATERMARK = "@redlightsoff5"

# UI colors (CSS drives most of this; keep here for charts)
COL_BG    = "#9f9797"
COL_CARD  = "#ffffff"
COL_TEXT  = "#111111"
COL_MUTED = "rgba(0,0,0,0.62)"
COL_RED   = "#c0001a"

# Broadcast-style team colors (same for both drivers of a team)
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
}

# Map possible FastF1 team strings to canonical keys
TEAM_ALIASES = {
    "red bull": "Red Bull",
    "oracle red bull": "Red Bull",
    "red bull racing": "Red Bull",

    "racing bulls": "Racing Bulls",
    "visa cash app rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "rb f1": "Racing Bulls",
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
    "audi": "Sauber",

    "haas": "Haas",
    "haas f1": "Haas",
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
    if pd.isna(x):
        return ""
    x = float(x)
    m = int(x // 60)
    s = int(x % 60)
    ms = int(round((x - int(x)) * 1000))
    return f"{m}:{s:02d}.{ms:03d}"

def _utc_today_token() -> str:
    # used to refresh cached schedules daily without redeploys
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

@lru_cache(maxsize=8)
def get_schedule_df(year: int, date_token: str) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=False).copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    df = df[["RoundNumber", "EventName", "EventDate"]].sort_values("RoundNumber").reset_index(drop=True)
    return df

def build_gp_options(year: int):
    df = get_schedule_df(year, _utc_today_token())
    return [
        {
            "label": f"R{int(r.RoundNumber)} — {r.EventName} ({r.EventDate.date()})",
            "value": r.EventName,
        }
        for _, r in df.iterrows()
    ]

def default_event_value(year: int):
    df = get_schedule_df(year, _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df["EventDate"] <= today]
    if not past.empty:
        return past.iloc[-1]["EventName"]
    return None

def default_year_value() -> int:
    """Pick newest allowed year with at least one completed event; else 2025."""
    today = pd.Timestamp.utcnow().tz_localize(None)
    for y in sorted(YEARS_ALLOWED, reverse=True):
        try:
            df = get_schedule_df(y, _utc_today_token())
            if not df.empty and (df["EventDate"] <= today).any():
                return y
        except Exception:
            continue
    return 2025

SESSION_OPTIONS = [
    {"label": "FP1",               "value": "FP1"},
    {"label": "FP2",               "value": "FP2"},
    {"label": "FP3",               "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},
    {"label": "Qualifying",        "value": "Q"},
    {"label": "Sprint",            "value": "SR"},
    {"label": "Race",              "value": "R"},
]

def is_race(ses) -> bool:
    t = (getattr(ses, "session_type", "") or "").upper()
    n = (getattr(ses, "name", "") or "").upper()
    return t == "R" or "RACE" in n

# ---------- Loaders ----------
@lru_cache(maxsize=64)
def load_session_laps(year: int, event_name: str, sess_code: str):
    """Load a session by official EventName and session code."""
    try:
        ses = ff1.get_session(int(year), event_name, str(sess_code))
    except Exception:
        if str(sess_code).upper() == "SQ":
            ses = ff1.get_session(int(year), event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

# ================= Dash =================
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>{SITE_TITLE}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  {{%favicon%}}
  {{%css%}}
</head>
<body>
  {{%app_entry%}}
  {{%config%}}
  {{%scripts%}}
  {{%renderer%}}
</body>
</html>
"""

server = app.server

# ---- Compression ----
Compress(server)

# ---- Server-side cache for processed data ----
cache = Cache(server, config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": os.path.join(CACHE_DIR, "flask_cache"),
    "CACHE_DEFAULT_TIMEOUT": 3600,
    "CACHE_THRESHOLD": 200,
})

# ================= Utility: server routes =================
@server.route("/health")
def health():
    return "ok", 200

@server.route("/warmup")
def warmup():
    out = []
    for y in YEARS_ALLOWED:
        try:
            ev = default_event_value(y)
            if not ev:
                out.append(f"{y}: no completed events")
                continue
            for sc in ("R", "Q", "SR"):
                try:
                    _ = get_processed(int(y), str(ev), str(sc))
                    out.append(f"{y} {ev} {sc}: ok")
                except Exception:
                    out.append(f"{y} {ev} {sc}: fail")
        except Exception:
            out.append(f"{y}: fail")
    return "\n".join(out), 200

# ================= Processed data layer (compute once, reuse everywhere) =================
def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")

def _df_from_json(s: str) -> pd.DataFrame:
    if not s:
        return pd.DataFrame()
    try:
        return pd.read_json(s, orient="split")
    except ValueError:
        from io import StringIO
        return pd.read_json(StringIO(s), orient="split")

def driver_team_color_map_from_laps(laps: pd.DataFrame) -> Dict[str, str]:
    if laps is None or laps.empty or not {"Driver", "Team"}.issubset(laps.columns):
        return {}
    lt = laps[["Driver", "Team"]].dropna()
    if lt.empty:
        return {}
    team = (lt.groupby("Driver")["Team"]
              .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
              .apply(canonical_team))
    return {drv: TEAM_COLORS.get(t, "#999999") for drv, t in team.items()}

def gap_to_leader_df_from_session(ses) -> pd.DataFrame:
    laps = ses.laps.copy().dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    if is_race(ses):
        laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
        laps["Cum"] = laps.groupby("Driver", dropna=False)["LapSeconds"].cumsum()
        lead = laps.groupby("LapNumber", dropna=False)["Cum"].min().rename("Lead").reset_index()
        d = laps.merge(lead, on="LapNumber", how="left")
        d["Gap_s"] = d["Cum"] - d["Lead"]
        return d[["Driver", "LapNumber", "Gap_s"]]
    best = laps.groupby("Driver", dropna=False)["LapTime"].min().rename("Best").reset_index()
    gbest = best["Best"].min()
    best["Gap_s"] = (best["Best"] - gbest).dt.total_seconds()
    best["LapNumber"] = 1
    return best[["Driver", "LapNumber", "Gap_s"]]

def positions_gained_df_from_session(ses) -> pd.DataFrame:
    res = ses.results
    if res is None or res.empty or not {"Abbreviation", "GridPosition", "Position"}.issubset(res.columns):
        return pd.DataFrame()
    df = res[["Abbreviation", "GridPosition", "Position"]].copy()
    df["Driver"] = df["Abbreviation"]
    df["PositionsGained"] = df["GridPosition"] - df["Position"]
    return df.sort_values("PositionsGained", ascending=False)

def tyre_stints_df_from_session(ses) -> pd.DataFrame:
    laps = ses.laps.copy()
    if laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    laps["Compound"] = laps["Compound"].astype(str).str.upper()
    agg = (laps.groupby(["Driver", "Stint", "Compound"], dropna=False)
               .agg(LapStart=("LapNumber", "min"), LapEnd=("LapNumber", "max")))
    agg["Laps"] = agg["LapEnd"] - agg["LapStart"] + 1
    return agg.reset_index().sort_values(["Driver", "Stint"])

def pace_df_from_session(ses) -> pd.DataFrame:
    laps = ses.laps.copy().dropna(subset=["LapTime"])
    if laps.empty:
        return pd.DataFrame()
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds().astype(float)
    return laps[["Driver", "LapNumber", "LapSeconds"]]

def sector_records_df_from_session(ses) -> pd.DataFrame:
    laps = ses.laps.copy()
    out = []
    for i, c in enumerate(["Sector1Time", "Sector2Time", "Sector3Time"], start=1):
        if c not in laps.columns:
            continue
        try:
            idx = laps[c].idxmin()
        except Exception:
            continue
        if pd.isna(idx):
            continue
        row = laps.loc[idx]
        if pd.isna(row.get(c)):
            continue
        out.append({"Sector": f"S{i}", "Driver": row.get("Driver", ""), "Time_s": float(row[c].total_seconds())})
    return pd.DataFrame(out)

def speed_records_df_from_session(ses) -> pd.DataFrame:
    laps = ses.laps.copy()
    cols = [c for c in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if c in laps.columns]
    if not cols:
        return pd.DataFrame()
    grp = laps.groupby("Driver", dropna=False)[cols].max().reset_index()
    return grp.rename(columns={"SpeedI1": "I1", "SpeedI2": "I2", "SpeedFL": "Finish", "SpeedST": "Trap"})

@cache.memoize(timeout=60 * 60)
def get_processed(year: int, event_name: str, sess_code: str) -> Dict[str, Any]:
    ses = load_session_laps(int(year), str(event_name), str(sess_code))

    gap = gap_to_leader_df_from_session(ses)
    lapchart = ses.laps[["Driver", "LapNumber", "Position"]].dropna() if not ses.laps.empty else pd.DataFrame()
    pace = pace_df_from_session(ses)
    tyres = tyre_stints_df_from_session(ses)
    posg = positions_gained_df_from_session(ses)
    sectors = sector_records_df_from_session(ses)
    speeds = speed_records_df_from_session(ses)

    evo = pd.DataFrame()
    if not pace.empty:
        evo = pace.sort_values(["Driver", "LapNumber"]).copy()
        evo["MA3"] = evo.groupby("Driver", dropna=False)["LapSeconds"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    drivers = sorted(pace["Driver"].dropna().unique().tolist()) if not pace.empty else []
    colors = driver_team_color_map_from_laps(ses.laps[["Driver", "Team"]].dropna()) if not ses.laps.empty else {}

    return {
        "meta": {"year": int(year), "event": str(event_name), "sess": str(sess_code), "is_race": bool(is_race(ses))},
        "drivers": drivers,
        "colors": colors,
        "gap": _df_to_json(gap),
        "lapchart": _df_to_json(lapchart),
        "pace": _df_to_json(pace),
        "evo": _df_to_json(evo),
        "tyres": _df_to_json(tyres),
        "posg": _df_to_json(posg),
        "sectors": _df_to_json(sectors),
        "speeds": _df_to_json(speeds),
    }

def _legend_selected_map(drivers, selected):
    selected = selected or []
    if not selected:
        return {d: False for d in drivers}
    selset = set(selected)
    return {d: (d in selset) for d in drivers}

def _add_watermark(option: Dict[str, Any], text: str = "@redlightsoff5") -> Dict[str, Any]:
    option = dict(option or {})
    option.setdefault("graphic", [])
    option["graphic"].append({
        "type": "text",
        "left": "center",
        "top": "middle",
        "rotation": -0.5,  # radians
        "style": {
            "text": text,
            "fontSize": 46,
            "fontWeight": 700,
            "fontFamily": "Inter, system-ui, Arial",
            "fill": "rgba(192,0,26,0.12)",
        },
        "silent": True,
        "z": 0
    })
    return option

# ================= ECharts option builders =================
def option_empty(title: str, subtitle: str = "") -> Dict[str, Any]:
    # Subtitle is accepted for compatibility but intentionally not displayed
    opt = {
        "title": {"text": title, "left": "center", "textStyle": {"color": COL_TEXT, "fontFamily": "Inter, system-ui, Arial"}},
        "grid": {"left": 50, "right": 25, "top": 55, "bottom": 45},
        "xAxis": {"type": "category", "data": []},
        "yAxis": {"type": "value"},
        "series": [],
    }
    return _add_watermark(opt)

def option_line_by_driver(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_name: str,
    color_map: Dict[str, str],
    y_inverse: bool = False,
    smooth: bool = True,
    selected_drivers: Optional[list] = None,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return option_empty(title, "No data available for this session.")
    df = df.dropna(subset=[x_col, y_col, "Driver"])
    if df.empty:
        return option_empty(title, "No data available for this session.")

    drivers_sorted = sorted([str(d) for d in df["Driver"].dropna().unique().tolist()])

    series = []
    for drv, g in df.groupby("Driver", dropna=False):
        pts = [[int(x), float(y)] for x, y in zip(g[x_col].astype(int), g[y_col].astype(float))]
        series.append({
            "name": str(drv),
            "type": "line",
            "data": pts,
            "showSymbol": False,
            "smooth": smooth,
            "lineStyle": {"width": 2, "color": (color_map or {}).get(str(drv), "#333")},
            "emphasis": {"focus": "series"},
        })

    opt = {
        "backgroundColor": "rgba(0,0,0,0)",
        "animation": True,
        "tooltip": {"trigger": "axis"},
        "legend": {
            "type": "scroll",
            "top": 0,
            "textStyle": {"color": COL_TEXT},
            "selected": _legend_selected_map(drivers_sorted, selected_drivers),
        },
        "grid": {"left": 55, "right": 20, "top": 40, "bottom": 55},
        "toolbox": {"feature": {"saveAsImage": {}, "dataZoom": {}, "restore": {}}},
        "xAxis": {"type": "value", "name": "Lap", "nameTextStyle": {"color": COL_TEXT}, "axisLabel": {"color": COL_TEXT}},
        "yAxis": {
            "type": "value",
            "name": y_name,
            "inverse": y_inverse,
            "nameTextStyle": {"color": COL_TEXT},
            "axisLabel": {"color": COL_TEXT},
            "splitLine": {"lineStyle": {"color": "rgba(0,0,0,0.10)"}},
        },
        "dataZoom": [
            {"type": "inside", "xAxisIndex": 0},
            {"type": "slider", "xAxisIndex": 0, "height": 22, "bottom": 10},
        ],
        "series": series,
        "title": {"text": title, "left": "center", "top": 22, "textStyle": {"color": COL_TEXT, "fontFamily": "Inter, system-ui, Arial"}},
    }
    return _add_watermark(opt)

def option_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str = COL_RED,
    selected_drivers: Optional[list] = None,
) -> Dict[str, Any]:
    # If explicitly told "no drivers selected", keep chart empty
    if selected_drivers is not None and len(selected_drivers) == 0:
        return option_empty(title, "")

    if df is None or df.empty:
        return option_empty(title, "No data available.")
    df = df.dropna(subset=[x_col, y_col])

    if selected_drivers and x_col in df.columns:
        df = df[df[x_col].astype(str).isin([str(d) for d in selected_drivers])]

    if df.empty:
        return option_empty(title, "No data available.")

    cats = df[x_col].astype(str).tolist()
    vals = df[y_col].astype(float).tolist()
    opt = {
        "backgroundColor": "rgba(0,0,0,0)",
        "animation": True,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 55, "right": 20, "top": 55, "bottom": 55},
        "xAxis": {"type": "category", "data": cats, "axisLabel": {"color": COL_TEXT, "rotate": 35}},
        "yAxis": {"type": "value", "axisLabel": {"color": COL_TEXT}, "splitLine": {"lineStyle": {"color": "rgba(0,0,0,0.10)"}}},
        "series": [{"type": "bar", "data": vals, "itemStyle": {"color": color}, "barMaxWidth": 28}],
        "title": {"text": title, "left": "center", "textStyle": {"color": COL_TEXT, "fontFamily": "Inter, system-ui, Arial"}},
    }
    return _add_watermark(opt)

# ================= Layout blocks =================
# ================= Layout blocks =================
def logo_block():
    """
    Put your own logo in: ./assets/logo.png
    """
    assets_logo = os.path.join(APP_DIR, "assets", "logo.png")
    if os.path.exists(assets_logo):
        return html.Img(src="/assets/logo.png", className="rlo-logo-img", alt="RedLightsOff")
    return html.Div(className="rlo-logo", children="RLO")

def header_bar():
    donate = dbc.Button("Support", href=BMC_URL, target="_blank", color="danger", className="rlo-btn-pill") if BMC_URL else None
    return html.Div(className="rlo-topbar", children=[
        html.Div(className="rlo-brand", children=[
            logo_block(),
            html.Div(children=[
                html.Div(className="rlo-title", children=["Telemetry ", html.Span("by RedLightsOff", className="rlo-title-accent")]),
                html.Div(className="rlo-subtitle", children=TAGLINE),
            ])
        ]),
        html.Div(className="rlo-topbar-actions", children=[
            donate,
            dbc.Button("Instagram", href="https://instagram.com/redlightsoff5", target="_blank", outline=True, color="danger", className="rlo-btn-pill"),
        ])
    ])

def hero_strip():
    return html.Div(className="rlo-hero", children=[
        html.Div(className="rlo-hero-left", children=[
            html.H2("How to use", className="rlo-h2"),
            html.Ul(className="rlo-bullets", children=[
                html.Li([html.B("1)"), " Select Year → GP → Session"]),
                html.Li([html.B("2)"), " Click Top 5 / Top 10 / All to quickly show drivers"]),
                html.Li([html.B("3)"), " Toggle drivers by tapping their name in the legend (team colours)"]),
                html.Li([html.B("4)"), " Zoom: drag. Reset: double‑click"]),
            ]),
        ]),
    ])

def filter_card():
    y0 = default_year_value()
    return html.Div(className="box rlo-filter", children=[
        dbc.Row([
            dbc.Col([
                dbc.Label("Year", className="dbc-label"),
                dcc.Dropdown(
                    id="year-dd",
                    options=[{"label": str(y), "value": y} for y in YEARS_ALLOWED],
                    value=y0,
                    clearable=False
                ),
                html.Div(id="year-warning", className="rlo-warning")
            ], md=3),

            dbc.Col([
                dbc.Label("Grand Prix", className="dbc-label"),
                dcc.Dropdown(
                    id="event-dd",
                    options=build_gp_options(y0) if y0 else [],
                    value=default_event_value(y0) if y0 else None,
                    clearable=False
                ),
            ], md=6),

            dbc.Col([
                dbc.Label("Session", className="dbc-label"),
                dcc.Dropdown(
                    id="session-dd",
                    options=SESSION_OPTIONS,
                    value="R",
                    clearable=False
                )
            ], md=3),
        ], className="g-2"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Quick driver select", className="dbc-label"),
                dbc.ButtonGroup([
                    dbc.Button("Top 5", id="btn-top5", className="rlo-btn-pill rlo-select-btn", color="light"),
                    dbc.Button("Top 10", id="btn-top10", className="rlo-btn-pill rlo-select-btn", color="light"),
                    dbc.Button("All", id="btn-all", className="rlo-btn-pill rlo-select-btn", color="light"),
                ], className="rlo-select-group"),
                html.Div("Tip: toggle drivers by tapping their name in the legend.", className="rlo-muted"),
            ], md=12),
        ], className="g-2 mt-2"),
    ])

def echarts_box(chart_id: str, title: str, chart_key: str, height: int = 520):
    return html.Div(className="box rlo-card", children=[
        html.Div(className="rlo-card-title", children=title),
        dcc.Loading(
            dash_echarts.DashECharts(
                id=chart_id,
                option=option_empty(title),
                style={"height": f"{height}px", "width": "100%"},
            ),
            type="dot",
            color=COL_RED,
        ),
    ])


def polish_plotly(fig: go.Figure, grid: bool = False) -> go.Figure:
    fig.update_layout(**COMMON_LAYOUT)
    if not grid:
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout(uirevision="keep")
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=42, color="rgba(0,0,0,0.10)", family="Inter, system-ui, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.22
    )
    return fig

def tyres_box(height: int = 560):
    def empty_fig(msg: str):
        f = go.Figure()
        f.update_layout(title="Tyre Strategy", **COMMON_LAYOUT)
        f.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=15, color="rgba(0,0,0,0.55)"))
        return polish_plotly(f)
    return html.Div(className="box rlo-card", children=[
        html.Div(className="rlo-card-title", children="Tyre Strategy"),
        dcc.Loading(
            dcc.Graph(
                id="tyre_strategy_p",
                figure=empty_fig("Select a completed event + session to load data."),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": f"{height}px"}
            ),
            type="default",
            className="rlo-loading"
        )
    ])

def tab_evolution():
    return html.Div([
        dbc.Row([
            dbc.Col(echarts_box("gap_e", "Gap", "gap"), md=6),
            dbc.Col(echarts_box("lapchart_e", "Lapchart", "lc"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(echarts_box("evo_pace_e", "Evolution Pace", "ep"), md=6),
            dbc.Col(echarts_box("pos_e", "Positions Gained", "pos"), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_tyres():
    return html.Div([tyres_box(height=560)])

def tab_pace():
    return html.Div([echarts_box("pace_e", "Lap-by-Lap Pace", "pace", height=560)])

def tab_records():
    return html.Div([
        dbc.Row([
            dbc.Col(echarts_box("best_laps_e", "Best Laps", "best"), md=6),
            dbc.Col(echarts_box("sectors_e", "Sector Records", "sec"), md=6),
        ], className="g-2"),
    ])

def tab_speeds():
    return html.Div([echarts_box("speeds_e", "Speed Metrics", "spd", height=560)])

def footer_faq():
    faq_items = [
        ("Why is the page empty sometimes?",
         "Most commonly: you selected a future session or a season with no completed events yet. Switch to the latest completed event or use 2025."),
        ("Do drivers always share team colours correctly?",
         "Yes — colours are derived from the team field in the selected session, so mid‑season swaps are handled per event."),
        ("Why is the first load slower?",
         "FastF1 must download and parse timing for a session at least once. After caching, repeat loads are faster."),
    ]
    return html.Div(className="rlo-footer", children=[
        html.Div(className="box rlo-card", children=[
            html.Div(className="rlo-card-title", children="FAQ"),
            dbc.Accordion(
                [dbc.AccordionItem(html.P(a, className="rlo-faq-text"), title=q) for q, a in faq_items],
                start_collapsed=True,
                always_open=False,
                className="rlo-accordion"
            )
        ]),
        html.Div(className="rlo-footnote", children=[
            "Built with FastF1 + ECharts. Not affiliated with Formula 1.",
        ])
    ])

app.layout = dbc.Container([
    dcc.Location(id="url"),
    header_bar(),
    html.Div(className="rlo-page", children=[
        hero_strip(),
        filter_card(),

        dcc.Tabs(
            id="tabs",
            value="evo",
            parent_className="rlo-tabs-parent",
            className="rlo-tabs",
            children=[
                dcc.Tab(label="Evolution", value="evo", className="rlo-tab", selected_className="rlo-tab--selected"),
                dcc.Tab(label="Tyres", value="tyres", className="rlo-tab", selected_className="rlo-tab--selected"),
                dcc.Tab(label="Pace", value="pace", className="rlo-tab", selected_className="rlo-tab--selected"),
                dcc.Tab(label="Records", value="records", className="rlo-tab", selected_className="rlo-tab--selected"),
                dcc.Tab(label="Speeds", value="speeds", className="rlo-tab", selected_className="rlo-tab--selected"),
            ],
        ),
        html.Div(id="tab-body", className="mt-2", children=tab_evolution()),

        dcc.Store(id="store"),
        dcc.Store(id="driver-select-store", data={"drivers": []}),
        footer_faq(),
    ])
], fluid=True, className="rlo-root")

@app.callback(
    Output("tyre_strategy_p", "figure"),
    Input("store", "data"),
    Input("driver-select-store", "data"),
)
def chart_tyres(store, sel_store):
    sel = (sel_store or {}).get("drivers", [])

    f = go.Figure()
    f.update_layout(title="Tyre Strategy", **COMMON_LAYOUT)

    # No drivers selected by default -> keep empty with watermark
    if not store or len(sel) == 0:
        f.add_annotation(
            text="@redlightsoff5",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=46, color="rgba(192,0,26,0.18)"),
            textangle=-28,
        )
        return polish_plotly(f)

    p = get_processed(int(store["year"]), store["event"], store["sess"])
    st = _df_from_json(p.get("tyres", ""))
    if st.empty:
        f.add_annotation(
            text="@redlightsoff5",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=46, color="rgba(192,0,26,0.18)"),
            textangle=-28,
        )
        return polish_plotly(f)

    # Filter to selected drivers
    if "Driver" in st.columns:
        st = st[st["Driver"].astype(str).isin([str(d) for d in sel])]

    if st.empty:
        f.add_annotation(
            text="@redlightsoff5",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=46, color="rgba(192,0,26,0.18)"),
            textangle=-28,
        )
        return polish_plotly(f)

    cmap = {"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0F0", "INTERMEDIATE": "#43B02A", "WET": "#00A3E0"}

    order = st["Driver"].astype(str).unique().tolist()
    st = st.copy()
    st["Compound"] = st["Compound"].astype(str).str.upper()

    for i, drv in enumerate(order):
        sd = st[st["Driver"].astype(str) == drv].sort_values("StartLap")
        for _, r in sd.iterrows():
            comp = str(r["Compound"]).upper()
            col = cmap.get(comp, "#666666")
            f.add_trace(go.Bar(
                x=[r["StintLength"]],
                y=[drv],
                orientation="h",
                base=[r["StartLap"]],
                marker_color=col,
                showlegend=False,
                hovertemplate=f"{drv}<br>{comp}<br>Lap {int(r['StartLap'])} → {int(r['EndLap'])}<extra></extra>",
            ))

    # Legend for compounds
    for n, c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))

    f.update_layout(
        title="Tyre Strategy",
        barmode="stack",
        yaxis=dict(categoryorder="array", categoryarray=order, title="Driver"),
        xaxis_title="Lap",
    )

    # Watermark
    f.add_annotation(
        text="@redlightsoff5",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=46, color="rgba(192,0,26,0.18)"),
        textangle=-28,
    )

    return polish_plotly(f, grid=True)

