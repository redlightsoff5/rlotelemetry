import os, warnings, logging, traceback
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

from functools import lru_cache

import pandas as pd
import numpy as np
import fastf1 as ff1

import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc


# =========================
# Config
# =========================
SUPPORTED_YEARS = [2025, 2026]
WATERMARK = "@redlightsoff5"

APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

# "Support" button (optional): set this in Render as an env var to your BuyMeACoffee URL
BMC_URL = os.getenv("BMC_URL", "").strip()

# Light chart theme (page background is handled by assets/styles.css)
COL_PANEL = "#ffffff"
COL_TEXT = "#111111"

COMMON_LAYOUT = dict(
    template=None,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COL_PANEL,
    font=dict(color=COL_TEXT),
    margin=dict(l=28, r=18, t=44, b=28),
    legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

SESSION_OPTIONS = [
    {"label": "FP1", "value": "FP1"},
    {"label": "FP2", "value": "FP2"},
    {"label": "FP3", "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},  # 2024/2025 naming; 2023 fallback handled
    {"label": "Qualifying", "value": "Q"},
    {"label": "Sprint", "value": "SR"},
    {"label": "Race", "value": "R"},
]

# Team colors (canonical names) — adjust anytime
TEAM_COLORS = {
    "Red Bull":     "#3671C6",
    "Ferrari":      "#E8002D",
    "McLaren":      "#FF8000",
    "Mercedes":     "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine":       "#FF87BC",
    "Williams":     "#1868DB",
    "RB":           "#6692FF",
    "Stake":        "#01C00E",  # Sauber / Kick / Audi bucket
    "Haas":         "#9C9FA2",
}

TEAM_ALIASES = {
    "red bull": "Red Bull",
    "oracle red bull": "Red Bull",
    "red bull racing": "Red Bull",

    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",

    "mclaren": "McLaren",
    "mclaren f1 team": "McLaren",

    "mercedes": "Mercedes",
    "mercedes amg": "Mercedes",
    "mercedes-amg": "Mercedes",

    "aston martin": "Aston Martin",
    "aston martin aramco": "Aston Martin",

    "alpine": "Alpine",
    "bwt alpine": "Alpine",

    "williams": "Williams",
    "williams racing": "Williams",

    "rb f1": "RB",
    "racing bulls": "RB",
    "visa cash app rb": "RB",
    "rb": "RB",

    "sauber": "Stake",
    "kick sauber": "Stake",
    "stake": "Stake",
    "audi": "Stake",

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
    # title-case fallback (kept for unknown)
    return name.strip().title()


def brand(fig: go.Figure) -> go.Figure:
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=44, color="rgba(0,0,0,0.16)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.16
    )
    return fig


def fig_empty() -> go.Figure:
    f = go.Figure()
    f.update_layout(**COMMON_LAYOUT, showlegend=False)
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False)
    return brand(f)


def apply_visibility(fig: go.Figure, selected) -> go.Figure:
    selected_set = set(selected or [])
    for tr in fig.data:
        name = str(getattr(tr, "name", ""))
        if not selected_set:
            tr.visible = "legendonly"
        else:
            tr.visible = True if name in selected_set else "legendonly"
    return fig


def set_trace_color(fig: go.Figure, color_map: dict) -> go.Figure:
    """Force line/marker colors to driver team colors where possible."""
    if not color_map:
        return fig
    for tr in fig.data:
        drv = str(getattr(tr, "name", ""))
        col = color_map.get(drv)
        if not col:
            continue
        try:
            if hasattr(tr, "line"):
                tr.line.color = col
            if hasattr(tr, "marker"):
                tr.marker.color = col
        except Exception:
            pass
    return fig


def s_to_mssmmm(x: float) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    m = int(x // 60)
    s = int(x % 60)
    ms = int(round((x - int(x)) * 1000))
    return f"{m}:{s:02d}.{ms:03d}"


def is_race(session) -> bool:
    code = str(getattr(session, "name", "")).upper()
    return code in ("RACE", "SPRINT", "SPRINT RACE", "SPR")


def get_schedule_df(year: int) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=False).copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    df = df[["RoundNumber", "EventName", "EventDate"]].sort_values("RoundNumber").reset_index(drop=True)
    return df


def build_gp_options(year: int):
    df = get_schedule_df(year)
    return [
        {"label": f"R{int(r.RoundNumber)} — {r.EventName} ({r.EventDate.date()})", "value": r.EventName}
        for _, r in df.iterrows()
    ]


def default_event_value(year: int):
    df = get_schedule_df(year)
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df["EventDate"] <= today]
    if not past.empty:
        return past.iloc[-1]["EventName"]
    return df.iloc[0]["EventName"] if not df.empty else None


def has_past_event(year: int) -> bool:
    df = get_schedule_df(year)
    today = pd.Timestamp.utcnow().tz_localize(None)
    return not df[df["EventDate"] <= today].empty


def default_year_value() -> int:
    for y in sorted(SUPPORTED_YEARS, reverse=True):
        if has_past_event(y):
            return y
    return SUPPORTED_YEARS[0]


@lru_cache(maxsize=64)
def load_session(year: int, event_name: str, sess_code: str):
    """Load FastF1 session (laps only)."""
    try:
        ses = ff1.get_session(int(year), event_name, str(sess_code))
    except Exception:
        # Back-compat: 2023 used "SS" for Sprint Shootout
        if str(sess_code).upper() == "SQ":
            ses = ff1.get_session(int(year), event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses


def driver_team_color_map(ses) -> dict:
    laps = ses.laps[["Driver", "Team"]].dropna()
    if laps.empty:
        return {}
    team = (
        laps.groupby("Driver", dropna=False)["Team"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1])
        .apply(canonical_team)
    )
    return {drv: TEAM_COLORS.get(t, "#cccccc") for drv, t in team.items()}


# =========================
# Layout
# =========================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="RLO Telemetry"
)
server = app.server


def header_controls():
    logo_path = os.path.join(APP_DIR, "assets", "logo.png")
    has_logo = os.path.exists(logo_path)
    y0 = default_year_value()
    return html.Div(className="rlo-top", children=[
        dbc.Row([
            dbc.Col([
                html.Div(className="rlo-brand", children=[
                    (html.Img(src=app.get_asset_url("logo.png"), className="rlo-logo") if has_logo else None),
                    html.Div([
                        html.Div("RLO Telemetry", className="rlo-title"),
                        html.Div("Tap legend to show/hide drivers", className="rlo-subtitle"),
                    ])
                ])
            ], md=7),
            dbc.Col([
                dbc.Button("Support", className="rlo-support-btn", href=BMC_URL, target="_blank",
                           style={"display": "inline-block" if BMC_URL else "none"})
            ], md=5, className="text-md-end")
        ], align="center", className="mb-2"),

        html.Div(className="rlo-filter-card", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Label("Year", className="rlo-label"),
                    dcc.Dropdown(
                        id="year-dd",
                        options=[{"label": str(y), "value": y} for y in SUPPORTED_YEARS],
                        value=y0,
                        clearable=False
                    )
                ], md=2),

                dbc.Col([
                    dbc.Label("Grand Prix", className="rlo-label"),
                    dcc.Dropdown(
                        id="event-dd",
                        options=build_gp_options(y0),
                        value=default_event_value(y0),
                        clearable=False
                    )
                ], md=5),

                dbc.Col([
                    dbc.Label("Session", className="rlo-label"),
                    dcc.Dropdown(
                        id="session-dd",
                        options=SESSION_OPTIONS,
                        value="R",
                        clearable=False
                    )
                ], md=2),

                dbc.Col([
                    dbc.Label("Quick select", className="rlo-label"),
                    dbc.ButtonGroup([
                        dbc.Button("Top 5", id="btn-top5", className="rlo-pill-btn", n_clicks=0),
                        dbc.Button("Top 10", id="btn-top10", className="rlo-pill-btn", n_clicks=0),
                        dbc.Button("All", id="btn-all", className="rlo-pill-btn", n_clicks=0),
                    ], className="w-100")
                ], md=3),
            ], className="g-2")
        ]),
    ])


def graph_box(graph_id, title):
    return html.Div(className="box", children=[
        html.Div(className="box-head", children=[html.H6(title, className="mb-0")]),
        dcc.Graph(id=graph_id, config={"displaylogo": False, "responsive": True}),
    ])


def tab_evolution():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("gap", "Gap"), md=6),
            dbc.Col(graph_box("lapchart", "Lapchart"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box("evo", "Evolution Pace (MA3)"), md=6),
            dbc.Col(graph_box("pos", "Positions Gained"), md=6),
        ], className="g-2 mt-2"),
    ])


def tab_tyres():
    return dbc.Row([dbc.Col(graph_box("tyres", "Tyre Strategy"), md=12)], className="g-2")


def tab_pace():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box("pace", "Lap-by-Lap Pace"), md=7),
            dbc.Col(graph_box("best", "Best Laps"), md=5),
        ], className="g-2"),
    ])


def tab_records():
    return dbc.Row([dbc.Col(graph_box("sectors", "Sector Records"), md=12)], className="g-2")


def tab_speeds():
    return dbc.Row([dbc.Col(graph_box("speeds", "Speed Records"), md=12)], className="g-2")


app.layout = dbc.Container([
    header_controls(),
    dcc.Tabs(
        id="tabs",
        value="evo",
        children=[
            dcc.Tab(label="Evolution", value="evo"),
            dcc.Tab(label="Tyres", value="tyres"),
            dcc.Tab(label="Pace", value="pace"),
            dcc.Tab(label="Records", value="records"),
            dcc.Tab(label="Speeds", value="speeds"),
        ]
    ),
    html.Div(id="tab-body", className="mt-2", children=tab_evolution()),

    dcc.Store(id="store"),
    dcc.Store(id="ranked-drivers"),
    dcc.Store(id="selected-drivers", data=[]),
    dcc.Store(id="team-color-store"),
], fluid=True)


@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tabs(tab):
    return {
        "evo": tab_evolution(),
        "tyres": tab_tyres(),
        "pace": tab_pace(),
        "records": tab_records(),
        "speeds": tab_speeds(),
    }.get(tab, tab_evolution())


# =========================
# Meta callbacks
# =========================
@app.callback(
    Output("event-dd", "options"),
    Output("event-dd", "value"),
    Input("year-dd", "value"),
)
def update_event_dropdown(year):
    if not year:
        return [], None
    year = int(year)
    return build_gp_options(year), default_event_value(year)


@app.callback(
    Output("store", "data"),
    Output("ranked-drivers", "data"),
    Output("team-color-store", "data"),
    Output("selected-drivers", "data"),
    Input("year-dd", "value"),
    Input("event-dd", "value"),
    Input("session-dd", "value"),
)
def load_session_meta(year, event_name, sess_code):
    if not year or not event_name or not sess_code:
        return no_update, [], {}, []
    try:
        ses = load_session(int(year), str(event_name), str(sess_code))
        laps = ses.laps.dropna(subset=["LapTime"]).copy()

        if laps.empty:
            ranked = []
        else:
            best = laps.groupby("Driver", dropna=False)["LapTime"].min().dropna()
            ranked = best.sort_values().index.tolist()

        colors = driver_team_color_map(ses)
        # Reset selection every time the session changes
        return {"year": int(year), "event": str(event_name), "sess": str(sess_code)}, ranked, colors, []
    except Exception:
        traceback.print_exc()
        return no_update, [], {}, []


@app.callback(
    Output("selected-drivers", "data"),
    Input("btn-top5", "n_clicks"),
    Input("btn-top10", "n_clicks"),
    Input("btn-all", "n_clicks"),
    State("ranked-drivers", "data"),
    prevent_initial_call=True
)
def quick_select(n5, n10, na, ranked):
    ranked = ranked or []
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""
    if trig == "btn-top5":
        return ranked[:5]
    if trig == "btn-top10":
        return ranked[:10]
    if trig == "btn-all":
        return ranked
    return []


# =========================
# Charts
# =========================
@app.callback(
    Output("gap", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_gap(data, selected, color_map):
    if not data:
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps.dropna(subset=["LapTime"]).copy()
    if laps.empty:
        return fig_empty()

    if is_race(ses):
        laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
        laps["Cum"] = laps.groupby("Driver", dropna=False)["LapSeconds"].cumsum()
        lead = laps.groupby("LapNumber", dropna=False)["Cum"].min().rename("Lead").reset_index()
        d = laps.merge(lead, on="LapNumber", how="left")
        d["Gap_s"] = d["Cum"] - d["Lead"]
        f = px.line(d, x="LapNumber", y="Gap_s", color="Driver", title="Gap to Leader")
        f.update_yaxes(title="sec", tickformat=".3f")
    else:
        best = laps.groupby("Driver", dropna=False)["LapTime"].min().dropna()
        if best.empty:
            return fig_empty()
        base = best.min().total_seconds()
        df = (best.dt.total_seconds() - base).rename("Gap_s").reset_index()
        df["GapStr"] = df["Gap_s"].apply(s_to_mssmmm)
        f = px.bar(df.sort_values("Gap_s"), x="Driver", y="Gap_s", custom_data=["GapStr"], title="Gap to Session Best")
        f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="sec", tickformat=".3f")

    f = brand(f)
    f = set_trace_color(f, color_map or {})
    f = apply_visibility(f, selected)
    return f


@app.callback(
    Output("lapchart", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_lapchart(data, selected, color_map):
    if not data:
        return fig_empty()
    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps[["Driver", "LapNumber", "Position"]].dropna()
    if laps.empty:
        return fig_empty()
    f = px.line(laps, x="LapNumber", y="Position", color="Driver", title="Lapchart (lower = better)")
    f.update_yaxes(autorange="reversed", dtick=1)

    f = brand(f)
    f = set_trace_color(f, color_map or {})
    f = apply_visibility(f, selected)
    return f


@app.callback(
    Output("evo", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_evolution(data, selected, color_map):
    if not data:
        return fig_empty()
    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps[["Driver", "LapNumber", "LapTime"]].dropna(subset=["LapTime"]).copy()
    if laps.empty:
        return fig_empty()
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["MA3"] = laps.groupby("Driver", dropna=False)["LapSeconds"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    f = go.Figure()
    for drv, d in laps.groupby("Driver"):
        f.add_trace(go.Scatter(x=d["LapNumber"], y=d["MA3"], mode="lines", name=str(drv),
                               hovertemplate=f"{drv} — Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_layout(title="Evolution Pace (3-lap moving average)")
    f.update_yaxes(title="sec (MA3)", tickformat=".3f")

    f = brand(f)
    f = set_trace_color(f, color_map or {})
    f = apply_visibility(f, selected)
    return f


@app.callback(
    Output("pos", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_positions_gained(data, selected, color_map):
    if not data:
        return fig_empty()
    ses = load_session(data["year"], data["event"], data["sess"])

    # Race-only: use results grid vs finish
    try:
        res = ses.results.copy()
    except Exception:
        res = pd.DataFrame()

    if res is None or res.empty or ("GridPosition" not in res.columns) or ("Position" not in res.columns):
        return fig_empty()

    df = res[["Abbreviation", "DriverNumber", "GridPosition", "Position"]].copy()
    df = df.dropna(subset=["GridPosition", "Position"])
    df["PositionsGained"] = df["GridPosition"].astype(int) - df["Position"].astype(int)

    if selected:
        # selected are driver codes like VER; match by Abbreviation
        df = df[df["Abbreviation"].isin(selected)]

    if df.empty:
        return fig_empty()

    f = px.bar(df.sort_values("PositionsGained", ascending=False), x="Abbreviation", y="PositionsGained",
               text="PositionsGained", title="Positions Gained (Grid → Finish)")
    f.update_traces(marker_line_width=0)

    f = brand(f)
    return f


@app.callback(
    Output("tyres", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
)
def chart_tyres(data, selected):
    if not data:
        return fig_empty()
    if not (selected or []):
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps.copy()
    if laps.empty:
        return fig_empty()

    # Build stints from laps: group by (Driver, Stint)
    cols = ["Driver", "Stint", "Compound", "LapNumber"]
    keep = [c for c in cols if c in laps.columns]
    if len(keep) < 4:
        return fig_empty()

    st = laps[keep].dropna()
    st = st[st["Driver"].isin(selected)]

    if st.empty:
        return fig_empty()

    g = st.groupby(["Driver", "Stint", "Compound"], dropna=False)["LapNumber"].agg(["min", "max", "count"]).reset_index()
    g.rename(columns={"min": "LapStart", "max": "LapEnd", "count": "Laps"}, inplace=True)

    f = go.Figure()
    order = g["Driver"].unique().tolist()[::-1]
    cmap = {
        "SOFT": "#DA291C",
        "MEDIUM": "#FFD12E",
        "HARD": "#F0F0F0",
        "INTERMEDIATE": "#43B02A",
        "WET": "#00A3E0",
    }

    for _, r in g.iterrows():
        comp = str(r["Compound"]).upper()
        f.add_trace(go.Bar(
            x=[int(r["Laps"])],
            y=[r["Driver"]],
            base=[int(r["LapStart"]) - 1],
            orientation="h",
            marker_color=cmap.get(comp, "#888888"),
            showlegend=False,
            hovertemplate=f"{r['Driver']} — {comp}<br>Lap {int(r['LapStart'])}–{int(r['LapEnd'])}<extra></extra>"
        ))

    # Legend entries for compounds
    for n, c in cmap.items():
        f.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))

    f.update_layout(title="Tyre Strategy", barmode="stack",
                    yaxis=dict(categoryorder="array", categoryarray=order, title="Driver"),
                    xaxis_title="Lap")

    return brand(f)


@app.callback(
    Output("pace", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_pace(data, selected, color_map):
    if not data:
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps[["Driver", "LapNumber", "LapTime"]].dropna(subset=["LapTime"]).copy()
    if laps.empty:
        return fig_empty()

    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()

    f = go.Figure()
    for drv, d in laps.groupby("Driver"):
        f.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapSeconds"],
            mode="lines+markers", name=str(drv),
            hovertemplate=f"{drv} — Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"
        ))

    f.update_layout(title="Lap-by-Lap Pace")
    f.update_yaxes(title="sec", tickformat=".3f")

    f = brand(f)
    f = set_trace_color(f, color_map or {})
    f = apply_visibility(f, selected)
    return f


@app.callback(
    Output("best", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
    State("team-color-store", "data"),
)
def chart_best_laps(data, selected, color_map):
    if not data:
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps.dropna(subset=["LapTime"]).copy()
    if laps.empty:
        return fig_empty()

    best = laps.loc[laps.groupby("Driver")["LapTime"].idxmin()].copy()
    best["Best_s"] = best["LapTime"].dt.total_seconds()
    best["BestStr"] = best["Best_s"].apply(s_to_mssmmm)

    f = px.bar(best.sort_values("Best_s"), x="Driver", y="Best_s", custom_data=["BestStr"], title="Best Laps")
    f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
    f.update_yaxes(title="sec", tickformat=".3f")

    f = brand(f)
    f = set_trace_color(f, color_map or {})
    f = apply_visibility(f, selected)
    return f


@app.callback(
    Output("sectors", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
)
def chart_sectors(data, selected):
    if not data:
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps.copy()
    if laps.empty:
        return fig_empty()

    sector_cols = []
    for c in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if c in laps.columns:
            sector_cols.append(c)

    if not sector_cols:
        return fig_empty()

    rows = []
    for i, c in enumerate(sector_cols, start=1):
        s = laps.dropna(subset=[c])[["Driver", c]].copy()
        if selected:
            s = s[s["Driver"].isin(selected)]
        if s.empty:
            continue
        s["t"] = s[c].dt.total_seconds()
        best_row = s.loc[s["t"].idxmin()]
        rows.append({"Sector": f"S{i}", "Driver": best_row["Driver"], "Time (s)": float(best_row["t"])})

    if not rows:
        return fig_empty()

    df = pd.DataFrame(rows)

    f = go.Figure(data=[go.Table(
        header=dict(values=["Sector", "Driver", "Time (s)"],
                    fill_color="#ffffff", font=dict(color="#000000", size=12)),
        cells=dict(values=[df["Sector"], df["Driver"], df["Time (s)"].round(3)],
                   fill_color="#ffffff", font=dict(color="#000000"))
    )])

    f.update_layout(title="Sector Records", paper_bgcolor="rgba(0,0,0,0)")
    return brand(f)


@app.callback(
    Output("speeds", "figure"),
    Input("store", "data"),
    Input("selected-drivers", "data"),
)
def chart_speeds(data, selected):
    if not data:
        return fig_empty()

    ses = load_session(data["year"], data["event"], data["sess"])
    laps = ses.laps.copy()
    if laps.empty:
        return fig_empty()

    speed_cols = [c for c in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"] if c in laps.columns]
    if not speed_cols:
        return fig_empty()

    spd = laps.dropna(subset=speed_cols)[["Driver"] + speed_cols].copy()
    if spd.empty:
        return fig_empty()

    spd = spd.groupby("Driver", dropna=False)[speed_cols].max().reset_index()

    if selected:
        spd = spd[spd["Driver"].isin(selected)]
    if spd.empty:
        return fig_empty()

    dm = spd.melt(id_vars="Driver", var_name="Metric", value_name="km/h")
    f = px.bar(dm, x="Driver", y="km/h", color="Metric", barmode="group", title="Speed Records")
    f.update_traces(marker_line_width=0)
    f.update_layout(yaxis_title="km/h")

    return brand(f)


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
