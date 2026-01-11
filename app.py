
import os, warnings, logging, traceback
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, no_update, ctx, MATCH
import dash_bootstrap_components as dbc

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

YEARS_ALLOWED = [2025, 2026]
SITE_TITLE = "RLO Telemetry"
WATERMARK  = "@redlightsoff5"

# UI palette (matches assets/styles.css)
COL_BG    = "#9f9797"
COL_CARD  = "#ffffff"
COL_TEXT  = "#111111"
COL_RED   = "#c0001a"
COL_GRID  = "rgba(0,0,0,0.10)"

# Team colors (same for both drivers of a team; swaps handled by session Team string)
TEAM_COLORS = {
    'Red Bull':      '#3671C6',
    'McLaren':       '#FF8000',
    'Ferrari':       '#E80020',
    'Mercedes':      '#27F4D2',
    'Aston Martin':  '#229971',
    'Alpine':        '#0093CC',
    'Williams':      '#64C4FF',
    'Racing Bulls':  '#6692FF',
    'Sauber':        '#52E252',
    'Haas':          '#B6BABD'
}
TEAM_ALIASES = {
    'red bull': 'Red Bull',
    'oracle red bull': 'Red Bull',
    'red bull racing': 'Red Bull',

    'racing bulls': 'Racing Bulls',
    'visa cash app rb': 'Racing Bulls',
    'vcarb': 'Racing Bulls',
    'rb f1': 'Racing Bulls',
    'rb': 'Racing Bulls',

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

    'sauber': 'Sauber',
    'kick sauber': 'Sauber',
    'stake': 'Sauber',
    'stake f1': 'Sauber',
    'audi': 'Sauber',

    'haas': 'Haas',
    'haas f1': 'Haas',
}

def canonical_team(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    for key, canon in TEAM_ALIASES.items():
        if key in s:
            return canon
    return name

def _utc_today_token() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

# ---------- Schedule helpers ----------
@lru_cache(maxsize=16)
def get_schedule_df(year: int, token: str) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=False).copy()
    df['EventDate'] = pd.to_datetime(df['EventDate'])
    df = df[['RoundNumber','EventName','EventDate']].sort_values('RoundNumber').reset_index(drop=True)
    return df

def build_gp_options(year: int):
    df = get_schedule_df(int(year), _utc_today_token())
    return [{'label': f"R{int(r.RoundNumber)} — {r.EventName} ({r.EventDate.date()})", 'value': r.EventName}
            for _, r in df.iterrows()]

def default_event_value(year: int):
    df = get_schedule_df(int(year), _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df['EventDate'] <= today]
    if not past.empty:
        return past.iloc[-1]['EventName']
    return None

def default_year_value():
    today = pd.Timestamp.utcnow().tz_localize(None)
    for y in sorted(YEARS_ALLOWED, reverse=True):
        try:
            df = get_schedule_df(int(y), _utc_today_token())
            if not df.empty and (df['EventDate'] <= today).any():
                return int(y)
        except Exception:
            continue
    return int(YEARS_ALLOWED[0])

SESSION_OPTIONS = [
    {"label": "FP1", "value": "FP1"},
    {"label": "FP2", "value": "FP2"},
    {"label": "FP3", "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},  # modern naming
    {"label": "Qualifying", "value": "Q"},
    {"label": "Sprint", "value": "SR"},
    {"label": "Race", "value": "R"},
]

# ---------- Load session (cached) ----------
@lru_cache(maxsize=64)
def load_session_laps(year: int, event_name: str, sess_code: str):
    y = int(year)
    sess_code = str(sess_code)
    event_name = str(event_name)
    try:
        ses = ff1.get_session(y, event_name, sess_code)
    except Exception:
        # Back-compat: some years used "SS" for Sprint Shootout
        if sess_code.upper() == "SQ":
            ses = ff1.get_session(y, event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

# ---------- Small utilities ----------
def s_to_mssmmm(x):
    if pd.isna(x): return ""
    x = float(x)
    m = int(x // 60); s = int(x % 60); ms = int(round((x - int(x))*1000))
    return f"{m}:{s:02d}.{ms:03d}"

def is_race(ses) -> bool:
    t = (getattr(ses, 'session_type', '') or '').upper()
    n = (getattr(ses, 'name', '') or '').upper()
    return t == 'R' or 'RACE' in n

def driver_team_color_map(ses) -> dict:
    laps = ses.laps[['Driver','Team']].dropna()
    if laps.empty:
        return {}
    team = laps.groupby('Driver')['Team'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
    ).apply(canonical_team)
    return {drv: TEAM_COLORS.get(t, '#cccccc') for drv, t in team.items()}

def top_drivers_from_session(ses, n: int) -> list:
    laps = ses.laps.dropna(subset=['LapTime'])
    if laps.empty:
        return []
    best = laps.groupby('Driver')['LapTime'].min().sort_values()
    return best.head(int(n)).index.tolist()

# ---------- Plotly styling ----------
COMMON_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COL_TEXT, family="Inter, system-ui, -apple-system, Segoe UI, Arial"),
    margin=dict(l=16, r=16, t=52, b=16),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)")
)

def apply_brand(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, **COMMON_LAYOUT, uirevision="keep")
    fig.update_xaxes(showgrid=True, gridcolor=COL_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COL_GRID, zeroline=False)
    # watermark
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=44, color="rgba(0,0,0,0.10)"),
        xanchor="center", yanchor="middle",
    )
    return fig

def empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(showlegend=False)
    return apply_brand(fig, title)

def set_trace_colors(fig: go.Figure, color_map: dict) -> go.Figure:
    if not color_map:
        return fig
    for tr in fig.data:
        name = getattr(tr, "name", None)
        c = color_map.get(name)
        if c:
            try:
                tr.update(line=dict(color=c))
            except Exception:
                pass
            try:
                tr.update(marker=dict(color=c))
            except Exception:
                pass
    return fig

def visible_for_driver(drv: str, selected: list) -> str | bool:
    if not selected:
        return "legendonly"
    return True if drv in selected else "legendonly"

# ---------- Data builders ----------
def gap_df(ses):
    laps = ses.laps.copy().dropna(subset=['LapTime'])
    if laps.empty:
        return pd.DataFrame()
    if is_race(ses):
        laps['LapSeconds'] = laps['LapTime'].dt.total_seconds()
        laps['Cum'] = laps.groupby('Driver', dropna=False)['LapSeconds'].cumsum()
        lead = laps.groupby('LapNumber', dropna=False)['Cum'].min().rename('Lead').reset_index()
        d = laps.merge(lead, on='LapNumber', how='left')
        d['Gap_s'] = d['Cum'] - d['Lead']
        return d[['Driver','LapNumber','Gap_s']]
    best = laps.groupby('Driver', dropna=False)['LapTime'].min().rename('Best').reset_index()
    gbest = best['Best'].min()
    best['Gap_s'] = (best['Best'] - gbest).dt.total_seconds()
    best['LapNumber'] = 1
    return best[['Driver','LapNumber','Gap_s']]

def pace_df(ses):
    laps = ses.laps.copy().dropna(subset=['LapTime'])
    if laps.empty:
        return pd.DataFrame()
    laps['LapSeconds'] = laps['LapTime'].dt.total_seconds().astype(float)
    return laps[['Driver','LapNumber','LapSeconds']]

def positions_df(ses):
    res = ses.results
    if res is None or res.empty or not {'Abbreviation','GridPosition','Position'}.issubset(res.columns):
        return pd.DataFrame()
    df = res[['Abbreviation','GridPosition','Position']].copy()
    df['Driver'] = df['Abbreviation']
    df['PositionsGained'] = df['GridPosition'] - df['Position']
    return df.sort_values('PositionsGained', ascending=False)

def tyre_stints_df(ses):
    laps = ses.laps.copy()
    if laps.empty or 'Compound' not in laps.columns:
        return pd.DataFrame()
    laps['Compound'] = laps['Compound'].astype(str).str.upper()
    agg = (laps.groupby(['Driver','Stint','Compound'], dropna=False)
               .agg(LapStart=('LapNumber','min'), LapEnd=('LapNumber','max')))
    agg['Laps'] = agg['LapEnd'] - agg['LapStart'] + 1
    return agg.reset_index().sort_values(['Driver','Stint'])

def best_laps_df(ses):
    laps = ses.laps.dropna(subset=['LapTime'])
    if laps.empty:
        return pd.DataFrame()
    best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
    best['Best_s'] = best['LapTime'].dt.total_seconds()
    return best[['Driver','Best_s']].sort_values('Best_s')

def sector_best_df(ses):
    laps = ses.laps.copy()
    out = []
    for i, c in enumerate(['Sector1Time','Sector2Time','Sector3Time'], start=1):
        if c not in laps.columns:
            continue
        idx = laps[c].idxmin()
        if pd.isna(idx):
            continue
        row = laps.loc[idx]
        out.append({'Sector': f'S{i}', 'Driver': row['Driver'], 'Time_s': row[c].total_seconds()})
    return pd.DataFrame(out)

def speed_df(ses):
    laps = ses.laps.copy()
    cols = [c for c in ['SpeedI1','SpeedI2','SpeedFL','SpeedST'] if c in laps.columns]
    if not cols:
        return pd.DataFrame()
    grp = laps.groupby('Driver')[cols].max().reset_index()
    dm = grp.melt(id_vars='Driver', var_name='Metric', value_name='kmh')
    metric_map = {'SpeedI1':'I1', 'SpeedI2':'I2', 'SpeedFL':'Finish', 'SpeedST':'Trap'}
    dm['Metric'] = dm['Metric'].map(metric_map).fillna(dm['Metric'])
    return dm

# ================= Dash app =================
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE
server = app.server

CHARTS = [
    # key, graph_id, title, tab
    ("gap", "gap", "Gap to Leader", "evo"),
    ("lapchart", "lapchart", "Lapchart (pace)", "evo"),
    ("evo_pace", "evo-pace", "Evolution Pace (rolling)", "evo"),
    ("pos", "pos", "Positions Gained", "evo"),
    ("ty", "tyre-strategy", "Tyre Strategy", "tyres"),
    ("pace", "pace", "Lap-by-Lap Pace", "pace"),
    ("best", "best-laps", "Best Laps", "records"),
    ("sec", "sectors", "Sector Records", "records"),
    ("spd", "speeds", "Speed Metrics", "speeds"),
]
CHART_KEYS = [k for k,_,_,_ in CHARTS]

def quick_buttons(chart_key: str):
    return dbc.ButtonGroup(
        [
            dbc.Button("Top 5", id={"role":"q5","chart":chart_key}, className="rlo-quick-btn", n_clicks=0),
            dbc.Button("Top 10", id={"role":"q10","chart":chart_key}, className="rlo-quick-btn", n_clicks=0),
            dbc.Button("All", id={"role":"qall","chart":chart_key}, className="rlo-quick-btn", n_clicks=0),
            dbc.Button("Clear", id={"role":"qclr","chart":chart_key}, className="rlo-quick-btn", n_clicks=0),
        ],
        className="rlo-quick-group",
        size="sm"
    )

def graph_card(graph_id: str, title: str, chart_key: str, height: int = 420):
    return html.Div(
        className="box graph-card",
        children=[
            dbc.Row(
                [
                    dbc.Col(html.Div(title, className="graph-title"), md=6),
                    dbc.Col(quick_buttons(chart_key), md=6, style={"textAlign":"right"})
                ],
                className="g-2 align-items-center mb-1"
            ),
            dcc.Loading(
                dcc.Graph(
                    id=graph_id,
                    figure=empty_fig(title),
                    config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
                    style={"height": f"{height}px"}
                ),
                type="default"
            )
        ]
    )

def header_bar():
    logo = html.Img(src="/assets/logo.png", className="rlo-logo", alt="logo")
    return html.Div(className="rlo-navbar", children=[
        html.Div([logo, html.Div([html.Div("RLO Telemetry", className="rlo-brand"),
                                  html.Div("Select event/session, then use Top 5 / Top 10 / All or click legend.", className="rlo-sub")],
                                 className="rlo-brand-wrap")],
                 className="rlo-left"),
    ])

def controls_card():
    y0 = default_year_value()
    return html.Div(className="box controls-card", children=[
        dbc.Row([
            dbc.Col([
                dbc.Label("Year", className="dbc-label"),
                dcc.Dropdown(
                    id="year-dd",
                    options=[{"label": str(y), "value": y} for y in YEARS_ALLOWED],
                    value=y0,
                    clearable=False
                )
            ], md=3),
            dbc.Col([
                dbc.Label("Grand Prix", className="dbc-label"),
                dcc.Dropdown(
                    id="event-dd",
                    options=build_gp_options(y0),
                    value=default_event_value(y0),
                    clearable=False
                )
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
        ], className="g-2")
    ])

def tab_evolution():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_card("gap", "Gap to Leader", "gap"), md=6),
            dbc.Col(graph_card("lapchart", "Lapchart (pace)", "lapchart"), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_card("evo-pace", "Evolution Pace (rolling)", "evo_pace"), md=6),
            dbc.Col(graph_card("pos", "Positions Gained", "pos", height=360), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_tyres():
    return html.Div([graph_card("tyre-strategy", "Tyre Strategy", "ty", height=480)])

def tab_pace():
    return html.Div([graph_card("pace", "Lap-by-Lap Pace", "pace", height=520)])

def tab_records():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_card("best-laps", "Best Laps", "best", height=420), md=6),
            dbc.Col(graph_card("sectors", "Sector Records", "sec", height=420), md=6),
        ], className="g-2"),
    ])

def tab_speeds():
    return html.Div([graph_card("speeds", "Speed Metrics", "spd", height=520)])

app.layout = dbc.Container(
    [
        header_bar(),
        html.Div(className="rlo-page", children=[
            controls_card(),
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
        ]),
        dcc.Store(id="session-store"),
        dcc.Store(id="drivers-store"),
        dcc.Store(id="colors-store"),
        # selection per chart (persist across tabs)
        *[dcc.Store(id={"role":"sel", "chart":k}, data=[]) for k in CHART_KEYS],
    ],
    fluid=True
)

@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def _render_tabs(tab):
    return {
        "evo": tab_evolution,
        "tyres": tab_tyres,
        "pace": tab_pace,
        "records": tab_records,
        "speeds": tab_speeds
    }.get(tab, tab_evolution)()

# Update GP list when year changes (and guard for future-only)
@app.callback(
    Output("event-dd", "options"),
    Output("event-dd", "value"),
    Input("year-dd", "value"),
    State("event-dd", "value"),
)
def _year_changed(year, current_event):
    if year is None:
        return [], None
    opts = build_gp_options(int(year))
    valid = {o["value"] for o in opts}
    default_ev = default_event_value(int(year))
    value = current_event if current_event in valid else default_ev
    return opts, value

# Load session meta once
@app.callback(
    Output("session-store","data"),
    Output("drivers-store","data"),
    Output("colors-store","data"),
    Input("year-dd","value"),
    Input("event-dd","value"),
    Input("session-dd","value"),
)
def _load_meta(year, event_name, sess_code):
    if not year or not event_name or not sess_code:
        return no_update, [], {}
    try:
        ses = load_session_laps(int(year), str(event_name), str(sess_code))
        laps = ses.laps.dropna(subset=["LapTime"])
        drivers = sorted(laps["Driver"].dropna().unique().tolist())
        colors = driver_team_color_map(ses)
        return {"year": int(year), "event": str(event_name), "sess": str(sess_code)}, drivers, colors
    except Exception:
        traceback.print_exc()
        return no_update, [], {}

# The real MATCH callback (Dash requires concrete dict IDs; define separately)
@app.callback(
    Output({"role":"sel", "chart":MATCH}, "data"),
    Input({"role":"q5","chart":MATCH}, "n_clicks"),
    Input({"role":"q10","chart":MATCH}, "n_clicks"),
    Input({"role":"qall","chart":MATCH}, "n_clicks"),
    Input({"role":"qclr","chart":MATCH}, "n_clicks"),
    State("session-store","data"),
    prevent_initial_call=True,
)
def quick_select(c5, c10, call, cclr, sess_meta):
    if not sess_meta:
        return []
    trig = ctx.triggered_id
    year = sess_meta["year"]; event = sess_meta["event"]; sess = sess_meta["sess"]
    try:
        ses = load_session_laps(int(year), str(event), str(sess))
    except Exception:
        return []
    if isinstance(trig, dict) and trig.get("role") == "qclr":
        return []
    if isinstance(trig, dict) and trig.get("role") == "q5":
        return top_drivers_from_session(ses, 5)
    if isinstance(trig, dict) and trig.get("role") == "q10":
        return top_drivers_from_session(ses, 10)
    if isinstance(trig, dict) and trig.get("role") == "qall":
        laps = ses.laps.dropna(subset=["LapTime"])
        return sorted(laps["Driver"].dropna().unique().tolist())
    return no_update

# ---------- Figures ----------
@app.callback(
    Output("gap", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"gap"}, "data"),
    State("colors-store","data"),
)
def fig_gap(sess_meta, selected, colors):
    title = "Gap to Leader"
    if not sess_meta:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = gap_df(ses)
    if df.empty:
        return empty_fig(title)
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["Gap_s"], mode="lines",
            name=str(drv),
            visible=visible_for_driver(str(drv), selected),
            hovertemplate=f"{drv}<br>Lap %{{x}}<br>Gap %{{y:.3f}}s<extra></extra>"
        ))
    fig.update_yaxes(title="Gap (s)", tickformat=".3f")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("lapchart", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"lapchart"}, "data"),
    State("colors-store","data"),
)
def fig_lapchart(sess_meta, selected, colors):
    title = "Lapchart (pace)"
    if not sess_meta:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = pace_df(ses)
    if df.empty:
        return empty_fig(title)
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapSeconds"], mode="lines",
            name=str(drv),
            visible=visible_for_driver(str(drv), selected),
            hovertemplate=f"{drv}<br>Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"
        ))
    fig.update_yaxes(title="Lap time (s)", tickformat=".3f", autorange="reversed")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("evo-pace", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"evo_pace"}, "data"),
    State("colors-store","data"),
)
def fig_evo_pace(sess_meta, selected, colors):
    title = "Evolution Pace (rolling)"
    if not sess_meta:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = pace_df(ses)
    if df.empty:
        return empty_fig(title)
    # rolling median to smooth
    df = df.sort_values(["Driver","LapNumber"])
    df["Roll"] = df.groupby("Driver")["LapSeconds"].transform(lambda s: s.rolling(5, min_periods=1).median())
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["Roll"], mode="lines",
            name=str(drv),
            visible=visible_for_driver(str(drv), selected),
            hovertemplate=f"{drv}<br>Lap %{{x}}<br>%{{y:.3f}}s (roll)<extra></extra>"
        ))
    fig.update_yaxes(title="Rolling lap time (s)", tickformat=".3f", autorange="reversed")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("pos", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"pos"}, "data"),
    State("colors-store","data"),
)
def fig_pos(sess_meta, selected, colors):
    title = "Positions Gained"
    if not sess_meta:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = positions_df(ses)
    if df.empty:
        return empty_fig(title)
    # One trace per driver so legend-click works
    fig = go.Figure()
    for _, r in df.iterrows():
        drv = str(r["Driver"])
        fig.add_trace(go.Bar(
            x=[drv], y=[r["PositionsGained"]],
            name=drv,
            visible=visible_for_driver(drv, selected),
            hovertemplate=f"{drv}<br>Gained {r['PositionsGained']}<extra></extra>"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Positions", zeroline=True)
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("tyre-strategy", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"ty"}, "data"),
)
def fig_tyres(sess_meta, selected):
    title = "Tyre Strategy"
    if not sess_meta:
        return empty_fig(title)
    if not selected:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    st = tyre_stints_df(ses)
    if st.empty:
        return empty_fig(title)
    st = st[st["Driver"].isin(selected)]
    if st.empty:
        return empty_fig(title)

    cmap = {'SOFT':'#DA291C','MEDIUM':'#FFD12E','HARD':'#F0F0F0','INTERMEDIATE':'#43B02A','WET':'#00A3E0'}
    fig = go.Figure()
    order = list(reversed([d for d in selected if d in st["Driver"].unique().tolist()]))
    for _, r in st.iterrows():
        fig.add_trace(go.Bar(
            x=[int(r["Laps"])], y=[r["Driver"]], base=[int(r["LapStart"])-1],
            orientation="h",
            marker_color=cmap.get(str(r["Compound"]).upper(), "#888"),
            showlegend=False,
            hovertemplate=f"{r['Driver']} — {r['Compound']}<br>Lap {int(r['LapStart'])}–{int(r['LapEnd'])}<extra></extra>"
        ))
    # legend for compounds
    for n,c in cmap.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=n, showlegend=True))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(categoryorder="array", categoryarray=order, title="Driver")
    fig.update_xaxes(title="Lap")
    return apply_brand(fig, title)

@app.callback(
    Output("pace", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"pace"}, "data"),
    State("colors-store","data"),
)
def fig_pace(sess_meta, selected, colors):
    title = "Lap-by-Lap Pace"
    if not sess_meta:
        return empty_fig(title)
    if not selected:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = pace_df(ses)
    df = df[df["Driver"].isin(selected)]
    if df.empty:
        return empty_fig(title)
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Scatter(
            x=d["LapNumber"], y=d["LapSeconds"], mode="lines+markers",
            name=str(drv),
            hovertemplate=f"{drv}<br>Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"
        ))
    fig.update_yaxes(title="Lap time (s)", tickformat=".3f", autorange="reversed")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("best-laps", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"best"}, "data"),
    State("colors-store","data"),
)
def fig_best(sess_meta, selected, colors):
    title = "Best Laps"
    if not sess_meta:
        return empty_fig(title)
    if not selected:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = best_laps_df(ses)
    df = df[df["Driver"].isin(selected)]
    if df.empty:
        return empty_fig(title)
    fig = go.Figure()
    for _, r in df.iterrows():
        drv = str(r["Driver"])
        fig.add_trace(go.Bar(
            x=[drv], y=[r["Best_s"]],
            name=drv,
            visible=True,
            hovertemplate=f"{drv}<br>%{{y:.3f}}s<extra></extra>"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Best lap (s)", tickformat=".3f")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("sectors", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"sec"}, "data"),
    State("colors-store","data"),
)
def fig_sectors(sess_meta, selected, colors):
    title = "Sector Records"
    if not sess_meta:
        return empty_fig(title)
    if not selected:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = sector_best_df(ses)
    df = df[df["Driver"].isin(selected)]
    if df.empty:
        return empty_fig(title)
    # bar per sector with per-driver traces (legend toggles drivers)
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Bar(
            x=d["Sector"], y=d["Time_s"],
            name=str(drv),
            visible=True,
            hovertemplate=f"{drv}<br>%{{x}}: %{{y:.3f}}s<extra></extra>"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Time (s)", tickformat=".3f", autorange=False)
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

@app.callback(
    Output("speeds", "figure"),
    Input("session-store","data"),
    Input({"role":"sel","chart":"spd"}, "data"),
    State("colors-store","data"),
)
def fig_speeds(sess_meta, selected, colors):
    title = "Speed Metrics"
    if not sess_meta:
        return empty_fig(title)
    if not selected:
        return empty_fig(title)
    ses = load_session_laps(sess_meta["year"], sess_meta["event"], sess_meta["sess"])
    df = speed_df(ses)
    df = df[df["Driver"].isin(selected)]
    if df.empty:
        return empty_fig(title)
    fig = go.Figure()
    for drv, d in df.groupby("Driver"):
        fig.add_trace(go.Bar(
            x=d["Metric"], y=d["kmh"],
            name=str(drv),
            hovertemplate=f"{drv}<br>%{{x}}: %{{y:.0f}} km/h<extra></extra>"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="km/h")
    fig = apply_brand(fig, title)
    return set_trace_colors(fig, colors)

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
