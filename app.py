import os, warnings, logging, traceback, glob
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'plotly_white'

from dash import Dash, dcc, html, Input, Output, State, no_update, ALL, MATCH
import dash_bootstrap_components as dbc

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

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

COL_BG    = "#9f9797"
COL_PANEL = "#ffffff"
COL_RED   = "#e11d2e"
COL_TEXT  = "#000000"

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
    font=dict(color=COL_TEXT),
    margin=dict(l=12, r=12, t=48, b=12)
)

def brand(fig):
    """Apply common styling + centered watermark to a Plotly figure."""
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_annotation(
        text=WATERMARK,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=42, color="rgba(255,255,255,0.16)", family="Montserrat, Arial"),
        xanchor="center", yanchor="middle",
        opacity=0.25
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

def build_gp_options(year: int):
    df = get_schedule_df(year, _utc_today_token())

    testing_df = df[df["EventFormat"] == "testing"].sort_values("EventDate")

    if not testing_df.empty:
        g = (
            testing_df.groupby("EventName", dropna=False)["EventDate"]
            .agg(["min", "max"])
            .reset_index()
            .sort_values("min")
            .reset_index(drop=True)
        )
        test_number_map = {row["EventName"]: int(i + 1) for i, row in g.iterrows()}
    else:
        g = pd.DataFrame(columns=["EventName", "min", "max"])
        test_number_map = {}

    opts = []
    added_tests = set()

    for _, r in df.sort_values(["EventDate", "RoundNumber"]).iterrows():
        fmt = str(r.EventFormat).lower()
        name = str(r.EventName)

        if fmt == "testing":
            tn = test_number_map.get(name, 1)
            val = f"TEST|{tn}"
            if val in added_tests:
                continue

            if not g.empty and (g["EventName"] == name).any():
                row = g[g["EventName"] == name].iloc[0]
                d0 = row["min"].date()
                d1 = row["max"].date()
                label = f"Pre-Season Testing #{tn} ({d0}–{d1})"
            else:
                label = f"Pre-Season Testing #{tn}"

            opts.append({"label": label, "value": val})
            added_tests.add(val)
        else:
            date = r.EventDate.date()
            opts.append(
                {
                    "label": f"R{int(r.RoundNumber)} — {name} ({date})",
                    "value": f"GP|{name}",
                }
            )

    return opts

def default_event_value(year: int):
    df = get_schedule_df(year, _utc_today_token())
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df["EventDate"] <= today].sort_values("EventDate")
    if past.empty:
        return None

    last = past.iloc[-1]
    if str(last["EventFormat"]).lower() == "testing":
        testing_df = df[df["EventFormat"] == "testing"].sort_values("EventDate")
        g = (
            testing_df.groupby("EventName", dropna=False)["EventDate"]
            .min()
            .sort_values()
            .reset_index(drop=False)
            .reset_index()
            .rename(columns={"index": "test_number"})
        )
        last_test_name = str(last["EventName"])
        tn_row = g[g["EventName"] == last_test_name]
        tn = int(tn_row["test_number"].iloc[0]) + 1 if not tn_row.empty else 1
        return f"TEST|{tn}"

    return f"GP|{str(last['EventName'])}"

SESSION_OPTIONS = [
    {"label": "FP1",               "value": "FP1"},
    {"label": "FP2",               "value": "FP2"},
    {"label": "FP3",               "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},
    {"label": "Qualifying",        "value": "Q"},
    {"label": "Sprint",            "value": "SR"},
    {"label": "Race",              "value": "R"},
]

TEST_SESSION_OPTIONS = [
    {"label": "Day 1", "value": "T1"},
    {"label": "Day 2", "value": "T2"},
    {"label": "Day 3", "value": "T3"},
]

# ---------- Loaders ----------
@lru_cache(maxsize=64)
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

    # Telemetry charts (only if you add that telemetry tab later)
    # Example keys from earlier: tel_spd, tel_ctl, tel_gr, tel_map
    if chart_key in {"tel_spd", "tel_ctl", "tel_gr", "tel_map"}:
        # you will need your telemetry loader + fastest lap extractor
        return pd.DataFrame()

    return pd.DataFrame()

# ================= Dash =================
external_stylesheets=[dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE  # browser tab title

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
    y0 = default_year_value()
    return dbc.Row([
        dbc.Col([
            dbc.Label("Year"),
            dcc.Dropdown(
                id='year-dd',
                options=[{'label': str(y), 'value': y} for y in YEARS_ALLOWED],
                value=y0,
                clearable=False
            ),
            html.Div(id='year-warning', className="mt-1", style={'fontSize':'0.85rem','opacity':0.85})
        ], md=3),

        dbc.Col([
            dbc.Label("Grand Prix"),
            dcc.Dropdown(
                id='event-dd',
                options=build_gp_options(y0),
                value=default_event_value(y0),
                clearable=False,
                placeholder="Select event..."
            )
        ], md=6),

        dbc.Col([
            dbc.Label("Session"),
            dcc.Dropdown(
                id='session-dd',
                options=SESSION_OPTIONS,
                value='R',
                clearable=False
            )
        ], md=3),
    ], className="g-2")

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
                    placeholder="Filter drivers (optional)",
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
            dbc.Col(graph_box('gap','Gap','gap'), md=6),
            dbc.Col(graph_box('lapchart','Lapchart','lc'), md=6),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(graph_box('evo-pace','Evolution Pace','ep'), md=6),
            dbc.Col(graph_box('pos','Positions Gained','pos'), md=6),
        ], className="g-2 mt-1"),
    ])

def tab_tyres():
    return html.Div([ graph_box('tyre-strategy','Tyre Strategy','ty') ])

def tab_pace():
    return html.Div([ graph_box('pace','Pace','pace') ])

def tab_records():
    return html.Div([
        dbc.Row([
            dbc.Col(graph_box('best-laps','Best Laps','best'), md=6),
            dbc.Col(graph_box('sectors','Sector Records','sec'), md=6),
        ], className="g-2")
    ])

def tab_speeds():
    return html.Div([ graph_box('speeds','Speed Metrics','spd') ])

app.layout = dbc.Container([
    header_controls(),
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
    dcc.Store(id='store'),
    dcc.Store(id='drivers-store'),
    dcc.Store(id='team-color-store')
], fluid=True, className='rlo-page')

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"evo":tab_evolution, "tyres":tab_tyres, "pace":tab_pace,
            "records":tab_records, "speeds":tab_speeds}.get(val, tab_evolution)()

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
        try:
            laps = ses.laps.dropna(subset=["LapTime"])
        except ff1.exceptions.DataNotLoadedError:
            # one forced reload in case FastF1 returned an unloaded session object
            try:
                ses.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = ses.laps.dropna(subset=["LapTime"])
            except Exception:
                raise
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
    opts = [{'label': d, 'value': d} for d in (drivers or [])]
    val = drivers or []
    n = len(ids)
    return [opts]*n, [val]*n

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
            warn = f"No completed events yet for {year}. Select 2025 to view data."
            value = None
        return opts, value, warn
    except Exception:
        return [], None, f"Schedule unavailable for {year}."

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
    if df.empty: return fig_empty("Gap — no data")
    if is_race(ses):
        f = px.line(df, x='LapNumber', y='Gap_s', color='Driver', custom_data=['GapStr'], title='Gap to Leader (MM:SS.mmm)')
        f.update_traces(hovertemplate="%{fullData.name} — Lap %{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="sec", tickformat=".3f")
    else:
        gg = df.sort_values('Gap_s')
        f = px.bar(gg, x='Driver', y='Gap_s', custom_data=['GapStr'], title='Gap to Session Best (MM:SS.mmm)')
        f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
        f.update_yaxes(title="sec", tickformat=".3f")
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
    if laps.empty: return fig_empty("Lapchart — no data")
    f = px.line(laps, x='LapNumber', y='Position', color='Driver', title="Lapchart (lower = better)")
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
    if pdf.empty: return fig_empty("Evolution — no lap data")
    pdf = pdf.sort_values(['Driver','LapNumber'])
    pdf['MA3'] = pdf.groupby('Driver', dropna=False)['LapSeconds'].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f = go.Figure()
    for drv, d in pdf.groupby('Driver'):
        f.add_trace(go.Scatter(x=d['LapNumber'], y=d['MA3'], mode='lines', name=str(drv),
                               hovertemplate=f"{drv} — Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_yaxes(title="sec (MA3)", tickformat=".3f")
    f.update_layout(title="Evolution (3-lap MA)")
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
    if df.empty: return fig_empty("Positions Gained — (Race only / no data)")
    f = px.bar(df, x='Abbreviation', y='PositionsGained', title='Positions Gained', text='PositionsGained')
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
    if st.empty: return fig_empty("Tyre Strategy — no data")
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
    f.update_layout(title='Tyre Strategy', barmode='stack',
                    yaxis=dict(categoryorder='array', categoryarray=order, title='Driver'),
                    xaxis_title='Lap')
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
    if pdf.empty: return fig_empty("Pace — no lap data")
    f = go.Figure()
    for drv, d in pdf.groupby('Driver'):
        f.add_trace(go.Scatter(x=d['LapNumber'], y=d['LapSeconds'], mode='lines+markers',
                               name=str(drv),
                               hovertemplate=f"{drv} — Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>"))
    f.update_yaxes(title="sec", tickformat=".3f")
    f.update_layout(title="Lap-by-Lap Pace")
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
    if laps.empty: return fig_empty("Best Laps — no data")
    best = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].copy()
    best['Best_s'] = best['LapTime'].dt.total_seconds()
    best['BestStr'] = best['Best_s'].apply(s_to_mssmmm)
    f = px.bar(best.sort_values('Best_s'), x='Driver', y='Best_s', custom_data=['BestStr'], title="Best Laps")
    f.update_traces(hovertemplate="%{x}<br>%{y:.3f}s (%{customdata[0]})<extra></extra>")
    f.update_yaxes(title="sec", tickformat=".3f")
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
    if df.empty: return fig_empty("Sector Records — no data")
    f = go.Figure(data=[go.Table(
        header=dict(values=['Sector','Driver','Time (s)'],
                    fill_color='#FFFFFF', font=dict(color='#000000', size=12)),
        cells=dict(values=[df['Sector'], df['Driver'], df['Time (s)']],
                   fill_color='#FFFFFF', font=dict(color='#000000'))
    )])
    f.update_layout(title="Sector Records", paper_bgcolor=COL_PANEL)
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
    if spd.empty: return fig_empty("Speed Records — no data")
    if spd.shape[1] > 2:
        dm = spd.melt(id_vars='Driver', var_name='Metric', value_name='km/h')
        f = px.bar(dm, x='Driver', y='km/h', color='Metric', barmode='group', title='Speed Records')
    else:
        ycol = spd.columns[-1]
        f = px.bar(spd, x='Driver', y=ycol, title='Speed Records')
    f.update_traces(marker_line_width=0)
    f.update_layout(yaxis_title="km/h")
    return polish(f)

@app.callback(
    Output({"role":"csv-dl", "chart": MATCH}, "data"),
    Input({"role":"csv", "chart": MATCH}, "n_clicks"),
    State({"role":"csv", "chart": MATCH}, "id"),
    State("store", "data"),
    State({"role":"drv", "chart": MATCH}, "value"),
    prevent_initial_call=True
)
def download_chart_csv(n_clicks, btn_id, store_data, selected):
    if not n_clicks:
        return no_update
    if not store_data:
        return no_update

    chart_key = btn_id.get("chart")
    try:
        ses = load_session_laps(int(store_data.get('year', 2025)), store_data['event'], store_data['sess'])
        df = export_df_for_chart(chart_key, ses, selected)

        if df is None or df.empty:
            # still return a valid CSV with headers if you want; here we just no-op
            return no_update

        # filename that helps you track what it is
        safe_event = str(store_data["event"]).replace("|", "_").replace(" ", "_")
        safe_sess = str(store_data["sess"])
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
    return jsonify(status="ok")

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
	            g = (testing_df.groupby('EventName', dropna=False)['EventDate']
                         .min()
                         .sort_values()
                         .reset_index(drop=False)
                         .reset_index()
                         .rename(columns={'index':'test_number'}))
                    last_test_name = str(last["EventName"])
                    tn_row = g[g['EventName'] == last_test_name]
                    test_number = int(tn_row['test_number'].iloc[0]) + 1 if not tn_row.empty else 1

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
