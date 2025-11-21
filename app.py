import os, warnings, logging, traceback, glob
warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.WARNING)

import fastf1 as ff1
import pandas as pd
import numpy as np
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc

# ================= Setup & cache =================
APP_DIR = os.path.dirname(__file__)
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

YEAR = 2025

# ---- one-time schedule cache refresh (set REFRESH_SCHEDULE=1 in env, redeploy once) ----
if os.environ.get("REFRESH_SCHEDULE") == "1":
    try:
        ydir = os.path.join(CACHE_DIR, str(YEAR))
        for pat in ("*EventSchedule*", "*schedule*"):
            for f in glob.glob(os.path.join(ydir, pat)):
                try:
                    os.remove(f)
                except Exception:
                    pass
    except Exception:
        pass
# ----------------------------------------------------------------------------------------

# Brand
SITE_TITLE = "Telemetry by RedLightsOff"
WATERMARK  = "@redlightsoff5"

COL_BG    = "#0b0b0d"
COL_PANEL = "#141418"
COL_RED   = "#e11d2e"
COL_TEXT  = "#ffffff"

TEAM_COLORS = {
    'Red Bull':    '#4781D7',
    'RB':          '#6C98FF',
    'Ferrari':     '#ED1131',
    'Mercedes':    '#00D7B6',
    'McLaren':     '#F47600',
    'Aston Martin':'#229971',
    'Alpine':      '#00A1E8',
    'Williams':    '#1868DB',
    'Stake':       '#01C00E',   # Sauber/Kick/Audi bucket
    'Haas':        '#9C9FA2'
}

# Map many possible FastF1 team strings to a canonical key in TEAM_COLORS
TEAM_ALIASES = {
    'red bull': 'Red Bull',
    'oracle red bull': 'Red Bull',
    'red bull racing': 'Red Bull',

    'rb f1': 'RB',
    'racing bulls': 'RB',
    'visa cash app rb': 'RB',
    'rb': 'RB',   # keep last as a catch-all

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

    'sauber': 'Stake',
    'kick sauber': 'Stake',
    'stake': 'Stake',
    'audi': 'Stake',   # 2026 branding bucketed here for now

    'haas': 'Haas',
    'haas f1': 'Haas'
}

def canonical_team(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    for key, canon in TEAM_ALIASES.items():
        if key in s:
            return canon
    return name  # fallback to raw string (may still match a TEAM_COLORS key)


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
    # watermark (center of plotting area)
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
def get_schedule_df(year:int) -> pd.DataFrame:
    df = ff1.get_event_schedule(year, include_testing=False).copy()
    df['EventDate'] = pd.to_datetime(df['EventDate'])
    df = df[['RoundNumber','EventName','EventDate']].sort_values('RoundNumber').reset_index(drop=True)
    return df

def build_gp_options(year:int):
    df = get_schedule_df(year)
    return [{'label': f"R{int(r.RoundNumber)} — {r.EventName} ({r.EventDate.date()})",
             'value': r.EventName} for _, r in df.iterrows()]

def default_event_value(year:int):
    df = get_schedule_df(year)
    # pick the latest event up to "now"; if none, first event
    today = pd.Timestamp.utcnow().tz_localize(None)
    past = df[df['EventDate'] <= today]
    if not past.empty:
        return past.iloc[-1]['EventName']
    return df.iloc[0]['EventName'] if not df.empty else None

SESSION_OPTIONS = [
    {"label": "FP1",               "value": "FP1"},
    {"label": "FP2",               "value": "FP2"},
    {"label": "FP3",               "value": "FP3"},
    {"label": "Sprint Qualifying", "value": "SQ"},  # 2024/2025 naming
    {"label": "Qualifying",        "value": "Q"},
    {"label": "Sprint",            "value": "SR"},   # Sprint race
    {"label": "Race",              "value": "R"},
]

# ---------- Loaders ----------
@lru_cache(maxsize=64)
def load_session_laps(event_name:str, sess_code:str):
    """Load a session by official EventName (e.g., 'United States Grand Prix') and session code."""
    try:
        ses = ff1.get_session(YEAR, event_name, str(sess_code))
    except Exception:
        # Back-compat: 2023 used "SS" for Sprint Shootout
        if str(sess_code).upper() == "SQ":
            ses = ff1.get_session(YEAR, event_name, "SS")
        else:
            raise
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    return ses

# ---------- Builders ----------
def driver_team_color_map(ses):
    laps = ses.laps[['Driver','Team']].dropna()
    if laps.empty:
        return {}
    # pick the most frequent team per driver, then normalize it
    team = laps.groupby('Driver')['Team'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
    ).apply(canonical_team)

    # look up color from canonical name; default light grey if truly unknown
    return {drv: TEAM_COLORS.get(t, '#cccccc') for drv, t in team.items()}

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

# ================= Dash =================
external_stylesheets=[dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = SITE_TITLE  # browser tab title

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>{SITE_TITLE}</title>
  {{%favicon%}}
  {{%css%}}
  <style>
    body {{ background:{COL_BG}; color:{COL_TEXT}; }}
    .rlo-navbar {{
      display:flex; align-items:center; justify-content:space-between;
      padding:12px 16px; background:{COL_BG}; border-bottom:1px solid #1e1e24;
    }}
    .rlo-title {{ font-family: 'Montserrat', system-ui, Arial; font-weight:700; font-size:22px; margin:0; }}
    .rlo-title span {{ color:{COL_RED}; }}
    .logo-wrap img {{ height:34px; border-radius:6px; }}
    .dbc-label {{ margin-bottom:6px; font-weight:600; }}
    .tab {{ background:#fff !important; color:#000 !important; border:1px solid #ddd !important; }}
    .tab--selected {{ border-bottom:3px solid {COL_RED} !important; color:#000 !important; }}
    .Select-control, .Select-menu-outer, .Select-value-label {{ color:#000 !important; background:#fff !important; }}
    .Select-placeholder {{ color:#333 !important; }}
    .Select-input > input {{ color:#000 !important; }}
    .box {{ background:{COL_PANEL}; border-radius:12px; padding:8px; }}
  </style>
</head>
<body>
  <div class="rlo-navbar">
    <div class="logo-wrap">
      <img src="/assets/logo.png" alt="logo"/>
    </div>
    <h1 class="rlo-title"><span>Telemetry</span> by RedLightsOff</h1>
  </div>
  {{%app_entry%}}
  {{%config%}}
  {{%scripts%}}
  {{%renderer%}}
</body>
</html>
"""

def header_controls():
    return dbc.Row([
        dbc.Col([
            dbc.Label("Grand Prix"),
            dcc.Dropdown(id='event-dd',
                         options=build_gp_options(YEAR),
                         value=default_event_value(YEAR),
                         clearable=False)
        ], md=7),
        dbc.Col([
            dbc.Label("Session"),
            dcc.Dropdown(id='session-dd',
                         options=SESSION_OPTIONS,
                         value='R',
                         clearable=False)
        ], md=5)
    ], className="mt-3 mb-2")

def graph_box(graph_id, title, chart_key):
    dd_id = {'role':'drv','chart':chart_key}
    return html.Div(className="box", children=[
        dbc.Row([
            dbc.Col(html.H6(title, className="mb-1"), md="auto"),
            dbc.Col(dcc.Dropdown(id=dd_id, options=[], value=[], multi=True, placeholder="Drivers…"), md=True),
        ], align="center", className="mb-2"),
        dcc.Graph(id=graph_id, config={"displaylogo": False, "responsive": True}),
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
    dcc.Tabs(id="tabs", value="evo",
             children=[dcc.Tab(label="Evolution", value="evo"),
                       dcc.Tab(label="Tyres", value="tyres"),
                       dcc.Tab(label="Pace", value="pace"),
                       dcc.Tab(label="Records", value="records"),
                       dcc.Tab(label="Speeds", value="speeds")]),
    html.Div(id="tab-body", className="mt-2", children=tab_evolution()),
    dcc.Store(id='store'),
    dcc.Store(id='drivers-store'),
    dcc.Store(id='team-color-store')
], fluid=True)

@app.callback(Output("tab-body","children"), Input("tabs","value"))
def _render_tabs(val):
    return {"evo":tab_evolution, "tyres":tab_tyres, "pace":tab_pace,
            "records":tab_records, "speeds":tab_speeds}.get(val, tab_evolution)()

# ===== Load session once =====
@app.callback(
    Output('store','data'),
    Output('drivers-store','data'),
    Output('team-color-store','data'),
    Input('event-dd','value'),
    Input('session-dd','value')
)
def load_session_meta(event_name, sess_code):
    if not event_name or not sess_code:
        return no_update, [], {}
    try:
        ses = load_session_laps(str(event_name), str(sess_code))
        laps = ses.laps.dropna(subset=['LapTime'])
        drivers = sorted(laps['Driver'].dropna().unique().tolist())
        colors = driver_team_color_map(ses)
        return {'event': str(event_name), 'sess': str(sess_code)}, drivers, colors
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

# ---------- helper to color by team ----------
def set_trace_color(fig, name_to_color):
    for tr in fig.data:
        c = (name_to_color or {}).get(tr.name)
        if c: tr.update(line=dict(color=c), marker=dict(color=c))
    return fig

# ---------- Evolution charts ----------
@app.callback(
    Output('gap','figure'),
    Input('store','data'), Input({'role':'drv','chart':'gap'}, 'value'),
    State('team-color-store','data')
)
def chart_gap(data, selected, color_map):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    ses = load_session_laps(data['event'], data['sess'])
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
    return brand(f)  # watermark tables too

# ---------- Speeds ----------
@app.callback(
    Output('speeds','figure'),
    Input('store','data'), Input({'role':'drv','chart':'spd'}, 'value')
)
def chart_speeds(data, selected):
    if not data: return fig_empty("(no data)")
    ses = load_session_laps(data['event'], data['sess'])
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

# ================= Run =================
server = app.server

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
