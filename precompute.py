"""Pre-compute FastF1 session data into parquet files under ./data so the web
app can serve charts instantly (no live FastF1 download on each click).

Run by .github/workflows/precompute.yml (which has network). Locally it only
succeeds for sessions already present in the FastF1 disk cache; everything else
is skipped gracefully.

Layout written:
  data/<year>/<event-slug>/<SESS>.laps.parquet
  data/<year>/<event-slug>/<SESS>.results.parquet
  data/<year>/<event-slug>/<SESS>.meta.json
"""
import os, sys, json, warnings, traceback
warnings.filterwarnings("ignore")

import fastf1 as ff1
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(APP_DIR, "cache"))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(CACHE_DIR, exist_ok=True)
ff1.Cache.enable_cache(CACHE_DIR)

YEARS = [int(y) for y in os.environ.get("PRECOMPUTE_YEARS", "2025,2026").split(",") if y.strip()]
SESSIONS = ["R", "SR", "Q", "SQ"]   # most-viewed; practice/testing stay live

# Only the columns the app's chart builders actually use (keeps files tiny).
LAP_COLS = ['Driver', 'Team', 'LapNumber', 'LapTime', 'Position', 'Compound', 'Stint',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']


def slug(name: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in str(name))
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def write_session(year: int, event_name: str, sess_code: str) -> bool:
    try:
        ses = ff1.get_session(year, event_name, sess_code)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  skip {year} {event_name!r} {sess_code}: {type(e).__name__}")
        return False

    try:
        laps = ses.laps
        if laps is None or len(laps) == 0:
            print(f"  no laps  {year} {event_name!r} {sess_code}")
            return False

        out_dir = os.path.join(DATA_DIR, str(year), slug(event_name))
        os.makedirs(out_dir, exist_ok=True)

        cols = [c for c in LAP_COLS if c in laps.columns]
        laps[cols].reset_index(drop=True).to_parquet(
            os.path.join(out_dir, f"{sess_code}.laps.parquet"), index=False)

        res = getattr(ses, "results", None)
        if res is not None and len(res) > 0:
            try:
                res.reset_index(drop=True).to_parquet(
                    os.path.join(out_dir, f"{sess_code}.results.parquet"), index=False)
            except Exception:
                print(f"  (results not serialisable for {event_name} {sess_code})")

        meta = {
            "session_type": str(getattr(ses, "session_type", "") or ""),
            "name": str(getattr(ses, "name", "") or ""),
            "year": int(year), "event": str(event_name), "sess": sess_code,
        }
        with open(os.path.join(out_dir, f"{sess_code}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        print(f"  OK   {year} {event_name!r} {sess_code}: {len(laps)} laps")
        return True
    except Exception:
        traceback.print_exc()
        return False


def main():
    today = pd.Timestamp.utcnow().tz_localize(None)
    written = 0
    for year in YEARS:
        try:
            sched = ff1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"schedule unavailable for {year}: {type(e).__name__}")
            continue
        sched = sched[pd.to_datetime(sched["EventDate"]) <= today]
        print(f"[{year}] {len(sched)} completed events")
        for _, ev in sched.iterrows():
            name = str(ev["EventName"])
            for sc in SESSIONS:
                if write_session(year, name, sc):
                    written += 1
    print(f"DONE: wrote/updated {written} sessions into {DATA_DIR}")
    return written


if __name__ == "__main__":
    main()
