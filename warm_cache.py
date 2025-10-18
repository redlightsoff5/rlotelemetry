# warm_cache.py — prefetch sessions into FastF1 cache on deploy
import os, time
import fastf1

# Use the Render disk if present, fall back to local ./cache
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2025
SESSIONS_TO_WARM = [
    ("United States Grand Prix", "FP1"),
    # add more if you like:
    # ("United States Grand Prix", "FP2"),
    # ("United States Grand Prix", "FP3"),
    # ("United States Grand Prix", "Q"),
    # ("United States Grand Prix", "R"),
]

for gp, kind in SESSIONS_TO_WARM:
    try:
        print(f"[warm] {YEAR} {gp} {kind}", flush=True)
        ses = fastf1.get_session(YEAR, gp, kind)
        # laps=True is enough to build most cache files fast
        ses.load(telemetry=False, weather=False, laps=True)
        # Touch some properties to force writes
        _ = ses.results
        _ = ses.laps.head(1)
        time.sleep(0.25)
    except Exception as e:
        print(f"[warm] skip {gp} {kind}: {e}", flush=True)
