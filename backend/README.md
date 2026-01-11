# RLO Telemetry API (FastAPI)

## Local run
```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Render start command
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `CACHE_DIR` to a persistent disk mount for speed.
