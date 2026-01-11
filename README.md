# RLO Telemetry (Proper Architecture)

Frontend: Next.js (Vercel) + Apache ECharts  
Backend: FastAPI (Render) + FastF1

## Why this
- Fancy charts without fragile Dash component wrappers
- Independent scaling (frontend static + backend compute)
- Clean "product" UI and consistent styling

## Deploy quick
1) Deploy `backend/` as a Render Web Service.
2) Deploy `frontend/` on Vercel and set `NEXT_PUBLIC_API_BASE` to the backend URL.
