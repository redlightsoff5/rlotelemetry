# RLO Telemetry Frontend (Next.js + ECharts)

## Local run
```bash
cp .env.example .env.local
# edit NEXT_PUBLIC_API_BASE
npm install
npm run dev
```

## Deploy
Deploy on Vercel. Set environment variable:
- `NEXT_PUBLIC_API_BASE` = your Render API base URL
