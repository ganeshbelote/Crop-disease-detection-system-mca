# Frontend — Crop Disease Detection UI

React + TypeScript + Vite + Tailwind CSS interface for the Crop Disease
Detection and Analysis System.

## Setup

```bash
cd frontend
npm install
```

By default the app talks to the backend at `http://localhost:8000`. To point
at a different backend, create a `.env` file:

```
VITE_API_BASE_URL=http://localhost:8000
```

## Run (development)

```bash
npm run dev
```

Open http://localhost:5173

## Build

```bash
npm run build
npm run preview
```

## Pages

- **Dashboard** (`/`) — project overview, supported classes, workflow
- **Analyze Leaf** (`/analyze`) — upload, run analysis, view result
- **History** (`/history`) — prediction history stored in MySQL, with delete
- **Methodology** (`/about`) — dataset, models, evaluation and limitations
