# GeoSpot — The AI Location Scout (uv project)

GeoSpot helps entrepreneurs pick the best neighborhood for a first brick‑and‑mortar business using public data:
- **OpenStreetMap (Overpass)** for POIs (competition, transit, vibe)
- **US Census ACS** for demographics (median income & population)
- Optional **Mistral** LLM for a consultant‑style narrative
- Optional **ElevenLabs** for a 60‑second audio briefing

This repo uses **[uv](https://docs.astral.sh/uv/)** (fast Python package manager) and standard Python packaging.

## Quickstart

```bash
# 1) Install uv (if needed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex

# 2) Clone & enter
git clone <YOUR-REPO-URL>.git
cd geospot-uv

# 3) Sync deps
uv sync

# 4) (Optional) set environment variables
# See .env.example for keys and city/neighborhoods.
cp .env.example .env
# Edit .env as needed

# 5) Run (CLI)
uv run geospot --help
uv run geospot rank --city "Austin, Texas" --neighborhoods "Downtown;East Austin;South Congress;Hyde Park"
```

## Environment Variables

Copy `.env.example` to `.env` and fill values:
- `GEOSPOT_CITY` — default city
- `GEOSPOT_NEIGHBORHOODS` — semicolon list of neighborhoods
- `CENSUS_API_KEY` — optional, recommended
- `MISTRAL_API_KEY` — optional, enables narrative
- `ELEVEN_API_KEY` — optional, enables audio
- `ELEVEN_VOICE_ID` — optional, default `Rachel`

## Commands

```bash
uv run geospot rank --city "Austin, Texas" --neighborhoods "Downtown;East Austin;South Congress;Hyde Park"
```

Outputs:
- `geospot_report.json` — ranked picks + features and narrative
- `geospot_briefing.mp3` — if ElevenLabs key provided

## Dev notes

- Core logic in `src/geospot/core.py`
- CLI in `src/geospot/cli.py`
- MIT Licensed — see `LICENSE`



---

## Run as API (FastAPI)

```bash
uv run fastapi dev src/geospot/server/app.py
```

Endpoints:
- `POST /rank` — body: `{ "city": "...", "neighborhoods": ["..."], ... }`
- `GET /health` — health check

## Docker

```bash
docker build -t geospot .
docker run -it -p 8000:8000 --env-file .env geospot
```

Then visit: http://localhost:8000/docs


## Run the API Server (FastAPI)

### With uv
```bash
uv sync
uv run geospot-api
# Server on http://localhost:8000
# Health check:
curl http://localhost:8000/health
# Rank neighborhoods:
curl -X POST http://localhost:8000/rank -H "Content-Type: application/json" -d '{
  "city": "Austin, Texas",
  "neighborhoods": ["Downtown", "East Austin", "Hyde Park"],
  "business": {"type":"coffee_shop", "ideal_pop_density":[4000,15000], "price_position":"mid", "foot_traffic_importance":"high"}
}'
```

### With Docker
```bash
docker build -t geospot:latest .
docker run --rm -p 8000:8000 --env-file .env geospot:latest

# Test:
curl http://localhost:8000/health
```

