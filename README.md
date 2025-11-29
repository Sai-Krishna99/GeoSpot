# GeoSpot

AI Location Scout for Entrepreneurs – De-risking business location decisions with data-driven insights.

## Overview

GeoSpot empowers entrepreneurs to make informed decisions about physical business locations by analyzing public data from OpenStreetMap, U.S. Census Bureau, and generating strategic narratives. Built for the "Internet of Agents" hackathon using Coral Protocol for agent orchestration.

## Features

- **Data Acquisition**: Fetches POI (Points of Interest), demographics, and competition data.
- **Scoring Engine**: Ranks neighborhoods based on foot traffic, demographics, and competition.
- **AI Narrative**: Generates persuasive pro/con analyses using AIML API.
- **Audio Briefing**: Converts reports to voice using ElevenLabs.
- **Agent-Based**: Uses Coral Protocol for collaborative, visible agent workflows.

## Setup

### Prerequisites

Run the dependency check script:

```bash
./check-dependencies.sh
```

This verifies and installs:
- Python 3.10+
- uv (Python package manager)
- Node.js 18+
- Java 21+ (for Coral Server)
- Git

### API Keys

Required:
- **AIML API Key**: For AI narrative generation (get from [aimlapi.com](https://aimlapi.com))
- **ElevenLabs API Key**: For audio generation (optional, get from [elevenlabs.io](https://elevenlabs.io))
- **U.S. Census API Key**: For demographics data (optional, get from [census.gov](https://www.census.gov/data/developers/data-sets.html))

Add keys to `.env`:
```
AIML_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
CENSUS_API_KEY=your_key_here
```

### Coral Protocol Setup

1. **Initialize Submodules**:
   ```bash
   git submodule init
   git submodule update
   ```

2. **Start Coral Server**:
   ```bash
   cd coral-server
   REGISTRY_FILE_PATH="../registry.toml" ./gradlew run
   ```
   Leave running in background.
   If it doesnt work, provide full path to `registry.toml`. 
   example -
   ```bash
   cd <path to your GeoSpot directory>/GeoSpot && REGISTRY_FILE_PATH="<path to your GeoSpot directory>/GeoSpot/registry.toml" ./coral-server/gradlew -p coral-server run
```

3. **Start Coral Studio**:
   ```bash
   npx @coral-protocol/coral-studio
   ```
   Access at [http://localhost:3000](http://localhost:3000).

### Running Agents

- In Coral Studio, register agents from `registry.toml`.
- Add agents to a session, provide business profile inputs.
- Agents collaborate: DataFetcher → Scorer → Narrator → Orchestrator.

### Development

- Install Python dependencies: `uv sync`
- Run CLI: `uv run geospot --help`
- Run API: `uv run geospot-api`

## Roadmap

- [ ] Refactor core logic into Coral agents
- [ ] Add Streamlit frontend
- [ ] Enhance error handling and resilience
- [ ] Add more data sources

## License

MIT 
