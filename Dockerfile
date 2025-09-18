
# Lightweight Python base
FROM python:3.11-slim

# Install curl for uv installer
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh &&     /root/.local/bin/uv --version

WORKDIR /app
COPY . /app

# Sync deps (prod)
RUN /root/.local/bin/uv sync --no-dev

# Expose API port
EXPOSE 8000

# Env passthrough (optional)
ENV PYTHONUNBUFFERED=1

# Start server
CMD ["/root/.local/bin/uv", "run", "geospot-api"]
