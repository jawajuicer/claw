# ── Build stage ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY mcp_tools/ mcp_tools/

# Build wheels
RUN pip wheel --wheel-dir=/wheels -e ".[all]" || pip wheel --wheel-dir=/wheels -e .

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="The Claw"
LABEL org.opencontainers.image.description="Local-first voice AI assistant with MCP tool calling"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    ffmpeg \
    mpv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy source and tools
COPY src/ src/
COPY mcp_tools/ mcp_tools/
COPY config.yaml.example config.yaml

# Create data directories
RUN mkdir -p data/chromadb data/notes data/secrets data/conversations data/google data/models && \
    chmod 700 data/secrets data/google

# Default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -sf http://localhost:8080/api/health || exit 1

# Run
ENTRYPOINT ["python", "-m", "claw"]
CMD ["--mode", "both"]
