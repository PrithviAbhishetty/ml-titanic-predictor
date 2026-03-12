FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install production dependencies only
RUN uv sync --no-dev

# Copy application code
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Start the API server
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]