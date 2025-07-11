FROM python:3.13-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache --no-dev

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "api.py", "--port", "8000", "--host", "0.0.0.0"]