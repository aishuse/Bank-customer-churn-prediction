FROM python:3.11-slim

WORKDIR /app

# Install uv (fast package manager for pyproject.toml)
RUN pip install uv

# Copy only dependency files first
COPY pyproject.toml .

# Install dependencies from pyproject.toml
RUN uv pip install --system -e .

# Copy the rest of the application
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
