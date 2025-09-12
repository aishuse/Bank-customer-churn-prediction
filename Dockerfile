# Use slim Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y supervisor curl bash && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install uv streamlit supervisor

# Copy pyproject.toml and install your package
COPY pyproject.toml .
RUN uv pip install --system -e .

# Copy application code
COPY . .

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8080

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
