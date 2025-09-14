# Use slim Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y supervisor curl bash \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install uvicorn streamlit supervisor mlflow pandas requests langchain langchain-groq python-dotenv

# Copy Python package metadata and install package (if you have one)
COPY pyproject.toml .
RUN pip install -e .

# Copy source code
COPY StreamFast /app/StreamFast

# Copy artifacts explicitly
COPY artifacts /app/artifacts

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8080



# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
