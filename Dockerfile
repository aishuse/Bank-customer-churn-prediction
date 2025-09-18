FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    supervisor curl bash net-tools \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --upgrade pip
RUN pip install uvicorn streamlit supervisor mlflow pandas requests \
    langchain langchain-groq python-dotenv langgraph pypdf langchain-community

# Copy Python package metadata and install package
COPY pyproject.toml .
RUN pip install -e .

# Copy source code, artifacts and data
COPY StreamFast /app/StreamFast
COPY artifacts /app/artifacts
COPY data/csv /app/data/csv
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Environment
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PATH="/usr/local/bin:$PATH"

# Expose ports
EXPOSE 8000 8080

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
