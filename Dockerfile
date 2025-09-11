FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install uv supervisor

COPY pyproject.toml .
RUN uv pip install --system -e .

# Copy application code
COPY . .

# Expose both ports
EXPOSE 8000 8080

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
