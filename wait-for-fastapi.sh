#!/bin/bash
# Wait until FastAPI responds on port 8000
until curl -s http://localhost:8000/predict > /dev/null; do
  echo "Waiting for FastAPI..."
  sleep 2
done

# Start Streamlit
streamlit run StreamFast/app.py --server.port=8080 --server.address=0.0.0.0
