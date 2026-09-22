#!/bin/bash
PORT=${PORT:-${WEBSITES_PORT:-8000}}
echo "Starting ResearchLens AI Streamlit on port $PORT..."
python -m streamlit run app.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
