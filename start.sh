#!/bin/bash

# Start the FastAPI backend in the background on port 8000
# Streamlit will communicate with this internally using localhost:8000
echo "Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for 3 seconds to ensure backend is ready
sleep 3

# Start the Streamlit frontend on port 7860 (The port Hugging Face exposes)
echo "Starting Streamlit frontend..."
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0
