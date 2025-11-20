#!/usr/bin/env bash
set -e
echo "Creating venv..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Starting Streamlit app..."
streamlit run webui/streamlit_app.py
