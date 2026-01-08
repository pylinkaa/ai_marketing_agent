#!/bin/bash
# Script to run Streamlit app

cd "$(dirname "$0")"
python3 -m streamlit run src/app/streamlit_app.py
