#!/bin/bash
# Launch script for Vioce GUI
# Usage: ./launch.sh [--port PORT] [--share]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the app
python app.py "$@"
