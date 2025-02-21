#!/bin/bash

# Activate virtual environment - try both Windows and Unix paths
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Check if .env exists and has API key
if [ ! -f .env ] || ! grep -q "GOOGLE_API_KEY=" .env; then
    echo "ERROR: .env file missing or API key not set"
    echo "Please create .env file and add your Google API key:"
    echo "GOOGLE_API_KEY=your_api_key_here"
    exit 1
fi

# Run the main script
python main.py

# Deactivate virtual environment when done
deactivate