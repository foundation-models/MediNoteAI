#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it first using:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements using uv
echo "Installing requirements..."
uv pip install -r requirements.txt

# Run the Chainlit app
echo "Starting Chainlit app..."
chainlit run validation_via_model_chainlit.py 