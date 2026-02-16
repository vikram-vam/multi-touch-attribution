#!/bin/bash
# Erie MCA Demo - Quick Setup Script

echo "===================================="
echo "Erie MCA Demo - Setup & Execution"
echo "===================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run data generation pipeline
echo "[5/5] Generating data and running attribution models..."
python main.py
echo ""

echo "===================================="
echo "✓ Setup Complete!"
echo "===================================="
echo ""
echo "To launch the dashboard:"
echo "  1. Ensure virtual environment is active: source venv/bin/activate"
echo "  2. Run: python app.py"
echo "  3. Open browser to: http://localhost:8050"
echo ""
echo "Quick commands:"
echo "  make run          - Launch dashboard"
echo "  make generate-data - Regenerate data"
echo "  make test         - Run tests"
echo "  make clean        - Remove generated data"
echo ""
