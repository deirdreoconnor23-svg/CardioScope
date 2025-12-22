#!/bin/bash

# CardioScope Launch Script
# Launches the Streamlit web application from src/ folder

echo "=========================================="
echo "  CardioScope - CVD Risk Predictor"
echo "=========================================="
echo ""

# Get the directory where this script is located (CardioScope/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if we're in conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Using conda environment: $CONDA_DEFAULT_ENV"
    echo ""
else
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✓ Virtual environment created"
        echo ""
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
fi

# Install/update dependencies
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if model files exist
echo "Checking for model files..."
if [ ! -f "models/lr_model.pkl" ]; then
    echo "⚠️  Warning: models/lr_model.pkl not found"
fi
if [ ! -f "models/rf_model.pkl" ]; then
    echo "⚠️  Warning: models/rf_model.pkl not found"
fi
if [ ! -f "models/feature_columns.pkl" ]; then
    echo "⚠️  Warning: models/feature_columns.pkl not found"
fi
echo ""

# Check if app.py exists in src/
if [ ! -f "src/app.py" ]; then
    echo "❌ Error: src/app.py not found"
    echo "Please ensure app.py is in the src/ folder"
    exit 1
fi

# Display project info
echo "Project structure:"
echo "  ~/Documents/CardioScope/"
echo "  ├── launch.sh                 (this script)"
echo "  ├── requirements.txt          (dependencies)"
echo "  ├── src/"
echo "  │   └── app.py                (main application)"
echo "  ├── models/"
echo "  │   ├── lr_model.pkl         (logistic regression model)"
echo "  │   ├── rf_model.pkl         (random forest model)"
echo "  │   └── feature_columns.pkl  (feature definitions)"
echo "  ├── data/                     (original datasets)"
echo "  ├── notebooks/                (Jupyter notebooks)"
echo "  └── results/                  (outputs & visualizations)"
echo ""

# Launch Streamlit
echo "=========================================="
echo "  Starting Streamlit Server"
echo "=========================================="
echo ""
echo "CardioScope will open in your browser at:"
echo "  → http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run src/app.py
