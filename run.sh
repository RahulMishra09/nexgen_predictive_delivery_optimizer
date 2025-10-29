#!/bin/bash

# NexGen Predictive Delivery Optimizer - Run Script
# Quick launcher for the Streamlit application

echo "=========================================="
echo "NexGen Delivery Optimizer"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed"
    echo ""
    echo "Please run setup first:"
    echo "  bash setup.sh"
    echo ""
    echo "Or install manually:"
    echo "  pip3 install streamlit"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "⚠️  Warning: data/ directory not found"
    echo "Creating data directory..."
    mkdir -p data
    echo "✅ Created data/ directory"
    echo ""
    echo "Please add your CSV files to data/ before proceeding"
    echo ""
fi

# Check for orders.csv
if [ ! -f "data/orders.csv" ]; then
    echo "⚠️  Warning: data/orders.csv not found"
    echo ""
    echo "At minimum, you need orders.csv to run the app."
    echo "The app will still launch, but you'll need to add data before training models."
    echo ""
fi

echo "🚀 Launching NexGen Delivery Optimizer..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Run streamlit
streamlit run app.py
