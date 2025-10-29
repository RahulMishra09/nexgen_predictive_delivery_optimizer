#!/bin/bash

# NexGen Predictive Delivery Optimizer - Setup Script
# This script sets up the environment and installs dependencies

echo "=========================================="
echo "NexGen Delivery Optimizer - Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
echo ""

# Check if pip is installed
echo "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pip3 is installed${NC}"
echo ""

# Create virtual environment (recommended)
read -p "Do you want to create a virtual environment? (recommended) [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo ""

    echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
    echo "  source venv/bin/activate  (on macOS/Linux)"
    echo "  venv\\Scripts\\activate     (on Windows)"
    echo ""

    # Activate venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All packages installed successfully${NC}"
else
    echo -e "${RED}❌ Error installing packages${NC}"
    exit 1
fi
echo ""

# Check data directory
echo "Checking data directory..."
if [ -d "data" ]; then
    echo -e "${GREEN}✅ Data directory exists${NC}"

    # Check for required CSV files
    REQUIRED_FILES=("orders.csv" "customers.csv" "warehouses.csv" "carriers.csv" "fleet.csv" "tracking.csv" "costs.csv")
    MISSING_FILES=()

    echo ""
    echo "Checking for required CSV files:"
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "data/$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (missing)"
            MISSING_FILES+=("$file")
        fi
    done

    if [ ${#MISSING_FILES[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  Warning: Some CSV files are missing${NC}"
        echo "Missing files: ${MISSING_FILES[*]}"
        echo "Note: At minimum, orders.csv is required"
    fi
else
    echo -e "${YELLOW}⚠️  Data directory does not exist${NC}"
    echo "Creating data directory..."
    mkdir -p data
    echo -e "${GREEN}✅ Created data/ directory${NC}"
    echo ""
    echo -e "${YELLOW}Please add your CSV files to the data/ directory:${NC}"
    echo "  - orders.csv (required)"
    echo "  - customers.csv"
    echo "  - warehouses.csv"
    echo "  - carriers.csv"
    echo "  - fleet.csv"
    echo "  - tracking.csv"
    echo "  - costs.csv"
fi
echo ""

# Create models directory
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
    echo -e "${GREEN}✅ Created models/ directory${NC}"
fi
echo ""

# Setup complete
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure your CSV files are in the data/ directory"
echo "  2. Run the application:"
echo "     streamlit run app.py"
echo ""
echo "For more information, see README.md"
echo ""
