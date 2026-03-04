#!/bin/bash

# Sepsis Prediction API - Quick Start Script (Linux/Mac)
# Usage: bash run_api.sh

set -e

echo "======================================================================"
echo "  SEPSIS PREDICTION SYSTEM - API QUICK START"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${BLUE}✓ Python ${PYTHON_VERSION} found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
if pip install -r requirements_api.txt > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some dependencies may have failed to install${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo -e "${GREEN}✓ Logs directory created${NC}"
fi

# Display start options
echo ""
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Choose how to start the API:${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo "1) Development mode (with auto-reload)"
echo "2) Production mode (no reload)"
echo "3) Run tests"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${YELLOW}Starting API in development mode...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:8000${NC}"
        echo -e "${YELLOW}Docs at: http://localhost:8000/docs${NC}"
        echo ""
        python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
        ;;
    2)
        echo -e "${YELLOW}Starting API in production mode...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:8000${NC}"
        echo -e "${YELLOW}Docs at: http://localhost:8000/docs${NC}"
        echo ""
        python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
        ;;
    3)
        echo -e "${YELLOW}Running tests...${NC}"
        python3 -m pytest test_app.py -v
        ;;
    4)
        echo -e "${YELLOW}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Exiting...${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=====================================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}=====================================================================${NC}"
