#!/bin/bash

echo "========================================="
echo "AI Trading Bot - Test Runner"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run backend tests
run_backend_tests() {
    echo -e "\n${YELLOW}Running Backend API Tests...${NC}"
    echo "========================================="
    
    # Check if pytest is installed
    if ! pip list | grep -q pytest; then
        echo "Installing pytest..."
        pip install pytest pytest-cov
    fi
    
    # Run tests
    pytest tests/test_api.py -v --tb=short
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Backend tests passed!${NC}"
    else
        echo -e "\n${RED}✗ Backend tests failed!${NC}"
        return 1
    fi
}

# Function to run frontend tests
run_frontend_tests() {
    echo -e "\n${YELLOW}Running Frontend Tests...${NC}"
    echo "========================================="
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    # Run tests
    npm test -- --watchAll=false --coverage
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Frontend tests passed!${NC}"
    else
        echo -e "\n${RED}✗ Frontend tests failed!${NC}"
        return 1
    fi
    
    cd ..
}

# Function to run integration tests
run_integration_tests() {
    echo -e "\n${YELLOW}Running Integration Tests...${NC}"
    echo "========================================="
    
    # Start backend in background
    echo "Starting backend server..."
    python test_simple.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 3
    
    # Test if backend is running
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✓ Backend is running${NC}"
        
        # Run API tests against live server
        pytest tests/test_api.py -v --tb=short
        
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}✓ Integration tests passed!${NC}"
        else
            echo -e "\n${RED}✗ Integration tests failed!${NC}"
        fi
    else
        echo -e "${RED}✗ Backend failed to start${NC}"
    fi
    
    # Kill backend
    kill $BACKEND_PID 2>/dev/null
}

# Main menu
echo ""
echo "Select test type:"
echo "1) Run Backend Tests Only"
echo "2) Run Frontend Tests Only"
echo "3) Run All Tests"
echo "4) Run Integration Tests"
echo "5) Run Tests with Coverage"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        run_backend_tests
        ;;
    2)
        run_frontend_tests
        ;;
    3)
        run_backend_tests
        run_frontend_tests
        ;;
    4)
        run_integration_tests
        ;;
    5)
        echo -e "\n${YELLOW}Running tests with coverage...${NC}"
        pytest tests/test_api.py -v --cov=. --cov-report=html --cov-report=term
        echo -e "\n${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Test run completed!${NC}"
