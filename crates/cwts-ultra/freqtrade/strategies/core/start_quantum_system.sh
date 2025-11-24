#!/bin/bash

# Start Integrated Quantum Trading System
# This script starts all components with proper dependencies

echo "=================================================="
echo "INTEGRATED QUANTUM TRADING SYSTEM STARTUP"
echo "=================================================="

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export QUANTUM_SYSTEM_CONFIG="quantum_system_config.json"

# Function to check if Redis is running
check_redis() {
    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli ping >/dev/null 2>&1; then
            echo "✓ Redis is running"
            return 0
        else
            echo "✗ Redis is not running"
            return 1
        fi
    else
        echo "✗ Redis CLI not found"
        return 1
    fi
}

# Function to start Redis if needed
start_redis() {
    echo "Starting Redis server..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port 6379
        sleep 2
        if check_redis; then
            echo "✓ Redis started successfully"
        else
            echo "✗ Failed to start Redis"
            return 1
        fi
    else
        echo "✗ Redis server not found. Please install Redis."
        return 1
    fi
}

# Check dependencies
echo "Checking system dependencies..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Check Redis
if ! check_redis; then
    echo "Attempting to start Redis..."
    if ! start_redis; then
        echo "Warning: Redis not available. System will run with limited messaging."
    fi
fi

# Check Python packages
echo "Checking Python packages..."
required_packages=("asyncio" "redis" "zmq" "numpy" "pandas")
for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package available"
    else
        echo "✗ $package not available"
    fi
done

echo ""
echo "Starting Integrated Quantum Trading System..."
echo "Press Ctrl+C to stop the system"
echo ""

# Start the main system
python3 integrated_quantum_trading_system.py

echo ""
echo "System shutdown complete."