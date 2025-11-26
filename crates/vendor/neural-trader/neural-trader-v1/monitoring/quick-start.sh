#!/bin/bash

# Neural Trader Monitoring Quick Start Script
# Deployment ID: neural-trader-1763096012878

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Neural Trader Monitoring - Quick Start                       ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Deployment ID: neural-trader-1763096012878                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if in monitoring directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the monitoring directory"
    echo "   cd /workspaces/neural-trader/monitoring"
    exit 1
fi

# Function to check if node_modules exists
check_dependencies() {
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing dependencies..."
        npm install
        echo "✅ Dependencies installed"
        echo ""
    else
        echo "✅ Dependencies already installed"
        echo ""
    fi
}

# Function to run validation
run_validation() {
    echo "🔍 Running deployment validation..."
    echo ""
    npm run validate
    echo ""
}

# Function to show menu
show_menu() {
    echo "What would you like to do?"
    echo ""
    echo "  1) Comprehensive Status Display (Recommended)"
    echo "  2) Real-Time Dashboard"
    echo "  3) Health Check System"
    echo "  4) Run Validation Tests"
    echo "  5) Generate Performance Report"
    echo "  6) Dashboard + Health Checks (Concurrent)"
    echo "  7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice
}

# Main menu loop
main() {
    check_dependencies

    while true; do
        show_menu

        case $choice in
            1)
                echo ""
                echo "🚀 Launching comprehensive status display..."
                echo "   (Press Ctrl+C to exit)"
                echo ""
                sleep 2
                ts-node status-display.ts
                ;;
            2)
                echo ""
                echo "🚀 Launching real-time dashboard..."
                echo "   (Press 'q' or ESC to exit)"
                echo ""
                sleep 2
                npm run dashboard
                ;;
            3)
                echo ""
                echo "💓 Starting health check system..."
                echo "   (Press Ctrl+C to stop)"
                echo ""
                sleep 2
                npm run health-check
                ;;
            4)
                echo ""
                run_validation
                read -p "Press Enter to continue..."
                ;;
            5)
                echo ""
                echo "📋 Generating performance report..."
                echo ""
                npm run report
                echo ""
                echo "✅ Reports generated in: reports/output/"
                ls -lh reports/output/
                echo ""
                read -p "Press Enter to continue..."
                ;;
            6)
                echo ""
                echo "🚀 Starting dashboard + health checks..."
                echo "   (Press Ctrl+C to stop all)"
                echo ""
                sleep 2
                npm run monitor-all
                ;;
            7)
                echo ""
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo ""
                echo "❌ Invalid choice. Please enter 1-7."
                echo ""
                sleep 2
                ;;
        esac
    done
}

main
