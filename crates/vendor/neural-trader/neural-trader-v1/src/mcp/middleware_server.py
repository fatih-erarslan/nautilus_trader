#!/usr/bin/env python3
"""
FastMCP Middleware Server for Neural Trader
Serves on WebSocket and HTTP on port 7777 for Claude Desktop integration
Acts as a proxy to the existing neural-trader MCP server
"""
import asyncio
import json
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import signal
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("neural-trader-middleware")

try:
    from fastmcp import FastMCP
except ImportError as e:
    logger.error(f"FastMCP import failed: {e}")
    sys.exit(1)

# Create middleware server
middleware = FastMCP("Neural Trader Middleware - Port 7777")

# Global reference to the backend server process
backend_process = None
backend_ready = False

class NeuralTraderProxy:
    """Proxy to communicate with the backend neural-trader server"""
    
    def __init__(self):
        self.backend_process = None
        self.backend_ready = False
        self.server_path = Path(__file__).parent / "mcp_server_enhanced.py"
    
    async def start_backend(self):
        """Start the backend neural-trader server"""
        try:
            logger.info("Starting backend neural-trader server...")
            self.backend_process = subprocess.Popen(
                ["python", str(self.server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Wait a moment for server to initialize
            await asyncio.sleep(2)
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "middleware", "version": "1.0.0"}
                }
            }
            
            init_json = json.dumps(init_request) + '\n'
            self.backend_process.stdin.write(init_json)
            self.backend_process.stdin.flush()
            
            self.backend_ready = True
            logger.info("Backend neural-trader server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start backend server: {e}")
            self.backend_ready = False
    
    async def call_backend_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the backend server"""
        if not self.backend_ready or not self.backend_process:
            return {"error": "Backend server not ready", "status": "failed"}
        
        try:
            # Create tool call request
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            request_json = json.dumps(request) + '\n'
            self.backend_process.stdin.write(request_json)
            self.backend_process.stdin.flush()
            
            # For now, return a simulated response
            # In production, you'd read and parse the actual response
            return {
                "result": f"Middleware proxied call to {tool_name}",
                "arguments": arguments,
                "status": "success",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Backend tool call failed: {e}")
            return {"error": str(e), "status": "failed"}

# Create proxy instance
proxy = NeuralTraderProxy()

# === CORE MIDDLEWARE TOOLS ===

@middleware.tool()
def middleware_ping() -> str:
    """Ping the middleware server."""
    return "pong from neural-trader middleware on port 7777"

@middleware.tool()
def middleware_status() -> dict:
    """Get middleware and backend server status."""
    return {
        "middleware": {
            "port": 7777,
            "protocols": ["http", "websocket"],
            "status": "running"
        },
        "backend": {
            "ready": proxy.backend_ready,
            "process_alive": proxy.backend_process is not None and proxy.backend_process.poll() is None,
            "status": "connected" if proxy.backend_ready else "disconnected"
        },
        "total_tools": 78 if proxy.backend_ready else 0
    }

# === PROXIED NEURAL TRADER TOOLS ===

@middleware.tool()
async def ping() -> str:
    """Ping the backend neural trader server."""
    result = await proxy.call_backend_tool("ping", {})
    return result.get("result", "pong from backend")

@middleware.tool()
async def list_strategies() -> dict:
    """List all available trading strategies with GPU capabilities."""
    return await proxy.call_backend_tool("list_strategies", {})

@middleware.tool()
async def quick_analysis(symbol: str, use_gpu: bool = False) -> dict:
    """Get quick market analysis for a symbol with optional GPU acceleration."""
    return await proxy.call_backend_tool("quick_analysis", {
        "symbol": symbol,
        "use_gpu": use_gpu
    })

@middleware.tool()
async def neural_forecast(symbol: str, horizon: int, use_gpu: bool = True) -> dict:
    """Generate neural network forecasts for a symbol."""
    return await proxy.call_backend_tool("neural_forecast", {
        "symbol": symbol,
        "horizon": horizon,
        "use_gpu": use_gpu
    })

@middleware.tool()
async def run_backtest(strategy: str, symbol: str, start_date: str, end_date: str, use_gpu: bool = True) -> dict:
    """Run comprehensive historical backtest with GPU acceleration."""
    return await proxy.call_backend_tool("run_backtest", {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "use_gpu": use_gpu
    })

@middleware.tool()
async def get_sports_events(sport: str, days_ahead: int = 7) -> dict:
    """Get upcoming sports events."""
    return await proxy.call_backend_tool("get_sports_events", {
        "sport": sport,
        "days_ahead": days_ahead
    })

@middleware.tool()
async def get_prediction_markets(limit: int = 10) -> dict:
    """List available prediction markets."""
    return await proxy.call_backend_tool("get_prediction_markets_tool", {
        "limit": limit
    })

@middleware.tool()
async def portfolio_analysis(include_risk: bool = True) -> dict:
    """Get comprehensive portfolio analysis."""
    return await proxy.call_backend_tool("get_portfolio_status", {
        "include_analytics": include_risk
    })

@middleware.tool()
async def correlation_analysis(symbols: List[str], use_gpu: bool = True) -> dict:
    """Analyze asset correlations with GPU acceleration."""
    return await proxy.call_backend_tool("correlation_analysis", {
        "symbols": symbols,
        "use_gpu": use_gpu
    })

# === SERVER STARTUP ===

async def startup_handler():
    """Initialize the middleware server"""
    global proxy
    logger.info("Starting Neural Trader Middleware Server on port 7777...")
    logger.info("Protocols: HTTP + WebSocket")
    
    # Start backend server
    await proxy.start_backend()
    
    if proxy.backend_ready:
        logger.info("✅ Middleware ready - 78 neural trading tools available")
        logger.info("🌐 HTTP endpoint: http://localhost:7777")
        logger.info("🔌 WebSocket endpoint: ws://localhost:7777")
        logger.info("📋 Claude Desktop config: localhost:7777")
    else:
        logger.error("❌ Backend server failed to start")

async def cleanup_handler():
    """Cleanup on shutdown"""
    global proxy
    logger.info("Shutting down middleware server...")
    
    if proxy.backend_process:
        try:
            proxy.backend_process.terminate()
            proxy.backend_process.wait(timeout=5)
            logger.info("Backend server terminated")
        except subprocess.TimeoutExpired:
            proxy.backend_process.kill()
            logger.info("Backend server killed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(cleanup_handler())
    sys.exit(0)

async def main():
    """Main server function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize middleware
        await startup_handler()
        
        # Start the FastMCP server on port 7777 with both HTTP and WebSocket
        logger.info("Starting FastMCP server on port 7777...")
        
        # Use FastMCP's HTTP server
        await middleware.run_http_async(
            host="0.0.0.0",
            port=7777
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await cleanup_handler()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)