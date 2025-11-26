# Developer Integration Guide

Comprehensive guide for developers integrating with the AI News Trading Platform's Neural Forecasting capabilities.

## Overview

The AI News Trading Platform provides multiple integration points for developers:

- **Python API**: Direct integration with neural forecasting models
- **MCP Tools**: JSON-RPC 2.0 protocol for tool integration
- **Claude-Flow CLI**: Command-line orchestration and automation
- **REST API**: HTTP-based service integration
- **WebSocket API**: Real-time data streaming
- **Plugin System**: Custom tool and strategy development

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Developer Integration Layer                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Python API  │  │ REST API    │  │ WebSocket   │  │ Plugins  │ │
│  │             │  │             │  │ API         │  │          │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    MCP Protocol Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Neural      │  │ Trading     │  │ Risk        │  │ Market   │ │
│  │ Forecasting │  │ Strategies  │  │ Management  │  │ Data     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Core Platform Services                        │
└─────────────────────────────────────────────────────────────────┘
```

## Python API Integration

### Basic Integration

```python
# Basic neural forecasting integration
import sys
import os
sys.path.append('/path/to/ai-news-trader/src')

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import pandas as pd

class TradingSystem:
    """Example trading system integration"""
    
    def __init__(self):
        self.forecaster = None
        self.initialize_forecasting()
    
    def initialize_forecasting(self):
        """Initialize neural forecasting capabilities"""
        
        # Configure NHITS model for trading
        model = NHITS(
            input_size=84,      # 12 weeks of data
            h=30,               # 30-day forecast
            max_epochs=100,
            batch_size=32,
            accelerator='auto',  # GPU if available
            alias='TradingNHITS'
        )
        
        self.forecaster = NeuralForecast(
            models=[model], 
            freq='D'
        )
        
        print("✓ Neural forecasting initialized")
    
    def train_model(self, historical_data):
        """Train forecasting model with historical data"""
        
        # Data must have columns: ds, unique_id, y
        # ds: datetime
        # unique_id: symbol identifier
        # y: price/value to forecast
        
        self.forecaster.fit(historical_data)
        print("✓ Model training completed")
    
    def generate_forecast(self, symbol, horizon=30):
        """Generate forecast for trading symbol"""
        
        forecasts = self.forecaster.predict(h=horizon, level=[80, 95])
        
        # Filter for specific symbol
        symbol_forecast = forecasts[
            forecasts['unique_id'] == symbol
        ]
        
        return symbol_forecast
    
    def get_trading_signal(self, symbol, forecast_data):
        """Convert forecast to trading signal"""
        
        if len(forecast_data) == 0:
            return {'action': 'HOLD', 'confidence': 0}
        
        # Get forecast values
        forecast_values = forecast_data['TradingNHITS'].values
        
        # Calculate trend
        trend = (forecast_values[-1] - forecast_values[0]) / forecast_values[0]
        
        # Generate signal based on trend
        if trend > 0.05:  # 5% upward trend
            return {'action': 'BUY', 'confidence': min(trend * 10, 1.0)}
        elif trend < -0.05:  # 5% downward trend
            return {'action': 'SELL', 'confidence': min(abs(trend) * 10, 1.0)}
        else:
            return {'action': 'HOLD', 'confidence': 0.5}

# Usage example
trading_system = TradingSystem()

# Prepare sample data
sample_data = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=365, freq='D'),
    'unique_id': 'AAPL',
    'y': 150 + pd.Series(range(365)).apply(lambda x: x * 0.1 + np.random.normal(0, 2))
})

# Train and generate signals
trading_system.train_model(sample_data)
forecast = trading_system.generate_forecast('AAPL', horizon=30)
signal = trading_system.get_trading_signal('AAPL', forecast)

print(f"Trading signal: {signal}")
```

### Advanced Integration with Custom Models

```python
# Advanced integration with custom neural architectures
import torch
import torch.nn as nn
from neuralforecast.core import TimeSeriesDataset
from neuralforecast.models.base import BaseModel

class CustomTradingModel(BaseModel):
    """Custom neural model for trading forecasts"""
    
    def __init__(self, input_size, h, hidden_size=256, num_layers=2):
        super().__init__()
        
        self.input_size = input_size
        self.h = h
        self.hidden_size = hidden_size
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, h)
        )
    
    def forward(self, x):
        """Forward pass"""
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep
        last_hidden = attn_out[:, -1, :]
        
        # Generate forecast
        forecast = self.output_layers(last_hidden)
        
        return forecast

# Integration with custom model
class AdvancedTradingSystem(TradingSystem):
    """Advanced trading system with custom models"""
    
    def __init__(self):
        super().__init__()
        self.custom_models = {}
    
    def add_custom_model(self, symbol, model_config):
        """Add custom model for specific symbol"""
        
        custom_model = CustomTradingModel(
            input_size=model_config['input_size'],
            h=model_config['horizon'],
            hidden_size=model_config.get('hidden_size', 256)
        )
        
        self.custom_models[symbol] = NeuralForecast(
            models=[custom_model],
            freq='D'
        )
    
    def train_custom_model(self, symbol, data):
        """Train custom model for symbol"""
        
        if symbol in self.custom_models:
            self.custom_models[symbol].fit(data)
            print(f"✓ Custom model trained for {symbol}")
    
    def ensemble_forecast(self, symbol, horizon=30):
        """Generate ensemble forecast using multiple models"""
        
        forecasts = []
        
        # Standard NHITS forecast
        if self.forecaster:
            standard_forecast = self.generate_forecast(symbol, horizon)
            forecasts.append(standard_forecast)
        
        # Custom model forecast
        if symbol in self.custom_models:
            custom_forecast = self.custom_models[symbol].predict(h=horizon)
            forecasts.append(custom_forecast)
        
        if not forecasts:
            return None
        
        # Simple ensemble averaging
        ensemble_values = sum([f['forecast_column'].values for f in forecasts]) / len(forecasts)
        
        # Create ensemble result
        ensemble_forecast = forecasts[0].copy()
        ensemble_forecast['ensemble'] = ensemble_values
        
        return ensemble_forecast
```

## MCP Tools Integration

### Direct MCP Client

```python
# Direct MCP client integration
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class MCPClient:
    """Client for MCP server integration"""
    
    def __init__(self, base_url: str = "http://localhost:3000/mcp"):
        self.base_url = base_url
        self.session = None
        self.request_id = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def call_tool(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Call MCP tool method"""
        
        self.request_id += 1
        
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }
        
        async with self.session.post(
            self.base_url,
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            result = await response.json()
            
            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")
            
            return result.get("result", {})
    
    async def neural_forecast(self, symbol: str, horizon: int = 30) -> Dict[str, Any]:
        """Generate neural forecast via MCP"""
        
        return await self.call_tool("quick_analysis", {
            "symbol": symbol,
            "use_gpu": True
        })
    
    async def simulate_trade(self, strategy: str, symbol: str, action: str) -> Dict[str, Any]:
        """Simulate trade via MCP"""
        
        return await self.call_tool("simulate_trade", {
            "strategy": strategy,
            "symbol": symbol,
            "action": action,
            "use_gpu": True
        })
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get portfolio status via MCP"""
        
        return await self.call_tool("get_portfolio_status", {
            "include_analytics": True
        })
    
    async def run_backtest(self, strategy: str, symbol: str, 
                          start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest via MCP"""
        
        return await self.call_tool("run_backtest", {
            "strategy": strategy,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "use_gpu": True
        })

# Usage example
async def trading_bot_with_mcp():
    """Example trading bot using MCP integration"""
    
    async with MCPClient() as mcp:
        
        # Get neural forecast
        forecast_result = await mcp.neural_forecast("AAPL", horizon=5)
        print("Forecast result:", forecast_result)
        
        # Simulate trade based on forecast
        neural_forecast = forecast_result.get("neural_forecast", {})
        expected_direction = neural_forecast.get("trend_direction", "neutral")
        
        if expected_direction == "up":
            trade_result = await mcp.simulate_trade(
                "momentum_trading_optimized", "AAPL", "buy"
            )
            print("Trade simulation:", trade_result)
        
        # Check portfolio status
        portfolio = await mcp.get_portfolio_status()
        print("Portfolio status:", portfolio.get("portfolio", {}))

# Run the trading bot
# asyncio.run(trading_bot_with_mcp())
```

### Synchronous MCP Integration

```python
# Synchronous MCP integration for simpler use cases
import requests
import json
from typing import Dict, Any, Optional

class SyncMCPClient:
    """Synchronous MCP client"""
    
    def __init__(self, base_url: str = "http://localhost:3000/mcp"):
        self.base_url = base_url
        self.request_id = 0
    
    def call_tool(self, method: str, params: Optional[Dict] = None, timeout: int = 60) -> Dict[str, Any]:
        """Call MCP tool method synchronously"""
        
        self.request_id += 1
        
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id
        }
        
        response = requests.post(
            self.base_url,
            json=request_data,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        result = response.json()
        
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        
        return result.get("result", {})
    
    def get_available_strategies(self) -> list:
        """Get list of available trading strategies"""
        
        result = self.call_tool("list_strategies")
        return result.get("strategies", [])
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze symbol with neural forecasting"""
        
        return self.call_tool("quick_analysis", {
            "symbol": symbol,
            "use_gpu": True
        })
    
    def optimize_strategy(self, strategy: str, symbol: str, 
                         parameter_ranges: Dict[str, list]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        
        return self.call_tool("optimize_strategy", {
            "strategy": strategy,
            "symbol": symbol,
            "parameter_ranges": parameter_ranges,
            "use_gpu": True,
            "max_iterations": 100
        })

# Simple integration example
def simple_trading_analysis():
    """Simple trading analysis using sync MCP client"""
    
    client = SyncMCPClient()
    
    # Get available strategies
    strategies = client.get_available_strategies()
    print(f"Available strategies: {[s['name'] for s in strategies]}")
    
    # Analyze multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    analyses = {}
    
    for symbol in symbols:
        try:
            analysis = client.analyze_symbol(symbol)
            analyses[symbol] = analysis
            print(f"{symbol}: {analysis.get('analysis', {}).get('trend', 'unknown')}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    return analyses

# Run simple analysis
# analyses = simple_trading_analysis()
```

## REST API Integration

### FastAPI Server Setup

```python
# REST API server for external integrations
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn

app = FastAPI(
    title="AI News Trading Platform API",
    description="REST API for neural forecasting and trading operations",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ForecastRequest(BaseModel):
    symbol: str
    horizon: int = 30
    confidence_levels: List[int] = [80, 95]
    use_gpu: bool = True

class ForecastResponse(BaseModel):
    symbol: str
    forecast: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class TradeSignalRequest(BaseModel):
    symbols: List[str]
    strategy: str = "momentum_trading_optimized"
    risk_tolerance: float = 0.02

class TradeSignalResponse(BaseModel):
    signals: List[Dict[str, Any]]
    portfolio_impact: Dict[str, Any]

# Initialize MCP client
mcp_client = SyncMCPClient()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MCP server connectivity
        result = mcp_client.call_tool("ping")
        return {
            "status": "healthy",
            "mcp_server": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate neural forecast for symbol"""
    
    try:
        # Call MCP quick analysis
        result = mcp_client.call_tool("quick_analysis", {
            "symbol": request.symbol,
            "use_gpu": request.use_gpu
        })
        
        neural_forecast = result.get("neural_forecast", {})
        
        # Format response
        forecast_data = [
            {
                "date": (datetime.now() + timedelta(days=i)).isoformat(),
                "predicted_value": neural_forecast.get("next_day", 0) * (1 + i * 0.001),
                "confidence": neural_forecast.get("confidence", 0.5)
            }
            for i in range(1, request.horizon + 1)
        ]
        
        return ForecastResponse(
            symbol=request.symbol,
            forecast=forecast_data,
            metadata={
                "model": "nhits",
                "horizon": request.horizon,
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {e}")

@app.post("/signals", response_model=TradeSignalResponse)
async def generate_trade_signals(request: TradeSignalRequest):
    """Generate trading signals for multiple symbols"""
    
    signals = []
    
    for symbol in request.symbols:
        try:
            # Get analysis
            analysis = mcp_client.call_tool("quick_analysis", {
                "symbol": symbol,
                "use_gpu": True
            })
            
            # Simulate trade
            simulation = mcp_client.call_tool("simulate_trade", {
                "strategy": request.strategy,
                "symbol": symbol,
                "action": "buy",
                "use_gpu": True
            })
            
            signals.append({
                "symbol": symbol,
                "action": simulation.get("action", "HOLD"),
                "confidence": analysis.get("neural_forecast", {}).get("confidence", 0.5),
                "expected_return": simulation.get("expected_return", 0),
                "risk_score": simulation.get("risk_metrics", {}).get("var_95", 0)
            })
            
        except Exception as e:
            signals.append({
                "symbol": symbol,
                "action": "ERROR",
                "error": str(e)
            })
    
    # Calculate portfolio impact
    portfolio_impact = {
        "total_symbols": len(request.symbols),
        "buy_signals": len([s for s in signals if s.get("action") == "BUY"]),
        "sell_signals": len([s for s in signals if s.get("action") == "SELL"]),
        "hold_signals": len([s for s in signals if s.get("action") == "HOLD"]),
        "average_confidence": sum([s.get("confidence", 0) for s in signals]) / len(signals) if signals else 0
    }
    
    return TradeSignalResponse(
        signals=signals,
        portfolio_impact=portfolio_impact
    )

@app.get("/strategies")
async def get_strategies():
    """Get available trading strategies"""
    
    try:
        strategies = mcp_client.call_tool("list_strategies")
        return strategies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategies: {e}")

@app.post("/backtest")
async def run_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    background_tasks: BackgroundTasks
):
    """Run backtesting operation"""
    
    # Start backtest in background
    def run_backtest_task():
        try:
            result = mcp_client.call_tool("run_backtest", {
                "strategy": strategy,
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "use_gpu": True
            }, timeout=300)  # 5 minute timeout
            
            # Store result (implement your storage logic)
            print(f"Backtest completed: {result}")
            
        except Exception as e:
            print(f"Backtest failed: {e}")
    
    background_tasks.add_task(run_backtest_task)
    
    return {
        "status": "started",
        "message": "Backtest started in background",
        "strategy": strategy,
        "symbol": symbol
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### REST API Client

```python
# REST API client for external applications
import requests
from typing import List, Dict, Any, Optional

class TradingAPIClient:
    """Client for REST API integration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_forecast(self, symbol: str, horizon: int = 30) -> Dict[str, Any]:
        """Get neural forecast for symbol"""
        
        response = self.session.post(
            f"{self.base_url}/forecast",
            json={
                "symbol": symbol,
                "horizon": horizon,
                "use_gpu": True
            }
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_trade_signals(self, symbols: List[str], strategy: str = "momentum_trading_optimized") -> Dict[str, Any]:
        """Get trading signals for symbols"""
        
        response = self.session.post(
            f"{self.base_url}/signals",
            json={
                "symbols": symbols,
                "strategy": strategy,
                "risk_tolerance": 0.02
            }
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get available trading strategies"""
        
        response = self.session.get(f"{self.base_url}/strategies")
        response.raise_for_status()
        return response.json()
    
    def start_backtest(self, strategy: str, symbol: str, 
                      start_date: str, end_date: str) -> Dict[str, Any]:
        """Start backtesting operation"""
        
        response = self.session.post(
            f"{self.base_url}/backtest",
            params={
                "strategy": strategy,
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        response.raise_for_status()
        return response.json()

# Usage example
def external_application_integration():
    """Example external application using REST API"""
    
    client = TradingAPIClient()
    
    # Get forecast
    forecast = client.get_forecast("AAPL", horizon=14)
    print(f"Forecast for AAPL: {forecast['metadata']}")
    
    # Get trading signals
    signals = client.get_trade_signals(["AAPL", "GOOGL", "MSFT"])
    print(f"Trading signals: {signals['portfolio_impact']}")
    
    # Start backtest
    backtest = client.start_backtest(
        "momentum_trading_optimized",
        "AAPL",
        "2023-01-01",
        "2024-01-01"
    )
    print(f"Backtest started: {backtest['status']}")

# Run example
# external_application_integration()
```

## Plugin System

### Custom Tool Development

```python
# Custom MCP tool development
from typing import Dict, Any, Optional
import asyncio

class CustomMCPTool:
    """Base class for custom MCP tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        raise NotImplementedError
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        raise NotImplementedError

class SentimentAnalysisTool(CustomMCPTool):
    """Custom sentiment analysis tool"""
    
    def __init__(self):
        super().__init__(
            name="custom_sentiment_analysis",
            description="Advanced sentiment analysis with custom models"
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        
        symbol = params.get("symbol")
        sources = params.get("sources", ["news", "social"])
        
        # Your custom sentiment analysis logic here
        sentiment_score = self._analyze_sentiment(symbol, sources)
        
        return {
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "confidence": 0.85,
            "sources_analyzed": sources,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_sentiment(self, symbol: str, sources: list) -> float:
        """Custom sentiment analysis implementation"""
        # Implement your custom logic here
        return 0.72  # Example sentiment score
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Trading symbol to analyze"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data sources for sentiment analysis"
                }
            },
            "required": ["symbol"]
        }

class PortfolioOptimizerTool(CustomMCPTool):
    """Custom portfolio optimization tool"""
    
    def __init__(self):
        super().__init__(
            name="neural_portfolio_optimizer",
            description="Portfolio optimization using neural forecasts"
        )
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio optimization"""
        
        symbols = params.get("symbols", [])
        risk_tolerance = params.get("risk_tolerance", 0.1)
        
        # Get neural forecasts for all symbols
        forecasts = await self._get_neural_forecasts(symbols)
        
        # Optimize portfolio weights
        optimal_weights = self._optimize_weights(forecasts, risk_tolerance)
        
        return {
            "symbols": symbols,
            "optimal_weights": optimal_weights,
            "expected_return": self._calculate_expected_return(forecasts, optimal_weights),
            "risk_score": self._calculate_risk_score(forecasts, optimal_weights),
            "optimization_method": "neural_enhanced_markowitz"
        }
    
    async def _get_neural_forecasts(self, symbols: list) -> Dict[str, Any]:
        """Get neural forecasts for symbols"""
        # Implement neural forecast retrieval
        return {symbol: {"forecast": [1.0] * 30, "confidence": 0.8} for symbol in symbols}
    
    def _optimize_weights(self, forecasts: Dict, risk_tolerance: float) -> Dict[str, float]:
        """Optimize portfolio weights"""
        # Implement your optimization algorithm
        num_symbols = len(forecasts)
        equal_weight = 1.0 / num_symbols
        return {symbol: equal_weight for symbol in forecasts.keys()}
    
    def _calculate_expected_return(self, forecasts: Dict, weights: Dict) -> float:
        """Calculate expected portfolio return"""
        # Implement expected return calculation
        return 0.08  # Example 8% expected return
    
    def _calculate_risk_score(self, forecasts: Dict, weights: Dict) -> float:
        """Calculate portfolio risk score"""
        # Implement risk calculation
        return 0.15  # Example 15% volatility
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols to optimize"
                },
                "risk_tolerance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Risk tolerance (0-1)"
                }
            },
            "required": ["symbols"]
        }

# Plugin registration
class PluginRegistry:
    """Registry for custom tools and plugins"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: CustomMCPTool):
        """Register a custom tool"""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[CustomMCPTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())
    
    async def execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool by name"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return await tool.execute(params)

# Usage example
async def custom_plugin_example():
    """Example of custom plugin usage"""
    
    # Create plugin registry
    registry = PluginRegistry()
    
    # Register custom tools
    registry.register_tool(SentimentAnalysisTool())
    registry.register_tool(PortfolioOptimizerTool())
    
    # Use custom sentiment analysis
    sentiment_result = await registry.execute_tool(
        "custom_sentiment_analysis",
        {"symbol": "AAPL", "sources": ["news", "social"]}
    )
    print(f"Sentiment analysis: {sentiment_result}")
    
    # Use custom portfolio optimizer
    portfolio_result = await registry.execute_tool(
        "neural_portfolio_optimizer",
        {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "risk_tolerance": 0.1
        }
    )
    print(f"Portfolio optimization: {portfolio_result}")

# Run example
# asyncio.run(custom_plugin_example())
```

## WebSocket Integration

### Real-time Data Streaming

```python
# WebSocket server for real-time data streaming
import asyncio
import websockets
import json
from typing import Set, Dict, Any
import threading
import time

class TradingWebSocketServer:
    """WebSocket server for real-time trading data"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.running = False
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister client"""
        self.clients.discard(websocket)
        
        # Remove from subscriptions
        for symbol, subscribers in self.subscriptions.items():
            subscribers.discard(websocket)
        
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def subscribe_to_symbol(self, websocket: websockets.WebSocketServerProtocol, symbol: str):
        """Subscribe client to symbol updates"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        
        self.subscriptions[symbol].add(websocket)
        print(f"Client subscribed to {symbol}")
    
    async def broadcast_to_symbol(self, symbol: str, data: Dict[str, Any]):
        """Broadcast data to symbol subscribers"""
        if symbol in self.subscriptions:
            message = json.dumps({
                "type": "symbol_update",
                "symbol": symbol,
                "data": data,
                "timestamp": time.time()
            })
            
            # Send to all subscribers
            disconnected = set()
            for websocket in self.subscriptions[symbol]:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            for websocket in disconnected:
                self.subscriptions[symbol].discard(websocket)
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                symbol = data.get("symbol")
                if symbol:
                    await self.subscribe_to_symbol(websocket, symbol)
                    
                    # Send confirmation
                    await websocket.send(json.dumps({
                        "type": "subscription_confirmed",
                        "symbol": symbol
                    }))
            
            elif msg_type == "forecast_request":
                symbol = data.get("symbol", "AAPL")
                
                # Generate forecast using MCP client
                mcp_client = SyncMCPClient()
                try:
                    result = mcp_client.call_tool("quick_analysis", {
                        "symbol": symbol,
                        "use_gpu": True
                    })
                    
                    await websocket.send(json.dumps({
                        "type": "forecast_response",
                        "symbol": symbol,
                        "forecast": result.get("neural_forecast", {}),
                        "analysis": result.get("analysis", {})
                    }))
                    
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Forecast failed: {e}"
                    }))
            
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON"
            }))
    
    async def client_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start WebSocket server"""
        print(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.client_handler, self.host, self.port):
            self.running = True
            await asyncio.Future()  # Run forever
    
    def start_background_updates(self):
        """Start background thread for data updates"""
        def update_loop():
            while self.running:
                # Simulate real-time data updates
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
                
                for symbol in symbols:
                    # Generate mock data (replace with real data source)
                    data = {
                        "price": 150.0 + (time.time() % 100),
                        "volume": 1000000,
                        "change": 0.5,
                        "timestamp": time.time()
                    }
                    
                    # Broadcast to subscribers
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast_to_symbol(symbol, data),
                        asyncio.get_event_loop()
                    )
                
                time.sleep(1)  # Update every second
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()

# WebSocket client example
class TradingWebSocketClient:
    """WebSocket client for consuming real-time data"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.uri)
        print("Connected to WebSocket server")
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from WebSocket server")
    
    async def subscribe_to_symbol(self, symbol: str):
        """Subscribe to symbol updates"""
        if self.websocket:
            message = json.dumps({
                "type": "subscribe",
                "symbol": symbol
            })
            await self.websocket.send(message)
    
    async def request_forecast(self, symbol: str):
        """Request neural forecast for symbol"""
        if self.websocket:
            message = json.dumps({
                "type": "forecast_request",
                "symbol": symbol
            })
            await self.websocket.send(message)
    
    async def listen_for_messages(self):
        """Listen for messages from server"""
        if self.websocket:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle message from server"""
        msg_type = data.get("type")
        
        if msg_type == "symbol_update":
            print(f"Symbol update: {data['symbol']} - {data['data']}")
        
        elif msg_type == "forecast_response":
            print(f"Forecast for {data['symbol']}: {data['forecast']}")
        
        elif msg_type == "subscription_confirmed":
            print(f"Subscribed to {data['symbol']}")
        
        elif msg_type == "error":
            print(f"Error: {data['message']}")

# Usage example
async def websocket_example():
    """Example WebSocket client usage"""
    
    client = TradingWebSocketClient()
    
    try:
        await client.connect()
        
        # Subscribe to symbols
        await client.subscribe_to_symbol("AAPL")
        await client.subscribe_to_symbol("GOOGL")
        
        # Request forecasts
        await client.request_forecast("AAPL")
        
        # Listen for messages
        await client.listen_for_messages()
        
    except KeyboardInterrupt:
        print("Stopping client...")
    finally:
        await client.disconnect()

# Start server and client
# Server: asyncio.run(TradingWebSocketServer().start_server())
# Client: asyncio.run(websocket_example())
```

## Integration Best Practices

### Error Handling and Resilience

```python
# Robust integration with error handling
import logging
import time
from functools import wraps
from typing import Callable, Any

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed operations"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

class ResilientMCPClient:
    """MCP client with error handling and resilience"""
    
    def __init__(self, base_url: str = "http://localhost:3000/mcp"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def call_tool_with_retry(self, method: str, params: dict = None) -> dict:
        """Call MCP tool with retry logic"""
        
        client = SyncMCPClient(self.base_url)
        return client.call_tool(method, params)
    
    def safe_neural_forecast(self, symbol: str, horizon: int = 30) -> dict:
        """Generate neural forecast with error handling"""
        
        try:
            result = self.call_tool_with_retry("quick_analysis", {
                "symbol": symbol,
                "use_gpu": True
            })
            
            self.logger.info(f"Neural forecast generated for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Neural forecast failed for {symbol}: {e}")
            
            # Return fallback result
            return {
                "symbol": symbol,
                "error": str(e),
                "fallback": True,
                "neural_forecast": {
                    "next_day": None,
                    "confidence": 0.0,
                    "trend_direction": "unknown"
                }
            }
    
    def bulk_analysis_with_fallback(self, symbols: list) -> dict:
        """Analyze multiple symbols with individual error handling"""
        
        results = {}
        successful = 0
        failed = 0
        
        for symbol in symbols:
            try:
                result = self.safe_neural_forecast(symbol)
                results[symbol] = result
                
                if not result.get("fallback", False):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Critical error for {symbol}: {e}")
                results[symbol] = {"error": str(e), "critical_failure": True}
                failed += 1
        
        self.logger.info(f"Bulk analysis completed: {successful} successful, {failed} failed")
        
        return {
            "results": results,
            "summary": {
                "total": len(symbols),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(symbols) if symbols else 0
            }
        }

# Circuit breaker pattern
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
            
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Usage with circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
resilient_client = ResilientMCPClient()

def protected_forecast(symbol: str) -> dict:
    """Neural forecast with circuit breaker protection"""
    return circuit_breaker.call(
        resilient_client.safe_neural_forecast,
        symbol
    )
```

### Performance Optimization

```python
# Performance optimization techniques
import asyncio
import concurrent.futures
from functools import lru_cache
import time

class PerformanceOptimizedClient:
    """MCP client optimized for performance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    @lru_cache(maxsize=100)
    def cached_strategy_info(self, strategy: str) -> dict:
        """Cache strategy information"""
        client = SyncMCPClient()
        return client.call_tool("get_strategy_info", {"strategy": strategy})
    
    def cache_forecast(self, symbol: str, result: dict):
        """Cache forecast result"""
        self.cache[symbol] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def get_cached_forecast(self, symbol: str) -> dict:
        """Get cached forecast if valid"""
        if symbol in self.cache:
            cached = self.cache[symbol]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["result"]
        return None
    
    def parallel_forecasts(self, symbols: list) -> dict:
        """Generate forecasts in parallel"""
        
        def generate_forecast(symbol):
            # Check cache first
            cached = self.get_cached_forecast(symbol)
            if cached:
                return symbol, cached
            
            # Generate new forecast
            client = SyncMCPClient()
            result = client.call_tool("quick_analysis", {
                "symbol": symbol,
                "use_gpu": True
            })
            
            # Cache result
            self.cache_forecast(symbol, result)
            
            return symbol, result
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(generate_forecast, symbol)
                for symbol in symbols
            ]
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    symbol, result = future.result(timeout=30)
                    results[symbol] = result
                except Exception as e:
                    print(f"Forecast failed: {e}")
            
            return results
    
    async def async_batch_analysis(self, symbols: list) -> dict:
        """Asynchronous batch analysis"""
        
        async def analyze_symbol(symbol):
            client = MCPClient()
            async with client:
                return await client.neural_forecast(symbol)
        
        # Run analyses concurrently
        tasks = [analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analysis_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                analysis_results[symbol] = {"error": str(result)}
            else:
                analysis_results[symbol] = result
        
        return analysis_results

# Connection pooling for high-volume usage
class ConnectionPooledClient:
    """MCP client with connection pooling"""
    
    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.available_clients = []
        self.in_use_clients = set()
        self.lock = asyncio.Lock()
    
    async def get_client(self) -> MCPClient:
        """Get client from pool"""
        async with self.lock:
            if self.available_clients:
                client = self.available_clients.pop()
                self.in_use_clients.add(client)
                return client
            
            if len(self.in_use_clients) < self.pool_size:
                client = MCPClient()
                await client.__aenter__()
                self.in_use_clients.add(client)
                return client
        
        # Pool exhausted, wait for available client
        while True:
            await asyncio.sleep(0.1)
            async with self.lock:
                if self.available_clients:
                    client = self.available_clients.pop()
                    self.in_use_clients.add(client)
                    return client
    
    async def return_client(self, client: MCPClient):
        """Return client to pool"""
        async with self.lock:
            self.in_use_clients.discard(client)
            self.available_clients.append(client)
    
    async def close_all(self):
        """Close all connections"""
        async with self.lock:
            for client in self.available_clients + list(self.in_use_clients):
                await client.__aexit__(None, None, None)
            
            self.available_clients.clear()
            self.in_use_clients.clear()

# Usage example
async def high_performance_trading_system():
    """Example high-performance trading system"""
    
    pool = ConnectionPooledClient(pool_size=10)
    optimized = PerformanceOptimizedClient()
    
    try:
        # Parallel forecast generation
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        # Method 1: Thread-based parallel processing
        parallel_results = optimized.parallel_forecasts(symbols)
        print(f"Parallel forecasts completed: {len(parallel_results)} symbols")
        
        # Method 2: Async batch processing
        async_results = await optimized.async_batch_analysis(symbols)
        print(f"Async analysis completed: {len(async_results)} symbols")
        
        # Method 3: Connection pooling for high-volume operations
        tasks = []
        for symbol in symbols * 3:  # Process each symbol 3 times
            client = await pool.get_client()
            task = client.neural_forecast(symbol)
            tasks.append((task, client))
        
        # Execute all tasks
        for task, client in tasks:
            try:
                result = await task
                print(f"Pooled forecast for {result.get('symbol', 'unknown')}")
            finally:
                await pool.return_client(client)
    
    finally:
        await pool.close_all()

# Run performance example
# asyncio.run(high_performance_trading_system())
```

## Testing and Validation

### Integration Testing

```python
# Integration testing framework
import unittest
import asyncio
from unittest.mock import Mock, patch

class IntegrationTestSuite(unittest.TestCase):
    """Integration test suite for AI News Trading Platform"""
    
    def setUp(self):
        """Set up test environment"""
        self.mcp_client = SyncMCPClient()
        self.test_symbols = ["AAPL", "GOOGL", "MSFT"]
    
    def test_mcp_server_connectivity(self):
        """Test MCP server connectivity"""
        try:
            result = self.mcp_client.call_tool("ping")
            self.assertIn("status", result)
            self.assertEqual(result["status"], "ok")
        except Exception as e:
            self.fail(f"MCP server connectivity failed: {e}")
    
    def test_neural_forecasting_basic(self):
        """Test basic neural forecasting functionality"""
        for symbol in self.test_symbols:
            with self.subTest(symbol=symbol):
                result = self.mcp_client.call_tool("quick_analysis", {
                    "symbol": symbol,
                    "use_gpu": False  # Use CPU for testing
                })
                
                self.assertIn("neural_forecast", result)
                self.assertIn("analysis", result)
                
                neural_forecast = result["neural_forecast"]
                self.assertIn("confidence", neural_forecast)
                self.assertIsInstance(neural_forecast["confidence"], (int, float))
                self.assertGreaterEqual(neural_forecast["confidence"], 0)
                self.assertLessEqual(neural_forecast["confidence"], 1)
    
    def test_strategy_listing(self):
        """Test strategy listing functionality"""
        result = self.mcp_client.call_tool("list_strategies")
        
        self.assertIn("strategies", result)
        strategies = result["strategies"]
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        
        # Validate strategy structure
        for strategy in strategies:
            self.assertIn("name", strategy)
            self.assertIn("description", strategy)
            self.assertIn("gpu_accelerated", strategy)
    
    def test_trade_simulation(self):
        """Test trade simulation"""
        # Get available strategies first
        strategies_result = self.mcp_client.call_tool("list_strategies")
        strategies = strategies_result["strategies"]
        
        if strategies:
            strategy_name = strategies[0]["name"]
            
            for symbol in self.test_symbols[:2]:  # Test with 2 symbols
                with self.subTest(symbol=symbol, strategy=strategy_name):
                    result = self.mcp_client.call_tool("simulate_trade", {
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "action": "buy",
                        "use_gpu": False
                    })
                    
                    self.assertIn("action", result)
                    self.assertIn("expected_return", result)
                    self.assertIn("risk_metrics", result)
    
    def test_portfolio_status(self):
        """Test portfolio status retrieval"""
        result = self.mcp_client.call_tool("get_portfolio_status", {
            "include_analytics": True
        })
        
        self.assertIn("portfolio", result)
        portfolio = result["portfolio"]
        
        self.assertIn("total_value", portfolio)
        self.assertIn("cash", portfolio)
        self.assertIn("positions", portfolio)
        
        # Test analytics if available
        if "analytics" in result:
            analytics = result["analytics"]
            self.assertIn("total_return", analytics)
            self.assertIn("sharpe_ratio", analytics)
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        
        # Test invalid symbol
        with self.assertRaises(Exception):
            self.mcp_client.call_tool("quick_analysis", {
                "symbol": "INVALID_SYMBOL_12345"
            })
        
        # Test invalid strategy
        with self.assertRaises(Exception):
            self.mcp_client.call_tool("get_strategy_info", {
                "strategy": "non_existent_strategy"
            })
        
        # Test invalid method
        with self.assertRaises(Exception):
            self.mcp_client.call_tool("non_existent_method")

class PerformanceTestSuite(unittest.TestCase):
    """Performance test suite"""
    
    def setUp(self):
        self.mcp_client = SyncMCPClient()
        self.performance_thresholds = {
            "quick_analysis": 5.0,  # 5 seconds max
            "simulate_trade": 3.0,  # 3 seconds max
            "portfolio_status": 2.0,  # 2 seconds max
        }
    
    def test_neural_forecast_performance(self):
        """Test neural forecasting performance"""
        
        start_time = time.time()
        
        result = self.mcp_client.call_tool("quick_analysis", {
            "symbol": "AAPL",
            "use_gpu": True
        })
        
        duration = time.time() - start_time
        
        self.assertLess(
            duration, 
            self.performance_thresholds["quick_analysis"],
            f"Neural forecast took {duration:.2f}s, expected < {self.performance_thresholds['quick_analysis']}s"
        )
        
        # Check if GPU acceleration is working
        processing_time = result.get("processing_time_ms", 0)
        if processing_time > 0:
            self.assertLess(processing_time, 1000, "GPU acceleration should provide sub-second processing")
    
    def test_concurrent_requests(self):
        """Test system performance under concurrent load"""
        
        def make_request(symbol):
            start_time = time.time()
            result = self.mcp_client.call_tool("quick_analysis", {
                "symbol": symbol,
                "use_gpu": True
            })
            duration = time.time() - start_time
            return duration, result
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, symbol) for symbol in symbols]
            results = [future.result() for future in futures]
        
        # Check that all requests completed successfully
        self.assertEqual(len(results), len(symbols))
        
        # Check average performance
        avg_duration = sum(duration for duration, _ in results) / len(results)
        self.assertLess(avg_duration, 10.0, f"Average concurrent request time: {avg_duration:.2f}s")

# Run tests
if __name__ == "__main__":
    # Run integration tests
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    performance_suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceTestSuite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    
    print("Running Integration Tests...")
    integration_result = runner.run(integration_suite)
    
    print("\nRunning Performance Tests...")
    performance_result = runner.run(performance_suite)
    
    # Summary
    total_tests = integration_result.testsRun + performance_result.testsRun
    total_failures = len(integration_result.failures) + len(performance_result.failures)
    total_errors = len(integration_result.errors) + len(performance_result.errors)
    
    print(f"\nTest Summary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
```

## Best Practices Summary

### Integration Guidelines

1. **Error Handling**:
   - Always implement retry logic for network calls
   - Use circuit breakers for external dependencies
   - Provide meaningful fallback responses

2. **Performance**:
   - Use connection pooling for high-volume applications
   - Implement caching for frequently accessed data
   - Leverage parallel processing for bulk operations

3. **Security**:
   - Validate all inputs before processing
   - Use secure connections (HTTPS/WSS)
   - Implement proper authentication and authorization

4. **Monitoring**:
   - Log all integration points
   - Monitor performance metrics
   - Set up alerts for failures

5. **Testing**:
   - Test both success and failure scenarios
   - Include performance testing
   - Test with realistic data volumes

For more information, see:
- [API Documentation](../api/)
- [Configuration Guide](../configuration/system_config.md)
- [Deployment Guide](../guides/deployment.md)
- [Examples](../examples/)