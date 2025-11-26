# MCP Implementation Guide for AI News Trading Platform

## Quick Start

This guide provides practical implementation examples for building MCP servers for the AI News Trading platform.

## 1. Basic MCP Server Setup

### Python Implementation

```python
# src/mcp_servers/basic_server.py
from mcp import Server, Tool, Resource
import asyncio
import json

class BasicTradingMCPServer(Server):
    def __init__(self):
        super().__init__(
            name="basic-trading-server",
            version="1.0.0"
        )
        
    async def initialize(self):
        """Initialize server resources"""
        # Connect to databases, load models, etc.
        self.db = await self._connect_database()
        self.market_api = await self._connect_market_api()
        
    @Tool(
        name="get_market_status",
        description="Get current market status and trading hours"
    )
    async def get_market_status(self) -> dict:
        """Simple tool example"""
        return {
            "status": "open",
            "next_close": "16:00 EST",
            "trading_enabled": True
        }

# Run the server
if __name__ == "__main__":
    server = BasicTradingMCPServer()
    server.run_stdio()  # Run with stdio transport
```

### TypeScript Implementation

```typescript
// src/mcp_servers/basic_server.ts
import { Server, Tool, Resource } from '@modelcontextprotocol/sdk';

class BasicTradingMCPServer extends Server {
  constructor() {
    super({
      name: 'basic-trading-server',
      version: '1.0.0'
    });
  }
  
  async initialize(): Promise<void> {
    // Initialize resources
    await this.connectDatabase();
    await this.connectMarketAPI();
  }
  
  @Tool({
    name: 'get_market_status',
    description: 'Get current market status and trading hours'
  })
  async getMarketStatus(): Promise<object> {
    return {
      status: 'open',
      next_close: '16:00 EST',
      trading_enabled: true
    };
  }
}

// Run the server
const server = new BasicTradingMCPServer();
server.runStdio();
```

## 2. Advanced Trading MCP Server

```python
# src/mcp_servers/advanced_trading_server.py
from mcp import Server, Tool, Resource
from typing import List, Dict, Optional
import numpy as np
import torch
from dataclasses import dataclass
import asyncio
from datetime import datetime

@dataclass
class TradeSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    quantity: int
    reasoning: str

class AdvancedTradingMCPServer(Server):
    def __init__(self, config: dict):
        super().__init__(
            name="advanced-trading-server",
            version="2.0.0",
            capabilities={
                "tools": True,
                "resources": True,
                "streaming": True
            }
        )
        self.config = config
        self.models = {}
        self.active_streams = {}
        
    async def initialize(self):
        """Initialize ML models and connections"""
        # Load GPU models
        self.models['sentiment'] = await self._load_model('sentiment_analyzer')
        self.models['pattern'] = await self._load_model('pattern_detector')
        
        # Initialize connections
        self.news_feed = await self._connect_news_feed()
        self.market_data = await self._connect_market_data()
        self.broker = await self._connect_broker()
    
    @Tool(
        name="analyze_news_and_trade",
        description="Analyze news and generate trading signals",
        input_schema={
            "type": "object",
            "properties": {
                "news_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "published_at": {"type": "string"},
                            "symbols": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                    "default": "moderate"
                },
                "max_position_size": {
                    "type": "number",
                    "default": 10000
                }
            },
            "required": ["news_items"]
        }
    )
    async def analyze_news_and_trade(self, news_items: List[Dict], 
                                   risk_level: str = "moderate",
                                   max_position_size: float = 10000) -> Dict:
        """Complex tool that analyzes news and generates trading signals"""
        
        # 1. Batch process news through sentiment model
        sentiments = await self._analyze_sentiments_batch(news_items)
        
        # 2. Get affected symbols and their current prices
        affected_symbols = self._extract_affected_symbols(news_items)
        market_data = await self._get_market_data_batch(affected_symbols)
        
        # 3. Detect trading patterns
        patterns = await self._detect_patterns(market_data)
        
        # 4. Generate trading signals
        signals = self._generate_signals(
            sentiments, patterns, market_data, risk_level
        )
        
        # 5. Filter by position size
        filtered_signals = [
            s for s in signals 
            if s.quantity * market_data[s.symbol]['price'] <= max_position_size
        ]
        
        # 6. Prepare response
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "news_analyzed": len(news_items),
            "symbols_affected": affected_symbols,
            "signals": [self._signal_to_dict(s) for s in filtered_signals],
            "risk_metrics": self._calculate_risk_metrics(filtered_signals, market_data)
        }
    
    @Tool(
        name="execute_trading_strategy",
        description="Execute a complete trading strategy with risk management",
        input_schema={
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "enum": ["momentum", "mean_reversion", "arbitrage", "news_based"]
                },
                "capital": {"type": "number"},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "duration_minutes": {"type": "integer", "default": 60}
            },
            "required": ["strategy_name", "capital"]
        }
    )
    async def execute_trading_strategy(self, strategy_name: str, capital: float,
                                     symbols: Optional[List[str]] = None,
                                     duration_minutes: int = 60) -> Dict:
        """Execute a trading strategy with real-time monitoring"""
        
        strategy_id = f"{strategy_name}_{datetime.utcnow().timestamp()}"
        
        # Create strategy instance
        strategy = self._create_strategy(strategy_name, capital, symbols)
        
        # Start execution in background
        execution_task = asyncio.create_task(
            self._run_strategy(strategy_id, strategy, duration_minutes)
        )
        
        # Store for monitoring
        self.active_strategies[strategy_id] = {
            "strategy": strategy,
            "task": execution_task,
            "start_time": datetime.utcnow(),
            "capital": capital
        }
        
        return {
            "strategy_id": strategy_id,
            "status": "running",
            "monitoring_url": f"/strategies/{strategy_id}/status",
            "stop_command": f"stop_strategy --id {strategy_id}"
        }
    
    @Resource(
        uri_template="trading://positions/{account_id}",
        mime_type="application/json",
        description="Real-time position information"
    )
    async def get_positions_resource(self, account_id: str):
        """Provide positions as MCP resource"""
        positions = await self.broker.get_positions(account_id)
        
        # Add real-time P&L
        for position in positions:
            current_price = await self.market_data.get_price(position['symbol'])
            position['current_price'] = current_price
            position['unrealized_pnl'] = (
                (current_price - position['avg_price']) * position['quantity']
            )
        
        return {
            "uri": f"trading://positions/{account_id}",
            "content": {
                "account_id": account_id,
                "positions": positions,
                "total_value": sum(p['current_price'] * p['quantity'] for p in positions),
                "total_pnl": sum(p['unrealized_pnl'] for p in positions),
                "updated_at": datetime.utcnow().isoformat()
            }
        }
    
    async def _analyze_sentiments_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Batch process news through GPU sentiment model"""
        texts = [f"{item['title']} {item['content']}" for item in news_items]
        
        # Prepare for GPU processing
        with torch.cuda.device(0):
            # Tokenize and pad
            inputs = self.models['sentiment'].tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to('cuda')
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['sentiment'].model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to CPU for processing
            sentiments = predictions.cpu().numpy()
        
        # Format results
        return [
            {
                "positive": float(sent[2]),
                "neutral": float(sent[1]),
                "negative": float(sent[0]),
                "overall": "positive" if sent[2] > 0.6 else "negative" if sent[0] > 0.6 else "neutral"
            }
            for sent in sentiments
        ]
    
    def _generate_signals(self, sentiments: List[Dict], patterns: List[Dict],
                         market_data: Dict, risk_level: str) -> List[TradeSignal]:
        """Generate trading signals based on analysis"""
        signals = []
        
        risk_multipliers = {
            "conservative": 0.5,
            "moderate": 1.0,
            "aggressive": 2.0
        }
        risk_mult = risk_multipliers[risk_level]
        
        for symbol, data in market_data.items():
            # Aggregate sentiment for this symbol
            symbol_sentiment = self._aggregate_sentiment_for_symbol(
                symbol, sentiments
            )
            
            # Check patterns
            symbol_patterns = [p for p in patterns if p['symbol'] == symbol]
            
            # Generate signal
            if symbol_sentiment['positive'] > 0.7 and any(
                p['pattern'] == 'breakout' for p in symbol_patterns
            ):
                quantity = int(1000 * risk_mult * symbol_sentiment['positive'])
                signals.append(TradeSignal(
                    symbol=symbol,
                    action="BUY",
                    confidence=symbol_sentiment['positive'],
                    quantity=quantity,
                    reasoning="Strong positive sentiment with breakout pattern"
                ))
            elif symbol_sentiment['negative'] > 0.7:
                quantity = int(500 * risk_mult)
                signals.append(TradeSignal(
                    symbol=symbol,
                    action="SELL",
                    confidence=symbol_sentiment['negative'],
                    quantity=quantity,
                    reasoning="Strong negative sentiment detected"
                ))
        
        return signals
```

## 3. Streaming MCP Server

```javascript
// src/mcp_servers/streaming_server.js
import { Server, Tool, Resource } from '@modelcontextprotocol/sdk';
import WebSocket from 'ws';

class StreamingMarketDataServer extends Server {
  constructor() {
    super({
      name: 'streaming-market-server',
      version: '1.0.0',
      capabilities: {
        streaming: true,
        tools: true,
        resources: true
      }
    });
    
    this.activeStreams = new Map();
    this.marketConnections = new Map();
  }
  
  @Tool({
    name: 'start_market_stream',
    description: 'Start streaming real-time market data',
    inputSchema: {
      type: 'object',
      properties: {
        symbols: {
          type: 'array',
          items: { type: 'string' }
        },
        data_types: {
          type: 'array',
          items: {
            type: 'string',
            enum: ['trades', 'quotes', 'bars', 'news']
          }
        },
        filters: {
          type: 'object',
          properties: {
            min_volume: { type: 'number' },
            min_price_change: { type: 'number' }
          }
        }
      },
      required: ['symbols', 'data_types']
    }
  })
  async startMarketStream({ symbols, data_types, filters = {} }) {
    const streamId = crypto.randomUUID();
    
    // Create WebSocket connection to market data provider
    const ws = new WebSocket(process.env.MARKET_DATA_WS_URL);
    
    // Set up stream configuration
    const streamConfig = {
      symbols,
      data_types,
      filters,
      startTime: new Date(),
      messageCount: 0
    };
    
    ws.on('open', () => {
      // Subscribe to requested data
      ws.send(JSON.stringify({
        action: 'subscribe',
        symbols,
        types: data_types
      }));
    });
    
    // Create readable stream for MCP
    const stream = new ReadableStream({
      start(controller) {
        ws.on('message', (data) => {
          const message = JSON.parse(data);
          
          // Apply filters
          if (this.shouldFilterMessage(message, filters)) {
            return;
          }
          
          // Enrich message
          const enrichedMessage = {
            ...message,
            stream_id: streamId,
            received_at: new Date().toISOString(),
            sequence: ++streamConfig.messageCount
          };
          
          // Send to MCP client
          controller.enqueue(enrichedMessage);
        });
        
        ws.on('error', (error) => {
          controller.error(error);
        });
        
        ws.on('close', () => {
          controller.close();
        });
      },
      
      cancel() {
        // Clean up when stream is cancelled
        ws.close();
        this.activeStreams.delete(streamId);
      }
    });
    
    // Store stream reference
    this.activeStreams.set(streamId, {
      config: streamConfig,
      websocket: ws,
      stream
    });
    
    return {
      stream_id: streamId,
      stream,
      config: streamConfig
    };
  }
  
  @Tool({
    name: 'stop_market_stream',
    description: 'Stop an active market data stream',
    inputSchema: {
      type: 'object',
      properties: {
        stream_id: { type: 'string' }
      },
      required: ['stream_id']
    }
  })
  async stopMarketStream({ stream_id }) {
    const streamInfo = this.activeStreams.get(stream_id);
    
    if (!streamInfo) {
      throw new Error(`Stream ${stream_id} not found`);
    }
    
    // Close WebSocket
    streamInfo.websocket.close();
    
    // Remove from active streams
    this.activeStreams.delete(stream_id);
    
    return {
      stream_id,
      status: 'stopped',
      duration_seconds: (Date.now() - streamInfo.config.startTime) / 1000,
      messages_processed: streamInfo.config.messageCount
    };
  }
  
  shouldFilterMessage(message, filters) {
    if (filters.min_volume && message.volume < filters.min_volume) {
      return true;
    }
    
    if (filters.min_price_change) {
      const changePercent = Math.abs(message.change_percent || 0);
      if (changePercent < filters.min_price_change) {
        return true;
      }
    }
    
    return false;
  }
}

// Run the server
const server = new StreamingMarketDataServer();
server.runWebSocket({ port: 8080 });
```

## 4. GPU-Accelerated Model Server

```python
# src/mcp_servers/gpu_model_server.py
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import List, Dict
import asyncio

class GPUModelMCPServer(Server):
    def __init__(self, gpu_devices: List[int] = [0, 1]):
        super().__init__(
            name="gpu-model-server",
            version="1.0.0"
        )
        self.gpu_devices = gpu_devices
        self.models = {}
        self.model_queues = {}
        
    async def initialize(self):
        """Initialize GPU models with load balancing"""
        # Load models across multiple GPUs
        for i, device_id in enumerate(self.gpu_devices):
            with torch.cuda.device(device_id):
                # Load sentiment model
                sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert"
                ).to(f'cuda:{device_id}')
                sentiment_model.eval()
                
                # Load pattern detection model
                pattern_model = torch.jit.load(
                    "models/chart_pattern_detector.pt"
                ).to(f'cuda:{device_id}')
                
                self.models[f'sentiment_{device_id}'] = {
                    'model': sentiment_model,
                    'tokenizer': AutoTokenizer.from_pretrained("ProsusAI/finbert"),
                    'device': device_id,
                    'usage_count': 0
                }
                
                self.models[f'pattern_{device_id}'] = {
                    'model': pattern_model,
                    'device': device_id,
                    'usage_count': 0
                }
    
    @Tool(
        name="batch_sentiment_analysis",
        description="GPU-accelerated batch sentiment analysis",
        input_schema={
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 1000
                },
                "return_probabilities": {
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["texts"]
        }
    )
    async def batch_sentiment_analysis(self, texts: List[str], 
                                     return_probabilities: bool = False) -> Dict:
        """Analyze sentiment using GPU acceleration"""
        
        # Select least used GPU
        model_key = self._select_gpu_model('sentiment')
        model_info = self.models[model_key]
        model_info['usage_count'] += 1
        
        try:
            # Move to selected GPU
            with torch.cuda.device(model_info['device']):
                # Tokenize batch
                inputs = model_info['tokenizer'](
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(f"cuda:{model_info['device']}")
                
                # Run inference
                with torch.no_grad():
                    with torch.cuda.amp.autocast():  # Mixed precision
                        outputs = model_info['model'](**inputs)
                        logits = outputs.logits
                        
                        if return_probabilities:
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            results = probs.cpu().numpy()
                        else:
                            predictions = torch.argmax(logits, dim=-1)
                            results = predictions.cpu().numpy()
                
                # Format results
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
                if return_probabilities:
                    formatted_results = [
                        {
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'probabilities': {
                                'negative': float(probs[0]),
                                'neutral': float(probs[1]),
                                'positive': float(probs[2])
                            },
                            'sentiment': sentiment_map[np.argmax(probs)]
                        }
                        for text, probs in zip(texts, results)
                    ]
                else:
                    formatted_results = [
                        {
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'sentiment': sentiment_map[int(pred)]
                        }
                        for text, pred in zip(texts, results)
                    ]
                
                return {
                    'results': formatted_results,
                    'batch_size': len(texts),
                    'gpu_device': model_info['device'],
                    'processing_time_ms': self._get_processing_time()
                }
                
        finally:
            model_info['usage_count'] -= 1
    
    @Tool(
        name="detect_chart_patterns",
        description="Detect chart patterns using GPU-accelerated CV model",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {
                    "type": "string",
                    "enum": ["1m", "5m", "15m", "1h", "1d"]
                },
                "lookback_periods": {
                    "type": "integer",
                    "default": 100
                }
            },
            "required": ["symbol", "timeframe"]
        }
    )
    async def detect_chart_patterns(self, symbol: str, timeframe: str,
                                  lookback_periods: int = 100) -> Dict:
        """Detect chart patterns using computer vision on GPU"""
        
        # Get price data
        price_data = await self.market_data.get_bars(
            symbol, timeframe, lookback_periods
        )
        
        # Convert to tensor format
        prices = torch.tensor([
            [bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']]
            for bar in price_data
        ], dtype=torch.float32)
        
        # Select GPU and run pattern detection
        model_key = self._select_gpu_model('pattern')
        model_info = self.models[model_key]
        
        with torch.cuda.device(model_info['device']):
            prices_gpu = prices.to(f"cuda:{model_info['device']}")
            
            with torch.no_grad():
                # Run pattern detection model
                patterns = model_info['model'](prices_gpu.unsqueeze(0))
                pattern_probs = torch.sigmoid(patterns).cpu().numpy()[0]
        
        # Map to pattern names
        pattern_names = [
            'head_and_shoulders', 'double_top', 'double_bottom',
            'triangle', 'flag', 'wedge', 'channel'
        ]
        
        detected_patterns = [
            {
                'pattern': name,
                'confidence': float(prob),
                'timeframe': timeframe
            }
            for name, prob in zip(pattern_names, pattern_probs)
            if prob > 0.7  # Confidence threshold
        ]
        
        return {
            'symbol': symbol,
            'patterns': detected_patterns,
            'analyzed_periods': lookback_periods,
            'gpu_device': model_info['device']
        }
    
    def _select_gpu_model(self, model_type: str) -> str:
        """Select least loaded GPU for model execution"""
        relevant_models = {
            k: v for k, v in self.models.items() 
            if k.startswith(model_type)
        }
        
        # Select model with lowest usage count
        selected = min(relevant_models.items(), 
                      key=lambda x: x[1]['usage_count'])
        
        return selected[0]
```

## 5. Client Integration Examples

### Python Client

```python
# src/clients/python_mcp_client.py
from mcp import Client
import asyncio

class TradingMCPClient:
    def __init__(self):
        self.servers = {}
        
    async def connect(self):
        """Connect to all MCP servers"""
        # Connect to news analyzer
        self.servers['news'] = await Client.connect(
            transport='stdio',
            command=['python', '-m', 'mcp_servers.news_analyzer']
        )
        
        # Connect to market data (WebSocket)
        self.servers['market'] = await Client.connect(
            transport='websocket',
            url='ws://localhost:8080/mcp'
        )
        
        # Connect to trade executor (HTTP+SSE)
        self.servers['trading'] = await Client.connect(
            transport='http',
            url='https://trading-api.example.com/mcp',
            headers={'Authorization': f'Bearer {API_TOKEN}'}
        )
    
    async def analyze_and_trade(self, news_items):
        """Complete trading workflow"""
        # 1. Analyze news
        analysis = await self.servers['news'].call_tool(
            'analyze_news_batch',
            {
                'news_items': news_items,
                'symbols': ['AAPL', 'GOOGL', 'MSFT']
            }
        )
        
        # 2. Get market data for affected symbols
        market_data = await self.servers['market'].call_tool(
            'get_real_time_quote',
            {
                'symbol': analysis['affected_symbols'][0],
                'include_depth': True
            }
        )
        
        # 3. Execute trades based on signals
        for signal in analysis['signals']:
            if signal['confidence'] > 0.8:
                order = await self.servers['trading'].call_tool(
                    'place_order',
                    {
                        'symbol': signal['symbol'],
                        'side': signal['action'],
                        'quantity': signal['quantity'],
                        'order_type': 'LIMIT',
                        'limit_price': market_data['ask'] if signal['action'] == 'BUY' else market_data['bid']
                    }
                )
                print(f"Order placed: {order}")

# Usage
async def main():
    client = TradingMCPClient()
    await client.connect()
    
    # Example news
    news = [
        {
            'title': 'Apple announces record profits',
            'content': 'Apple Inc. reported better than expected...',
            'symbols': ['AAPL']
        }
    ]
    
    await client.analyze_and_trade(news)

if __name__ == "__main__":
    asyncio.run(main())
```

### TypeScript Client

```typescript
// src/clients/typescript_mcp_client.ts
import { Client } from '@modelcontextprotocol/client';

class TradingMCPClient {
  private servers: Map<string, Client> = new Map();
  
  async connect(): Promise<void> {
    // Connect to servers
    this.servers.set('news', await Client.connect({
      transport: 'stdio',
      command: ['node', 'dist/news-analyzer.js']
    }));
    
    this.servers.set('market', await Client.connect({
      transport: 'websocket',
      url: 'ws://localhost:8080/mcp'
    }));
    
    this.servers.set('trading', await Client.connect({
      transport: 'http',
      url: 'https://trading-api.example.com/mcp',
      headers: {
        'Authorization': `Bearer ${process.env.API_TOKEN}`
      }
    }));
  }
  
  async executeStrategy(strategyName: string, capital: number): Promise<void> {
    const tradingServer = this.servers.get('trading');
    
    // Start strategy
    const result = await tradingServer.callTool('execute_trading_strategy', {
      strategy_name: strategyName,
      capital: capital,
      symbols: ['AAPL', 'GOOGL', 'MSFT'],
      duration_minutes: 120
    });
    
    console.log(`Strategy started: ${result.strategy_id}`);
    
    // Monitor strategy
    const positions = await tradingServer.getResource(
      `trading://positions/${result.strategy_id}`
    );
    
    console.log('Current positions:', positions);
  }
}

// Usage
(async () => {
  const client = new TradingMCPClient();
  await client.connect();
  await client.executeStrategy('momentum', 100000);
})();
```

## 6. Testing MCP Servers

```python
# tests/test_mcp_server.py
import pytest
from mcp.testing import MCPTestClient
from src.mcp_servers.trading_server import TradingMCPServer

@pytest.fixture
async def mcp_client():
    """Create test client"""
    server = TradingMCPServer(test_mode=True)
    client = MCPTestClient(server)
    await client.initialize()
    yield client
    await client.cleanup()

@pytest.mark.asyncio
async def test_place_order(mcp_client):
    """Test order placement"""
    # Call tool
    result = await mcp_client.call_tool('place_order', {
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 100,
        'order_type': 'MARKET'
    })
    
    # Verify
    assert result['status'] == 'ACCEPTED'
    assert 'order_id' in result

@pytest.mark.asyncio
async def test_rate_limiting(mcp_client):
    """Test rate limiting"""
    # Make many rapid requests
    tasks = []
    for i in range(20):
        tasks.append(
            mcp_client.call_tool('get_quote', {'symbol': 'AAPL'})
        )
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Should have some rate limit errors
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) > 0
    assert any('rate limit' in str(e).lower() for e in errors)

@pytest.mark.asyncio
async def test_streaming(mcp_client):
    """Test streaming functionality"""
    # Start stream
    stream_result = await mcp_client.call_tool('start_market_stream', {
        'symbols': ['AAPL'],
        'data_types': ['quotes']
    })
    
    # Collect messages
    messages = []
    async for message in stream_result['stream']:
        messages.append(message)
        if len(messages) >= 10:
            break
    
    # Verify
    assert len(messages) == 10
    assert all('symbol' in m for m in messages)
```

## 7. Deployment Configuration

```yaml
# deployment/docker-compose.yml
version: '3.8'

services:
  mcp-gateway:
    image: trading-platform/mcp-gateway:latest
    ports:
      - "8443:8443"
    environment:
      - LOG_LEVEL=INFO
      - AUTH_PROVIDER=oauth2
    volumes:
      - ./config:/config
      - ./certs:/certs
    depends_on:
      - redis
      - postgres
  
  news-analyzer:
    image: trading-platform/news-analyzer-mcp:latest
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/models
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - model-cache:/models
  
  market-data:
    image: trading-platform/market-data-mcp:latest
    deploy:
      replicas: 2
    environment:
      - MARKET_PROVIDER=polygon
      - REDIS_URL=redis://redis:6379
  
  trade-executor:
    image: trading-platform/trade-executor-mcp:latest
    environment:
      - BROKER_API=${BROKER_API_URL}
      - RISK_LIMITS_FILE=/config/risk_limits.json
    volumes:
      - ./config:/config
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=trading
      - POSTGRES_USER=trading
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  model-cache:
  redis-data:
  postgres-data:
```

## Conclusion

This implementation guide provides practical examples for building MCP servers for the AI News Trading platform. Key implementation considerations:

1. **Transport Selection**: Choose appropriate transport for use case
2. **GPU Optimization**: Implement batching and load balancing
3. **Error Handling**: Comprehensive error handling and recovery
4. **Security**: Authentication and rate limiting from the start
5. **Testing**: Thorough testing including edge cases
6. **Monitoring**: Built-in metrics and logging

The modular architecture allows for easy extension and maintenance while providing robust trading capabilities through the MCP protocol.