"""
Base templates for E2B sandboxes
"""

from typing import Dict, Any
from .models import (
    TemplateConfig, 
    TemplateType, 
    TemplateMetadata,
    TemplateRequirements,
    TemplateFiles,
    RuntimeEnvironment
)


class BaseTemplates:
    """Collection of base templates"""
    
    @staticmethod
    def python_base() -> TemplateConfig:
        """Base Python template with common dependencies"""
        return TemplateConfig(
            template_type=TemplateType.PYTHON_BASE,
            metadata=TemplateMetadata(
                name="Python Base Environment",
                description="Base Python environment with common data science and trading libraries",
                version="1.0.0",
                tags=["python", "base", "data-science"],
                category="base"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.PYTHON_3_10,
                cpu_cores=2,
                memory_mb=1024,
                python_packages=[
                    "numpy>=1.24.0",
                    "pandas>=2.0.0",
                    "scipy>=1.10.0",
                    "scikit-learn>=1.3.0",
                    "matplotlib>=3.7.0",
                    "seaborn>=0.12.0",
                    "requests>=2.31.0",
                    "python-dotenv>=1.0.0",
                    "pydantic>=2.0.0",
                    "fastapi>=0.100.0",
                    "yfinance>=0.2.0",
                    "ta>=0.10.0",
                    "asyncio",
                    "aiohttp>=3.8.0"
                ],
                env_vars={
                    "PYTHONUNBUFFERED": "1",
                    "ENVIRONMENT": "sandbox"
                }
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env python3
"""Base Python template main script"""

import os
import sys
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    logger.info("Python base template initialized")
    
    # Get configuration from environment or arguments
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            logger.warning("Failed to parse config from arguments")
    
    logger.info(f"Configuration: {config}")
    
    # Your code here
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "message": "Base template executed successfully"
    }
    
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
                modules={
                    "utils.py": '''"""Utility functions"""

def load_config(path):
    """Load configuration from file"""
    import json
    with open(path, 'r') as f:
        return json.load(f)

def save_results(results, path):
    """Save results to file"""
    import json
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
'''
                }
            )
        )
    
    @staticmethod
    def node_base() -> TemplateConfig:
        """Base Node.js template"""
        return TemplateConfig(
            template_type=TemplateType.NODE_BASE,
            metadata=TemplateMetadata(
                name="Node.js Base Environment",
                description="Base Node.js environment for JavaScript/TypeScript applications",
                version="1.0.0",
                tags=["nodejs", "base", "javascript"],
                category="base"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.NODE_20,
                cpu_cores=2,
                memory_mb=1024,
                node_packages=[
                    "express",
                    "axios",
                    "dotenv",
                    "lodash",
                    "moment",
                    "ws",
                    "node-fetch"
                ],
                env_vars={
                    "NODE_ENV": "production"
                }
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env node
/**
 * Base Node.js template main script
 */

const process = require('process');

async function main() {
    console.log('Node.js base template initialized');
    
    // Get configuration from arguments
    let config = {};
    if (process.argv.length > 2) {
        try {
            config = JSON.parse(process.argv[2]);
        } catch (e) {
            console.warn('Failed to parse config from arguments');
        }
    }
    
    console.log('Configuration:', config);
    
    // Your code here
    const result = {
        status: 'success',
        timestamp: new Date().toISOString(),
        config: config,
        message: 'Base template executed successfully'
    };
    
    console.log(JSON.stringify(result, null, 2));
    return 0;
}

// Run main function
main().then(code => {
    process.exit(code);
}).catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
''',
                configs={
                    "package.json": '''{
  "name": "e2b-node-template",
  "version": "1.0.0",
  "description": "E2B Node.js base template",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {}
}'''
                }
            )
        )
    
    @staticmethod
    def trading_agent_base() -> TemplateConfig:
        """Base trading agent template"""
        return TemplateConfig(
            template_type=TemplateType.TRADING_AGENT,
            metadata=TemplateMetadata(
                name="Trading Agent Base",
                description="Base template for trading agents with market data access",
                version="1.0.0",
                tags=["trading", "agent", "finance"],
                category="trading"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.PYTHON_3_10,
                cpu_cores=2,
                memory_mb=2048,
                python_packages=[
                    "yfinance>=0.2.0",
                    "pandas>=2.0.0",
                    "numpy>=1.24.0",
                    "ta>=0.10.0",
                    "alpaca-trade-api>=3.0.0",
                    "ccxt>=4.0.0",
                    "pandas-ta>=0.3.0",
                    "backtrader>=1.9.0"
                ],
                env_vars={
                    "TRADING_MODE": "simulation",
                    "RISK_LEVEL": "moderate"
                }
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env python3
"""Trading Agent Base Template"""

import sys
import json
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgent:
    """Base trading agent class"""
    
    def __init__(self, config):
        self.config = config
        self.symbols = config.get('symbols', ['AAPL'])
        self.strategy = config.get('strategy', 'momentum')
        self.risk_limit = config.get('risk_limit', 0.02)
        self.positions = []
        self.trades = []
        
    def fetch_data(self, symbol, period='1mo'):
        """Fetch market data"""
        logger.info(f"Fetching data for {symbol}")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    
    def analyze(self, data):
        """Analyze market data and generate signals"""
        signals = []
        
        # Calculate indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Generate signals based on strategy
        if self.strategy == 'momentum':
            if data['Close'].iloc[-1] > data['SMA20'].iloc[-1]:
                signals.append({
                    'action': 'buy',
                    'strength': 0.7,
                    'reason': 'Price above SMA20'
                })
        
        return signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def execute_trade(self, symbol, signal):
        """Execute a trade based on signal"""
        trade = {
            'symbol': symbol,
            'action': signal['action'],
            'quantity': 100,  # Default quantity
            'price': 0,  # Would be filled with actual price
            'timestamp': datetime.now().isoformat(),
            'signal': signal
        }
        self.trades.append(trade)
        logger.info(f"Executed trade: {trade}")
        return trade
    
    def run(self):
        """Run the trading agent"""
        results = {
            'status': 'success',
            'trades': [],
            'analysis': {}
        }
        
        for symbol in self.symbols:
            try:
                # Fetch and analyze data
                data = self.fetch_data(symbol)
                signals = self.analyze(data)
                
                # Execute trades based on signals
                for signal in signals:
                    if signal['strength'] > 0.6:  # Threshold
                        trade = self.execute_trade(symbol, signal)
                        results['trades'].append(trade)
                
                # Store analysis
                results['analysis'][symbol] = {
                    'last_price': float(data['Close'].iloc[-1]),
                    'volume': int(data['Volume'].iloc[-1]),
                    'signals': signals
                }
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                
        return results

def main():
    """Main entry point"""
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except:
            pass
    
    agent = TradingAgent(config)
    results = agent.run()
    
    print(json.dumps(results, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
                modules={
                    "strategies.py": '''"""Trading strategies module"""

class MomentumStrategy:
    """Momentum trading strategy"""
    
    def generate_signals(self, data):
        signals = []
        # Implementation here
        return signals

class MeanReversionStrategy:
    """Mean reversion trading strategy"""
    
    def generate_signals(self, data):
        signals = []
        # Implementation here
        return signals
''',
                    "risk_manager.py": '''"""Risk management module"""

class RiskManager:
    """Manage trading risks"""
    
    def __init__(self, max_position_size=0.1, stop_loss=0.02):
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
    
    def check_risk(self, trade, portfolio):
        """Check if trade meets risk criteria"""
        # Implementation here
        return True
'''
                }
            )
        )