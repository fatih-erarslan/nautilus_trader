"""
E2B Sandbox Manager for centralized sandbox lifecycle management
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from e2b import Sandbox
from dotenv import load_dotenv

from .models import (
    SandboxConfig, 
    SandboxStatus, 
    SandboxInfo,
    ProcessResult,
    AgentResult
)

load_dotenv()
logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages E2B sandbox instances and their lifecycle"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize sandbox manager"""
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY not found in environment")
        
        self.sandboxes: Dict[str, Sandbox] = {}
        self.sandbox_info: Dict[str, SandboxInfo] = {}
        self.max_sandboxes = 10  # Limit concurrent sandboxes
        
    def create_sandbox(self, config: SandboxConfig) -> str:
        """Create a new E2B sandbox"""
        try:
            # Check sandbox limit
            if len(self.sandboxes) >= self.max_sandboxes:
                self._cleanup_idle_sandboxes()
                
            # Create sandbox with configuration
            sandbox = Sandbox(
                api_key=self.api_key,
                template=config.template,
                timeout=config.timeout,
                metadata=config.metadata,
                envs=config.envs,
                allow_internet_access=config.allow_internet
            )
            
            sandbox_id = sandbox.sandbox_id
            self.sandboxes[sandbox_id] = sandbox
            
            # Store sandbox info
            self.sandbox_info[sandbox_id] = SandboxInfo(
                sandbox_id=sandbox_id,
                name=config.name,
                status=SandboxStatus.IDLE,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata=config.metadata
            )
            
            # Setup sandbox environment
            self._setup_sandbox_environment(sandbox, config)
            
            logger.info(f"Created sandbox {sandbox_id} ({config.name})")
            return sandbox_id
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise
    
    def _setup_sandbox_environment(self, sandbox: Sandbox, config: SandboxConfig):
        """Setup the sandbox environment with required dependencies"""
        try:
            # Skip heavy setup for now - just create basic directories
            sandbox.commands.run("mkdir -p /tmp/workspace")
            
            # Only copy essential code if needed
            if config.metadata.get("setup_type") == "full":
                # Install Python dependencies
                requirements = """
numpy
pandas
yfinance
ta
scikit-learn
requests
python-dotenv
                """.strip()
                
                sandbox.files.write("/tmp/requirements.txt", requirements)
                sandbox.commands.run("pip install -r /tmp/requirements.txt")
                
                # Copy trading strategies code
                self._copy_trading_code(sandbox)
            
            # Set environment variables
            for key, value in config.envs.items():
                sandbox.commands.run(f"export {key}='{value}'")
                
            logger.info(f"Sandbox environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup sandbox environment: {e}")
            raise
    
    def _copy_trading_code(self, sandbox: Sandbox):
        """Copy essential trading code to sandbox"""
        # Create directory structure
        dirs = [
            "/tmp/trading",
            "/tmp/trading/strategies",
            "/tmp/trading/indicators",
            "/tmp/news",
            "/tmp/optimization"
        ]
        
        for dir_path in dirs:
            sandbox.commands.run(f"mkdir -p {dir_path}")
        
        # Copy base trading strategy template
        strategy_template = '''
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class BaseTrader:
    """Base trading strategy template"""
    
    def __init__(self, symbol, config=None):
        self.symbol = symbol
        self.config = config or {}
        self.positions = []
        self.trades = []
        
    def fetch_data(self, period="1mo"):
        """Fetch market data"""
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period)
        return data
        
    def calculate_signals(self, data):
        """Calculate trading signals"""
        # Implement strategy logic
        return []
        
    def execute_trade(self, signal):
        """Execute a trade based on signal"""
        trade = {
            "symbol": self.symbol,
            "action": signal["action"],
            "quantity": signal.get("quantity", 100),
            "price": signal["price"],
            "timestamp": datetime.now().isoformat()
        }
        self.trades.append(trade)
        return trade
        
    def run(self):
        """Run the trading strategy"""
        data = self.fetch_data()
        signals = self.calculate_signals(data)
        
        for signal in signals:
            self.execute_trade(signal)
            
        return {
            "trades": self.trades,
            "performance": self.calculate_performance()
        }
        
    def calculate_performance(self):
        """Calculate strategy performance"""
        if not self.trades:
            return {"total_trades": 0}
            
        return {
            "total_trades": len(self.trades),
            "symbols_traded": list(set(t["symbol"] for t in self.trades))
        }
'''
        
        sandbox.files.write("/tmp/trading/base_trader.py", strategy_template)
        
        # Copy momentum trader
        momentum_trader = '''
from trading.base_trader import BaseTrader
import numpy as np

class MomentumTrader(BaseTrader):
    """Momentum-based trading strategy"""
    
    def calculate_signals(self, data):
        signals = []
        
        # Calculate momentum indicators
        data["returns"] = data["Close"].pct_change()
        data["momentum"] = data["returns"].rolling(window=10).mean()
        
        # Generate signals
        for i in range(len(data)):
            if i < 10:
                continue
                
            if data["momentum"].iloc[i] > 0.01:  # Positive momentum
                signals.append({
                    "action": "buy",
                    "price": data["Close"].iloc[i],
                    "quantity": 100,
                    "reason": "positive_momentum"
                })
            elif data["momentum"].iloc[i] < -0.01:  # Negative momentum
                signals.append({
                    "action": "sell",
                    "price": data["Close"].iloc[i],
                    "quantity": 100,
                    "reason": "negative_momentum"
                })
                
        return signals
'''
        
        sandbox.files.write("/tmp/trading/strategies/momentum_trader.py", momentum_trader)
    
    def get_sandbox(self, sandbox_id: str) -> Optional[Sandbox]:
        """Get a sandbox instance by ID"""
        return self.sandboxes.get(sandbox_id)
    
    def get_sandbox_info(self, sandbox_id: str) -> Optional[SandboxInfo]:
        """Get sandbox information"""
        return self.sandbox_info.get(sandbox_id)
    
    def list_sandboxes(self) -> List[SandboxInfo]:
        """List all active sandboxes"""
        return list(self.sandbox_info.values())
    
    def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a sandbox"""
        try:
            if sandbox_id in self.sandboxes:
                sandbox = self.sandboxes[sandbox_id]
                sandbox.kill()
                
                del self.sandboxes[sandbox_id]
                
                if sandbox_id in self.sandbox_info:
                    self.sandbox_info[sandbox_id].status = SandboxStatus.TERMINATED
                    
                logger.info(f"Terminated sandbox {sandbox_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to terminate sandbox {sandbox_id}: {e}")
            
        return False
    
    def _cleanup_idle_sandboxes(self):
        """Clean up idle sandboxes to free resources"""
        idle_threshold = 600  # 10 minutes
        current_time = datetime.now()
        
        for sandbox_id, info in list(self.sandbox_info.items()):
            if info.status == SandboxStatus.IDLE:
                idle_time = (current_time - info.last_activity).total_seconds()
                if idle_time > idle_threshold:
                    self.terminate_sandbox(sandbox_id)
                    logger.info(f"Cleaned up idle sandbox {sandbox_id}")
    
    def execute_command(self, sandbox_id: str, command: str, 
                       timeout: int = 30) -> ProcessResult:
        """Execute a command in a sandbox"""
        sandbox = self.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        started_at = datetime.now()
        
        try:
            # Update sandbox status
            if sandbox_id in self.sandbox_info:
                self.sandbox_info[sandbox_id].status = SandboxStatus.PROCESSING
                self.sandbox_info[sandbox_id].last_activity = started_at
            
            # Execute command
            result = sandbox.commands.run(command)
            
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            # Update status back to idle
            if sandbox_id in self.sandbox_info:
                self.sandbox_info[sandbox_id].status = SandboxStatus.IDLE
            
            return ProcessResult(
                sandbox_id=sandbox_id,
                command=command,
                exit_code=result.exit_code if hasattr(result, 'exit_code') else 0,
                stdout=result.stdout if hasattr(result, 'stdout') else "",
                stderr=result.stderr if hasattr(result, 'stderr') else "",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration
            )
            
        except Exception as e:
            if sandbox_id in self.sandbox_info:
                self.sandbox_info[sandbox_id].status = SandboxStatus.ERROR
                
            return ProcessResult(
                sandbox_id=sandbox_id,
                command=command,
                started_at=started_at,
                error=str(e)
            )
    
    def upload_file(self, sandbox_id: str, local_path: str, 
                   sandbox_path: str) -> bool:
        """Upload a file to sandbox"""
        sandbox = self.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        try:
            with open(local_path, 'r') as f:
                content = f.read()
            
            sandbox.files.write(sandbox_path, content)
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to sandbox: {e}")
            return False
    
    def download_file(self, sandbox_id: str, sandbox_path: str) -> Optional[str]:
        """Download a file from sandbox"""
        sandbox = self.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        try:
            content = sandbox.files.read(sandbox_path)
            return content
            
        except Exception as e:
            logger.error(f"Failed to download file from sandbox: {e}")
            return None
    
    def cleanup_all(self):
        """Clean up all sandboxes"""
        for sandbox_id in list(self.sandboxes.keys()):
            self.terminate_sandbox(sandbox_id)
        
        self.sandboxes.clear()
        self.sandbox_info.clear()
        logger.info("Cleaned up all sandboxes")