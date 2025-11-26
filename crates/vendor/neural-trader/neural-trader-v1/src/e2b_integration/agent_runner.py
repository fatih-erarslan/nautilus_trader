"""
Agent Runner for executing trading agents in E2B sandboxes
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .sandbox_manager import SandboxManager
from .models import (
    AgentConfig,
    AgentResult,
    AgentType,
    SandboxConfig,
    SandboxStatus
)

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs trading agents in isolated E2B sandboxes"""
    
    def __init__(self, sandbox_manager: SandboxManager):
        """Initialize agent runner"""
        self.sandbox_manager = sandbox_manager
        self.agent_scripts = self._load_agent_scripts()
        
    def _load_agent_scripts(self) -> Dict[AgentType, str]:
        """Load agent implementation scripts"""
        scripts = {}
        
        # Momentum Trader Script
        scripts[AgentType.MOMENTUM_TRADER] = '''
import json
import sys
sys.path.append('/tmp')

from trading.strategies.momentum_trader import MomentumTrader

def run_momentum_trader(config):
    """Run momentum trading strategy"""
    symbol = config.get("symbols", ["AAPL"])[0]
    params = config.get("strategy_params", {})
    
    trader = MomentumTrader(symbol, params)
    result = trader.run()
    
    return {
        "status": "success",
        "trades": result["trades"],
        "performance": result["performance"]
    }

if __name__ == "__main__":
    import sys
    config = json.loads(sys.argv[1])
    result = run_momentum_trader(config)
    print(json.dumps(result))
'''

        # Mean Reversion Trader Script
        scripts[AgentType.MEAN_REVERSION_TRADER] = '''
import json
import sys
import numpy as np
sys.path.append('/tmp')

from trading.base_trader import BaseTrader

class MeanReversionTrader(BaseTrader):
    """Mean reversion trading strategy"""
    
    def calculate_signals(self, data):
        signals = []
        
        # Calculate moving average and standard deviation
        window = self.config.get("window", 20)
        data["ma"] = data["Close"].rolling(window=window).mean()
        data["std"] = data["Close"].rolling(window=window).std()
        
        # Calculate z-score
        data["z_score"] = (data["Close"] - data["ma"]) / data["std"]
        
        # Generate signals based on z-score
        threshold = self.config.get("z_threshold", 2.0)
        
        for i in range(window, len(data)):
            z = data["z_score"].iloc[i]
            
            if z < -threshold:  # Oversold - buy signal
                signals.append({
                    "action": "buy",
                    "price": data["Close"].iloc[i],
                    "quantity": 100,
                    "reason": f"oversold_z_{z:.2f}"
                })
            elif z > threshold:  # Overbought - sell signal
                signals.append({
                    "action": "sell",
                    "price": data["Close"].iloc[i],
                    "quantity": 100,
                    "reason": f"overbought_z_{z:.2f}"
                })
                
        return signals

def run_mean_reversion_trader(config):
    """Run mean reversion trading strategy"""
    symbol = config.get("symbols", ["AAPL"])[0]
    params = config.get("strategy_params", {})
    
    trader = MeanReversionTrader(symbol, params)
    result = trader.run()
    
    return {
        "status": "success",
        "trades": result["trades"],
        "performance": result["performance"]
    }

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    result = run_mean_reversion_trader(config)
    print(json.dumps(result))
'''

        # Neural Forecaster Script
        scripts[AgentType.NEURAL_FORECASTER] = '''
import json
import sys
import numpy as np
import pandas as pd
sys.path.append('/tmp')

def run_neural_forecaster(config):
    """Run neural forecasting model"""
    symbols = config.get("symbols", ["AAPL"])
    
    # Simulate neural network predictions
    predictions = []
    for symbol in symbols:
        # Generate mock predictions
        forecast = {
            "symbol": symbol,
            "predictions": {
                "1_day": np.random.uniform(-0.05, 0.05),
                "5_day": np.random.uniform(-0.10, 0.10),
                "10_day": np.random.uniform(-0.15, 0.15)
            },
            "confidence": np.random.uniform(0.5, 0.95),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        predictions.append(forecast)
    
    return {
        "status": "success",
        "predictions": predictions,
        "model_metrics": {
            "accuracy": np.random.uniform(0.6, 0.85),
            "sharpe_ratio": np.random.uniform(0.5, 2.0)
        }
    }

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    result = run_neural_forecaster(config)
    print(json.dumps(result))
'''

        # News Analyzer Script
        scripts[AgentType.NEWS_ANALYZER] = '''
import json
import sys
import random
sys.path.append('/tmp')

def run_news_analyzer(config):
    """Run news sentiment analysis"""
    symbols = config.get("symbols", ["AAPL"])
    
    # Simulate news analysis
    analysis = []
    for symbol in symbols:
        sentiment = {
            "symbol": symbol,
            "overall_sentiment": random.choice(["bullish", "neutral", "bearish"]),
            "sentiment_score": random.uniform(-1, 1),
            "news_volume": random.randint(5, 50),
            "key_topics": ["earnings", "product_launch", "market_trends"],
            "recommendation": random.choice(["buy", "hold", "sell"])
        }
        analysis.append(sentiment)
    
    return {
        "status": "success",
        "analysis": analysis,
        "summary": {
            "total_articles_analyzed": sum(s["news_volume"] for s in analysis),
            "average_sentiment": sum(s["sentiment_score"] for s in analysis) / len(analysis)
        }
    }

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    result = run_news_analyzer(config)
    print(json.dumps(result))
'''

        # Risk Manager Script
        scripts[AgentType.RISK_MANAGER] = '''
import json
import sys
import numpy as np
sys.path.append('/tmp')

def run_risk_manager(config):
    """Run risk management analysis"""
    portfolio = config.get("portfolio", {})
    risk_limits = config.get("risk_limits", {})
    
    # Calculate portfolio metrics
    total_value = sum(portfolio.values()) if portfolio else 100000
    
    # Simulate risk metrics
    metrics = {
        "portfolio_value": total_value,
        "var_95": total_value * 0.02,  # 2% VaR
        "cvar_95": total_value * 0.03,  # 3% CVaR
        "max_drawdown": np.random.uniform(0.05, 0.15),
        "sharpe_ratio": np.random.uniform(0.5, 2.0),
        "beta": np.random.uniform(0.8, 1.2),
        "correlation_spy": np.random.uniform(0.6, 0.95)
    }
    
    # Check risk limits
    violations = []
    if risk_limits:
        if "max_position_size" in risk_limits:
            for symbol, value in portfolio.items():
                if value > risk_limits["max_position_size"]:
                    violations.append(f"Position size violation: {symbol}")
        
        if "max_drawdown" in risk_limits:
            if metrics["max_drawdown"] > risk_limits["max_drawdown"]:
                violations.append("Max drawdown exceeded")
    
    return {
        "status": "success",
        "risk_metrics": metrics,
        "violations": violations,
        "recommendations": [
            "Diversify portfolio",
            "Reduce position sizes",
            "Add hedging instruments"
        ] if violations else ["Portfolio within risk limits"]
    }

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    result = run_risk_manager(config)
    print(json.dumps(result))
'''
        
        return scripts
    
    async def run_agent(self, config: AgentConfig) -> AgentResult:
        """Run an agent in a sandbox"""
        started_at = datetime.now()
        
        try:
            # Create sandbox for agent
            sandbox_config = SandboxConfig(
                name=f"agent_{config.agent_type.value}_{started_at.strftime('%Y%m%d_%H%M%S')}",
                timeout=600,  # 10 minutes
                memory_mb=1024 if config.use_gpu else 512,
                cpu_count=2 if config.use_gpu else 1,
                envs={
                    "AGENT_TYPE": config.agent_type.value,
                    "EXECUTION_MODE": config.execution_mode
                }
            )
            
            sandbox_id = self.sandbox_manager.create_sandbox(sandbox_config)
            
            # Get the sandbox
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            if not sandbox:
                raise ValueError(f"Failed to get sandbox {sandbox_id}")
            
            # Write agent script
            script = self.agent_scripts.get(config.agent_type)
            if not script:
                script = self._get_custom_agent_script(config)
            
            agent_script_path = f"/workspace/agent_{config.agent_type.value}.py"
            sandbox.files.write(agent_script_path, script)
            
            # Prepare configuration
            agent_config = {
                "symbols": config.symbols,
                "strategy_params": config.strategy_params,
                "risk_limits": config.risk_limits,
                "data_source": config.data_source
            }
            
            # Run the agent
            command = f"cd /workspace && python {agent_script_path} '{json.dumps(agent_config)}'"
            result = self.sandbox_manager.execute_command(sandbox_id, command, timeout=300)
            
            # Parse agent output
            agent_output = {}
            if result.stdout:
                try:
                    agent_output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse agent output: {result.stdout}")
                    agent_output = {"raw_output": result.stdout}
            
            completed_at = datetime.now()
            
            # Create agent result
            agent_result = AgentResult(
                sandbox_id=sandbox_id,
                agent_type=config.agent_type,
                status=agent_output.get("status", "completed"),
                trades=agent_output.get("trades", []),
                performance=agent_output.get("performance", {}),
                logs=[result.stdout] if result.stdout else [],
                errors=[result.stderr] if result.stderr else [],
                started_at=started_at,
                completed_at=completed_at,
                metadata={
                    "config": config.dict(),
                    "output": agent_output
                }
            )
            
            # Clean up sandbox (optional - keep for debugging)
            # self.sandbox_manager.terminate_sandbox(sandbox_id)
            
            return agent_result
            
        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            
            return AgentResult(
                sandbox_id="error",
                agent_type=config.agent_type,
                status="error",
                errors=[str(e)],
                started_at=started_at,
                completed_at=datetime.now()
            )
    
    def _get_custom_agent_script(self, config: AgentConfig) -> str:
        """Get custom agent script"""
        # Default custom agent template
        return '''
import json
import sys

def run_custom_agent(config):
    """Run custom agent logic"""
    return {
        "status": "success",
        "message": "Custom agent executed",
        "config": config
    }

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    result = run_custom_agent(config)
    print(json.dumps(result))
'''
    
    async def run_multiple_agents(self, configs: List[AgentConfig]) -> List[AgentResult]:
        """Run multiple agents concurrently"""
        import asyncio
        
        tasks = [self.run_agent(config) for config in configs]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_agent_types(self) -> List[str]:
        """Get available agent types"""
        return [agent_type.value for agent_type in AgentType]
    
    def validate_agent_config(self, config: AgentConfig) -> bool:
        """Validate agent configuration"""
        if not config.symbols and config.agent_type != AgentType.CUSTOM:
            logger.warning("No symbols provided for agent")
            
        if config.agent_type not in AgentType:
            logger.error(f"Invalid agent type: {config.agent_type}")
            return False
            
        return True