"""
Neural Trader - Alpaca Integration
Integrates Alpaca trading with neural trader tools and MCP
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .alpaca_client import AlpacaClient, OrderSide, OrderType
from .trading_strategies import TradingBot, MomentumStrategy, MeanReversionStrategy

logger = logging.getLogger(__name__)

class NeuralAlpacaIntegration:
    """
    Integration bridge between Alpaca trading and Neural Trader ecosystem
    Provides MCP integration and neural trading capabilities
    """

    def __init__(self, alpaca_client: AlpacaClient):
        self.alpaca_client = alpaca_client
        self.trading_bot = TradingBot(alpaca_client)
        self.neural_predictions = {}
        self.mcp_enabled = False

        # Initialize neural components
        self.setup_neural_strategies()

    def setup_neural_strategies(self):
        """Setup neural-enhanced trading strategies"""

        # Add neural-enhanced momentum strategy
        neural_momentum = NeuralMomentumStrategy(
            self.alpaca_client,
            lookback_days=20,
            neural_enhancement=True
        )
        self.trading_bot.add_strategy(neural_momentum)

        # Add neural mean reversion
        neural_mean_reversion = NeuralMeanReversionStrategy(
            self.alpaca_client,
            lookback_days=20,
            neural_enhancement=True
        )
        self.trading_bot.add_strategy(neural_mean_reversion)

        logger.info("Neural strategies initialized")

    async def initialize_mcp_integration(self):
        """Initialize MCP (Model Context Protocol) integration"""
        try:
            # Check if MCP tools are available
            import importlib

            # Try to import neural trader MCP tools
            try:
                neural_trader_mcp = importlib.import_module('mcp.neural_trader')
                self.mcp_enabled = True
                logger.info("MCP neural trader tools available")
            except ImportError:
                logger.warning("MCP neural trader tools not available")

            # Try to import flow-nexus MCP tools
            try:
                flow_nexus_mcp = importlib.import_module('mcp.flow_nexus')
                logger.info("Flow Nexus MCP tools available")
            except ImportError:
                logger.warning("Flow Nexus MCP tools not available")

            # Try to import claude-flow MCP tools
            try:
                claude_flow_mcp = importlib.import_module('mcp.claude_flow')
                logger.info("Claude Flow MCP tools available")
            except ImportError:
                logger.warning("Claude Flow MCP tools not available")

        except Exception as e:
            logger.error(f"MCP integration initialization failed: {e}")

    async def get_neural_prediction(self, symbol: str, timeframe: str = '1D') -> Dict[str, Any]:
        """
        Get neural network prediction for a symbol
        Integrates with neural trader tools if available
        """
        try:
            # Get market data
            bars = self.alpaca_client.get_bars(symbol, timeframe, limit=100)

            if bars.empty:
                return {"error": "No market data available"}

            # Basic technical analysis
            bars['sma_20'] = bars['close'].rolling(20).mean()
            bars['sma_50'] = bars['close'].rolling(50).mean()
            bars['rsi'] = self.calculate_rsi(bars['close'])

            # Neural prediction (simulated for now)
            prediction = self.simulate_neural_prediction(bars)

            self.neural_predictions[symbol] = {
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': prediction.get('confidence', 0.5),
                'timeframe': timeframe
            }

            return prediction

        except Exception as e:
            logger.error(f"Neural prediction failed for {symbol}: {e}")
            return {"error": str(e)}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def simulate_neural_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate neural network prediction
        In production, this would connect to actual neural models
        """
        latest = data.iloc[-1]

        # Simple prediction based on technical indicators
        price_momentum = (latest['close'] - latest['sma_20']) / latest['sma_20']
        trend_strength = abs(latest['sma_20'] - latest['sma_50']) / latest['sma_50']
        rsi_signal = (latest['rsi'] - 50) / 50  # Normalized RSI

        # Combine signals
        prediction_score = (price_momentum * 0.4 + trend_strength * 0.3 + rsi_signal * 0.3)

        # Determine direction and confidence
        if prediction_score > 0.1:
            direction = "bullish"
            confidence = min(abs(prediction_score) * 2, 0.95)
        elif prediction_score < -0.1:
            direction = "bearish"
            confidence = min(abs(prediction_score) * 2, 0.95)
        else:
            direction = "neutral"
            confidence = 0.5

        return {
            "direction": direction,
            "confidence": confidence,
            "score": prediction_score,
            "signals": {
                "price_momentum": price_momentum,
                "trend_strength": trend_strength,
                "rsi_signal": rsi_signal
            },
            "next_price_estimate": latest['close'] * (1 + prediction_score * 0.1),
            "time_horizon": "1-5 days"
        }

    async def execute_neural_trading_session(
        self,
        symbols: List[str],
        max_positions: int = 5,
        risk_per_trade: float = 0.02
    ) -> Dict[str, Any]:
        """
        Execute a neural-enhanced trading session
        """
        session_results = {
            "session_id": f"neural_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now(),
            "symbols": symbols,
            "predictions": {},
            "trades": [],
            "errors": []
        }

        try:
            # Get neural predictions for all symbols
            for symbol in symbols:
                prediction = await self.get_neural_prediction(symbol)
                session_results["predictions"][symbol] = prediction

            # Get current account status
            account = self.alpaca_client.get_account()
            portfolio_value = float(account.get('portfolio_value', 100000))

            # Execute trading based on neural predictions
            for symbol, prediction in session_results["predictions"].items():
                if "error" in prediction:
                    session_results["errors"].append(f"{symbol}: {prediction['error']}")
                    continue

                # Check if we should trade based on prediction
                if prediction.get("confidence", 0) > 0.7:
                    trade_result = await self.execute_neural_trade(
                        symbol,
                        prediction,
                        portfolio_value,
                        risk_per_trade
                    )
                    session_results["trades"].append(trade_result)

            session_results["end_time"] = datetime.now()
            session_results["duration"] = (session_results["end_time"] - session_results["start_time"]).total_seconds()

            return session_results

        except Exception as e:
            logger.error(f"Neural trading session failed: {e}")
            session_results["errors"].append(str(e))
            return session_results

    async def execute_neural_trade(
        self,
        symbol: str,
        prediction: Dict[str, Any],
        portfolio_value: float,
        risk_per_trade: float
    ) -> Dict[str, Any]:
        """Execute a trade based on neural prediction"""

        trade_result = {
            "symbol": symbol,
            "prediction": prediction,
            "action": "none",
            "timestamp": datetime.now(),
            "success": False
        }

        try:
            # Determine trade action
            direction = prediction.get("direction", "neutral")
            confidence = prediction.get("confidence", 0.5)

            if direction == "bullish" and confidence > 0.7:
                # Calculate position size
                risk_amount = portfolio_value * risk_per_trade
                current_price = prediction.get("signals", {}).get("current_price", 100)  # Fallback

                # Get actual current price
                bars = self.alpaca_client.get_bars(symbol, '1Day', limit=1)
                if not bars.empty:
                    current_price = bars['close'].iloc[-1]

                qty = max(1, int(risk_amount / current_price))  # At least 1 share

                # Place buy order (in paper trading mode)
                order = self.alpaca_client.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )

                trade_result.update({
                    "action": "buy",
                    "quantity": qty,
                    "price": current_price,
                    "order_id": order.id,
                    "success": True
                })

            elif direction == "bearish" and confidence > 0.7:
                # Check if we have position to sell
                position = self.alpaca_client.get_position(symbol)
                if position and float(position.qty) > 0:
                    # Sell existing position
                    order = self.alpaca_client.place_order(
                        symbol=symbol,
                        qty=float(position.qty),
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET
                    )

                    trade_result.update({
                        "action": "sell",
                        "quantity": float(position.qty),
                        "order_id": order.id,
                        "success": True
                    })

        except Exception as e:
            trade_result["error"] = str(e)
            logger.error(f"Neural trade execution failed for {symbol}: {e}")

        return trade_result

    def get_neural_portfolio_analysis(self) -> Dict[str, Any]:
        """Analyze portfolio using neural predictions"""
        try:
            positions = self.alpaca_client.get_positions()
            account = self.alpaca_client.get_account()

            analysis = {
                "timestamp": datetime.now(),
                "total_value": float(account.get('portfolio_value', 0)),
                "cash": float(account.get('cash', 0)),
                "positions": [],
                "neural_scores": {},
                "recommendations": []
            }

            for position in positions:
                symbol = position.symbol

                # Get neural prediction for this position
                if symbol in self.neural_predictions:
                    neural_data = self.neural_predictions[symbol]
                else:
                    # Get fresh prediction
                    import asyncio
                    neural_data = asyncio.run(self.get_neural_prediction(symbol))

                position_analysis = {
                    "symbol": symbol,
                    "quantity": float(position.qty),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                    "neural_prediction": neural_data
                }

                analysis["positions"].append(position_analysis)
                analysis["neural_scores"][symbol] = neural_data.get("confidence", 0.5)

                # Generate recommendations
                if neural_data.get("direction") == "bearish" and neural_data.get("confidence", 0) > 0.7:
                    analysis["recommendations"].append(f"Consider reducing {symbol} position - bearish signal")
                elif neural_data.get("direction") == "bullish" and neural_data.get("confidence", 0) > 0.7:
                    analysis["recommendations"].append(f"Consider increasing {symbol} position - bullish signal")

            return analysis

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {"error": str(e)}

class NeuralMomentumStrategy(MomentumStrategy):
    """Neural-enhanced momentum strategy"""

    def __init__(self, client: AlpacaClient, lookback_days: int = 20, neural_enhancement: bool = True):
        super().__init__(client, lookback_days)
        self.name = "NeuralMomentum"
        self.neural_enhancement = neural_enhancement

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List:
        """Generate signals with neural enhancement"""
        # Get base momentum signals
        base_signals = super().generate_signals(data)

        if not self.neural_enhancement:
            return base_signals

        # Enhance signals with neural predictions
        enhanced_signals = []
        for signal in base_signals:
            # Add neural confidence scoring
            neural_score = self.get_neural_confidence(signal.symbol, data[signal.symbol])
            signal.strength = signal.strength * neural_score
            signal.reason += f" (Neural confidence: {neural_score:.2f})"
            enhanced_signals.append(signal)

        return enhanced_signals

    def get_neural_confidence(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate neural confidence multiplier"""
        try:
            # Simple neural-like calculation
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            trend_consistency = (returns > 0).sum() / len(returns)

            # Confidence based on trend consistency and volatility
            confidence = trend_consistency * (1 - min(volatility * 10, 0.5))
            return max(0.1, min(confidence, 1.0))

        except Exception:
            return 0.5  # Default confidence

class NeuralMeanReversionStrategy(MeanReversionStrategy):
    """Neural-enhanced mean reversion strategy"""

    def __init__(self, client: AlpacaClient, lookback_days: int = 20, neural_enhancement: bool = True):
        super().__init__(client, lookback_days)
        self.name = "NeuralMeanReversion"
        self.neural_enhancement = neural_enhancement

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List:
        """Generate signals with neural enhancement"""
        base_signals = super().generate_signals(data)

        if not self.neural_enhancement:
            return base_signals

        # Apply neural filtering
        filtered_signals = []
        for signal in base_signals:
            neural_filter = self.apply_neural_filter(signal.symbol, data[signal.symbol])
            if neural_filter > 0.6:  # Only keep high-confidence signals
                signal.strength = signal.strength * neural_filter
                signal.reason += f" (Neural filter: {neural_filter:.2f})"
                filtered_signals.append(signal)

        return filtered_signals

    def apply_neural_filter(self, symbol: str, df: pd.DataFrame) -> float:
        """Apply neural filtering to mean reversion signals"""
        try:
            # Calculate multiple timeframe analysis
            short_ma = df['close'].rolling(5).mean()
            long_ma = df['close'].rolling(20).mean()

            # Check if short-term and long-term trends align
            short_trend = short_ma.iloc[-1] > short_ma.iloc[-5]
            long_trend = long_ma.iloc[-1] > long_ma.iloc[-10]

            # Higher confidence when trends don't align (better for mean reversion)
            trend_score = 0.8 if short_trend != long_trend else 0.4

            # Volume confirmation
            volume_ma = df['volume'].rolling(10).mean()
            volume_score = min(df['volume'].iloc[-1] / volume_ma.iloc[-1], 2.0) / 2.0

            return (trend_score + volume_score) / 2

        except Exception:
            return 0.5

# Integration functions for external use
async def create_neural_alpaca_integration(api_key: str = None, secret_key: str = None, base_url: str = None) -> NeuralAlpacaIntegration:
    """Create and initialize neural Alpaca integration"""
    client = AlpacaClient(api_key, secret_key, base_url)
    integration = NeuralAlpacaIntegration(client)
    await integration.initialize_mcp_integration()
    return integration

def run_neural_trading_demo(symbols: List[str] = None) -> Dict[str, Any]:
    """Run a demonstration of neural trading capabilities"""
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT']

    async def demo():
        integration = await create_neural_alpaca_integration()
        return await integration.execute_neural_trading_session(symbols)

    return asyncio.run(demo())

if __name__ == "__main__":
    # Demo execution
    print("🧠 Neural Alpaca Integration Demo")
    print("=" * 50)

    result = run_neural_trading_demo(['AAPL', 'TSLA'])
    print(json.dumps(result, indent=2, default=str))