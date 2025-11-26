#!/usr/bin/env python3
"""
Interactive Brokers Canada - Production Usage Example

This script demonstrates production-ready usage patterns for the IB Canada client,
including error handling, risk management, and integration with neural signals.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from canadian_trading import (
    IBCanadaClient,
    ConnectionConfig,
    OrderType,
    ConnectionState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Example trading system using IB Canada client"""
    
    def __init__(self, paper_trading: bool = True):
        """
        Initialize trading system
        
        Args:
            paper_trading: Use paper trading account
        """
        # Configure connection
        config = ConnectionConfig(
            host="127.0.0.1",
            port=7497 if paper_trading else 7496,  # Paper vs Live ports
            client_id=1,
            is_paper=paper_trading,
            max_retries=5,
            retry_delay=5
        )
        
        self.client = IBCanadaClient(config)
        self.paper_trading = paper_trading
        self.active_orders = {}
        
        # Register event handlers
        self.client.on('connected', self._on_connected)
        self.client.on('order_status', self._on_order_status)
        self.client.on('position', self._on_position_update)
        self.client.on('error', self._on_error)
        
    async def _on_connected(self, data: Dict):
        """Handle connection event"""
        logger.info(f"Connected to IB. Account: {data.get('account')}")
        
    async def _on_order_status(self, data: Dict):
        """Handle order status updates"""
        order_id = data['order_id']
        status = data['status']
        filled = data['filled']
        
        logger.info(f"Order {order_id} - Status: {status}, Filled: {filled}")
        
        # Update active orders
        if status in ['Filled', 'Cancelled']:
            self.active_orders.pop(order_id, None)
            
    async def _on_position_update(self, data: Dict):
        """Handle position updates"""
        logger.info(f"Position update - {data['symbol']}: {data['position']} @ {data['avg_cost']}")
        
    async def _on_error(self, data: Dict):
        """Handle errors"""
        logger.error(f"IB Error: {data['error_string']} (Code: {data['error_code']})")
        
    async def start(self):
        """Start the trading system"""
        logger.info(f"Starting trading system (Paper Trading: {self.paper_trading})")
        
        # Connect to IB
        connected = await self.client.connect()
        if not connected:
            raise ConnectionError("Failed to connect to Interactive Brokers")
            
        # Display account info
        await self.display_account_info()
        
    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            await self.client.cancel_order(order_id)
            
        # Disconnect
        await self.client.disconnect()
        
    async def display_account_info(self):
        """Display account information"""
        # Get account values
        net_liq = self.client.get_account_value("NetLiquidation", "BASE")
        cash = self.client.get_account_value("TotalCashValue", "BASE")
        
        logger.info("=== Account Information ===")
        logger.info(f"Account: {self.client.config.account}")
        logger.info(f"Net Liquidation: ${net_liq:,.2f}" if net_liq else "Net Liquidation: N/A")
        logger.info(f"Total Cash: ${cash:,.2f}" if cash else "Total Cash: N/A")
        
        # Get positions
        positions = self.client.get_positions()
        if positions:
            logger.info("\n=== Current Positions ===")
            for symbol, pos in positions.items():
                logger.info(f"{symbol}: {pos.position} shares @ ${pos.avg_cost:.2f} "
                          f"(Market Value: ${pos.market_value:,.2f}, "
                          f"Unrealized P&L: ${pos.unrealized_pnl:,.2f})")
        else:
            logger.info("No open positions")
            
    async def execute_neural_trade(self, 
                                 symbol: str,
                                 neural_signal: Dict,
                                 max_position_value: float = 10000):
        """
        Execute trade based on neural signal
        
        Args:
            symbol: Stock symbol (e.g., "SHOP.TO" for Canadian, "AAPL" for US)
            neural_signal: Neural prediction signal
            max_position_value: Maximum position value allowed
        """
        try:
            # Extract signal data
            confidence = neural_signal.get('confidence', 0)
            prediction = neural_signal.get('prediction', 0)
            volatility = neural_signal.get('volatility_prediction', 0.02)
            
            logger.info(f"Neural signal for {symbol} - Prediction: {prediction:.3f}, "
                       f"Confidence: {confidence:.3f}, Volatility: {volatility:.3f}")
            
            # Check confidence threshold
            if confidence < 0.7:
                logger.info(f"Confidence too low ({confidence:.3f} < 0.7), skipping trade")
                return
                
            # Determine if Canadian or US stock
            if symbol.endswith('.TO'):
                contract = self.client.create_canadian_stock(symbol[:-3], "CAD")
            else:
                contract = self.client.create_us_stock(symbol)
                
            # Get market data
            ticker = await self.client.get_market_data(contract, snapshot=True)
            if not ticker or not ticker.marketPrice():
                logger.error(f"Unable to get market price for {symbol}")
                return
                
            market_price = ticker.marketPrice()
            
            # Determine action based on prediction
            if prediction > 0.6:
                action = "BUY"
            elif prediction < 0.4:
                action = "SELL"
            else:
                logger.info("Neutral prediction, no action taken")
                return
                
            # Check existing position
            position = self.client.get_position(contract.symbol)
            current_position = position.position if position else 0
            
            # Skip if already have position in same direction
            if (action == "BUY" and current_position > 0) or \
               (action == "SELL" and current_position < 0):
                logger.info(f"Already have {action} position, skipping")
                return
                
            # Calculate position size based on confidence and volatility
            base_position_value = max_position_value * confidence
            volatility_adjustment = max(0.5, 1 - volatility * 10)  # Reduce size for high volatility
            adjusted_position_value = base_position_value * volatility_adjustment
            
            quantity = int(adjusted_position_value / market_price)
            if quantity == 0:
                logger.info("Position size too small, skipping")
                return
                
            # Validate order risk
            is_valid, error_msg = await self.client.validate_order_risk(
                contract, action, quantity, OrderType.MARKET
            )
            
            if not is_valid:
                logger.warning(f"Order failed risk validation: {error_msg}")
                return
                
            # Determine order type based on confidence
            if confidence > 0.85:
                # High confidence - market order
                order_type = OrderType.MARKET
                logger.info(f"Placing MARKET order: {action} {quantity} {symbol}")
                
                trade = await self.client.place_order(
                    contract=contract,
                    order_type=order_type,
                    action=action,
                    quantity=quantity
                )
            else:
                # Medium confidence - limit order
                order_type = OrderType.LIMIT
                limit_price = market_price * (0.995 if action == "BUY" else 1.005)
                
                logger.info(f"Placing LIMIT order: {action} {quantity} {symbol} @ ${limit_price:.2f}")
                
                trade = await self.client.place_order(
                    contract=contract,
                    order_type=order_type,
                    action=action,
                    quantity=quantity,
                    price=limit_price,
                    tif="DAY"
                )
                
            if trade:
                self.active_orders[trade.order.orderId] = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'neural_signal': neural_signal,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Order placed successfully. Order ID: {trade.order.orderId}")
                
                # Set stop loss based on volatility
                stop_distance = market_price * volatility * 2
                stop_price = market_price - stop_distance if action == "BUY" else market_price + stop_distance
                
                stop_order = await self.client.place_order(
                    contract=contract,
                    order_type=OrderType.STOP,
                    action="SELL" if action == "BUY" else "BUY",
                    quantity=quantity,
                    stop_price=stop_price,
                    tif="GTC",
                    parentId=trade.order.orderId
                )
                
                if stop_order:
                    logger.info(f"Stop loss set at ${stop_price:.2f}")
                    
        except Exception as e:
            logger.error(f"Error executing neural trade: {e}")
            
    async def get_market_snapshot(self, symbols: list):
        """Get market snapshot for multiple symbols"""
        logger.info(f"Getting market snapshot for {len(symbols)} symbols")
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # Create contract
                if symbol.endswith('.TO'):
                    contract = self.client.create_canadian_stock(symbol[:-3], "CAD")
                else:
                    contract = self.client.create_us_stock(symbol)
                    
                # Get market data
                ticker = await self.client.get_market_data(contract, snapshot=True)
                
                if ticker:
                    market_data[symbol] = {
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'last': ticker.last,
                        'volume': ticker.volume,
                        'market_price': ticker.marketPrice()
                    }
                    
                    logger.info(f"{symbol}: ${ticker.marketPrice():.2f} "
                              f"(Bid: {ticker.bid}, Ask: {ticker.ask})")
                              
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
                
        return market_data
        
    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        positions = self.client.get_positions()
        
        for symbol, position in positions.items():
            logger.info(f"Monitoring {symbol}: {position.position} shares, "
                       f"Unrealized P&L: ${position.unrealized_pnl:.2f}")
            
            # Example: Close position if profit target reached
            if position.unrealized_pnl > 0:
                profit_pct = position.unrealized_pnl / (position.position * position.avg_cost)
                
                if profit_pct > 0.02:  # 2% profit target
                    logger.info(f"Profit target reached for {symbol} ({profit_pct:.1%}), closing position")
                    await self.client.close_position(symbol)


async def main():
    """Main function demonstrating usage"""
    
    # Create trading system
    system = TradingSystem(paper_trading=True)
    
    try:
        # Start system
        await system.start()
        
        # Example 1: Get market snapshot
        canadian_symbols = ["SHOP.TO", "RY.TO", "CNR.TO"]
        us_symbols = ["AAPL", "GOOGL", "MSFT"]
        
        await system.get_market_snapshot(canadian_symbols + us_symbols)
        
        # Example 2: Execute neural trade (simulated signal)
        neural_signal = {
            'prediction': 0.75,  # Bullish prediction
            'confidence': 0.82,  # High confidence
            'volatility_prediction': 0.015,  # 1.5% expected volatility
            'expected_return': 0.03  # 3% expected return
        }
        
        # Uncomment to execute trade
        # await system.execute_neural_trade("SHOP.TO", neural_signal)
        
        # Example 3: Monitor positions
        await system.monitor_positions()
        
        # Keep running for a while to see updates
        await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        
    finally:
        # Stop system
        await system.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())