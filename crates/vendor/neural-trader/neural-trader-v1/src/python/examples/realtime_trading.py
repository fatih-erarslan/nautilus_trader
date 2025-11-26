"""
Real-Time Trading Example
========================

This example demonstrates real-time trading operations using WebSocket
connections for live market data and trading signals.
"""

import asyncio
import os
import json
from uuid import uuid4
from datetime import datetime
import random

from supabase_client import NeuralTradingClient
from supabase_client.clients.trading_bots import (
    CreateBotRequest,
    PlaceOrderRequest,
    OrderSide,
    OrderType
)
from supabase_client.real_time.channel_manager import RealtimeChannelManager

class RealTimeTradingSystem:
    """Real-time trading system with event-driven architecture."""
    
    def __init__(self, client: NeuralTradingClient):
        self.client = client
        self.realtime = client.realtime
        self.active_bots = {}
        self.market_data = {}
        self.running = False
        
    async def setup_event_handlers(self):
        """Set up event handlers for real-time data."""
        
        # Market data handler
        async def handle_market_data(data):
            symbol = data.get('symbol')
            price = data.get('price')
            volume = data.get('volume', 0)
            
            print(f"📈 {symbol}: ${price:.2f} (Volume: {volume:,})")
            
            # Update internal market data
            self.market_data[symbol] = data
            
            # Check for trading opportunities
            await self.evaluate_trading_signals(symbol, data)
        
        # Trading signal handler
        async def handle_trading_signal(data):
            signal_type = data.get('signal_type')
            symbol = data.get('symbol')
            strength = data.get('strength', 0)
            bot_id = data.get('bot_id')
            
            print(f"🚨 Trading Signal: {signal_type} {symbol} (Strength: {strength:.2f})")
            
            if bot_id in self.active_bots and strength > 0.7:
                await self.execute_signal(bot_id, signal_type, symbol, strength)
        
        # Bot status handler
        async def handle_bot_status(data):
            bot_id = data.get('bot_id')
            status = data.get('status')
            message = data.get('message', '')
            
            print(f"🤖 Bot {bot_id}: {status}")
            if message:
                print(f"   Message: {message}")
            
            # Update bot status
            if bot_id in self.active_bots:
                self.active_bots[bot_id]['status'] = status
        
        # Order execution handler
        async def handle_order_execution(data):
            order_id = data.get('order_id')
            symbol = data.get('symbol')
            side = data.get('side')
            quantity = data.get('quantity')
            price = data.get('price')
            status = data.get('status')
            
            print(f"💼 Order {order_id}: {side} {quantity} {symbol} @ ${price:.2f} - {status}")
        
        # Performance alert handler
        async def handle_performance_alert(data):
            alert_type = data.get('alert_type')
            message = data.get('message')
            severity = data.get('severity', 'info')
            
            emoji = {'info': '💡', 'warning': '⚠️', 'error': '❌', 'critical': '🚨'}.get(severity, '📢')
            print(f"{emoji} Alert ({alert_type}): {message}")
        
        # Set handlers
        self.realtime.set_market_data_handler(handle_market_data)
        self.realtime.set_trading_signal_handler(handle_trading_signal)
        self.realtime.set_bot_status_handler(handle_bot_status)
        self.realtime.set_order_execution_handler(handle_order_execution)
        self.realtime.set_performance_alert_handler(handle_performance_alert)
        
        print("✅ Event handlers configured")
    
    async def evaluate_trading_signals(self, symbol, market_data):
        """Evaluate market data for trading opportunities."""
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        
        # Simple momentum strategy example
        if symbol in self.market_data:
            previous_price = self.market_data[symbol].get('price', price)
            price_change = (price - previous_price) / previous_price if previous_price > 0 else 0
            
            # Generate signal based on price movement and volume
            if abs(price_change) > 0.02 and volume > 1000000:  # 2% move with high volume
                signal_type = 'buy' if price_change > 0 else 'sell'
                strength = min(abs(price_change) * 10, 1.0)  # Scale to 0-1
                
                # Simulate signal generation (in real system, this would come from ML models)
                signal_data = {
                    'signal_type': signal_type,
                    'symbol': symbol,
                    'strength': strength,
                    'price': price,
                    'timestamp': datetime.utcnow().isoformat(),
                    'strategy': 'momentum'
                }
                
                # In a real system, this would be published to the signals channel
                print(f"🔍 Generated {signal_type} signal for {symbol} (strength: {strength:.2f})")
    
    async def execute_signal(self, bot_id, signal_type, symbol, strength):
        """Execute a trading signal."""
        try:
            bot_info = self.active_bots.get(bot_id)
            if not bot_info:
                return
            
            # Calculate position size based on signal strength and risk parameters
            risk_params = bot_info.get('risk_params', {})
            max_position_size = risk_params.get('max_position_size', 0.05)
            position_size = max_position_size * strength
            
            # Get current market price
            current_price = self.market_data.get(symbol, {}).get('price', 0)
            if current_price == 0:
                return
            
            # Calculate quantity (simplified)
            account_balance = 10000  # In real system, get from account
            quantity = int((account_balance * position_size) / current_price)
            
            if quantity > 0:
                # Place order
                order_request = PlaceOrderRequest(
                    bot_id=bot_id,
                    symbol=symbol,
                    side=OrderSide.BUY if signal_type == 'buy' else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
                
                order, error = await self.client.trading_bots.place_order(order_request)
                if error:
                    print(f"❌ Error placing order: {error}")
                else:
                    print(f"✅ Placed {signal_type} order: {quantity} {symbol}")
        
        except Exception as e:
            print(f"❌ Error executing signal: {e}")
    
    async def start_trading(self, symbols, bot_configs):
        """Start real-time trading system."""
        print("🚀 Starting real-time trading system...")
        
        # Create trading bots
        user_id = uuid4()
        
        # Create user profile
        profile_data = {
            "id": str(user_id),
            "email": "realtime-trader@example.com",
            "full_name": "Real-time Trader",
            "is_active": True
        }
        await self.client.supabase.insert("profiles", profile_data)
        
        # Create trading account
        account_data = {
            "user_id": str(user_id),
            "account_name": "Real-time Trading Account",
            "account_type": "demo",
            "broker": "demo_broker",
            "balance": 10000.0,
            "currency": "USD",
            "is_active": True
        }
        accounts = await self.client.supabase.insert("trading_accounts", account_data)
        account_id = accounts[0]["id"]
        
        # Create bots
        for bot_config in bot_configs:
            bot_request = CreateBotRequest(
                name=bot_config["name"],
                strategy=bot_config["strategy"],
                account_id=account_id,
                symbols=symbols,
                risk_params=bot_config["risk_params"],
                strategy_params=bot_config["strategy_params"]
            )
            
            bot, error = await self.client.trading_bots.create_bot(user_id, bot_request)
            if error:
                print(f"❌ Error creating bot {bot_config['name']}: {error}")
                continue
            
            # Start bot
            success, error = await self.client.trading_bots.start_bot(bot["id"])
            if error:
                print(f"❌ Error starting bot {bot['name']}: {error}")
                continue
            
            self.active_bots[bot["id"]] = {
                **bot,
                "config": bot_config
            }
            print(f"✅ Started bot: {bot['name']}")
        
        # Set up real-time subscriptions
        await self.setup_event_handlers()
        
        # Subscribe to channels
        await self.realtime.subscribe_to_market_data(symbols=symbols)
        await self.realtime.subscribe_to_trading_signals()
        await self.realtime.subscribe_to_bot_status()
        await self.realtime.subscribe_to_order_executions()
        await self.realtime.subscribe_to_performance_alerts()
        
        print(f"✅ Subscribed to real-time data for symbols: {symbols}")
        
        self.running = True
        
        # Start market data simulation (in real system, this would be live data)
        asyncio.create_task(self.simulate_market_data(symbols))
        
        # Start performance monitoring
        asyncio.create_task(self.monitor_performance())
        
        print("🎯 Real-time trading system is now active!")
    
    async def simulate_market_data(self, symbols):
        """Simulate real-time market data for demonstration."""
        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
            "TSLA": 800.0
        }
        
        while self.running:
            for symbol in symbols:
                if symbol in base_prices:
                    # Simulate price movement
                    change = random.uniform(-0.02, 0.02)  # ±2% random change
                    new_price = base_prices[symbol] * (1 + change)
                    base_prices[symbol] = new_price
                    
                    # Simulate volume
                    volume = random.randint(500000, 3000000)
                    
                    market_data = {
                        'symbol': symbol,
                        'price': new_price,
                        'volume': volume,
                        'timestamp': datetime.utcnow().isoformat(),
                        'high': new_price * 1.01,
                        'low': new_price * 0.99,
                        'open': new_price * 0.995,
                        'close': new_price
                    }
                    
                    # Process market data
                    await self.realtime._handle_market_data(market_data)
            
            await asyncio.sleep(2)  # Update every 2 seconds
    
    async def monitor_performance(self):
        """Monitor trading performance."""
        while self.running:
            try:
                for bot_id, bot_info in self.active_bots.items():
                    # Get bot performance
                    performance, error = await self.client.trading_bots.calculate_bot_performance(
                        bot_id,
                        time_range_hours=1
                    )
                    
                    if not error and performance:
                        pnl = performance.get('total_pnl', 0)
                        win_rate = performance.get('win_rate', 0)
                        
                        if abs(pnl) > 100:  # Alert if P&L > $100
                            alert_type = 'profit' if pnl > 0 else 'loss'
                            await self.realtime._handle_performance_alert({
                                'alert_type': alert_type,
                                'message': f"Bot {bot_info['name']} P&L: ${pnl:.2f}",
                                'severity': 'warning' if abs(pnl) > 200 else 'info',
                                'bot_id': bot_id,
                                'value': pnl
                            })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"❌ Error in performance monitoring: {e}")
                await asyncio.sleep(30)
    
    async def stop_trading(self):
        """Stop the trading system."""
        print("🛑 Stopping real-time trading system...")
        
        self.running = False
        
        # Stop all bots
        for bot_id, bot_info in self.active_bots.items():
            success, error = await self.client.trading_bots.stop_bot(bot_id)
            if error:
                print(f"❌ Error stopping bot {bot_info['name']}: {error}")
            else:
                print(f"✅ Stopped bot: {bot_info['name']}")
        
        # Unsubscribe from channels
        await self.realtime.unsubscribe_all()
        
        print("✅ Real-time trading system stopped")

async def main():
    """Main function demonstrating real-time trading."""
    
    # Initialize client
    client = NeuralTradingClient(
        url=os.getenv("SUPABASE_URL", "https://your-project.supabase.co"),
        key=os.getenv("SUPABASE_ANON_KEY", "your-anon-key"),
        service_key=os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    try:
        await client.connect()
        print("✅ Connected to Supabase")
        
        # Create trading system
        trading_system = RealTimeTradingSystem(client)
        
        # Define trading symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        # Define bot configurations
        bot_configs = [
            {
                "name": "Momentum Scalper",
                "strategy": "momentum",
                "risk_params": {
                    "max_position_size": 0.02,
                    "stop_loss": 0.01,
                    "take_profit": 0.03,
                    "max_daily_trades": 20
                },
                "strategy_params": {
                    "momentum_window": 5,
                    "signal_threshold": 0.015,
                    "volume_filter": True
                }
            },
            {
                "name": "Mean Reversion Bot",
                "strategy": "mean_reversion",
                "risk_params": {
                    "max_position_size": 0.05,
                    "stop_loss": 0.02,
                    "take_profit": 0.04,
                    "max_daily_trades": 10
                },
                "strategy_params": {
                    "lookback_period": 20,
                    "deviation_threshold": 2.0,
                    "min_volume": 1000000
                }
            }
        ]
        
        # Start trading system
        await trading_system.start_trading(symbols, bot_configs)
        
        # Run for a specified duration
        print("📊 Trading system running... Press Ctrl+C to stop")
        
        try:
            # Run for 2 minutes for demonstration
            await asyncio.sleep(120)
        except KeyboardInterrupt:
            print("\n⚠️ Received interrupt signal")
        
        # Stop trading system
        await trading_system.stop_trading()
        
        print("🎉 Real-time trading example completed!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await client.disconnect()
        print("👋 Disconnected from Supabase")

if __name__ == "__main__":
    print("📡 Starting real-time trading example...")
    print("This example simulates market data and trading signals.")
    print("In a production environment, you would connect to real market data feeds.")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example stopped by user")