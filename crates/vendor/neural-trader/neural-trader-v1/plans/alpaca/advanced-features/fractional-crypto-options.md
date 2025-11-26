# Alpaca API Advanced Trading Features Research

## Overview
Alpaca's 2024 advanced features include comprehensive fractional shares trading, 24/7 cryptocurrency trading, multi-leg options strategies, and institutional-grade tools. These features enable sophisticated trading strategies and expand market access for all types of traders.

## Fractional Shares Trading

### Core Capabilities (2024 Updates)
- **Minimum Investment**: As little as $1 worth of shares
- **Supported Assets**: Over 2,000 US equities and ETFs
- **Order Types**: Market, Limit, Stop, Stop-Limit orders
- **Extended Hours**: Pre-market (4:00-9:30 AM ET), After-hours (4:00-8:00 PM ET), Overnight (8:00 PM-4:00 AM ET)
- **Time in Force**: Day orders supported

### Implementation Examples
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class FractionalTrading:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

    def buy_fractional_shares(self, symbol, quantity):
        """Buy fractional shares by quantity"""
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,  # Can be fractional like 0.5
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing fractional order: {e}")
            return None

    def buy_notional_amount(self, symbol, dollar_amount):
        """Buy specific dollar amount of shares"""
        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=dollar_amount,  # Dollar amount to invest
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing notional order: {e}")
            return None

    def fractional_limit_order(self, symbol, quantity, limit_price):
        """Place fractional limit order"""
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing fractional limit order: {e}")
            return None

    def fractional_extended_hours(self, symbol, quantity, extended_hours=True):
        """Trade fractional shares in extended hours"""
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            extended_hours=extended_hours
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing extended hours fractional order: {e}")
            return None

    def dollar_cost_averaging(self, symbol, monthly_amount, days_per_month=30):
        """Implement dollar cost averaging with fractional shares"""
        daily_amount = monthly_amount / days_per_month

        # This would be scheduled to run daily
        order = self.buy_notional_amount(symbol, daily_amount)
        return order

# Example usage
fractional_trader = FractionalTrading("api_key", "secret_key", paper=True)

# Buy 0.5 shares of AAPL
fractional_trader.buy_fractional_shares("AAPL", 0.5)

# Invest exactly $100 in TSLA
fractional_trader.buy_notional_amount("TSLA", 100.00)

# Set up $500/month DCA for S&P 500 ETF
fractional_trader.dollar_cost_averaging("SPY", 500.00)
```

### Fractional Shares Portfolio Management
```python
class FractionalPortfolioManager:
    def __init__(self, trading_client):
        self.trading_client = trading_client

    def rebalance_with_fractionals(self, target_allocations, total_portfolio_value):
        """Rebalance portfolio using fractional shares for precision"""
        rebalance_orders = []

        for symbol, target_percentage in target_allocations.items():
            target_value = total_portfolio_value * target_percentage

            # Get current position
            try:
                current_position = self.trading_client.get_open_position(symbol)
                current_value = float(current_position.market_value)
            except:
                current_value = 0

            value_difference = target_value - current_value

            if abs(value_difference) > 1:  # Only rebalance if difference > $1
                if value_difference > 0:
                    # Need to buy more
                    order = MarketOrderRequest(
                        symbol=symbol,
                        notional=value_difference,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                else:
                    # Need to sell some
                    # Calculate shares to sell
                    current_price = float(current_position.current_price)
                    shares_to_sell = abs(value_difference) / current_price

                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=shares_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )

                try:
                    submitted_order = self.trading_client.submit_order(order)
                    rebalance_orders.append(submitted_order)
                except Exception as e:
                    print(f"Error rebalancing {symbol}: {e}")

        return rebalance_orders

    def micro_investing_strategy(self, symbols_list, daily_budget):
        """Implement micro-investing across multiple symbols"""
        amount_per_symbol = daily_budget / len(symbols_list)

        orders = []
        for symbol in symbols_list:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=amount_per_symbol,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            try:
                submitted_order = self.trading_client.submit_order(order)
                orders.append(submitted_order)
            except Exception as e:
                print(f"Error with micro-investment in {symbol}: {e}")

        return orders
```

## Cryptocurrency Trading

### 24/7 Crypto Trading Features
- **Trading Hours**: 24/7, including weekends
- **Supported Cryptocurrencies**: BTC, ETH, LTC, BCH, and more
- **Order Types**: Market, Limit, Stop-Limit
- **Time in Force**: GTC (Good Till Canceled), IOC (Immediate or Cancel)
- **Fractional Support**: All crypto assets are fractionable

### Crypto Trading Implementation
```python
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest

class CryptoTrading:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

    def buy_crypto_market(self, symbol, notional_amount):
        """Buy crypto with market order using notional amount"""
        order_request = MarketOrderRequest(
            symbol=symbol,  # e.g., "BTC/USD"
            notional=notional_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC  # Good till canceled
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing crypto market order: {e}")
            return None

    def buy_crypto_limit(self, symbol, quantity, limit_price):
        """Buy crypto with limit order"""
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            limit_price=limit_price
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing crypto limit order: {e}")
            return None

    def crypto_stop_limit(self, symbol, quantity, stop_price, limit_price):
        """Place stop-limit order for crypto"""
        order_request = StopLimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
            limit_price=limit_price
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing crypto stop-limit order: {e}")
            return None

    def crypto_dca_strategy(self, symbol, weekly_amount):
        """Implement weekly DCA for cryptocurrency"""
        # This would be scheduled to run weekly
        order = self.buy_crypto_market(symbol, weekly_amount)
        return order

    def crypto_grid_trading(self, symbol, base_price, grid_spacing, grid_levels, order_size):
        """Implement grid trading strategy for crypto"""
        buy_orders = []
        sell_orders = []

        for i in range(1, grid_levels + 1):
            # Buy orders below current price
            buy_price = base_price - (grid_spacing * i)
            buy_order = LimitOrderRequest(
                symbol=symbol,
                qty=order_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                limit_price=buy_price
            )

            # Sell orders above current price
            sell_price = base_price + (grid_spacing * i)
            sell_order = LimitOrderRequest(
                symbol=symbol,
                qty=order_size,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                limit_price=sell_price
            )

            try:
                buy_orders.append(self.trading_client.submit_order(buy_order))
                sell_orders.append(self.trading_client.submit_order(sell_order))
            except Exception as e:
                print(f"Error placing grid orders at level {i}: {e}")

        return {"buy_orders": buy_orders, "sell_orders": sell_orders}

# Example usage
crypto_trader = CryptoTrading("api_key", "secret_key", paper=True)

# Buy $100 worth of Bitcoin
crypto_trader.buy_crypto_market("BTC/USD", 100.00)

# Set up weekly $50 DCA for Ethereum
crypto_trader.crypto_dca_strategy("ETH/USD", 50.00)

# Implement grid trading for Bitcoin
crypto_trader.crypto_grid_trading("BTC/USD", 45000, 500, 5, 0.01)
```

### Real-Time Crypto Streaming
```python
from alpaca.data.live.crypto import CryptoDataStream

class CryptoStreamTrader:
    def __init__(self, api_key, secret_key):
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

        self.crypto_stream = CryptoDataStream(
            api_key=api_key,
            secret_key=secret_key
        )

        self.setup_stream_handlers()

    def setup_stream_handlers(self):
        @self.crypto_stream.on_trade("BTC/USD")
        async def btc_trade_handler(trade):
            await self.process_crypto_trade(trade)

        @self.crypto_stream.on_quote("BTC/USD", "ETH/USD")
        async def crypto_quote_handler(quote):
            await self.process_crypto_quote(quote)

    async def process_crypto_trade(self, trade):
        """Process crypto trade data for trading signals"""
        # Implement your crypto trading logic here
        price = trade.price
        volume = trade.size

        # Example: Buy dip strategy
        if self.is_significant_dip(price):
            await self.execute_dip_buy("BTC/USD", 50.00)

    async def execute_dip_buy(self, symbol, amount):
        """Execute buy order on dip"""
        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )

        try:
            order = self.trading_client.submit_order(order_request)
            print(f"Dip buy executed: {order.id}")
        except Exception as e:
            print(f"Error executing dip buy: {e}")

    def start_crypto_streaming(self):
        """Start 24/7 crypto streaming"""
        self.crypto_stream.subscribe_trades("BTC/USD", "ETH/USD", "LTC/USD")
        self.crypto_stream.subscribe_quotes("BTC/USD", "ETH/USD", "LTC/USD")
        self.crypto_stream.run()
```

## Options Trading

### Options Trading Capabilities
- **Multi-leg Strategies**: Spreads, straddles, strangles, iron condors
- **Contract Types**: Calls and Puts on US equities and ETFs
- **Order Types**: Market, Limit, Stop orders
- **Expiration Management**: Automatic exercise and assignment handling
- **Paper Trading**: Full options testing in paper environment

### Options Trading Implementation
```python
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, AssetClass

class OptionsTrading:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

    def get_option_chain(self, underlying_symbol, expiration_date=None):
        """Get option chain for underlying asset"""
        # Note: This would typically use a separate market data call
        # For demonstration, showing the structure
        try:
            assets = self.trading_client.get_all_assets(
                status='active',
                asset_class=AssetClass.US_OPTION
            )

            # Filter options for specific underlying
            option_chain = []
            for asset in assets:
                if underlying_symbol in asset.symbol:
                    option_chain.append({
                        'symbol': asset.symbol,
                        'name': asset.name,
                        'tradable': asset.tradable,
                        'underlying': underlying_symbol
                    })

            return option_chain
        except Exception as e:
            print(f"Error getting option chain: {e}")
            return []

    def buy_call_option(self, option_symbol, quantity):
        """Buy call option"""
        order_request = MarketOrderRequest(
            symbol=option_symbol,  # e.g., "AAPL240315C00150000"
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error buying call option: {e}")
            return None

    def sell_put_option(self, option_symbol, quantity):
        """Sell put option (cash-secured put strategy)"""
        order_request = MarketOrderRequest(
            symbol=option_symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error selling put option: {e}")
            return None

    def covered_call_strategy(self, underlying_symbol, shares_owned, call_option_symbol):
        """Implement covered call strategy"""
        # Assume we already own the underlying shares
        # Sell call option against the position

        contracts_to_sell = shares_owned // 100  # Each contract covers 100 shares

        if contracts_to_sell > 0:
            order = self.sell_call_option(call_option_symbol, contracts_to_sell)
            return order
        else:
            print("Not enough shares for covered call strategy")
            return None

    def sell_call_option(self, option_symbol, quantity):
        """Sell call option"""
        order_request = MarketOrderRequest(
            symbol=option_symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error selling call option: {e}")
            return None

    def bull_call_spread(self, lower_strike_call, higher_strike_call, quantity):
        """Implement bull call spread"""
        orders = []

        # Buy lower strike call
        buy_order = MarketOrderRequest(
            symbol=lower_strike_call,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        # Sell higher strike call
        sell_order = MarketOrderRequest(
            symbol=higher_strike_call,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        try:
            buy_result = self.trading_client.submit_order(buy_order)
            sell_result = self.trading_client.submit_order(sell_order)
            orders = [buy_result, sell_result]
        except Exception as e:
            print(f"Error executing bull call spread: {e}")

        return orders

    def iron_condor(self, put_spread_lower, put_spread_higher, call_spread_lower, call_spread_higher, quantity):
        """Implement iron condor strategy"""
        orders = []

        # Sell put spread (sell higher strike, buy lower strike)
        sell_put_order = MarketOrderRequest(
            symbol=put_spread_higher,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        buy_put_order = MarketOrderRequest(
            symbol=put_spread_lower,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        # Sell call spread (sell lower strike, buy higher strike)
        sell_call_order = MarketOrderRequest(
            symbol=call_spread_lower,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        buy_call_order = MarketOrderRequest(
            symbol=call_spread_higher,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        order_requests = [sell_put_order, buy_put_order, sell_call_order, buy_call_order]

        for order_request in order_requests:
            try:
                order = self.trading_client.submit_order(order_request)
                orders.append(order)
            except Exception as e:
                print(f"Error in iron condor leg: {e}")

        return orders

# Example usage
options_trader = OptionsTrading("api_key", "secret_key", paper=True)

# Buy AAPL call option
options_trader.buy_call_option("AAPL240315C00150000", 1)

# Implement covered call strategy
options_trader.covered_call_strategy("AAPL", 100, "AAPL240315C00160000")

# Execute bull call spread
options_trader.bull_call_spread("AAPL240315C00150000", "AAPL240315C00160000", 1)
```

## 2024 Institutional Features

### FIX API for High-Frequency Trading
```python
# Note: FIX API implementation would require separate FIX protocol library
class InstitutionalFeatures:
    def __init__(self):
        self.fix_enabled = True  # Available for institutional accounts

    def high_frequency_execution(self):
        """
        Features for institutional high-frequency trading:
        - Direct Market Access (DMA)
        - Reduced latency execution
        - Dedicated connection
        - High throughput capacity
        """
        return {
            'fix_api': True,
            'dedicated_connection': True,
            'low_latency': True,
            'high_throughput': True
        }

    def algorithmic_execution(self):
        """
        Advanced algorithmic execution features:
        - TWAP (Time Weighted Average Price)
        - VWAP (Volume Weighted Average Price)
        - Implementation Shortfall
        - Participation Rate
        """
        return {
            'twap': True,
            'vwap': True,
            'implementation_shortfall': True,
            'participation_rate': True
        }
```

### Local Currency Trading
```python
class LocalCurrencyTrading:
    def __init__(self, api_key, secret_key):
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    def trade_in_local_currency(self, symbol, amount_usd, target_currency='EUR'):
        """
        Trade using local currency pricing
        - Reduced FX volatility
        - Stable pricing in USD
        - Extended hours support
        """
        # This feature allows international users to trade
        # with USD-denominated pricing while thinking in local currency

        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=amount_usd,  # Priced in stable USD
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            extended_hours=True  # Available in extended hours
        )

        try:
            order = self.trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error with local currency trading: {e}")
            return None
```

## Advanced Strategy Integration

### Multi-Asset Strategy Framework
```python
class AdvancedTradingFramework:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.fractional_trader = FractionalTrading(api_key, secret_key, paper)
        self.crypto_trader = CryptoTrading(api_key, secret_key, paper)
        self.options_trader = OptionsTrading(api_key, secret_key, paper)

    def diversified_portfolio_strategy(self, total_capital):
        """
        Advanced strategy using all asset classes:
        - 60% Equities (including fractional shares)
        - 20% Crypto
        - 15% Options strategies
        - 5% Cash
        """
        equity_allocation = total_capital * 0.60
        crypto_allocation = total_capital * 0.20
        options_allocation = total_capital * 0.15

        orders = []

        # Equity investments with fractional shares
        equity_symbols = ["SPY", "QQQ", "VTI", "AAPL", "MSFT"]
        equity_per_symbol = equity_allocation / len(equity_symbols)

        for symbol in equity_symbols:
            order = self.fractional_trader.buy_notional_amount(symbol, equity_per_symbol)
            if order:
                orders.append(order)

        # Crypto investments
        crypto_symbols = ["BTC/USD", "ETH/USD"]
        crypto_per_symbol = crypto_allocation / len(crypto_symbols)

        for symbol in crypto_symbols:
            order = self.crypto_trader.buy_crypto_market(symbol, crypto_per_symbol)
            if order:
                orders.append(order)

        # Options strategies (covered calls, cash-secured puts)
        # Implementation would depend on current positions and market conditions

        return orders

    def adaptive_rebalancing(self):
        """
        Intelligent rebalancing across all asset classes
        """
        # Get current portfolio state
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()

        # Analyze current allocations vs targets
        # Rebalance using fractional shares for precision
        # Adjust crypto positions for 24/7 availability
        # Manage options positions for income generation

        pass

    def risk_parity_implementation(self):
        """
        Risk parity strategy using advanced features
        """
        # Use fractional shares for precise allocation
        # Include crypto for diversification
        # Use options for downside protection
        pass
```

This comprehensive research covers Alpaca's advanced trading features available in 2024, providing the foundation for building sophisticated trading strategies that leverage fractional shares, cryptocurrency trading, and options strategies.