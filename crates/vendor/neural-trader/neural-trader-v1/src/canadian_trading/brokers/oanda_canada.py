"""
OANDA Canada Forex Trading Integration
Implements v20 REST API with streaming rates, position management, and advanced forex features
"""

import json
import asyncio
import websockets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
import numpy as np
import pandas as pd
import requests
from collections import defaultdict, deque
import threading
import time

# OANDA v20 API imports
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints import (
    accounts, orders, positions, pricing, trades,
    instruments, transactions
)
from oandapyV20.contrib.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    TakeProfitDetails, StopLossDetails, TrailingStopLossDetails
)
from oandapyV20.exceptions import V20Error, StreamTerminated

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ForexSignal:
    """Forex trading signal with confidence and risk metrics"""
    instrument: str
    direction: str  # 'buy' or 'sell'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    volatility: float
    spread_impact: float
    optimal_execution_time: datetime
    risk_reward_ratio: float
    kelly_position_size: float
    market_session: str  # 'asian', 'european', 'american'


@dataclass
class SpreadAnalysis:
    """Spread analysis for forex pairs"""
    current_spread: float
    average_spread: float
    spread_percentile: float
    is_favorable: bool
    spread_history: List[float]
    session_analysis: Dict[str, float]  # Average spread by session


@dataclass
class MarginInfo:
    """Margin calculation information"""
    required_margin: float
    available_margin: float
    margin_ratio: float
    margin_call_level: float
    margin_closeout_level: float
    position_margin: float
    order_margin: float
    

class OANDACanada:
    """
    OANDA Canada Forex Trading Integration
    Supports v20 REST API, WebSocket streaming, advanced risk management
    """
    
    # CAD pairs specialization
    CAD_MAJOR_PAIRS = [
        'USD_CAD', 'EUR_CAD', 'GBP_CAD', 'AUD_CAD', 
        'CAD_JPY', 'CAD_CHF', 'NZD_CAD', 'CAD_SGD'
    ]
    
    # Optimal trading sessions for CAD pairs
    TRADING_SESSIONS = {
        'asian': {'start': '00:00', 'end': '08:00'},      # Tokyo session
        'european': {'start': '08:00', 'end': '16:00'},   # London session
        'american': {'start': '13:00', 'end': '21:00'}    # New York session
    }
    
    def __init__(self, api_token: str, account_id: str, environment: str = 'practice'):
        """
        Initialize OANDA Canada integration
        
        Args:
            api_token: OANDA API v20 token
            account_id: OANDA account ID
            environment: 'practice' or 'live'
        """
        self.api_token = api_token
        self.account_id = account_id
        self.environment = environment
        
        # Initialize v20 API client
        self.api = API(access_token=api_token, environment=environment)
        
        # Streaming components
        self.stream_api = None
        self.price_stream_task = None
        self.transaction_stream_task = None
        self.streaming_active = False
        
        # Market data storage
        self.price_cache = defaultdict(lambda: deque(maxlen=1000))
        self.spread_history = defaultdict(lambda: deque(maxlen=10000))
        self.tick_data = defaultdict(list)
        
        # Position and risk tracking
        self.positions = {}
        self.pending_orders = {}
        self.account_info = None
        self.margin_info = None
        
        # Performance metrics
        self.execution_metrics = defaultdict(list)
        self.slippage_history = deque(maxlen=1000)
        
        # WebSocket URLs based on environment
        self.stream_url = self._get_stream_url()
        
        # Initialize account
        self._initialize_account()
        
    def _get_stream_url(self) -> str:
        """Get WebSocket streaming URL based on environment"""
        if self.environment == 'practice':
            return "wss://stream-fxpractice.oanda.com/v3/accounts/{}/pricing/stream"
        else:
            return "wss://stream-fxtrade.oanda.com/v3/accounts/{}/pricing/stream"
            
    def _initialize_account(self):
        """Initialize account information and settings"""
        try:
            # Get account details
            r = accounts.AccountDetails(self.account_id)
            response = self.api.request(r)
            self.account_info = response['account']
            
            # Update margin information
            self._update_margin_info()
            
            logger.info(f"OANDA Canada account initialized: {self.account_id}")
            logger.info(f"Account currency: {self.account_info['currency']}")
            logger.info(f"Balance: {self.account_info['balance']}")
            
        except V20Error as e:
            logger.error(f"Failed to initialize OANDA account: {e}")
            raise
            
    def _update_margin_info(self):
        """Update margin information from account"""
        if self.account_info:
            self.margin_info = MarginInfo(
                required_margin=float(self.account_info.get('marginUsed', 0)),
                available_margin=float(self.account_info.get('marginAvailable', 0)),
                margin_ratio=float(self.account_info.get('marginRate', 0)),
                margin_call_level=float(self.account_info.get('marginCallPercent', 0)),
                margin_closeout_level=float(self.account_info.get('marginCloseoutPercent', 0)),
                position_margin=float(self.account_info.get('positionValue', 0)),
                order_margin=float(self.account_info.get('unrealizedPL', 0))
            )
    
    async def start_streaming(self, instruments: List[str]):
        """
        Start WebSocket streaming for real-time forex rates
        
        Args:
            instruments: List of instruments to stream (e.g., ['USD_CAD', 'EUR_CAD'])
        """
        self.streaming_active = True
        
        # Start price streaming
        self.price_stream_task = asyncio.create_task(
            self._stream_prices(instruments)
        )
        
        # Start transaction streaming
        self.transaction_stream_task = asyncio.create_task(
            self._stream_transactions()
        )
        
        logger.info(f"Started streaming for instruments: {instruments}")
        
    async def stop_streaming(self):
        """Stop all streaming connections"""
        self.streaming_active = False
        
        if self.price_stream_task:
            self.price_stream_task.cancel()
            
        if self.transaction_stream_task:
            self.transaction_stream_task.cancel()
            
        logger.info("Stopped all streaming connections")
        
    async def _stream_prices(self, instruments: List[str]):
        """Stream real-time prices via WebSocket"""
        headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        instruments_str = ','.join(instruments)
        url = f"{self.stream_url.format(self.account_id)}?instruments={instruments_str}"
        
        while self.streaming_active:
            try:
                async with websockets.connect(url, extra_headers=headers) as websocket:
                    logger.info("Price streaming connected")
                    
                    async for message in websocket:
                        if not self.streaming_active:
                            break
                            
                        data = json.loads(message)
                        await self._process_price_update(data)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Price stream connection closed, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Price streaming error: {e}")
                await asyncio.sleep(5)
                
    async def _stream_transactions(self):
        """Stream real-time transaction updates"""
        # Similar WebSocket implementation for transaction streaming
        pass
        
    async def _process_price_update(self, data: Dict):
        """Process incoming price updates"""
        if data.get('type') == 'PRICE':
            instrument = data['instrument']
            
            # Extract bid/ask prices
            bid = float(data['bids'][0]['price']) if data.get('bids') else None
            ask = float(data['asks'][0]['price']) if data.get('asks') else None
            
            if bid and ask:
                # Calculate spread
                spread = ask - bid
                spread_pips = self._calculate_pips(instrument, spread)
                
                # Update price cache
                price_data = {
                    'timestamp': datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'spread_pips': spread_pips
                }
                
                self.price_cache[instrument].append(price_data)
                self.spread_history[instrument].append((price_data['timestamp'], spread_pips))
                
                # Analyze spread
                await self._analyze_spread(instrument)
                
    def _calculate_pips(self, instrument: str, price_difference: float) -> float:
        """Calculate pips for a given price difference"""
        if 'JPY' in instrument:
            return price_difference * 100
        else:
            return price_difference * 10000
            
    async def _analyze_spread(self, instrument: str):
        """Analyze spread patterns and optimal execution times"""
        if len(self.spread_history[instrument]) < 100:
            return
            
        spreads = [s[1] for s in self.spread_history[instrument]]
        current_spread = spreads[-1]
        
        # Calculate spread statistics
        avg_spread = np.mean(spreads)
        spread_percentile = np.percentile(spreads, [25, 50, 75])
        
        # Determine if current spread is favorable
        is_favorable = current_spread <= spread_percentile[1]  # Below median
        
        # Session-based analysis
        session_spreads = self._analyze_session_spreads(instrument)
        
        # Store analysis
        self.spread_analysis = SpreadAnalysis(
            current_spread=current_spread,
            average_spread=avg_spread,
            spread_percentile=np.percentile(spreads, int((current_spread / max(spreads)) * 100)),
            is_favorable=is_favorable,
            spread_history=spreads[-100:],
            session_analysis=session_spreads
        )
        
    def _analyze_session_spreads(self, instrument: str) -> Dict[str, float]:
        """Analyze average spreads by trading session"""
        session_spreads = defaultdict(list)
        
        for timestamp, spread in list(self.spread_history[instrument])[-1000:]:
            hour = timestamp.hour
            
            # Determine session
            if 0 <= hour < 8:
                session = 'asian'
            elif 8 <= hour < 16:
                session = 'european'
            else:
                session = 'american'
                
            session_spreads[session].append(spread)
            
        # Calculate averages
        return {
            session: np.mean(spreads) if spreads else 0
            for session, spreads in session_spreads.items()
        }
        
    def get_optimal_execution_time(self, instrument: str, direction: str) -> datetime:
        """
        Determine optimal execution time based on spread analysis and market sessions
        
        Args:
            instrument: Forex pair (e.g., 'USD_CAD')
            direction: 'buy' or 'sell'
            
        Returns:
            Optimal execution datetime
        """
        now = datetime.now()
        
        # Get session analysis
        if hasattr(self, 'spread_analysis'):
            session_spreads = self.spread_analysis.session_analysis
            
            # Find session with lowest average spread
            best_session = min(session_spreads.items(), key=lambda x: x[1])[0]
            
            # Calculate next occurrence of best session
            session_times = self.TRADING_SESSIONS[best_session]
            session_start = datetime.strptime(session_times['start'], '%H:%M').time()
            
            optimal_time = datetime.combine(now.date(), session_start)
            
            # If session already passed today, use tomorrow
            if optimal_time < now:
                optimal_time += timedelta(days=1)
                
            return optimal_time
        else:
            # Default to current time if no spread analysis available
            return now
            
    def calculate_position_size(self, 
                              instrument: str,
                              signal: ForexSignal,
                              risk_percentage: float = 0.02) -> int:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        
        Args:
            instrument: Forex pair
            signal: Trading signal with confidence and risk metrics
            risk_percentage: Maximum risk per trade (default 2%)
            
        Returns:
            Position size in units
        """
        if not self.account_info:
            self._initialize_account()
            
        account_balance = float(self.account_info['balance'])
        
        # Kelly Criterion calculation
        win_probability = signal.confidence
        win_loss_ratio = signal.risk_reward_ratio
        
        # Kelly percentage = (bp - q) / b
        # where b = odds, p = probability of winning, q = probability of losing
        kelly_percentage = (win_loss_ratio * win_probability - (1 - win_probability)) / win_loss_ratio
        
        # Apply Kelly fraction (typically 0.25 to be conservative)
        kelly_fraction = 0.25
        position_percentage = kelly_percentage * kelly_fraction
        
        # Apply maximum risk constraint
        position_percentage = min(position_percentage, risk_percentage)
        
        # Calculate position value
        position_value = account_balance * position_percentage
        
        # Convert to units based on current price
        current_price = self._get_current_price(instrument)
        
        # Adjust for instrument precision
        if 'JPY' in instrument:
            units = int(position_value / (current_price * 0.01))
        else:
            units = int(position_value / current_price)
            
        # Ensure minimum position size
        min_units = 1000  # OANDA minimum
        units = max(units, min_units)
        
        # Check margin requirements
        margin_required = self._calculate_margin_requirement(instrument, units)
        if margin_required > self.margin_info.available_margin:
            # Reduce position size to fit margin
            units = int(units * (self.margin_info.available_margin / margin_required) * 0.95)
            
        return units
        
    def _calculate_margin_requirement(self, instrument: str, units: int) -> float:
        """Calculate margin requirement for a position"""
        # Get instrument details
        r = instruments.InstrumentsDetails(accountID=self.account_id, instruments=instrument)
        response = self.api.request(r)
        
        margin_rate = float(response['instruments'][0]['marginRate'])
        current_price = self._get_current_price(instrument)
        
        # Calculate margin requirement
        position_value = units * current_price
        margin_required = position_value * margin_rate
        
        return margin_required
        
    def _get_current_price(self, instrument: str) -> float:
        """Get current mid price for instrument"""
        if instrument in self.price_cache and self.price_cache[instrument]:
            latest = self.price_cache[instrument][-1]
            return (latest['bid'] + latest['ask']) / 2
        else:
            # Fetch from API if not in cache
            params = {"instruments": instrument}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            
            prices = response['prices'][0]
            bid = float(prices['bids'][0]['price'])
            ask = float(prices['asks'][0]['price'])
            
            return (bid + ask) / 2
            
    async def execute_forex_signal(self, signal: ForexSignal) -> Dict:
        """
        Execute forex trade based on signal with advanced order management
        
        Args:
            signal: ForexSignal with entry, stop loss, take profit
            
        Returns:
            Execution result with order details
        """
        try:
            # Check if we should wait for optimal execution time
            optimal_time = signal.optimal_execution_time
            current_time = datetime.now()
            
            if optimal_time > current_time:
                wait_seconds = (optimal_time - current_time).total_seconds()
                if wait_seconds < 3600:  # Wait up to 1 hour
                    logger.info(f"Waiting {wait_seconds/60:.1f} minutes for optimal execution time")
                    await asyncio.sleep(wait_seconds)
                    
            # Calculate position size
            units = self.calculate_position_size(
                signal.instrument,
                signal,
                risk_percentage=0.02
            )
            
            # Adjust units for direction
            if signal.direction == 'sell':
                units = -units
                
            # Create order with stop loss and take profit
            order_data = MarketOrderRequest(
                instrument=signal.instrument,
                units=units,
                takeProfitOnFill=TakeProfitDetails(price=signal.take_profit),
                stopLossOnFill=StopLossDetails(price=signal.stop_loss)
            )
            
            # Add trailing stop if high confidence
            if signal.confidence > 0.8:
                trailing_distance = abs(signal.entry_price - signal.stop_loss) * 0.5
                order_data.trailingStopLossOnFill = TrailingStopLossDetails(
                    distance=trailing_distance
                )
                
            # Submit order
            r = orders.OrderCreate(self.account_id, data=order_data.data)
            response = self.api.request(r)
            
            # Track execution metrics
            execution_time = datetime.now()
            order_result = response.get('orderFillTransaction', {})
            
            if order_result:
                # Calculate slippage
                executed_price = float(order_result.get('price', signal.entry_price))
                slippage = executed_price - signal.entry_price
                slippage_pips = self._calculate_pips(signal.instrument, abs(slippage))
                
                # Store metrics
                self.execution_metrics[signal.instrument].append({
                    'timestamp': execution_time,
                    'signal_price': signal.entry_price,
                    'executed_price': executed_price,
                    'slippage_pips': slippage_pips,
                    'units': units,
                    'spread_at_execution': self.spread_analysis.current_spread if hasattr(self, 'spread_analysis') else None
                })
                
                self.slippage_history.append(slippage_pips)
                
                return {
                    'status': 'success',
                    'order_id': order_result.get('id'),
                    'instrument': signal.instrument,
                    'units': units,
                    'executed_price': executed_price,
                    'slippage_pips': slippage_pips,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'execution_time': execution_time.isoformat(),
                    'margin_used': self._calculate_margin_requirement(signal.instrument, abs(units))
                }
            else:
                return {
                    'status': 'failed',
                    'reason': response.get('errorMessage', 'Unknown error'),
                    'instrument': signal.instrument,
                    'signal': signal
                }
                
        except V20Error as e:
            logger.error(f"Order execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instrument': signal.instrument
            }
            
    def get_positions(self) -> Dict[str, Any]:
        """Get all open positions with detailed information"""
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.api.request(r)
            
            positions = {}
            for position in response['positions']:
                instrument = position['instrument']
                
                # Calculate current P&L
                long_units = float(position['long']['units'])
                short_units = float(position['short']['units'])
                
                if long_units > 0:
                    units = long_units
                    average_price = float(position['long']['averagePrice'])
                    unrealized_pl = float(position['long']['unrealizedPL'])
                else:
                    units = short_units
                    average_price = float(position['short']['averagePrice'])
                    unrealized_pl = float(position['short']['unrealizedPL'])
                    
                current_price = self._get_current_price(instrument)
                
                positions[instrument] = {
                    'units': units,
                    'average_price': average_price,
                    'current_price': current_price,
                    'unrealized_pl': unrealized_pl,
                    'margin_used': float(position.get('marginUsed', 0)),
                    'pl_percentage': (unrealized_pl / abs(units * average_price)) * 100 if average_price else 0
                }
                
            self.positions = positions
            return positions
            
        except V20Error as e:
            logger.error(f"Failed to get positions: {e}")
            return {}
            
    def close_position(self, instrument: str, units: Optional[int] = None) -> Dict:
        """
        Close a position (fully or partially)
        
        Args:
            instrument: Forex pair to close
            units: Number of units to close (None for full close)
            
        Returns:
            Close transaction details
        """
        try:
            if units is None:
                # Close all units
                r = positions.PositionClose(accountID=self.account_id, instrument=instrument)
            else:
                # Close specific units
                data = {"units": str(units)}
                r = positions.PositionClose(
                    accountID=self.account_id,
                    instrument=instrument,
                    data=data
                )
                
            response = self.api.request(r)
            
            return {
                'status': 'success',
                'instrument': instrument,
                'units_closed': units,
                'transaction': response
            }
            
        except V20Error as e:
            logger.error(f"Failed to close position: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instrument': instrument
            }
            
    def modify_position(self, 
                       instrument: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None,
                       trailing_stop: Optional[float] = None) -> Dict:
        """Modify an existing position's risk parameters"""
        try:
            # Get current position
            positions = self.get_positions()
            if instrument not in positions:
                return {
                    'status': 'error',
                    'error': f'No position found for {instrument}'
                }
                
            # Build modification data
            data = {}
            if stop_loss is not None:
                data['stopLoss'] = {"price": str(stop_loss)}
            if take_profit is not None:
                data['takeProfit'] = {"price": str(take_profit)}
            if trailing_stop is not None:
                data['trailingStopLoss'] = {"distance": str(trailing_stop)}
                
            # Submit modification
            r = trades.TradeCRCDO(
                accountID=self.account_id,
                tradeID=self._get_trade_id(instrument),
                data=data
            )
            
            response = self.api.request(r)
            
            return {
                'status': 'success',
                'instrument': instrument,
                'modifications': data,
                'response': response
            }
            
        except V20Error as e:
            logger.error(f"Failed to modify position: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instrument': instrument
            }
            
    def _get_trade_id(self, instrument: str) -> str:
        """Get trade ID for an instrument"""
        r = trades.TradesList(accountID=self.account_id, params={"instrument": instrument})
        response = self.api.request(r)
        
        if response['trades']:
            return response['trades'][0]['id']
        else:
            raise ValueError(f"No trade found for {instrument}")
            
    def get_spread_analysis(self, instrument: str) -> SpreadAnalysis:
        """Get current spread analysis for an instrument"""
        if hasattr(self, 'spread_analysis'):
            return self.spread_analysis
        else:
            # Return basic analysis if streaming not active
            current_price = self._get_current_price(instrument)
            params = {"instruments": instrument}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            response = self.api.request(r)
            
            prices = response['prices'][0]
            bid = float(prices['bids'][0]['price'])
            ask = float(prices['asks'][0]['price'])
            spread = ask - bid
            spread_pips = self._calculate_pips(instrument, spread)
            
            return SpreadAnalysis(
                current_spread=spread_pips,
                average_spread=spread_pips,
                spread_percentile=50.0,
                is_favorable=True,
                spread_history=[spread_pips],
                session_analysis={'current': spread_pips}
            )
            
    def calculate_forex_risk_metrics(self, positions: Dict) -> Dict:
        """Calculate comprehensive risk metrics for forex positions"""
        if not positions:
            return {
                'total_exposure': 0,
                'margin_utilization': 0,
                'largest_position': None,
                'correlation_risk': 'low',
                'recommendations': []
            }
            
        # Calculate total exposure
        total_exposure = sum(abs(pos['units'] * pos['current_price']) for pos in positions.values())
        
        # Margin utilization
        self._update_margin_info()
        margin_utilization = (self.margin_info.required_margin / 
                            (self.margin_info.required_margin + self.margin_info.available_margin)) * 100
                            
        # Find largest position
        largest_position = max(positions.items(), 
                             key=lambda x: abs(x[1]['units'] * x[1]['current_price']))
                             
        # Analyze correlation risk for CAD pairs
        cad_exposure = sum(abs(pos['units'] * pos['current_price']) 
                          for pair, pos in positions.items() 
                          if 'CAD' in pair)
        correlation_risk = 'high' if cad_exposure / total_exposure > 0.6 else 'medium' if cad_exposure / total_exposure > 0.4 else 'low'
        
        # Generate recommendations
        recommendations = []
        if margin_utilization > 80:
            recommendations.append("High margin utilization - consider reducing position sizes")
        if correlation_risk == 'high':
            recommendations.append("High CAD correlation - diversify into other currencies")
        if len(positions) < 3:
            recommendations.append("Low diversification - consider adding more currency pairs")
            
        return {
            'total_exposure': total_exposure,
            'margin_utilization': margin_utilization,
            'largest_position': {
                'instrument': largest_position[0],
                'exposure': abs(largest_position[1]['units'] * largest_position[1]['current_price'])
            },
            'correlation_risk': correlation_risk,
            'cad_exposure_percentage': (cad_exposure / total_exposure * 100) if total_exposure > 0 else 0,
            'recommendations': recommendations,
            'var_estimate': self._calculate_portfolio_var(positions),
            'margin_call_price_levels': self._calculate_margin_call_levels(positions)
        }
        
    def _calculate_portfolio_var(self, positions: Dict, confidence: float = 0.95) -> float:
        """Calculate Value at Risk for forex portfolio"""
        if not positions:
            return 0.0
            
        # Simplified VaR calculation using historical volatility
        # In production, use more sophisticated methods
        portfolio_value = sum(pos['units'] * pos['current_price'] for pos in positions.values())
        
        # Assume 2% daily volatility for forex (adjustable)
        daily_volatility = 0.02
        
        # Calculate VaR
        z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99% confidence
        var = portfolio_value * daily_volatility * z_score
        
        return var
        
    def _calculate_margin_call_levels(self, positions: Dict) -> Dict[str, float]:
        """Calculate price levels that would trigger margin calls"""
        margin_call_levels = {}
        
        for instrument, position in positions.items():
            # Calculate adverse price movement that would trigger margin call
            current_price = position['current_price']
            units = position['units']
            
            # Simplified calculation - in production use exact margin formulas
            if units > 0:  # Long position
                margin_call_price = current_price * (1 - self.margin_info.margin_call_level)
            else:  # Short position
                margin_call_price = current_price * (1 + self.margin_info.margin_call_level)
                
            margin_call_levels[instrument] = margin_call_price
            
        return margin_call_levels
        
    def get_execution_analytics(self, lookback_days: int = 30) -> Dict:
        """Get execution quality analytics"""
        analytics = {
            'average_slippage': 0,
            'positive_slippage_rate': 0,
            'best_execution_session': None,
            'worst_execution_session': None,
            'execution_by_pair': {},
            'recommendations': []
        }
        
        if not self.execution_metrics:
            return analytics
            
        # Analyze slippage
        all_slippages = []
        for instrument, executions in self.execution_metrics.items():
            recent_executions = [
                e for e in executions 
                if (datetime.now() - e['timestamp']).days <= lookback_days
            ]
            
            if recent_executions:
                slippages = [e['slippage_pips'] for e in recent_executions]
                all_slippages.extend(slippages)
                
                analytics['execution_by_pair'][instrument] = {
                    'average_slippage': np.mean(slippages),
                    'execution_count': len(slippages),
                    'positive_slippage_rate': sum(1 for s in slippages if s < 0) / len(slippages)
                }
                
        if all_slippages:
            analytics['average_slippage'] = np.mean(all_slippages)
            analytics['positive_slippage_rate'] = sum(1 for s in all_slippages if s < 0) / len(all_slippages)
            
            # Analyze by session
            session_slippages = defaultdict(list)
            for instrument, executions in self.execution_metrics.items():
                for e in executions:
                    hour = e['timestamp'].hour
                    if 0 <= hour < 8:
                        session = 'asian'
                    elif 8 <= hour < 16:
                        session = 'european'
                    else:
                        session = 'american'
                    session_slippages[session].append(e['slippage_pips'])
                    
            if session_slippages:
                session_averages = {s: np.mean(slips) for s, slips in session_slippages.items()}
                analytics['best_execution_session'] = min(session_averages.items(), key=lambda x: x[1])
                analytics['worst_execution_session'] = max(session_averages.items(), key=lambda x: x[1])
                
            # Recommendations
            if analytics['average_slippage'] > 1.0:
                analytics['recommendations'].append("High average slippage - consider using limit orders")
            if analytics['positive_slippage_rate'] < 0.3:
                analytics['recommendations'].append("Low positive slippage rate - review execution timing")
                
        return analytics
        
    def create_advanced_order(self,
                            instrument: str,
                            units: int,
                            order_type: str = 'MARKET',
                            price: Optional[float] = None,
                            distance: Optional[float] = None,
                            stop_loss: Optional[float] = None,
                            take_profit: Optional[float] = None,
                            trailing_stop: Optional[float] = None,
                            expiry: Optional[datetime] = None) -> Dict:
        """
        Create advanced order types including OCO (One Cancels Other) and contingent orders
        
        Args:
            instrument: Forex pair
            units: Position size (positive for buy, negative for sell)
            order_type: 'MARKET', 'LIMIT', 'STOP', 'MARKET_IF_TOUCHED'
            price: Price for limit/stop orders
            distance: Distance for trailing stops
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop: Trailing stop distance
            expiry: Order expiration time
            
        Returns:
            Order creation result
        """
        try:
            # Build order data based on type
            if order_type == 'MARKET':
                order_data = MarketOrderRequest(
                    instrument=instrument,
                    units=units
                )
            elif order_type == 'LIMIT':
                order_data = LimitOrderRequest(
                    instrument=instrument,
                    units=units,
                    price=price
                )
            elif order_type == 'STOP':
                order_data = StopOrderRequest(
                    instrument=instrument,
                    units=units,
                    price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
                
            # Add risk management parameters
            if stop_loss:
                order_data.stopLossOnFill = StopLossDetails(price=stop_loss)
            if take_profit:
                order_data.takeProfitOnFill = TakeProfitDetails(price=take_profit)
            if trailing_stop:
                order_data.trailingStopLossOnFill = TrailingStopLossDetails(distance=trailing_stop)
                
            # Add expiry if specified
            if expiry:
                order_data.timeInForce = "GTD"
                order_data.gtdTime = expiry.isoformat()
                
            # Submit order
            r = orders.OrderCreate(self.account_id, data=order_data.data)
            response = self.api.request(r)
            
            return {
                'status': 'success',
                'order_id': response['orderCreateTransaction']['id'],
                'instrument': instrument,
                'type': order_type,
                'units': units,
                'response': response
            }
            
        except V20Error as e:
            logger.error(f"Failed to create advanced order: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instrument': instrument,
                'type': order_type
            }
            
    def get_margin_closeout_calculator(self, positions: Dict) -> Dict:
        """Calculate detailed margin closeout scenarios"""
        if not positions or not self.margin_info:
            return {
                'current_margin_level': 0,
                'closeout_level': 0,
                'buffer_percentage': 100,
                'worst_case_scenarios': {}
            }
            
        current_nav = float(self.account_info['NAV'])
        margin_used = self.margin_info.required_margin
        margin_level = (current_nav / margin_used * 100) if margin_used > 0 else float('inf')
        
        closeout_level = self.margin_info.margin_closeout_level * 100
        buffer_percentage = ((margin_level - closeout_level) / closeout_level * 100) if closeout_level > 0 else 100
        
        # Calculate worst-case scenarios
        worst_case_scenarios = {}
        
        for instrument, position in positions.items():
            units = position['units']
            current_price = position['current_price']
            
            # Calculate price movement that would trigger closeout
            if units > 0:  # Long position
                closeout_price = current_price * (1 - (buffer_percentage / 100))
                price_movement_pips = self._calculate_pips(instrument, current_price - closeout_price)
            else:  # Short position
                closeout_price = current_price * (1 + (buffer_percentage / 100))
                price_movement_pips = self._calculate_pips(instrument, closeout_price - current_price)
                
            worst_case_scenarios[instrument] = {
                'current_price': current_price,
                'closeout_price': closeout_price,
                'price_movement_pips': price_movement_pips,
                'percentage_move': abs((closeout_price - current_price) / current_price * 100)
            }
            
        return {
            'current_margin_level': margin_level,
            'closeout_level': closeout_level,
            'buffer_percentage': buffer_percentage,
            'margin_used': margin_used,
            'margin_available': self.margin_info.available_margin,
            'worst_case_scenarios': worst_case_scenarios,
            'recommendations': self._generate_margin_recommendations(margin_level, buffer_percentage)
        }
        
    def _generate_margin_recommendations(self, margin_level: float, buffer: float) -> List[str]:
        """Generate margin management recommendations"""
        recommendations = []
        
        if margin_level < 200:
            recommendations.append("CRITICAL: Margin level below 200% - immediate risk reduction needed")
        elif margin_level < 300:
            recommendations.append("WARNING: Low margin level - consider reducing position sizes")
            
        if buffer < 50:
            recommendations.append("Low margin buffer - vulnerable to moderate market movements")
            
        if self.margin_info.margin_ratio > 0.3:
            recommendations.append("High leverage usage - consider more conservative position sizing")
            
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize OANDA Canada client
    oanda = OANDACanada(
        api_token="your_api_token_here",
        account_id="your_account_id_here",
        environment="practice"
    )
    
    # Example: Get optimal execution time
    optimal_time = oanda.get_optimal_execution_time("USD_CAD", "buy")
    print(f"Optimal execution time for USD/CAD: {optimal_time}")
    
    # Example: Create a trading signal
    signal = ForexSignal(
        instrument="USD_CAD",
        direction="buy",
        confidence=0.75,
        entry_price=1.3500,
        stop_loss=1.3450,
        take_profit=1.3600,
        volatility=0.008,
        spread_impact=0.0002,
        optimal_execution_time=optimal_time,
        risk_reward_ratio=2.0,
        kelly_position_size=0.05,
        market_session="american"
    )
    
    # Example: Calculate position size
    position_size = oanda.calculate_position_size("USD_CAD", signal)
    print(f"Calculated position size: {position_size} units")
    
    # Example: Get spread analysis
    spread_analysis = oanda.get_spread_analysis("USD_CAD")
    print(f"Current spread: {spread_analysis.current_spread} pips")
    print(f"Is favorable: {spread_analysis.is_favorable}")