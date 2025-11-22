"""
CWTS Ultra High-Performance FreqTrade Strategy
Base class for strategies using CWTS Ultra execution engine
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import talib.abstract as ta
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

# Import the CWTS client
try:
    # Try Cython version first (fastest)
    import cwts_client
    CWTS_AVAILABLE = True
    CWTS_MODE = "cython"
    logger.info("CWTS Ultra using Cython module (fastest)")
except ImportError:
    try:
        # Fall back to pure Python version
        import cwts_client_simple as cwts_client
        CWTS_AVAILABLE = True
        CWTS_MODE = "python"
        logger.info("CWTS Ultra using Python module with shared memory")
    except ImportError as e:
        # Try one more time with explicit path
        try:
            import sys
            import os
            strategy_dir = os.path.dirname(os.path.abspath(__file__))
            if strategy_dir not in sys.path:
                sys.path.insert(0, strategy_dir)
            import cwts_client_simple as cwts_client
            CWTS_AVAILABLE = True
            CWTS_MODE = "python"
            logger.info("CWTS Ultra using Python module (path adjusted)")
        except ImportError:
            CWTS_AVAILABLE = False
            CWTS_MODE = "websocket"
            logger.info("CWTS Ultra using WebSocket mode (still fast!)")

# WebSocket fallback - import only if needed
if CWTS_MODE == "websocket":
    try:
        import websockets
        import json
        import msgpack
    except ImportError:
        logger.warning("WebSocket dependencies not available")


class CWTSUltraStrategy(IStrategy):
    """
    Base strategy class for CWTS Ultra integration.
    
    Features:
    - Sub-millisecond signal transmission via shared memory
    - Real-time order book access
    - GPU-accelerated indicator computation (via CWTS)
    - Ultra-low latency execution
    """
    
    # Strategy version
    INTERFACE_VERSION = 3
    
    # CWTS Ultra specific settings
    use_cwts_ultra = True
    cwts_shm_path = "/dev/shm/cwts_ultra"
    cwts_websocket = "ws://localhost:4000"
    cwts_latency_mode = "ultra"  # "ultra", "low", "normal"
    
    # Can this strategy go short?
    can_short = False  # Set to False for spot trading
    
    # Minimal ROI designed for high-frequency trading (QUANTUM-INSPIRED)
    minimal_roi = {
        "0": 0.03,    # 3% profit (like Quantum)
        "10": 0.02,   # 2% after 10 minutes
        "30": 0.015,  # 1.5% after 30 minutes
        "60": 0.01,   # 1% after 60 minutes
        "120": 0.005  # 0.5% after 2 hours
    }
    
    # Stop loss (WIDER like Quantum)
    stoploss = -0.025  # 2.5% stop loss (was 2%)
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.002
    trailing_only_offset_is_reached = True
    
    # Timeframe for the strategy (QUANTUM SUCCESS)
    timeframe = '5m'  # Changed from 1m to 5m (like QuantumMomentum)
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    
    # CWTS Ultra parameters (LIBERAL like Quantum)
    cwts_confidence_threshold = DecimalParameter(0.3, 0.7, default=0.45, space="buy")  # Much lower!
    cwts_signal_strength = IntParameter(1, 10, default=5, space="buy")
    cwts_use_orderbook = True
    cwts_orderbook_depth = 20
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Initialize CWTS Ultra client
        self.cwts_client = None
        self.ws_client = None
        
        if self.use_cwts_ultra:
            self._init_cwts_client()
    
    def _init_cwts_client(self) -> None:
        """Initialize CWTS Ultra client with fallback to WebSocket."""
        try:
            if CWTS_AVAILABLE and self.cwts_latency_mode in ["ultra", "low"]:
                # Use high-performance Cython client
                self.cwts_client = cwts_client.create_client(
                    self.cwts_shm_path,
                    self.cwts_websocket
                )
                logger.info("CWTS Ultra client initialized (shared memory mode)")
            else:
                # Fallback to WebSocket
                self._init_websocket_client()
        except Exception as e:
            logger.error(f"Failed to initialize CWTS client: {e}")
            self._init_websocket_client()
    
    def _init_websocket_client(self) -> None:
        """Initialize WebSocket client as fallback."""
        self.ws_client = CWTSWebSocketClient(self.cwts_websocket)
        logger.info("CWTS Ultra client initialized (WebSocket mode)")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the dataframe.
        Uses CWTS Ultra for real-time data when available.
        """
        
        # Standard indicators (computed locally for backtesting)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Volume indicators
        dataframe['volume_ema'] = ta.EMA(dataframe['volume'], timeperiod=20)
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Custom CWTS indicators (if live trading)
        if self.cwts_client and hasattr(self, 'dp') and self.dp and hasattr(self.dp, 'runmode'):
            if self.dp.runmode.value in ['live', 'dry_run']:
                self._add_cwts_indicators(dataframe, metadata)
        
        return dataframe
    
    def _add_cwts_indicators(self, dataframe: DataFrame, metadata: dict) -> None:
        """Add real-time indicators from CWTS Ultra."""
        try:
            pair = metadata.get('pair')
            
            if self.cwts_client:
                # Get real-time market data
                market_data = self.cwts_client.get_market_data(pair)
                
                if market_data:
                    # Add real-time spread
                    dataframe.loc[dataframe.index[-1], 'rt_spread'] = market_data['spread']
                    dataframe.loc[dataframe.index[-1], 'rt_mid'] = market_data['mid']
                    
                    # Get order book imbalance
                    if self.cwts_use_orderbook:
                        orderbook = self.cwts_client.get_order_book(pair, self.cwts_orderbook_depth)
                        if len(orderbook) > 0:
                            bid_volume = orderbook[:, 1].sum()
                            ask_volume = orderbook[:, 3].sum()
                            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                            dataframe.loc[dataframe.index[-1], 'orderbook_imbalance'] = imbalance
                            
                            # Weighted mid price
                            weighted_bid = np.average(orderbook[:, 0], weights=orderbook[:, 1])
                            weighted_ask = np.average(orderbook[:, 2], weights=orderbook[:, 3])
                            dataframe.loc[dataframe.index[-1], 'weighted_mid'] = (weighted_bid + weighted_ask) / 2
        
        except Exception as e:
            logger.error(f"Error adding CWTS indicators: {e}")
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy/entry signals.
        Uses CWTS Ultra for signal transmission.
        """
        conditions_long = []
        conditions_short = []
        
        # Long entry conditions
        conditions_long.append(
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['rsi'] < 70) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['close'] > dataframe['bb_lower']) &
            (dataframe['volume'] > dataframe['volume_ema'])
        )
        
        # Short entry conditions (if enabled)
        if self.can_short:
            conditions_short.append(
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                (dataframe['rsi'] > 30) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['close'] < dataframe['bb_upper']) &
                (dataframe['volume'] > dataframe['volume_ema'])
            )
        
        # Apply conditions
        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'
            ] = 1
            
            # Send signal to CWTS Ultra
            self._send_entry_signal(dataframe, metadata, 'long')
        
        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'enter_short'
            ] = 1
            
            # Send signal to CWTS Ultra
            self._send_entry_signal(dataframe, metadata, 'short')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell/exit signals.
        Uses CWTS Ultra for signal transmission.
        """
        conditions_exit_long = []
        conditions_exit_short = []
        
        # Exit long conditions
        conditions_exit_long.append(
            (dataframe['ema_fast'] < dataframe['ema_slow']) |
            (dataframe['rsi'] > 80) |
            (dataframe['close'] < dataframe['bb_lower'])
        )
        
        # Exit short conditions
        if self.can_short:
            conditions_exit_short.append(
                (dataframe['ema_fast'] > dataframe['ema_slow']) |
                (dataframe['rsi'] < 20) |
                (dataframe['close'] > dataframe['bb_upper'])
            )
        
        # Apply conditions
        if conditions_exit_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_long),
                'exit_long'
            ] = 1
            
            # Send signal to CWTS Ultra
            self._send_exit_signal(dataframe, metadata, 'long')
        
        if conditions_exit_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_short),
                'exit_short'
            ] = 1
            
            # Send signal to CWTS Ultra
            self._send_exit_signal(dataframe, metadata, 'short')
        
        return dataframe
    
    def _send_entry_signal(self, dataframe: DataFrame, metadata: dict, direction: str) -> None:
        """Send entry signal to CWTS Ultra for ultra-low latency execution."""
        if not self.cwts_client or not hasattr(self, 'dp') or not self.dp:
            return
        
        if self.dp.runmode.value not in ['live', 'dry_run']:
            return
        
        try:
            pair = metadata.get('pair')
            last_row = dataframe.iloc[-1]
            
            # Calculate confidence based on multiple indicators
            confidence = self._calculate_signal_confidence(last_row, direction)
            
            if confidence >= self.cwts_confidence_threshold.value:
                # Send signal via shared memory (microsecond latency)
                action = 1 if direction == 'long' else 2  # 1=buy, 2=sell
                
                success = self.cwts_client.send_signal(
                    symbol=pair,
                    action=action,
                    confidence=confidence,
                    size=0.0,  # Use default size
                    price=0.0,  # Market order
                    stop_loss=last_row['close'] * (0.98 if direction == 'long' else 1.02),
                    take_profit=last_row['close'] * (1.02 if direction == 'long' else 0.98),
                    strategy_id=hash(self.__class__.__name__) & 0xFFFFFFFF
                )
                
                if success:
                    logger.info(f"Sent {direction} entry signal for {pair} with confidence {confidence:.2f}")
        
        except Exception as e:
            logger.error(f"Error sending entry signal: {e}")
    
    def _send_exit_signal(self, dataframe: DataFrame, metadata: dict, direction: str) -> None:
        """Send exit signal to CWTS Ultra for ultra-low latency execution."""
        if not self.cwts_client or not hasattr(self, 'dp') or not self.dp:
            return
        
        if self.dp.runmode.value not in ['live', 'dry_run']:
            return
        
        try:
            pair = metadata.get('pair')
            
            # For exit, we send opposite signal or hold
            action = 2 if direction == 'long' else 1  # Opposite of entry
            
            success = self.cwts_client.send_signal(
                symbol=pair,
                action=action,
                confidence=1.0,  # High confidence for exit
                size=0.0,  # Close full position
                price=0.0,  # Market order
                strategy_id=hash(self.__class__.__name__) & 0xFFFFFFFF
            )
            
            if success:
                logger.info(f"Sent {direction} exit signal for {pair}")
        
        except Exception as e:
            logger.error(f"Error sending exit signal: {e}")
    
    def _calculate_signal_confidence(self, row: pd.Series, direction: str) -> float:
        """
        Calculate signal confidence based on multiple indicators.
        Returns value between 0.0 and 1.0.
        """
        confidence = 0.0
        factors = 0
        
        # RSI confidence
        if direction == 'long':
            if row['rsi'] < 30:
                confidence += 1.0
                factors += 1
            elif row['rsi'] < 50:
                confidence += 0.5
                factors += 1
        else:  # short
            if row['rsi'] > 70:
                confidence += 1.0
                factors += 1
            elif row['rsi'] > 50:
                confidence += 0.5
                factors += 1
        
        # MACD confidence
        macd_signal = row['macd'] > row['macdsignal'] if direction == 'long' else row['macd'] < row['macdsignal']
        if macd_signal:
            confidence += 0.8
            factors += 1
        
        # Bollinger Bands confidence
        if direction == 'long' and row['close'] > row['bb_middle']:
            confidence += 0.6
            factors += 1
        elif direction == 'short' and row['close'] < row['bb_middle']:
            confidence += 0.6
            factors += 1
        
        # Order book imbalance (if available)
        if 'orderbook_imbalance' in row and not pd.isna(row['orderbook_imbalance']):
            if direction == 'long' and row['orderbook_imbalance'] > 0.1:
                confidence += 0.9
                factors += 1
            elif direction == 'short' and row['orderbook_imbalance'] < -0.1:
                confidence += 0.9
                factors += 1
        
        # Calculate average confidence
        return confidence / factors if factors > 0 else 0.0


class CWTSWebSocketClient:
    """WebSocket client fallback for CWTS Ultra communication."""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.loop = asyncio.new_event_loop()
    
    async def connect(self):
        """Connect to CWTS Ultra WebSocket server."""
        self.ws = await websockets.connect(self.url)
    
    async def send_signal(self, signal: dict):
        """Send signal via WebSocket."""
        if self.ws:
            await self.ws.send(msgpack.packb(signal))
    
    async def get_market_data(self, symbol: str) -> dict:
        """Get market data via WebSocket."""
        if self.ws:
            request = {"method": "get_market_data", "params": {"symbol": symbol}}
            await self.ws.send(msgpack.packb(request))
            response = await self.ws.recv()
            return msgpack.unpackb(response)
        return {}


# Import reduce for combining conditions
from functools import reduce