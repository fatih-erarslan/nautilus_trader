"""
Quantum Momentum Strategy - Recreated based on 95% win rate performance
Based on analysis of the highly successful QuantumMomentumStrategy trades
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class QuantumMomentumRecreated(IStrategy):
    """
    Recreation of the 95% win rate QuantumMomentumStrategy.
    
    Based on trade analysis showing:
    - 20 trades from July 12-13, 2025
    - +$629.48 profit 
    - 95% win rate
    - Average +4.96% per trade
    
    This strategy likely uses:
    - Longer timeframes (15m-1h) to avoid noise
    - Strong momentum confirmation
    - Wider stops to avoid premature exits
    - Lets winners run with trailing stops
    """
    
    # Strategy name
    STRATEGY_NAME = "Quantum_Momentum_Recreated"
    
    # ROI table - let winners run
    minimal_roi = {
        "0": 0.20,    # 20% for huge moves only
        "30": 0.10,   # 10% after 30 min
        "60": 0.05,   # 5% after 1 hour  
        "120": 0.03,  # 3% after 2 hours
        "240": 0.02,  # 2% after 4 hours
        "480": 0.01,  # 1% after 8 hours
    }
    
    # Wider stop loss for 95% win rate
    stoploss = -0.05  # 5% stop loss (wide enough to avoid noise)
    
    # Trailing stop to lock profits
    trailing_stop = True
    trailing_stop_positive = 0.01  # Start trailing after 1% profit
    trailing_stop_positive_offset = 0.02  # Trail by 2%
    trailing_only_offset_is_reached = True
    
    # Use 15m or 30m timeframe (less noise)
    timeframe = '30m'
    
    # Can short
    can_short = False
    
    # Process only new candles
    process_only_new_candles = True
    startup_candle_count = 200
    
    # Optimizable parameters
    
    # Momentum
    momentum_period = IntParameter(10, 30, default=20, space="buy")
    momentum_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy")
    
    # Volume
    volume_threshold = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    
    # Trend
    ema_fast = IntParameter(8, 20, default=12, space="buy")
    ema_slow = IntParameter(20, 50, default=26, space="buy")
    ema_trend = IntParameter(50, 200, default=100, space="buy")
    
    # RSI
    rsi_buy = IntParameter(30, 50, default=40, space="buy")
    rsi_sell = IntParameter(70, 90, default=80, space="sell")
    
    # MACD
    macd_buy_signal = DecimalParameter(-0.002, 0.002, default=0, space="buy")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add quantum momentum indicators.
        """
        
        # Momentum calculation
        dataframe['momentum'] = dataframe['close'].pct_change(periods=self.momentum_period.value)
        dataframe['momentum_abs'] = dataframe['momentum'].abs()
        dataframe['momentum_sma'] = dataframe['momentum'].rolling(window=10).mean()
        dataframe['momentum_std'] = dataframe['momentum'].rolling(window=20).std()
        
        # Momentum quality (consistency)
        dataframe['momentum_quality'] = dataframe['momentum'] / (dataframe['momentum_std'] + 0.0001)
        
        # Price action
        dataframe['high_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20).min()
        dataframe['price_position'] = (dataframe['close'] - dataframe['low_20']) / (dataframe['high_20'] - dataframe['low_20'] + 0.0001)
        
        # EMAs for trend
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.ema_trend.value)
        
        # Trend strength
        dataframe['trend_strength'] = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['close']
        dataframe['uptrend'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slow'] > dataframe['ema_trend'])
        ).astype(int)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = dataframe['rsi'].rolling(window=10).mean()
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        
        # Money Flow Index
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Quantum indicator (combines multiple factors)
        dataframe['quantum_score'] = (
            (dataframe['uptrend'] * 0.25) +
            ((dataframe['momentum_quality'] > 1) * 0.25) +
            ((dataframe['volume_ratio'] > 1.5) * 0.25) +
            ((dataframe['rsi'] > 40) * 0.25)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Quantum momentum entry signals.
        """
        
        # Strong momentum entry
        strong_momentum = (
            # Strong uptrend
            (dataframe['uptrend'] == 1) &
            
            # Momentum is positive and strong
            (dataframe['momentum'] > self.momentum_threshold.value) &
            (dataframe['momentum_quality'] > 1) &
            
            # Not overbought
            (dataframe['rsi'] > self.rsi_buy.value) &
            (dataframe['rsi'] < 70) &
            
            # Volume confirmation
            (dataframe['volume_ratio'] > self.volume_threshold.value) &
            
            # MACD confirmation
            (dataframe['macdhist'] > self.macd_buy_signal.value) &
            
            # Price not at resistance
            (dataframe['price_position'] < 0.9) &
            
            # Quantum score is high
            (dataframe['quantum_score'] >= 0.75)
        )
        
        # Breakout entry
        breakout = (
            # Uptrend
            (dataframe['uptrend'] == 1) &
            
            # Breaking above BB upper
            (dataframe['close'] > dataframe['bb_upper']) &
            
            # Strong volume
            (dataframe['volume_ratio'] > 2.5) &
            
            # RSI not extreme
            (dataframe['rsi'] < 75) &
            
            # Momentum positive
            (dataframe['momentum'] > 0.01)
        )
        
        # Pullback bounce entry
        bounce = (
            # In uptrend
            (dataframe['uptrend'] == 1) &
            
            # Pulled back to support
            (dataframe['close'] <= dataframe['ema_fast'] * 1.01) &
            
            # RSI oversold in uptrend
            (dataframe['rsi'] < 45) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            
            # MACD turning up
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &
            
            # Volume present
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Apply entry conditions
        dataframe.loc[
            strong_momentum | breakout | bounce,
            'enter_long'
        ] = 1
        
        # Tag entries
        dataframe.loc[strong_momentum, 'enter_tag'] = 'quantum_momentum'
        dataframe.loc[breakout, 'enter_tag'] = 'quantum_breakout'
        dataframe.loc[bounce, 'enter_tag'] = 'quantum_bounce'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Quantum momentum exit signals.
        """
        
        # Take profit exit
        profit_exit = (
            # RSI overbought
            (dataframe['rsi'] > self.rsi_sell.value) |
            
            # Price at resistance
            (dataframe['price_position'] > 0.95) |
            
            # Momentum reversing
            (
                (dataframe['momentum'] < 0) &
                (dataframe['momentum'] < dataframe['momentum'].shift(1))
            )
        )
        
        # Trend reversal exit
        trend_exit = (
            # Lost uptrend
            (dataframe['uptrend'] == 0) &
            
            # MACD bearish
            (dataframe['macdhist'] < 0) &
            
            # Price below EMA
            (dataframe['close'] < dataframe['ema_slow'])
        )
        
        # Quantum score dropped
        quantum_exit = (
            dataframe['quantum_score'] < 0.25
        )
        
        # Apply exit conditions
        dataframe.loc[
            profit_exit | trend_exit | quantum_exit,
            'exit_long'
        ] = 1
        
        # Tag exits
        dataframe.loc[profit_exit, 'exit_tag'] = 'profit_target'
        dataframe.loc[trend_exit, 'exit_tag'] = 'trend_reversal'
        dataframe.loc[quantum_exit, 'exit_tag'] = 'quantum_exit'
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, side: str, **kwargs) -> float:
        """
        Adjust position size based on quantum score.
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return proposed_stake
        
        last_row = dataframe.iloc[-1]
        
        # Base position on quantum score
        quantum_multiplier = last_row.get('quantum_score', 0.5) * 2  # 0 to 2x
        
        # Reduce size in high volatility
        if 'atr_pct' in last_row:
            if last_row['atr_pct'] > 0.03:
                quantum_multiplier *= 0.7
        
        # Increase size for strong signals
        if entry_tag == 'quantum_momentum':
            quantum_multiplier *= 1.2
        
        stake = proposed_stake * quantum_multiplier
        
        return max(min_stake, min(stake, max_stake))