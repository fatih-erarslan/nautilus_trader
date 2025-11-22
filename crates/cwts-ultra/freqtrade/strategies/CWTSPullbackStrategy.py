"""
CWTS Pullback Strategy - Buy Dips in Uptrends
A profitable strategy that buys pullbacks instead of chasing momentum
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class CWTSPullbackStrategy(IStrategy):
    """
    Pullback strategy that buys dips in uptrends.
    
    Key principles:
    - Buy weakness in strength (pullbacks in uptrends)
    - Sell strength in weakness (rallies in downtrends)
    - Use wider stops for volatility
    - Realistic profit targets
    """
    
    # Strategy name
    STRATEGY_NAME = "CWTS_Pullback"
    
    # Realistic ROI table - let winners run
    minimal_roi = {
        "0": 0.10,    # 10% for exceptional moves
        "60": 0.04,   # 4% after 1 hour
        "120": 0.03,  # 3% after 2 hours
        "240": 0.02,  # 2% after 4 hours
        "480": 0.01,  # 1% after 8 hours
        "720": 0.005  # 0.5% after 12 hours (cover fees)
    }
    
    # Wider stop loss for volatility
    stoploss = -0.03  # 3% stop loss
    trailing_stop = True
    trailing_stop_positive = 0.005  # Start trailing after 0.5% profit
    trailing_stop_positive_offset = 0.01  # Trail by 1%
    trailing_only_offset_is_reached = True
    
    # 15-minute timeframe - less noise than 1m, more signals than 1h
    timeframe = '15m'
    
    # Process only new candles
    process_only_new_candles = True
    
    # Startup candles needed
    startup_candle_count = 200
    
    # Can short (set to False for spot)
    can_short = False
    
    # Optimizable parameters
    
    # Trend parameters
    ema_fast_period = IntParameter(8, 25, default=12, space="buy")
    ema_slow_period = IntParameter(20, 100, default=50, space="buy")
    
    # Pullback parameters
    rsi_buy_level = IntParameter(25, 45, default=35, space="buy")
    pullback_pct = DecimalParameter(0.005, 0.03, default=0.015, space="buy")
    
    # Volume confirmation
    volume_threshold = DecimalParameter(1.0, 2.5, default=1.5, space="buy")
    
    # Exit parameters
    rsi_sell_level = IntParameter(65, 85, default=75, space="sell")
    profit_threshold = DecimalParameter(0.01, 0.03, default=0.02, space="sell")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators for pullback detection.
        """
        
        # EMAs for trend
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)
        
        # Additional EMAs for dynamic support/resistance
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # RSI for oversold/overbought
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD for momentum shifts
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands for volatility
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_position'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ATR for volatility-based stops
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        
        # Price action
        dataframe['high_max'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_min'] = dataframe['low'].rolling(window=20).min()
        dataframe['price_range'] = (dataframe['close'] - dataframe['low_min']) / (dataframe['high_max'] - dataframe['low_min'])
        
        # Trend strength
        dataframe['trend_strength'] = abs(dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['close']
        dataframe['uptrend'] = (dataframe['ema_fast'] > dataframe['ema_slow']).astype(int)
        dataframe['strong_uptrend'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slow'] > dataframe['ema_200']) &
            (dataframe['trend_strength'] > 0.01)
        ).astype(int)
        
        # Pullback detection
        dataframe['pullback_from_high'] = (dataframe['high_max'] - dataframe['close']) / dataframe['high_max']
        dataframe['bounce_from_ema'] = abs(dataframe['close'] - dataframe['ema_20']) / dataframe['close']
        
        # Support levels
        dataframe['near_support'] = (
            (abs(dataframe['close'] - dataframe['ema_20']) < dataframe['atr'] * 0.5) |
            (abs(dataframe['close'] - dataframe['ema_slow']) < dataframe['atr'] * 0.5) |
            (abs(dataframe['close'] - dataframe['bb_lower']) < dataframe['atr'] * 0.5)
        ).astype(int)
        
        # Money Flow Index
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Stochastic for oversold/overbought
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy pullbacks in uptrends.
        """
        
        # Condition 1: Strong Uptrend Pullback
        uptrend_pullback = (
            # Uptrend confirmed
            (dataframe['uptrend'] == 1) &
            # Price pulled back to support
            (
                (dataframe['close'] <= dataframe['ema_fast']) |
                (dataframe['close'] <= dataframe['ema_20'])
            ) &
            # RSI oversold in uptrend
            (dataframe['rsi'] < self.rsi_buy_level.value) &
            # Volume confirmation
            (dataframe['volume_ratio'] > self.volume_threshold.value) &
            # Not at resistance
            (dataframe['price_range'] < 0.8)
        )
        
        # Condition 2: Bollinger Band Bounce
        bb_bounce = (
            # Uptrend
            (dataframe['uptrend'] == 1) &
            # Touch lower band
            (dataframe['close'] <= dataframe['bb_lower'] * 1.01) &
            # RSI not too oversold (avoiding knife catching)
            (dataframe['rsi'] > 25) &
            (dataframe['rsi'] < 45) &
            # Volume spike
            (dataframe['volume_ratio'] > 1.3)
        )
        
        # Condition 3: MACD Divergence Buy
        macd_divergence = (
            # Uptrend
            (dataframe['uptrend'] == 1) &
            # MACD turning up
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &
            (dataframe['macdhist'].shift(1) < dataframe['macdhist'].shift(2)) &
            # RSI oversold
            (dataframe['rsi'] < 40) &
            # Near support
            (dataframe['near_support'] == 1)
        )
        
        # Condition 4: Oversold Bounce
        oversold_bounce = (
            # Not in strong downtrend
            (dataframe['ema_fast'] > dataframe['ema_200']) &
            # Multiple oversold indicators
            (
                (dataframe['rsi'] < 30) |
                (dataframe['stoch_k'] < 20) |
                (dataframe['mfi'] < 25)
            ) &
            # Price at support
            (
                (dataframe['close'] <= dataframe['bb_lower']) |
                (dataframe['pullback_from_high'] > self.pullback_pct.value)
            ) &
            # Volume confirmation
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Combine conditions with OR logic
        conditions = uptrend_pullback | bb_bounce | macd_divergence | oversold_bounce
        
        dataframe.loc[conditions, 'enter_long'] = 1
        
        # Tag entries for analysis
        dataframe.loc[uptrend_pullback, 'enter_tag'] = 'uptrend_pullback'
        dataframe.loc[bb_bounce, 'enter_tag'] = 'bb_bounce'
        dataframe.loc[macd_divergence, 'enter_tag'] = 'macd_divergence'
        dataframe.loc[oversold_bounce, 'enter_tag'] = 'oversold_bounce'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit on strength or weakness.
        """
        
        # Exit Condition 1: Take Profit at Resistance
        resistance_exit = (
            # RSI overbought
            (dataframe['rsi'] > self.rsi_sell_level.value) |
            # At Bollinger upper band
            (dataframe['close'] >= dataframe['bb_upper'] * 0.99) |
            # At recent high
            (dataframe['price_range'] > 0.95)
        )
        
        # Exit Condition 2: Trend Reversal
        trend_exit = (
            # Trend changed
            (dataframe['uptrend'] == 0) &
            # MACD bearish
            (dataframe['macd'] < dataframe['macdsignal']) &
            # Price below fast EMA
            (dataframe['close'] < dataframe['ema_fast'])
        )
        
        # Exit Condition 3: Stop Loss Approaching
        stoploss_exit = (
            # Big red candle
            ((dataframe['close'] - dataframe['open']) / dataframe['open'] < -0.02) |
            # Breaking support
            (dataframe['close'] < dataframe['ema_slow'] * 0.97)
        )
        
        # Exit Condition 4: Time-based Profit
        time_exit = (
            # Small profit after time
            ((dataframe['close'] - dataframe['close'].shift(48)) / dataframe['close'].shift(48) > self.profit_threshold.value)
        )
        
        # Combine exit conditions
        conditions = resistance_exit | trend_exit | stoploss_exit | time_exit
        
        dataframe.loc[conditions, 'exit_long'] = 1
        
        # Tag exits
        dataframe.loc[resistance_exit, 'exit_tag'] = 'resistance'
        dataframe.loc[trend_exit, 'exit_tag'] = 'trend_reversal'
        dataframe.loc[stoploss_exit, 'exit_tag'] = 'stoploss_approaching'
        dataframe.loc[time_exit, 'exit_tag'] = 'time_profit'
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, side: str, **kwargs) -> float:
        """
        Risk 1% per trade with 3% stop loss.
        """
        
        # Get latest data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return proposed_stake
        
        last_row = dataframe.iloc[-1]
        
        # Base position size on volatility
        if 'atr_pct' in last_row:
            # Lower position size for high volatility
            if last_row['atr_pct'] > 0.03:  # High volatility
                stake = proposed_stake * 0.5
            elif last_row['atr_pct'] > 0.02:  # Medium volatility
                stake = proposed_stake * 0.75
            else:  # Low volatility
                stake = proposed_stake
        else:
            stake = proposed_stake
        
        # Adjust by entry type confidence
        confidence = {
            'uptrend_pullback': 1.2,
            'bb_bounce': 1.0,
            'macd_divergence': 0.9,
            'oversold_bounce': 0.8
        }
        
        stake = stake * confidence.get(entry_tag, 1.0)
        
        return max(min_stake, min(stake, max_stake))