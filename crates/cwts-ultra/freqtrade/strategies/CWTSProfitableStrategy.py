"""
CWTS Profitable Strategy - Simplified for Actual Trading
A pragmatic approach that prioritizes profitability over complexity
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class CWTSProfitableStrategy(IStrategy):
    """
    Simplified profitable strategy that actually enters trades.
    
    Key changes from parasitic strategy:
    - Relaxed entry conditions (OR logic instead of AND)
    - Lower quality thresholds
    - Focus on momentum and volume
    - Pragmatic risk management
    """
    
    # Strategy name
    STRATEGY_NAME = "CWTS_Profitable"
    
    # Reasonable ROI table
    minimal_roi = {
        "0": 0.025,   # 2.5% immediate target
        "10": 0.02,   # 2% after 10 minutes
        "30": 0.015,  # 1.5% after 30 minutes
        "60": 0.01,   # 1% after 60 minutes
        "120": 0.005, # 0.5% after 2 hours
    }
    
    # Reasonable stop loss
    stoploss = -0.02  # 2% stop loss (more room to breathe)
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'  # 5-minute for better signals (not too fast, not too slow)
    
    # Can short (set to False for spot trading)
    can_short = False
    
    # Optimizable parameters with REASONABLE defaults
    # Momentum
    momentum_threshold = DecimalParameter(0.002, 0.01, default=0.004, space="buy")
    momentum_period = IntParameter(10, 30, default=20, space="buy")
    
    # Volume
    volume_threshold = DecimalParameter(1.2, 2.0, default=1.5, space="buy")
    
    # RSI
    rsi_buy = IntParameter(25, 45, default=35, space="buy")
    rsi_sell = IntParameter(65, 85, default=75, space="sell")
    
    # Quality threshold (MUCH LOWER)
    quality_threshold = DecimalParameter(0.5, 0.8, default=0.6, space="buy")
    
    # Risk score threshold (REASONABLE)
    risk_threshold = DecimalParameter(0.2, 0.5, default=0.35, space="buy")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add technical indicators - keep it simple and effective.
        """
        
        # Momentum
        dataframe['momentum'] = dataframe['close'].pct_change(periods=self.momentum_period.value)
        dataframe['momentum_sma'] = dataframe['momentum'].rolling(window=10).mean()
        
        # Volume
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # EMA
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Simple trend detection
        dataframe['trend_up'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['close'] > dataframe['ema_fast'])
        ).astype(int)
        
        # Money Flow Index
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Quality score (simplified)
        dataframe['quality_score'] = self.calculate_quality_score(dataframe)
        
        # Risk score (simplified)
        dataframe['risk_score'] = self.calculate_risk_score(dataframe)
        
        return dataframe
    
    def calculate_quality_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Simple quality score based on multiple factors.
        Returns value between 0 and 1.
        """
        score = pd.Series(index=dataframe.index, data=0.5)
        
        # Trend alignment (25%)
        score += 0.25 * dataframe['trend_up']
        
        # Volume confirmation (25%)
        score += 0.25 * (dataframe['volume_ratio'] > 1.0).astype(float)
        
        # RSI not extreme (25%)
        score += 0.25 * ((dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)).astype(float)
        
        # MACD positive (25%)
        score += 0.25 * (dataframe['macdhist'] > 0).astype(float)
        
        return score
    
    def calculate_risk_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Simple risk score based on volatility and extremes.
        Returns value between 0 and 1 (lower is better).
        """
        score = pd.Series(index=dataframe.index, data=0.0)
        
        # High volatility increases risk
        atr_pct = dataframe['atr'] / dataframe['close']
        score += 0.33 * (atr_pct / atr_pct.rolling(100).max()).fillna(0.5)
        
        # RSI extremes increase risk
        score += 0.33 * (
            ((dataframe['rsi'] < 20) | (dataframe['rsi'] > 80)).astype(float)
        )
        
        # Wide Bollinger Bands increase risk
        bb_width_norm = dataframe['bb_width'] / dataframe['bb_width'].rolling(100).mean()
        score += 0.34 * ((bb_width_norm > 1.5).astype(float))
        
        return score.clip(0, 1)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy signals with RELAXED conditions using OR logic.
        """
        
        # Buy Condition Set 1: Momentum Play
        momentum_buy = (
            (dataframe['momentum'] > self.momentum_threshold.value) &
            (dataframe['momentum'] > dataframe['momentum_sma']) &
            (dataframe['volume_ratio'] > self.volume_threshold.value)
        )
        
        # Buy Condition Set 2: Oversold Bounce
        oversold_buy = (
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['close'] < dataframe['bb_lower']) &
            (dataframe['mfi'] < 30) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Buy Condition Set 3: Trend Continuation
        trend_buy = (
            (dataframe['trend_up'] == 1) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['close'] > dataframe['bb_middle']) &
            (dataframe['rsi'] < 65)
        )
        
        # Buy Condition Set 4: MACD Cross
        macd_buy = (
            (ta.crossover(dataframe['macd'], dataframe['macdsignal'])) &
            (dataframe['volume_ratio'] > 1.2) &
            (dataframe['rsi'] > 30) &
            (dataframe['rsi'] < 70)
        )
        
        # Quality and Risk Filters (RELAXED)
        quality_filter = dataframe['quality_score'] >= self.quality_threshold.value
        risk_filter = dataframe['risk_score'] <= self.risk_threshold.value
        
        # Combine with OR logic (any condition can trigger)
        buy_condition = (
            (momentum_buy | oversold_buy | trend_buy | macd_buy) &
            quality_filter &
            risk_filter
        )
        
        # Set buy signal
        dataframe.loc[buy_condition, 'enter_long'] = 1
        
        # Add tags for different buy reasons
        dataframe.loc[momentum_buy & quality_filter & risk_filter, 'enter_tag'] = 'momentum'
        dataframe.loc[oversold_buy & quality_filter & risk_filter, 'enter_tag'] = 'oversold'
        dataframe.loc[trend_buy & quality_filter & risk_filter, 'enter_tag'] = 'trend'
        dataframe.loc[macd_buy & quality_filter & risk_filter, 'enter_tag'] = 'macd_cross'
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell signals - be quick to take profits or cut losses.
        """
        
        # Sell Condition 1: Take Profit on Overbought
        overbought_sell = (
            (dataframe['rsi'] > self.rsi_sell.value) |
            (dataframe['mfi'] > 80) |
            (dataframe['close'] > dataframe['bb_upper'])
        )
        
        # Sell Condition 2: Momentum Reversal
        momentum_sell = (
            (dataframe['momentum'] < 0) &
            (dataframe['momentum'] < dataframe['momentum_sma'])
        )
        
        # Sell Condition 3: Trend Reversal
        trend_sell = (
            (dataframe['trend_up'] == 0) &
            (dataframe['macd'] < dataframe['macdsignal'])
        )
        
        # Sell Condition 4: Stop Loss Approaching
        risk_sell = (
            dataframe['risk_score'] > 0.7
        )
        
        # Combine with OR logic (any condition triggers sell)
        sell_condition = (
            overbought_sell | 
            momentum_sell | 
            trend_sell | 
            risk_sell
        )
        
        # Set sell signal
        dataframe.loc[sell_condition, 'exit_long'] = 1
        
        # Add tags for different sell reasons
        dataframe.loc[overbought_sell, 'exit_tag'] = 'overbought'
        dataframe.loc[momentum_sell, 'exit_tag'] = 'momentum_reversal'
        dataframe.loc[trend_sell, 'exit_tag'] = 'trend_reversal'
        dataframe.loc[risk_sell, 'exit_tag'] = 'risk_limit'
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, side: str, **kwargs) -> float:
        """
        Adjust position size based on confidence.
        """
        
        # Get latest data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return proposed_stake
        
        last_row = dataframe.iloc[-1]
        
        # Start with proposed stake
        stake = proposed_stake
        
        # Adjust based on quality score
        quality_multiplier = 0.5 + last_row['quality_score']  # 0.5x to 1.5x
        
        # Adjust based on risk score (inverse)
        risk_multiplier = 1.5 - last_row['risk_score']  # 0.5x to 1.5x
        
        # Adjust based on entry type
        tag_multipliers = {
            'momentum': 1.2,
            'oversold': 0.8,
            'trend': 1.0,
            'macd_cross': 0.9
        }
        tag_multiplier = tag_multipliers.get(entry_tag, 1.0)
        
        # Calculate final stake
        stake = stake * quality_multiplier * risk_multiplier * tag_multiplier
        
        # Ensure within limits
        return max(min_stake, min(stake, max_stake))