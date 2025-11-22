"""
CWTS Ultra Momentum Strategy with FreqAI Support
High-frequency momentum trading strategy using CWTS Ultra engine and FreqAI predictions
"""

from typing import Optional, Union
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from CWTSMomentumStrategy import CWTSMomentumStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class CWTSMomentumFreqAI(CWTSMomentumStrategy):
    """
    CWTS Momentum Strategy with FreqAI integration.
    
    This strategy combines:
    - Ultra-low latency signals from CWTS Ultra
    - FreqAI predictions using CatBoost
    - Momentum-based trading signals
    """
    
    # Strategy name
    STRATEGY_NAME = "CWTS_Momentum_FreqAI"
    
    # Override parent's can_short for spot trading
    can_short = False
    
    # FreqAI parameters
    use_freqai = True
    
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: dict, **kwargs) -> DataFrame:
        """
        Create features for FreqAI model.
        This method is called for all timeframes and periods configured in FreqAI.
        """
        
        # Price-based features
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)
        
        # Momentum features
        dataframe[f"%-momentum-period_{period}"] = dataframe['close'].pct_change(periods=period)
        dataframe[f"%-momentum_abs-period_{period}"] = dataframe[f"%-momentum-period_{period}"].abs()
        
        # Volume features
        dataframe[f"%-volume_ratio-period_{period}"] = (
            dataframe['volume'] / dataframe['volume'].rolling(window=period).mean()
        )
        
        # Volatility features
        dataframe[f"%-bb_width-period_{period}"] = (
            ta.BBANDS(dataframe, timeperiod=period)['upperband'] -
            ta.BBANDS(dataframe, timeperiod=period)['lowerband']
        ) / ta.BBANDS(dataframe, timeperiod=period)['middleband']
        
        # MACD features
        macd = ta.MACD(dataframe, fastperiod=period, slowperiod=period*2)
        dataframe[f"%-macd-period_{period}"] = macd['macd']
        dataframe[f"%-macdsignal-period_{period}"] = macd['macdsignal']
        dataframe[f"%-macdhist-period_{period}"] = macd['macdhist']
        
        # Price position features
        dataframe[f"%-close_to_bb_upper-period_{period}"] = (
            dataframe['close'] / ta.BBANDS(dataframe, timeperiod=period)['upperband']
        )
        dataframe[f"%-close_to_bb_lower-period_{period}"] = (
            dataframe['close'] / ta.BBANDS(dataframe, timeperiod=period)['lowerband']
        )
        
        return dataframe
    
    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create basic features that are not period-dependent.
        """
        
        # Price change features
        dataframe["%-pct_change"] = dataframe['close'].pct_change()
        dataframe["%-raw_volume"] = dataframe['volume']
        dataframe["%-raw_price"] = dataframe['close']
        
        # High/Low ratios
        dataframe["%-high_low_ratio"] = dataframe['high'] / dataframe['low']
        dataframe["%-close_open_ratio"] = dataframe['close'] / dataframe['open']
        
        # Spread
        dataframe["%-spread"] = (dataframe['high'] - dataframe['low']) / dataframe['close']
        
        return dataframe
    
    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create standard features used across all pairs.
        """
        
        # Time-based features
        dataframe["%-hour"] = dataframe.index.hour
        dataframe["%-day_of_week"] = dataframe.index.dayofweek
        
        # Rolling statistics
        for window in [5, 10, 20]:
            dataframe[f"%-rolling_std_{window}"] = dataframe['close'].rolling(window=window).std()
            dataframe[f"%-rolling_mean_{window}"] = dataframe['close'].rolling(window=window).mean()
            dataframe[f"%-rolling_max_{window}"] = dataframe['high'].rolling(window=window).max()
            dataframe[f"%-rolling_min_{window}"] = dataframe['low'].rolling(window=window).min()
        
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Set the targets for FreqAI model training.
        In this case, we're predicting future price movement.
        """
        
        # Classification target: Will price go up by X% in next N candles?
        n_candles = 10  # Look ahead period
        threshold = 0.005  # 0.5% movement threshold
        
        dataframe['&s-price_change'] = (
            dataframe['close'].shift(-n_candles) - dataframe['close']
        ) / dataframe['close']
        
        # Multi-class target
        dataframe['&s-trend'] = 0  # Neutral
        dataframe.loc[dataframe['&s-price_change'] > threshold, '&s-trend'] = 1  # Up
        dataframe.loc[dataframe['&s-price_change'] < -threshold, '&s-trend'] = -1  # Down
        
        # Regression target (actual price change)
        dataframe['&s-price_change_reg'] = dataframe['&s-price_change']
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators including FreqAI predictions.
        """
        
        # Call parent to get base indicators
        dataframe = super().populate_indicators(dataframe, metadata)
        
        # FreqAI adds predictions automatically with columns like:
        # 'prediction', 'do_predict', 'DI_values', etc.
        
        # Add any additional indicators needed for entry/exit logic
        # that use FreqAI predictions
        if 'do_predict' in dataframe.columns:
            # Smooth predictions
            dataframe['prediction_smooth'] = dataframe['do_predict'].rolling(window=3).mean()
            
            # Confidence-based features
            if 'DI_values' in dataframe.columns:
                dataframe['high_confidence'] = dataframe['DI_values'] < 0.5
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy/entry signals using both momentum and FreqAI predictions.
        """
        
        conditions_long = []
        
        # Base momentum conditions from parent
        conditions_long.append(
            (dataframe['momentum'] > self.momentum_threshold.value) &
            (dataframe['momentum'] > dataframe['momentum_sma']) &
            (dataframe['acceleration'] > 0)
        )
        
        # Volume confirmation
        conditions_long.append(
            (dataframe['volume_ratio'] > self.volume_factor.value) |
            (dataframe['volume_momentum'] > 0.5)
        )
        
        # Technical confirmation
        conditions_long.append(
            (dataframe['close'] > dataframe['vwap']) &
            (dataframe['mfi'] < 80) &
            (dataframe['rsi'] > 40) &
            (dataframe['rsi'] < 70)
        )
        
        # FreqAI confirmation (if available)
        if 'do_predict' in dataframe.columns:
            # Only enter when FreqAI predicts upward movement
            conditions_long.append(
                (dataframe['do_predict'] > 0.5)  # Bullish prediction
            )
            
            # Optional: Use DI values for confidence filtering
            if 'DI_values' in dataframe.columns:
                conditions_long.append(
                    (dataframe['DI_values'] < 0.5)  # High confidence
                )
        
        # Apply long conditions
        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                ['enter_long', 'enter_tag']
            ] = (1, 'momentum_freqai_long')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell/exit signals using both momentum and FreqAI predictions.
        """
        
        conditions_exit_long = []
        
        # Momentum reversal
        conditions_exit_long.append(
            (dataframe['momentum'] < 0) |
            (dataframe['acceleration'] < -self.momentum_threshold.value/2)
        )
        
        # Technical exit signals
        conditions_exit_long.append(
            (dataframe['rsi'] > 75) |
            (dataframe['mfi'] > 85) |
            (dataframe['close'] < dataframe['ema_fast'])
        )
        
        # FreqAI exit signal (if available)
        if 'do_predict' in dataframe.columns:
            # Exit when FreqAI predicts downward movement
            conditions_exit_long.append(
                (dataframe['do_predict'] < -0.3)  # Bearish prediction
            )
        
        # Apply exit conditions
        if conditions_exit_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_exit_long),
                ['exit_long', 'exit_tag']
            ] = (1, 'momentum_freqai_exit')
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Additional trade entry confirmation using FreqAI predictions.
        """
        
        # Get the latest prediction
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return False
        
        last_row = dataframe.iloc[-1]
        
        # Check FreqAI prediction if available
        if 'do_predict' in last_row:
            if side == "long" and last_row['do_predict'] < 0.3:
                logger.info(f"Rejecting long entry for {pair} due to weak FreqAI prediction")
                return False
            elif side == "short" and last_row['do_predict'] > -0.3:
                logger.info(f"Rejecting short entry for {pair} due to weak FreqAI prediction")
                return False
        
        # Check DI values for confidence
        if 'DI_values' in last_row and last_row['DI_values'] > 0.7:
            logger.info(f"Rejecting entry for {pair} due to low FreqAI confidence (high DI)")
            return False
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time, **kwargs) -> bool:
        """
        Additional trade exit confirmation using FreqAI predictions.
        """
        
        # Get the latest prediction
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) == 0:
            return True  # Allow exit if no data
        
        last_row = dataframe.iloc[-1]
        
        # Check FreqAI prediction
        if 'do_predict' in last_row:
            # If FreqAI strongly disagrees with exit, reconsider
            if trade.is_short:
                if last_row['do_predict'] < -0.7:  # Strong bearish, don't exit short
                    logger.info(f"Delaying short exit for {pair} due to strong bearish FreqAI prediction")
                    return False
            else:  # Long trade
                if last_row['do_predict'] > 0.7:  # Strong bullish, don't exit long
                    logger.info(f"Delaying long exit for {pair} due to strong bullish FreqAI prediction")
                    return False
        
        return True