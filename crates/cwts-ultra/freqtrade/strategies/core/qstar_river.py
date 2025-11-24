#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:29:34 2025

@author: ashina
"""

"""
Q* Learning Integration with RiverML for Crypto Trading

This module integrates the Q* Learning algorithm with RiverML's online
learning capabilities to create a sophisticated crypto trading prediction system.
The integration leverages River's drift detection and anomaly detection with
Q*'s reinforcement learning framework to adapt to changing market conditions.
"""
import os
import pickle
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from collections import deque
import threading
# Import Q* Learning
from q_star_learning import (
    SophisticatedQLearningAgent, 
    EnvironmentWrapper,
    ExperienceBuffer,
    QLearningMetrics
)

# Import RiverML
from river_ml import RiverOnlineML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Q*RiverTrading")


class MarketState:
    """
    Market state representation combining technical indicators and RiverML features.
    """
    
    def __init__(self, num_states: int = 100, feature_keys=None, history_size=100, feature_window: int = 50):
        """Initialize market state with default features to prevent warnings."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_keys = feature_keys or []
        self.feature_history = []
        self.history_size = history_size
        self.logger = logging.getLogger("MarketState")
        self.num_states = num_states
        self.feature_window = feature_window
        self.feature_history = deque(maxlen=feature_window)
        
        # Initialize with defaults for all common features
        self.normalized_features = {
            'qerc_trend': 0.5,
            'volatility_regime': 0.5,
            'qerc_momentum': 0.5,
            'momentum': 0.5,
            'rsi_14': 0.5,
            'adx': 0.5,
            'trend': 0.5,
            'volume': 0.5,
            'close': 0.5,
            'high': 0.5,
            'low': 0.5,
            'open': 0.5
        }
        
        self.feature_mins = {}
        self.feature_maxs = {}
        
        # Start initialized with defaults
        self._initialized = True
        
    def update(self, features: Dict[str, float]) -> None:
        """
        Update state with new market features - enhanced for robustness.
        
        Args:
            features: Dictionary of feature values
        """
        # Handle empty features case
        if not features:
            self.logger.warning("Empty features dictionary provided to MarketState")
            return
            
        # Store original features for reference
        self.feature_history.append(features.copy())
        
        # Filter out non-numeric values
        valid_features = {}
        for key, value in features.items():
            try:
                # Convert to float and check for NaN/inf
                float_val = float(value)
                if not np.isnan(float_val) and not np.isinf(float_val):
                    valid_features[key] = float_val
            except (ValueError, TypeError):
                continue
        
        # Check if we have any valid features after filtering
        if not valid_features:
            self.logger.warning("No valid numeric features available after filtering")
            return
            
        # Update min/max values for normalization
        for key, value in valid_features.items():
            if key not in self.feature_mins or value < self.feature_mins[key]:
                self.feature_mins[key] = value
            if key not in self.feature_maxs or value > self.feature_maxs[key]:
                self.feature_maxs[key] = value
                
        # Update normalized features (preserve existing ones)
        for key, value in valid_features.items():
            if key in self.feature_mins and key in self.feature_maxs:
                min_val = self.feature_mins[key]
                max_val = self.feature_maxs[key]
                if max_val > min_val:
                    self.normalized_features[key] = (value - min_val) / (max_val - min_val)
                else:
                    self.normalized_features[key] = 0.5
            else:
                self.normalized_features[key] = 0.5
                
        # Set initialization flag once we have normalized features
        if not self._initialized and self.normalized_features:
            self._initialized = True
                    
    def get_state_index(self) -> int:
        """
        Map current normalized features to a discrete state index with improved error handling.
        
        Returns:
            State index (0 to num_states-1)
        """
        if not self._initialized or not self.normalized_features:
            self.logger.warning("No normalized features available, returning default state 0")
            return 0
            
        try:
            # Use key features for state determination (with defaults if not available)
            state_features = {
                'trend': self.normalized_features.get('qerc_trend', 0.5),
                'volatility': self.normalized_features.get('volatility_regime', 0.5),
                'momentum': self.normalized_features.get('qerc_momentum', 0.5),
                'rsi': self.normalized_features.get('rsi_14', 0.5),
                'adx': self.normalized_features.get('adx', 0.5)
            }
            
            # Create a hash from discretized feature values
            hash_value = 0
            bins = 5  # Number of bins per feature
            for i, (key, value) in enumerate(state_features.items()):
                # Ensure value is valid
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    binned_value = min(bins - 1, max(0, int(value * bins)))
                else:
                    binned_value = 0
                hash_value += binned_value * (bins ** i)
            
            # Map hash to state index
            state_index = hash_value % self.num_states
            return state_index
        except Exception as e:
            self.logger.error(f"Error in get_state_index: {e}")
            return 0  # Return default state on error

class TradingAction:
    """Trading action definitions and utilities."""
    
    # Action definitions
    BUY = 0
    SELL = 1
    HOLD = 2
    REDUCE = 3  # Reduce position
    INCREASE = 4  # Increase position
    
    @staticmethod
    def get_num_actions() -> int:
        """Get number of possible actions."""
        return 5
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """
        Get human-readable action name.
        
        Args:
            action: Action index
            
        Returns:
            Action name
        """
        actions = {
            TradingAction.BUY: "BUY",
            TradingAction.SELL: "SELL",
            TradingAction.HOLD: "HOLD",
            TradingAction.REDUCE: "REDUCE",
            TradingAction.INCREASE: "INCREASE"
        }
        return actions.get(action, "UNKNOWN")


class TradingEnvironment(EnvironmentWrapper):
    """
    Crypto trading environment for Q* Learning agent.
    
    This environment integrates market data and RiverML predictions
    to provide state representation, action space, and rewards for
    the reinforcement learning agent.
    """
    
    def __init__(self, river_ml=None, price_data=None, window_size=50, 
                 initial_balance=10000.0, transaction_fee=0.001, 
                 reward_scaling=0.01, use_position_limits=True, 
                 max_position_size=1.0):
        """Initialize trading environment with proper attribute initialization"""
        # Initialize logger first
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Explicitly initialize attributes to prevent cleanup errors
        self.env = None
        self.gym_env = None
        self._resources = []
        
        # Set consistent number of states
        self.num_states = 200  # Standardize on 200 states
        self.num_actions = 5   # Standardize on 5 actions
        
        # Regular initialization
        self.river_ml = river_ml
        self.price_data = price_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_position_limits = use_position_limits
        self.max_position_size = max_position_size
        
        # Market state
        self.market_state = MarketState(num_states=self.num_states, feature_window=window_size)
        
        # Trading state
        self.balance = initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = initial_balance
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        
        # Market data
        self.current_price = 0.0
        self.current_idx = 0
        self.prices = []
        
        if price_data is not None:
            self.prices = price_data['close'].values
        
        # Drift and anomaly detection
        self.drift_detected = False
        self.anomaly_detected = False
        
        self.logger.info(f"Trading environment initialized with {self.num_states} states and {self.num_actions} actions")
    
    def __del__(self):
        """Only log destruction, don't perform cleanup to prevent double-free"""
        try:
            self.logger.debug("TradingEnvironment being garbage collected")
        except:
            pass  # Silent during garbage collection
        
    def close(self):
        """Close all resources safely"""
        try:
            # Close env if it exists
            if hasattr(self, 'env') and self.env is not None:
                if hasattr(self.env, 'close'):
                    self.env.close()
                self.env = None  # Clear reference
            
            # Close gym_env if it exists
            if hasattr(self, 'gym_env') and self.gym_env is not None:
                self.gym_env.close()
                self.gym_env = None  # Clear reference
                
            # Close any other resources
            if hasattr(self, '_resources'):
                for resource in self._resources:
                    if hasattr(resource, 'close'):
                        resource.close()
                self._resources = []
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during resource cleanup: {e}")
            
        
    def reset(self) -> int:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state index
        """
        # Reset trading state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.initial_balance
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        
        # Reset market position
        self.current_idx = self.window_size
        if self.price_data is not None and len(self.prices) > 0:
            self.current_price = self.prices[self.current_idx]
        else:
            self.current_price = 100.0  # Default starting price
            
        # Reset history
        self.returns = []
        self.positions = []
        self.portfolio_values = []
        self.actions_taken = []
        
        # Reset metrics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        # Reset drift detection
        self.drift_detected = False
        self.anomaly_detected = False
        
        # Update market state
        self._update_market_state()
        
        # Return initial state
        return self.market_state.get_state_index()
        
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        try:
            # Track the action
            self.actions_taken.append(action)
            
            # Execute action
            reward = self._execute_action(action)
            
            # Move to next time step
            done = self._next_time_step()
            
            # If we're done (reached end of data), return immediately
            if done:
                # Return the last valid state for consistency
                return self.market_state.get_state_index(), reward, done
            
            # Update market state with new data
            self._update_market_state()
            
            # Get new state
            next_state = self.market_state.get_state_index()
            
            # Track drift and anomalies with RiverML
            self._detect_drift_and_anomalies()
            
            # Return step results
            return next_state, reward, done
        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in step: {str(e)}", exc_info=True)
            else:
                print(f"Error in step: {str(e)}")
            
            # Return current state, no reward, and done=True to terminate episode
            return self.market_state.get_state_index(), 0.0, True
        
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action and calculate reward.
        
        Args:
            action: Action to take
            
        Returns:
            Reward
        """
        previous_portfolio_value = self.portfolio_value
        reward = 0.0
        
        # Execute based on action type
        if action == TradingAction.BUY and self.balance > 0:
            # Full buy - use all available balance
            buy_amount = self.balance
            if self.use_position_limits:
                # Limit position size based on settings
                max_buy = self.portfolio_value * self.max_position_size - self.position_value
                buy_amount = min(buy_amount, max(0, max_buy))
                
            if buy_amount > 0:
                # Calculate position after fees
                fee = buy_amount * self.transaction_fee
                buy_amount_after_fee = buy_amount - fee
                new_position = buy_amount_after_fee / self.current_price
                
                # Update state
                self.balance -= buy_amount
                self.position += new_position
                self.position_value = self.position * self.current_price
                self.last_buy_price = self.current_price
                self.total_trades += 1
                logger.debug(f"BUY: {new_position:.4f} units at {self.current_price:.2f}")
                
        elif action == TradingAction.SELL and self.position > 0:
            # Full sell - liquidate entire position
            sell_amount = self.position
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L
            position_cost = self.position_value
            pnl = sell_value_after_fee - position_cost
            
            # Update state
            self.balance += sell_value_after_fee
            self.position = 0.0
            self.position_value = 0.0
            self.last_sell_price = self.current_price
            self.total_trades += 1
            self.total_pnl += pnl
            
            if pnl > 0:
                self.profitable_trades += 1
                
            logger.debug(f"SELL: {sell_amount:.4f} units at {self.current_price:.2f}, PnL: {pnl:.2f}")
            
        elif action == TradingAction.REDUCE and self.position > 0:
            # Reduce position by 50%
            sell_amount = self.position * 0.5
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L for the sold portion
            position_cost = self.position_value * 0.5
            pnl = sell_value_after_fee - position_cost
            
            # Update state
            self.balance += sell_value_after_fee
            self.position -= sell_amount
            self.position_value = self.position * self.current_price
            self.last_sell_price = self.current_price
            self.total_trades += 1
            self.total_pnl += pnl
            
            if pnl > 0:
                self.profitable_trades += 1
                
            logger.debug(f"REDUCE: {sell_amount:.4f} units at {self.current_price:.2f}, PnL: {pnl:.2f}")
            
        elif action == TradingAction.INCREASE and self.balance > 0:
            # Increase position by 50% of available balance
            buy_amount = self.balance * 0.5
            if self.use_position_limits:
                # Limit position size based on settings
                max_buy = self.portfolio_value * self.max_position_size - self.position_value
                buy_amount = min(buy_amount, max(0, max_buy))
                
            if buy_amount > 0:
                # Calculate position after fees
                fee = buy_amount * self.transaction_fee
                buy_amount_after_fee = buy_amount - fee
                new_position = buy_amount_after_fee / self.current_price
                
                # Update state
                self.balance -= buy_amount
                self.position += new_position
                self.position_value = self.position * self.current_price
                self.last_buy_price = self.current_price
                self.total_trades += 1
                logger.debug(f"INCREASE: {new_position:.4f} units at {self.current_price:.2f}")
        
        # For HOLD action, do nothing
        
        # Calculate current portfolio value
        self.position_value = self.position * self.current_price
        self.portfolio_value = self.balance + self.position_value
        
        # Track positions and portfolio value
        self.positions.append(self.position)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward based on portfolio change
        portfolio_return = (self.portfolio_value / previous_portfolio_value) - 1.0
        self.returns.append(portfolio_return)
        
        # Calculate reward - scaled return with penalty for excessive trading
        reward = portfolio_return * self.reward_scaling
        
        # Add drift and anomaly detection bonuses/penalties
        if self.drift_detected:
            if action == TradingAction.HOLD:
                # Bonus for holding during drift (being cautious)
                reward += 0.001
        
        if self.anomaly_detected:
            if action == TradingAction.SELL or action == TradingAction.REDUCE:
                # Bonus for reducing risk during anomaly
                reward += 0.002
                
        return reward
        
    def _next_time_step(self) -> bool:
        """
        Move to next time step.
        
        Returns:
            Whether the episode is done
        """
        self.current_idx += 1
        
        # Check if we've reached the end of available data
        if self.price_data is not None:
            if self.current_idx >= len(self.prices):
                # Important: Set the done flag and return True
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Reached end of price data at index {self.current_idx}, total length: {len(self.prices)}")
                return True
            
            # Update current price
            self.current_price = self.prices[self.current_idx]
        else:
            # Simulate random price movement if no data provided
            price_change = np.random.normal(0, 0.01)  # 1% standard deviation
            self.current_price *= (1 + price_change)
            
            # End after 200 steps if no data provided
            if self.current_idx > 200:
                return True
                
        return False
            
    def _update_market_state(self) -> None:
        """Update market state with latest data and features."""
        if self.price_data is None:
            # Create synthetic features
            features = {
                'price': self.current_price,
                'volatility_regime': 0.5,
                'qerc_trend': 0.5,
                'qerc_momentum': 0.5,
                'iqad_score': 0.0,
                'performance_metric': 0.5
            }
        else:
            # Check if current_idx is within bounds
            if self.current_idx >= len(self.price_data):
                self.logger.warning(f"Current index {self.current_idx} exceeds price data length {len(self.price_data)}. Using last valid index.")
                self.current_idx = len(self.price_data) - 1
                
            # Extract window of recent data
            end_idx = self.current_idx
            start_idx = max(0, end_idx - self.window_size)
            price_window = self.prices[start_idx:end_idx+1]
            
            # Calculate basic features
            if len(price_window) > 1:
                returns = np.diff(price_window) / price_window[:-1]
                vol = np.std(returns) if len(returns) > 0 else 0.0
                trend = np.mean(returns) * 10 if len(returns) > 0 else 0.0
                
                features = {
                    'price': self.current_price,
                    'returns': returns[-1] if len(returns) > 0 else 0.0,
                    'volatility': vol,
                    'trend': trend
                }
                
                # Add additional features from dataframe if available
                if isinstance(self.price_data, pd.DataFrame):
                    for column in ['volatility_regime', 'qerc_trend', 'qerc_momentum', 
                                  'iqad_score', 'performance_metric']:
                        if column in self.price_data.columns:
                            # Safely access the column with bounds checking
                            features[column] = self.price_data.iloc[self.current_idx][column] if self.current_idx < len(self.price_data) else 0.0
            else:
                features = {
                    'price': self.current_price,
                    'returns': 0.0,
                    'volatility': 0.0,
                    'trend': 0.0
                }
                
        # Update market state
        self.market_state.update(features)
        
    def _detect_drift_and_anomalies(self) -> None:
        """Detect drift and anomalies using RiverML."""
        if self.river_ml is None:
            return
            
        try:
            # Prepare features for drift detection
            if isinstance(self.price_data, pd.DataFrame) and self.current_idx > 0:
                # Get returns for drift detection
                price_series = self.price_data['close'].values[:self.current_idx+1]
                if len(price_series) > 1:
                    returns = price_series[-1] / price_series[-2] - 1
                    
                    # Check for drift
                    drift_result = self.river_ml.detect_drift(returns)
                    self.drift_detected = drift_result.get('drift_detected', False)
                    
                    # Prepare features for anomaly detection
                    row = self.price_data.iloc[self.current_idx]
                    feature_dict = {}
                    
                    # Add key indicators
                    for column in ['rsi_14', 'adx', 'volatility_regime', 
                                  'antifragility', 'soc_equilibrium']:
                        if column in row:
                            feature_dict[column] = float(row[column])
                    
                    # Detect anomalies
                    anomaly_result = self.river_ml.detect_anomalies(feature_dict)
                    self.anomaly_detected = anomaly_result.get('is_anomaly', False)
                    
                    logger.debug(f"Drift: {self.drift_detected}, Anomaly: {self.anomaly_detected}")
                    
        except Exception as e:
            logger.error(f"Error in drift/anomaly detection: {str(e)}")
            self.drift_detected = False
            self.anomaly_detected = False
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current episode.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns metrics
        returns = np.array(self.returns)
        
        metrics = {
            "total_return": (self.portfolio_value / self.initial_balance) - 1.0,
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(),
            "win_rate": self.profitable_trades / max(1, self.total_trades),
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "final_portfolio": self.portfolio_value
        }
        
        return metrics
        
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as positive percentage
        """
        if not self.portfolio_values:
            return 0.0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (running_max - self.portfolio_values) / running_max
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0


class QStarTradingPredictor:
    """
    Q* Trading predictor that integrates RiverML and Q* Learning
    for crypto trading predictions.
    """
    
    def __init__(self, 
                 river_ml: RiverOnlineML = None,
                 use_quantum_representation: bool = True,
                 initial_states: int = 200,
                 initial_actions: int = 5,
                 experience_buffer_size: int = 20000,
                 learning_rate: float = 0.05,
                 discount_factor: float = 0.95,
                 training_episodes: int = 200,
                 batch_size: int = 64):
        """
        Initialize Q* Trading predictor.
        
        Args:
            river_ml: RiverML instance for online learning
            use_quantum_representation: Whether to use quantum representation
            initial_states: Initial number of states
            initial_actions: Initial number of actions
            experience_buffer_size: Size of experience buffer
            learning_rate: Initial learning rate
            discount_factor: Discount factor for future rewards
            training_episodes: Maximum training episodes
            batch_size: Batch size for experience replay
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize state flags first to avoid attribute errors
        self._is_initializing = True
        self._is_trained = False
        
        self.river_ml = river_ml if river_ml is not None else RiverOnlineML()
        
        # Initialize Q* Learning agent
        self.agent = SophisticatedQLearningAgent(
            states=initial_states,
            actions=initial_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=1.0,
            min_exploration_rate=0.05,
            exploration_decay_rate=0.99,
            use_adaptive_learning_rate=True,
            use_experience_replay=True,
            experience_buffer_size=experience_buffer_size,
            batch_size=batch_size,
            max_episodes=training_episodes,
            max_steps_per_episode=10000,  # Long episodes for backtesting
            use_quantum_representation=use_quantum_representation
        )
        
        try:
            # Trading environment
            self.env = TradingEnvironment(
                river_ml=self.river_ml,
                window_size=50,
                #initial_balance,
                #transaction_fee
            )
            self.logger.info(f"Trading environment initialized with {self.env.num_states} states and {self.env.num_actions} actions")
        except Exception as e_env_init:
            self.logger.error(f"Failed to initialize TradingEnvironment: {e_env_init}", exc_info=True)
            self.env = None  # Set env to None on failure
        
        river_config = { 
            'drift_detector_type': 'adwin', 
            'anomaly_detector_type': 'hst',
            'feature_window': 50, 
            'drift_sensitivity': 0.05, 
            'anomaly_threshold': 0.75,
            'enable_feature_selection': False,  # Disable feature selection temporarily
            'log_level': self.logger.level
        }
        self.river_ml = RiverOnlineML(**river_config)
        
        # Initialize model state
        self.model_state = {}        
    
        # Load existing state if available
        success = self.load_state()
        
        # Set trained flag based on state loading
        self._is_trained = success
        
        # Set initialization complete
        self._is_initializing = False    
        
        # Only show warning if no state was loaded
        if not success and not self._is_initializing:
            self.logger.warning("Model not trained yet - predictions may be unreliable")
            
        # For backward compatibility - use the same attribute name throughout the class
        self.is_trained = self._is_trained
        self.backtest_results = None
        
        self.logger.info(f"Q* Trading Predictor initialized. Trained: {self.is_trained}")
    
    import os
    def train(self, price_data: pd.DataFrame, window_size: int = 50,
             initial_balance: float = 10000.0, transaction_fee: float = 0.001,
             epochs: int = 3) -> Dict[str, Any]:
        """
        Train the Q* agent on historical price data.
        """
        logger.info(f"Training Q* Trading Predictor on {len(price_data)} data points")
        
        # Prepare data - ensure it has necessary columns
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Ensure we have sufficient data
        if len(price_data) <= window_size:
            raise ValueError(f"Not enough data points. Need more than {window_size} but got {len(price_data)}")
            
        # Create trading environment
        self.env = TradingEnvironment(
            river_ml=self.river_ml,
            price_data=price_data,
            window_size=window_size,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee
        )
        
        # Ensure agent dimensions match environment
        if self.agent.states < self.env.num_states:
            logger.info(f"Resizing agent from {self.agent.states} to {self.env.num_states} states")
            self.agent.resize_q_table(self.env.num_states, self.agent.actions)
        
        # Train for multiple epochs
        best_return = -float('inf')
        best_agent_path = None
        
        try:
            for epoch in range(epochs):
                logger.info(f"Training epoch {epoch+1}/{epochs}")
                
                # Train agent with exception handling
                try:
                    converged, episodes = self.agent.train(self.env)
                    
                    # Evaluate performance
                    eval_metrics = self.agent.evaluate(self.env, num_episodes=1)
                    current_return = self.env.get_performance_metrics()["total_return"]
                    
                    logger.info(f"Epoch {epoch+1} - Return: {current_return:.4f}, "
                              f"Win Rate: {self.env.get_performance_metrics()['win_rate']:.4f}")
                    
                    # Save if best performance
                    if current_return > best_return:
                        best_return = current_return
                        best_agent_path = f"temp_qstar_trading_agent_{epoch}.pkl"
                        self.agent.save(best_agent_path)
                except Exception as e:
                    logger.error(f"Error during training epoch {epoch+1}: {str(e)}", exc_info=True)
                    continue
            
            # Load best agent
            if best_agent_path and os.path.exists(best_agent_path):
                self.agent = SophisticatedQLearningAgent.load(best_agent_path)
                import os
                for epoch in range(epochs):
                    path = f"temp_qstar_trading_agent_{epoch}.pkl"
                    if os.path.exists(path):
                        os.remove(path)
        
            # Run final backtest
            self.backtest_results = self._run_backtest(price_data)
            
            # Mark as trained
            self.is_trained = True
            
            # Return performance metrics
            return {
                "training_metrics": self.agent.metrics.get_summary(),
                "backtest_metrics": self.backtest_results
            }
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            self.is_trained = False
            return {
                "error": str(e),
                "training_metrics": {},
                "backtest_metrics": {}
            }
        
    def _run_backtest(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data with improved error handling.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Backtest metrics
        """
        # Default return value in case of error
        default_results = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "final_portfolio": self.env.initial_balance if hasattr(self.env, 'initial_balance') else 10000.0,
            "actions": [],
            "positions": [],
            "portfolio_values": []
        }
        
        try:
            # Reset environment for clean backtest
            self.env.reset()
            
            # Set agent to no exploration for backtest
            original_exploration = self.agent.exploration_rate
            self.agent.exploration_rate = 0.0
            
            # Tracking variables
            state = self.env.reset()
            done = False
            actions = []
            positions = []
            portfolio_values = []
            
            # Safety counter to prevent infinite loops
            max_steps = len(price_data) * 2
            step_count = 0
            
            while not done and step_count < max_steps:
                # Choose action from trained policy
                action = self.agent.choose_action(state)
                actions.append(action)
                
                # Take step in environment with error handling
                try:
                    next_state, reward, done = self.env.step(action)
                    
                    # Track positions and portfolio
                    positions.append(self.env.position)
                    portfolio_values.append(self.env.portfolio_value)
                    
                    # Move to next state
                    state = next_state
                except Exception as e:
                    self.logger.error(f"Error during backtest step: {str(e)}", exc_info=True)
                    done = True
                
                step_count += 1
            
            # Restore exploration rate
            self.agent.exploration_rate = original_exploration
            
            # Get performance metrics
            try:
                metrics = self.env.get_performance_metrics()
            except Exception as e:
                self.logger.error(f"Error getting performance metrics: {str(e)}", exc_info=True)
                metrics = default_results
            
            # Add actions and positions to results
            metrics["actions"] = actions
            metrics["positions"] = positions
            metrics["portfolio_values"] = portfolio_values
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            return default_results
        
    def predict(self, dataframe: pd.DataFrame, current_position: float = 0.0) -> Dict[str, Any]:
        """
        Generate trading prediction with memory optimization.
        
        Args:
            dataframe: Current market data
            current_position: Current position size
            
        Returns:
            Prediction dictionary with action and confidence
        """
        # Force garbage collection before prediction to clean up memory
        import gc
        gc.collect()
        
        
        # Skip warning during initialization
        if not hasattr(self, '_is_trained') and not getattr(self, '_is_initializing', False):
            self.logger.warning("Model not trained yet - predictions may be unreliable")
            
        # Default results
        results = {
            "action": 0,
            "action_name": "HOLD",
            "confidence": 0.5,
            "q_values": [0.0] * 5,
            "drift_detected": False,
            "anomaly_detected": False,
            "state": 0
        }
        
        try:
            # Create temporary environment with current data
            temp_env = None
            try:
                temp_env = TradingEnvironment(
                    river_ml=self.river_ml,
                    price_data=dataframe,
                    window_size=50
                )
                
                # Set the current position
                temp_env.position = current_position
                temp_env.position_value = current_position * temp_env.current_price
                
                # Get current state
                state = temp_env.market_state.get_state_index()
                
                # Get drift and anomaly information
                temp_env._detect_drift_and_anomalies()
                drift_detected = temp_env.drift_detected
                anomaly_detected = temp_env.anomaly_detected
                
                # Get Q-values for current state
                q_values = self.agent.q_table[state, :]
                
                # Get action with highest Q-value
                action = np.argmax(q_values)
                
                # Calculate confidence
                sorted_q_values = np.sort(q_values)[::-1]  # Descending order
                if len(sorted_q_values) > 1:
                    max_separation = sorted_q_values[0] - sorted_q_values[1]
                    confidence = min(0.95, max(0.5, max_separation))
                else:
                    confidence = 0.5
                    
                # Adjust confidence based on drift and anomalies
                if drift_detected:
                    confidence *= 0.8  # Reduce confidence during regime changes
                if anomaly_detected:
                    confidence *= 0.7  # Reduce confidence during anomalies
                    
                # Update results
                results["action"] = int(action)
                results["action_name"] = TradingAction.get_action_name(action)
                results["confidence"] = float(confidence)
                results["q_values"] = q_values.tolist()
                results["drift_detected"] = drift_detected
                results["anomaly_detected"] = anomaly_detected
                results["state"] = int(state)
                return results
            finally:
                # Proper cleanup
                if temp_env is not None:
                    temp_env.close()  # Call close first
                    temp_env = None   # Then remove reference
                    
            # Force garbage collection after prediction
            gc.collect()
            
            return results
                
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return results
        
    def save_model(self, filepath: str) -> None:
        """
        Save Q* trading model to file.
        
        Args:
            filepath: Path to save model
        """
        self.agent.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath: str, river_ml: Optional[RiverOnlineML] = None) -> 'QStarTradingPredictor':
        """
        Load Q* trading model from file.
        
        Args:
            filepath: Path to load model from
            river_ml: RiverML instance or None to create new one
            
        Returns:
            Loaded model
        """
        agent = SophisticatedQLearningAgent.load(filepath)
        
        if river_ml is None:
            river_ml = RiverOnlineML()
            
        predictor = cls(river_ml=river_ml)
        predictor.agent = agent
        predictor.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return predictor


    def load_state(self, filename: str = None) -> bool:
        """
        Load QStar River state from a pickle file.
        
        Args:
            filename: Pickle file to load state from. If None, uses standard filename.
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            # Determine models directory path
            if hasattr(self, 'config') and 'user_data_dir' in self.config:
                models_dir = os.path.join(self.config['user_data_dir'], 'models')
            else:
                # Look for standard Freqtrade models directory
                freqtrade_dir = os.path.expanduser('/home/ashina/freqtrade')
                models_dir = os.path.join(freqtrade_dir, 'user_data', 'models')
                
                # Alternative using environment variable if available
                if 'FREQTRADE_USER_DATA_DIR' in os.environ:
                    models_dir = os.path.join(os.environ['FREQTRADE_USER_DATA_DIR'], 'models')
            
            # Use standard filename if none provided
            if filename is None:
                filename = os.path.join(models_dir, "qstar_river_ml_state.pkl")
            elif not os.path.isabs(filename):
                # If relative path, make it absolute
                filename = os.path.join(models_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filename):
                self.logger.warning(f"State file not found: {filename}")
                return False
            
            # Load state dictionary
            with open(filename, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Restore River ML state if available
            if 'river_ml_state' in state_dict and hasattr(self, 'river_ml') and self.river_ml is not None:
                river_state = state_dict['river_ml_state']
                
                # Restore regression models
                if 'regression_models' in river_state and hasattr(self.river_ml, 'regression_models'):
                    self.river_ml.regression_models = river_state['regression_models']
                
                # Restore classification models
                if 'classification_models' in river_state and hasattr(self.river_ml, 'classification_models'):
                    self.river_ml.classification_models = river_state['classification_models']
                
                # Restore drift detectors
                if 'drift_detectors' in river_state and hasattr(self.river_ml, 'drift_detectors'):
                    self.river_ml.drift_detectors = river_state['drift_detectors']
                
                # Restore anomaly detectors
                if 'anomaly_detectors' in river_state and hasattr(self.river_ml, 'anomaly_detectors'):
                    self.river_ml.anomaly_detectors = river_state['anomaly_detectors']
                
                # Restore feature selectors
                if 'feature_selectors' in river_state and hasattr(self.river_ml, 'feature_selectors'):
                    self.river_ml.feature_selectors = river_state['feature_selectors']
                
                # Restore other attributes
                for attr in ['statistics', 'feature_window', 'drift_sensitivity', 'anomaly_threshold']:
                    if attr in river_state and hasattr(self.river_ml, attr):
                        setattr(self.river_ml, attr, river_state[attr])
            
            # Restore QStar Predictor state if available
            if 'model_state' in state_dict and hasattr(self, 'model_state'):
                self.model_state = state_dict['model_state']
                
            # Restore hyperparameters and configuration
            if 'hyperparameters' in state_dict and hasattr(self, 'hyperparameters'):
                self.hyperparameters = state_dict['hyperparameters']
                
            # Restore feature information
            if 'feature_columns' in state_dict and hasattr(self, 'feature_columns'):
                self.feature_columns = state_dict['feature_columns']
                
            # Restore normalization parameters
            if 'feature_stats' in state_dict and hasattr(self, 'feature_stats'):
                self.feature_stats = state_dict['feature_stats']
                
            # Restore embeddings
            if 'market_embeddings' in state_dict and hasattr(self, 'market_embeddings'):
                self.market_embeddings = state_dict['market_embeddings']
                
            # Restore quantum parameters
            if 'quantum_params' in state_dict and hasattr(self, 'quantum_params'):
                self.quantum_params = state_dict['quantum_params']
                
            # Restore performance metrics
            if 'performance_metrics' in state_dict and hasattr(self, 'performance_metrics'):
                self.performance_metrics = state_dict['performance_metrics']
                
            # Restore preprocessors
            if 'preprocessors' in state_dict and hasattr(self, 'preprocessors'):
                self.preprocessors = state_dict['preprocessors']
                
            # Restore agent state if available
            if 'q_table' in state_dict and hasattr(self, 'agent') and hasattr(self.agent, 'q_table'):
                self.agent.q_table = state_dict['q_table']
                
            # Restore agent parameters
            if 'agent_params' in state_dict and hasattr(self, 'agent'):
                for key, value in state_dict['agent_params'].items():
                    if hasattr(self.agent, key):
                        setattr(self.agent, key, value)
                
            # Set trained flag
            self._is_trained = True
            if hasattr(self, 'is_trained'):
                self.is_trained = True
            
            # Log success
            self.logger.info(f"QStar River state successfully loaded from {filename}")
            if 'metadata' in state_dict and 'saved_at' in state_dict['metadata']:
                self.logger.info(f"State was saved at: {state_dict['metadata']['saved_at']}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading QStar River state: {e}", exc_info=True)
            return False
    
    def save_state(self, custom_filename=None, force=False) -> bool:
        """
        Save QStar River state to pickle file for continued learning.
        
        Args:
            custom_filename: Optional custom filename to use
            force: Force save regardless of timing conditions
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Get current time for metadata
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            
            # Determine models directory path
            if hasattr(self, 'config') and 'user_data_dir' in self.config:
                models_dir = os.path.join(self.config['user_data_dir'], 'models')
            else:
                # Look for standard Freqtrade models directory
                freqtrade_dir = os.path.expanduser('/home/ashina/freqtrade')
                models_dir = os.path.join(freqtrade_dir, 'user_data', 'models')
                
                # Alternative using environment variable if available
                if 'FREQTRADE_USER_DATA_DIR' in os.environ:
                    models_dir = os.path.join(os.environ['FREQTRADE_USER_DATA_DIR'], 'models')
            
            # Create directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Use standard filename unless custom is provided
            if custom_filename:
                if not custom_filename.endswith('.pkl'):
                    custom_filename += '.pkl'
                filename = os.path.join(models_dir, custom_filename)
            else:
                filename = os.path.join(models_dir, "qstar_river_ml_state.pkl")
            
            # Create state dictionary containing all essential model components
            state_dict = {
                'timestamp': timestamp,
                'version': getattr(self, 'VERSION', '1.0.0'),
                'metadata': {
                    'saved_at': current_time.isoformat(),
                    'description': 'QStar River ML State',
                }
            }
            
            # Save River ML state if available
            if hasattr(self, 'river_ml') and self.river_ml is not None:
                # For River ML objects, we need to save the individual components
                river_state = {}
                
                # Save regression models
                if hasattr(self.river_ml, 'regression_models'):
                    river_state['regression_models'] = self.river_ml.regression_models
                
                # Save classification models
                if hasattr(self.river_ml, 'classification_models'):
                    river_state['classification_models'] = self.river_ml.classification_models
                
                # Save drift detectors
                if hasattr(self.river_ml, 'drift_detectors'):
                    river_state['drift_detectors'] = self.river_ml.drift_detectors
                
                # Save anomaly detectors
                if hasattr(self.river_ml, 'anomaly_detectors'):
                    river_state['anomaly_detectors'] = self.river_ml.anomaly_detectors
                
                # Save feature selectors
                if hasattr(self.river_ml, 'feature_selectors'):
                    river_state['feature_selectors'] = self.river_ml.feature_selectors
                
                # Save any other important attributes
                for attr in ['statistics', 'feature_window', 'drift_sensitivity', 'anomaly_threshold']:
                    if hasattr(self.river_ml, attr):
                        river_state[attr] = getattr(self.river_ml, attr)
                
                state_dict['river_ml_state'] = river_state
            
            # Save agent state if available
            if hasattr(self, 'agent'):
                if hasattr(self.agent, 'q_table'):
                    state_dict['q_table'] = self.agent.q_table
                
                # Save agent parameters
                state_dict['agent_params'] = {}
                for param in ['learning_rate', 'discount_factor', 'exploration_rate', 
                              'min_exploration_rate', 'exploration_decay_rate']:
                    if hasattr(self.agent, param):
                        state_dict['agent_params'][param] = getattr(self.agent, param)
                        
            # Save hyperparameters and configuration
            if hasattr(self, 'hyperparameters'):
                state_dict['hyperparameters'] = self.hyperparameters
                
            # Save feature information
            if hasattr(self, 'feature_columns'):
                state_dict['feature_columns'] = self.feature_columns
                
            # Save normalization parameters if available
            if hasattr(self, 'feature_stats'):
                state_dict['feature_stats'] = self.feature_stats
                
            # Save any embeddings or representations
            if hasattr(self, 'market_embeddings'):
                state_dict['market_embeddings'] = self.market_embeddings
                
            # Save quantum representation parameters if applicable
            if hasattr(self, 'quantum_params'):
                state_dict['quantum_params'] = self.quantum_params
                
            # Save performance tracking metrics
            if hasattr(self, 'performance_metrics'):
                state_dict['performance_metrics'] = self.performance_metrics
                
            # Save preprocessors
            if hasattr(self, 'preprocessors'):
                state_dict['preprocessors'] = self.preprocessors
                
            # Save QStar Predictor state if available
            if hasattr(self, 'model_state'):
                state_dict['model_state'] = self.model_state
            
            # Write the state dictionary to file - NOT nested in any if block
            try:
                # Create temporary file
                temp_filename = f"{filename}.tmp"
                with open(temp_filename, 'wb') as f:
                    pickle.dump(state_dict, f)
                
                # Rename to final filename (atomic operation)
                os.replace(temp_filename, filename)
                
                self.logger.info(f"QStar River state successfully saved to {filename}")
                return True
            except Exception as e:
                self.logger.error(f"Error writing state file: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preparing QStar River state: {e}", exc_info=True)
            return False


    def _cleanup_old_state_files(self, models_dir: str, keep_count: int = 5) -> None:
        """
        Clean up old state files, keeping only the most recent ones.
        
        Args:
            models_dir: Directory containing model files
            keep_count: Number of recent files to keep
        """
        try:
            # Find all QStar River state files
            state_files = []
            
            # Check if directory exists before attempting to list files
            if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
                self.logger.warning(f"Models directory not found for cleanup: {models_dir}")
                return
                
            for filename in os.listdir(models_dir):
                if filename.startswith("qstar_river_state_") and filename.endswith(".pkl"):
                    filepath = os.path.join(models_dir, filename)
                    state_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            state_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove older files beyond keep_count
            if len(state_files) > keep_count:
                for filepath, _ in state_files[keep_count:]:
                    os.remove(filepath)
                    self.logger.debug(f"Removed old QStar River state file: {filepath}")
                    
        except Exception as e:
            self.logger.warning(f"Error cleaning up old state files: {e}")
        
    def plot_backtest_results(self, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save plot or None to display
        """
        if self.backtest_results is None:
            logger.warning("No backtest results available")
            return
            
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Get data
        portfolio_values = self.backtest_results["portfolio_values"]
        positions = self.backtest_results["positions"]
        actions = self.backtest_results["actions"]
        
        # Create time indices
        time_indices = list(range(len(portfolio_values)))
        
        # Plot portfolio value
        axes[0].plot(time_indices, portfolio_values)
        axes[0].set_title("Portfolio Value")
        axes[0].set_ylabel("Value ($)")
        axes[0].grid(True)
        
        # Plot position size
        axes[1].plot(time_indices, positions)
        axes[1].set_title("Position Size")
        axes[1].set_ylabel("Position")
        axes[1].grid(True)
        
        # Plot actions
        action_names = {
            TradingAction.BUY: "BUY",
            TradingAction.SELL: "SELL",
            TradingAction.HOLD: "HOLD",
            TradingAction.REDUCE: "REDUCE",
            TradingAction.INCREASE: "INCREASE"
        }
        
        # Convert actions to names
        action_texts = [action_names.get(a, str(a)) for a in actions]
        
        # Create markers for non-HOLD actions
        for idx, action in enumerate(actions):
            if action != TradingAction.HOLD:
                color = 'g' if action in [TradingAction.BUY, TradingAction.INCREASE] else 'r'
                axes[0].scatter(idx, portfolio_values[idx], color=color, marker='o')
        
        # Plot actions as a heatmap
        actions_array = np.array(actions)
        unique_actions = sorted(list(set(actions)))
        action_matrix = np.zeros((len(unique_actions), len(actions)))
        
        for i, action_type in enumerate(unique_actions):
            action_matrix[i, :] = (actions_array == action_type).astype(int)
        
        im = axes[2].imshow(action_matrix, aspect='auto', cmap='viridis')
        axes[2].set_title("Trading Actions")
        axes[2].set_ylabel("Action Type")
        
        # Set y-ticks to action names
        axes[2].set_yticks(range(len(unique_actions)))
        axes[2].set_yticklabels([action_names.get(a, str(a)) for a in unique_actions])
        
        # Add performance metrics as text
        metrics_text = (
            f"Total Return: {self.backtest_results['total_return']:.2%}\n"
            f"Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {self.backtest_results['max_drawdown']:.2%}\n"
            f"Win Rate: {self.backtest_results['win_rate']:.2%}\n"
            f"Total Trades: {self.backtest_results['total_trades']}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def stop(self):
        """Called when strategy is stopped."""
        try:
            # Save models state
            if hasattr(self, 'qstar_predictor') and self.qstar_predictor is not None:
                self.qstar_predictor.save_state(force=True)  # Force save regardless of timing
                
            self.logger.info("QStar strategy models saved")
        except Exception as e:
            self.logger.error(f"Error saving models during shutdown: {e}")
        
        # Call parent method
        super().stop()

def prepare_trading_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare trading data for Q* Learning.
    
    Args:
        df: Raw dataframe with OHLCV data
        
    Returns:
        Processed dataframe with features
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Ensure essential columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Calculate common technical indicators using TA-Lib if available
    try:
        import talib
        
        # Momentum indicators
        data['rsi_14'] = talib.RSI(data['close'], timeperiod=14)
        
        # Trend indicators
        data['sma_20'] = talib.SMA(data['close'], timeperiod=20)
        data['sma_50'] = talib.SMA(data['close'], timeperiod=50)
        data['sma_200'] = talib.SMA(data['close'], timeperiod=200)
        
        # Volatility indicators
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Calculate MACD
        data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
    except ImportError:
        logger.warning("TA-Lib not available, using pandas for basic indicators")
        
        # Basic indicators without TA-Lib
        # Moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Simple RSI implementation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate volatility regime
    returns = data['close'].pct_change()
    data['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
    data['volatility_regime'] = (data['volatility'] - data['volatility'].rolling(window=100).min()) / (
        data['volatility'].rolling(window=100).max() - data['volatility'].rolling(window=100).min()
    )
    
    # Calculate trend features
    data['trend'] = data['close'].diff(10) / data['close'].shift(10)
    data['momentum'] = data['close'].pct_change(10)
    
    # Fill missing values
    data = data.fillna(method='bfill').fillna(0)
    
    # Add placeholder columns for quantum components
    # These would normally be calculated by the actual components
    data['qerc_trend'] = 0.5
    data['qerc_momentum'] = 0.5
    data['qerc_volatility'] = 0.5
    data['qerc_regime'] = 0.5
    data['iqad_score'] = 0.0
    data['performance_metric'] = 0.5
    
    return data


def run_trading_example():
    """Run example of Q* Trading integration with sample data."""
    logger.info("Starting Q* Trading example")
    
    # Create RiverML instance
    river_ml = RiverOnlineML(
        drift_detector_type='adwin',
        anomaly_detector_type='hst',
        feature_window=50,
        drift_sensitivity=0.05,
        anomaly_threshold=0.95
    )
    
    # Create sample price data
    try:
        # Try to load sample data using pandas-datareader
        import pandas_datareader as pdr
        import datetime as dt
        
        start_date = dt.datetime.now() - dt.timedelta(days=1000)
        end_date = dt.datetime.now()
        
        logger.info(f"Fetching BTC-USD data from {start_date.date()} to {end_date.date()}")
        
        # Fetch data
        df = pdr.data.get_data_yahoo('BTC-USD', start=start_date, end=end_date)
        logger.info(f"Fetched {len(df)} data points")
        
    except (ImportError, Exception) as e:
        logger.warning(f"Error fetching data: {str(e)}")
        logger.info("Using synthetic price data")
        
        # Create synthetic price data
        np.random.seed(42)
        n_days = 500
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Create OHLCV data
        price = 10000.0  # Starting price
        prices = [price]
        
        for _ in range(n_days - 1):
            # Log-normal returns
            ret = np.random.normal(0.0002, 0.02)
            price *= (1 + ret)
            prices.append(price)
        
        prices = np.array(prices)
        
        # Create dataframe
        df = pd.DataFrame({
            'open': prices * (1 - np.random.uniform(0, 0.01, n_days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_days) * prices
        }, index=dates)
    
    # Prepare data
    processed_data = prepare_trading_data(df)
    
    # Split data for training and testing
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data.iloc[:train_size]
    test_data = processed_data.iloc[train_size:]
    
    logger.info(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    
    # Create and train Q* Trading predictor
    predictor = QStarTradingPredictor(
        river_ml=river_ml,
        use_quantum_representation=True,
        initial_states=200,
        training_episodes=100
    )
    
    # Train the predictor
    training_result = predictor.train(
        price_data=train_data,
        window_size=50,
        initial_balance=10000.0,
        transaction_fee=0.001,
        epochs=2
    )
    
    logger.info("Training completed")
    logger.info(f"Training metrics: {training_result['training_metrics']}")
    logger.info(f"Backtest metrics: {training_result['backtest_metrics']}")
    
    # Plot backtest results
    predictor.plot_backtest_results()
    
    # Save model
    predictor.save_model("qstar_trading_model.pkl")
    
    # Test on unseen data
    test_env = TradingEnvironment(
        river_ml=river_ml,
        price_data=test_data,
        window_size=50,
        initial_balance=10000.0,
        transaction_fee=0.001
    )
    
    # Evaluate on test data
    logger.info("Evaluating on test data")
    
    # Turn off exploration for testing
    original_exploration = predictor.agent.exploration_rate
    predictor.agent.exploration_rate = 0.0
    
    # Run evaluation
    eval_metrics = predictor.agent.evaluate(test_env, num_episodes=1)
    test_metrics = test_env.get_performance_metrics()
    
    # Restore exploration rate
    predictor.agent.exploration_rate = original_exploration
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Generate sample predictions
    logger.info("Generating sample predictions")
    
    for i in range(5):
        idx = np.random.randint(0, len(test_data))
        window = test_data.iloc[max(0, idx-49):idx+1]
        
        prediction = predictor.predict(window)
        
        logger.info(f"Prediction {i+1}:")
        logger.info(f"  Action: {prediction['action_name']}")
        logger.info(f"  Confidence: {prediction['confidence']:.4f}")
        logger.info(f"  Drift detected: {prediction['drift_detected']}")
        logger.info(f"  Anomaly detected: {prediction['anomaly_detected']}")
    
    logger.info("Example completed")


if __name__ == "__main__":
    """Run demonstration of Q* Trading integration."""
    run_trading_example()