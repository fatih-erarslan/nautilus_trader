#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:43:45 2025

@author: ashina
"""

# --- qstar_learner.py ---

import os
import numpy as np
import pandas as pd
import time
import logging
import threading
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

class QStarLearner:
    """
    Manages incremental learning for QStar strategy using offline data.
    """
    
    def __init__(self, strategy=None, 
                 data_file: str = 'data/tengri_offline_data_ALL.csv',
                 model_file: str = 'models/qstar_model.pkl',
                 learning_rate: float = 0.05,
                 use_quantum: bool = True,
                 log_level: int = logging.INFO):
        """
        Initialize QStar learner.
        
        Args:
            strategy: QStar strategy instance or None
            data_file: Path to CSV data file
            model_file: Path to model file
            learning_rate: Learning rate for Q-learning
            use_quantum: Whether to use quantum-inspired features
            log_level: Logging level
        """
        self.strategy = strategy
        self.data_file = data_file
        self.model_file = model_file
        self.learning_rate = learning_rate
        self.use_quantum = use_quantum
        
        # Setup logging
        self.logger = logging.getLogger("QStarLearner")
        self.logger.setLevel(log_level)
        
        # State variables
        self.q_table = None
        self.feature_stats = {}
        self.is_initialized = False
        self.is_training = False
        self._training_lock = threading.RLock()
        
        # Get useful functions from strategy
        if strategy:
            self.dp = strategy.dp
            self.timeframe = strategy.timeframe
        
        # Initialize Q-table and load model
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize Q-table and load model if available.
        
        Returns:
            Success flag
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            # Initialize Q-table (states × actions)
            n_states = 100  # Market states
            n_actions = 5   # Trading actions (buy, sell, hold, reduce, increase)
            
            # Check if model file exists
            if os.path.exists(self.model_file):
                # Load existing model
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.q_table = model_data.get('q_table')
                self.feature_stats = model_data.get('feature_stats', {})
                
                self.logger.info(f"Loaded Q-table with shape {self.q_table.shape} from {self.model_file}")
            else:
                # Create new Q-table
                self.q_table = np.zeros((n_states, n_actions))
                self.logger.info(f"Created new Q-table with shape {self.q_table.shape}")
            
            # Check if data file exists
            if os.path.exists(self.data_file):
                self.logger.info(f"Found data file: {self.data_file}")
                
                # Start background learning thread
                threading.Thread(
                    target=self._background_learning,
                    daemon=True,
                    name="QStarLearningThread"
                ).start()
            else:
                self.logger.warning(f"Data file not found: {self.data_file}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    def _background_learning(self) -> None:
        """Learn from offline data in background thread."""
        try:
            self.logger.info("Starting background learning from offline data")
            
            # Wait for strategy to be fully initialized
            time.sleep(10)
            
            # Learn from data file
            self._learn_from_csv()
            
            self.logger.info("Background learning completed")
            
        except Exception as e:
            self.logger.error(f"Background learning error: {e}", exc_info=True)
    
    def _learn_from_csv(self) -> None:
        """Learn from CSV data file."""
        try:
            # Avoid training if already in progress
            with self._training_lock:
                if self.is_training:
                    self.logger.debug("Training already in progress, skipping")
                    return
                self.is_training = True
            
            self.logger.info(f"Learning from {self.data_file}")
            
            # Load data in chunks to reduce memory usage
            chunk_size = 10000
            total_rows = 0
            
            for chunk_idx, chunk in enumerate(pd.read_csv(self.data_file, chunksize=chunk_size)):
                self.logger.info(f"Processing chunk {chunk_idx+1} with {len(chunk)} rows")
                
                # Convert chunk to features and learn
                self._learn_from_dataframe(chunk)
                
                total_rows += len(chunk)
                
                # Save model periodically
                if chunk_idx % 5 == 0:
                    self._save_model()
            
            # Final save
            self._save_model()
            
            self.logger.info(f"Learned from {total_rows} rows in {self.data_file}")
            
        except Exception as e:
            self.logger.error(f"Error learning from CSV: {e}", exc_info=True)
        finally:
            with self._training_lock:
                self.is_training = False
    
    def _learn_from_dataframe(self, dataframe: pd.DataFrame) -> None:
        """
        Learn from dataframe.
        
        Args:
            dataframe: Input dataframe with feature columns
        """
        try:
            # Check for required columns
            required_cols = ['close', 'soc_index', 'performance_metric', 'qerc_trend']
            if not all(col in dataframe.columns for col in required_cols):
                missing = [col for col in required_cols if col not in dataframe.columns]
                self.logger.warning(f"Missing required columns: {missing}")
                return
            
            # Extract features and convert to states
            states = self._extract_states(dataframe)
            
            # Define actions and simulate trading
            current_state = states[0] if states else 0
            
            for i in range(1, len(states)):
                # Choose action based on Q-table (epsilon-greedy)
                action = self._choose_action(current_state)
                
                # Calculate reward based on outcome
                next_state = states[i]
                reward = self._calculate_reward(
                    action, 
                    dataframe['qerc_trend'].iloc[i-1], 
                    dataframe['qerc_trend'].iloc[i]
                )
                
                # Update Q-table (Q-learning algorithm)
                self._update_q_table(current_state, action, reward, next_state)
                
                # Move to next state
                current_state = next_state
            
        except Exception as e:
            self.logger.error(f"Error learning from dataframe: {e}", exc_info=True)
    
    def _extract_states(self, dataframe: pd.DataFrame) -> List[int]:
        """
        Extract states from dataframe.
        
        Args:
            dataframe: Input dataframe
            
        Returns:
            List of state indices
        """
        # Define feature columns to use
        feature_cols = [
            'close', 'volume', 'rsi_14', 'adx', 'macd',
            'volatility_regime', 'antifragility', 
            'soc_equilibrium', 'soc_fragility',
            'qerc_trend', 'performance_metric'
        ]
        
        # Use available columns
        available_cols = [col for col in feature_cols if col in dataframe.columns]
        
        if not available_cols:
            self.logger.warning("No feature columns available")
            return [0] * len(dataframe)
        
        # Extract and normalize features
        features = dataframe[available_cols].values
        normalized = self._normalize_features(features, available_cols)
        
        # Convert to discrete states
        n_states = self.q_table.shape[0]
        states = []
        
        for i in range(len(normalized)):
            # Simple discretization: combine features into a single state index
            feature_vals = normalized[i]
            
            # Map feature values to buckets (N buckets per feature)
            n_buckets = 3
            buckets = (feature_vals * n_buckets).astype(int).clip(0, n_buckets-1)
            
            # Calculate state index using positional encoding
            # This maps the combination of bucketed features to a unique state
            state_idx = 0
            for j, bucket in enumerate(buckets):
                state_idx += bucket * (n_buckets ** j)
            
            # Ensure state index is within bounds
            state_idx = state_idx % n_states
            states.append(state_idx)
        
        return states
    
    def _normalize_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Normalize features based on historical stats.
        
        Args:
            features: Feature array
            feature_names: Feature column names
            
        Returns:
            Normalized feature array
        """
        normalized = np.zeros_like(features, dtype=np.float32)
        
        # Update feature stats
        for i, name in enumerate(feature_names):
            col_values = features[:, i]
            
            # Calculate column stats
            col_min = np.nanmin(col_values)
            col_max = np.nanmax(col_values)
            col_mean = np.nanmean(col_values)
            col_std = np.nanstd(col_values)
            
            # Update feature stats
            if name not in self.feature_stats:
                self.feature_stats[name] = {
                    'min': col_min,
                    'max': col_max,
                    'mean': col_mean,
                    'std': col_std
                }
            else:
                # Progressive update of min/max
                if col_min < self.feature_stats[name]['min']:
                    self.feature_stats[name]['min'] = col_min
                if col_max > self.feature_stats[name]['max']:
                    self.feature_stats[name]['max'] = col_max
                
                # Simple update of mean/std (not strictly correct but sufficient)
                self.feature_stats[name]['mean'] = (
                    self.feature_stats[name]['mean'] * 0.9 + col_mean * 0.1
                )
                self.feature_stats[name]['std'] = (
                    self.feature_stats[name]['std'] * 0.9 + col_std * 0.1
                )
            
            # Normalize column (min-max scaling)
            min_val = self.feature_stats[name]['min']
            max_val = self.feature_stats[name]['max']
            
            if max_val > min_val:
                normalized[:, i] = (col_values - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5  # Default value
        
        return normalized
    
    def _choose_action(self, state: int) -> int:
        """
        Choose action based on state (epsilon-greedy).
        
        Args:
            state: Current state index
            
        Returns:
            Action index
        """
        # Exploration-exploitation tradeoff
        epsilon = 0.1  # 10% random actions for exploration
        
        if np.random.random() < epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.q_table.shape[1])
        else:
            # Best action (exploitation)
            return np.argmax(self.q_table[state])
    
    def _calculate_reward(self, action: int, current_value: float, next_value: float) -> float:
        """
        Calculate reward for action based on outcome.
        
        Args:
            action: Action taken
            current_value: Current trend value
            next_value: Next trend value
            
        Returns:
            Reward value
        """
        # Calculate trend change
        trend_change = next_value - current_value
        
        # Define actions: 0=buy, 1=sell, 2=hold, 3=reduce, 4=increase
        BUY = 0
        SELL = 1
        HOLD = 2
        REDUCE = 3
        INCREASE = 4
        
        # Base reward is trend change
        base_reward = trend_change * 10  # Scale for better learning
        
        # Adjust based on action
        if trend_change > 0:  # Trend going up
            if action in [BUY, INCREASE]:
                reward = base_reward * 1.5  # Bonus for correct action
            elif action in [SELL, REDUCE]:
                reward = -base_reward  # Penalty for incorrect action
            else:  # HOLD
                reward = base_reward * 0.5  # Reduced reward for hold
        elif trend_change < 0:  # Trend going down
            if action in [SELL, REDUCE]:
                reward = -base_reward * 1.5  # Bonus for correct action
            elif action in [BUY, INCREASE]:
                reward = base_reward  # Penalty for incorrect action
            else:  # HOLD
                reward = base_reward * 0.5  # Reduced penalty for hold
        else:  # No change
            reward = 0.0  # Neutral reward
        
        return reward
    
    def _update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Update Q-table using Q-learning algorithm.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Q-learning update formula:
        # Q(s,a) := Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        alpha = self.learning_rate
        gamma = 0.95  # Discount factor
        
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + alpha * (reward + gamma * max_next_q - current_q)
    
    def _save_model(self) -> None:
        """Save model to file."""
        try:
            # Create model data
            model_data = {
                'q_table': self.q_table,
                'feature_stats': self.feature_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info(f"Model saved to {self.model_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
    
    def predict(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading prediction.
        
        Args:
            dataframe: Input dataframe
            
        Returns:
            Dictionary with prediction
        """
        try:
            if not self.is_initialized or self.q_table is None:
                return {'action': 2, 'confidence': 0.0, 'error': 'Not initialized'}
            
            # Extract state from dataframe
            states = self._extract_states(dataframe)
            
            if not states:
                return {'action': 2, 'confidence': 0.0, 'error': 'No valid state'}
            
            # Use last state for prediction
            state = states[-1]
            
            # Get Q-values for this state
            q_values = self.q_table[state]
            
            # Choose best action
            action = np.argmax(q_values)
            
            # Calculate confidence based on Q-value separation
            sorted_q = np.sort(q_values)[::-1]  # Descending
            if len(sorted_q) > 1:
                separation = sorted_q[0] - sorted_q[1]
                confidence = min(0.95, max(0.5, separation / 2.0))
            else:
                confidence = 0.5
            
            # Convert to action value (-1 to 1 range)
            action_value = 0.0  # Default: neutral
            
            if action == 0:  # Buy
                action_value = 1.0
            elif action == 1:  # Sell
                action_value = -1.0
            elif action == 3:  # Reduce
                action_value = -0.5
            elif action == 4:  # Increase
                action_value = 0.5
                
            # Return prediction
            return {
                'action': int(action),
                'action_name': ['BUY', 'SELL', 'HOLD', 'REDUCE', 'INCREASE'][action],
                'action_value': action_value,
                'confidence': float(confidence),
                'q_values': q_values.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            return {'action': 2, 'confidence': 0.0, 'error': str(e)}
    
    def recover(self) -> bool:
        """
        Recovery method for Bluewolf integration.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Attempting to recover QStarLearner")
            
            # Reload model
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.q_table = model_data.get('q_table')
                self.feature_stats = model_data.get('feature_stats', {})
                
                self.logger.info(f"Reloaded Q-table from {self.model_file}")
                
                # Reset training flag
                with self._training_lock:
                    self.is_training = False
                
                return True
            else:
                self.logger.warning(f"Model file not found: {self.model_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery error: {e}", exc_info=True)
            return False

# Integration with QStar
def add_qstar_learner(qstar_strategy):
    """
    Add QStar learner to strategy.
    
    Args:
        qstar_strategy: QStar strategy instance
    """
    try:
        # Set up paths
        data_path = os.path.join(qstar_strategy.config['user_data_dir'], 'data', 'tengri_offline_data_ALL.csv')
        model_path = os.path.join(qstar_strategy.config['user_data_dir'], 'models', 'qstar_model.pkl')
        
        # Create learner
        qstar_strategy.qstar_learner = QStarLearner(
            strategy=qstar_strategy,
            data_file=data_path,
            model_file=model_path,
            learning_rate=0.05,
            use_quantum=not getattr(qstar_strategy, 'force_cpu', False),
            log_level=qstar_strategy.logger.level
        )
        
        # Register with bluewolf if available
        if hasattr(qstar_strategy, 'bluewolf') and qstar_strategy.bluewolf is not None:
            qstar_strategy.bluewolf.register_component('qstar_learner', qstar_strategy.qstar_learner)
        
        # Add update and predict methods to QStar class
        def update_learner_with_data(self, dataframe, metadata):
            """Update QStar learner with new data."""
            if hasattr(self, 'qstar_learner') and self.qstar_learner.is_initialized:
                # Ping bluewolf if available
                if hasattr(self, 'bluewolf') and self.bluewolf is not None:
                    self.bluewolf.ping('qstar_learner')
                
                # We don't need to explicitly update with live data in this implementation
                # The model is pre-trained from the CSV file
                pass
        
        def get_learner_prediction(self, dataframe, metadata):
            """Get prediction from QStar learner."""
            if hasattr(self, 'qstar_learner') and self.qstar_learner.is_initialized:
                # Ping bluewolf if available
                if hasattr(self, 'bluewolf') and self.bluewolf is not None:
                    self.bluewolf.ping('qstar_learner')
                
                # Get prediction
                prediction = self.qstar_learner.predict(dataframe)
                
                # Update dataframe with prediction
                if 'action' in prediction:
                    dataframe['qstar_action'] = prediction['action']
                    dataframe['qstar_confidence'] = prediction['confidence']
                    dataframe['qstar_action_value'] = prediction['action_value']
                
                return prediction
            
            return None
        
        # Add methods to QStar instance
        import types
        qstar_strategy.update_learner_with_data = types.MethodType(update_learner_with_data, qstar_strategy)
        qstar_strategy.get_learner_prediction = types.MethodType(get_learner_prediction, qstar_strategy)
        
        # Patch populate_indicators to include learner prediction
        original_populate_indicators = qstar_strategy.populate_indicators
        
        def patched_populate_indicators(self, dataframe, metadata):
            # Call original method first
            dataframe = original_populate_indicators(dataframe, metadata)
            
            # Add learner prediction
            if hasattr(self, 'get_learner_prediction'):
                self.get_learner_prediction(dataframe, metadata)
            
            return dataframe
        
        # Replace method
        qstar_strategy.populate_indicators = types.MethodType(patched_populate_indicators, qstar_strategy)
        
        return True
        
    except Exception as e:
        qstar_strategy.logger.error(f"Error adding QStar learner: {e}", exc_info=True)
        return False