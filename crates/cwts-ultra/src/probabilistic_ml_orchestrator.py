#!/usr/bin/env python3
"""
CWTS Probabilistic Machine Learning Orchestrator

This module provides Python integration for the probabilistic computing system,
leveraging ML libraries for advanced analytics and orchestration.

Key Features:
- Real-time market data processing with ML models
- Ensemble learning for regime detection
- Deep learning uncertainty quantification
- AutoML for hyperparameter optimization
- Integration with Rust/C++ high-performance backends
"""

import asyncio
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

# ML/Data Science Libraries
try:
    import scikit_learn as sklearn
    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                                 IsolationForest)
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    import xgboost as xgb
    import lightgbm as lgb
    
    # Deep Learning
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    
    # Time Series
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    
    # Bayesian
    import pymc as pm
    import arviz as az
    
    # Optimization
    import optuna
    
except ImportError as e:
    logging.warning(f"Some ML libraries not available: {e}")

# FFI bindings for C++ backend
import ctypes
from ctypes import Structure, c_double, c_size_t, c_int, POINTER

# Financial data
try:
    import yfinance as yf
    import websocket
    import json
except ImportError:
    logging.warning("Financial data libraries not available")

@dataclass
class ProbabilisticMetrics:
    """Probabilistic risk metrics from ML models"""
    var_95: float
    var_99: float
    expected_shortfall: float
    tail_risk_probability: float
    uncertainty_score: float
    regime_probabilities: Dict[str, float]
    model_confidence: float
    prediction_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketData:
    """Market data container"""
    prices: np.ndarray
    returns: np.ndarray
    volume: np.ndarray
    volatility: np.ndarray
    features: Dict[str, np.ndarray]
    timestamp: datetime

class UncertaintyQuantificationNet(nn.Module):
    """Deep learning model for uncertainty quantification"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        # Mean and variance heads for uncertainty quantification
        self.feature_layers = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.var_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # Ensure positive variance
        )
        
    def forward(self, x):
        features = self.feature_layers(x)
        mean = self.mean_head(features)
        variance = self.var_head(features)
        return mean, variance

class BayesianRegimeDetector:
    """Bayesian regime detection using PyMC"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.trace = None
        
    def fit(self, returns: np.ndarray, n_samples: int = 2000):
        """Fit Bayesian regime switching model"""
        with pm.Model() as model:
            # Regime transition probabilities
            p = pm.Dirichlet('p', a=np.ones(self.n_regimes), shape=(self.n_regimes, self.n_regimes))
            
            # Regime-specific parameters
            mu = pm.Normal('mu', mu=0, sigma=0.1, shape=self.n_regimes)
            sigma = pm.Exponential('sigma', lam=10, shape=self.n_regimes)
            
            # Hidden regime states
            regime = pm.Categorical('regime', p=p[0], shape=len(returns))
            
            # Observations
            obs = pm.Normal('obs', mu=mu[regime], sigma=sigma[regime], observed=returns)
            
            # Sample
            self.trace = pm.sample(n_samples, tune=1000, chains=2)
            self.model = model
            
    def predict_regime_probabilities(self, returns: np.ndarray) -> Dict[str, float]:
        """Predict current regime probabilities"""
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Use posterior predictive sampling
        with self.model:
            posterior_pred = pm.sample_posterior_predictive(self.trace, predictions=True)
            
        # Extract regime probabilities (simplified)
        regime_names = ['low_volatility', 'medium_volatility', 'high_volatility', 'crisis']
        regime_probs = {}
        
        for i, name in enumerate(regime_names[:self.n_regimes]):
            # Approximate probability based on posterior samples
            regime_probs[name] = float(np.mean(self.trace.posterior['regime'][-100:] == i))
            
        return regime_probs

class EnsembleRiskModel:
    """Ensemble model combining multiple ML approaches"""
    
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        self.scalers = {}
        self.feature_importance = {}
        self.is_fitted = False
        
    def prepare_features(self, data: MarketData) -> np.ndarray:
        """Create feature matrix from market data"""
        features = []
        
        # Price-based features
        features.extend([
            data.returns,
            np.roll(data.returns, 1),  # Lagged returns
            np.roll(data.returns, 2),
            np.cumsum(data.returns),   # Cumulative returns
        ])
        
        # Volatility features
        rolling_vol = pd.Series(data.returns).rolling(20).std().fillna(0).values
        features.extend([
            data.volatility,
            rolling_vol,
            data.volatility - rolling_vol,  # Volatility spread
        ])
        
        # Volume features
        if len(data.volume) > 0:
            volume_ma = pd.Series(data.volume).rolling(20).mean().fillna(data.volume.mean()).values
            features.extend([
                data.volume,
                volume_ma,
                data.volume / volume_ma,  # Relative volume
            ])
        
        # Technical indicators
        prices = np.cumsum(data.returns) + 100  # Reconstruct prices
        sma_20 = pd.Series(prices).rolling(20).mean().fillna(prices[0]).values
        sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0]).values
        
        features.extend([
            prices / sma_20 - 1,  # Price relative to SMA20
            sma_20 / sma_50 - 1,  # SMA20/SMA50 ratio
        ])
        
        # Additional custom features from data.features
        for key, values in data.features.items():
            features.append(values)
        
        # Stack and transpose to get (n_samples, n_features)
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def fit(self, data: MarketData, target: np.ndarray):
        """Fit ensemble models"""
        X = self.prepare_features(data)
        
        # Fit scalers and models
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[name] = scaler
            
            # Fit model
            model.fit(X_scaled, target)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        self.is_fitted = True
        
    def predict_with_uncertainty(self, data: MarketData) -> Tuple[float, float, Dict[str, float]]:
        """Make ensemble prediction with uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features(data)
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            scaler = self.scalers[name]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions[name] = pred[-1]  # Latest prediction
        
        # Ensemble statistics
        pred_values = list(predictions.values())
        ensemble_mean = np.mean(pred_values)
        ensemble_std = np.std(pred_values)
        
        # Model agreement as confidence measure
        model_agreement = 1.0 - (ensemble_std / (abs(ensemble_mean) + 1e-6))
        model_agreement = np.clip(model_agreement, 0.0, 1.0)
        
        return ensemble_mean, ensemble_std, predictions

class AutoMLOptimizer:
    """AutoML hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        self.best_params = {}
        
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                
                mse = mean_squared_error(y_val, pred)
                scores.append(mse)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params

class ProbabilisticMLOrchestrator:
    """Main orchestrator for probabilistic ML system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.ensemble_model = EnsembleRiskModel()
        self.uncertainty_net = None
        self.bayesian_detector = BayesianRegimeDetector()
        self.automl_optimizer = AutoMLOptimizer()
        
        # Data storage
        self.historical_data = []
        self.predictions_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'mse': [],
            'mae': []
        }
        
        # C++ backend interface
        self.cpp_backend = self._load_cpp_backend()
        
        logging.info("🤖 Probabilistic ML Orchestrator initialized")
        
    def _load_cpp_backend(self):
        """Load C++ backend for high-performance computing"""
        try:
            # Try to load compiled C++ library
            import ctypes.util
            lib_path = ctypes.util.find_library('probabilistic_compute')
            if lib_path:
                lib = ctypes.CDLL(lib_path)
                
                # Define C function signatures
                lib.compute_monte_carlo_var_c.argtypes = [
                    POINTER(c_double), c_double, c_double, c_double, c_size_t
                ]
                lib.estimate_tail_params_c.argtypes = [POINTER(c_double), c_size_t]
                
                return lib
        except Exception as e:
            logging.warning(f"C++ backend not available: {e}")
        
        return None
    
    async def process_real_time_data(self, market_data: MarketData) -> ProbabilisticMetrics:
        """Process real-time market data and generate probabilistic metrics"""
        try:
            # Store historical data
            self.historical_data.append(market_data)
            if len(self.historical_data) > 1000:
                self.historical_data.pop(0)
            
            # Prepare features
            if len(self.historical_data) < 50:
                logging.warning("Insufficient historical data for reliable predictions")
                return self._default_metrics()
            
            # Ensemble prediction with uncertainty
            ensemble_pred, ensemble_std, model_preds = self.ensemble_model.predict_with_uncertainty(market_data)
            
            # Regime detection
            returns_window = np.array([d.returns[-1] if len(d.returns) > 0 else 0.0 
                                     for d in self.historical_data[-100:]])
            regime_probs = self.bayesian_detector.predict_regime_probabilities(returns_window)
            
            # Deep learning uncertainty quantification
            if self.uncertainty_net is not None:
                dl_uncertainty = await self._deep_learning_uncertainty(market_data)
            else:
                dl_uncertainty = ensemble_std
            
            # C++ backend for heavy computations
            var_95, var_99, expected_shortfall = await self._compute_risk_metrics_cpp(market_data)
            
            # Feature importance
            feature_importance = self._aggregate_feature_importance()
            
            # Prediction interval
            confidence_level = 0.95
            pred_interval = (
                ensemble_pred - 1.96 * ensemble_std,
                ensemble_pred + 1.96 * ensemble_std
            )
            
            # Model confidence (based on agreement and historical performance)
            model_confidence = self._calculate_model_confidence(ensemble_std)
            
            metrics = ProbabilisticMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                tail_risk_probability=self._estimate_tail_risk_probability(market_data),
                uncertainty_score=min(ensemble_std, dl_uncertainty),
                regime_probabilities=regime_probs,
                model_confidence=model_confidence,
                prediction_interval=pred_interval,
                feature_importance=feature_importance
            )
            
            # Store prediction for performance tracking
            self.predictions_history.append({
                'timestamp': datetime.now(),
                'prediction': ensemble_pred,
                'uncertainty': ensemble_std,
                'actual': None  # To be filled when actual data arrives
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error processing real-time data: {e}")
            return self._default_metrics()
    
    async def _compute_risk_metrics_cpp(self, market_data: MarketData) -> Tuple[float, float, float]:
        """Use C++ backend for high-performance risk metric computation"""
        if self.cpp_backend is None:
            # Fallback to Python implementation
            return self._compute_risk_metrics_python(market_data)
        
        try:
            # Prepare data for C++
            returns = market_data.returns[-252:]  # Last year of data
            portfolio_value = 1000000.0  # $1M portfolio
            
            # Allocate output buffer
            buffer_size = 10000
            output_buffer = (c_double * buffer_size)()
            
            # Call C++ Monte Carlo function
            self.cpp_backend.compute_monte_carlo_var_c(
                output_buffer,
                c_double(portfolio_value),
                c_double(np.mean(returns)),
                c_double(np.std(returns)),
                c_size_t(buffer_size)
            )
            
            # Convert to numpy array and calculate percentiles
            monte_carlo_results = np.array([output_buffer[i] for i in range(buffer_size)])
            monte_carlo_results = np.sort(monte_carlo_results)
            
            var_95 = float(monte_carlo_results[int(0.05 * buffer_size)])
            var_99 = float(monte_carlo_results[int(0.01 * buffer_size)])
            expected_shortfall = float(np.mean(monte_carlo_results[:int(0.01 * buffer_size)]))
            
            return abs(var_95), abs(var_99), abs(expected_shortfall)
            
        except Exception as e:
            logging.error(f"C++ backend error: {e}")
            return self._compute_risk_metrics_python(market_data)
    
    def _compute_risk_metrics_python(self, market_data: MarketData) -> Tuple[float, float, float]:
        """Python fallback for risk metric computation"""
        returns = market_data.returns[-252:] if len(market_data.returns) >= 252 else market_data.returns
        
        if len(returns) < 10:
            return 5000.0, 10000.0, 15000.0  # Default values
        
        portfolio_value = 1000000.0
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Simple Monte Carlo simulation
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, volatility, 10000)
        portfolio_changes = portfolio_value * simulated_returns
        portfolio_changes.sort()
        
        var_95 = abs(portfolio_changes[int(0.05 * len(portfolio_changes))])
        var_99 = abs(portfolio_changes[int(0.01 * len(portfolio_changes))])
        expected_shortfall = abs(np.mean(portfolio_changes[:int(0.01 * len(portfolio_changes))]))
        
        return var_95, var_99, expected_shortfall
    
    async def _deep_learning_uncertainty(self, market_data: MarketData) -> float:
        """Deep learning based uncertainty quantification"""
        if self.uncertainty_net is None:
            return 0.1  # Default uncertainty
        
        try:
            # Prepare features for neural network
            features = self.ensemble_model.prepare_features(market_data)
            
            # Convert to torch tensor
            X_tensor = torch.FloatTensor(features[-1:])  # Latest observation
            
            # Forward pass
            with torch.no_grad():
                mean, variance = self.uncertainty_net(X_tensor)
                uncertainty = float(torch.sqrt(variance).item())
            
            return uncertainty
            
        except Exception as e:
            logging.error(f"Deep learning uncertainty error: {e}")
            return 0.1
    
    def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across ensemble models"""
        if not self.ensemble_model.is_fitted:
            return {}
        
        feature_names = [
            'returns', 'returns_lag1', 'returns_lag2', 'cumsum_returns',
            'volatility', 'rolling_vol', 'vol_spread',
            'volume', 'volume_ma', 'relative_volume',
            'price_sma20_ratio', 'sma_ratio'
        ]
        
        # Add custom feature names
        for i in range(len(feature_names), 20):  # Assume max 20 features
            feature_names.append(f'feature_{i}')
        
        aggregated_importance = {}
        
        for name, importance in self.ensemble_model.feature_importance.items():
            for i, imp in enumerate(importance[:len(feature_names)]):
                feature_name = feature_names[i]
                if feature_name not in aggregated_importance:
                    aggregated_importance[feature_name] = []
                aggregated_importance[feature_name].append(imp)
        
        # Average importance across models
        final_importance = {}
        for feature, importance_list in aggregated_importance.items():
            final_importance[feature] = float(np.mean(importance_list))
        
        return final_importance
    
    def _calculate_model_confidence(self, uncertainty: float) -> float:
        """Calculate model confidence based on uncertainty and historical performance"""
        # Normalize uncertainty to confidence score
        base_confidence = max(0.0, 1.0 - uncertainty)
        
        # Adjust based on recent prediction accuracy
        if len(self.predictions_history) > 10:
            recent_errors = []
            for pred in self.predictions_history[-10:]:
                if pred['actual'] is not None:
                    error = abs(pred['prediction'] - pred['actual'])
                    recent_errors.append(error)
            
            if recent_errors:
                avg_error = np.mean(recent_errors)
                accuracy_factor = 1.0 / (1.0 + avg_error)
                base_confidence *= accuracy_factor
        
        return float(np.clip(base_confidence, 0.1, 0.95))
    
    def _estimate_tail_risk_probability(self, market_data: MarketData) -> float:
        """Estimate probability of tail risk events"""
        returns = market_data.returns[-252:] if len(market_data.returns) >= 252 else market_data.returns
        
        if len(returns) < 20:
            return 0.05  # Default 5%
        
        # Use empirical quantiles
        extreme_threshold = np.percentile(np.abs(returns), 95)
        extreme_events = np.sum(np.abs(returns) > extreme_threshold)
        tail_probability = extreme_events / len(returns)
        
        return float(np.clip(tail_probability, 0.01, 0.2))
    
    def _default_metrics(self) -> ProbabilisticMetrics:
        """Return default metrics when computation fails"""
        return ProbabilisticMetrics(
            var_95=5000.0,
            var_99=10000.0,
            expected_shortfall=15000.0,
            tail_risk_probability=0.05,
            uncertainty_score=0.5,
            regime_probabilities={'low_volatility': 0.4, 'medium_volatility': 0.4, 
                                'high_volatility': 0.15, 'crisis': 0.05},
            model_confidence=0.5,
            prediction_interval=(-10000.0, 10000.0),
            feature_importance={}
        )
    
    async def train_models(self, training_data: List[MarketData], target_variable: str = 'next_return'):
        """Train all ML models with historical data"""
        if len(training_data) < 100:
            raise ValueError("Insufficient training data. Need at least 100 samples.")
        
        logging.info(f"Training models with {len(training_data)} samples...")
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for i, data in enumerate(training_data[:-1]):
            features = self.ensemble_model.prepare_features(data)
            next_return = training_data[i + 1].returns[0] if len(training_data[i + 1].returns) > 0 else 0.0
            
            X_list.append(features[-1])  # Latest features
            y_list.append(next_return)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Train ensemble model
        combined_data = MarketData(
            prices=np.concatenate([d.prices for d in training_data]),
            returns=np.concatenate([d.returns for d in training_data]),
            volume=np.concatenate([d.volume for d in training_data]),
            volatility=np.concatenate([d.volatility for d in training_data]),
            features={},
            timestamp=datetime.now()
        )
        
        self.ensemble_model.fit(combined_data, y)
        
        # Train deep learning uncertainty model
        await self._train_uncertainty_net(X, y)
        
        # Train Bayesian regime detector
        all_returns = np.concatenate([d.returns for d in training_data])
        self.bayesian_detector.fit(all_returns[-1000:])  # Use recent data
        
        # Hyperparameter optimization
        best_params = self.automl_optimizer.optimize_xgboost(X, y)
        logging.info(f"Best XGBoost parameters: {best_params}")
        
        logging.info("✅ Model training completed successfully")
    
    async def _train_uncertainty_net(self, X: np.ndarray, y: np.ndarray):
        """Train deep learning uncertainty quantification network"""
        try:
            input_size = X.shape[1]
            self.uncertainty_net = UncertaintyQuantificationNet(input_size)
            
            # Prepare data
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.Adam(self.uncertainty_net.parameters(), lr=0.001)
            criterion = nn.GaussianNLLLoss()
            
            # Training loop
            self.uncertainty_net.train()
            for epoch in range(100):
                total_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    mean, var = self.uncertainty_net(batch_x)
                    loss = criterion(mean, batch_y, var)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    logging.info(f"Uncertainty net epoch {epoch}, loss: {total_loss:.6f}")
            
            self.uncertainty_net.eval()
            logging.info("✅ Uncertainty quantification network trained")
            
        except Exception as e:
            logging.error(f"Failed to train uncertainty network: {e}")
            self.uncertainty_net = None
    
    async def run_performance_benchmark(self, problem_size: int = 10000) -> Dict[str, float]:
        """Run comprehensive performance benchmark"""
        logging.info(f"🚀 Running performance benchmark with problem size {problem_size}")
        
        results = {}
        
        # Python ML benchmarks
        start_time = datetime.now()
        
        # Generate synthetic data
        np.random.seed(42)
        synthetic_data = self._generate_synthetic_market_data(problem_size)
        
        # Benchmark ensemble prediction
        if self.ensemble_model.is_fitted:
            ensemble_start = datetime.now()
            try:
                pred, std, _ = self.ensemble_model.predict_with_uncertainty(synthetic_data)
                ensemble_duration = (datetime.now() - ensemble_start).total_seconds() * 1000
                results['ensemble_prediction_ms'] = ensemble_duration
            except Exception as e:
                logging.error(f"Ensemble benchmark failed: {e}")
                results['ensemble_prediction_ms'] = -1
        
        # Benchmark deep learning uncertainty
        if self.uncertainty_net is not None:
            dl_start = datetime.now()
            try:
                uncertainty = await self._deep_learning_uncertainty(synthetic_data)
                dl_duration = (datetime.now() - dl_start).total_seconds() * 1000
                results['deep_learning_uncertainty_ms'] = dl_duration
            except Exception as e:
                logging.error(f"Deep learning benchmark failed: {e}")
                results['deep_learning_uncertainty_ms'] = -1
        
        # C++ backend benchmark
        if self.cpp_backend is not None:
            cpp_start = datetime.now()
            try:
                var_95, var_99, es = await self._compute_risk_metrics_cpp(synthetic_data)
                cpp_duration = (datetime.now() - cpp_start).total_seconds() * 1000
                results['cpp_backend_ms'] = cpp_duration
                results['cpp_var_95'] = var_95
                results['cpp_var_99'] = var_99
            except Exception as e:
                logging.error(f"C++ benchmark failed: {e}")
                results['cpp_backend_ms'] = -1
        
        total_duration = (datetime.now() - start_time).total_seconds() * 1000
        results['total_benchmark_ms'] = total_duration
        
        logging.info(f"✅ Performance benchmark completed: {results}")
        return results
    
    def _generate_synthetic_market_data(self, size: int) -> MarketData:
        """Generate synthetic market data for testing"""
        np.random.seed(42)
        
        # Generate price series with realistic properties
        returns = np.random.normal(0.0005, 0.02, size)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volume = np.random.lognormal(10, 0.5, size)
        
        # Calculate volatility
        volatility = pd.Series(returns).rolling(20).std().fillna(0.02).values
        
        return MarketData(
            prices=prices,
            returns=returns,
            volume=volume,
            volatility=volatility,
            features={'synthetic': np.random.normal(0, 1, size)},
            timestamp=datetime.now()
        )
    
    def export_model_performance(self) -> pd.DataFrame:
        """Export model performance metrics to DataFrame"""
        performance_data = []
        
        for i, pred in enumerate(self.predictions_history):
            if pred['actual'] is not None:
                error = abs(pred['prediction'] - pred['actual'])
                performance_data.append({
                    'timestamp': pred['timestamp'],
                    'prediction': pred['prediction'],
                    'actual': pred['actual'],
                    'uncertainty': pred['uncertainty'],
                    'error': error,
                    'within_confidence': error <= 2 * pred['uncertainty']
                })
        
        return pd.DataFrame(performance_data)
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
CWTS PROBABILISTIC ML ORCHESTRATOR - ANALYSIS REPORT
==================================================
Generated: {datetime.now().isoformat()}

SYSTEM STATUS:
- Ensemble Model: {'✅ Trained' if self.ensemble_model.is_fitted else '❌ Not Trained'}
- Deep Learning Net: {'✅ Available' if self.uncertainty_net is not None else '❌ Not Available'}
- Bayesian Detector: {'✅ Fitted' if self.bayesian_detector.trace is not None else '❌ Not Fitted'}
- C++ Backend: {'✅ Available' if self.cpp_backend is not None else '❌ Not Available'}

DATA STATISTICS:
- Historical Data Points: {len(self.historical_data)}
- Prediction History: {len(self.predictions_history)}
- Training Data Coverage: {len(self.historical_data) * 0.016:.1f} hours (assuming 1-minute intervals)

PERFORMANCE METRICS:
"""
        
        if self.predictions_history:
            recent_predictions = self.predictions_history[-100:]  # Last 100 predictions
            valid_predictions = [p for p in recent_predictions if p['actual'] is not None]
            
            if valid_predictions:
                errors = [abs(p['prediction'] - p['actual']) for p in valid_predictions]
                uncertainties = [p['uncertainty'] for p in valid_predictions]
                
                report += f"""- Mean Absolute Error: {np.mean(errors):.6f}
- Mean Uncertainty: {np.mean(uncertainties):.6f}
- Prediction Accuracy: {sum(e <= 2*u for e, u in zip(errors, uncertainties)) / len(errors) * 100:.1f}%
"""
            else:
                report += "- No validated predictions available yet\n"
        
        # Model ensemble information
        if self.ensemble_model.is_fitted:
            report += f"""
ENSEMBLE MODELS:
- Number of Models: {len(self.ensemble_model.models)}
- Feature Importance Available: {len(self.ensemble_model.feature_importance) > 0}
"""
        
        report += f"""
RECOMMENDATIONS:
{'- ✅ System is fully operational' if all([
    self.ensemble_model.is_fitted,
    self.uncertainty_net is not None,
    self.bayesian_detector.trace is not None
]) else '- ⚠️  Complete model training for optimal performance'}
- Continue collecting real-time data for model improvement
- Monitor prediction accuracy and retrain if performance degrades
- Consider ensemble model expansion for better coverage

END REPORT
==========
"""
        
        return report

# Example usage and testing
async def main():
    """Example usage of the Probabilistic ML Orchestrator"""
    
    # Initialize orchestrator
    orchestrator = ProbabilisticMLOrchestrator()
    
    # Generate synthetic training data
    training_data = []
    np.random.seed(42)
    
    for i in range(200):  # 200 data points
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        volume = np.random.lognormal(10, 0.5, 100)
        volatility = np.random.exponential(0.02, 100)
        
        market_data = MarketData(
            prices=prices,
            returns=returns,
            volume=volume,
            volatility=volatility,
            features={'momentum': np.random.normal(0, 1, 100)},
            timestamp=datetime.now() - timedelta(hours=200-i)
        )
        training_data.append(market_data)
    
    # Train models
    await orchestrator.train_models(training_data)
    
    # Process real-time data
    current_data = training_data[-1]  # Use last data point as current
    metrics = await orchestrator.process_real_time_data(current_data)
    
    print("🎯 Probabilistic Metrics:")
    print(f"VaR 95%: ${metrics.var_95:,.2f}")
    print(f"VaR 99%: ${metrics.var_99:,.2f}")
    print(f"Expected Shortfall: ${metrics.expected_shortfall:,.2f}")
    print(f"Uncertainty Score: {metrics.uncertainty_score:.3f}")
    print(f"Model Confidence: {metrics.model_confidence:.3f}")
    
    # Run performance benchmark
    benchmark_results = await orchestrator.run_performance_benchmark(10000)
    print("\n⚡ Performance Benchmark:")
    for key, value in benchmark_results.items():
        print(f"{key}: {value:.2f}")
    
    # Generate analysis report
    report = orchestrator.generate_analysis_report()
    print("\n📊 Analysis Report:")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())