"""
Machine learning-based optimizer for trading system parameters.

Uses various ML models to learn the objective function landscape
and guide the optimization process.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import time
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

from .parameter_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class MLOptimizer(BaseOptimizer):
    """Machine learning-based optimization."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]],
                 model_type: str = 'random_forest',
                 acquisition_strategy: str = 'uncertainty',
                 exploration_rate: float = 0.2,
                 constraints: Optional[List[Callable]] = None,
                 adaptive_convergence: bool = False,
                 parallel_evaluations: int = 1):
        """
        Initialize ML optimizer.
        
        Args:
            search_space: Parameter definitions
            model_type: ML model type ('random_forest', 'gradient_boosting', 
                       'gaussian_process', 'neural_network')
            acquisition_strategy: How to select next points ('uncertainty', 
                                'expected_improvement', 'thompson')
            exploration_rate: Rate of exploration vs exploitation
            constraints: List of constraint functions
            adaptive_convergence: Use adaptive convergence criteria
            parallel_evaluations: Number of parallel evaluations
        """
        super().__init__(search_space, constraints)
        self.model_type = model_type
        self.acquisition_strategy = acquisition_strategy
        self.exploration_rate = exploration_rate
        self.adaptive_convergence = adaptive_convergence
        self.parallel_evaluations = parallel_evaluations
        
        # Initialize model
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Storage
        self.X_observed = []
        self.y_observed = []
        
        # Parameter encoding
        self._setup_encoding()
    
    def _create_model(self):
        """Create the ML model based on type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'gaussian_process':
            return GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
        elif self.model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _setup_encoding(self):
        """Setup parameter encoding for ML."""
        self.param_names = list(self.search_space.keys())
        self.feature_indices = {}
        self.categorical_encodings = {}
        
        idx = 0
        for param, config in self.search_space.items():
            if config['type'] in ['float', 'int']:
                self.feature_indices[param] = idx
                idx += 1
            elif config['type'] == 'categorical':
                # One-hot encoding for categorical
                values = config['values']
                self.categorical_encodings[param] = {
                    val: i for i, val in enumerate(values)
                }
                self.feature_indices[param] = list(range(idx, idx + len(values)))
                idx += len(values)
        
        self.n_features = idx
    
    def _params_to_features(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to feature vector."""
        features = np.zeros(self.n_features)
        
        for param, value in params.items():
            config = self.search_space[param]
            
            if config['type'] in ['float', 'int']:
                # Normalize to [0, 1]
                normalized = (value - config['min']) / (config['max'] - config['min'])
                features[self.feature_indices[param]] = normalized
            elif config['type'] == 'categorical':
                # One-hot encoding
                encoding_idx = self.categorical_encodings[param][value]
                feature_idx = self.feature_indices[param][encoding_idx]
                features[feature_idx] = 1.0
        
        return features
    
    def _features_to_params(self, features: np.ndarray) -> Dict[str, Any]:
        """Convert feature vector to parameters."""
        params = {}
        
        for param, config in self.search_space.items():
            if config['type'] == 'float':
                idx = self.feature_indices[param]
                normalized = features[idx]
                value = normalized * (config['max'] - config['min']) + config['min']
                params[param] = float(value)
            elif config['type'] == 'int':
                idx = self.feature_indices[param]
                normalized = features[idx]
                value = normalized * (config['max'] - config['min']) + config['min']
                params[param] = int(round(value))
            elif config['type'] == 'categorical':
                # Find which one-hot is active
                indices = self.feature_indices[param]
                one_hot = features[indices]
                
                if np.sum(one_hot) > 0:
                    active_idx = np.argmax(one_hot)
                else:
                    active_idx = np.random.randint(len(indices))
                
                # Find corresponding value
                for val, idx in self.categorical_encodings[param].items():
                    if idx == active_idx:
                        params[param] = val
                        break
        
        return params
    
    def _predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates."""
        if self.model_type == 'random_forest':
            # Use tree predictions for uncertainty
            predictions = []
            for tree in self.model.estimators_:
                predictions.append(tree.predict(X))
            predictions = np.array(predictions)
            
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            return mean, std
            
        elif self.model_type == 'gaussian_process':
            return self.model.predict(X, return_std=True)
            
        elif self.model_type == 'gradient_boosting':
            # Use staged predictions for uncertainty
            staged_preds = []
            for pred in self.model.staged_predict(X):
                staged_preds.append(pred)
            
            if len(staged_preds) > 10:
                # Use last 10 stages
                staged_preds = staged_preds[-10:]
            
            staged_preds = np.array(staged_preds)
            mean = staged_preds[-1]
            std = np.std(staged_preds, axis=0)
            return mean, std
            
        else:
            # No uncertainty for neural networks, use dropout approximation
            mean = self.model.predict(X)
            # Approximate uncertainty
            std = np.ones_like(mean) * 0.1
            return mean, std
    
    def _select_next_points(self, n_points: int = 1) -> List[Dict[str, Any]]:
        """Select next points to evaluate."""
        if not self.is_fitted:
            # Random sampling if no model
            points = []
            for _ in range(n_points):
                for _ in range(1000):
                    params = self.sample_params()
                    if self.validate_params(params):
                        points.append(params)
                        break
            return points
        
        # Generate candidate points
        n_candidates = max(1000, 100 * n_points)
        candidates = []
        candidate_features = []
        
        for _ in range(n_candidates):
            params = self.sample_params()
            if self.validate_params(params):
                candidates.append(params)
                candidate_features.append(self._params_to_features(params))
        
        if not candidates:
            return []
        
        candidate_features = np.array(candidate_features)
        
        # Scale features
        candidate_features_scaled = self.scaler.transform(candidate_features)
        
        # Predict values and uncertainties
        predictions, uncertainties = self._predict_with_uncertainty(candidate_features_scaled)
        
        # Select based on acquisition strategy
        if self.acquisition_strategy == 'uncertainty':
            # Select points with highest uncertainty
            if np.random.rand() < self.exploration_rate:
                # Explore: highest uncertainty
                indices = np.argsort(uncertainties)[-n_points:]
            else:
                # Exploit: best predicted value
                indices = np.argsort(predictions)[:n_points]
                
        elif self.acquisition_strategy == 'expected_improvement':
            # Calculate expected improvement
            best_observed = np.min(self.y_observed)
            improvement = best_observed - predictions
            z = improvement / (uncertainties + 1e-9)
            ei = improvement * norm.cdf(z) + uncertainties * norm.pdf(z)
            indices = np.argsort(ei)[-n_points:]
            
        elif self.acquisition_strategy == 'thompson':
            # Thompson sampling
            samples = predictions + uncertainties * np.random.randn(len(predictions))
            indices = np.argsort(samples)[:n_points]
        else:
            # Default to uncertainty
            indices = np.argsort(uncertainties)[-n_points:]
        
        return [candidates[i] for i in indices]
    
    def _fit_model(self):
        """Fit the ML model to observed data."""
        if len(self.X_observed) < 5:
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        try:
            if self.model_type == 'neural_network' and len(X) < 50:
                # Need more data for neural networks
                return
            
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # Log model performance
            if len(X) >= 10:
                try:
                    scores = cross_val_score(
                        self.model, X_scaled, y, 
                        cv=min(5, len(X)), 
                        scoring='neg_mean_squared_error'
                    )
                    logger.debug(f"Model CV score: {-np.mean(scores):.4f}")
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            self.is_fitted = False
    
    def optimize(self, objective_function: Callable,
                max_evaluations: int = 100,
                target_score: Optional[float] = None,
                patience: Optional[int] = None,
                min_improvement: float = 1e-6,
                batch_size: Optional[int] = None,
                parallel: bool = None,
                initial_tolerance: Optional[float] = None,
                final_tolerance: Optional[float] = None) -> OptimizationResult:
        """
        Run ML-based optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum evaluations
            target_score: Stop if score below target
            patience: Stop after no improvement
            min_improvement: Minimum improvement
            batch_size: Batch size for parallel evaluation
            parallel: Use parallel evaluation
            initial_tolerance: Initial convergence tolerance
            final_tolerance: Final convergence tolerance
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        history = []
        best_params = None
        best_score = float('inf')
        best_iteration = 0
        converged = False
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.parallel_evaluations
        if parallel is None:
            parallel = self.parallel_evaluations > 1
        
        # Initial random sampling (at least 2 * n_features)
        initial_samples = max(10, 2 * self.n_features)
        
        for i in range(min(initial_samples, max_evaluations)):
            params = None
            for _ in range(1000):
                candidate = self.sample_params()
                if self.validate_params(candidate):
                    params = candidate
                    break
            
            if params is None:
                continue
            
            # Evaluate
            score = objective_function(params)
            
            # Store
            features = self._params_to_features(params)
            self.X_observed.append(features)
            self.y_observed.append(score)
            
            # Update best
            if score < best_score - min_improvement:
                best_score = score
                best_params = params.copy()
                best_iteration = i
            
            history.append({
                'params': params.copy(),
                'score': score,
                'iteration': i,
                'phase': 'initial'
            })
            
            # Early stopping
            if target_score and best_score <= target_score:
                converged = True
                break
        
        # ML-guided optimization
        for i in range(len(history), max_evaluations, batch_size):
            # Fit model
            self._fit_model()
            
            # Select next points
            n_points = min(batch_size, max_evaluations - i)
            next_points = self._select_next_points(n_points)
            
            if not next_points:
                logger.warning("Failed to select next points")
                break
            
            # Evaluate points
            if parallel and len(next_points) > 1:
                # Parallel evaluation
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {
                        executor.submit(objective_function, params): params
                        for params in next_points
                    }
                    
                    for future in as_completed(futures):
                        params = futures[future]
                        try:
                            score = future.result()
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")
                            score = float('inf')
                        
                        # Store
                        features = self._params_to_features(params)
                        self.X_observed.append(features)
                        self.y_observed.append(score)
                        
                        # Update best
                        if score < best_score - min_improvement:
                            best_score = score
                            best_params = params.copy()
                            best_iteration = len(history)
                        
                        history.append({
                            'params': params.copy(),
                            'score': score,
                            'iteration': len(history),
                            'phase': 'ml_guided'
                        })
            else:
                # Sequential evaluation
                for params in next_points:
                    try:
                        score = objective_function(params)
                    except Exception as e:
                        logger.error(f"Evaluation failed: {e}")
                        score = float('inf')
                    
                    # Store
                    features = self._params_to_features(params)
                    self.X_observed.append(features)
                    self.y_observed.append(score)
                    
                    # Update best
                    if score < best_score - min_improvement:
                        best_score = score
                        best_params = params.copy()
                        best_iteration = len(history)
                    
                    history.append({
                        'params': params.copy(),
                        'score': score,
                        'iteration': len(history),
                        'phase': 'ml_guided'
                    })
            
            # Check convergence
            if target_score and best_score <= target_score:
                converged = True
                break
            
            if patience and len(history) - best_iteration >= patience:
                converged = True
                break
            
            # Adaptive convergence
            if self.adaptive_convergence and initial_tolerance and final_tolerance:
                progress = len(history) / max_evaluations
                current_tolerance = initial_tolerance * (1 - progress) + final_tolerance * progress
                
                if len(history) >= 10:
                    recent_scores = [h['score'] for h in history[-10:]]
                    if max(recent_scores) - min(recent_scores) < current_tolerance:
                        converged = True
                        break
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        
        metrics = {
            'time_elapsed': elapsed_time,
            'model_type': self.model_type,
            'acquisition_strategy': self.acquisition_strategy,
            'model_fitted': self.is_fitted,
            'evaluations_per_second': len(history) / elapsed_time
        }
        
        if self.adaptive_convergence and initial_tolerance:
            metrics['tolerance_history'] = [
                initial_tolerance * (1 - i/len(history)) + 
                (final_tolerance or 0) * i/len(history)
                for i in range(len(history))
            ]
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            converged=converged,
            iterations=len(history),
            convergence_iteration=best_iteration if converged else None,
            metrics=metrics,
            early_stopped=target_score and best_score <= target_score
        )