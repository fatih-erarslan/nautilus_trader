import numpy as np
import pandas as pd
import logging
import time
import os
import joblib
from typing import Dict, List, Optional, Tuple, Callable, Any, Union, Literal
from dataclasses import dataclass
from functools import partial
from enum import Enum

# --- JAX Ecosystem Imports ---
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from jax import grad, jit, vmap, lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # Fallback to NumPy

# Configure logging
logger = logging.getLogger("advanced_ml.ceflann_elm")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ExpansionType(Enum):
    """Types of functional expansion supported by CEFLANN-ELM"""
    TRIGONOMETRIC = "trigonometric"  # sin/cos expansion
    POLYNOMIAL = "polynomial"  # polynomial expansion
    CHEBYSHEV = "chebyshev"  # Chebyshev polynomials
    HERMITE = "hermite"  # Hermite polynomials


class CEFLANN_ELM:
    """
    Computationally Efficient Functional Link Artificial Neural Network (CEFLANN)
    using Extreme Learning Machine (ELM) principles for training the output layer.
    
    Based on concepts described in:
    Dash, R., & Dash, P. K. (2016). A hybrid stock trading framework integrating
    technical analysis with machine learning techniques. The Journal of Finance
    and Data Science, 2(1), 42-57.
    
    This implementation uses JAX for acceleration when available.
    
    Features:
    - Input expansion using various non-linear functions (trigonometric, polynomial, etc.)
    - Single-pass analytical training using Moore-Penrose pseudoinverse
    - Regularized solution (ridge regression) for better generalization
    - JAX acceleration for large datasets
    - Support for batched processing and incremental learning
    - Multiple output capabilities
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 1,
                 expansion_type: Union[str, ExpansionType] = ExpansionType.TRIGONOMETRIC,
                 expansion_order: int = 5,
                 per_feature_expansion: bool = True,
                 activation_scale: float = 1.0,
                 C: float = 1e-3,
                 use_jax: bool = True,
                 seed: int = 42):
        """
        Initialize CEFLANN-ELM model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions
            expansion_type: Type of functional expansion to use
                ("trigonometric", "polynomial", "chebyshev", "hermite")
            expansion_order: Order of expansion (number of terms)
            per_feature_expansion: Whether to apply expansion to each feature separately
                or to weighted combinations of features
            activation_scale: Scaling factor for activation functions
            C: Regularization parameter (inverse of lambda, larger C = less regularization)
            use_jax: Whether to use JAX acceleration if available
            seed: Random seed for reproducibility
        """
        if expansion_order <= 0:
            raise ValueError("Expansion order must be positive")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Handle expansion type
        if isinstance(expansion_type, str):
            try:
                self.expansion_type = ExpansionType(expansion_type.lower())
            except ValueError:
                logger.warning(f"Unknown expansion type '{expansion_type}', defaulting to TRIGONOMETRIC")
                self.expansion_type = ExpansionType.TRIGONOMETRIC
        else:
            self.expansion_type = expansion_type
            
        self.expansion_order = expansion_order
        self.per_feature_expansion = per_feature_expansion
        self.activation_scale = activation_scale
        self.regularization_param = C
        self.seed = seed
        self.use_jax = use_jax and JAX_AVAILABLE
        
        # Set random seed
        np.random.seed(seed)
        if self.use_jax:
            self.key = jrandom.PRNGKey(seed)
        
        # Initialize expansion parameters and calculate expanded dimension
        self.expanded_dim = self._initialize_expansion()
        
        # Placeholders for trained model
        self.output_weights = None
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self._trained = False
        
        # Feature names for explainability
        self.feature_names = None
        self.expanded_feature_names = None
        
        logger.info(
            f"CEFLANN-ELM Initialized: InputDim={input_dim}, OutputDim={output_dim}, "
            f"ExpansionType={self.expansion_type.value}, ExpansionOrder={expansion_order}, "
            f"ExpandedDim={self.expanded_dim}, PerFeatureExpansion={per_feature_expansion}, "
            f"UseJAX={self.use_jax}"
        )
        
        # Compile JAX functions if available
        if self.use_jax:
            self._compile_jax_functions()
    
    def _initialize_expansion(self) -> int:
        """
        Initialize expansion parameters and calculate expanded dimension.
        
        Returns:
            int: Expanded dimension size
        """
        # Calculate expanded dimension based on expansion type and method
        if self.per_feature_expansion:
            # Apply expansion to each feature separately
            if self.expansion_type == ExpansionType.TRIGONOMETRIC:
                # Each feature gets 1 + 2*order terms (original + sin/cos pairs)
                self.expanded_dim = self.input_dim * (1 + 2 * self.expansion_order)
            elif self.expansion_type == ExpansionType.POLYNOMIAL:
                # Each feature gets 1 + order terms (original + powers)
                self.expanded_dim = self.input_dim * (1 + self.expansion_order)
            elif self.expansion_type in (ExpansionType.CHEBYSHEV, ExpansionType.HERMITE):
                # Each feature gets 1 + order terms (original + polynomials)
                self.expanded_dim = self.input_dim * (1 + self.expansion_order)
            else:
                # Default dimension
                self.expanded_dim = self.input_dim * (1 + 2 * self.expansion_order)
        else:
            # Apply expansion to weighted combinations of features
            if self.expansion_type == ExpansionType.TRIGONOMETRIC:
                # input_dim original features + 2*order expansion terms
                self.expanded_dim = self.input_dim + 2 * self.expansion_order
            elif self.expansion_type == ExpansionType.POLYNOMIAL:
                # input_dim original features + order expansion terms
                self.expanded_dim = self.input_dim + self.expansion_order
            elif self.expansion_type in (ExpansionType.CHEBYSHEV, ExpansionType.HERMITE):
                # input_dim original features + order expansion terms
                self.expanded_dim = self.input_dim + self.expansion_order
            else:
                # Default dimension
                self.expanded_dim = self.input_dim + 2 * self.expansion_order
        
        # Initialize random weights for functional expansion
        if self.use_jax:
            # Use JAX random generator
            key, subkey = jrandom.split(self.key)
            self.key = key
            
            if self.per_feature_expansion:
                # One weight per feature per order
                self._expansion_weights = jrandom.normal(
                    subkey, 
                    (self.expansion_order, self.input_dim)
                ) * 0.5
            else:
                # One weight vector per order for all features
                self._expansion_weights = jrandom.normal(
                    subkey, 
                    (self.expansion_order, self.input_dim)
                ) * 0.5
            
            # Convert to numpy for compatibility
            self._expansion_weights = np.array(self._expansion_weights)
        else:
            # Use NumPy random generator
            if self.per_feature_expansion:
                self._expansion_weights = np.random.randn(
                    self.expansion_order, self.input_dim
                ) * 0.5
            else:
                self._expansion_weights = np.random.randn(
                    self.expansion_order, self.input_dim
                ) * 0.5
        
        return self.expanded_dim
    
    def _compile_jax_functions(self):
        """Compile JAX functions for acceleration"""
        logger.info("Compiling JAX functions for accelerated computation...")
        
        # JIT-compile the functional expansion
        @jax.jit
        def _functional_expansion_jit(X):
            """JIT-compiled functional expansion"""
            n_samples = X.shape[0]
            
            # Initialize expanded feature matrix
            H = jnp.zeros((n_samples, self.expanded_dim))
            
            # Include original features
            H = H.at[:, :self.input_dim].set(X)
            
            # Apply expansion based on type and method
            if self.expansion_type == ExpansionType.TRIGONOMETRIC:
                if self.per_feature_expansion:
                    # Per-feature trigonometric expansion
                    current_col = self.input_dim
                    for i in range(self.input_dim):
                        for n in range(self.expansion_order):
                            w_ni = self._expansion_weights[n, i]
                            H = H.at[:, current_col].set(
                                jnp.sin(X[:, i] * w_ni * jnp.pi * self.activation_scale)
                            )
                            current_col += 1
                            H = H.at[:, current_col].set(
                                jnp.cos(X[:, i] * w_ni * jnp.pi * self.activation_scale)
                            )
                            current_col += 1
                else:
                    # Combined features trigonometric expansion
                    current_col = self.input_dim
                    for n in range(self.expansion_order):
                        w_n = self._expansion_weights[n, :]
                        weighted_sum = X @ w_n
                        H = H.at[:, current_col].set(
                            jnp.sin(weighted_sum * jnp.pi * self.activation_scale)
                        )
                        current_col += 1
                        H = H.at[:, current_col].set(
                            jnp.cos(weighted_sum * jnp.pi * self.activation_scale)
                        )
                        current_col += 1
            
            elif self.expansion_type == ExpansionType.POLYNOMIAL:
                if self.per_feature_expansion:
                    # Per-feature polynomial expansion
                    current_col = self.input_dim
                    for i in range(self.input_dim):
                        for n in range(1, self.expansion_order + 1):
                            H = H.at[:, current_col].set(
                                X[:, i] ** (n * self.activation_scale)
                            )
                            current_col += 1
                else:
                    # Combined features polynomial expansion
                    current_col = self.input_dim
                    for n in range(self.expansion_order):
                        w_n = self._expansion_weights[n, :]
                        weighted_sum = X @ w_n
                        H = H.at[:, current_col].set(
                            weighted_sum ** ((n+1) * self.activation_scale)
                        )
                        current_col += 1
            
            elif self.expansion_type == ExpansionType.CHEBYSHEV:
                # Chebyshev polynomials implementation
                if self.per_feature_expansion:
                    current_col = self.input_dim
                    for i in range(self.input_dim):
                        # T_0(x) = 1 (constant), usually skipped as redundant
                        # T_1(x) = x (already included in original features)
                        # Start from T_2(x) = 2x² - 1
                        t_n_minus_2 = jnp.ones(n_samples)  # T_0(x) = 1
                        t_n_minus_1 = X[:, i]  # T_1(x) = x
                        
                        for n in range(2, self.expansion_order + 2):
                            # Chebyshev recurrence: T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
                            t_n = 2 * X[:, i] * t_n_minus_1 - t_n_minus_2
                            H = H.at[:, current_col].set(t_n)
                            current_col += 1
                            
                            # Update for next iteration
                            t_n_minus_2 = t_n_minus_1
                            t_n_minus_1 = t_n
                else:
                    # Combined features Chebyshev expansion
                    current_col = self.input_dim
                    for n in range(self.expansion_order):
                        w_n = self._expansion_weights[n, :]
                        x = X @ w_n  # Weighted sum
                        
                        # Ensure x is in [-1, 1] for Chebyshev stability
                        x = jnp.clip(x, -1.0, 1.0)
                        
                        # Compute approximation of Chebyshev polynomial
                        if n == 0:
                            result = jnp.ones_like(x)  # T_0(x) = 1
                        elif n == 1:
                            result = x  # T_1(x) = x
                        else:
                            # Calculate higher order polynomials
                            t_n_minus_2 = jnp.ones_like(x)
                            t_n_minus_1 = x
                            
                            for k in range(2, n + 1):
                                t_n = 2 * x * t_n_minus_1 - t_n_minus_2
                                t_n_minus_2 = t_n_minus_1
                                t_n_minus_1 = t_n
                            
                            result = t_n_minus_1
                        
                        H = H.at[:, current_col].set(result)
                        current_col += 1
            
            elif self.expansion_type == ExpansionType.HERMITE:
                # Hermite polynomials implementation (physicist's version)
                if self.per_feature_expansion:
                    current_col = self.input_dim
                    for i in range(self.input_dim):
                        # He_0(x) = 1 (constant)
                        # He_1(x) = x (already included in original features)
                        # Start from He_2(x) = x² - 1
                        h_n_minus_2 = jnp.ones(n_samples)  # He_0(x) = 1
                        h_n_minus_1 = X[:, i]  # He_1(x) = x
                        
                        for n in range(2, self.expansion_order + 2):
                            # Hermite recurrence: He_n(x) = x*He_{n-1}(x) - (n-1)*He_{n-2}(x)
                            h_n = X[:, i] * h_n_minus_1 - (n-1) * h_n_minus_2
                            H = H.at[:, current_col].set(h_n)
                            current_col += 1
                            
                            # Update for next iteration
                            h_n_minus_2 = h_n_minus_1
                            h_n_minus_1 = h_n
                else:
                    # Combined features Hermite expansion
                    current_col = self.input_dim
                    for n in range(self.expansion_order):
                        w_n = self._expansion_weights[n, :]
                        x = X @ w_n  # Weighted sum
                        
                        # Compute approximation of Hermite polynomial
                        if n == 0:
                            result = jnp.ones_like(x)  # He_0(x) = 1
                        elif n == 1:
                            result = x  # He_1(x) = x
                        else:
                            # Calculate higher order polynomials
                            h_n_minus_2 = jnp.ones_like(x)
                            h_n_minus_1 = x
                            
                            for k in range(2, n + 1):
                                h_n = x * h_n_minus_1 - (k-1) * h_n_minus_2
                                h_n_minus_2 = h_n_minus_1
                                h_n_minus_1 = h_n
                            
                            result = h_n_minus_1
                        
                        H = H.at[:, current_col].set(result)
                        current_col += 1
            
            return H
        
        # Store the compiled function
        self._functional_expansion_jit = _functional_expansion_jit
        
        # Create vectorized version for batch processing
        self._functional_expansion_batched = vmap(
            lambda x: self._functional_expansion_jit(x.reshape(1, -1))[0],
            in_axes=0, out_axes=0
        )
        
        logger.info("JAX functions compiled")
    
    def _functional_expansion(self, X: np.ndarray) -> np.ndarray:
        """
        Apply functional expansion to input features.
        
        Args:
            X: Input features, shape (n_samples, input_dim)
        
        Returns:
            Expanded features, shape (n_samples, expanded_dim)
        """
        n_samples = X.shape[0]
        
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input X has incorrect dimension {X.shape[1]}, expected {self.input_dim}")
        
        # Use JAX implementation if available
        if self.use_jax:
            try:
                # Convert to JAX array
                X_jax = jnp.array(X)
                
                # Apply the JIT-compiled function
                H = self._functional_expansion_jit(X_jax)
                
                # Convert back to numpy
                return np.array(H)
            except Exception as e:
                logger.warning(f"JAX expansion failed: {e}. Falling back to NumPy.")
        
        # NumPy implementation (fallback)
        # Initialize expanded feature matrix
        H = np.zeros((n_samples, self.expanded_dim))
        
        # Include original features
        H[:, :self.input_dim] = X
        
        # Apply expansion based on type and method
        if self.expansion_type == ExpansionType.TRIGONOMETRIC:
            if self.per_feature_expansion:
                # Per-feature trigonometric expansion
                current_col = self.input_dim
                for i in range(self.input_dim):
                    for n in range(self.expansion_order):
                        w_ni = self._expansion_weights[n, i]
                        H[:, current_col] = np.sin(X[:, i] * w_ni * np.pi * self.activation_scale)
                        current_col += 1
                        H[:, current_col] = np.cos(X[:, i] * w_ni * np.pi * self.activation_scale)
                        current_col += 1
            else:
                # Combined features trigonometric expansion
                current_col = self.input_dim
                for n in range(self.expansion_order):
                    w_n = self._expansion_weights[n, :]
                    weighted_sum = X @ w_n
                    H[:, current_col] = np.sin(weighted_sum * np.pi * self.activation_scale)
                    current_col += 1
                    H[:, current_col] = np.cos(weighted_sum * np.pi * self.activation_scale)
                    current_col += 1
        
        elif self.expansion_type == ExpansionType.POLYNOMIAL:
            if self.per_feature_expansion:
                # Per-feature polynomial expansion
                current_col = self.input_dim
                for i in range(self.input_dim):
                    for n in range(1, self.expansion_order + 1):
                        H[:, current_col] = X[:, i] ** (n * self.activation_scale)
                        current_col += 1
            else:
                # Combined features polynomial expansion
                current_col = self.input_dim
                for n in range(self.expansion_order):
                    w_n = self._expansion_weights[n, :]
                    weighted_sum = X @ w_n
                    H[:, current_col] = weighted_sum ** ((n+1) * self.activation_scale)
                    current_col += 1
        
        elif self.expansion_type == ExpansionType.CHEBYSHEV:
            # Chebyshev polynomials implementation
            if self.per_feature_expansion:
                current_col = self.input_dim
                for i in range(self.input_dim):
                    # T_0(x) = 1 (constant), usually skipped as redundant
                    # T_1(x) = x (already included in original features)
                    # Start from T_2(x) = 2x² - 1
                    t_n_minus_2 = np.ones(n_samples)  # T_0(x) = 1
                    t_n_minus_1 = X[:, i]  # T_1(x) = x
                    
                    for n in range(2, self.expansion_order + 2):
                        # Chebyshev recurrence: T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
                        t_n = 2 * X[:, i] * t_n_minus_1 - t_n_minus_2
                        H[:, current_col] = t_n
                        current_col += 1
                        
                        # Update for next iteration
                        t_n_minus_2 = t_n_minus_1
                        t_n_minus_1 = t_n
            else:
                # Combined features Chebyshev expansion
                current_col = self.input_dim
                for n in range(self.expansion_order):
                    w_n = self._expansion_weights[n, :]
                    x = X @ w_n  # Weighted sum
                    
                    # Ensure x is in [-1, 1] for Chebyshev stability
                    x = np.clip(x, -1.0, 1.0)
                    
                    # Compute approximation of Chebyshev polynomial
                    if n == 0:
                        result = np.ones_like(x)  # T_0(x) = 1
                    elif n == 1:
                        result = x  # T_1(x) = x
                    else:
                        # Calculate higher order polynomials
                        t_n_minus_2 = np.ones_like(x)
                        t_n_minus_1 = x
                        
                        for k in range(2, n + 1):
                            t_n = 2 * x * t_n_minus_1 - t_n_minus_2
                            t_n_minus_2 = t_n_minus_1
                            t_n_minus_1 = t_n
                        
                        result = t_n_minus_1
                    
                    H[:, current_col] = result
                    current_col += 1
        
        elif self.expansion_type == ExpansionType.HERMITE:
            # Hermite polynomials implementation (physicist's version)
            if self.per_feature_expansion:
                current_col = self.input_dim
                for i in range(self.input_dim):
                    # He_0(x) = 1 (constant)
                    # He_1(x) = x (already included in original features)
                    # Start from He_2(x) = x² - 1
                    h_n_minus_2 = np.ones(n_samples)  # He_0(x) = 1
                    h_n_minus_1 = X[:, i]  # He_1(x) = x
                    
                    for n in range(2, self.expansion_order + 2):
                        # Hermite recurrence: He_n(x) = x*He_{n-1}(x) - (n-1)*He_{n-2}(x)
                        h_n = X[:, i] * h_n_minus_1 - (n-1) * h_n_minus_2
                        H[:, current_col] = h_n
                        current_col += 1
                        
                        # Update for next iteration
                        h_n_minus_2 = h_n_minus_1
                        h_n_minus_1 = h_n
            else:
                # Combined features Hermite expansion
                current_col = self.input_dim
                for n in range(self.expansion_order):
                    w_n = self._expansion_weights[n, :]
                    x = X @ w_n  # Weighted sum
                    
                    # Compute approximation of Hermite polynomial
                    if n == 0:
                        result = np.ones_like(x)  # He_0(x) = 1
                    elif n == 1:
                        result = x  # He_1(x) = x
                    else:
                        # Calculate higher order polynomials
                        h_n_minus_2 = np.ones_like(x)
                        h_n_minus_1 = x
                        
                        for k in range(2, n + 1):
                            h_n = x * h_n_minus_1 - (k-1) * h_n_minus_2
                            h_n_minus_2 = h_n_minus_1
                            h_n_minus_1 = h_n
                        
                        result = h_n_minus_1
                    
                    H[:, current_col] = result
                    current_col += 1
        
        return H
    
    def _normalize_inputs(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize input features for numerical stability.
        
        Args:
            X: Input data, shape (n_samples, input_dim)
            fit: If True, calculate and store normalization parameters
            
        Returns:
            Normalized input data
        """
        if fit:
            self.input_mean = np.mean(X, axis=0)
            self.input_std = np.std(X, axis=0)
            # Avoid division by zero
            self.input_std = np.where(self.input_std < 1e-10, 1.0, self.input_std)
        
        # Check if normalization parameters exist
        if self.input_mean is None or self.input_std is None:
            logger.warning("Normalization parameters not set. Using input data as is.")
            return X
        
        # Normalize
        X_norm = (X - self.input_mean) / self.input_std
        return X_norm
    
    def _normalize_outputs(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize output values for numerical stability.
        
        Args:
            y: Output data, shape (n_samples, output_dim)
            fit: If True, calculate and store normalization parameters
            
        Returns:
            Normalized output data
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if fit:
            self.output_mean = np.mean(y, axis=0)
            self.output_std = np.std(y, axis=0)
            # Avoid division by zero
            self.output_std = np.where(self.output_std < 1e-10, 1.0, self.output_std)
        
        # Check if normalization parameters exist
        if self.output_mean is None or self.output_std is None:
            logger.warning("Output normalization parameters not set. Using output data as is.")
            return y
        
        # Normalize
        y_norm = (y - self.output_mean) / self.output_std
        return y_norm
    
    def _denormalize_outputs(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            y_norm: Normalized predictions
            
        Returns:
            Denormalized predictions
        """
        # Ensure y_norm is 2D
        if y_norm.ndim == 1:
            y_norm = y_norm.reshape(-1, 1)
        
        # Check if normalization parameters exist
        if self.output_mean is None or self.output_std is None:
            logger.warning("Output normalization parameters not set. Returning predictions as is.")
            return y_norm
        
        # Denormalize
        y = y_norm * self.output_std + self.output_mean
        
        # Return with original shape
        if self.output_dim == 1:
            return y.flatten()
        return y
    
    def _create_expanded_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        Create names for expanded features for explainability.
        
        Args:
            feature_names: Names of original input features
            
        Returns:
            Names of expanded features
        """
        if len(feature_names) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} feature names, got {len(feature_names)}")
        
        # Start with original feature names
        expanded_names = feature_names.copy()
        
        # Add expanded feature names based on expansion type
        if self.expansion_type == ExpansionType.TRIGONOMETRIC:
            if self.per_feature_expansion:
                # Per-feature trigonometric expansion
                for i, name in enumerate(feature_names):
                    for n in range(self.expansion_order):
                        expanded_names.append(f"sin({n+1}π·{name})")
                        expanded_names.append(f"cos({n+1}π·{name})")
            else:
                # Combined features trigonometric expansion
                for n in range(self.expansion_order):
                    weight_str = '+'.join([f"{w:.2f}·{name}" for w, name in 
                                         zip(self._expansion_weights[n], feature_names)])
                    expanded_names.append(f"sin(π·({weight_str}))")
                    expanded_names.append(f"cos(π·({weight_str}))")
        
        elif self.expansion_type == ExpansionType.POLYNOMIAL:
            if self.per_feature_expansion:
                # Per-feature polynomial expansion
                for i, name in enumerate(feature_names):
                    for n in range(1, self.expansion_order + 1):
                        expanded_names.append(f"{name}^{n}")
            else:
                # Combined features polynomial expansion
                for n in range(self.expansion_order):
                    weight_str = '+'.join([f"{w:.2f}·{name}" for w, name in 
                                         zip(self._expansion_weights[n], feature_names)])
                    expanded_names.append(f"({weight_str})^{n+1}")
        
        elif self.expansion_type == ExpansionType.CHEBYSHEV:
            if self.per_feature_expansion:
                # Per-feature Chebyshev expansion
                for i, name in enumerate(feature_names):
                    for n in range(2, self.expansion_order + 2):  # T_0 and T_1 skipped
                        expanded_names.append(f"T_{n}({name})")
            else:
                # Combined features Chebyshev expansion
                for n in range(self.expansion_order):
                    weight_str = '+'.join([f"{w:.2f}·{name}" for w, name in 
                                         zip(self._expansion_weights[n], feature_names)])
                    expanded_names.append(f"T_{n}({weight_str})")
        
        elif self.expansion_type == ExpansionType.HERMITE:
            if self.per_feature_expansion:
                # Per-feature Hermite expansion
                for i, name in enumerate(feature_names):
                    for n in range(2, self.expansion_order + 2):  # He_0 and He_1 skipped
                        expanded_names.append(f"He_{n}({name})")
            else:
                # Combined features Hermite expansion
                for n in range(self.expansion_order):
                    weight_str = '+'.join([f"{w:.2f}·{name}" for w, name in 
                                         zip(self._expansion_weights[n], feature_names)])
                    expanded_names.append(f"He_{n}({weight_str})")
        
        return expanded_names
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            normalize: bool = True,
            batch_size: Optional[int] = None,
            feature_names: Optional[List[str]] = None):
        """
        Train the CEFLANN-ELM model.
        
        Args:
            X_train: Training input features, shape (n_samples, input_dim)
            y_train: Training target values, shape (n_samples, [output_dim])
            normalize: Whether to normalize inputs and outputs
            batch_size: Batch size for large datasets (if None, process all at once)
            feature_names: Names of input features for explainability
        """
        start_time = time.time()
        n_samples = X_train.shape[0]
        
        logger.info(f"Fitting CEFLANN-ELM on {n_samples} samples...")
        
        # Ensure input has correct shape
        if X_train.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {X_train.shape[1]}")
        
        # Ensure y is 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Verify output dimension
        if y_train.shape[1] != self.output_dim:
            logger.warning(
                f"Output dimension mismatch: got {y_train.shape[1]}, expected {self.output_dim}. "
                f"Adjusting model to match data."
            )
            self.output_dim = y_train.shape[1]
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
            try:
                self.expanded_feature_names = self._create_expanded_feature_names(feature_names)
            except Exception as e:
                logger.warning(f"Failed to create expanded feature names: {e}")
        
        # Normalize inputs if requested
        if normalize:
            X_norm = self._normalize_inputs(X_train, fit=True)
            y_norm = self._normalize_outputs(y_train, fit=True)
        else:
            X_norm = X_train
            y_norm = y_train
        
        # Handle large datasets with batching if requested
        if batch_size is not None and n_samples > batch_size:
            logger.info(f"Using batched processing with batch size {batch_size}")
            self._fit_batched(X_norm, y_norm, batch_size)
        else:
            logger.debug("Applying functional expansion...")
            # Apply functional expansion to inputs
            H = self._functional_expansion(X_norm)
            
            logger.debug(f"Calculating output weights...")
            # Calculate output weights using regularized least squares solution
            if self.use_jax:
                self._fit_jax(H, y_norm)
            else:
                self._fit_numpy(H, y_norm)
        
        self._trained = True
        end_time = time.time()
        logger.info(f"CEFLANN-ELM training completed in {end_time - start_time:.3f} seconds")
    
    def _fit_numpy(self, H: np.ndarray, y_train: np.ndarray):
        """
        Calculate output weights using NumPy.
        
        Args:
            H: Expanded feature matrix
            y_train: Training target values
        """
        H_T = H.T
        
        if self.regularization_param > 0:
            # Ridge regression: Beta = (H^T * H + lambda * I)^-1 * H^T * Y
            n_expanded_features = H.shape[1]
            identity = np.identity(n_expanded_features)
            lambda_reg = 1.0 / self.regularization_param
            
            try:
                # Calculate (H^T * H + lambda * I)^-1 * H^T * Y
                term1 = H_T @ H + lambda_reg * identity
                term1_inv = np.linalg.inv(term1)
                self.output_weights = term1_inv @ H_T @ y_train
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix encountered during Ridge Regression. "
                              "Attempting pseudoinverse.")
                # Fallback to pseudoinverse
                self.output_weights = np.linalg.pinv(H) @ y_train
        else:
            # Standard pseudoinverse
            self.output_weights = np.linalg.pinv(H) @ y_train
    
    def _fit_jax(self, H: np.ndarray, y_train: np.ndarray):
        """
        Calculate output weights using JAX for acceleration.
        
        Args:
            H: Expanded feature matrix
            y_train: Training target values
        """
        # Convert to JAX arrays
        H_jax = jnp.array(H)
        y_jax = jnp.array(y_train)
        
        H_T = H_jax.T
        
        if self.regularization_param > 0:
            # Ridge regression with JAX
            n_expanded_features = H_jax.shape[1]
            identity = jnp.identity(n_expanded_features)
            lambda_reg = 1.0 / self.regularization_param
            
            try:
                # Use JAX's solver for better numerical stability
                term1 = H_T @ H_jax + lambda_reg * identity
                self.output_weights = jax.scipy.linalg.solve(term1, H_T @ y_jax)
            except Exception as e:
                logger.warning(f"Error with JAX solver: {e}. Falling back to pseudoinverse.")
                self.output_weights = jnp.linalg.pinv(H_jax) @ y_jax
        else:
            # Standard pseudoinverse with JAX
            self.output_weights = jnp.linalg.pinv(H_jax) @ y_jax
        
        # Convert back to numpy
        self.output_weights = np.array(self.output_weights)
    
    def _fit_batched(self, X_norm: np.ndarray, y_norm: np.ndarray, batch_size: int):
        """
        Implements online sequential ELM for large datasets using batched processing.
        Based on Liang et al. (2006) "Efficient extreme learning machine."
        
        Args:
            X_norm: Normalized input features
            y_norm: Normalized target values
            batch_size: Size of each batch
        """
        n_samples = X_norm.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        logger.info(f"Fitting in batches: {n_batches} batches of size {batch_size}")
        
        # Initial batch for initialization
        X_batch = X_norm[:batch_size]
        y_batch = y_norm[:batch_size]
        
        # Apply functional expansion to first batch
        H_batch = self._functional_expansion(X_batch)
        
        # Initialize with first batch
        H_T = H_batch.T
        
        # Initialize P = (H^T * H + lambda * I)^-1
        if self.use_jax:
            H_batch_jax = jnp.array(H_batch)
            H_T_jax = jnp.array(H_T)
            identity = jnp.identity(H_batch_jax.shape[1])
            lambda_reg = 1.0 / self.regularization_param if self.regularization_param > 0 else 0.0
            P = jnp.linalg.inv(H_T_jax @ H_batch_jax + lambda_reg * identity)
            
            # Initial output weights
            beta = P @ H_T_jax @ jnp.array(y_batch)
            
            # Convert back to numpy for incremental updates
            P = np.array(P)
            beta = np.array(beta)
        else:
            identity = np.identity(H_batch.shape[1])
            lambda_reg = 1.0 / self.regularization_param if self.regularization_param > 0 else 0.0
            P = np.linalg.inv(H_T @ H_batch + lambda_reg * identity)
            
            # Initial output weights
            beta = P @ H_T @ y_batch
        
        # Process remaining batches
        for i in range(1, n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            actual_batch_size = end_idx - start_idx
            
            logger.debug(f"Processing batch {i+1}/{n_batches} (samples {start_idx}:{end_idx})")
            
            X_batch = X_norm[start_idx:end_idx]
            y_batch = y_norm[start_idx:end_idx]
            
            # Apply functional expansion to this batch
            H_batch = self._functional_expansion(X_batch)
            H_T = H_batch.T
            
            # Update P and beta using recursive least squares
            # P_{k+1} = P_k - P_k * H_k^T * (I + H_k * P_k * H_k^T)^-1 * H_k * P_k
            # beta_{k+1} = beta_k + P_{k+1} * H_k^T * (y_k - H_k * beta_k)
            try:
                temp1 = P @ H_T
                temp2 = np.linalg.inv(np.identity(actual_batch_size) + H_batch @ temp1)
                P = P - temp1 @ temp2 @ H_batch @ P
                beta = beta + P @ H_T @ (y_batch - H_batch @ beta)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Singular matrix in batch {i+1}: {e}. Using simplified update.")
                # Simplified update as fallback
                H_all = self._functional_expansion(X_norm[:end_idx])
                y_all = y_norm[:end_idx]
                beta = np.linalg.pinv(H_all) @ y_all
        
        # Set final output weights
        self.output_weights = beta
    
    def predict(self, X: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            denormalize: Whether to denormalize outputs
            
        Returns:
            Predictions, shape (n_samples, output_dim) or (n_samples,) if output_dim=1
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        # Ensure input has correct shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, got {X.shape[1]}")
        
        # Normalize input
        X_norm = self._normalize_inputs(X)
        
        # Apply functional expansion
        H = self._functional_expansion(X_norm)
        
        # Calculate predictions
        predictions = H @ self.output_weights
        
        # Denormalize if requested
        if denormalize:
            predictions = self._denormalize_outputs(predictions)
        
        # Return with correct shape
        if self.output_dim == 1:
            return predictions.flatten()
        return predictions
    
    def predict_signal(self, X: np.ndarray, output_range=(0, 1)) -> np.ndarray:
        """
        Make predictions and normalize to a specified range (useful for signals).
        
        Args:
            X: Input features
            output_range: Tuple of (min, max) for output scaling
            
        Returns:
            Predictions scaled to output_range
        """
        predictions = self.predict(X)
        
        # Ensure predictions is 2D for consistent handling
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Scale to output range
        min_val, max_val = output_range
        range_size = max_val - min_val
        
        # Scale each output dimension separately
        scaled_predictions = np.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            col = predictions[:, i]
            col_min, col_max = np.min(col), np.max(col)
            
            # Avoid division by zero
            if col_max > col_min:
                normalized = (col - col_min) / (col_max - col_min)
            else:
                normalized = np.ones_like(col) * 0.5
            
            scaled_predictions[:, i] = normalized * range_size + min_val
        
        # Return with correct shape
        if self.output_dim == 1:
            return scaled_predictions.flatten()
        return scaled_predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Calculate feature importance based on output weights magnitude.
        
        Returns:
            Array of importance scores for each input feature
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        # Initialize importance array
        importance = np.zeros(self.input_dim)
        
        # Direct weights for original features
        direct_weights = self.output_weights[:self.input_dim, :]
        
        # Add direct contribution
        if self.output_dim > 1:
            importance += np.sum(np.abs(direct_weights), axis=1)
        else:
            importance += np.abs(direct_weights.flatten())
        
        # Add contribution from expanded features
        if self.per_feature_expansion:
            # For per-feature expansion, we can attribute expanded features back to original ones
            expanded_weights = self.output_weights[self.input_dim:, :]
            
            if self.expansion_type == ExpansionType.TRIGONOMETRIC:
                # Each feature has 2*order expanded terms
                terms_per_feature = 2 * self.expansion_order
                for i in range(self.input_dim):
                    start_idx = i * terms_per_feature
                    end_idx = (i + 1) * terms_per_feature
                    feature_expanded_weights = expanded_weights[start_idx:end_idx, :]
                    if self.output_dim > 1:
                        importance[i] += np.sum(np.abs(feature_expanded_weights))
                    else:
                        importance[i] += np.sum(np.abs(feature_expanded_weights.flatten()))
            
            elif self.expansion_type in (ExpansionType.POLYNOMIAL, ExpansionType.CHEBYSHEV, ExpansionType.HERMITE):
                # Each feature has order expanded terms
                terms_per_feature = self.expansion_order
                for i in range(self.input_dim):
                    start_idx = i * terms_per_feature
                    end_idx = (i + 1) * terms_per_feature
                    feature_expanded_weights = expanded_weights[start_idx:end_idx, :]
                    if self.output_dim > 1:
                        importance[i] += np.sum(np.abs(feature_expanded_weights))
                    else:
                        importance[i] += np.sum(np.abs(feature_expanded_weights.flatten()))
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def get_expanded_feature_importance(self) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Calculate importance for each expanded feature.
        
        Returns:
            Tuple of (importance array, feature names)
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        # Calculate importance based on weight magnitudes
        if self.output_dim > 1:
            importance = np.sum(np.abs(self.output_weights), axis=1)
        else:
            importance = np.abs(self.output_weights.flatten())
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance, self.expanded_feature_names
    
    def save(self, filepath: str):
        """
        Save the CEFLANN-ELM model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self._trained:
            logger.warning("Saving untrained model.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'expansion_type': self.expansion_type.value,
            'expansion_order': self.expansion_order,
            'per_feature_expansion': self.per_feature_expansion,
            'activation_scale': self.activation_scale,
            'regularization_param': self.regularization_param,
            'seed': self.seed,
            '_expansion_weights': self._expansion_weights,
            'output_weights': self.output_weights,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'feature_names': self.feature_names,
            'expanded_feature_names': self.expanded_feature_names,
            '_trained': self._trained
        }
        
        # Save model data
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a CEFLANN-ELM model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded CEFLANN-ELM model
        """
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Create model instance
            model = cls(
                input_dim=model_data['input_dim'],
                output_dim=model_data['output_dim'],
                expansion_type=model_data['expansion_type'],
                expansion_order=model_data['expansion_order'],
                per_feature_expansion=model_data['per_feature_expansion'],
                activation_scale=model_data['activation_scale'],
                C=model_data['regularization_param'],
                seed=model_data['seed']
            )
            
            # Restore model state
            model._expansion_weights = model_data['_expansion_weights']
            model.output_weights = model_data['output_weights']
            model.input_mean = model_data['input_mean']
            model.input_std = model_data['input_std']
            model.output_mean = model_data['output_mean']
            model.output_std = model_data['output_std']
            model.feature_names = model_data['feature_names']
            model.expanded_feature_names = model_data['expanded_feature_names']
            model._trained = model_data['_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained"""
        return self._trained
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model information
        """
        summary = {
            'type': 'CEFLANN-ELM',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'expanded_dim': self.expanded_dim,
            'expansion_type': self.expansion_type.value,
            'expansion_order': self.expansion_order,
            'per_feature_expansion': self.per_feature_expansion,
            'regularization': self.regularization_param,
            'trained': self.is_trained,
        }
        
        if self.is_trained:
            summary['feature_names'] = self.feature_names
            
            # Calculate feature importance if trained
            try:
                importance = self.get_feature_importance()
                if self.feature_names is not None:
                    feature_importance = {name: float(imp) for name, imp in 
                                        zip(self.feature_names, importance)}
                else:
                    feature_importance = {f"feature_{i}": float(imp) for i, imp in 
                                        enumerate(importance)}
                
                summary['feature_importance'] = feature_importance
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")
        
        return summary


# --- Utility Functions ---

def create_ceflann_elm(input_features: pd.DataFrame, target_values: pd.Series,
                     expansion_type: str = 'trigonometric',
                     expansion_order: int = 5,
                     regularization: float = 1e-3,
                     use_jax: bool = True,
                     batch_size: Optional[int] = None) -> CEFLANN_ELM:
    """
    Helper function to create and train a CEFLANN-ELM model from pandas DataFrame/Series.
    
    Args:
        input_features: Input features DataFrame
        target_values: Target values Series
        expansion_type: Type of functional expansion
        expansion_order: Order of expansion
        regularization: Regularization parameter
        use_jax: Whether to use JAX acceleration
        batch_size: Batch size for large datasets
        
    Returns:
        Trained CEFLANN-ELM model
    """
    # Convert DataFrame/Series to numpy arrays
    X = input_features.values
    y = target_values.values
    
    # Get feature names
    feature_names = input_features.columns.tolist()
    
    # Ensure y is 2D for multi-output
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Create model
    model = CEFLANN_ELM(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        expansion_type=expansion_type,
        expansion_order=expansion_order,
        C=regularization,
        use_jax=use_jax
    )
    
    # Train model
    model.fit(
        X_train=X,
        y_train=y,
        normalize=True,
        batch_size=batch_size,
        feature_names=feature_names
    )
    
    return model