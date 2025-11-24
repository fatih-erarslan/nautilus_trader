#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:14:28 2025

@author: ashina
"""

import logging
import time
import os
import pickle
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque
import copy

# Import River components with proper error handling
try:
    import river
    from river import drift
    from river import anomaly
    from river import preprocessing
    from river import metrics
    from river import stats
    from river import ensemble
    from river import compose
    from river import feature_selection
    from river import time_series
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

class RiverOnlineML:
    """
    Enterprise-grade River implementation for online machine learning in trading.
    
    This class provides drift detection, anomaly detection, and adaptive feature
    selection using River's online learning capabilities. It includes comprehensive
    error handling, thread safety, and performance monitoring.
    """
    
    def __init__(self, 
                 drift_detector_type: str = 'adwin',
                 anomaly_detector_type: str = 'hst',
                 feature_window: int = 50,
                 drift_sensitivity: float = 0.05,
                 anomaly_threshold: float = 0.95,
                 log_level: int = logging.INFO,
                 enable_feature_selection: bool = True,
                 max_selected_features: int = 10):
        """
        Initialize the River Online ML component.
        
        Args:
            drift_detector_type (str): Type of drift detector ('adwin', 'ddm', 'page_hinkley')
            anomaly_detector_type (str): Type of anomaly detector ('hst', 'robust_svm', 'loda')
            feature_window (int): Window size for feature tracking
            drift_sensitivity (float): Sensitivity parameter for drift detection
            anomaly_threshold (float): Threshold for anomaly detection
            log_level (int): Logging level
            enable_feature_selection (bool): Whether to enable online feature selection
            max_selected_features (int): Maximum number of features to select
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Configuration
        self.drift_detector_type = drift_detector_type
        self.anomaly_detector_type = anomaly_detector_type
        self.feature_window = feature_window
        self.drift_sensitivity = drift_sensitivity
        self.anomaly_threshold = anomaly_threshold
        self.enable_feature_selection = enable_feature_selection
        self.max_selected_features = max_selected_features
        
        # State tracking
        self.is_initialized = False
        self.metrics_trackers = {}
        self.feature_importance = {}
        self.recent_features = deque(maxlen=feature_window)
        self.drift_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=100)
        self.error_count = 0
        self.sequential_errors = 0
        self.last_update_time = None
        self.total_updates = 0
        self.execution_times = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize River components.

        Returns:
            bool: True if initialization succeeded
        """
        self.logger.info("Attempting River Online ML initialization...")
        # Ensure state is False before starting
        self.is_initialized = False

        if not RIVER_AVAILABLE:
            self.logger.warning("River library not available. Initialization skipped.")
            return False

        try:
            with self.lock:
                # Reset components to None before assignment
                self.drift_detector = None
                self.anomaly_detector = None
                self.feature_selector = None
                self.feature_scaler = None
                self.metrics_trackers = {}

                # Initialize drift detector
                self.drift_detector = self._create_drift_detector()
                if self.drift_detector:
                    self.logger.info(f"Drift detector ({self.drift_detector_type}) created.")
                else:
                    raise ValueError("Failed to create drift detector") # Force error if creation failed

                # Initialize anomaly detector
                self.anomaly_detector = self._create_anomaly_detector()
                if self.anomaly_detector:
                     self.logger.info(f"Anomaly detector ({self.anomaly_detector_type}) created.")
                else:
                     raise ValueError("Failed to create anomaly detector")

                # Initialize feature selector
                if self.enable_feature_selection:
                    self.feature_selector = self._create_feature_selector()
                    if self.feature_selector:
                         self.logger.info("Feature selector created.")
                    # Allow feature selector to be optional, don't raise error if None
                else:
                    self.feature_selector = None # Explicitly set to None if disabled

                # Initialize metrics trackers
                self.metrics_trackers = {
                    'drift_rate': stats.Mean(),
                    'anomaly_score': stats.Mean(),
                    'processing_time': stats.Mean(),
                    'feature_count': stats.Mean()
                }
                self.logger.info("Metrics trackers created.")

                # Initialize feature scalers
                self.feature_scaler = preprocessing.StandardScaler()
                if self.feature_scaler:
                     self.logger.info("Feature scaler (StandardScaler) created.")
                else:
                     # This should almost never happen unless river import itself is broken
                     raise ValueError("Failed to create StandardScaler")

                # --- Set is_initialized ONLY after all successful assignments ---
                self.is_initialized = True
                self.logger.info("--- River Online ML initialization complete ---")
                return True

        except Exception as e:
            self.logger.error(f"--- River initialization FAILED: {str(e)} ---", exc_info=True)
            self.error_count += 1
            self.sequential_errors += 1
            self.is_initialized = False # Ensure it's False on error
            # Explicitly set components to None on failure
            self.drift_detector = None
            self.anomaly_detector = None
            self.feature_selector = None
            self.feature_scaler = None
            self.metrics_trackers = {}
            return False    
    def _create_drift_detector(self):
        """
        Create the appropriate drift detector based on configuration.
        
        Returns:
            river.drift.base.DriftDetector: Configured drift detector
        """
        if self.drift_detector_type.lower() == 'adwin':
            detector = drift.ADWIN(delta=self.drift_sensitivity)
        elif self.drift_detector_type.lower() == 'ddm':
            detector = drift.DDM(min_num_instances=30, warning_level=self.drift_sensitivity*2, 
                                 drift_level=self.drift_sensitivity)
        elif self.drift_detector_type.lower() == 'page_hinkley':
            detector = drift.PageHinkley(min_instances=30, delta=self.drift_sensitivity, 
                                        threshold=20, alpha=0.9999)
        else:
            # Default to ensemble of drift detectors for robustness
            detector = drift.EnsembleDriftDetector(
                [
                    drift.ADWIN(delta=self.drift_sensitivity),
                    drift.DDM(min_num_instances=30, warning_level=self.drift_sensitivity*2, 
                             drift_level=self.drift_sensitivity),
                    drift.PageHinkley(min_instances=30, delta=self.drift_sensitivity, 
                                     threshold=20, alpha=0.9999)
                ],
                vote_threshold=2/3
            )
        
        return detector
    
    def _create_anomaly_detector(self):
        """
        Create the appropriate anomaly detector based on configuration.
        
        Returns:
            Object: Configured anomaly detector
        """
        if self.anomaly_detector_type.lower() == 'hst':
            detector = anomaly.HalfSpaceTrees(
                n_trees=50,
                height=8,
                window_size=self.feature_window,
                seed=42
            )
        elif self.anomaly_detector_type.lower() == 'robust_svm':
            detector = compose.Pipeline(
                preprocessing.StandardScaler(),
                anomaly.RobustOneClassSVM(nu=0.1)
            )
        elif self.anomaly_detector_type.lower() == 'loda':
            detector = anomaly.LODA(n_estimators=20)
        else:
            # Default to ensemble approach
            detector = compose.Pipeline(
                preprocessing.StandardScaler(),
                anomaly.HalfSpaceTrees(
                    n_trees=50,
                    height=8,
                    window_size=self.feature_window,
                    seed=42
                )
            )
        
        return detector
    
    def _create_feature_selector(self):
        """
        Create online feature selector.
        
        Returns:
            Object: Configured feature selector
        """
        # Create a feature selection pipeline
        selector = feature_selection.VarianceThreshold(threshold=0.05)
        return selector
    
    def _check_initialization(self) -> bool:
        """
        Check if initialization is complete and attempt re-initialization if needed.
        
        Returns:
            bool: True if initialized
        """
        # If already initialized, return True
        if self.is_initialized:
            return True
            
        # If not initialized and River is not available, return False
        if not RIVER_AVAILABLE:
            return False
            
        # Attempt re-initialization
        return self._initialize()
    
    # Inside RiverOnlineML class (`river_ml.py`)
    def _preprocess_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Preprocess features assuming learn_one updates IN-PLACE and returns None.
        """
        # Use a specific logger instance for this method for clarity
        method_logger = logging.getLogger(f"{self.__class__.__name__}._preprocess_features")
        method_logger.debug(f"Input features: {features}")

        try:
            # --- Step 1: Initial Checks & Filtering ---
            if self.feature_scaler is None:
                method_logger.error("Feature scaler (self.feature_scaler) is None. Cannot preprocess.")
                return {} # Cannot proceed without scaler

            # Filter non-finite values
            original_keys = list(features.keys())
            features_filtered = {k: float(v) for k, v in features.items()
                                 if isinstance(v, (int, float)) and np.isfinite(v)}
            filtered_keys = list(features_filtered.keys())

            if len(original_keys) != len(filtered_keys):
                method_logger.warning(f"Filtered invalid features: {set(original_keys) - set(filtered_keys)}")
            if not features_filtered:
                method_logger.warning("No valid features remain after filtering.")
                return {} # Cannot proceed if no features left

            # --- Step 2: Scaling (In-Place Learn + Transform) ---
            method_logger.debug(f"Calling IN-PLACE feature_scaler.learn_one on {features_filtered}")
            try:
                # Assume learn_one updates self.feature_scaler and might return None
                learn_return = self.feature_scaler.learn_one(features_filtered)
                if learn_return is not None:
                     # This is unexpected based on logs, but good to log if it happens
                     method_logger.warning(f"StandardScaler learn_one unexpectedly returned: {type(learn_return)}")
                # Double-check scaler didn't become None (highly defensive)
                if self.feature_scaler is None:
                     method_logger.critical("Feature scaler became None after learn_one!")
                     return {}
            except Exception as learn_err:
                 method_logger.error(f"Error during feature_scaler.learn_one: {learn_err}", exc_info=True)
                 # Attempt to transform anyway, maybe learning failure was partial
            # Now transform using the (presumably updated) self.feature_scaler
            try:
                method_logger.debug(f"Calling feature_scaler.transform_one")
                scaled_features = self.feature_scaler.transform_one(features_filtered)
                method_logger.debug(f"Scaled features: {scaled_features}")
                if scaled_features is None: # Check if transform returns None
                     method_logger.error("StandardScaler transform_one returned None!")
                     return {} # Cannot proceed
            except Exception as transform_err:
                 method_logger.error(f"Error during feature_scaler.transform_one: {transform_err}", exc_info=True)
                 return {} # Cannot proceed if transform fails

            # --- Step 3: Feature Selection (In-Place Learn + Transform, if enabled) ---
            selected_features = scaled_features # Default to scaled if no selection
            if self.enable_feature_selection and self.feature_selector is not None:
                method_logger.debug(f"Applying Feature Selection ({type(self.feature_selector)})")
                try:
                    # Assume learn_one updates self.feature_selector
                    learn_return_sel = self.feature_selector.learn_one(scaled_features)
                    if learn_return_sel is not None:
                         method_logger.warning(f"Feature selector learn_one unexpectedly returned: {type(learn_return_sel)}")
                    # Check selector didn't become None
                    if self.feature_selector is None:
                         method_logger.error("Feature selector became None after learn_one!")
                         # Continue with scaled_features as selected_features
                    else:
                        # Transform using the updated selector
                        method_logger.debug(f"Calling feature_selector.transform_one")
                        selected_features_candidate = self.feature_selector.transform_one(scaled_features)

                        # Handle transform returning None or empty
                        if selected_features_candidate is None:
                             method_logger.warning("Feature selector transform_one returned None. Using unselected (scaled) features.")
                             selected_features = scaled_features
                        elif not selected_features_candidate and scaled_features:
                              method_logger.warning("Feature selection resulted in empty features. Using unselected (scaled) features.")
                              selected_features = scaled_features
                        else:
                              selected_features = selected_features_candidate # Assign valid selection
                              method_logger.debug(f"Selected features after transform: {selected_features}")

                except Exception as fs_err:
                     method_logger.error(f"Error during feature selection learn/transform: {fs_err}", exc_info=True)
                     selected_features = scaled_features # Fallback to scaled on error
            # --- End Feature Selection ---

            # --- Step 4: Final Return ---
            # Ensure we return a dictionary, even if empty
            if selected_features is None:
                 method_logger.error("Preprocessing result 'selected_features' is None unexpectedly.")
                 return {}

            method_logger.info(f"_preprocess_features successful. Output keys: {list(selected_features.keys())}")
            return selected_features

        except Exception as e:
            method_logger.error(f"Unhandled error in _preprocess_features: {str(e)}", exc_info=True)
            return {} # Return empty dict on any unhandled error

    def _prepare_qstar_features(self, dataframe):
        """Prepare features for QStar prediction with better error handling"""
        try:
            # Create a deep copy to avoid modifying original
            df = dataframe.copy()
            
            # Ensure all required columns exist with default values
            required_columns = ['close', 'volume', 'rsi_14', 'adx']
            for col in required_columns:
                if col not in df:
                    self.logger.warning(f"Required column {col} missing, adding default")
                    df[col] = 0.0 if col != 'close' else df.index  # Use index as fallback for close
            
            # Replace NaN/inf values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill then backward fill to handle NaNs
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Convert all values to appropriate types
            for col in df.columns:
                if col in ['rsi_14', 'adx', 'volatility_regime', 'antifragility', 'soc_equilibrium']:
                    df[col] = df[col].astype(float)
            
            # Return clean dataframe with consistent types
            return df
        except Exception as e:
            self.logger.error(f"Error preparing QStar features: {e}")
            # Return minimal valid dataframe
            return pd.DataFrame({'close': [0.0], 'volume': [0.0]})
        
        
    def detect_drift(self, value: float) -> Dict[str, Any]:
        """
        Detect concept drift in data stream.
        
        Args:
            value (float): Current data point to check for drift
            
        Returns:
            Dict[str, Any]: Drift detection results
        """
        if not self._check_initialization():
            return {'drift_detected': False, 'warning': False, 'error': 'Not initialized'}
            
        start_time = time.time()
        
        try:
            with self.lock:
                # Update drift detector
                self.drift_detector.update(value)
                
                # Check for drift
                drift_detected = self.drift_detector.drift_detected
                
                # Check for warning if available
                warning = getattr(self.drift_detector, 'warning_detected', False)
                
                # Get drift level if available
                drift_level = getattr(self.drift_detector, 'drift_level', None)
                
                # Track drift events
                self.drift_history.append((drift_detected, time.time()))
                
                # Update metrics
                drift_rate = sum(1 for d, _ in self.drift_history if d) / len(self.drift_history) if self.drift_history else 0
                self.metrics_trackers['drift_rate'].update(drift_rate)
                
                # Calculate execution time
                execution_time = (time.time() - start_time) * 1000  # ms
                self.execution_times.append(execution_time)
                self.metrics_trackers['processing_time'].update(execution_time)
                
                return {
                    'drift_detected': drift_detected,
                    'warning': warning,
                    'drift_level': drift_level,
                    'drift_rate': drift_rate,
                    'execution_time_ms': execution_time
                }
                
        except Exception as e:
            self.logger.error(f"Drift detection error: {str(e)}", exc_info=True)
            self.error_count += 1
            self.sequential_errors += 1
            return {
                'drift_detected': False,
                'warning': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
    
    def detect_anomalies(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in feature vector. (With enhanced logging)
        """
        # Default result in case of early exit or error
        default_result = {'anomaly_score': 0.0, 'is_anomaly': False, 'error': None, 'execution_time_ms': 0}
        start_time = time.time() # Define start_time here
    
        if not self._check_initialization():
            self.logger.warning("Anomaly detection skipped: Not initialized")
            default_result['error'] = 'Not initialized'
            default_result['execution_time_ms'] = (time.time() - start_time) * 1000
            return default_result
    
        anomaly_score = 0.0 # Initialize local score variable
        is_anomaly = False
        processed_features_len = 0
    
        try:
            with self.lock:
                self.sequential_errors = 0 # Reset on successful entry
    
                # --- Step 1: Preprocess ---
                self.logger.debug(f"Calling _preprocess_features with raw features: {features}")
                processed_features = self._preprocess_features(features)
                self.logger.info(f"_preprocess_features returned: {processed_features} (type: {type(processed_features)})") # Use INFO level
    
                if not processed_features or not isinstance(processed_features, dict):
                    self.logger.warning("Anomaly detection skipped: Preprocessing returned no valid features or incorrect type.")
                    default_result['error'] = 'Preprocessing failed'
                    default_result['execution_time_ms'] = (time.time() - start_time) * 1000
                    return default_result
    
                processed_features_len = len(processed_features)
    
                # --- Step 2: Check Anomaly Detector ---
                self.logger.debug(f"Checking anomaly detector. Type: {type(self.anomaly_detector)}")
                if self.anomaly_detector is None:
                    self.logger.error("Anomaly detector (self.anomaly_detector) is None! Cannot score.")
                    default_result['error'] = 'Anomaly detector not initialized'
                    default_result['execution_time_ms'] = (time.time() - start_time) * 1000
                    return default_result
    
                # --- Step 3: Score ---
                self.logger.debug(f"Calling anomaly_detector.score_one with processed features: {processed_features}")
                # Add try-except specifically around score_one/learn_one
                try:
                    anomaly_score_raw = self.anomaly_detector.score_one(processed_features) # Get the raw score
                    self.logger.info(f"Raw anomaly score from detector.score_one: {anomaly_score_raw} (type: {type(anomaly_score_raw)})") # Log raw score
    
                    # Validate score (handle potential None/NaN from detector)
                    if anomaly_score_raw is None or pd.isna(anomaly_score_raw):
                        self.logger.warning(f"Anomaly detector returned invalid score ({anomaly_score_raw}). Treating as 0.0.")
                        anomaly_score = 0.0 # Force to 0.0 if invalid
                    else:
                        anomaly_score = float(anomaly_score_raw) # Convert valid score to float
    
                    # Determine if it's an anomaly based on the validated score
                    is_anomaly = anomaly_score > self.anomaly_threshold
    
                    # --- Step 4: Learn ---
                    self.logger.debug(f"Calling anomaly_detector.learn_one with processed features.")
                    self.anomaly_detector.learn_one(processed_features)
    
                except Exception as e_score_learn:
                    self.logger.error(f"Error during anomaly score_one/learn_one call: {e_score_learn}", exc_info=True)
                    self.error_count += 1 # Count this specific error
                    self.sequential_errors += 1
                    default_result['error'] = f"Score/Learn Error: {e_score_learn}"
                    default_result['execution_time_ms'] = (time.time() - start_time) * 1000
                    return default_result # Return default 0.0 score
    
                # --- Step 5: Update State & Metrics (only if score/learn succeeded) ---
                self.recent_features.append(processed_features)
                self.anomaly_history.append((is_anomaly, time.time()))
                self.metrics_trackers['anomaly_score'].update(anomaly_score) # Update with validated score
                self.metrics_trackers['feature_count'].update(processed_features_len)
    
                # Calculate execution time for this successful run
                execution_time = (time.time() - start_time) * 1000
    
                # Prepare successful result
                result = {
                    'anomaly_score': anomaly_score, # Use the validated score
                    'is_anomaly': is_anomaly,
                    'processed_features': processed_features_len,
                    'execution_time_ms': execution_time,
                    'error': None # Explicitly set error to None on success
                }
                self.total_updates += 1
                return result
    
        except Exception as e: # Catch errors in the main try block (e.g., lock issues)
            self.logger.error(f"Outer Anomaly detection error: {str(e)}", exc_info=True)
            self.error_count += 1
            self.sequential_errors += 1
            default_result['error'] = str(e)
            default_result['execution_time_ms'] = (time.time() - start_time) * 1000
            return default_result
    
    def update_feature_importance(self, features: Dict[str, float], target: float) -> Dict[str, float]:
        """
        Update feature importance based on their correlation with target.
        
        Args:
            features (Dict[str, float]): Feature dictionary
            target (float): Target value
            
        Returns:
            Dict[str, float]: Updated feature importance scores
        """
        if not self._check_initialization():
            return {}
            
        try:
            with self.lock:
                # Initialize feature importance tracking if needed
                for feature in features:
                    if feature not in self.feature_importance:
                        self.feature_importance[feature] = {
                            'correlation': stats.RollingPearson(self.feature_window),
                            'covariance': stats.RollingCovariance(self.feature_window),
                            'importance': 0.0
                        }
                
                # Update correlation and covariance statistics
                for feature, value in features.items():
                    if feature in self.feature_importance:
                        try:
                            # Update correlation
                            self.feature_importance[feature]['correlation'].update(value, target)
                            # Update covariance
                            self.feature_importance[feature]['covariance'].update(value, target)
                        except Exception as e:
                            self.logger.warning(f"Error updating feature {feature}: {str(e)}")
                
                # Calculate importance scores
                for feature in self.feature_importance:
                    corr = abs(self.feature_importance[feature]['correlation'].get())
                    cov = abs(self.feature_importance[feature]['covariance'].get())
                    
                    # Combine metrics (with safeguards against NaN/inf)
                    if np.isnan(corr) or np.isinf(corr):
                        corr = 0.0
                    if np.isnan(cov) or np.isinf(cov):
                        cov = 0.0
                        
                    # Simple weighted combination
                    self.feature_importance[feature]['importance'] = 0.7 * corr + 0.3 * (cov / (1 + cov))
                
                # Return current importance scores
                return {f: data['importance'] for f, data in self.feature_importance.items()}
                
        except Exception as e:
            self.logger.error(f"Feature importance update error: {str(e)}", exc_info=True)
            self.error_count += 1
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about online learning components.
        
        Returns:
            Dict[str, Any]: Statistics about drift, anomalies, and features
        """
        if not self._check_initialization():
            return {'error': 'Not initialized', 'is_initialized': False}
            
        try:
            with self.lock:
                # Calculate drift rate
                drift_rate = sum(1 for d, _ in self.drift_history if d) / len(self.drift_history) if self.drift_history else 0
                
                # Calculate anomaly rate
                anomaly_rate = sum(1 for a, _ in self.anomaly_history if a) / len(self.anomaly_history) if self.anomaly_history else 0
                
                # Get feature importance
                top_features = sorted(
                    [(f, data['importance']) for f, data in self.feature_importance.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 features
                
                # Get average processing time
                avg_processing_time = self.metrics_trackers['processing_time'].get() if 'processing_time' in self.metrics_trackers else 0
                
                return {
                    'is_initialized': self.is_initialized,
                    'drift_rate': drift_rate,
                    'anomaly_rate': anomaly_rate,
                    'top_features': dict(top_features),
                    'feature_count': len(self.feature_importance),
                    'processing_time_ms': avg_processing_time,
                    'error_count': self.error_count,
                    'total_updates': self.total_updates,
                    'drift_detector_type': self.drift_detector_type,
                    'anomaly_detector_type': self.anomaly_detector_type
                }
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'is_initialized': self.is_initialized,
                'error_count': self.error_count
            }
    
    def reset_drift_detector(self) -> bool:
        """
        Reset drift detector state.
        
        Returns:
            bool: True if reset successful
        """
        if not self._check_initialization():
            return False
            
        try:
            with self.lock:
                # Create new drift detector
                self.drift_detector = self._create_drift_detector()
                self.drift_history.clear()
                self.logger.info("Drift detector reset")
                return True
                
        except Exception as e:
            self.logger.error(f"Error resetting drift detector: {str(e)}", exc_info=True)
            return False
    
    def reset_anomaly_detector(self) -> bool:
        """
        Reset anomaly detector state.
        
        Returns:
            bool: True if reset successful
        """
        if not self._check_initialization():
            return False
            
        try:
            with self.lock:
                # Create new anomaly detector
                self.anomaly_detector = self._create_anomaly_detector()
                self.anomaly_history.clear()
                self.logger.info("Anomaly detector reset")
                return True
                
        except Exception as e:
            self.logger.error(f"Error resetting anomaly detector: {str(e)}", exc_info=True)
            return False
    
    def save_state(self, filepath: str) -> bool:
        """
        Save the current state to a file.
        
        Args:
            filepath (str): Path to save state
            
        Returns:
            bool: True if save successful
        """
        if not self._check_initialization():
            return False
            
        try:
            with self.lock:
                # Create state dictionary
                state = {
                    'drift_detector': self.drift_detector,
                    'anomaly_detector': self.anomaly_detector,
                    'feature_scaler': self.feature_scaler,
                    'feature_importance': self.feature_importance,
                    'metrics_trackers': self.metrics_trackers,
                    'config': {
                        'drift_detector_type': self.drift_detector_type,
                        'anomaly_detector_type': self.anomaly_detector_type,
                        'feature_window': self.feature_window,
                        'drift_sensitivity': self.drift_sensitivity,
                        'anomaly_threshold': self.anomaly_threshold,
                        'enable_feature_selection': self.enable_feature_selection,
                        'max_selected_features': self.max_selected_features
                    },
                    'metadata': {
                        'version': '1.0',
                        'timestamp': time.time(),
                        'error_count': self.error_count,
                        'total_updates': self.total_updates
                    }
                }
                
                # Save feature selector if enabled
                if self.enable_feature_selection and hasattr(self, 'feature_selector'):
                    state['feature_selector'] = self.feature_selector
                
                # Serialize to file
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
                    
                self.logger.info(f"State saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}", exc_info=True)
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load state from a file.
        
        Args:
            filepath (str): Path to load state from
            
        Returns:
            bool: True if load successful
        """
        if not RIVER_AVAILABLE:
            self.logger.warning("River library not available. Cannot load state.")
            return False
            
        try:
            with self.lock:
                # Load serialized state
                with open(filepath, 'rb') as f:
                    state = pickle.load(f)
                
                # Restore components
                self.drift_detector = state['drift_detector']
                self.anomaly_detector = state['anomaly_detector']
                self.feature_scaler = state['feature_scaler']
                self.feature_importance = state['feature_importance']
                self.metrics_trackers = state['metrics_trackers']
                
                # Restore configuration
                config = state.get('config', {})
                self.drift_detector_type = config.get('drift_detector_type', self.drift_detector_type)
                self.anomaly_detector_type = config.get('anomaly_detector_type', self.anomaly_detector_type)
                self.feature_window = config.get('feature_window', self.feature_window)
                self.drift_sensitivity = config.get('drift_sensitivity', self.drift_sensitivity)
                self.anomaly_threshold = config.get('anomaly_threshold', self.anomaly_threshold)
                self.enable_feature_selection = config.get('enable_feature_selection', self.enable_feature_selection)
                self.max_selected_features = config.get('max_selected_features', self.max_selected_features)
                
                # Restore feature selector if available
                if 'feature_selector' in state and self.enable_feature_selection:
                    self.feature_selector = state['feature_selector']
                    
                # Restore metadata
                metadata = state.get('metadata', {})
                self.error_count = metadata.get('error_count', 0)
                self.total_updates = metadata.get('total_updates', 0)
                
                self.is_initialized = True
                self.logger.info(f"State loaded from {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}", exc_info=True)
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the component.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            with self.lock:
                # Check if initialized
                initialized = self.is_initialized and RIVER_AVAILABLE
                
                # Check for excessive errors
                error_rate = self.error_count / max(1, self.total_updates) if self.total_updates > 0 else 0
                excessive_errors = error_rate > 0.1 or self.sequential_errors > 5
                
                # Check processing time
                avg_processing_time = self.metrics_trackers.get('processing_time', stats.Mean()).get()
                processing_time_ok = avg_processing_time < 100  # Less than 100ms is acceptable
                
                # Overall health status
                status = "healthy" if (initialized and not excessive_errors and processing_time_ok) else "degraded"
                
                return {
                    'status': status,
                    'initialized': initialized,
                    'river_available': RIVER_AVAILABLE,
                    'error_rate': error_rate,
                    'sequential_errors': self.sequential_errors,
                    'avg_processing_time_ms': avg_processing_time,
                    'last_update_time': self.last_update_time,
                    'total_updates': self.total_updates
                }
                
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'river_available': RIVER_AVAILABLE
            }