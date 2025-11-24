#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:49:27 2025

@author: ashina
"""

# --- qstar_incremental_learning.py ---

import os
import numpy as np
import pandas as pd
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import pickle
import hashlib
from functools import lru_cache
import traceback
import gc
from collections import deque
import json
import random

logger = logging.getLogger("QStarIncrementalLearning")

class DataChunkManager:
    """
    Manages data chunks for incremental learning with efficient disk I/O.
    """
    
    def __init__(self, data_dir: str = 'data', 
                 chunk_size: int = 5000,
                 max_chunks_in_memory: int = 3,
                 data_ttl: int = 86400):  # 24 hours default TTL
        """
        Initialize data chunk manager.
        
        Args:
            data_dir: Directory to store data chunks
            chunk_size: Number of rows per chunk
            max_chunks_in_memory: Maximum chunks to keep in memory
            data_ttl: Time-to-live for data chunks in seconds
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.max_chunks_in_memory = max_chunks_in_memory
        self.data_ttl = data_ttl
        
        self._chunk_cache = {}  # chunk_id -> (data, timestamp)
        self._lock = threading.RLock()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load chunk metadata
        self._chunk_metadata = self._load_chunk_metadata()
        
        # Start cleanup thread
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_chunks, daemon=True)
        self._cleanup_thread.start()
    
    def __del__(self):
        """Stop cleanup thread on deletion."""
        self._stop_event.set()
        
    def load_csv_data(self, csv_path: str, columns: Optional[List[str]] = None) -> bool:
        """
        Load CSV data into chunks.
        
        Args:
            csv_path: Path to CSV file
            columns: Specific columns to load or None for all
            
        Returns:
            Success flag
        """
        try:
            logger.info(f"Loading CSV data from {csv_path}")
            
            # Calculate file hash for tracking
            file_hash = self._calculate_file_hash(csv_path)
            
            # Check if file was already processed
            with self._lock:
                if file_hash in self._chunk_metadata.get('processed_files', {}):
                    last_modified = os.path.getmtime(csv_path)
                    last_processed = self._chunk_metadata['processed_files'][file_hash].get('timestamp', 0)
                    
                    # Skip if file hasn't changed since last processing
                    if last_modified <= last_processed:
                        logger.info(f"File {csv_path} already processed and unchanged.")
                        return True
            
            # Read CSV in chunks
            chunk_id = 0
            for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size, usecols=columns):
                # Process and store chunk
                chunk_path = os.path.join(self.data_dir, f"chunk_{file_hash}_{chunk_id}.pkl")
                
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk, f)
                
                # Update metadata
                with self._lock:
                    if 'chunks' not in self._chunk_metadata:
                        self._chunk_metadata['chunks'] = {}
                    
                    self._chunk_metadata['chunks'][f"{file_hash}_{chunk_id}"] = {
                        'path': chunk_path,
                        'rows': len(chunk),
                        'timestamp': time.time()
                    }
                
                chunk_id += 1
            
            # Update processed files metadata
            with self._lock:
                if 'processed_files' not in self._chunk_metadata:
                    self._chunk_metadata['processed_files'] = {}
                
                self._chunk_metadata['processed_files'][file_hash] = {
                    'path': csv_path,
                    'chunks': chunk_id,
                    'timestamp': time.time()
                }
            
            # Save metadata
            self._save_chunk_metadata()
            
            logger.info(f"CSV data loaded into {chunk_id} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}", exc_info=True)
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[pd.DataFrame]:
        """
        Get data chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DataFrame chunk or None if not found
        """
        try:
            # Check memory cache first
            with self._lock:
                if chunk_id in self._chunk_cache:
                    data, _ = self._chunk_cache[chunk_id]
                    return data
                
                # Check if chunk exists in metadata
                if 'chunks' not in self._chunk_metadata or chunk_id not in self._chunk_metadata['chunks']:
                    logger.warning(f"Chunk {chunk_id} not found in metadata")
                    return None
                
                # Load from disk
                chunk_path = self._chunk_metadata['chunks'][chunk_id]['path']
                
                if not os.path.exists(chunk_path):
                    logger.warning(f"Chunk file {chunk_path} not found")
                    return None
                
                with open(chunk_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Cache in memory
                self._cache_chunk(chunk_id, data)
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}", exc_info=True)
            return None
    
    def get_next_batch(self, batch_size: int = 100, 
                      feature_columns: Optional[List[str]] = None,
                      target_column: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Get next batch of data for training.
        
        Args:
            batch_size: Number of rows in batch
            feature_columns: Feature column names
            target_column: Target column name
            
        Returns:
            Tuple of (feature_df, target_series) or (None, None) if no data
        """
        try:
            # Get list of available chunks
            with self._lock:
                if 'chunks' not in self._chunk_metadata:
                    logger.warning("No data chunks available")
                    return None, None
                
                chunk_ids = list(self._chunk_metadata['chunks'].keys())
            
            if not chunk_ids:
                logger.warning("No data chunks available")
                return None, None
            
            # Randomly select a chunk
            chunk_id = random.choice(chunk_ids)
            chunk_data = self.get_chunk(chunk_id)
            
            if chunk_data is None or len(chunk_data) == 0:
                logger.warning(f"Empty or missing chunk: {chunk_id}")
                return None, None
            
            # Select random batch from chunk
            if len(chunk_data) <= batch_size:
                batch = chunk_data
            else:
                # Select random starting index
                start_idx = random.randint(0, len(chunk_data) - batch_size)
                batch = chunk_data.iloc[start_idx:start_idx + batch_size]
            
            # Split features and target
            if feature_columns and target_column:
                # Validate columns
                valid_features = [col for col in feature_columns if col in batch.columns]
                
                if not valid_features:
                    logger.warning("No valid feature columns found in data")
                    return None, None
                
                if target_column not in batch.columns:
                    logger.warning(f"Target column {target_column} not found in data")
                    return None, None
                
                features = batch[valid_features]
                target = batch[target_column]
                
                return features, target
            else:
                # Return full batch if no specific columns requested
                return batch, None
                
        except Exception as e:
            logger.error(f"Error getting next batch: {e}", exc_info=True)
            return None, None
    
    def get_file_chunks(self, file_hash: str) -> List[str]:
        """
        Get all chunk IDs for a specific file.
        
        Args:
            file_hash: File hash
            
        Returns:
            List of chunk IDs
        """
        with self._lock:
            if 'processed_files' not in self._chunk_metadata or file_hash not in self._chunk_metadata['processed_files']:
                return []
            
            chunks = self._chunk_metadata['processed_files'][file_hash].get('chunks', 0)
            return [f"{file_hash}_{i}" for i in range(chunks)]
    
    def _cache_chunk(self, chunk_id: str, data: pd.DataFrame) -> None:
        """
        Cache chunk in memory with LRU policy.
        
        Args:
            chunk_id: Chunk identifier
            data: Chunk data
        """
        with self._lock:
            # Add to cache
            self._chunk_cache[chunk_id] = (data, time.time())
            
            # Enforce cache size limit
            if len(self._chunk_cache) > self.max_chunks_in_memory:
                # Find oldest chunk
                oldest_id = min(self._chunk_cache.items(), key=lambda x: x[1][1])[0]
                del self._chunk_cache[oldest_id]
    
    def _cleanup_old_chunks(self) -> None:
        """Clean up old chunks based on TTL in a background thread."""
        while not self._stop_event.is_set():
            try:
                now = time.time()
                chunks_to_remove = []
                
                with self._lock:
                    if 'chunks' in self._chunk_metadata:
                        for chunk_id, info in self._chunk_metadata['chunks'].items():
                            if now - info.get('timestamp', 0) > self.data_ttl:
                                chunks_to_remove.append((chunk_id, info.get('path')))
                
                # Remove old chunks
                for chunk_id, path in chunks_to_remove:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                        
                        with self._lock:
                            if 'chunks' in self._chunk_metadata and chunk_id in self._chunk_metadata['chunks']:
                                del self._chunk_metadata['chunks'][chunk_id]
                        
                        logger.debug(f"Removed old chunk: {chunk_id}")
                    except Exception as e:
                        logger.error(f"Error removing chunk {chunk_id}: {e}")
                
                if chunks_to_remove:
                    self._save_chunk_metadata()
            
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
            
            # Sleep before next check
            for _ in range(60):  # Check stop event every second
                if self._stop_event.is_set():
                    break
                time.sleep(1)
    
    def _load_chunk_metadata(self) -> Dict:
        """
        Load chunk metadata from disk.
        
        Returns:
            Metadata dictionary
        """
        metadata_path = os.path.join(self.data_dir, "chunk_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading chunk metadata: {e}")
        
        return {}
    
    def _save_chunk_metadata(self) -> None:
        """Save chunk metadata to disk."""
        metadata_path = os.path.join(self.data_dir, "chunk_metadata.json")
        
        try:
            with self._lock:
                with open(metadata_path, 'w') as f:
                    json.dump(self._chunk_metadata, f)
        except Exception as e:
            logger.error(f"Error saving chunk metadata: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hash string
        """
        try:
            # Use file size and modification time for quick hash
            file_stat = os.stat(file_path)
            file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            
            return hashlib.md5(file_info.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return f"error_{int(time.time())}"


class ModelStateManager:
    """Manages saving and loading of model states with versioning."""
    
    def __init__(self, model_dir: str = 'models', 
                 max_versions: int = 3,
                 save_interval: int = 3600):  # 1 hour default
        """
        Initialize model state manager.
        
        Args:
            model_dir: Directory to store model states
            max_versions: Maximum number of model versions to keep
            save_interval: Minimum interval between saves in seconds
        """
        self.model_dir = model_dir
        self.max_versions = max_versions
        self.save_interval = save_interval
        
        self._last_save_time = {}  # model_id -> timestamp
        self._lock = threading.RLock()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model_state(self, model_id: str, model_object: Any, force: bool = False) -> bool:
        """
        Save model state to disk with versioning.
        
        Args:
            model_id: Model identifier
            model_object: Model object to save
            force: Force save even if interval hasn't elapsed
            
        Returns:
            Success flag
        """
        with self._lock:
            current_time = time.time()
            
            # Check save interval
            if not force and model_id in self._last_save_time:
                time_since_last_save = current_time - self._last_save_time[model_id]
                if time_since_last_save < self.save_interval:
                    logger.debug(f"Skipping save for {model_id} - last saved {time_since_last_save:.1f}s ago")
                    return False
        
        try:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_id}_{timestamp}.pkl"
            filepath = os.path.join(self.model_dir, filename)
            
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump(model_object, f)
            
            logger.info(f"Saved model state: {filepath}")
            
            # Update last save time
            with self._lock:
                self._last_save_time[model_id] = current_time
            
            # Cleanup old versions
            self._cleanup_old_versions(model_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model state: {e}", exc_info=True)
            return False
    
    def load_latest_model(self, model_id: str) -> Optional[Any]:
        """
        Load latest model state from disk.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model object or None if not found
        """
        try:
            # Find all matching model files
            model_pattern = f"{model_id}_*.pkl"
            model_files = []
            
            for filename in os.listdir(self.model_dir):
                if filename.startswith(f"{model_id}_") and filename.endswith(".pkl"):
                    filepath = os.path.join(self.model_dir, filename)
                    model_files.append((filepath, os.path.getmtime(filepath)))
            
            if not model_files:
                logger.warning(f"No model states found for {model_id}")
                return None
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x[1], reverse=True)
            latest_filepath = model_files[0][0]
            
            # Load model
            with open(latest_filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model state: {latest_filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model state: {e}", exc_info=True)
            return None
    
    def _cleanup_old_versions(self, model_id: str) -> None:
        """
        Remove old model versions beyond max_versions.
        
        Args:
            model_id: Model identifier
        """
        try:
            # Find all matching model files
            model_files = []
            
            for filename in os.listdir(self.model_dir):
                if filename.startswith(f"{model_id}_") and filename.endswith(".pkl"):
                    filepath = os.path.join(self.model_dir, filename)
                    model_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old versions
            if len(model_files) > self.max_versions:
                for filepath, _ in model_files[self.max_versions:]:
                    os.remove(filepath)
                    logger.debug(f"Removed old model version: {filepath}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")


class FeatureExtractor:
    """Extracts and normalizes features for Q* Learning."""
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize feature extractor.
        
        Args:
            cache_size: Size of LRU cache for feature extraction
        """
        self.cache_size = cache_size
        self._feature_stats = {}  # column -> (min, max, mean, std)
        self._lock = threading.RLock()
        
        # Setup caching
        self.extract_features = lru_cache(maxsize=cache_size)(self._extract_features_impl)
    
    def _extract_features_impl(self, data_key: str, dataframe: pd.DataFrame, 
                            feature_columns: List[str]) -> np.ndarray:
        """
        Extract features from dataframe (for LRU cache).
        
        Args:
            data_key: Cache key
            dataframe: Input dataframe
            feature_columns: Feature column names
            
        Returns:
            Numpy array of features
        """
        # Validate columns
        valid_columns = [col for col in feature_columns if col in dataframe.columns]
        
        if not valid_columns:
            raise ValueError("No valid feature columns found in dataframe")
        
        # Extract features
        feature_values = dataframe[valid_columns].values
        
        # Update feature statistics
        with self._lock:
            for i, col in enumerate(valid_columns):
                if col not in self._feature_stats:
                    # Initialize stats
                    col_values = feature_values[:, i]
                    self._feature_stats[col] = {
                        'min': np.nanmin(col_values),
                        'max': np.nanmax(col_values),
                        'mean': np.nanmean(col_values),
                        'std': np.nanstd(col_values)
                    }
                else:
                    # Incremental update (simple approximation)
                    n_old = self._feature_stats[col].get('count', 1000)
                    n_new = len(feature_values)
                    total_n = n_old + n_new
                    
                    # Update min/max
                    col_values = feature_values[:, i]
                    col_min = np.nanmin(col_values)
                    col_max = np.nanmax(col_values)
                    
                    if col_min < self._feature_stats[col]['min']:
                        self._feature_stats[col]['min'] = col_min
                    
                    if col_max > self._feature_stats[col]['max']:
                        self._feature_stats[col]['max'] = col_max
                    
                    # Update mean/std (approximation)
                    old_mean = self._feature_stats[col]['mean']
                    new_mean = np.nanmean(col_values)
                    updated_mean = (old_mean * n_old + new_mean * n_new) / total_n
                    
                    # Simple combined std approximation
                    old_std = self._feature_stats[col]['std']
                    new_std = np.nanstd(col_values)
                    combined_var = (
                        (n_old * old_std**2 + n_new * new_std**2) / total_n +
                        (n_old * n_new) * (old_mean - new_mean)**2 / (total_n**2)
                    )
                    updated_std = np.sqrt(combined_var)
                    
                    self._feature_stats[col]['mean'] = updated_mean
                    self._feature_stats[col]['std'] = updated_std
                    self._feature_stats[col]['count'] = total_n
        
        return feature_values
    
    def normalize_features(self, features: np.ndarray, feature_columns: List[str],
                         method: str = 'minmax') -> np.ndarray:
        """
        Normalize features based on collected statistics.
        
        Args:
            features: Feature array
            feature_columns: Feature column names
            method: Normalization method ('minmax', 'zscore', or 'robust')
            
        Returns:
            Normalized feature array
        """
        normalized = np.zeros_like(features, dtype=np.float32)
        
        # Normalize each column
        for i, col in enumerate(feature_columns):
            col_values = features[:, i]
            
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                
                if method == 'minmax':
                    # Min-max normalization
                    min_val = stats['min']
                    max_val = stats['max']
                    if max_val > min_val:
                        normalized[:, i] = (col_values - min_val) / (max_val - min_val)
                    else:
                        normalized[:, i] = 0.5  # Default for constant features
                
                elif method == 'zscore':
                    # Z-score normalization
                    mean = stats['mean']
                    std = stats['std']
                    if std > 0:
                        normalized[:, i] = (col_values - mean) / std
                    else:
                        normalized[:, i] = 0.0  # Default for constant features
                
                elif method == 'robust':
                    # Robust scaling (using percentiles would be better but we use min/max)
                    min_val = stats['min']
                    max_val = stats['max']
                    if max_val > min_val:
                        center = (min_val + max_val) / 2
                        scale = (max_val - min_val) / 2
                        normalized[:, i] = (col_values - center) / scale
                    else:
                        normalized[:, i] = 0.0  # Default for constant features
                
                else:
                    # Unknown method, use identity
                    normalized[:, i] = col_values
            else:
                # No stats available, use original values
                normalized[:, i] = col_values
        
        return normalized


class QStarIncrementalLearner:
    """
    Incremental learning component for QStar with offline data integration.
    
    This component handles:
    1. Loading and processing indicator data from CSV files
    2. Incremental updates to Q* Learning models
    3. Model state persistence with versioning
    4. Memory-efficient batch processing
    5. Integration with Bluewolf for monitoring
    """
    
    def __init__(self, 
                 data_dir: str = 'data',
                 model_dir: str = 'models', 
                 learning_rate: float = 0.05,
                 batch_size: int = 100,
                 training_interval: int = 300,  # 5 minutes
                 training_iterations: int = 5,
                 autosave_interval: int = 3600,  # 1 hour
                 enable_river: bool = True,
                 use_quantum_representation: bool = True,
                 log_level: int = logging.INFO):
        """
        Initialize QStar Incremental Learner.
        
        Args:
            data_dir: Directory for data storage
            model_dir: Directory for model storage
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            training_interval: Seconds between training sessions
            training_iterations: Number of batches per training session
            autosave_interval: Seconds between automatic model saves
            enable_river: Whether to enable River ML integration
            use_quantum_representation: Whether to use quantum-inspired representation
            log_level: Logging level
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_interval = training_interval
        self.training_iterations = training_iterations
        self.autosave_interval = autosave_interval
        self.enable_river = enable_river
        self.use_quantum_representation = use_quantum_representation
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Initialize managers
        self.data_manager = DataChunkManager(data_dir=data_dir)
        self.model_manager = ModelStateManager(model_dir=model_dir, save_interval=autosave_interval)
        self.feature_extractor = FeatureExtractor()
        
        # Load required modules
        self._load_required_modules()
        
        # Models state
        self.qstar_agent = None
        self.river_ml = None
        self.qstar_predictor = None
        
        # Training state
        self._training_thread = None
        self._stop_event = threading.Event()
        self._last_update_time = 0
        self._is_training = False
        self._lock = threading.RLock()
        self._metrics = {
            'training_iterations': 0,
            'total_batches': 0,
            'last_training_time': 0,
            'avg_training_duration': 0,
            'avg_batch_size': 0
        }
        
        # Performance monitoring
        self._performance_metrics = {
            'training_durations': deque(maxlen=100),
            'prediction_durations': deque(maxlen=100),
            'batch_counts': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
        
        # Default feature configuration
        self.default_feature_columns = [
            'close', 'volume', 'rsi_14', 'adx', 'ema_50', 'macd',
            'volatility_regime', 'antifragility', 'soc_equilibrium', 'soc_fragility',
            'panarchy_P', 'panarchy_C', 'panarchy_R', 'performance_metric',
            'qerc_trend', 'qerc_momentum', 'qerc_volatility', 'qerc_regime'
        ]
        
        self.default_target_column = 'qerc_trend'
    
    def _load_required_modules(self):
        """Load required modules if available."""
        try:
            # Attempt to import QStar components
            from q_star_learning import SophisticatedQLearningAgent
            self.SophisticatedQLearningAgent = SophisticatedQLearningAgent
            
            if self.enable_river:
                from river_ml import RiverOnlineML
                from qstar_river import QStarTradingPredictor
                
                self.RiverOnlineML = RiverOnlineML
                self.QStarTradingPredictor = QStarTradingPredictor
                self.logger.info("River ML modules loaded successfully")
            else:
                self.RiverOnlineML = None
                self.QStarTradingPredictor = None
                self.logger.info("River ML integration disabled")
                
            self.modules_loaded = True
            
        except ImportError as e:
            self.logger.error(f"Failed to import required modules: {e}")
            self.modules_loaded = False
    
    def initialize(self, qstar_strategy) -> bool:
            """
            Initialize learner with QStar strategy instance.
            
            Args:
                qstar_strategy: QStar strategy instance
                
            Returns:
                Success flag
            """
            if not self.modules_loaded:
                self.logger.error("Required modules not loaded, initialization failed")
                return False
            
            try:
                self.logger.info("Initializing QStar Incremental Learner")
                self.qstar_strategy = qstar_strategy
                
                # Load saved models if available
                loaded = self._load_saved_models()
                
                if not loaded:
                    # Create new models
                    self._create_new_models()
                
                # Initialize training thread
                self._start_training_thread()
                
                # Create directories if they don't exist
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(self.model_dir, exist_ok=True)
                
                # Initialize data from CSV if available
                self._initialize_from_csv()
                
                self.logger.info("QStar Incremental Learner initialized successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error initializing QStar Incremental Learner: {e}", exc_info=True)
                return False
    
    def _load_saved_models(self) -> bool:
        """
        Load saved models from disk.
        
        Returns:
            Success flag
        """
        try:
            # Load QStar agent
            self.qstar_agent = self.model_manager.load_latest_model("qstar_agent")
            
            if self.enable_river:
                # Load River ML
                self.river_ml = self.model_manager.load_latest_model("river_ml")
                
                # Load QStar predictor
                self.qstar_predictor = self.model_manager.load_latest_model("qstar_predictor")
            
            # Check if all required models are loaded
            if self.qstar_agent is None:
                self.logger.warning("QStar agent not found, need to create new models")
                return False
            
            if self.enable_river and (self.river_ml is None or self.qstar_predictor is None):
                self.logger.warning("River models not found, need to create new models")
                return False
            
            self.logger.info("Saved models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading saved models: {e}", exc_info=True)
            return False
    
    def _create_new_models(self) -> None:
        """Create new models for learning."""
        try:
            self.logger.info("Creating new QStar learning models")
            
            # Create QStar agent
            self.qstar_agent = self.SophisticatedQLearningAgent(
                states=200,         # States for market representation
                actions=5,          # Trading actions (buy, sell, hold, increase, reduce)
                learning_rate=self.learning_rate,
                discount_factor=0.95,
                exploration_rate=1.0,
                min_exploration_rate=0.05,
                exploration_decay_rate=0.99,
                use_adaptive_learning_rate=True,
                use_experience_replay=True,
                experience_buffer_size=10000,
                batch_size=64,
                max_episodes=1000,
                max_steps_per_episode=500,
                use_quantum_representation=self.use_quantum_representation
            )
            
            if self.enable_river:
                # Create River ML
                self.river_ml = self.RiverOnlineML(
                    drift_detector_type='adwin',
                    anomaly_detector_type='hst',
                    feature_window=50,
                    drift_sensitivity=0.05,
                    anomaly_threshold=0.75,
                    log_level=self.logger.level
                )
                
                # Create QStar predictor
                self.qstar_predictor = self.QStarTradingPredictor(
                    river_ml=self.river_ml,
                    use_quantum_representation=self.use_quantum_representation,
                    initial_states=200,
                    training_episodes=100
                )
            
            # Save initial models
            self.model_manager.save_model_state("qstar_agent", self.qstar_agent, force=True)
            
            if self.enable_river:
                self.model_manager.save_model_state("river_ml", self.river_ml, force=True)
                self.model_manager.save_model_state("qstar_predictor", self.qstar_predictor, force=True)
            
            self.logger.info("New QStar learning models created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating new models: {e}", exc_info=True)
            raise
    
    def _initialize_from_csv(self) -> None:
        """Initialize from offline data CSV if available."""
        try:
            # Check for tengri_offline_data_ALL.csv in data directory
            csv_path = os.path.join(self.data_dir, "tengri_offline_data_ALL.csv")
            
            if os.path.exists(csv_path):
                self.logger.info(f"Found offline data at {csv_path}, loading...")
                self.data_manager.load_csv_data(csv_path)
                self.logger.info("Offline data loaded successfully")
            else:
                self.logger.info("No offline data found at default location")
                
        except Exception as e:
            self.logger.error(f"Error initializing from CSV: {e}", exc_info=True)
    
    def _start_training_thread(self) -> None:
        """Start background training thread."""
        if self._training_thread is not None and self._training_thread.is_alive():
            self.logger.debug("Training thread already running")
            return
        
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="QStarTrainingThread"
        )
        self._training_thread.start()
        self.logger.info("Training thread started")
    
    def _training_loop(self) -> None:
        """Main training loop in background thread."""
        while not self._stop_event.is_set():
            try:
                # Check if it's time to train
                current_time = time.time()
                time_since_last_update = current_time - self._last_update_time
                
                if time_since_last_update >= self.training_interval:
                    self._perform_training()
                    self._last_update_time = current_time
                
                # Check if it's time to save models
                time_since_last_save = current_time - self._metrics.get('last_save_time', 0)
                if time_since_last_save >= self.autosave_interval:
                    self._save_models()
                    self._metrics['last_save_time'] = current_time
            
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}", exc_info=True)
            
            # Sleep for a while before next check
            for _ in range(min(30, self.training_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
    
    def _perform_training(self) -> None:
        """Perform a training session."""
        if self._is_training:
            self.logger.debug("Training already in progress, skipping")
            return
        
        with self._lock:
            self._is_training = True
        
        try:
            self.logger.info("Starting training session")
            start_time = time.time()
            
            # Track metrics
            total_training_time = 0
            batch_count = 0
            
            # Perform multiple training iterations
            for i in range(self.training_iterations):
                # Train QStar agent
                if self.qstar_agent is not None:
                    agent_success = self._train_agent_batch()
                    if agent_success:
                        batch_count += 1
                
                # Train River ML if enabled
                if self.enable_river and self.river_ml is not None and self.qstar_predictor is not None:
                    river_success = self._train_river_batch()
                    if river_success:
                        batch_count += 1
            
            # Update metrics
            training_duration = time.time() - start_time
            
            with self._lock:
                self._metrics['training_iterations'] += 1
                self._metrics['total_batches'] += batch_count
                self._metrics['last_training_time'] = time.time()
                self._metrics['avg_batch_size'] = (
                    (self._metrics['avg_batch_size'] * (self._metrics['training_iterations'] - 1) + 
                     self.batch_size) / self._metrics['training_iterations']
                )
                
                # Track performance metrics
                self._performance_metrics['training_durations'].append(training_duration)
                self._performance_metrics['batch_counts'].append(batch_count)
                
                # Estimate memory usage
                try:
                    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                    self._performance_metrics['memory_usage'].append(memory_usage)
                except Exception:
                    pass
                
                self._metrics['avg_training_duration'] = sum(self._performance_metrics['training_durations']) / len(self._performance_metrics['training_durations'])
            
            self.logger.info(f"Training session completed in {training_duration:.2f}s with {batch_count} batches")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
        
        finally:
            # Run garbage collection after training
            gc.collect()
            
            with self._lock:
                self._is_training = False
    
    def _train_agent_batch(self) -> bool:
        """
        Train QStar agent with a batch of data.
        
        Returns:
            Success flag
        """
        try:
            # Get batch of data
            features, targets = self.data_manager.get_next_batch(
                batch_size=self.batch_size,
                feature_columns=self.default_feature_columns,
                target_column=self.default_target_column
            )
            
            if features is None or targets is None:
                self.logger.debug("No data available for agent training")
                return False
            
            # Check data size
            if len(features) < 10:  # Minimum batch size
                self.logger.debug(f"Batch too small for agent training: {len(features)} < 10")
                return False
            
            # Process data for training
            states = self._features_to_states(features)
            
            # Target drives the reward function
            targets = targets.values
            
            # Simple environment simulation for agent training
            current_state = states[0]
            
            for i in range(1, len(states)):
                # Choose action based on current state
                action = self.qstar_agent.choose_action(current_state)
                
                # Determine reward based on action and target
                reward = self._calculate_reward(action, targets[i-1], targets[i])
                
                # Learn from state-action-reward-next_state
                next_state = states[i]
                self.qstar_agent.learn(current_state, action, reward, next_state)
                
                # Move to next state
                current_state = next_state
            
            # Run experience replay if available
            if hasattr(self.qstar_agent, 'replay_experiences'):
                self.qstar_agent.replay_experiences()
            
            # Update exploration rate
            self.qstar_agent.update_exploration_rate()
            
            self.logger.debug(f"Trained QStar agent on batch of {len(features)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training agent batch: {e}", exc_info=True)
            return False
    
    def _train_river_batch(self) -> bool:
        """
        Train River ML with a batch of data.
        
        Returns:
            Success flag
        """
        if not self.enable_river or self.river_ml is None:
            return False
        
        try:
            # Get batch of data
            features, targets = self.data_manager.get_next_batch(
                batch_size=self.batch_size,
                feature_columns=self.default_feature_columns,
                target_column=self.default_target_column
            )
            
            if features is None or targets is None:
                self.logger.debug("No data available for River training")
                return False
            
            # Check data size
            if len(features) < 10:  # Minimum batch size
                self.logger.debug(f"Batch too small for River training: {len(features)} < 10")
                return False
            
            # Update River ML components
            for i in range(len(features)):
                # Extract feature row
                feature_row = features.iloc[i].to_dict()
                
                # Extract target
                target_value = float(targets.iloc[i])
                
                # Update drift detector
                self.river_ml.detect_drift(target_value)
                
                # Update anomaly detector
                self.river_ml.detect_anomalies(feature_row)
            
            # Update QStar predictor with River ML
            if self.qstar_predictor is not None:
                # Prepare sample data
                sample_data = pd.concat([features, targets], axis=1)
                
                # Update QStar predictor
                self.qstar_predictor.river_ml = self.river_ml
            
            self.logger.debug(f"Trained River ML on batch of {len(features)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training River batch: {e}", exc_info=True)
            return False
    
    def _features_to_states(self, features: pd.DataFrame) -> List[int]:
        """
        Convert features to state indices for Q* Learning.
        
        Args:
            features: Feature dataframe
            
        Returns:
            List of state indices
        """
        try:
            if features.empty:
                return []
            
            # Extract feature values
            feature_data = features.values
            
            # Normalize features
            normalized = self.feature_extractor.normalize_features(
                feature_data, 
                features.columns.tolist(),
                method='minmax'
            )
            
            # Convert to states (simple discretization)
            states = []
            n_features = normalized.shape[1]
            n_states = 200  # Match agent's state space
            
            for i in range(len(normalized)):
                # Convert normalized features to a single state index
                # This is a simple hash-based approach
                feature_vec = normalized[i, :]
                
                # Discretize each feature (e.g., to 3 levels) and combine
                discrete_features = (feature_vec * 3).astype(int).clip(0, 2)
                
                # Calculate state index using base-3 encoding
                state_idx = sum(discrete_features[j] * (3 ** j) for j in range(n_features)) % n_states
                states.append(int(state_idx))
            
            return states
            
        except Exception as e:
            self.logger.error(f"Error converting features to states: {e}", exc_info=True)
            return [0] * len(features)  # Default to state 0
    
    def _calculate_reward(self, action: int, current_target: float, next_target: float) -> float:
        """
        Calculate reward for action based on targets.
        
        Args:
            action: Action taken (0=buy, 1=sell, 2=hold, 3=reduce, 4=increase)
            current_target: Current target value
            next_target: Next target value
            
        Returns:
            Reward value
        """
        try:
            # Calculate target change
            target_change = next_target - current_target
            
            # Define actions
            BUY = 0
            SELL = 1
            HOLD = 2
            REDUCE = 3
            INCREASE = 4
            
            # Base reward is target change (positive = good, negative = bad)
            base_reward = target_change * 10  # Scale for better learning
            
            # Modify reward based on action and target change
            if target_change > 0:
                # Price is going up - good for buy/increase, bad for sell/reduce
                if action in [BUY, INCREASE]:
                    reward = base_reward * 1.5  # Bonus for correct action
                elif action in [SELL, REDUCE]:
                    reward = -base_reward  # Penalty for incorrect action
                else:  # HOLD
                    reward = base_reward * 0.5  # Partial reward for hold
            
            elif target_change < 0:
                # Price is going down - good for sell/reduce, bad for buy/increase
                if action in [SELL, REDUCE]:
                    reward = -base_reward * 1.5  # Bonus for correct action (negative * -1.5 = positive)
                elif action in [BUY, INCREASE]:
                    reward = base_reward  # Penalty for incorrect action (negative stays negative)
                else:  # HOLD
                    reward = base_reward * 0.5  # Partial penalty for hold (smaller negative)
            
            else:
                # No change - HOLD is best
                if action == HOLD:
                    reward = 0.1  # Small positive reward for correct action
                else:
                    reward = -0.1  # Small negative reward for unnecessary action
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}", exc_info=True)
            return 0.0  # Neutral reward on error
    
    def predict(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading prediction from current models.
        
        Args:
            dataframe: Input dataframe with features
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Check if models are available
            if self.qstar_agent is None:
                return {
                    'error': 'QStar agent not initialized',
                    'action': 2,  # Default to HOLD
                    'action_name': 'HOLD',
                    'confidence': 0.0
                }
            
            # Extract features
            valid_features = [col for col in self.default_feature_columns if col in dataframe.columns]
            
            if not valid_features:
                return {
                    'error': 'No valid features found in dataframe',
                    'action': 2,  # Default to HOLD
                    'action_name': 'HOLD',
                    'confidence': 0.0
                }
            
            # Prepare feature subset
            features = dataframe[valid_features]
            
            # Convert to state
            states = self._features_to_states(features)
            
            if not states:
                return {
                    'error': 'Failed to convert features to states',
                    'action': 2,  # Default to HOLD
                    'action_name': 'HOLD',
                    'confidence': 0.0
                }
            
            # Use last state for prediction
            current_state = states[-1]
            
            # Get Q-values for current state
            q_values = self.qstar_agent.q_table[current_state, :]
            
            # Choose action with highest Q-value
            action = int(np.argmax(q_values))
            
            # Determine confidence based on Q-value separation
            sorted_q_values = np.sort(q_values)[::-1]  # Descending order
            if len(sorted_q_values) > 1:
                max_separation = sorted_q_values[0] - sorted_q_values[1]
                confidence = min(0.95, max(0.5, max_separation))
            else:
                confidence = 0.5
            
            # Define action names for logging
            action_names = ['BUY', 'SELL', 'HOLD', 'REDUCE', 'INCREASE']
            action_name = action_names[action] if 0 <= action < len(action_names) else 'UNKNOWN'
            
            # Add prediction from River ML if available
            river_prediction = None
            if self.enable_river and self.qstar_predictor is not None:
                try:
                    river_prediction = self.qstar_predictor.predict(dataframe)
                except Exception as e:
                    self.logger.error(f"Error getting River prediction: {e}")
            
            # Merge predictions
            prediction = {
                'action': action,
                'action_name': action_name,
                'confidence': float(confidence),
                'q_values': q_values.tolist(),
                'river_prediction': river_prediction
            }
            
            # Track performance
            prediction_duration = time.time() - start_time
            with self._lock:
                self._performance_metrics['prediction_durations'].append(prediction_duration)
            
            self.logger.debug(f"Generated prediction: {action_name} with confidence {confidence:.2f} in {prediction_duration:.3f}s")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}", exc_info=True)
            return {
                'error': str(e),
                'action': 2,  # Default to HOLD
                'action_name': 'HOLD',
                'confidence': 0.0
            }
    
    def _save_models(self) -> None:
        """Save current model state."""
        try:
            self.logger.info("Saving model state")
            
            # Save QStar agent
            if self.qstar_agent is not None:
                self.model_manager.save_model_state("qstar_agent", self.qstar_agent)
            
            # Save River ML if enabled
            if self.enable_river:
                if self.river_ml is not None:
                    self.model_manager.save_model_state("river_ml", self.river_ml)
                
                if self.qstar_predictor is not None:
                    self.model_manager.save_model_state("qstar_predictor", self.qstar_predictor)
            
            self.logger.info("Model state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model state: {e}", exc_info=True)
    
    def shutdown(self) -> None:
        """Clean shutdown of learner."""
        try:
            self.logger.info("Shutting down QStar Incremental Learner")
            
            # Stop training thread
            self._stop_event.set()
            
            if self._training_thread is not None and self._training_thread.is_alive():
                self._training_thread.join(timeout=30)
            
            # Save models before exit
            self._save_models()
            
            self.logger.info("QStar Incremental Learner shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check health status for Bluewolf monitoring.
        
        Returns:
            Health status dictionary
        """
        try:
            status = {
                'healthy': True,
                'components': {},
                'metrics': {},
                'errors': []
            }
            
            # Check component health
            if self.qstar_agent is None:
                status['healthy'] = False
                status['errors'].append("QStar agent not initialized")
                status['components']['qstar_agent'] = False
            else:
                status['components']['qstar_agent'] = True
            
            if self.enable_river:
                if self.river_ml is None:
                    status['components']['river_ml'] = False
                    status['errors'].append("River ML not initialized")
                else:
                    status['components']['river_ml'] = True
                
                if self.qstar_predictor is None:
                    status['components']['qstar_predictor'] = False
                    status['errors'].append("QStar predictor not initialized")
                else:
                    status['components']['qstar_predictor'] = True
            
            # Check thread health
            if self._training_thread is None or not self._training_thread.is_alive():
                status['healthy'] = False
                status['errors'].append("Training thread not running")
                status['components']['training_thread'] = False
            else:
                status['components']['training_thread'] = True
            
            # Add metrics
            with self._lock:
                status['metrics'] = {
                    'training_iterations': self._metrics.get('training_iterations', 0),
                    'total_batches': self._metrics.get('total_batches', 0),
                    'avg_training_duration': self._metrics.get('avg_training_duration', 0),
                    'last_training_time': self._metrics.get('last_training_time', 0),
                    'avg_batch_size': self._metrics.get('avg_batch_size', 0),
                    'avg_prediction_time': (
                        sum(self._performance_metrics['prediction_durations']) / 
                        len(self._performance_metrics['prediction_durations'])
                        if self._performance_metrics['prediction_durations'] else 0
                    )
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}", exc_info=True)
            return {
                'healthy': False,
                'errors': [str(e)],
                'components': {},
                'metrics': {}
            }
    
    def recover(self) -> bool:
        """
        Recovery method for Bluewolf integration.
        
        Returns:
            Success flag
        """
        try:
            self.logger.info("Attempting recovery of QStar Incremental Learner")
            
            # Stop training thread if running
            self._stop_event.set()
            
            if self._training_thread is not None and self._training_thread.is_alive():
                self._training_thread.join(timeout=10)
            
            # Reset flags
            self._is_training = False
            self._stop_event.clear()
            
            # Attempt to reload models
            models_reloaded = self._load_saved_models()
            
            if not models_reloaded:
                # Create new models if reload failed
                self._create_new_models()
            
            # Restart training thread
            self._start_training_thread()
            
            self.logger.info("QStar Incremental Learner recovery successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recovering QStar Incremental Learner: {e}", exc_info=True)
            return False