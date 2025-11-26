"""
Neural Model Manager for NHITS Forecasting Models.

This module provides comprehensive model lifecycle management including:
- Model training orchestration
- Version control and model registry
- Performance monitoring and evaluation
- Automated retraining workflows
- Model deployment and serving
"""

import asyncio
import logging
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import hashlib

from .nhits_forecaster import NHITSForecaster


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    r2_score: float
    training_time: float
    prediction_time: float
    timestamp: str


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_id: str
    created_at: str
    model_path: str
    metrics: ModelMetrics
    config: Dict[str, Any]
    status: str  # 'training', 'active', 'deprecated', 'failed'
    notes: str = ""


class NeuralModelManager:
    """
    Comprehensive neural model lifecycle management system.
    
    Features:
    - Model version control and registry
    - Automated training and retraining
    - Performance monitoring and evaluation
    - Model deployment and serving
    - A/B testing support
    - Model rollback capabilities
    """
    
    def __init__(
        self,
        model_registry_path: str = "models/neural_forecast",
        auto_retrain_enabled: bool = True,
        performance_threshold: float = 0.85,
        max_model_versions: int = 10,
        enable_monitoring: bool = True
    ):
        """
        Initialize Neural Model Manager.
        
        Args:
            model_registry_path: Path to model registry directory
            auto_retrain_enabled: Enable automatic retraining
            performance_threshold: Performance threshold for retraining trigger
            max_model_versions: Maximum number of model versions to keep
            enable_monitoring: Enable performance monitoring
        """
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.auto_retrain_enabled = auto_retrain_enabled
        self.performance_threshold = performance_threshold
        self.max_model_versions = max_model_versions
        self.enable_monitoring = enable_monitoring
        
        # Initialize registry database
        self.registry_db_path = self.registry_path / "model_registry.json"
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.active_models: Dict[str, str] = {}  # model_id -> version
        
        # Load existing registry
        self._load_registry()
        
        # Initialize forecaster
        self.forecaster = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        self.monitoring_enabled = enable_monitoring
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.logger.info(f"Neural Model Manager initialized with registry at {self.registry_path}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_registry(self):
        """Load model registry from disk."""
        try:
            if self.registry_db_path.exists():
                with open(self.registry_db_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Convert dictionaries back to ModelVersion objects
                for model_id, versions in registry_data.get('model_versions', {}).items():
                    self.model_versions[model_id] = []
                    for version_data in versions:
                        # Convert metrics dict to ModelMetrics
                        metrics_data = version_data['metrics']
                        metrics = ModelMetrics(**metrics_data)
                        
                        # Create ModelVersion with converted metrics
                        version_data['metrics'] = metrics
                        model_version = ModelVersion(**version_data)
                        self.model_versions[model_id].append(model_version)
                
                self.active_models = registry_data.get('active_models', {})
                self.performance_history = registry_data.get('performance_history', {})
                
                self.logger.info(f"Loaded registry with {len(self.model_versions)} model families")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {str(e)}")
            self.model_versions = {}
            self.active_models = {}
            self.performance_history = {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        try:
            registry_data = {
                'model_versions': {},
                'active_models': self.active_models,
                'performance_history': self.performance_history,
                'updated_at': datetime.now().isoformat()
            }
            
            # Convert ModelVersion objects to dictionaries
            for model_id, versions in self.model_versions.items():
                registry_data['model_versions'][model_id] = []
                for version in versions:
                    version_dict = asdict(version)
                    # Convert ModelMetrics to dict
                    version_dict['metrics'] = asdict(version.metrics)
                    registry_data['model_versions'][model_id].append(version_dict)
            
            with open(self.registry_db_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            self.logger.debug("Registry saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save registry: {str(e)}")
    
    def _generate_model_id(self, config: Dict[str, Any]) -> str:
        """Generate unique model ID based on configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _generate_version(self, model_id: str) -> str:
        """Generate next version number for model."""
        if model_id not in self.model_versions:
            return "v1.0.0"
        
        versions = self.model_versions[model_id]
        if not versions:
            return "v1.0.0"
        
        # Extract version numbers and increment
        latest_version = max(versions, key=lambda v: v.created_at)
        version_parts = latest_version.version.replace('v', '').split('.')
        major, minor, patch = map(int, version_parts)
        
        return f"v{major}.{minor}.{patch + 1}"
    
    async def train_model(
        self,
        data: Union[pd.DataFrame, Dict],
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "nhits_default",
        validation_data: Optional[pd.DataFrame] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Train a new NHITS model with comprehensive tracking.
        
        Args:
            data: Training data
            config: Model configuration
            model_name: Human-readable model name
            validation_data: Optional validation data
            notes: Training notes
            
        Returns:
            Training results with model version info
        """
        self.logger.info(f"Starting model training for {model_name}")
        
        try:
            # Use default config if not provided
            if config is None:
                config = {
                    'input_size': 24,
                    'horizon': 12,
                    'batch_size': 32,
                    'max_epochs': 100,
                    'learning_rate': 1e-3,
                    'enable_gpu': True
                }
            
            # Generate model ID and version
            model_id = self._generate_model_id(config)
            version = self._generate_version(model_id)
            
            # Initialize forecaster
            self.forecaster = NHITSForecaster(**config)
            
            # Train model
            start_time = datetime.now()
            
            training_result = await self.forecaster.fit(
                data=data,
                val_data=validation_data,
                verbose=True
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            if not training_result['success']:
                raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
            
            # Evaluate model performance
            metrics = await self._evaluate_model(data, validation_data)
            
            # Save model to registry
            model_path = await self._save_model_to_registry(
                model_id, version, config, metrics, notes
            )
            
            # Create model version record
            model_version = ModelVersion(
                version=version,
                model_id=model_id,
                created_at=datetime.now().isoformat(),
                model_path=model_path,
                metrics=metrics,
                config=config,
                status='active',
                notes=notes
            )
            
            # Update registry
            if model_id not in self.model_versions:
                self.model_versions[model_id] = []
            
            self.model_versions[model_id].append(model_version)
            self.active_models[model_name] = version
            
            # Cleanup old versions if needed
            await self._cleanup_old_versions(model_id)
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"Model training completed: {model_name} {version}")
            
            return {
                'success': True,
                'model_id': model_id,
                'version': version,
                'model_name': model_name,
                'training_time': training_time,
                'metrics': asdict(metrics),
                'model_path': model_path
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    async def _evaluate_model(
        self,
        train_data: Union[pd.DataFrame, Dict],
        val_data: Optional[pd.DataFrame] = None
    ) -> ModelMetrics:
        """
        Evaluate model performance on validation data.
        
        Args:
            train_data: Training data for context
            val_data: Validation data
            
        Returns:
            Performance metrics
        """
        try:
            # Use last portion of training data if no validation data
            if val_data is None:
                if isinstance(train_data, dict):
                    df = pd.DataFrame(train_data)
                else:
                    df = train_data.copy()
                
                # Use last 20% for validation
                split_idx = int(len(df) * 0.8)
                val_data = df.iloc[split_idx:].copy()
                eval_data = df.iloc[:split_idx].copy()
            else:
                eval_data = train_data
            
            # Generate predictions
            start_time = datetime.now()
            prediction_result = await self.forecaster.predict(eval_data)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            if not prediction_result['success']:
                raise Exception("Prediction failed during evaluation")
            
            # Calculate metrics
            y_true = val_data['y'].values
            y_pred = np.array(prediction_result['point_forecast'])
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate error metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Percentage errors
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))
            
            return ModelMetrics(
                mae=float(mae),
                mse=float(mse),
                rmse=float(rmse),
                mape=float(mape),
                smape=float(smape),
                r2_score=float(r2_score),
                training_time=0.0,  # Set by caller
                prediction_time=prediction_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            # Return default metrics
            return ModelMetrics(
                mae=float('inf'),
                mse=float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                smape=float('inf'),
                r2_score=-1.0,
                training_time=0.0,
                prediction_time=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    async def _save_model_to_registry(
        self,
        model_id: str,
        version: str,
        config: Dict,
        metrics: ModelMetrics,
        notes: str
    ) -> str:
        """
        Save model to registry with proper organization.
        
        Args:
            model_id: Unique model identifier
            version: Model version
            config: Model configuration
            metrics: Performance metrics
            notes: Training notes
            
        Returns:
            Path where model was saved
        """
        # Create model directory
        model_dir = self.registry_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        self.forecaster.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'model_id': model_id,
            'version': version,
            'config': config,
            'metrics': asdict(metrics),
            'notes': notes,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(model_path)
    
    async def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Model name to load
            version: Specific version (latest if None)
            
        Returns:
            Load result with model info
        """
        try:
            # Get model version info
            if model_name not in self.active_models:
                raise ValueError(f"Model {model_name} not found in registry")
            
            if version is None:
                version = self.active_models[model_name]
            
            # Find model version
            model_version = None
            for model_id, versions in self.model_versions.items():
                for v in versions:
                    if v.version == version:
                        model_version = v
                        break
                if model_version:
                    break
            
            if not model_version:
                raise ValueError(f"Version {version} not found")
            
            # Load model
            self.forecaster = NHITSForecaster(**model_version.config)
            
            if not self.forecaster.load_model(model_version.model_path):
                raise Exception("Failed to load model from file")
            
            self.logger.info(f"Loaded model {model_name} {version}")
            
            return {
                'success': True,
                'model_name': model_name,
                'version': version,
                'model_id': model_version.model_id,
                'metrics': asdict(model_version.metrics),
                'config': model_version.config
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        model_name: Optional[str] = None,
        return_intervals: bool = True
    ) -> Dict[str, Any]:
        """
        Generate predictions using loaded model.
        
        Args:
            data: Input data
            model_name: Model to use (current if None)
            return_intervals: Include prediction intervals
            
        Returns:
            Prediction results
        """
        if self.forecaster is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        try:
            result = await self.forecaster.predict(
                data=data,
                return_intervals=return_intervals
            )
            
            # Track prediction for monitoring
            if self.monitoring_enabled and model_name:
                await self._track_prediction(model_name, data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _track_prediction(
        self,
        model_name: str,
        input_data: Union[pd.DataFrame, Dict],
        prediction_result: Dict
    ):
        """Track prediction for monitoring purposes."""
        try:
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'input_size': len(input_data) if isinstance(input_data, (pd.DataFrame, dict)) else 0,
                'prediction_time': prediction_result.get('prediction_time', 0),
                'success': prediction_result.get('success', False)
            }
            
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            self.performance_history[model_name].append(prediction_log)
            
            # Keep only recent predictions (last 1000)
            if len(self.performance_history[model_name]) > 1000:
                self.performance_history[model_name] = self.performance_history[model_name][-1000:]
            
        except Exception as e:
            self.logger.warning(f"Failed to track prediction: {str(e)}")
    
    async def _cleanup_old_versions(self, model_id: str):
        """Remove old model versions to save space."""
        if model_id not in self.model_versions:
            return
        
        versions = self.model_versions[model_id]
        if len(versions) <= self.max_model_versions:
            return
        
        # Sort by creation date and keep the newest ones
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        # Remove old versions
        old_versions = versions[self.max_model_versions:]
        for version in old_versions:
            try:
                # Remove model files
                model_path = Path(version.model_path)
                if model_path.parent.exists():
                    shutil.rmtree(model_path.parent)
                
                self.logger.info(f"Removed old model version {version.version}")
                
            except Exception as e:
                self.logger.warning(f"Failed to remove old version {version.version}: {str(e)}")
        
        # Update registry
        self.model_versions[model_id] = versions[:self.max_model_versions]
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all models in the registry.
        
        Returns:
            Model registry information
        """
        models_info = {}
        
        for model_name, active_version in self.active_models.items():
            # Find model info
            model_info = None
            for model_id, versions in self.model_versions.items():
                for version in versions:
                    if version.version == active_version:
                        model_info = {
                            'model_id': model_id,
                            'version': version.version,
                            'status': version.status,
                            'created_at': version.created_at,
                            'metrics': asdict(version.metrics),
                            'notes': version.notes
                        }
                        break
                if model_info:
                    break
            
            models_info[model_name] = model_info
        
        return {
            'active_models': models_info,
            'total_model_families': len(self.model_versions),
            'total_versions': sum(len(versions) for versions in self.model_versions.values()),
            'registry_path': str(self.registry_path)
        }
    
    def get_model_performance_history(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance history for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Performance history
        """
        if model_name not in self.performance_history:
            return {'model_name': model_name, 'history': [], 'summary': {}}
        
        history = self.performance_history[model_name]
        
        # Calculate summary statistics
        successful_predictions = [h for h in history if h.get('success', False)]
        
        summary = {
            'total_predictions': len(history),
            'successful_predictions': len(successful_predictions),
            'success_rate': len(successful_predictions) / len(history) if history else 0,
            'avg_prediction_time': np.mean([h.get('prediction_time', 0) for h in successful_predictions]) if successful_predictions else 0,
            'last_prediction': history[-1]['timestamp'] if history else None
        }
        
        return {
            'model_name': model_name,
            'history': history[-100:],  # Return last 100 predictions
            'summary': summary
        }
    
    async def retrain_if_needed(
        self,
        model_name: str,
        new_data: Union[pd.DataFrame, Dict],
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Retrain model if performance has degraded.
        
        Args:
            model_name: Model to check
            new_data: New training data
            force_retrain: Force retraining regardless of performance
            
        Returns:
            Retraining result
        """
        if not self.auto_retrain_enabled and not force_retrain:
            return {'retrain_needed': False, 'reason': 'Auto-retrain disabled'}
        
        try:
            # Check if retraining is needed
            performance_history = self.get_model_performance_history(model_name)
            
            if not force_retrain:
                success_rate = performance_history['summary'].get('success_rate', 1.0)
                
                if success_rate >= self.performance_threshold:
                    return {
                        'retrain_needed': False,
                        'reason': f'Performance acceptable: {success_rate:.2%}'
                    }
            
            # Retrain model
            self.logger.info(f"Retraining model {model_name}")
            
            # Get current model config
            model_info = self.list_models()['active_models'].get(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not found")
            
            # Find current config
            current_config = None
            for model_id, versions in self.model_versions.items():
                for version in versions:
                    if version.version == model_info['version']:
                        current_config = version.config
                        break
                if current_config:
                    break
            
            if not current_config:
                raise ValueError("Current model configuration not found")
            
            # Train new version
            retrain_result = await self.train_model(
                data=new_data,
                config=current_config,
                model_name=model_name,
                notes=f"Automatic retraining due to performance degradation"
            )
            
            return {
                'retrain_needed': True,
                'retrain_result': retrain_result,
                'reason': 'Performance below threshold' if not force_retrain else 'Forced retrain'
            }
            
        except Exception as e:
            self.logger.error(f"Retraining failed: {str(e)}")
            return {
                'retrain_needed': True,
                'retrain_result': {'success': False, 'error': str(e)},
                'reason': 'Retraining failed'
            }
    
    def export_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Export model configuration for reproducibility.
        
        Args:
            model_name: Model name
            
        Returns:
            Model configuration
        """
        try:
            models_info = self.list_models()['active_models']
            
            if model_name not in models_info:
                raise ValueError(f"Model {model_name} not found")
            
            model_info = models_info[model_name]
            
            # Find full config
            for model_id, versions in self.model_versions.items():
                for version in versions:
                    if version.version == model_info['version']:
                        return {
                            'model_name': model_name,
                            'version': version.version,
                            'config': version.config,
                            'metrics': asdict(version.metrics),
                            'created_at': version.created_at,
                            'notes': version.notes
                        }
            
            raise ValueError("Model configuration not found in registry")
            
        except Exception as e:
            self.logger.error(f"Failed to export config: {str(e)}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model manager.
        
        Returns:
            Health status
        """
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'registry_path': str(self.registry_path),
                'registry_exists': self.registry_path.exists(),
                'active_models': len(self.active_models),
                'total_versions': sum(len(versions) for versions in self.model_versions.values()),
                'auto_retrain_enabled': self.auto_retrain_enabled,
                'monitoring_enabled': self.monitoring_enabled,
                'performance_threshold': self.performance_threshold
            }
            
            # Check if forecaster is loaded
            health_status['forecaster_loaded'] = self.forecaster is not None
            if self.forecaster:
                health_status['forecaster_fitted'] = self.forecaster.is_fitted
            
            # Check registry integrity
            try:
                self._save_registry()
                health_status['registry_writable'] = True
            except Exception:
                health_status['registry_writable'] = False
                health_status['status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }