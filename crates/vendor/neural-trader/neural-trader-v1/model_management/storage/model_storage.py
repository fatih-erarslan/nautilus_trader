"""Advanced Model Storage System with Versioning and Compression."""

import pickle
import joblib
import json
import gzip
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    JSON = "json"
    COMPRESSED_PICKLE = "cpickle"
    COMPRESSED_JOBLIB = "cjoblib"


class CompressionLevel(Enum):
    """Compression levels for model storage."""
    NONE = 0
    LOW = 3
    MEDIUM = 6
    HIGH = 9


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    name: str
    version: str
    created_at: datetime
    model_type: str
    strategy_name: str
    performance_metrics: Dict[str, float]
    parameters: Dict[str, Any]
    training_data_hash: Optional[str] = None
    file_size_bytes: int = 0
    compression_ratio: float = 1.0
    storage_format: str = "pickle"
    tags: List[str] = None
    description: str = ""
    author: str = "AI Trading System"
    validation_passed: bool = True
    benchmark_results: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.benchmark_results is None:
            self.benchmark_results = {}
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelStorage:
    """Advanced model storage system with versioning and optimization."""
    
    def __init__(self, base_path: str = "model_management/models", 
                 auto_cleanup: bool = True, max_versions: int = 10):
        """
        Initialize model storage system.
        
        Args:
            base_path: Base directory for model storage
            auto_cleanup: Enable automatic cleanup of old versions
            max_versions: Maximum versions to keep per model
        """
        self.base_path = Path(base_path)
        self.auto_cleanup = auto_cleanup
        self.max_versions = max_versions
        self._lock = threading.Lock()
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize model registry
        self.registry_file = self.base_path / "registry.json"
        self._model_registry = self._load_registry()
        
        logger.info(f"Model storage initialized at {self.base_path}")
    
    def _setup_directories(self):
        """Setup directory structure for model storage."""
        directories = [
            self.base_path,
            self.base_path / "optimized_models",
            self.base_path / "backup",
            self.base_path / "versions",
            self.base_path / "metadata",
            self.base_path / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self) -> Dict:
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        with self._lock:
            try:
                with open(self.registry_file, 'w') as f:
                    json.dump(self._model_registry, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save registry: {e}")
    
    def _generate_model_id(self, model_data: Any, metadata: ModelMetadata) -> str:
        """Generate unique model ID based on content and metadata."""
        # Create hash from model data and key parameters
        content_str = f"{metadata.name}_{metadata.strategy_name}_{metadata.version}"
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{metadata.strategy_name}_{content_hash}_{timestamp}"
    
    def _get_model_path(self, model_id: str, version: str = None) -> Path:
        """Get file path for model storage."""
        if version:
            return self.base_path / "versions" / f"{model_id}_v{version}"
        return self.base_path / "optimized_models" / model_id
    
    def _get_metadata_path(self, model_id: str) -> Path:
        """Get metadata file path."""
        return self.base_path / "metadata" / f"{model_id}_metadata.json"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def save_model(self, model: Any, metadata: ModelMetadata, 
                   format_type: ModelFormat = ModelFormat.COMPRESSED_PICKLE,
                   compression_level: CompressionLevel = CompressionLevel.MEDIUM) -> str:
        """
        Save model with comprehensive metadata and versioning.
        
        Args:
            model: Model object to save
            metadata: Model metadata
            format_type: Serialization format
            compression_level: Compression level for storage
            
        Returns:
            Model ID for the saved model
        """
        with self._lock:
            # Generate model ID
            model_id = self._generate_model_id(model, metadata)
            metadata.model_id = model_id
            
            # Determine file paths
            model_path = self._get_model_path(model_id)
            metadata_path = self._get_metadata_path(model_id)
            
            try:
                # Save model with appropriate format
                original_size = self._save_model_data(model, model_path, format_type, compression_level)
                
                # Calculate compression ratio
                compressed_size = model_path.stat().st_size
                metadata.compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                metadata.file_size_bytes = compressed_size
                metadata.storage_format = format_type.value
                
                # Save metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2, default=str)
                
                # Update registry
                self._model_registry[model_id] = {
                    'name': metadata.name,
                    'version': metadata.version,
                    'created_at': metadata.created_at.isoformat(),
                    'strategy_name': metadata.strategy_name,
                    'performance_metrics': metadata.performance_metrics,
                    'file_path': str(model_path),
                    'metadata_path': str(metadata_path),
                    'tags': metadata.tags
                }
                
                self._save_registry()
                
                # Perform auto-cleanup if enabled
                if self.auto_cleanup:
                    self._cleanup_old_versions(metadata.strategy_name)
                
                logger.info(f"Model saved: {model_id} (compression: {metadata.compression_ratio:.2f}x)")
                return model_id
                
            except Exception as e:
                logger.error(f"Failed to save model {model_id}: {e}")
                # Cleanup partial files
                for path in [model_path, metadata_path]:
                    if path.exists():
                        path.unlink()
                raise
    
    def _save_model_data(self, model: Any, model_path: Path, 
                        format_type: ModelFormat, compression_level: CompressionLevel) -> int:
        """Save model data with specified format and compression."""
        original_size = 0
        
        if format_type == ModelFormat.PICKLE:
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = f.tell()
        
        elif format_type == ModelFormat.JOBLIB:
            joblib.dump(model, model_path.with_suffix('.joblib'))
            original_size = model_path.with_suffix('.joblib').stat().st_size
        
        elif format_type == ModelFormat.JSON:
            with open(model_path.with_suffix('.json'), 'w') as f:
                json.dump(model, f, indent=2, default=str)
                original_size = f.tell()
        
        elif format_type == ModelFormat.COMPRESSED_PICKLE:
            # First save to pickle, then compress
            temp_path = model_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = f.tell()
            
            # Compress the pickle file
            with open(temp_path, 'rb') as f_in:
                with gzip.open(model_path.with_suffix('.cpkl'), 'wb', 
                             compresslevel=compression_level.value) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            temp_path.unlink()  # Remove temporary file
        
        elif format_type == ModelFormat.COMPRESSED_JOBLIB:
            # Save with joblib compression
            joblib.dump(model, model_path.with_suffix('.cjoblib'), 
                       compress=compression_level.value)
            original_size = model_path.with_suffix('.cjoblib').stat().st_size * 2  # Estimate
        
        return original_size
    
    def load_model(self, model_id: str, version: str = None) -> Tuple[Any, ModelMetadata]:
        """
        Load model and metadata by ID.
        
        Args:
            model_id: Model identifier
            version: Specific version to load (latest if None)
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self._get_model_path(model_id, version)
        metadata_path = self._get_metadata_path(model_id)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for model {model_id}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Determine actual model file path based on format
        format_type = ModelFormat(metadata.storage_format)
        actual_model_path = self._get_actual_model_path(model_path, format_type)
        
        if not actual_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {actual_model_path}")
        
        # Load model based on format
        model = self._load_model_data(actual_model_path, format_type)
        
        logger.info(f"Model loaded: {model_id}")
        return model, metadata
    
    def _get_actual_model_path(self, base_path: Path, format_type: ModelFormat) -> Path:
        """Get actual model file path based on format."""
        suffixes = {
            ModelFormat.PICKLE: '.pkl',
            ModelFormat.JOBLIB: '.joblib',
            ModelFormat.JSON: '.json',
            ModelFormat.COMPRESSED_PICKLE: '.cpkl',
            ModelFormat.COMPRESSED_JOBLIB: '.cjoblib'
        }
        return base_path.with_suffix(suffixes[format_type])
    
    def _load_model_data(self, model_path: Path, format_type: ModelFormat) -> Any:
        """Load model data based on format."""
        if format_type == ModelFormat.PICKLE:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        elif format_type == ModelFormat.JOBLIB:
            return joblib.load(model_path)
        
        elif format_type == ModelFormat.JSON:
            with open(model_path, 'r') as f:
                return json.load(f)
        
        elif format_type == ModelFormat.COMPRESSED_PICKLE:
            with gzip.open(model_path, 'rb') as f:
                return pickle.load(f)
        
        elif format_type == ModelFormat.COMPRESSED_JOBLIB:
            return joblib.load(model_path)
        
        raise ValueError(f"Unsupported format: {format_type}")
    
    def list_models(self, strategy_name: str = None, tags: List[str] = None) -> List[Dict]:
        """
        List models with optional filtering.
        
        Args:
            strategy_name: Filter by strategy name
            tags: Filter by tags
            
        Returns:
            List of model information
        """
        models = []
        
        for model_id, info in self._model_registry.items():
            # Apply filters
            if strategy_name and info.get('strategy_name') != strategy_name:
                continue
            
            if tags and not any(tag in info.get('tags', []) for tag in tags):
                continue
            
            models.append({
                'model_id': model_id,
                'name': info['name'],
                'version': info['version'],
                'strategy_name': info['strategy_name'],
                'created_at': info['created_at'],
                'performance_metrics': info['performance_metrics'],
                'tags': info.get('tags', [])
            })
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model and associated files.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful
        """
        if model_id not in self._model_registry:
            return False
        
        with self._lock:
            try:
                info = self._model_registry[model_id]
                
                # Remove files
                model_path = Path(info['file_path'])
                metadata_path = Path(info['metadata_path'])
                
                for path in [model_path, metadata_path]:
                    if path.exists():
                        path.unlink()
                
                # Remove from registry
                del self._model_registry[model_id]
                self._save_registry()
                
                logger.info(f"Model deleted: {model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                return False
    
    def backup_model(self, model_id: str) -> str:
        """
        Create backup of model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Backup file path
        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        info = self._model_registry[model_id]
        model_path = Path(info['file_path'])
        metadata_path = Path(info['metadata_path'])
        
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.base_path / "backup" / f"{model_id}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to backup
        shutil.copy2(model_path, backup_dir / model_path.name)
        shutil.copy2(metadata_path, backup_dir / metadata_path.name)
        
        # Create backup manifest
        manifest = {
            'model_id': model_id,
            'backup_timestamp': timestamp,
            'original_created_at': info['created_at'],
            'files': [model_path.name, metadata_path.name]
        }
        
        with open(backup_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Model backup created: {backup_dir}")
        return str(backup_dir)
    
    def _cleanup_old_versions(self, strategy_name: str):
        """Clean up old versions of models for a strategy."""
        strategy_models = [
            (model_id, info) for model_id, info in self._model_registry.items()
            if info.get('strategy_name') == strategy_name
        ]
        
        if len(strategy_models) <= self.max_versions:
            return
        
        # Sort by creation date and remove oldest
        strategy_models.sort(key=lambda x: x[1]['created_at'])
        models_to_remove = strategy_models[:-self.max_versions]
        
        for model_id, _ in models_to_remove:
            logger.info(f"Auto-cleanup: removing old model {model_id}")
            self.delete_model(model_id)
    
    @contextmanager
    def temporary_model(self, model: Any, metadata: ModelMetadata):
        """
        Context manager for temporary model storage.
        
        Args:
            model: Model to store temporarily
            metadata: Model metadata
        """
        model_id = None
        try:
            # Save model temporarily
            model_id = self.save_model(model, metadata)
            yield model_id
        finally:
            # Clean up temporary model
            if model_id:
                self.delete_model(model_id)
    
    def get_storage_stats(self) -> Dict:
        """Get storage system statistics."""
        total_models = len(self._model_registry)
        total_size = 0
        strategies = set()
        formats = {}
        
        for model_id, info in self._model_registry.items():
            try:
                model_path = Path(info['file_path'])
                if model_path.exists():
                    total_size += model_path.stat().st_size
            except Exception:
                pass
            
            strategies.add(info.get('strategy_name', 'unknown'))
            
            # Load metadata for format info
            try:
                metadata_path = self._get_metadata_path(model_id)
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    format_type = metadata.get('storage_format', 'unknown')
                    formats[format_type] = formats.get(format_type, 0) + 1
            except Exception:
                pass
        
        return {
            'total_models': total_models,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'strategies_count': len(strategies),
            'strategies': list(strategies),
            'formats': formats,
            'storage_path': str(self.base_path)
        }


# Utility functions for easy model management
def save_strategy_parameters(parameters: Dict, strategy_name: str, 
                           performance_metrics: Dict, 
                           storage: ModelStorage = None) -> str:
    """Convenience function to save strategy parameters."""
    if storage is None:
        storage = ModelStorage()
    
    metadata = ModelMetadata(
        model_id="",  # Will be generated
        name=f"{strategy_name}_parameters",
        version="1.0",
        created_at=datetime.now(),
        model_type="strategy_parameters",
        strategy_name=strategy_name,
        performance_metrics=performance_metrics,
        parameters=parameters,
        tags=["parameters", "optimized"],
        description=f"Optimized parameters for {strategy_name} strategy"
    )
    
    return storage.save_model(parameters, metadata, ModelFormat.JSON)


def load_strategy_parameters(model_id: str, storage: ModelStorage = None) -> Tuple[Dict, Dict]:
    """Convenience function to load strategy parameters."""
    if storage is None:
        storage = ModelStorage()
    
    parameters, metadata = storage.load_model(model_id)
    return parameters, metadata.performance_metrics