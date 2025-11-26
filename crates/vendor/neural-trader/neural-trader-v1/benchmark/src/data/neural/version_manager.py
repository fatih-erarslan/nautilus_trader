"""
Data Version Manager for Neural Pipeline

This module provides comprehensive data versioning and lineage tracking
for neural forecasting pipelines, ensuring reproducibility and auditability.

Key Features:
- Data versioning and lineage tracking
- Processing step recording
- Reproducibility management
- Change detection
- Rollback capabilities
"""

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import pickle
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataVersion:
    """Data version information."""
    version_id: str
    parent_version_id: Optional[str]
    created_at: float
    created_by: str
    description: str
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_steps: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    size_info: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProcessingStep:
    """Processing step information."""
    step_id: str
    step_name: str
    step_type: str
    timestamp: float
    input_version: str
    output_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class DataLineage:
    """Data lineage tracking."""
    data_id: str
    versions: List[DataVersion] = field(default_factory=list)
    processing_history: List[ProcessingStep] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class DataVersionManager:
    """
    Comprehensive data version manager for neural pipelines.
    
    This class manages data versions, tracks lineage, and ensures
    reproducibility across the neural forecasting pipeline.
    """
    
    def __init__(self, storage_path: str = "data_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.versions_path = self.storage_path / "versions"
        self.lineage_path = self.storage_path / "lineage"
        self.metadata_path = self.storage_path / "metadata"
        
        for path in [self.versions_path, self.lineage_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory caches
        self.version_cache = {}
        self.lineage_cache = {}
        
        # Load existing data
        self._load_existing_data()
        
        self.logger.info(f"DataVersionManager initialized at {storage_path}")
    
    def _load_existing_data(self):
        """Load existing versions and lineage data."""
        try:
            # Load versions
            for version_file in self.versions_path.glob("*.json"):
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                    version = DataVersion(**version_data)
                    self.version_cache[version.version_id] = version
            
            # Load lineage
            for lineage_file in self.lineage_path.glob("*.json"):
                with open(lineage_file, 'r') as f:
                    lineage_data = json.load(f)
                    # Convert dict back to DataLineage
                    lineage = DataLineage(
                        data_id=lineage_data['data_id'],
                        versions=[DataVersion(**v) for v in lineage_data['versions']],
                        processing_history=[ProcessingStep(**s) for s in lineage_data['processing_history']],
                        dependencies=lineage_data['dependencies'],
                        created_at=lineage_data['created_at']
                    )
                    self.lineage_cache[lineage.data_id] = lineage
            
            self.logger.info(f"Loaded {len(self.version_cache)} versions and {len(self.lineage_cache)} lineage records")
            
        except Exception as e:
            self.logger.error(f"Error loading existing data: {e}")
    
    def create_version(
        self,
        data: Union[Dict[str, np.ndarray], pd.DataFrame],
        description: str,
        parent_version_id: Optional[str] = None,
        processing_step: Optional[str] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new data version.
        
        Args:
            data: Data to version
            description: Version description
            parent_version_id: Parent version ID if this is derived data
            processing_step: Processing step that created this version
            quality_metrics: Quality metrics for this version
            metadata: Additional metadata
        
        Returns:
            Version ID
        """
        try:
            # Generate version ID
            version_id = str(uuid.uuid4())
            
            # Calculate data hash
            data_hash = self._calculate_data_hash(data)
            
            # Calculate size info
            size_info = self._calculate_size_info(data)
            
            # Create version
            version = DataVersion(
                version_id=version_id,
                parent_version_id=parent_version_id,
                created_at=time.time(),
                created_by="neural_pipeline",
                description=description,
                data_hash=data_hash,
                metadata=metadata or {},
                processing_steps=[processing_step] if processing_step else [],
                quality_metrics=quality_metrics or {},
                size_info=size_info
            )
            
            # Store version
            self._store_version(version)
            
            # Store data
            self._store_data(version_id, data)
            
            # Cache version
            self.version_cache[version_id] = version
            
            self.logger.info(f"Created version {version_id}: {description}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error creating version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version information."""
        return self.version_cache.get(version_id)
    
    def load_data(self, version_id: str) -> Optional[Union[Dict[str, np.ndarray], pd.DataFrame]]:
        """Load data for a specific version."""
        try:
            data_file = self.versions_path / f"{version_id}.pkl"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data for version {version_id}: {e}")
            return None
    
    def track_processing(
        self,
        symbol: str,
        result: Any,
        processing_step: str = "preprocessing",
        input_version_id: Optional[str] = None
    ) -> str:
        """
        Track a processing step and create new version.
        
        Args:
            symbol: Symbol being processed
            result: Processing result
            processing_step: Name of processing step
            input_version_id: Input version ID
        
        Returns:
            New version ID
        """
        try:
            # Extract data from result
            if hasattr(result, 'data') and result.data is not None:
                data = result.data
                description = f"{processing_step} for {symbol}"
                
                # Extract quality metrics if available
                quality_metrics = None
                if hasattr(result, 'quality_metrics'):
                    quality_metrics = result.quality_metrics
                
                # Extract metadata
                metadata = {
                    'symbol': symbol,
                    'processing_step': processing_step,
                    'success': getattr(result, 'success', True)
                }
                
                if hasattr(result, 'metadata'):
                    metadata.update(result.metadata)
                
                # Create new version
                version_id = self.create_version(
                    data=data,
                    description=description,
                    parent_version_id=input_version_id,
                    processing_step=processing_step,
                    quality_metrics=quality_metrics,
                    metadata=metadata
                )
                
                # Update lineage
                self._update_lineage(symbol, version_id, processing_step, input_version_id)
                
                return version_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error tracking processing: {e}")
            return None
    
    def create_lineage(self, data_id: str) -> str:
        """Create new data lineage tracking."""
        lineage = DataLineage(data_id=data_id)
        self.lineage_cache[data_id] = lineage
        self._store_lineage(lineage)
        return data_id
    
    def _update_lineage(
        self,
        data_id: str,
        version_id: str,
        processing_step: str,
        input_version_id: Optional[str]
    ):
        """Update lineage information."""
        try:
            # Get or create lineage
            if data_id not in self.lineage_cache:
                self.create_lineage(data_id)
            
            lineage = self.lineage_cache[data_id]
            
            # Add version to lineage
            version = self.version_cache.get(version_id)
            if version:
                lineage.versions.append(version)
            
            # Create processing step record
            step = ProcessingStep(
                step_id=str(uuid.uuid4()),
                step_name=processing_step,
                step_type="preprocessing",
                timestamp=time.time(),
                input_version=input_version_id or "initial",
                output_version=version_id,
                success=True
            )
            
            lineage.processing_history.append(step)
            
            # Update dependencies
            if input_version_id:
                if version_id not in lineage.dependencies:
                    lineage.dependencies[version_id] = []
                lineage.dependencies[version_id].append(input_version_id)
            
            # Store updated lineage
            self._store_lineage(lineage)
            
        except Exception as e:
            self.logger.error(f"Error updating lineage: {e}")
    
    def get_lineage(self, data_id: str) -> Optional[DataLineage]:
        """Get lineage information for data ID."""
        return self.lineage_cache.get(data_id)
    
    def get_version_history(self, data_id: str) -> List[DataVersion]:
        """Get version history for data ID."""
        lineage = self.get_lineage(data_id)
        return lineage.versions if lineage else []
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two data versions."""
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            return {'error': 'Version not found'}
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'hash_different': version1.data_hash != version2.data_hash,
            'size_comparison': {},
            'quality_comparison': {},
            'time_difference': version2.created_at - version1.created_at,
            'processing_steps_diff': list(set(version2.processing_steps) - set(version1.processing_steps))
        }
        
        # Compare sizes
        for key in version1.size_info:
            if key in version2.size_info:
                comparison['size_comparison'][key] = {
                    'version1': version1.size_info[key],
                    'version2': version2.size_info[key],
                    'difference': version2.size_info[key] - version1.size_info[key]
                }
        
        # Compare quality metrics
        for metric in version1.quality_metrics:
            if metric in version2.quality_metrics:
                comparison['quality_comparison'][metric] = {
                    'version1': version1.quality_metrics[metric],
                    'version2': version2.quality_metrics[metric],
                    'difference': version2.quality_metrics[metric] - version1.quality_metrics[metric]
                }
        
        return comparison
    
    def rollback_to_version(self, data_id: str, target_version_id: str) -> bool:
        """Rollback to a specific version."""
        try:
            # Load the target version data
            data = self.load_data(target_version_id)
            if data is None:
                return False
            
            # Create a new version as rollback
            rollback_version_id = self.create_version(
                data=data,
                description=f"Rollback to version {target_version_id}",
                parent_version_id=target_version_id,
                processing_step="rollback"
            )
            
            # Update lineage
            self._update_lineage(data_id, rollback_version_id, "rollback", target_version_id)
            
            self.logger.info(f"Rolled back {data_id} to version {target_version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back: {e}")
            return False
    
    def cleanup_old_versions(self, data_id: str, keep_count: int = 10):
        """Clean up old versions, keeping only the most recent ones."""
        try:
            lineage = self.get_lineage(data_id)
            if not lineage or len(lineage.versions) <= keep_count:
                return
            
            # Sort versions by creation time
            sorted_versions = sorted(lineage.versions, key=lambda v: v.created_at, reverse=True)
            
            # Keep recent versions, remove old ones
            versions_to_keep = sorted_versions[:keep_count]
            versions_to_remove = sorted_versions[keep_count:]
            
            for version in versions_to_remove:
                self._remove_version(version.version_id)
            
            # Update lineage
            lineage.versions = versions_to_keep
            self._store_lineage(lineage)
            
            self.logger.info(f"Cleaned up {len(versions_to_remove)} old versions for {data_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up versions: {e}")
    
    def export_lineage(self, data_id: str, output_path: str):
        """Export lineage information to file."""
        try:
            lineage = self.get_lineage(data_id)
            if not lineage:
                return False
            
            export_data = {
                'data_id': lineage.data_id,
                'exported_at': time.time(),
                'versions': [asdict(v) for v in lineage.versions],
                'processing_history': [asdict(s) for s in lineage.processing_history],
                'dependencies': lineage.dependencies,
                'created_at': lineage.created_at
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting lineage: {e}")
            return False
    
    def _calculate_data_hash(self, data: Union[Dict[str, np.ndarray], pd.DataFrame]) -> str:
        """Calculate hash of data for version identification."""
        try:
            if isinstance(data, dict):
                # For dictionary data (typical neural data format)
                combined_data = b""
                for key in sorted(data.keys()):
                    if isinstance(data[key], np.ndarray):
                        combined_data += data[key].tobytes()
                    else:
                        combined_data += str(data[key]).encode()
            
            elif isinstance(data, pd.DataFrame):
                # For DataFrame data
                combined_data = data.to_string().encode()
            
            else:
                # For other data types
                combined_data = str(data).encode()
            
            return hashlib.sha256(combined_data).hexdigest()
            
        except Exception:
            # Fallback hash
            return hashlib.sha256(str(time.time()).encode()).hexdigest()
    
    def _calculate_size_info(self, data: Union[Dict[str, np.ndarray], pd.DataFrame]) -> Dict[str, int]:
        """Calculate size information for data."""
        size_info = {}
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        size_info[f"{key}_shape"] = value.shape
                        size_info[f"{key}_size"] = value.size
                        size_info[f"{key}_bytes"] = value.nbytes
                    else:
                        size_info[f"{key}_size"] = len(str(value))
            
            elif isinstance(data, pd.DataFrame):
                size_info['rows'] = len(data)
                size_info['columns'] = len(data.columns)
                size_info['memory_usage'] = data.memory_usage(deep=True).sum()
            
        except Exception as e:
            self.logger.warning(f"Error calculating size info: {e}")
        
        return size_info
    
    def _store_version(self, version: DataVersion):
        """Store version metadata."""
        version_file = self.metadata_path / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)
    
    def _store_data(self, version_id: str, data: Union[Dict[str, np.ndarray], pd.DataFrame]):
        """Store actual data."""
        data_file = self.versions_path / f"{version_id}.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    
    def _store_lineage(self, lineage: DataLineage):
        """Store lineage information."""
        lineage_file = self.lineage_path / f"{lineage.data_id}.json"
        lineage_dict = {
            'data_id': lineage.data_id,
            'versions': [asdict(v) for v in lineage.versions],
            'processing_history': [asdict(s) for s in lineage.processing_history],
            'dependencies': lineage.dependencies,
            'created_at': lineage.created_at
        }
        
        with open(lineage_file, 'w') as f:
            json.dump(lineage_dict, f, indent=2)
    
    def _remove_version(self, version_id: str):
        """Remove a version and its data."""
        try:
            # Remove from cache
            if version_id in self.version_cache:
                del self.version_cache[version_id]
            
            # Remove files
            metadata_file = self.metadata_path / f"{version_id}.json"
            data_file = self.versions_path / f"{version_id}.pkl"
            
            if metadata_file.exists():
                metadata_file.unlink()
            if data_file.exists():
                data_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Error removing version {version_id}: {e}")
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            total_size = 0
            file_count = 0
            
            for path in [self.versions_path, self.lineage_path, self.metadata_path]:
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'version_count': len(self.version_cache),
                'lineage_count': len(self.lineage_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating storage usage: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version manager statistics."""
        return {
            'total_versions': len(self.version_cache),
            'total_lineages': len(self.lineage_cache),
            'storage_usage': self.get_storage_usage(),
            'recent_versions': [
                {
                    'version_id': v.version_id,
                    'description': v.description,
                    'created_at': v.created_at
                }
                for v in sorted(self.version_cache.values(), key=lambda x: x.created_at, reverse=True)[:5]
            ]
        }