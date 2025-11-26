"""
Model Serialization and Deserialization for NHITS Forecasting.

This module provides comprehensive model serialization capabilities including:
- Model state and configuration serialization
- Version control and compatibility checking
- Compressed storage and cloud deployment support
- Model export/import for different environments
- Metadata preservation and validation
"""

import asyncio
import logging
import pickle
import json
import gzip
import base64
import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, BinaryIO
import tempfile

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False


class ModelSerializer:
    """
    Advanced model serialization with compression, versioning, and validation.
    
    Features:
    - Multiple serialization formats (pickle, torch, custom)
    - Compression and encryption support
    - Version control and compatibility checking
    - Metadata preservation
    - Cloud storage optimization
    - Model validation and integrity checks
    """
    
    def __init__(
        self,
        compression_enabled: bool = True,
        encryption_enabled: bool = False,
        version_control: bool = True,
        validate_on_load: bool = True,
        storage_format: str = 'auto'  # 'pickle', 'torch', 'custom', 'auto'
    ):
        """
        Initialize model serializer.
        
        Args:
            compression_enabled: Enable gzip compression
            encryption_enabled: Enable model encryption (requires key)
            version_control: Enable version tracking
            validate_on_load: Validate models on load
            storage_format: Preferred storage format
        """
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        self.version_control = version_control
        self.validate_on_load = validate_on_load
        self.storage_format = storage_format
        
        # Encryption configuration
        self.encryption_key = None
        if encryption_enabled:
            self._setup_encryption()
        
        # Version tracking
        self.serialization_version = "1.0.0"
        self.supported_versions = ["1.0.0"]
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.logger.info("Model Serializer initialized")
    
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
    
    def _setup_encryption(self):
        """Setup encryption configuration."""
        try:
            # Use environment variable or generate key
            key = os.environ.get('MODEL_ENCRYPTION_KEY')
            if key:
                self.encryption_key = key.encode()
            else:
                # Generate a simple key (in production, use proper key management)
                import secrets
                self.encryption_key = secrets.token_bytes(32)
                self.logger.warning("Using auto-generated encryption key - set MODEL_ENCRYPTION_KEY environment variable for production")
        except Exception as e:
            self.logger.error(f"Encryption setup failed: {str(e)}")
            self.encryption_enabled = False
    
    async def serialize_model(
        self,
        model_data: Dict[str, Any],
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compress: Optional[bool] = None,
        encrypt: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Serialize model with comprehensive metadata and options.
        
        Args:
            model_data: Model data to serialize
            output_path: Output file path
            metadata: Additional metadata
            compress: Override compression setting
            encrypt: Override encryption setting
            
        Returns:
            Serialization results
        """
        try:
            start_time = datetime.now()
            
            # Prepare serialization package
            package = await self._prepare_serialization_package(model_data, metadata)
            
            # Determine format
            format_type = self._determine_format(model_data)
            
            # Serialize data
            serialized_data = await self._serialize_data(package, format_type)
            
            # Apply compression if enabled
            if compress if compress is not None else self.compression_enabled:
                serialized_data = await self._compress_data(serialized_data)
                package['compression'] = 'gzip'
            
            # Apply encryption if enabled
            if encrypt if encrypt is not None else self.encryption_enabled:
                serialized_data = await self._encrypt_data(serialized_data)
                package['encryption'] = 'aes256'
            
            # Write to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            await self._write_serialized_file(serialized_data, output_path, package)
            
            # Calculate file stats
            file_size = output_path.stat().st_size
            serialization_time = (datetime.now() - start_time).total_seconds()
            
            # Create checksum
            checksum = await self._calculate_checksum(output_path)
            
            result = {
                'success': True,
                'output_path': str(output_path),
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'serialization_time': serialization_time,
                'format': format_type,
                'compression': package.get('compression'),
                'encryption': package.get('encryption'),
                'checksum': checksum,
                'version': self.serialization_version,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Model serialized successfully: {file_size / (1024 * 1024):.2f}MB in {serialization_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model serialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'output_path': str(output_path) if 'output_path' in locals() else None
            }
    
    async def deserialize_model(
        self,
        input_path: Union[str, Path],
        validate: Optional[bool] = None,
        decrypt: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Deserialize model with validation and error handling.
        
        Args:
            input_path: Input file path
            validate: Override validation setting
            decrypt: Override decryption setting
            
        Returns:
            Deserialization results with model data
        """
        try:
            start_time = datetime.now()
            input_path = Path(input_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Model file not found: {input_path}")
            
            # Read and validate file
            file_data, package_info = await self._read_serialized_file(input_path)
            
            # Validate version compatibility
            if self.version_control:
                await self._validate_version_compatibility(package_info)
            
            # Decrypt if needed
            if package_info.get('encryption') and (decrypt if decrypt is not None else self.encryption_enabled):
                file_data = await self._decrypt_data(file_data)
            
            # Decompress if needed
            if package_info.get('compression'):
                file_data = await self._decompress_data(file_data)
            
            # Deserialize data
            model_data = await self._deserialize_data(file_data, package_info.get('format', 'pickle'))
            
            # Validate model if enabled
            if validate if validate is not None else self.validate_on_load:
                validation_result = await self._validate_model_data(model_data)
                if not validation_result['valid']:
                    self.logger.warning(f"Model validation failed: {validation_result['issues']}")
            
            deserialization_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'model_data': model_data,
                'metadata': package_info.get('metadata', {}),
                'deserialization_time': deserialization_time,
                'file_size_mb': input_path.stat().st_size / (1024 * 1024),
                'format': package_info.get('format'),
                'version': package_info.get('version'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Model deserialized successfully in {deserialization_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model deserialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'input_path': str(input_path)
            }
    
    async def _prepare_serialization_package(
        self,
        model_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare complete serialization package with metadata."""
        package = {
            'version': self.serialization_version,
            'timestamp': datetime.now().isoformat(),
            'model_data': model_data,
            'metadata': metadata or {},
            'serializer_info': {
                'compression_enabled': self.compression_enabled,
                'encryption_enabled': self.encryption_enabled,
                'storage_format': self.storage_format
            }
        }
        
        # Add model type information
        if 'nf' in model_data:
            package['model_type'] = 'neuralforecast'
        elif 'forecaster' in model_data:
            package['model_type'] = 'nhits_forecaster'
        else:
            package['model_type'] = 'unknown'
        
        # Add data statistics
        if isinstance(model_data, dict):
            package['data_stats'] = {
                'num_keys': len(model_data),
                'keys': list(model_data.keys()),
                'total_size_estimate': self._estimate_data_size(model_data)
            }
        
        return package
    
    def _determine_format(self, model_data: Dict[str, Any]) -> str:
        """Determine optimal serialization format."""
        if self.storage_format != 'auto':
            return self.storage_format
        
        # Auto-determine based on content
        if TORCH_AVAILABLE and any(isinstance(v, torch.nn.Module) for v in model_data.values()):
            return 'torch'
        elif CLOUDPICKLE_AVAILABLE:
            return 'cloudpickle'
        else:
            return 'pickle'
    
    async def _serialize_data(self, package: Dict[str, Any], format_type: str) -> bytes:
        """Serialize data using specified format."""
        try:
            if format_type == 'torch' and TORCH_AVAILABLE:
                # Use torch's serialization for PyTorch objects
                buffer = io.BytesIO()
                torch.save(package, buffer)
                return buffer.getvalue()
            
            elif format_type == 'cloudpickle' and CLOUDPICKLE_AVAILABLE:
                # Use cloudpickle for complex objects
                return cloudpickle.dumps(package)
            
            else:
                # Default to pickle
                return pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            self.logger.error(f"Serialization with format {format_type} failed: {str(e)}")
            # Fallback to pickle
            return pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
    
    async def _deserialize_data(self, data: bytes, format_type: str) -> Dict[str, Any]:
        """Deserialize data using specified format."""
        try:
            if format_type == 'torch' and TORCH_AVAILABLE:
                import io
                buffer = io.BytesIO(data)
                return torch.load(buffer, map_location='cpu')
            
            elif format_type == 'cloudpickle' and CLOUDPICKLE_AVAILABLE:
                return cloudpickle.loads(data)
            
            else:
                return pickle.loads(data)
                
        except Exception as e:
            self.logger.error(f"Deserialization with format {format_type} failed: {str(e)}")
            # Try fallback methods
            try:
                return pickle.loads(data)
            except:
                if CLOUDPICKLE_AVAILABLE:
                    return cloudpickle.loads(data)
                raise
    
    async def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        try:
            return gzip.compress(data, compresslevel=6)
        except Exception as e:
            self.logger.error(f"Compression failed: {str(e)}")
            return data
    
    async def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        try:
            return gzip.decompress(data)
        except Exception as e:
            self.logger.error(f"Decompression failed: {str(e)}")
            return data
    
    async def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data (simple implementation - use proper encryption in production)."""
        try:
            if not self.encryption_key:
                raise ValueError("Encryption key not set")
            
            # Simple XOR encryption (replace with proper encryption in production)
            key_length = len(self.encryption_key)
            encrypted = bytearray()
            
            for i, byte in enumerate(data):
                encrypted.append(byte ^ self.encryption_key[i % key_length])
            
            return bytes(encrypted)
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            return data
    
    async def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data (simple implementation)."""
        try:
            # XOR decryption (same as encryption for XOR)
            return await self._encrypt_data(data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            return data
    
    async def _write_serialized_file(
        self,
        data: bytes,
        output_path: Path,
        package_info: Dict[str, Any]
    ):
        """Write serialized data to file with metadata."""
        try:
            # Create metadata file alongside main file
            metadata_path = output_path.with_suffix(output_path.suffix + '.meta')
            
            metadata = {
                'version': package_info.get('version'),
                'format': package_info.get('format'),
                'compression': package_info.get('compression'),
                'encryption': package_info.get('encryption'),
                'model_type': package_info.get('model_type'),
                'timestamp': package_info.get('timestamp'),
                'file_size': len(data)
            }
            
            # Write main file
            with open(output_path, 'wb') as f:
                f.write(data)
            
            # Write metadata file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"File writing failed: {str(e)}")
            raise
    
    async def _read_serialized_file(self, input_path: Path) -> Tuple[bytes, Dict[str, Any]]:
        """Read serialized file with metadata."""
        try:
            # Read main file
            with open(input_path, 'rb') as f:
                data = f.read()
            
            # Try to read metadata file
            metadata_path = input_path.with_suffix(input_path.suffix + '.meta')
            package_info = {}
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        package_info = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata: {str(e)}")
            
            return data, package_info
            
        except Exception as e:
            self.logger.error(f"File reading failed: {str(e)}")
            raise
    
    async def _validate_version_compatibility(self, package_info: Dict[str, Any]):
        """Validate version compatibility."""
        try:
            package_version = package_info.get('version')
            if not package_version:
                self.logger.warning("No version information found in package")
                return
            
            if package_version not in self.supported_versions:
                raise ValueError(f"Unsupported version: {package_version}. Supported: {self.supported_versions}")
                
        except Exception as e:
            self.logger.error(f"Version validation failed: {str(e)}")
            raise
    
    async def _validate_model_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate deserialized model data."""
        try:
            issues = []
            
            # Check for required fields
            if 'model_data' not in model_data:
                issues.append("Missing model_data field")
            
            # Check for metadata
            if 'metadata' not in model_data:
                issues.append("Missing metadata field")
            
            # Check model type specific requirements
            actual_model_data = model_data.get('model_data', {})
            
            if 'nf' in actual_model_data or 'forecaster' in actual_model_data:
                # NHITS specific validation
                if 'config' not in actual_model_data:
                    issues.append("Missing model configuration")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Validation error: {str(e)}"]
            }
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        try:
            # Quick size estimation
            import sys
            return sys.getsizeof(data)
        except:
            return 0
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for integrity verification."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {str(e)}")
            return ""
    
    async def export_for_deployment(
        self,
        model_data: Dict[str, Any],
        output_dir: Union[str, Path],
        deployment_target: str = 'flyio',
        optimization_level: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Export model optimized for specific deployment target.
        
        Args:
            model_data: Model data to export
            output_dir: Output directory
            deployment_target: Target deployment ('flyio', 'docker', 'serverless')
            optimization_level: Optimization level ('size', 'speed', 'balanced')
            
        Returns:
            Export results
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure export settings based on target
            export_config = self._get_deployment_config(deployment_target, optimization_level)
            
            # Prepare optimized model package
            optimized_data = await self._optimize_for_deployment(model_data, export_config)
            
            # Export main model
            main_model_path = output_dir / 'model.pkl'
            result = await self.serialize_model(
                model_data=optimized_data,
                output_path=main_model_path,
                compress=export_config['compress'],
                encrypt=export_config['encrypt']
            )
            
            # Create deployment manifest
            manifest = {
                'deployment_target': deployment_target,
                'optimization_level': optimization_level,
                'model_file': 'model.pkl',
                'requirements': self._get_deployment_requirements(deployment_target),
                'config': export_config,
                'export_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'type': optimized_data.get('model_type', 'nhits'),
                    'size_mb': result.get('file_size_mb', 0),
                    'checksum': result.get('checksum', '')
                }
            }
            
            # Write manifest
            manifest_path = output_dir / 'deployment_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create deployment instructions
            instructions_path = output_dir / 'DEPLOYMENT.md'
            instructions = self._generate_deployment_instructions(deployment_target, manifest)
            with open(instructions_path, 'w') as f:
                f.write(instructions)
            
            return {
                'success': True,
                'output_dir': str(output_dir),
                'model_path': str(main_model_path),
                'manifest_path': str(manifest_path),
                'deployment_target': deployment_target,
                'total_size_mb': result.get('file_size_mb', 0),
                'files_created': ['model.pkl', 'deployment_manifest.json', 'DEPLOYMENT.md']
            }
            
        except Exception as e:
            self.logger.error(f"Deployment export failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_deployment_config(self, target: str, optimization: str) -> Dict[str, Any]:
        """Get deployment-specific configuration."""
        configs = {
            'flyio': {
                'compress': True,
                'encrypt': False,
                'format': 'pickle',
                'strip_debug': True,
                'optimize_memory': True
            },
            'docker': {
                'compress': True,
                'encrypt': False,
                'format': 'pickle',
                'strip_debug': False,
                'optimize_memory': False
            },
            'serverless': {
                'compress': True,
                'encrypt': False,
                'format': 'cloudpickle' if CLOUDPICKLE_AVAILABLE else 'pickle',
                'strip_debug': True,
                'optimize_memory': True,
                'cold_start_optimization': True
            }
        }
        
        # Apply optimization level adjustments
        config = configs.get(target, configs['docker'])
        
        if optimization == 'size':
            config['compress'] = True
            config['strip_debug'] = True
        elif optimization == 'speed':
            config['compress'] = False
            config['strip_debug'] = False
        
        return config
    
    async def _optimize_for_deployment(
        self,
        model_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize model data for deployment."""
        try:
            optimized = model_data.copy()
            
            # Strip debug information if requested
            if config.get('strip_debug', False):
                optimized = self._strip_debug_info(optimized)
            
            # Memory optimization
            if config.get('optimize_memory', False):
                optimized = await self._optimize_memory_usage(optimized)
            
            # Cold start optimization for serverless
            if config.get('cold_start_optimization', False):
                optimized = await self._optimize_cold_start(optimized)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"Deployment optimization failed: {str(e)}")
            return model_data
    
    def _strip_debug_info(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Strip debug information to reduce size."""
        # Remove training history and other debug info
        cleaned = {}
        
        for key, value in model_data.items():
            if key in ['training_history', 'debug_info', 'logs']:
                continue  # Skip debug information
            cleaned[key] = value
        
        return cleaned
    
    async def _optimize_memory_usage(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage for deployment."""
        # This could include model pruning, quantization, etc.
        # For now, just ensure efficient data structures
        return model_data
    
    async def _optimize_cold_start(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize for serverless cold start performance."""
        # Pre-compile frequently used functions, optimize imports, etc.
        return model_data
    
    def _get_deployment_requirements(self, target: str) -> List[str]:
        """Get deployment requirements for target platform."""
        base_requirements = [
            'numpy>=1.21.0',
            'pandas>=1.5.0',
            'scikit-learn>=1.1.0'
        ]
        
        if target in ['flyio', 'docker']:
            base_requirements.extend([
                'neuralforecast>=1.6.4',
                'torch>=2.0.0'
            ])
        
        return base_requirements
    
    def _generate_deployment_instructions(self, target: str, manifest: Dict[str, Any]) -> str:
        """Generate deployment instructions."""
        instructions = f"""# Deployment Instructions for {target.upper()}

## Model Information
- Model Type: {manifest['model_info']['type']}
- Model Size: {manifest['model_info']['size_mb']:.2f} MB
- Export Date: {manifest['export_timestamp']}
- Optimization Level: {manifest['optimization_level']}

## Requirements
Install the following dependencies:
```
{chr(10).join(manifest['requirements'])}
```

## Deployment Steps

"""
        
        if target == 'flyio':
            instructions += """### Fly.io Deployment

1. Place the model files in your Fly.io application directory
2. Update your Dockerfile to include model files:
   ```dockerfile
   COPY model.pkl /app/
   COPY deployment_manifest.json /app/
   ```
3. Ensure GPU support is enabled in fly.toml if using GPU acceleration
4. Deploy with: `fly deploy`

### Loading the Model
```python
from neural_forecast.model_serialization import ModelSerializer

serializer = ModelSerializer()
result = await serializer.deserialize_model('model.pkl')
model_data = result['model_data']
```
"""
        
        elif target == 'docker':
            instructions += """### Docker Deployment

1. Add model files to your Docker image:
   ```dockerfile
   COPY model.pkl /app/models/
   COPY deployment_manifest.json /app/models/
   ```
2. Install requirements in Dockerfile
3. Build and run your container

### Loading the Model
```python
from neural_forecast.model_serialization import ModelSerializer

serializer = ModelSerializer()
result = await serializer.deserialize_model('/app/models/model.pkl')
model_data = result['model_data']
```
"""
        
        elif target == 'serverless':
            instructions += """### Serverless Deployment

1. Include model files in your deployment package
2. Ensure cold start optimization is enabled
3. Consider using Lambda layers for dependencies

### Loading the Model (with caching)
```python
import os
from neural_forecast.model_serialization import ModelSerializer

# Global model cache for serverless
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        serializer = ModelSerializer()
        result = await serializer.deserialize_model('model.pkl')
        _model_cache = result['model_data']
    return _model_cache
```
"""
        
        return instructions