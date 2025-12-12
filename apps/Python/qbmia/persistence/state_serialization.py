"""
State serialization utilities for QBMIA components.
"""

import json
import pickle
import numpy as np
import msgpack
import msgpack_numpy as m
from typing import Dict, Any, Union, List, Optional
import logging
import base64
import zlib
from datetime import datetime
import h5py
from pathlib import Path

m.patch()  # Enable msgpack numpy support

logger = logging.getLogger(__name__)

class StateSerializer:
    """
    Comprehensive state serialization for different formats and use cases.
    """

    def __init__(self, compression_level: int = 6):
        """
        Initialize state serializer.

        Args:
            compression_level: Compression level (0-9)
        """
        self.compression_level = compression_level

        # Format handlers
        self.format_handlers = {
            'json': self._serialize_json,
            'pickle': self._serialize_pickle,
            'msgpack': self._serialize_msgpack,
            'hdf5': self._serialize_hdf5,
            'npz': self._serialize_npz
        }

        self.deserialize_handlers = {
            'json': self._deserialize_json,
            'pickle': self._deserialize_pickle,
            'msgpack': self._deserialize_msgpack,
            'hdf5': self._deserialize_hdf5,
            'npz': self._deserialize_npz
        }

    def serialize(self, state: Dict[str, Any], format: str = 'msgpack',
                 compress: bool = True) -> Union[bytes, str]:
        """
        Serialize state to specified format.

        Args:
            state: State dictionary to serialize
            format: Serialization format
            compress: Whether to compress output

        Returns:
            Serialized data
        """
        if format not in self.format_handlers:
            raise ValueError(f"Unsupported format: {format}")

        # Add metadata
        serialized_state = {
            'version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'format': format,
            'compressed': compress,
            'state': state
        }

        # Serialize
        data = self.format_handlers[format](serialized_state)

        # Compress if requested
        if compress and format not in ['hdf5', 'npz']:  # These handle compression internally
            data = zlib.compress(data, level=self.compression_level)

        return data

    def deserialize(self, data: Union[bytes, str, Path],
                   format: Optional[str] = None) -> Dict[str, Any]:
        """
        Deserialize state from data.

        Args:
            data: Serialized data or file path
            format: Format (auto-detect if None)

        Returns:
            Deserialized state
        """
        # Handle file paths
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == '.json':
                format = 'json'
            elif path.suffix in ['.pkl', '.pickle']:
                format = 'pickle'
            elif path.suffix == '.msgpack':
                format = 'msgpack'
            elif path.suffix == '.h5' or path.suffix == '.hdf5':
                format = 'hdf5'
            elif path.suffix == '.npz':
                format = 'npz'

            if format in ['hdf5', 'npz']:
                # These formats need file paths
                return self.deserialize_handlers[format](path)
            else:
                with open(path, 'rb') as f:
                    data = f.read()

        # Try decompression
        try:
            decompressed = zlib.decompress(data)
            data = decompressed
            was_compressed = True
        except:
            was_compressed = False

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(data)

        if format not in self.deserialize_handlers:
            raise ValueError(f"Unsupported format: {format}")

        # Deserialize
        result = self.deserialize_handlers[format](data)

        # Extract state
        if isinstance(result, dict) and 'state' in result:
            return result['state']
        return result

    def _serialize_json(self, state: Dict[str, Any]) -> bytes:
        """Serialize to JSON format."""
        # Convert numpy arrays and other non-JSON types
        json_state = self._prepare_for_json(state)
        return json.dumps(json_state, indent=2).encode('utf-8')

    def _deserialize_json(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from JSON format."""
        json_state = json.loads(data.decode('utf-8'))
        return self._restore_from_json(json_state)

    def _serialize_pickle(self, state: Dict[str, Any]) -> bytes:
        """Serialize to pickle format."""
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_pickle(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from pickle format."""
        return pickle.loads(data)

    def _serialize_msgpack(self, state: Dict[str, Any]) -> bytes:
        """Serialize to msgpack format."""
        return msgpack.packb(state, use_bin_type=True)

    def _deserialize_msgpack(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from msgpack format."""
        return msgpack.unpackb(data, raw=False)

    def _serialize_hdf5(self, state: Dict[str, Any]) -> str:
        """Serialize to HDF5 format (returns filename)."""
        filename = f"state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.h5"

        with h5py.File(filename, 'w') as f:
            self._write_hdf5_recursive(f, '/', state)

        return filename

    def _deserialize_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Deserialize from HDF5 format."""
        state = {}

        with h5py.File(filepath, 'r') as f:
            state = self._read_hdf5_recursive(f)

        return state

    def _write_hdf5_recursive(self, h5file, path: str, data: Any):
        """Recursively write data to HDF5."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}/{key}" if path != '/' else f"/{key}"
                self._write_hdf5_recursive(h5file, new_path, value)

        elif isinstance(data, (list, tuple)):
            # Convert to numpy array
            arr = np.array(data)
            h5file.create_dataset(path, data=arr, compression='gzip',
                                compression_opts=self.compression_level)

        elif isinstance(data, np.ndarray):
            h5file.create_dataset(path, data=data, compression='gzip',
                                compression_opts=self.compression_level)

        elif isinstance(data, (str, int, float, bool)):
            # Store as attribute
            parent_path = '/'.join(path.split('/')[:-1]) or '/'
            attr_name = path.split('/')[-1]

            if parent_path not in h5file:
                h5file.create_group(parent_path)

            h5file[parent_path].attrs[attr_name] = data

        else:
            # Fallback: pickle and store as bytes
            pickled = pickle.dumps(data)
            h5file.create_dataset(path, data=np.frombuffer(pickled, dtype=np.uint8))

    def _read_hdf5_recursive(self, h5group) -> Dict[str, Any]:
        """Recursively read data from HDF5."""
        result = {}

        # Read attributes
        for attr_name, attr_value in h5group.attrs.items():
            result[attr_name] = attr_value

        # Read datasets and groups
        for key in h5group.keys():
            item = h5group[key]

            if isinstance(item, h5py.Dataset):
                result[key] = item[:]
            elif isinstance(item, h5py.Group):
                result[key] = self._read_hdf5_recursive(item)

        return result

    def _serialize_npz(self, state: Dict[str, Any]) -> str:
        """Serialize to NPZ format (returns filename)."""
        filename = f"state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.npz"

        # Flatten nested structure
        flat_data = {}
        self._flatten_for_npz(state, flat_data)

        # Save
        if self.compression_level > 0:
            np.savez_compressed(filename, **flat_data)
        else:
            np.savez(filename, **flat_data)

        return filename

    def _deserialize_npz(self, filepath: Path) -> Dict[str, Any]:
        """Deserialize from NPZ format."""
        with np.load(filepath, allow_pickle=True) as data:
            # Reconstruct nested structure
            flat_data = dict(data)
            return self._unflatten_from_npz(flat_data)

    def _flatten_for_npz(self, data: Dict[str, Any], flat: Dict[str, Any],
                        prefix: str = ''):
        """Flatten nested dictionary for NPZ storage."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_for_npz(value, flat, full_key)
            elif isinstance(value, np.ndarray):
                flat[full_key] = value
            elif isinstance(value, (list, tuple)):
                flat[full_key] = np.array(value)
            else:
                # Store as pickled object
                flat[f"{full_key}.__pickle__"] = np.array(pickle.dumps(value))

    def _unflatten_from_npz(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Unflatten NPZ data to nested dictionary."""
        result = {}

        for key, value in flat.items():
            if key.endswith('.__pickle__'):
                # Unpickle object
                real_key = key[:-11]  # Remove .__pickle__
                value = pickle.loads(value.tobytes())
            else:
                real_key = key

            # Reconstruct nested structure
            parts = real_key.split('.')
            current = result

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return result

    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif isinstance(obj, complex):
            return {
                '__type__': 'complex',
                'real': obj.real,
                'imag': obj.imag
            }
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                'isoformat': obj.isoformat()
            }
        elif hasattr(obj, '__dict__'):
            # Custom objects
            return {
                '__type__': 'custom',
                '__class__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                '__dict__': self._prepare_for_json(obj.__dict__)
            }
        else:
            return obj

    def _restore_from_json(self, obj: Any) -> Any:
        """Restore object from JSON representation."""
        if isinstance(obj, dict) and '__type__' in obj:
            obj_type = obj['__type__']

            if obj_type == 'ndarray':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
            elif obj_type == 'complex':
                return complex(obj['real'], obj['imag'])
            elif obj_type == 'datetime':
                return datetime.fromisoformat(obj['isoformat'])
            elif obj_type == 'custom':
                # Limited custom object restoration
                return obj  # Return dict representation

        elif isinstance(obj, dict):
            return {k: self._restore_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_json(v) for v in obj]

        return obj

    def _detect_format(self, data: bytes) -> str:
        """Auto-detect serialization format."""
        # Try JSON
        try:
            json.loads(data.decode('utf-8'))
            return 'json'
        except:
            pass

        # Try msgpack
        try:
            msgpack.unpackb(data, raw=False)
            return 'msgpack'
        except:
            pass

        # Try pickle
        try:
            pickle.loads(data)
            return 'pickle'
        except:
            pass

        raise ValueError("Unable to detect serialization format")

    def calculate_size_reduction(self, original: bytes, compressed: bytes) -> Dict[str, float]:
        """Calculate compression statistics."""
        original_size = len(original)
        compressed_size = len(compressed)

        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'size_reduction_percent': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        }
