"""
State management and persistence for QBMIA agent.
"""

import os
import json
import pickle
import hashlib
import gzip
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import threading
import logging

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages state persistence, checkpointing, and recovery for QBMIA.
    """

    def __init__(self, agent_id: str, checkpoint_dir: str = './checkpoints'):
        """
        Initialize state manager.

        Args:
            agent_id: Unique agent identifier
            checkpoint_dir: Directory for storing checkpoints
        """
        self.agent_id = agent_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_lock = threading.Lock()

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Checkpoint metadata
        self.checkpoint_history = []
        self.max_checkpoints = 10
        self.compression_enabled = True

        # Load checkpoint history
        self._load_checkpoint_history()

    def save_checkpoint(self, state: Dict[str, Any],
                       filepath: Optional[str] = None) -> str:
        """
        Save state checkpoint with compression and validation.

        Args:
            state: State dictionary to save
            filepath: Optional custom filepath

        Returns:
            Path to saved checkpoint
        """
        with self.checkpoint_lock:
            # Generate checkpoint filename
            if filepath is None:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.agent_id}_checkpoint_{timestamp}"
                filepath = os.path.join(self.checkpoint_dir, filename)

            try:
                # Add metadata
                checkpoint_data = {
                    'version': '1.0.0',
                    'agent_id': self.agent_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'state': state
                }

                # Calculate checksum
                checksum = self._calculate_checksum(checkpoint_data)
                checkpoint_data['checksum'] = checksum

                # Save based on format
                if filepath.endswith('.npz'):
                    self._save_binary(checkpoint_data, filepath)
                elif filepath.endswith('.json'):
                    self._save_json(checkpoint_data, filepath)
                else:
                    # Default to compressed pickle
                    self._save_pickle(checkpoint_data, filepath + '.pkl.gz')
                    filepath = filepath + '.pkl.gz'

                # Update history
                self._update_checkpoint_history(filepath)

                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()

                logger.info(f"Checkpoint saved to {filepath}")
                return filepath

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                raise

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load state from checkpoint with validation.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Loaded state dictionary
        """
        with self.checkpoint_lock:
            try:
                # Load based on format
                if filepath.endswith('.npz'):
                    checkpoint_data = self._load_binary(filepath)
                elif filepath.endswith('.json'):
                    checkpoint_data = self._load_json(filepath)
                elif filepath.endswith('.pkl.gz'):
                    checkpoint_data = self._load_pickle(filepath)
                else:
                    raise ValueError(f"Unknown checkpoint format: {filepath}")

                # Validate checksum
                stored_checksum = checkpoint_data.pop('checksum', None)
                calculated_checksum = self._calculate_checksum(checkpoint_data)

                if stored_checksum != calculated_checksum:
                    raise ValueError("Checkpoint validation failed: checksum mismatch")

                # Validate version compatibility
                version = checkpoint_data.get('version', '0.0.0')
                if not self._check_version_compatibility(version):
                    logger.warning(f"Loading checkpoint with version {version}")

                logger.info(f"Checkpoint loaded from {filepath}")
                return checkpoint_data['state']

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise

    def _save_binary(self, data: Dict[str, Any], filepath: str):
        """Save checkpoint in NumPy binary format."""
        # Extract numpy arrays
        arrays = {}
        metadata = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif isinstance(value, dict) and self._contains_arrays(value):
                # Flatten nested arrays
                flattened = self._flatten_arrays(value, prefix=key)
                arrays.update(flattened['arrays'])
                metadata[key] = flattened['structure']
            else:
                metadata[key] = value

        # Save arrays and metadata
        np.savez_compressed(
            filepath,
            **arrays,
            __metadata__=json.dumps(metadata).encode('utf-8')
        )

    def _load_binary(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from NumPy binary format."""
        with np.load(filepath, allow_pickle=False) as data:
            # Extract metadata
            metadata = json.loads(data['__metadata__'].decode('utf-8'))

            # Reconstruct data structure
            result = metadata.copy()

            # Add arrays back
            for key in data.files:
                if key != '__metadata__':
                    result[key] = data[key]

            return result

    def _save_json(self, data: Dict[str, Any], filepath: str):
        """Save checkpoint in JSON format."""
        # Convert numpy arrays to lists
        json_data = self._prepare_for_json(data)

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)

        if self.compression_enabled:
            # Compress the JSON file
            with open(filepath, 'rb') as f_in:
                with gzip.open(filepath + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
            return filepath + '.gz'

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from JSON format."""
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                json_data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                json_data = json.load(f)

        # Convert lists back to numpy arrays where needed
        return self._restore_from_json(json_data)

    def _save_pickle(self, data: Dict[str, Any], filepath: str):
        """Save checkpoint using compressed pickle."""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from compressed pickle."""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA256 checksum for data validation."""
        # Convert data to bytes
        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(data_bytes).hexdigest()

    def _check_version_compatibility(self, version: str) -> bool:
        """Check if checkpoint version is compatible."""
        current_major = int('1'.split('.')[0])  # Current version 1.0.0
        checkpoint_major = int(version.split('.')[0])
        return current_major == checkpoint_major

    def _contains_arrays(self, obj: Any) -> bool:
        """Check if object contains numpy arrays."""
        if isinstance(obj, np.ndarray):
            return True
        elif isinstance(obj, dict):
            return any(self._contains_arrays(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_arrays(v) for v in obj)
        return False

    def _flatten_arrays(self, obj: Any, prefix: str = '') -> Dict[str, Any]:
        """Flatten nested structure containing arrays."""
        arrays = {}
        structure = {}

        if isinstance(obj, dict):
            structure['_type'] = 'dict'
            structure['_keys'] = {}

            for key, value in obj.items():
                key_prefix = f"{prefix}.{key}" if prefix else key

                if isinstance(value, np.ndarray):
                    arrays[key_prefix] = value
                    structure['_keys'][key] = {'_type': 'array', '_key': key_prefix}
                elif isinstance(value, (dict, list, tuple)):
                    nested = self._flatten_arrays(value, key_prefix)
                    arrays.update(nested['arrays'])
                    structure['_keys'][key] = nested['structure']
                else:
                    structure['_keys'][key] = value

        elif isinstance(obj, (list, tuple)):
            structure['_type'] = 'list' if isinstance(obj, list) else 'tuple'
            structure['_items'] = []

            for i, value in enumerate(obj):
                item_prefix = f"{prefix}.{i}"

                if isinstance(value, np.ndarray):
                    arrays[item_prefix] = value
                    structure['_items'].append({'_type': 'array', '_key': item_prefix})
                elif isinstance(value, (dict, list, tuple)):
                    nested = self._flatten_arrays(value, item_prefix)
                    arrays.update(nested['arrays'])
                    structure['_items'].append(nested['structure'])
                else:
                    structure['_items'].append(value)

        return {'arrays': arrays, 'structure': structure}

    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return {
                '_type': 'ndarray',
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                'shape': obj.shape
            }
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def _restore_from_json(self, obj: Any) -> Any:
        """Restore object from JSON representation."""
        if isinstance(obj, dict) and '_type' in obj:
            if obj['_type'] == 'ndarray':
                return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        elif isinstance(obj, dict):
            return {k: self._restore_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_json(v) for v in obj]
        return obj

    def _load_checkpoint_history(self):
        """Load checkpoint history from metadata file."""
        history_file = os.path.join(self.checkpoint_dir, f"{self.agent_id}_history.json")

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.checkpoint_history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint history: {e}")
                self.checkpoint_history = []
        else:
            self.checkpoint_history = []

    def _update_checkpoint_history(self, filepath: str):
        """Update checkpoint history with new entry."""
        entry = {
            'filepath': filepath,
            'timestamp': datetime.utcnow().isoformat(),
            'size': os.path.getsize(filepath)
        }

        self.checkpoint_history.append(entry)

        # Save updated history
        history_file = os.path.join(self.checkpoint_dir, f"{self.agent_id}_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond retention limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by timestamp
            sorted_history = sorted(
                self.checkpoint_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )

            # Keep only recent checkpoints
            to_remove = sorted_history[self.max_checkpoints:]

            for entry in to_remove:
                filepath = entry['filepath']
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old checkpoint: {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {filepath}: {e}")

            # Update history
            self.checkpoint_history = sorted_history[:self.max_checkpoints]

    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints."""
        return sorted(
            self.checkpoint_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )

    def get_last_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the most recent checkpoint."""
        if self.checkpoint_history:
            return max(self.checkpoint_history, key=lambda x: x['timestamp'])
        return None

    def validate_checkpoint(self, filepath: str) -> bool:
        """Validate checkpoint integrity without loading full state."""
        try:
            # Quick validation based on file format
            if filepath.endswith('.npz'):
                with np.load(filepath, allow_pickle=False) as data:
                    return '__metadata__' in data
            elif filepath.endswith('.json') or filepath.endswith('.json.gz'):
                # Try to load just the metadata
                if filepath.endswith('.gz'):
                    with gzip.open(filepath, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                return 'version' in data and 'checksum' in data
            elif filepath.endswith('.pkl.gz'):
                # Basic file integrity check
                with gzip.open(filepath, 'rb') as f:
                    # Try to read first few bytes
                    header = f.read(100)
                    return len(header) > 0

            return False

        except Exception as e:
            logger.error(f"Checkpoint validation failed for {filepath}: {e}")
            return False
