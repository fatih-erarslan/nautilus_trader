"""
Checkpoint management for QBMIA state persistence.
"""

import os
import json
import time
import hashlib
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
import threading
import asyncio
from pathlib import Path
import pickle
import gzip

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages checkpointing for QBMIA components with versioning and recovery.
    """

    def __init__(self, base_dir: str = "./checkpoints",
                 agent_id: str = "QBMIA_001",
                 max_checkpoints: int = 10,
                 compression: bool = True):
        """
        Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
            agent_id: Agent identifier
            max_checkpoints: Maximum checkpoints to retain
            compression: Enable checkpoint compression
        """
        self.base_dir = Path(base_dir)
        self.agent_id = agent_id
        self.max_checkpoints = max_checkpoints
        self.compression = compression

        # Create checkpoint directory structure
        self.checkpoint_dir = self.base_dir / agent_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint metadata
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.checkpoint_history = self._load_metadata()

        # Async checkpoint queue
        self.checkpoint_queue = asyncio.Queue()
        self.checkpoint_lock = threading.Lock()

        # Performance tracking
        self.checkpoint_stats = {
            'total_checkpoints': 0,
            'average_size_mb': 0.0,
            'average_time_seconds': 0.0,
            'compression_ratio': 0.0
        }

        logger.info(f"CheckpointManager initialized for {agent_id}")

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return []
        return []

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.checkpoint_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def create_checkpoint(self, state: Dict[str, Any],
                              checkpoint_type: str = "periodic") -> str:
        """
        Create a new checkpoint asynchronously.

        Args:
            state: State dictionary to checkpoint
            checkpoint_type: Type of checkpoint (periodic, manual, critical)

        Returns:
            Checkpoint ID
        """
        checkpoint_id = self._generate_checkpoint_id()
        timestamp = datetime.utcnow()

        # Add to queue for async processing
        await self.checkpoint_queue.put({
            'id': checkpoint_id,
            'state': state,
            'type': checkpoint_type,
            'timestamp': timestamp
        })

        # Process checkpoint
        checkpoint_path = await self._process_checkpoint(checkpoint_id, state,
                                                       checkpoint_type, timestamp)

        return checkpoint_id

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{self.agent_id}_{timestamp}_{random_suffix}"

    async def _process_checkpoint(self, checkpoint_id: str, state: Dict[str, Any],
                                 checkpoint_type: str, timestamp: datetime) -> str:
        """Process and save checkpoint."""
        start_time = time.time()

        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)

        # Save different components
        saved_files = []
        total_size = 0

        # Save quantum states
        if 'quantum_states' in state:
            quantum_file = await self._save_component(
                checkpoint_path / "quantum_states.npz",
                state['quantum_states'],
                compress=self.compression
            )
            saved_files.append(quantum_file)
            total_size += quantum_file.stat().st_size

        # Save memory state
        if 'memory_state' in state:
            memory_file = await self._save_component(
                checkpoint_path / "memory_state.pkl",
                state['memory_state'],
                compress=self.compression
            )
            saved_files.append(memory_file)
            total_size += memory_file.stat().st_size

        # Save component states
        if 'component_states' in state:
            for component_name, component_state in state['component_states'].items():
                component_file = await self._save_component(
                    checkpoint_path / f"{component_name}_state.pkl",
                    component_state,
                    compress=self.compression
                )
                saved_files.append(component_file)
                total_size += component_file.stat().st_size

        # Save metadata
        metadata = {
            'checkpoint_id': checkpoint_id,
            'timestamp': timestamp.isoformat(),
            'type': checkpoint_type,
            'agent_id': self.agent_id,
            'files': [str(f.relative_to(checkpoint_path)) for f in saved_files],
            'total_size_bytes': total_size,
            'compressed': self.compression,
            'checksum': self._calculate_checksum(state)
        }

        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update checkpoint history
        with self.checkpoint_lock:
            self.checkpoint_history.append({
                'checkpoint_id': checkpoint_id,
                'timestamp': timestamp.isoformat(),
                'type': checkpoint_type,
                'size_mb': total_size / (1024 * 1024),
                'path': str(checkpoint_path)
            })
            self._save_metadata()

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

        # Update statistics
        elapsed_time = time.time() - start_time
        self._update_statistics(total_size, elapsed_time)

        logger.info(f"Checkpoint {checkpoint_id} saved ({total_size / (1024*1024):.2f} MB in {elapsed_time:.2f}s)")

        return str(checkpoint_path)

    async def _save_component(self, filepath: Path, data: Any,
                            compress: bool = True) -> Path:
        """Save individual component with optional compression."""
        if filepath.suffix == '.npz':
            # NumPy array data
            if isinstance(data, dict):
                if compress:
                    np.savez_compressed(filepath, **data)
                else:
                    np.savez(filepath, **data)
            else:
                if compress:
                    np.savez_compressed(filepath, data=data)
                else:
                    np.savez(filepath, data=data)

        elif filepath.suffix == '.pkl':
            # Pickle data
            if compress:
                with gzip.open(str(filepath) + '.gz', 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                filepath = Path(str(filepath) + '.gz')
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # JSON fallback
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        return filepath

    def _calculate_checksum(self, state: Dict[str, Any]) -> str:
        """Calculate checksum for state validation."""
        # Convert state to bytes for hashing
        state_bytes = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(state_bytes).hexdigest()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond retention limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by timestamp
            sorted_history = sorted(self.checkpoint_history,
                                  key=lambda x: x['timestamp'])

            # Remove oldest checkpoints
            to_remove = sorted_history[:-self.max_checkpoints]

            for checkpoint in to_remove:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    try:
                        shutil.rmtree(checkpoint_path)
                        logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_id']}")
                    except Exception as e:
                        logger.error(f"Failed to remove checkpoint: {e}")

                # Remove from history
                self.checkpoint_history.remove(checkpoint)

    def _update_statistics(self, size_bytes: int, time_seconds: float):
        """Update checkpoint statistics."""
        stats = self.checkpoint_stats

        # Update counters
        stats['total_checkpoints'] += 1

        # Update averages
        n = stats['total_checkpoints']
        stats['average_size_mb'] = (
            (stats['average_size_mb'] * (n - 1) + size_bytes / (1024 * 1024)) / n
        )
        stats['average_time_seconds'] = (
            (stats['average_time_seconds'] * (n - 1) + time_seconds) / n
        )

    async def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Loaded state dictionary
        """
        # Find checkpoint in history
        checkpoint_info = None
        for cp in self.checkpoint_history:
            if cp['checkpoint_id'] == checkpoint_id:
                checkpoint_info = cp
                break

        if not checkpoint_info:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint_path = Path(checkpoint_info['path'])

        # Load metadata
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load state components
        state = {}

        # Load quantum states
        quantum_path = checkpoint_path / "quantum_states.npz"
        if quantum_path.exists():
            state['quantum_states'] = dict(np.load(quantum_path))

        # Load memory state
        memory_path = checkpoint_path / "memory_state.pkl"
        if not memory_path.exists():
            memory_path = checkpoint_path / "memory_state.pkl.gz"

        if memory_path.exists():
            if memory_path.suffix == '.gz':
                with gzip.open(memory_path, 'rb') as f:
                    state['memory_state'] = pickle.load(f)
            else:
                with open(memory_path, 'rb') as f:
                    state['memory_state'] = pickle.load(f)

        # Load component states
        state['component_states'] = {}
        for filepath in checkpoint_path.glob("*_state.pkl*"):
            component_name = filepath.stem.replace('_state', '')

            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rb') as f:
                    state['component_states'][component_name] = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    state['component_states'][component_name] = pickle.load(f)

        # Validate checksum
        calculated_checksum = self._calculate_checksum(state)
        if calculated_checksum != metadata.get('checksum'):
            logger.warning(f"Checksum mismatch for checkpoint {checkpoint_id}")

        logger.info(f"Loaded checkpoint {checkpoint_id}")

        return state

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get ID of the latest checkpoint."""
        if not self.checkpoint_history:
            return None

        # Sort by timestamp
        sorted_history = sorted(self.checkpoint_history,
                              key=lambda x: x['timestamp'],
                              reverse=True)

        return sorted_history[0]['checkpoint_id']

    def list_checkpoints(self, checkpoint_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            checkpoint_type: Filter by checkpoint type

        Returns:
            List of checkpoint information
        """
        checkpoints = self.checkpoint_history.copy()

        if checkpoint_type:
            checkpoints = [cp for cp in checkpoints if cp.get('type') == checkpoint_type]

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        return checkpoints

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific checkpoint."""
        for cp in self.checkpoint_history:
            if cp['checkpoint_id'] == checkpoint_id:
                return cp.copy()
        return None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete specific checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            Success status
        """
        with self.checkpoint_lock:
            # Find checkpoint
            checkpoint_info = None
            for cp in self.checkpoint_history:
                if cp['checkpoint_id'] == checkpoint_id:
                    checkpoint_info = cp
                    break

            if not checkpoint_info:
                return False

            # Remove from disk
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint: {e}")
                    return False

            # Remove from history
            self.checkpoint_history.remove(checkpoint_info)
            self._save_metadata()

            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        stats = self.checkpoint_stats.copy()
        stats['total_checkpoints_stored'] = len(self.checkpoint_history)
        stats['checkpoint_types'] = {}

        # Count by type
        for cp in self.checkpoint_history:
            cp_type = cp.get('type', 'unknown')
            stats['checkpoint_types'][cp_type] = stats['checkpoint_types'].get(cp_type, 0) + 1

        # Calculate total storage
        total_storage_mb = sum(cp.get('size_mb', 0) for cp in self.checkpoint_history)
        stats['total_storage_mb'] = total_storage_mb

        return stats
