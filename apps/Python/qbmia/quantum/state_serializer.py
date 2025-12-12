"""
Quantum state serialization for efficient storage and transfer.
"""

import numpy as np
import json
import gzip
import h5py
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
from scipy.sparse import coo_matrix, save_npz, load_npz
import msgpack
import msgpack_numpy

# Patch msgpack for numpy support
msgpack_numpy.patch()

logger = logging.getLogger(__name__)

class QuantumStateSerializer:
    """
    Efficient serialization of quantum states for storage and orchestration.
    """

    def __init__(self, compression_level: int = 6):
        """
        Initialize quantum state serializer.

        Args:
            compression_level: Compression level (0-9)
        """
        self.compression_level = compression_level
        self.state_cache = {}
        self.format_handlers = {
            'dense': self._serialize_dense_state,
            'sparse': self._serialize_sparse_state,
            'parameter': self._serialize_parameters,
            'measurement': self._serialize_measurements
        }

    def serialize_state_vector(self, state: np.ndarray,
                             sparse_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Serialize quantum state vector with automatic format selection.

        Args:
            state: Complex state vector
            sparse_threshold: Sparsity threshold for format selection

        Returns:
            Serialized state dictionary
        """
        # Check sparsity
        sparsity = np.sum(np.abs(state) < 1e-10) / len(state)

        if sparsity > sparse_threshold:
            return self._serialize_sparse_state(state)
        else:
            return self._serialize_dense_state(state)

    def _serialize_dense_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Serialize dense state vector."""
        return {
            'format': 'dense',
            'shape': state.shape,
            'dtype': str(state.dtype),
            'real': state.real.astype(np.float32),  # Reduce precision for storage
            'imag': state.imag.astype(np.float32),
            'norm': np.linalg.norm(state)
        }

    def _serialize_sparse_state(self, state: np.ndarray) -> Dict[str, Any]:
        """Serialize sparse state vector."""
        # Find non-zero elements
        nonzero_mask = np.abs(state) > 1e-10
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = state[nonzero_mask]

        return {
            'format': 'sparse',
            'shape': state.shape,
            'dtype': str(state.dtype),
            'indices': nonzero_indices.astype(np.uint32),
            'values_real': nonzero_values.real.astype(np.float32),
            'values_imag': nonzero_values.imag.astype(np.float32),
            'nnz': len(nonzero_indices),
            'norm': np.linalg.norm(state)
        }

    def _serialize_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Serialize circuit parameters."""
        serialized = {
            'format': 'parameter',
            'parameters': {}
        }

        for name, values in parameters.items():
            if isinstance(values, np.ndarray):
                serialized['parameters'][name] = {
                    'data': values.astype(np.float32).tolist(),
                    'shape': values.shape
                }
            else:
                serialized['parameters'][name] = values

        return serialized

    def _serialize_measurements(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize measurement results."""
        return {
            'format': 'measurement',
            'shots': measurements.get('shots', 1024),
            'counts': measurements.get('counts', {}),
            'probabilities': measurements.get('probabilities', {}),
            'expectation_values': measurements.get('expectation_values', {})
        }

    def deserialize_state_vector(self, serialized: Dict[str, Any]) -> np.ndarray:
        """
        Deserialize quantum state vector.

        Args:
            serialized: Serialized state dictionary

        Returns:
            Complex state vector
        """
        format_type = serialized.get('format', 'dense')

        if format_type == 'dense':
            real = serialized['real'].astype(np.float64)
            imag = serialized['imag'].astype(np.float64)
            return real + 1j * imag

        elif format_type == 'sparse':
            shape = serialized['shape']
            indices = serialized['indices']
            values = (serialized['values_real'].astype(np.float64) +
                     1j * serialized['values_imag'].astype(np.float64))

            # Reconstruct sparse state
            state = np.zeros(shape, dtype=np.complex128)
            state[indices] = values
            return state

        else:
            raise ValueError(f"Unknown format: {format_type}")

    def save_to_hdf5(self, filename: str, states: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Save multiple quantum states to HDF5 file.

        Args:
            filename: Output filename
            states: Dictionary of states to save
            metadata: Optional metadata
        """
        with h5py.File(filename, 'w') as f:
            # Save metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.attrs[key] = json.dumps(value)

            # Save states
            states_group = f.create_group('states')
            for name, state_data in states.items():
                state_group = states_group.create_group(name)

                # Save format info
                state_group.attrs['format'] = state_data.get('format', 'unknown')

                # Save data based on format
                if state_data['format'] == 'dense':
                    state_group.create_dataset(
                        'real', data=state_data['real'],
                        compression='gzip', compression_opts=self.compression_level
                    )
                    state_group.create_dataset(
                        'imag', data=state_data['imag'],
                        compression='gzip', compression_opts=self.compression_level
                    )

                elif state_data['format'] == 'sparse':
                    state_group.create_dataset('indices', data=state_data['indices'])
                    state_group.create_dataset('values_real', data=state_data['values_real'])
                    state_group.create_dataset('values_imag', data=state_data['values_imag'])
                    state_group.attrs['nnz'] = state_data['nnz']

                # Save common attributes
                state_group.attrs['shape'] = state_data.get('shape', [])
                state_group.attrs['dtype'] = state_data.get('dtype', 'complex128')
                state_group.attrs['norm'] = state_data.get('norm', 1.0)

    def load_from_hdf5(self, filename: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load quantum states from HDF5 file.

        Args:
            filename: Input filename

        Returns:
            Tuple of (states, metadata)
        """
        states = {}
        metadata = {}

        with h5py.File(filename, 'r') as f:
            # Load metadata
            if 'metadata' in f:
                meta_group = f['metadata']
                for key, value in meta_group.attrs.items():
                    try:
                        metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        metadata[key] = value

            # Load states
            if 'states' in f:
                states_group = f['states']
                for name in states_group:
                    state_group = states_group[name]
                    state_data = {
                        'format': state_group.attrs.get('format', 'unknown'),
                        'shape': state_group.attrs.get('shape', []),
                        'dtype': state_group.attrs.get('dtype', 'complex128'),
                        'norm': state_group.attrs.get('norm', 1.0)
                    }

                    # Load data based on format
                    if state_data['format'] == 'dense':
                        state_data['real'] = state_group['real'][:]
                        state_data['imag'] = state_group['imag'][:]

                    elif state_data['format'] == 'sparse':
                        state_data['indices'] = state_group['indices'][:]
                        state_data['values_real'] = state_group['values_real'][:]
                        state_data['values_imag'] = state_group['values_imag'][:]
                        state_data['nnz'] = state_group.attrs.get('nnz', 0)

                    states[name] = state_data

        return states, metadata

    def to_msgpack(self, states: Dict[str, Any]) -> bytes:
        """
        Serialize states to MessagePack format for IPC.

        Args:
            states: States to serialize

        Returns:
            MessagePack bytes
        """
        # Convert numpy arrays to msgpack-compatible format
        packed_states = {}
        for name, state_data in states.items():
            packed = state_data.copy()

            # msgpack-numpy handles numpy arrays automatically
            packed_states[name] = packed

        return msgpack.packb(packed_states, use_bin_type=True)

    def from_msgpack(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize states from MessagePack format.

        Args:
            data: MessagePack bytes

        Returns:
            Deserialized states
        """
        return msgpack.unpackb(data, raw=False)

    def estimate_memory_usage(self, state: np.ndarray, format: str = 'auto') -> int:
        """
        Estimate memory usage for storing a quantum state.

        Args:
            state: Quantum state vector
            format: Storage format ('dense', 'sparse', 'auto')

        Returns:
            Estimated memory usage in bytes
        """
        if format == 'auto':
            sparsity = np.sum(np.abs(state) < 1e-10) / len(state)
            format = 'sparse' if sparsity > 0.9 else 'dense'

        if format == 'dense':
            # Complex128 = 16 bytes per element, float32 storage = 8 bytes
            return state.size * 8 + 100  # Plus metadata overhead

        elif format == 'sparse':
            nnz = np.sum(np.abs(state) > 1e-10)
            # Indices (uint32) + values (2 * float32) + overhead
            return nnz * (4 + 8) + 200

        else:
            raise ValueError(f"Unknown format: {format}")

    def serialize_density_matrix(self, rho: np.ndarray,
                               hermitian_check: bool = True) -> Dict[str, Any]:
        """
        Serialize density matrix efficiently.

        Args:
            rho: Density matrix
            hermitian_check: Verify Hermitian property

        Returns:
            Serialized density matrix
        """
        if hermitian_check:
            if not np.allclose(rho, rho.conj().T):
                logger.warning("Density matrix is not Hermitian")

        # Only store upper triangle for Hermitian matrices
        n = rho.shape[0]
        upper_indices = np.triu_indices(n)
        upper_values = rho[upper_indices]

        return {
            'format': 'density_matrix',
            'n': n,
            'upper_real': upper_values.real.astype(np.float32),
            'upper_imag': upper_values.imag.astype(np.float32),
            'trace': np.trace(rho).real
        }

    def deserialize_density_matrix(self, serialized: Dict[str, Any]) -> np.ndarray:
        """Deserialize density matrix."""
        n = serialized['n']
        upper_real = serialized['upper_real'].astype(np.float64)
        upper_imag = serialized['upper_imag'].astype(np.float64)
        upper_values = upper_real + 1j * upper_imag

        # Reconstruct full matrix
        rho = np.zeros((n, n), dtype=np.complex128)
        upper_indices = np.triu_indices(n)
        rho[upper_indices] = upper_values

        # Fill lower triangle (Hermitian)
        lower_indices = np.tril_indices(n, -1)
        rho[lower_indices] = rho.T[lower_indices].conj()

        return rho

    def serialize_all_states(self) -> Dict[str, Any]:
        """Serialize all cached states."""
        return {name: data for name, data in self.state_cache.items()}

    def restore_all_states(self, states: Dict[str, Any]):
        """Restore all states from serialized data."""
        self.state_cache = states.copy()
