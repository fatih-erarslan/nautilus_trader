"""
State recovery and validation for QBMIA.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import json
from datetime import datetime, timedelta
import traceback
import hashlib

logger = logging.getLogger(__name__)

class RecoveryHandler:
    """
    Handles state recovery, validation, and migration for QBMIA.
    """

    def __init__(self, checkpoint_manager: Any, state_serializer: Any):
        """
        Initialize recovery handler.

        Args:
            checkpoint_manager: Checkpoint manager instance
            state_serializer: State serializer instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.state_serializer = state_serializer

        # Recovery configuration
        self.max_recovery_attempts = 3
        self.validation_strict = False
        self.auto_repair = True

        # Recovery statistics
        self.recovery_stats = {
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'auto_repairs': 0,
            'validation_failures': 0
        }

    async def recover_state(self, checkpoint_id: Optional[str] = None,
                          recovery_strategy: str = 'latest') -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Recover state from checkpoint with validation and repair.

        Args:
            checkpoint_id: Specific checkpoint to recover
            recovery_strategy: Strategy ('latest', 'stable', 'fallback')

        Returns:
            Tuple of (recovered_state, recovery_metadata)
        """
        start_time = time.time()
        recovery_metadata = {
            'strategy': recovery_strategy,
            'attempts': 0,
            'repairs_applied': [],
            'warnings': []
        }

        # Determine checkpoint to recover
        if checkpoint_id is None:
            checkpoint_id = self._select_checkpoint(recovery_strategy)
            if checkpoint_id is None:
                raise ValueError("No valid checkpoint found for recovery")

        # Recovery attempts
        for attempt in range(self.max_recovery_attempts):
            recovery_metadata['attempts'] = attempt + 1

            try:
                # Load checkpoint
                logger.info(f"Attempting recovery from checkpoint {checkpoint_id} (attempt {attempt + 1})")
                state = await self.checkpoint_manager.load_checkpoint(checkpoint_id)

                # Validate state
                validation_result = self._validate_state(state)

                if validation_result['valid']:
                    # State is valid
                    self.recovery_stats['successful_recoveries'] += 1
                    recovery_metadata['recovery_time'] = time.time() - start_time
                    recovery_metadata['status'] = 'success'

                    logger.info(f"Successfully recovered state from {checkpoint_id}")
                    return state, recovery_metadata

                else:
                    # State validation failed
                    self.recovery_stats['validation_failures'] += 1
                    recovery_metadata['warnings'].extend(validation_result['errors'])

                    if self.auto_repair:
                        # Attempt auto-repair
                        repaired_state = self._repair_state(state, validation_result)

                        if repaired_state:
                            # Re-validate repaired state
                            revalidation = self._validate_state(repaired_state)

                            if revalidation['valid']:
                                self.recovery_stats['auto_repairs'] += 1
                                recovery_metadata['repairs_applied'] = validation_result['errors']
                                recovery_metadata['status'] = 'success_with_repairs'

                                logger.info(f"Successfully recovered and repaired state from {checkpoint_id}")
                                return repaired_state, recovery_metadata

                    # If strict validation or repair failed, try fallback
                    if self.validation_strict or attempt == self.max_recovery_attempts - 1:
                        # Try fallback checkpoint
                        fallback_id = self._get_fallback_checkpoint(checkpoint_id)
                        if fallback_id:
                            checkpoint_id = fallback_id
                            recovery_metadata['warnings'].append(f"Falling back to checkpoint {fallback_id}")
                        else:
                            raise ValueError(f"State validation failed: {validation_result['errors']}")

            except Exception as e:
                logger.error(f"Recovery attempt {attempt + 1} failed: {e}")

                if attempt == self.max_recovery_attempts - 1:
                    # Final attempt failed
                    self.recovery_stats['failed_recoveries'] += 1
                    recovery_metadata['status'] = 'failed'
                    recovery_metadata['error'] = str(e)
                    recovery_metadata['traceback'] = traceback.format_exc()
                    raise

                # Try alternative checkpoint
                fallback_id = self._get_fallback_checkpoint(checkpoint_id)
                if fallback_id:
                    checkpoint_id = fallback_id
                    recovery_metadata['warnings'].append(f"Recovery failed, trying fallback {fallback_id}")
                else:
                    time.sleep(1)  # Brief pause before retry

    def _select_checkpoint(self, strategy: str) -> Optional[str]:
        """Select checkpoint based on recovery strategy."""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        if not checkpoints:
            return None

        if strategy == 'latest':
            # Most recent checkpoint
            return checkpoints[0]['checkpoint_id']

        elif strategy == 'stable':
            # Most recent periodic checkpoint
            periodic_checkpoints = [cp for cp in checkpoints if cp.get('type') == 'periodic']
            if periodic_checkpoints:
                return periodic_checkpoints[0]['checkpoint_id']
            return checkpoints[0]['checkpoint_id']

        elif strategy == 'fallback':
            # Second most recent checkpoint
            if len(checkpoints) > 1:
                return checkpoints[1]['checkpoint_id']
            return checkpoints[0]['checkpoint_id']

        else:
            return checkpoints[0]['checkpoint_id']

    def _validate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate recovered state for completeness and consistency.

        Args:
            state: State to validate

        Returns:
            Validation result with errors
        """
        errors = []
        warnings = []

        # Check required components
        required_components = [
            'agent_id',
            'config',
            'quantum_states',
            'memory_state',
            'component_states'
        ]

        for component in required_components:
            if component not in state:
                errors.append(f"Missing required component: {component}")

        # Validate quantum states
        if 'quantum_states' in state:
            quantum_validation = self._validate_quantum_states(state['quantum_states'])
            errors.extend(quantum_validation['errors'])
            warnings.extend(quantum_validation['warnings'])

        # Validate memory state
        if 'memory_state' in state:
            memory_validation = self._validate_memory_state(state['memory_state'])
            errors.extend(memory_validation['errors'])
            warnings.extend(memory_validation['warnings'])

        # Validate component states
        if 'component_states' in state:
            for component_name, component_state in state['component_states'].items():
                component_validation = self._validate_component_state(
                    component_name, component_state
                )
                errors.extend(component_validation['errors'])
                warnings.extend(component_validation['warnings'])

        # Check state consistency
        consistency_check = self._check_state_consistency(state)
        errors.extend(consistency_check['errors'])
        warnings.extend(consistency_check['warnings'])

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _validate_quantum_states(self, quantum_states: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate quantum state components."""
        errors = []
        warnings = []

        # Check state vector normalization
        for state_name, state_data in quantum_states.items():
            if isinstance(state_data, np.ndarray):
                norm = np.linalg.norm(state_data)
                if abs(norm - 1.0) > 1e-6:
                    warnings.append(f"Quantum state '{state_name}' not normalized: {norm}")

                # Check for NaN or Inf
                if np.any(np.isnan(state_data)) or np.any(np.isinf(state_data)):
                    errors.append(f"Quantum state '{state_name}' contains NaN or Inf values")

        return {'errors': errors, 'warnings': warnings}

    def _validate_memory_state(self, memory_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate memory state components."""
        errors = []
        warnings = []

        # Check memory structure
        if 'capacity' not in memory_state:
            errors.append("Memory state missing capacity")

        if 'memory_index' in memory_state:
            if memory_state['memory_index'] < 0:
                errors.append("Invalid memory index")
            elif 'capacity' in memory_state and memory_state['memory_index'] > memory_state['capacity']:
                warnings.append("Memory index exceeds capacity")

        # Check memory arrays
        if 'long_term_memory' in memory_state:
            memory_array = np.array(memory_state['long_term_memory'])
            if np.any(np.isnan(memory_array)):
                warnings.append("Long-term memory contains NaN values")

        return {'errors': errors, 'warnings': warnings}

    def _validate_component_state(self, component_name: str,
                                component_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate individual component state."""
        errors = []
        warnings = []

        # Component-specific validation
        if component_name == 'machiavellian':
            if 'sensitivity' not in component_state:
                warnings.append(f"{component_name}: Missing sensitivity parameter")

        elif component_name == 'temporal_nash':
            if 'memory_decay' not in component_state:
                warnings.append(f"{component_name}: Missing memory decay parameter")

        # Add more component-specific validations...

        return {'errors': errors, 'warnings': warnings}

    def _check_state_consistency(self, state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check overall state consistency."""
        errors = []
        warnings = []

        # Check timestamp consistency
        if 'timestamp' in state:
            try:
                timestamp = datetime.fromisoformat(state['timestamp'])
                age = datetime.utcnow() - timestamp

                if age > timedelta(days=30):
                    warnings.append(f"State is {age.days} days old")

                if age < timedelta(seconds=-60):  # Future timestamp
                    errors.append("State timestamp is in the future")

            except Exception:
                errors.append("Invalid timestamp format")

        # Check version compatibility
        if 'version' in state:
            if not self._check_version_compatibility(state['version']):
                warnings.append(f"State version {state['version']} may have compatibility issues")

        return {'errors': errors, 'warnings': warnings}

    def _repair_state(self, state: Dict[str, Any],
                     validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair invalid state.

        Args:
            state: Invalid state
            validation_result: Validation errors

        Returns:
            Repaired state or None if unrepairable
        """
        repaired = state.copy()
        repairs_applied = []

        for error in validation_result['errors']:
            if "Missing required component" in error:
                # Add default component
                component = error.split(': ')[1]
                if component == 'quantum_states':
                    repaired['quantum_states'] = {}
                    repairs_applied.append(f"Added empty {component}")
                elif component == 'memory_state':
                    repaired['memory_state'] = {
                        'capacity': 10000,
                        'memory_index': 0,
                        'long_term_memory': []
                    }
                    repairs_applied.append(f"Added default {component}")
                elif component == 'component_states':
                    repaired['component_states'] = {}
                    repairs_applied.append(f"Added empty {component}")

            elif "not normalized" in error:
                # Renormalize quantum states
                for state_name, state_data in repaired.get('quantum_states', {}).items():
                    if isinstance(state_data, np.ndarray):
                        norm = np.linalg.norm(state_data)
                        if norm > 0:
                            repaired['quantum_states'][state_name] = state_data / norm
                            repairs_applied.append(f"Renormalized {state_name}")

            elif "contains NaN" in error:
                # Replace NaN values
                if "quantum state" in error:
                    state_name = error.split("'")[1]
                    if state_name in repaired.get('quantum_states', {}):
                        state_data = repaired['quantum_states'][state_name]
                        repaired['quantum_states'][state_name] = np.nan_to_num(state_data, 0)
                        repairs_applied.append(f"Replaced NaN in {state_name}")

        logger.info(f"Applied {len(repairs_applied)} repairs to state")

        return repaired if repairs_applied else None

    def _get_fallback_checkpoint(self, current_id: str) -> Optional[str]:
        """Get fallback checkpoint ID."""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        # Find current checkpoint index
        current_idx = None
        for i, cp in enumerate(checkpoints):
            if cp['checkpoint_id'] == current_id:
                current_idx = i
                break

        if current_idx is not None and current_idx < len(checkpoints) - 1:
            # Return next older checkpoint
            return checkpoints[current_idx + 1]['checkpoint_id']

        return None

    def _check_version_compatibility(self, version: str) -> bool:
        """Check if state version is compatible."""
        # Simple major version check
        current_major = 1  # Current version 1.x.x

        try:
            state_major = int(version.split('.')[0])
            return state_major == current_major
        except:
            return False

    async def migrate_state(self, state: Dict[str, Any],
                          from_version: str, to_version: str) -> Dict[str, Any]:
        """
        Migrate state between versions.

        Args:
            state: State to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated state
        """
        logger.info(f"Migrating state from {from_version} to {to_version}")

        # Version-specific migrations
        if from_version == "0.9.0" and to_version.startswith("1."):
            state = self._migrate_0_9_to_1_0(state)

        # Add more migration paths as needed

        return state

    def _migrate_0_9_to_1_0(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 0.9.0 to 1.0.0."""
        migrated = state.copy()

        # Example migrations
        # Rename old fields
        if 'old_field' in migrated:
            migrated['new_field'] = migrated.pop('old_field')

        # Add new required fields
        if 'version' not in migrated:
            migrated['version'] = '1.0.0'

        return migrated

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery handler statistics."""
        total_attempts = (self.recovery_stats['successful_recoveries'] +
                         self.recovery_stats['failed_recoveries'])

        success_rate = (self.recovery_stats['successful_recoveries'] /
                       max(1, total_attempts))

        return {
            **self.recovery_stats,
            'total_recovery_attempts': total_attempts,
            'success_rate': success_rate,
            'auto_repair_rate': (self.recovery_stats['auto_repairs'] /
                               max(1, self.recovery_stats['validation_failures']))
        }
