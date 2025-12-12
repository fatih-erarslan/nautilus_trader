"""
Biological memory patterns implementation for QBMIA.
"""

import numpy as np
import numba as nb
from typing import Dict, Any, List, Optional, Tuple, Deque
import logging
from collections import deque
from scipy import signal

logger = logging.getLogger(__name__)

@nb.jit(nopython=True, fastmath=True, cache=True)
def _consolidate_memory_numba(short_term: np.ndarray, long_term: np.ndarray,
                            consolidation_rate: float) -> np.ndarray:
    """
    Numba-accelerated memory consolidation from short-term to long-term.

    Args:
        short_term: Short-term memory buffer
        long_term: Long-term memory buffer
        consolidation_rate: Rate of memory consolidation

    Returns:
        Updated long-term memory
    """
    # Weighted combination with decay
    consolidated = long_term * (1 - consolidation_rate) + short_term * consolidation_rate

    # Apply forgetting curve to older memories
    forgetting_factor = 0.99
    for i in range(len(consolidated)):
        age_factor = forgetting_factor ** (i / 100.0)
        consolidated[i] *= age_factor

    return consolidated

@nb.jit(nopython=True, fastmath=True, cache=True)
def _pattern_similarity_numba(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """
    Calculate similarity between two patterns.

    Args:
        pattern1: First pattern
        pattern2: Second pattern

    Returns:
        Similarity score [0, 1]
    """
    # Cosine similarity
    dot_product = np.dot(pattern1, pattern2)
    norm1 = np.linalg.norm(pattern1)
    norm2 = np.linalg.norm(pattern2)

    if norm1 * norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    return max(0.0, min(1.0, similarity))

class BiologicalMemory:
    """
    Biologically-inspired memory system with short-term and long-term storage.
    """

    def __init__(self, capacity: int = 10000, hw_optimizer: Any = None):
        """
        Initialize biological memory system.

        Args:
            capacity: Maximum memory capacity
            hw_optimizer: Hardware optimizer for acceleration
        """
        self.capacity = capacity
        self.hw_optimizer = hw_optimizer

        # Memory structures
        self.short_term_memory = deque(maxlen=100)  # Recent experiences
        self.long_term_memory = np.zeros((capacity, 128))  # Consolidated patterns
        self.episodic_memory = deque(maxlen=1000)  # Specific episodes

        # Memory indices
        self.memory_index = 0
        self.pattern_index = {}  # Fast pattern lookup

        # Memory parameters
        self.consolidation_rate = 0.1
        self.recall_threshold = 0.7
        self.attention_weights = np.ones(128) / 128

        # Performance tracking
        self.memory_stats = {
            'total_stored': 0,
            'successful_recalls': 0,
            'failed_recalls': 0,
            'consolidations': 0
        }

        logger.info(f"Biological memory initialized with capacity {capacity}")

    def store_experience(self, experience: Dict[str, Any]):
        """
        Store new experience in memory.

        Args:
            experience: Experience data to store
        """
        # Extract features from experience
        features = self._extract_features(experience)

        # Store in short-term memory
        self.short_term_memory.append({
            'features': features,
            'timestamp': experience.get('timestamp'),
            'importance': self._calculate_importance(experience),
            'raw_data': experience
        })

        # Check for consolidation opportunity
        if len(self.short_term_memory) >= 10:
            self._consolidate_memories()

        # Store significant episodes
        if self._is_significant_episode(experience):
            self.episodic_memory.append(experience)

        self.memory_stats['total_stored'] += 1

    def _extract_features(self, experience: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from experience."""
        features = np.zeros(128)

        # Market features
        if 'market_snapshot' in experience:
            snapshot = experience['market_snapshot']
            features[0] = snapshot.get('price', 0) / 10000  # Normalized price
            features[1] = snapshot.get('volume', 0) / 1e6   # Normalized volume
            features[2] = snapshot.get('volatility', 0)
            features[3] = snapshot.get('trend', 0)

        # Decision features
        if 'integrated_decision' in experience:
            decision = experience['integrated_decision']
            if decision:
                actions = ['buy', 'sell', 'hold', 'wait']
                action_idx = actions.index(decision.get('action', 'hold'))
                features[10:14] = np.eye(4)[action_idx]  # One-hot encoding
                features[14] = decision.get('confidence', 0)

        # Component results features
        if 'component_results' in experience:
            results = experience['component_results']

            # Quantum Nash features
            if 'quantum_nash' in results:
                qn = results['quantum_nash']
                if 'equilibrium' in qn:
                    features[20] = qn['equilibrium'].get('convergence_score', 0)

            # Machiavellian features
            if 'machiavellian' in results:
                mach = results['machiavellian']
                if 'manipulation_detected' in mach:
                    features[30] = float(mach['manipulation_detected'].get('detected', False))
                    features[31] = mach['manipulation_detected'].get('confidence', 0)

            # Add more component features...

        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def _calculate_importance(self, experience: Dict[str, Any]) -> float:
        """Calculate importance score for memory prioritization."""
        importance = 0.0

        # High confidence decisions are important
        if 'integrated_decision' in experience:
            decision = experience['integrated_decision']
            if decision:
                importance += decision.get('confidence', 0) * 0.3

        # Unusual market conditions are important
        if 'market_snapshot' in experience:
            snapshot = experience['market_snapshot']
            volatility = snapshot.get('volatility', 0)
            if volatility > 0.03:  # High volatility
                importance += 0.3

        # Successful outcomes are important (would need outcome tracking)
        # importance += outcome_success * 0.4

        return min(1.0, importance)

    def _is_significant_episode(self, experience: Dict[str, Any]) -> bool:
        """Determine if experience is significant enough for episodic memory."""
        # High importance experiences
        importance = self._calculate_importance(experience)
        if importance > 0.7:
            return True

        # Manipulation detected
        if 'component_results' in experience:
            results = experience['component_results']
            if 'machiavellian' in results:
                if results['machiavellian'].get('manipulation_detected', {}).get('detected', False):
                    return True

        # High confidence decisions
        if 'integrated_decision' in experience:
            decision = experience['integrated_decision']
            if decision and decision.get('confidence', 0) > 0.8:
                return True

        return False

    def _consolidate_memories(self):
        """Consolidate short-term memories into long-term storage."""
        if len(self.short_term_memory) < 5:
            return

        # Extract patterns from short-term memory
        patterns = []
        importances = []

        for memory in list(self.short_term_memory)[-10:]:  # Last 10 memories
            patterns.append(memory['features'])
            importances.append(memory['importance'])

        patterns = np.array(patterns)
        importances = np.array(importances)

        # Weighted average pattern
        weights = importances / (np.sum(importances) + 1e-8)
        consolidated_pattern = np.sum(patterns * weights[:, np.newaxis], axis=0)

        # Store in long-term memory
        if self.memory_index < self.capacity:
            self.long_term_memory[self.memory_index] = consolidated_pattern
            self.memory_index += 1
        else:
            # Overwrite oldest memory
            self.memory_index = 0
            self.long_term_memory[self.memory_index] = consolidated_pattern

        # Update pattern index for fast lookup
        pattern_hash = hash(consolidated_pattern.tobytes())
        self.pattern_index[pattern_hash] = self.memory_index - 1

        self.memory_stats['consolidations'] += 1

        # Apply biological consolidation
        if self.hw_optimizer and hasattr(self.hw_optimizer, 'accelerate'):
            # Use hardware acceleration if available
            self.long_term_memory = self.hw_optimizer.accelerate(
                _consolidate_memory_numba,
                self.long_term_memory[:self.memory_index],
                consolidated_pattern,
                self.consolidation_rate
            )
        else:
            # CPU fallback
            self.long_term_memory[:self.memory_index] = _consolidate_memory_numba(
                self.long_term_memory[:self.memory_index],
                np.tile(consolidated_pattern, (self.memory_index, 1)),
                self.consolidation_rate
            )

    def recall_similar_experiences(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recall similar experiences from memory.

        Args:
            query: Query experience to match
            top_k: Number of similar experiences to return

        Returns:
            List of similar experiences
        """
        # Extract query features
        query_features = self._extract_features(query)

        # Search in long-term memory
        similarities = []

        for i in range(min(self.memory_index, self.capacity)):
            similarity = _pattern_similarity_numba(query_features, self.long_term_memory[i])
            if similarity > self.recall_threshold:
                similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        recalled = []
        for idx, similarity in similarities[:top_k]:
            # Try to find associated episodic memory
            recalled.append({
                'pattern_idx': idx,
                'similarity': similarity,
                'pattern': self.long_term_memory[idx]
            })

        # Also check recent short-term memory
        for memory in list(self.short_term_memory)[-20:]:
            similarity = _pattern_similarity_numba(query_features, memory['features'])
            if similarity > self.recall_threshold:
                recalled.append({
                    'type': 'short_term',
                    'similarity': similarity,
                    'data': memory['raw_data']
                })

        # Update stats
        if len(recalled) > 0:
            self.memory_stats['successful_recalls'] += 1
        else:
            self.memory_stats['failed_recalls'] += 1

        return recalled[:top_k]

    def get_recent_patterns(self, window: int = 50) -> List[np.ndarray]:
        """Get recent memory patterns for analysis."""
        patterns = []

        # From short-term memory
        for memory in list(self.short_term_memory)[-window:]:
            patterns.append(memory['features'])

        # From recent long-term consolidations
        start_idx = max(0, self.memory_index - window)
        for i in range(start_idx, self.memory_index):
            patterns.append(self.long_term_memory[i])

        return patterns

    def apply_attention(self, focus_areas: List[str]):
        """
        Apply attention mechanism to prioritize certain features.

        Args:
            focus_areas: Areas to focus on (e.g., 'volatility', 'manipulation')
        """
        # Reset attention weights
        self.attention_weights = np.ones(128) / 128

        # Increase weights for focus areas
        for area in focus_areas:
            if area == 'volatility':
                self.attention_weights[2] *= 2.0  # Volatility feature
                self.attention_weights[40:50] *= 1.5  # Volatility-related features
            elif area == 'manipulation':
                self.attention_weights[30:40] *= 2.0  # Manipulation features
            elif area == 'quantum':
                self.attention_weights[20:30] *= 1.5  # Quantum features

        # Normalize
        self.attention_weights /= np.sum(self.attention_weights)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory state."""
        return {
            'short_term_size': len(self.short_term_memory),
            'long_term_size': self.memory_index,
            'episodic_size': len(self.episodic_memory),
            'capacity_used': self.memory_index / self.capacity,
            'stats': self.memory_stats.copy(),
            'attention_focus': self._get_attention_focus()
        }

    def _get_attention_focus(self) -> List[str]:
        """Identify current attention focus areas."""
        focus = []

        # Check which features have high attention weights
        if self.attention_weights[2] > 0.02:
            focus.append('volatility')
        if np.mean(self.attention_weights[30:40]) > 0.01:
            focus.append('manipulation')
        if np.mean(self.attention_weights[20:30]) > 0.01:
            focus.append('quantum_patterns')

        return focus

    def serialize(self) -> Dict[str, Any]:
        """Serialize memory state."""
        return {
            'capacity': self.capacity,
            'memory_index': self.memory_index,
            'long_term_memory': self.long_term_memory[:self.memory_index].tolist(),
            'attention_weights': self.attention_weights.tolist(),
            'consolidation_rate': self.consolidation_rate,
            'memory_stats': self.memory_stats.copy()
        }

    def restore(self, state: Dict[str, Any]):
        """Restore memory from serialized state."""
        self.capacity = state.get('capacity', self.capacity)
        self.memory_index = state.get('memory_index', 0)

        if 'long_term_memory' in state:
            memory_data = np.array(state['long_term_memory'])
            self.long_term_memory[:len(memory_data)] = memory_data

        if 'attention_weights' in state:
            self.attention_weights = np.array(state['attention_weights'])

        self.consolidation_rate = state.get('consolidation_rate', self.consolidation_rate)
        self.memory_stats = state.get('memory_stats', self.memory_stats).copy()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'short_term_memory_mb': len(self.short_term_memory) * 128 * 8 / (1024 * 1024),
            'long_term_memory_mb': self.memory_index * 128 * 8 / (1024 * 1024),
            'total_memory_mb': (len(self.short_term_memory) + self.memory_index) * 128 * 8 / (1024 * 1024),
            'capacity_percentage': (self.memory_index / self.capacity) * 100
        }
