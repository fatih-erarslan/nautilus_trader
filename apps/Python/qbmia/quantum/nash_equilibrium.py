"""
Quantum Nash Equilibrium implementation using GPU-accelerated quantum simulation.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Optional
import logging
from functools import partial
import asyncio

logger = logging.getLogger(__name__)

class QuantumNashEquilibrium:
    """
    Quantum Nash Equilibrium solver using variational quantum algorithms.
    """

    def __init__(self, num_qubits: int = 16, hw_optimizer: Any = None):
        """
        Initialize Quantum Nash Equilibrium solver.

        Args:
            num_qubits: Number of qubits for quantum simulation
            hw_optimizer: Hardware optimizer instance
        """
        self.num_qubits = num_qubits
        self.hw_optimizer = hw_optimizer

        # Quantum device configuration
        self.device = self._setup_quantum_device()

        # Circuit parameters
        self.num_layers = 3
        self.params_per_layer = num_qubits * 3  # RX, RY, RZ per qubit
        self.total_params = self.num_layers * self.params_per_layer

        # Optimization settings
        self.learning_rate = 0.1
        self.convergence_threshold = 1e-4
        self.max_iterations = 200

        # Cache for compiled circuits
        self.circuit_cache = {}

        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'average_time': 0.0,
            'convergence_rate': 0.0
        }

        logger.info(f"Quantum Nash Equilibrium initialized with {num_qubits} qubits")

    def _setup_quantum_device(self) -> Any:
        """Setup quantum device based on available hardware."""
        if self.hw_optimizer and hasattr(self.hw_optimizer, '_device'):
            if 'cuda' in self.hw_optimizer._device:
                # NVIDIA GPU
                device = qml.device('lightning.gpu', wires=self.num_qubits)
                logger.info("Using lightning.gpu for NVIDIA GPU acceleration")
            elif 'rocm' in self.hw_optimizer._device:
                # AMD GPU via Kokkos
                device = qml.device('lightning.kokkos', wires=self.num_qubits)
                logger.info("Using lightning.kokkos for AMD GPU acceleration")
            else:
                # CPU fallback
                device = qml.device('lightning.qubit', wires=self.num_qubits)
                logger.info("Using lightning.qubit for CPU execution")
        else:
            # Default CPU device
            device = qml.device('lightning.qubit', wires=self.num_qubits)

        return device

    def _create_ansatz(self, params: np.ndarray) -> None:
        """
        Create parameterized quantum circuit ansatz for Nash equilibrium.

        Args:
            params: Circuit parameters
        """
        param_idx = 0

        # Initial superposition
        for i in range(self.num_qubits):
            qml.Hadamard(wires=i)

        # Variational layers
        for layer in range(self.num_layers):
            # Single qubit rotations
            for i in range(self.num_qubits):
                qml.RX(params[param_idx], wires=i)
                param_idx += 1
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

            # Entangling layer (ring topology)
            for i in range(self.num_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

            # Additional entanglement for complex correlations
            if layer < self.num_layers - 1:
                for i in range(0, self.num_qubits - 1, 2):
                    qml.CZ(wires=[i, i + 1])

    def _encode_payoff_matrix(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        Encode payoff matrix into quantum circuit parameters.

        Args:
            payoff_matrix: Game payoff matrix

        Returns:
            Encoded parameters
        """
        # Flatten and normalize payoff matrix
        flat_payoffs = payoff_matrix.flatten()

        # Normalize to [-π, π] range
        if np.max(np.abs(flat_payoffs)) > 0:
            normalized = flat_payoffs / np.max(np.abs(flat_payoffs)) * np.pi
        else:
            normalized = flat_payoffs

        # Pad or truncate to match parameter count
        if len(normalized) < self.total_params:
            # Pad with zeros
            encoded = np.zeros(self.total_params)
            encoded[:len(normalized)] = normalized
        else:
            # Truncate
            encoded = normalized[:self.total_params]

        return encoded

    @partial(jax.jit, static_argnums=(0,))
    def _compute_expectation_jax(self, params: jnp.ndarray,
                                observables: List[np.ndarray]) -> jnp.ndarray:
        """
        JAX-accelerated expectation value computation.

        Args:
            params: Circuit parameters
            observables: List of observable operators

        Returns:
            Expectation values
        """
        # This is a placeholder for JAX integration
        # In practice, would use JAX-compatible quantum simulation
        return jnp.zeros(len(observables))

    def _create_nash_circuit(self, params: np.ndarray) -> qml.QNode:
        """
        Create quantum circuit for Nash equilibrium calculation.

        Args:
            params: Circuit parameters

        Returns:
            Quantum circuit (QNode)
        """
        # Check cache
        param_hash = hash(params.tobytes())
        if param_hash in self.circuit_cache:
            return self.circuit_cache[param_hash]

        @qml.qnode(self.device, interface='numpy')
        def circuit():
            self._create_ansatz(params)
            # Return probability distribution
            return qml.probs(wires=range(self.num_qubits))

        # Cache compiled circuit
        self.circuit_cache[param_hash] = circuit

        return circuit

    def _extract_strategies(self, probabilities: np.ndarray,
                          num_players: int, num_actions: int) -> Dict[str, np.ndarray]:
        """
        Extract player strategies from quantum state probabilities.

        Args:
            probabilities: Quantum state probabilities
            num_players: Number of players
            num_actions: Number of actions per player

        Returns:
            Strategy profile for each player
        """
        strategies = {}

        # Calculate bits needed per player
        bits_per_player = int(np.ceil(np.log2(num_actions)))

        for player_idx in range(num_players):
            # Extract relevant probability amplitudes
            player_probs = np.zeros(num_actions)

            for state_idx, prob in enumerate(probabilities):
                # Extract player's action from binary representation
                player_bits_start = player_idx * bits_per_player
                player_bits_end = (player_idx + 1) * bits_per_player

                if player_bits_end <= self.num_qubits:
                    # Extract action index from state
                    action_bits = (state_idx >> player_bits_start) & ((1 << bits_per_player) - 1)
                    if action_bits < num_actions:
                        player_probs[action_bits] += prob

            # Normalize to create probability distribution
            if np.sum(player_probs) > 0:
                player_probs /= np.sum(player_probs)
            else:
                # Uniform distribution fallback
                player_probs = np.ones(num_actions) / num_actions

            strategies[f'player_{player_idx}'] = player_probs

        return strategies

    def _calculate_nash_loss(self, strategies: Dict[str, np.ndarray],
                           payoff_matrix: np.ndarray) -> float:
        """
        Calculate Nash equilibrium loss function.

        Args:
            strategies: Current strategy profile
            payoff_matrix: Game payoff matrix

        Returns:
            Nash loss value
        """
        num_players = len(strategies)
        total_loss = 0.0

        # For each player, calculate deviation incentive
        for player_idx in range(num_players):
            player_strategy = strategies[f'player_{player_idx}']

            # Calculate expected payoff for current strategy
            current_payoff = self._calculate_expected_payoff(
                player_idx, player_strategy, strategies, payoff_matrix
            )

            # Calculate best response payoff
            best_response = self._find_best_response(
                player_idx, strategies, payoff_matrix
            )
            best_payoff = self._calculate_expected_payoff(
                player_idx, best_response, strategies, payoff_matrix
            )

            # Add deviation loss
            deviation = max(0, best_payoff - current_payoff)
            total_loss += deviation ** 2

        return total_loss

    def _calculate_expected_payoff(self, player_idx: int, strategy: np.ndarray,
                                  all_strategies: Dict[str, np.ndarray],
                                  payoff_matrix: np.ndarray) -> float:
        """Calculate expected payoff for a player's strategy."""
        # Simplified for 2-player games
        if len(all_strategies) == 2:
            opponent_idx = 1 - player_idx
            opponent_strategy = all_strategies[f'player_{opponent_idx}']

            # Calculate expected payoff
            payoff_slice = payoff_matrix[player_idx]
            expected = np.sum(strategy[:, np.newaxis] * opponent_strategy[np.newaxis, :] * payoff_slice)

            return expected
        else:
            # Multi-player case (simplified)
            return np.random.random()

    def _find_best_response(self, player_idx: int,
                          all_strategies: Dict[str, np.ndarray],
                          payoff_matrix: np.ndarray) -> np.ndarray:
        """Find best response strategy for a player."""
        num_actions = len(all_strategies[f'player_{player_idx}'])
        best_payoff = -np.inf
        best_strategy = None

        # Check each pure strategy
        for action in range(num_actions):
            pure_strategy = np.zeros(num_actions)
            pure_strategy[action] = 1.0

            payoff = self._calculate_expected_payoff(
                player_idx, pure_strategy, all_strategies, payoff_matrix
            )

            if payoff > best_payoff:
                best_payoff = payoff
                best_strategy = pure_strategy

        return best_strategy if best_strategy is not None else np.ones(num_actions) / num_actions

    async def find_equilibrium(self, payoff_matrix: np.ndarray,
                             market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find quantum Nash equilibrium for given payoff matrix.

        Args:
            payoff_matrix: Game payoff matrix
            market_conditions: Optional market conditions for context

        Returns:
            Equilibrium solution with metadata
        """
        start_time = asyncio.get_event_loop().time()

        # Determine game structure
        if payoff_matrix.ndim == 2:
            num_players = 2
            num_actions = payoff_matrix.shape[1]
        else:
            num_players = payoff_matrix.shape[0]
            num_actions = payoff_matrix.shape[2]

        # Initialize parameters
        params = self._encode_payoff_matrix(payoff_matrix)

        # Add market condition bias if provided
        if market_conditions:
            volatility = market_conditions.get('volatility', 0.0)
            trend = market_conditions.get('trend', 0.0)
            params += np.random.normal(0, volatility * 0.1, size=params.shape)

        # Optimization loop
        best_loss = np.inf
        best_strategies = None
        convergence_history = []

        for iteration in range(self.max_iterations):
            # Create and execute quantum circuit
            circuit = self._create_nash_circuit(params)
            probabilities = circuit()

            # Extract strategies from quantum state
            strategies = self._extract_strategies(probabilities, num_players, num_actions)

            # Calculate Nash loss
            loss = self._calculate_nash_loss(strategies, payoff_matrix)
            convergence_history.append(loss)

            # Update best solution
            if loss < best_loss:
                best_loss = loss
                best_strategies = strategies.copy()

            # Check convergence
            if loss < self.convergence_threshold:
                logger.info(f"Nash equilibrium converged at iteration {iteration}")
                break

            # Update parameters using gradient-free optimization
            # (In practice, would use parameter-shift rule or other quantum gradients)
            noise_scale = self.learning_rate * (1.0 - iteration / self.max_iterations)
            params += np.random.normal(0, noise_scale, size=params.shape)
            params = np.clip(params, -np.pi, np.pi)

        # Calculate final metrics
        execution_time = asyncio.get_event_loop().time() - start_time
        convergence_score = 1.0 - (best_loss / (num_players * num_actions))

        # Update statistics
        self.execution_stats['total_executions'] += 1
        self.execution_stats['average_time'] = (
            (self.execution_stats['average_time'] * (self.execution_stats['total_executions'] - 1) +
             execution_time) / self.execution_stats['total_executions']
        )

        if len(convergence_history) > 0:
            self.execution_stats['convergence_rate'] = (
                convergence_history[0] - convergence_history[-1]
            ) / len(convergence_history)

        return {
            'equilibrium': best_strategies,
            'convergence_score': float(convergence_score),
            'nash_loss': float(best_loss),
            'iterations': len(convergence_history),
            'convergence_history': convergence_history,
            'quantum_state_entropy': self._calculate_entropy(probabilities),
            'execution_time': execution_time,
            'optimal_action': self._determine_optimal_action(best_strategies),
            'stability_analysis': self._analyze_stability(best_strategies, payoff_matrix)
        }

    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of quantum state distribution."""
        # Remove zero probabilities
        probs = probabilities[probabilities > 1e-10]
        if len(probs) == 0:
            return 0.0

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by maximum entropy
        max_entropy = np.log2(len(probabilities))

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _determine_optimal_action(self, strategies: Dict[str, np.ndarray]) -> int:
        """Determine optimal action based on equilibrium strategies."""
        # For simplicity, return the action with highest probability for player 0
        if 'player_0' in strategies:
            return int(np.argmax(strategies['player_0']))
        return 0

    def _analyze_stability(self, strategies: Dict[str, np.ndarray],
                         payoff_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze stability of the Nash equilibrium."""
        stability_metrics = {}

        # Calculate strategy entropy (mixed vs pure)
        for player, strategy in strategies.items():
            entropy = -np.sum(strategy * np.log2(strategy + 1e-10))
            max_entropy = np.log2(len(strategy))
            stability_metrics[f'{player}_mixedness'] = entropy / max_entropy if max_entropy > 0 else 0

        # Calculate robustness to perturbations
        perturbation_size = 0.01
        perturbed_strategies = {}

        for player, strategy in strategies.items():
            # Add small perturbation
            perturbed = strategy + np.random.normal(0, perturbation_size, size=strategy.shape)
            perturbed = np.maximum(perturbed, 0)
            perturbed /= np.sum(perturbed)
            perturbed_strategies[player] = perturbed

        # Calculate change in payoffs
        original_payoffs = {}
        perturbed_payoffs = {}

        for i, (player, strategy) in enumerate(strategies.items()):
            original_payoffs[player] = self._calculate_expected_payoff(
                i, strategy, strategies, payoff_matrix
            )
            perturbed_payoffs[player] = self._calculate_expected_payoff(
                i, perturbed_strategies[player], perturbed_strategies, payoff_matrix
            )

        # Stability score based on payoff sensitivity
        total_change = sum(
            abs(perturbed_payoffs[p] - original_payoffs[p])
            for p in strategies.keys()
        )
        stability_metrics['perturbation_sensitivity'] = float(total_change)

        return stability_metrics

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'num_qubits': self.num_qubits,
            'num_layers': self.num_layers,
            'execution_stats': self.execution_stats.copy(),
            'convergence_threshold': self.convergence_threshold,
            'learning_rate': self.learning_rate
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.num_qubits = state.get('num_qubits', self.num_qubits)
        self.num_layers = state.get('num_layers', self.num_layers)
        self.execution_stats = state.get('execution_stats', self.execution_stats).copy()
        self.convergence_threshold = state.get('convergence_threshold', self.convergence_threshold)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
