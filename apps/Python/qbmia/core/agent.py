"""
Main QBMIA Agent implementation with quantum-biological learning capabilities.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from ..quantum.nash_equilibrium import QuantumNashEquilibrium
from ..quantum.state_serializer import QuantumStateSerializer
from ..strategy.machiavellian import MachiavellianFramework
from ..strategy.robin_hood import RobinHoodProtocol
from ..strategy.temporal_nash import TemporalBiologicalNash
from ..strategy.antifragile_coalition import AntifragileCoalition
from .memory_patterns import BiologicalMemory
from .hardware_optimizer import QBMIAHardwareOptimizer
from .state_manager import StateManager
from ..orchestration.resource_manager import ResourceManager
from ..utils.performance_metrics import PerformanceTracker

logger = logging.getLogger(__name__)

class QBMIAAgent:
    """
    Quantum-Biological Market Intuition Agent

    Integrates quantum simulations, biological learning, and game-theoretic
    strategies for sophisticated market analysis and decision-making.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QBMIA Agent.

        Args:
            config: Configuration dictionary with parameters for all components
        """
        self.config = config or self._default_config()
        self.agent_id = self.config.get('agent_id', 'QBMIA_001')

        # Initialize logging
        self._setup_logging()

        # Hardware optimization
        self.hw_optimizer = QBMIAHardwareOptimizer(
            force_cpu=self.config.get('force_cpu', False),
            enable_profiling=self.config.get('enable_profiling', True)
        )

        # State management
        self.state_manager = StateManager(
            agent_id=self.agent_id,
            checkpoint_dir=self.config.get('checkpoint_dir', './checkpoints')
        )

        # Resource management for orchestration
        self.resource_manager = ResourceManager(
            agent_id=self.agent_id,
            hw_optimizer=self.hw_optimizer
        )

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Execution state
        self.is_running = False
        self.last_decision = None
        self.market_state = {}

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )

        # Lock for thread-safe operations
        self._lock = threading.RLock()

        logger.info(f"QBMIA Agent {self.agent_id} initialized successfully")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for QBMIA."""
        return {
            'agent_id': 'QBMIA_001',
            'checkpoint_dir': './checkpoints',
            'checkpoint_interval': 300,  # 5 minutes
            'num_qubits': 16,
            'memory_capacity': 10000,
            'learning_rate': 0.001,
            'force_cpu': False,
            'enable_profiling': True,
            'max_workers': 4,
            'log_level': 'INFO'
        }

    def _setup_logging(self):
        """Configure logging for the agent."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s] [{self.agent_id}] %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.setLevel(log_level)
        logger.addHandler(handler)

    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Quantum components
            self.quantum_nash = QuantumNashEquilibrium(
                num_qubits=self.config.get('num_qubits', 16),
                hw_optimizer=self.hw_optimizer
            )

            self.quantum_serializer = QuantumStateSerializer()

            # Strategic components
            self.machiavellian = MachiavellianFramework(
                hw_optimizer=self.hw_optimizer
            )

            self.robin_hood = RobinHoodProtocol(
                wealth_threshold=self.config.get('wealth_threshold', 0.8)
            )

            self.temporal_nash = TemporalBiologicalNash(
                memory_decay=self.config.get('memory_decay', 0.95)
            )

            self.antifragile = AntifragileCoalition(
                volatility_threshold=self.config.get('volatility_threshold', 0.3)
            )

            # Biological memory
            self.memory = BiologicalMemory(
                capacity=self.config.get('memory_capacity', 10000),
                hw_optimizer=self.hw_optimizer
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis using all components.

        Args:
            market_data: Current market state and data

        Returns:
            Analysis results and recommendations
        """
        start_time = time.time()

        with self._lock:
            self.market_state = market_data

        try:
            # Request resources from orchestrator
            resources = await self.resource_manager.request_resources({
                'operation': 'market_analysis',
                'num_qubits': self.config['num_qubits'],
                'estimated_memory': 4096,  # MB
                'priority': 'high'
            })

            # Parallel component analysis
            tasks = [
                self._quantum_nash_analysis(market_data),
                self._machiavellian_analysis(market_data),
                self._robin_hood_analysis(market_data),
                self._temporal_nash_analysis(market_data),
                self._antifragile_analysis(market_data)
            ]

            results = await asyncio.gather(*tasks)

            # Integrate results
            integrated_analysis = self._integrate_analyses(results, market_data)

            # Store in memory
            self.memory.store_experience(integrated_analysis)

            # Track performance
            execution_time = time.time() - start_time
            self.performance_tracker.record_execution(
                'market_analysis', execution_time
            )

            # Release resources
            await self.resource_manager.release_resources(resources)

            return integrated_analysis

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            if 'resources' in locals():
                await self.resource_manager.release_resources(resources)
            raise

    async def _quantum_nash_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Quantum Nash Equilibrium analysis."""
        try:
            # Prepare payoff matrix from market data
            payoff_matrix = self._extract_payoff_matrix(market_data)

            # Find quantum Nash equilibrium
            equilibrium = await self.quantum_nash.find_equilibrium(
                payoff_matrix,
                market_conditions=market_data.get('conditions', {})
            )

            return {
                'type': 'quantum_nash',
                'equilibrium': equilibrium,
                'confidence': equilibrium.get('convergence_score', 0.0)
            }

        except Exception as e:
            logger.error(f"Quantum Nash analysis failed: {e}")
            return {'type': 'quantum_nash', 'error': str(e)}

    async def _machiavellian_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Machiavellian strategic analysis."""
        try:
            # Detect market manipulation
            manipulation = await self.machiavellian.detect_manipulation(
                market_data.get('order_flow', []),
                market_data.get('price_history', [])
            )

            # Generate strategic recommendations
            strategy = await self.machiavellian.generate_strategy(
                manipulation,
                market_data.get('competitors', {})
            )

            return {
                'type': 'machiavellian',
                'manipulation_detected': manipulation,
                'strategy': strategy
            }

        except Exception as e:
            logger.error(f"Machiavellian analysis failed: {e}")
            return {'type': 'machiavellian', 'error': str(e)}

    async def _robin_hood_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Robin Hood protocol analysis."""
        try:
            # Analyze wealth distribution
            wealth_dist = await self.robin_hood.analyze_wealth_distribution(
                market_data.get('participant_wealth', {})
            )

            # Identify intervention opportunities
            interventions = await self.robin_hood.identify_interventions(
                wealth_dist,
                market_data.get('market_structure', {})
            )

            return {
                'type': 'robin_hood',
                'wealth_distribution': wealth_dist,
                'interventions': interventions
            }

        except Exception as e:
            logger.error(f"Robin Hood analysis failed: {e}")
            return {'type': 'robin_hood', 'error': str(e)}

    async def _temporal_nash_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Temporal-Biological Nash analysis."""
        try:
            # Extract time series data
            time_series = market_data.get('time_series', {})

            # Find temporal equilibrium
            temporal_eq = await self.temporal_nash.find_temporal_equilibrium(
                time_series,
                self.memory.get_recent_patterns()
            )

            return {
                'type': 'temporal_nash',
                'equilibrium': temporal_eq,
                'learning_progress': temporal_eq.get('convergence_rate', 0.0)
            }

        except Exception as e:
            logger.error(f"Temporal Nash analysis failed: {e}")
            return {'type': 'temporal_nash', 'error': str(e)}

    async def _antifragile_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Antifragile Coalition analysis."""
        try:
            # Analyze volatility patterns
            volatility = market_data.get('volatility', {})

            # Form antifragile coalitions
            coalitions = await self.antifragile.form_coalitions(
                volatility,
                market_data.get('crisis_indicators', {})
            )

            return {
                'type': 'antifragile',
                'coalitions': coalitions,
                'volatility_benefit': coalitions.get('expected_benefit', 0.0)
            }

        except Exception as e:
            logger.error(f"Antifragile analysis failed: {e}")
            return {'type': 'antifragile', 'error': str(e)}

    def _integrate_analyses(self, results: List[Dict[str, Any]],
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from all analytical components."""
        integrated = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': self.agent_id,
            'market_snapshot': market_data.get('snapshot', {}),
            'component_results': {},
            'integrated_decision': None,
            'confidence': 0.0
        }

        # Process each component result
        for result in results:
            component_type = result.get('type')
            if 'error' not in result:
                integrated['component_results'][component_type] = result

        # Generate integrated decision
        if len(integrated['component_results']) >= 3:  # Need at least 3 successful analyses
            decision = self._generate_decision(integrated['component_results'])
            integrated['integrated_decision'] = decision
            integrated['confidence'] = decision.get('confidence', 0.0)

        with self._lock:
            self.last_decision = integrated

        return integrated

    def _generate_decision(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated decision from component analyses."""
        # Extract key signals
        quantum_signal = component_results.get('quantum_nash', {}).get('equilibrium', {})
        machiavellian_signal = component_results.get('machiavellian', {}).get('strategy', {})
        temporal_signal = component_results.get('temporal_nash', {}).get('equilibrium', {})

        # Weighted decision making
        decision_vector = np.zeros(4)  # [buy, sell, hold, wait]
        confidence_weights = []

        # Process quantum Nash signal
        if quantum_signal:
            q_action = quantum_signal.get('optimal_action', 2)  # Default to hold
            q_confidence = quantum_signal.get('convergence_score', 0.5)
            decision_vector[q_action] += q_confidence
            confidence_weights.append(q_confidence)

        # Process Machiavellian signal
        if machiavellian_signal:
            m_action = machiavellian_signal.get('recommended_action', 2)
            m_confidence = machiavellian_signal.get('confidence', 0.5)
            decision_vector[m_action] += m_confidence * 0.8  # Slightly lower weight
            confidence_weights.append(m_confidence * 0.8)

        # Process temporal signal
        if temporal_signal:
            t_action = temporal_signal.get('predicted_action', 2)
            t_confidence = temporal_signal.get('learning_confidence', 0.5)
            decision_vector[t_action] += t_confidence * 0.9
            confidence_weights.append(t_confidence * 0.9)

        # Normalize decision vector
        if np.sum(decision_vector) > 0:
            decision_vector /= np.sum(decision_vector)

        # Select action with highest weight
        action_idx = np.argmax(decision_vector)
        actions = ['buy', 'sell', 'hold', 'wait']

        return {
            'action': actions[action_idx],
            'confidence': np.mean(confidence_weights) if confidence_weights else 0.0,
            'decision_vector': decision_vector.tolist(),
            'reasoning': self._generate_reasoning(component_results, actions[action_idx])
        }

    def _generate_reasoning(self, results: Dict[str, Any], action: str) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []

        if 'quantum_nash' in results:
            reasoning_parts.append(
                f"Quantum Nash equilibrium suggests {action} based on strategic superposition"
            )

        if 'machiavellian' in results:
            manipulation = results['machiavellian'].get('manipulation_detected', {})
            if manipulation.get('detected', False):
                reasoning_parts.append(
                    f"Market manipulation detected with {manipulation.get('confidence', 0):.1%} confidence"
                )

        if 'temporal_nash' in results:
            learning = results['temporal_nash'].get('learning_progress', 0)
            reasoning_parts.append(
                f"Temporal patterns indicate {action} with {learning:.1%} learning convergence"
            )

        return " | ".join(reasoning_parts) if reasoning_parts else "Insufficient data for reasoning"

    def _extract_payoff_matrix(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract payoff matrix from market data for game theory analysis."""
        # Simplified payoff extraction - in practice would be more sophisticated
        participants = market_data.get('participants', ['self', 'market_maker', 'retail'])
        actions = ['buy', 'sell', 'hold', 'wait']

        n_participants = len(participants)
        n_actions = len(actions)

        # Initialize random payoff matrix (would be calculated from real market data)
        payoff_matrix = np.random.randn(n_participants, n_participants, n_actions, n_actions)

        # Add market structure bias
        if market_data.get('trend', 'neutral') == 'bullish':
            payoff_matrix[:, :, 0, :] += 0.5  # Favor buying
        elif market_data.get('trend', 'neutral') == 'bearish':
            payoff_matrix[:, :, 1, :] += 0.5  # Favor selling

        return payoff_matrix

    def save_state(self, filepath: Optional[str] = None) -> str:
        """
        Save current agent state to disk.

        Args:
            filepath: Optional custom filepath for checkpoint

        Returns:
            Path to saved checkpoint
        """
        state = {
            'agent_id': self.agent_id,
            'config': self.config,
            'quantum_states': self.quantum_serializer.serialize_all_states(),
            'memory_state': self.memory.serialize(),
            'component_states': {
                'machiavellian': self.machiavellian.get_state(),
                'robin_hood': self.robin_hood.get_state(),
                'temporal_nash': self.temporal_nash.get_state(),
                'antifragile': self.antifragile.get_state()
            },
            'performance_metrics': self.performance_tracker.get_metrics(),
            'last_decision': self.last_decision,
            'timestamp': datetime.utcnow().isoformat()
        }

        return self.state_manager.save_checkpoint(state, filepath)

    def load_state(self, filepath: str) -> bool:
        """
        Load agent state from checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Success status
        """
        try:
            state = self.state_manager.load_checkpoint(filepath)

            # Restore quantum states
            self.quantum_serializer.restore_all_states(state['quantum_states'])

            # Restore memory
            self.memory.restore(state['memory_state'])

            # Restore component states
            self.machiavellian.set_state(state['component_states']['machiavellian'])
            self.robin_hood.set_state(state['component_states']['robin_hood'])
            self.temporal_nash.set_state(state['component_states']['temporal_nash'])
            self.antifragile.set_state(state['component_states']['antifragile'])

            # Restore metrics
            self.performance_tracker.restore_metrics(state['performance_metrics'])

            self.last_decision = state.get('last_decision')

            logger.info(f"Successfully loaded state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    async def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading decision with orchestrator coordination.

        Args:
            decision: Decision to execute

        Returns:
            Execution result
        """
        try:
            # Request execution resources
            resources = await self.resource_manager.request_resources({
                'operation': 'execute_decision',
                'priority': 'critical',
                'timeout': 1000  # ms
            })

            # Execute decision (mock implementation)
            result = {
                'decision': decision,
                'status': 'executed',
                'timestamp': datetime.utcnow().isoformat(),
                'execution_time': 0.0
            }

            # Track execution
            self.performance_tracker.record_decision(decision, result)

            # Release resources
            await self.resource_manager.release_resources(resources)

            return result

        except Exception as e:
            logger.error(f"Decision execution failed: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'is_running': self.is_running,
            'hardware': self.hw_optimizer.get_device_info(),
            'memory_usage': self.memory.get_usage_stats(),
            'performance': self.performance_tracker.get_summary(),
            'last_checkpoint': self.state_manager.get_last_checkpoint_info()
        }

    def shutdown(self):
        """Gracefully shutdown the agent."""
        logger.info(f"Shutting down QBMIA Agent {self.agent_id}")

        # Save final state
        self.save_state()

        # Cleanup resources
        self.executor.shutdown(wait=True)
        self.hw_optimizer.cleanup()
        self.resource_manager.cleanup()

        self.is_running = False
        logger.info("Shutdown complete")
