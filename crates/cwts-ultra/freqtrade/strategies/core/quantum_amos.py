#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 22:55:53 2025

@author: ashina
"""

#!/usr/bin/env python3
"""
quantum_amos.py

Quantum AMOS – A Sophisticated Quantum Hybrid CADM‑BDIA Agent with Accelerator Integration
and JIT-accelerated numerical functions.

This module implements an agent that:
  - Uses standard market factors (via StandardFactors enum) to form beliefs.
  - Computes an intention signal by fusing these beliefs with Prospect Theory adjustments and an intrinsic desire.
  - Determines its final decision via a quantum‑inspired fusion layer whose device is initialized
    using lightning.kokkos when available (via hardware acceleration logic).
  - Supports automatic parameter tuning via cognitive reappraisal (simulating reinforcement‑learning updates).
  - Uses Numba to JIT‑compile and vectorize key numerical functions for improved performance.

The accelerator initialization attempts to load hardware acceleration modules;
if not available, it falls back to dummy implementations.

Author: [Your Name]
Date: [Date]
"""

import os
import logging
import random
import warnings
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Import pennylane with 0.41.0 compatibility
import pennylane as qml

# Create a compatibility layer for PennyLane 0.41.0 where qml.math structure has changed
if not hasattr(qml, 'math'):
    # Create a minimal math namespace with required functions to avoid import errors
    class MathCompatLayer:
        def __init__(self):
            pass
            
        def get_interface(self, tensor):
            # Simple implementation to handle basic tensor type checks
            if hasattr(tensor, 'dtype') and hasattr(tensor, 'shape'):
                return "numpy"
            return None
            
        def is_abstract(self, tensor):
            # Simple implementation that should work for most cases
            return False
    
    # Attach the compatibility layer to qml
    qml.math = MathCompatLayer()

# Import numba for jit and vectorization.
from numba import njit

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("QuantumAMOS")

# -----------------------------------------------------------------------------
# Hardware Accelerator Imports and Dummy Fallbacks
# -----------------------------------------------------------------------------
try:
    from hardware_manager import HardwareManager
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType
    HARDWARE_ACCEL_AVAILABLE = True
    logger.info("Hardware acceleration modules successfully imported.")
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False
    warnings.warn("Hardware acceleration modules not available. Using fallback implementation.")
    logger.warning("Hardware acceleration modules not available. Using fallback implementation.")
    
    class HardwareManager:
        @classmethod
        def get_manager(cls, **kwargs):
            return cls(**kwargs)
            
        def __init__(self, **kwargs):
            self.quantum_available = False
            self.gpu_available = False
            self.default_quantum_wires = kwargs.get("qubits", 5)
            
        def initialize_hardware(self):
            logger.info("Fallback HardwareManager: Hardware not initialized.")
            return False
            
        def _get_quantum_device(self, qubits):
            return {"device": "lightning.kokkos", "wires": qubits, "shots": None}
            
        def get_optimal_device(self, **kwargs):
            return {"type": "gpu", "available": True}
    
    class HardwareAccelerator:
        def __init__(self, **kwargs):
            self.gpu_available = False
            
        def get_accelerator_type(self):
            return "cpu"
            
        def get_torch_device(self):
            return None
    
    class AcceleratorType(Enum):
        CPU = auto()
        CUDA = auto()
        ROCM = auto()
        MPS = auto()
    

def _is_pennylane_available() -> bool:
    try:
        import pennylane
        return True
    except ImportError:
        return False

# -----------------------------------------------------------------------------
# 1. Standard Enums and Data Classes
# -----------------------------------------------------------------------------

class StandardFactors(Enum):
    """Standard factors for market signal alignment."""
    TREND = "trend"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    CYCLE = "cycle"
    ANOMALY = "anomaly"
    
    @classmethod
    def get_ordered_list(cls) -> List[str]:
        return [factor.value for factor in cls]
    
    @classmethod
    def get_default_weights(cls) -> Dict[str, float]:
        return {
            cls.TREND.value: 0.60,
            cls.VOLATILITY.value: 0.50,
            cls.MOMENTUM.value: 0.55,
            cls.SENTIMENT.value: 0.45,
            cls.LIQUIDITY.value: 0.35,
            cls.CORRELATION.value: 0.40,
            cls.CYCLE.value: 0.50,
            cls.ANOMALY.value: 0.30,
        }
    
    @classmethod
    def validate_factor_name(cls, factor_name: str) -> bool:
        return factor_name in cls.get_ordered_list()

class DecisionType(Enum):
    """Possible trading decisions."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    EXIT = auto()
    HEDGE = auto()
    INCREASE = auto()
    DECREASE = auto()

class MarketPhase(Enum):
    """Market phases from Panarchy theory."""
    GROWTH = "growth"
    CONSERVATION = "conservation"
    RELEASE = "release"
    REORGANIZATION = "reorganization"
    UNKNOWN = "unknown"

# -----------------------------------------------------------------------------
# 2. Prospect Theory Functions (with Numba JIT/Vectorization)
# -----------------------------------------------------------------------------

@njit
def prospect_value(x: float, alpha: float = 0.88, beta: float = 0.88, lambda_loss: float = 2.25) -> float:
    """
    Compute the subjective value for outcome x using Prospect Theory.
    For x >= 0, returns x^alpha; otherwise returns -lambda_loss*((-x)^beta)
    """
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_loss * ((-x) ** beta)

@njit
def probability_weight(p: float, delta: float = 0.65) -> float:
    """
    Compute the subjective probability weight for probability p.
    """
    return (p ** delta) / (((p ** delta) + ((1 - p) ** delta)) ** (1/delta))

@njit
def normalize_signal(signal: float, min_signal: float = -2, max_signal: float = 2) -> float:
    """
    Normalize a raw signal into an angle (in radians) within [-pi, pi].
    """
    return np.pi * (2 * (signal - min_signal) / (max_signal - min_signal) - 1)

# -----------------------------------------------------------------------------
# 3. Quantum Hybrid CADM-BDIA Agent with Reinforcement Learning Updates
# -----------------------------------------------------------------------------

class QuantumAmosAgent:
    """
    A Quantum Hybrid CADM-BDIA agent that fuses market signals (beliefs) with Prospect Theory adjustments
    and an intrinsic desire factor to produce an intention. A quantum-inspired fusion process then maps 
    this intention into a decision. The agent supports cognitive reappraisal to update its parameters.
    """
    qubits = 5
    shots = 1024
    c_dtype = np.float64

    def __init__(self, name: str, weights: Dict[str, float], desire: float,
                 hw_manager: Optional[Any] = None, hw_accelerator: Optional[Any] = None):
        self.name = name
        self.weights = weights
        self.desire = desire
        
        # Initialize hardware
        self._init_hardware( hw_manager, hw_accelerator)
        
        # Initialize quantum device
        self.device, success = self._initialize_quantum_device(device_name="lightning.kokkos")
        if not success or self.device is None:
            logger.warning("Falling back to default device lightning.qubit")
            self.device = qml.device("lightning.qubit", wires=self.qubits, shots=self.shots, c_dtype=self.c_dtype)
        logger.info("[%s] Initialized with device: %s", self.name, self.device.name)

    def _init_hardware(self, hw_manager, hw_accelerator):
        if hw_manager is not None and isinstance(hw_manager, HardwareManager):
            self.hw_manager = hw_manager
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_manager = HardwareManager.get_manager()
        else:
            self.hw_manager = HardwareManager()
            logger.warning("Fallback: Using dummy HardwareManager.")
        
        if hw_accelerator is not None and isinstance(hw_accelerator, HardwareAccelerator):
            self.hw_accelerator = hw_accelerator
        elif HARDWARE_ACCEL_AVAILABLE:
            self.hw_accelerator = HardwareAccelerator(enable_gpu=True)
        else:
            self.hw_accelerator = HardwareAccelerator()
            logger.warning("Fallback: Using dummy HardwareAccelerator.")
        
        if not getattr(self.hw_manager, '_is_initialized', False):
            self.hw_manager.initialize_hardware()
            
        # Dummy settings for circuit wires
        self.qubits = getattr(self, 'qubits', 5)
        if self.hw_manager is not None:
            self.max_qubits = getattr(self.hw_manager, 'default_quantum_wires', self.qubits)
            self.qubits = min(self.qubits, self.max_qubits)
        else:
            self.max_qubits = self.qubits

    def _initialize_quantum_device(self, device_name: Optional[str] = None) -> Tuple[Any, bool]:
        """Initialize the quantum device for the agent with hardware acceleration support
        and PennyLane 0.41.0 compatibility.
        
        Args:
            device_name: Preferred device name to use, if available
            
        Returns:
            Tuple of (device, success_flag)
        """
        # Try hardware manager first if available
        if hasattr(self, 'hw_manager') and self.hw_manager is not None:
            if hasattr(self.hw_manager, 'get_quantum_device'):
                try:
                    device_config = self.hw_manager.get_quantum_device(self.qubits)
                    device = qml.device(
                        device_config.get('device', 'lightning.kokkos'),
                        wires=self.qubits,
                        shots=device_config.get('shots', self.shots if hasattr(self, 'shots') else 1024)
                    )
                    logger.info(f"[{self.name}] Using hardware manager quantum device: {device.name}")
                    return device, True
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to initialize quantum device from hardware manager: {e}")
        
        # Check if PennyLane is available
        if not _is_pennylane_available():
            logger.warning(f"[{self.name}] PennyLane not available. Quantum features disabled.")
            return None, False
        
        # Try to use the specified device directly if provided
        if device_name is not None:
            try:
                device = qml.device(
                    device_name,
                    wires=self.qubits,
                    shots=self.shots if hasattr(self, 'shots') else 1024
                )
                logger.info(f"[{self.name}] Using specified quantum device: {device_name}")
                return device, True
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to initialize specified device {device_name}: {e}")
        
        # Systematically try different device types for maximum compatibility
        # Order of preference: lightning.kokkos (GPU) > lightning.gpu > lightning.qubit > default.qubit
        
        # Try lightning.kokkos with GPU acceleration
        try:
            # Set optimal environment variables for kokkos performance
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
            os.environ["OMP_PROC_BIND"] = "spread"
            os.environ["OMP_PLACES"] = "threads"
            os.environ["KOKKOS_DEVICES"] = "OpenMP,Cuda,HIP"
            
            # For AMD GPUs, set ROCm variables if available
            if hasattr(self.hw_manager, 'gfx_version') and self.hw_manager.gfx_version:
                os.environ['HSA_OVERRIDE_GFX_VERSION'] = self.hw_manager.gfx_version
                logger.info(f"[{self.name}] Set HSA_OVERRIDE_GFX_VERSION={self.hw_manager.gfx_version}")
            
            # Try to optimize device initialization for PennyLane 0.41.0+
            try:
                # For newer PennyLane versions that support kokkos settings
                device = qml.device(
                    "lightning.kokkos",
                    wires=self.qubits,
                    shots=self.shots if hasattr(self, 'shots') else 1024,
                )
                logger.info(f"[{self.name}] Successfully initialized lightning.kokkos device")
                return device, True
            except Exception as e:
                logger.warning(f"[{self.name}] PennyLane 0.41.0+ kokkos initialization failed: {e}")
                
                # Try older initialization method
                try:
                    from pennylane_lightning.lightning_kokkos import LightningKokkos
                    device = LightningKokkos(wires=self.qubits)
                    logger.info(f"[{self.name}] Using legacy LightningKokkos initialization")
                    return device, True
                except Exception as e2:
                    logger.warning(f"[{self.name}] Legacy kokkos initialization failed: {e2}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to initialize kokkos: {e}")
        
        # Try lightning.gpu
        try:
            device = qml.device(
                "lightning.gpu",
                wires=self.qubits,
                shots=self.shots if hasattr(self, 'shots') else 1024
            )
            logger.info(f"[{self.name}] Using lightning.gpu device")
            return device, True
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to initialize lightning.gpu: {e}")
        
        # Try lightning.qubit
        try:
            device = qml.device(
                "lightning.qubit",
                wires=self.qubits,
                shots=self.shots if hasattr(self, 'shots') else 1024
            )
            logger.info(f"[{self.name}] Using lightning.qubit device")
            return device, True
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to initialize lightning.qubit: {e}")
        
        # Final fallback to default.qubit
        try:
            device = qml.device(
                "default.qubit",
                wires=self.qubits,
                shots=self.shots if hasattr(self, 'shots') else 1024
            )
            logger.info(f"[{self.name}] Using default.qubit device")
            return device, True
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize ANY quantum device: {e}")
            return None, False

        
    def compute_beliefs(self, market_data: Dict[str, float]) -> Dict[str, float]:
        beliefs = {}
        for factor in StandardFactors.get_ordered_list():
            beliefs[factor] = market_data.get(factor, random.uniform(-1, 1))
        logger.info("[%s] Beliefs: %s", self.name, beliefs)
        return beliefs
    
    def compute_intention(self, market_data: Dict[str, float], expected_outcome: float, probability: float) -> float:
        beliefs = self.compute_beliefs(market_data)
        cadm_signal = sum(self.weights.get(factor, 0.0) * beliefs[factor] for factor in beliefs)
        # Use the jit-compiled functions for prospect adjustment.
        prospect_adj = prospect_value(expected_outcome) * probability_weight(probability)
        intention = cadm_signal + prospect_adj + self.desire
        logger.info("[%s] CADM: %.3f, ProspectAdj: %.3f, Desire: %.3f, Intention: %.3f",
                    self.name, cadm_signal, prospect_adj, self.desire, intention)
        return intention
    

    def quantum_decision(self, intention_signal: float) -> Tuple[DecisionType, float, np.ndarray, float]: # Added float for confidence
        """Make a decision using a parameterized quantum circuit based on intention signal.
        
        This uses a 2-qubit circuit with entanglement for more sophisticated decision modeling.
        The circuit applies rotation gates based on normalized intention signal and
        performs entanglement between qubits to create more complex quantum states.
        
        Args:
            intention_signal: The computed intention value from market data and prospect theory
            
        Returns:
            Tuple of (decision, intention_signal, quantum_probabilities, confidence_score)
        """
        norm_angle = normalize_signal(intention_signal, -2, 2)
        market_phase_angle = 0.0
        if hasattr(self, 'market_phase'):
            if self.market_phase == MarketPhase.GROWTH:
                market_phase_angle = 0.2
            elif self.market_phase == MarketPhase.CONSERVATION:
                market_phase_angle = 0.1
            elif self.market_phase == MarketPhase.RELEASE:
                market_phase_angle = -0.2
            elif self.market_phase == MarketPhase.REORGANIZATION:
                market_phase_angle = -0.1
                
        @qml.qnode(self.device)
        def advanced_fusion_circuit(theta, phase_angle=0.0):
            qml.RY(theta, wires=0)
            qml.RY(theta + phase_angle, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(theta * 0.5, wires=0)
            qml.RX(phase_angle, wires=1)
            qml.CNOT(wires=[1, 0])
            return qml.probs(wires=[0, 1])
        
        probs = advanced_fusion_circuit(norm_angle, market_phase_angle)
        
        decision_map_probs = {
            DecisionType.BUY: probs[0],       # 00 state
            DecisionType.SELL: probs[3],      # 11 state
            DecisionType.INCREASE: probs[1],  # 01 state (positive intention)
            DecisionType.DECREASE: probs[1],  # 01 state (negative intention)
            DecisionType.HEDGE: probs[2],     # 10 state (positive intention)
            DecisionType.EXIT: probs[2],      # 10 state (negative intention)
        }
        
        # Determine decision based on max probability for dominant states, then specific conditions
        if probs[0] > 0.4:
            decision = DecisionType.BUY
            confidence = float(probs[0])
        elif probs[3] > 0.4:
            decision = DecisionType.SELL
            confidence = float(probs[3])
        elif probs[1] > 0.4:
            decision = DecisionType.INCREASE if intention_signal > 0 else DecisionType.DECREASE
            confidence = float(probs[1])
        elif probs[2] > 0.4:
            decision = DecisionType.HEDGE if intention_signal > 0 else DecisionType.EXIT
            confidence = float(probs[2])
        else:
            decision = DecisionType.HOLD
            # For HOLD, confidence could be 1 - (sum of dominant state probs if they are low)
            # or simply based on the spread or max of other probabilities
            confidence = float(1.0 - np.sum(probs[[0,1,2,3]])) # Example: confidence in HOLD is how much isn't in other defined states
            # A better HOLD confidence: if all main signals are weak
            if probs[0] < 0.3 and probs[1] < 0.3 and probs[2] < 0.3 and probs[3] < 0.3:
                 confidence = float(np.max(probs)) # Or some other metric e.g. 1.0 - (max_prob - second_max_prob)
            else: # If one of the directional signals was almost chosen.
                 confidence = float(np.sort(probs)[-1]) # Take the highest prob as confidence for HOLD

        logger.info("[%s] Advanced decision: %s (Intention: %.3f, Confidence: %.3f)", 
                    self.name, decision.name, intention_signal, confidence)
        return decision, intention_signal, probs, confidence
    
    def decide(self, market_data: Dict[str, float], expected_outcome: float, probability: float) -> DecisionType: # Ensure return type is DecisionType enum
        intention = self.compute_intention(market_data, expected_outcome, probability)
        # quantum_decision now returns: decision_enum, intention_signal, probs, confidence_val
        # We only need the decision_enum for this method's specified return type.
        # The confidence_val is logged within quantum_decision.
        decision_enum, _, _, _ = self.quantum_decision(intention) 
        return decision_enum # Return the enum directly
    
    def cognitive_reappraisal(self, market_data: Dict[str, float], predicted_return: float, 
                              actual_return: float, learning_rate: float = 0.01):
        error = actual_return - predicted_return
        logger.info("[%s] Reappraisal: predicted=%.3f, actual=%.3f, error=%.3f",
                    self.name, predicted_return, actual_return, error)
        self.desire += learning_rate * error
        for factor in self.weights:
            signal = market_data.get(factor, 0)
            self.weights[factor] += learning_rate * error * signal
            logger.info("[%s] New weight for %s: %.3f", self.name, factor, self.weights[factor])
        logger.info("[%s] Updated desire: %.3f, weights: %s", self.name, self.desire, self.weights)

# -----------------------------------------------------------------------------
# 4. Multi-Agent Quantum AMOS Network
# -----------------------------------------------------------------------------

class QuantumAmosNetwork:
    """
    A network that aggregates decisions from multiple QuantumAmosAgent instances using
    advanced quantum circuits and reinforcement learning for collective decision making.
    
    The network supports:
    1. Weighted consensus based on agent performance history
    2. Advanced multi-qubit entangled circuits for consensus modeling
    3. Market regime detection and adaptation
    4. Experience-based learning across the agent network
    """
    def __init__(self, agents: List[QuantumAmosAgent]):
        self.agents = agents
        # Initialize agent weights equally
        self.agent_weights = {agent.name: 1.0/len(agents) for agent in agents}
        # Track performance history for adaptive weighting
        self.performance_history = {agent.name: [] for agent in agents}
        # Current market phase detection
        self.market_phase = MarketPhase.GROWTH  # Default starting phase
        # Track network-level metrics
        self.decision_history = []
        self.intention_history = []
        
        logger.info("Network initialized with %d agents and adaptive weighting.", len(agents))
    
    def detect_market_phase(self, market_data: Dict[str, float]) -> MarketPhase:
        """
        Detect the current market phase based on market data indicators.
        Uses a simplified Panarchy theory approach to cycle through growth,
        conservation, release, and reorganization phases.
        
        Args:
            market_data: Dictionary of market factors and signals
            
        Returns:
            Current market phase
        """
        # Get key indicators for phase detection
        trend = market_data.get("trend", 0)
        volatility = market_data.get("volatility", 0)
        momentum = market_data.get("momentum", 0)
        cycle = market_data.get("cycle", 0)
        
        # Detect phase using a rule-based approach
        if trend > 0.3 and momentum > 0.2 and volatility < 0.4:
            phase = MarketPhase.GROWTH
        elif trend > 0 and momentum < 0 and volatility < 0.3:
            phase = MarketPhase.CONSERVATION
        elif trend < -0.3 or volatility > 0.7:
            phase = MarketPhase.RELEASE
        elif trend < 0 and momentum > 0:
            phase = MarketPhase.REORGANIZATION
        else:
            # Keep current phase if unclear
            phase = self.market_phase
            
        if phase != self.market_phase:
            logger.info(f"[Network] Market phase transition detected: {self.market_phase.value} -> {phase.value}")
            self.market_phase = phase
            
        # Update agents with the current market phase
        for agent in self.agents:
            agent.market_phase = phase
            
        return phase
    
    def update_agent_weights(self):
        """
        Update the influence weights of agents based on their recent performance history.
        Better performing agents receive higher weights in the consensus decision making.
        """
        if not all(len(hist) > 0 for hist in self.performance_history.values()):
            return  # Not enough history yet
            
        # Calculate average performance for each agent
        avg_performance = {}
        for agent_name, history in self.performance_history.items():
            # Use the most recent performance entries (up to 10)
            recent_history = history[-10:]
            avg_performance[agent_name] = np.mean(recent_history) if recent_history else 0
            
        # Normalize performances to sum to 1.0
        total_performance = sum(max(0.01, perf) for perf in avg_performance.values())
        for agent_name in avg_performance:
            # Ensure minimum weight of 0.1 for diversity
            self.agent_weights[agent_name] = max(0.1, avg_performance[agent_name] / total_performance)
            
        # Re-normalize weights
        total_weight = sum(self.agent_weights.values())
        for agent_name in self.agent_weights:
            self.agent_weights[agent_name] /= total_weight
            
        logger.info(f"[Network] Updated agent weights: {self.agent_weights}")
    
    def network_decide(self, market_data: Dict[str, float], expected_outcome: float, probability: float) -> DecisionType:
        """
        Make a consensus decision using all agents in the network, weighted by their performance.
        Uses an advanced quantum circuit with entanglement for higher-dimensional consensus.
        
        Args:
            market_data: Dictionary of market factors and signals
            expected_outcome: Expected return or outcome
            probability: Probability of the expected outcome
            
        Returns:
            Consensus decision from all agents
        """
        # First detect market phase to adapt decision-making
        current_phase = self.detect_market_phase(market_data)
        
        intentions = []
        decisions = []
        agent_outputs = {}
        
        for agent in self.agents:
            ti = agent.compute_intention(market_data, expected_outcome, probability)
            # agent.quantum_decision now returns: decision, intention_signal, probs, confidence
            # Unpack all four values, using _ for those not immediately used in this scope.
            decision, _, probs, _ = agent.quantum_decision(ti) # MODIFIED THIS LINE
            
            intentions.append(ti)
            decisions.append(decision)
            agent_outputs[agent.name] = (ti, decision, probs)
            
            weight = self.agent_weights.get(agent.name, 1.0/len(self.agents))
            logger.info(f"[{agent.name}] Intention: {ti:.3f}, Decision: {decision.name}, Weight: {weight:.3f}")
        
        # Calculate weighted average intention
        weighted_intention = 0
        for i, agent in enumerate(self.agents):
            weight = self.agent_weights.get(agent.name, 1.0/len(self.agents))
            weighted_intention += intentions[i] * weight
            
        self.intention_history.append(weighted_intention)
        logger.info(f"[Network] Weighted intention: {weighted_intention:.3f} (Market phase: {current_phase.value})")
        
        # Use the best performing agent's device for the consensus circuit
        device = self.agents[0].device
        
        # Create a more complex quantum circuit for consensus
        @qml.qnode(device)
        def consensus_circuit(theta, phase_param):
            # Use 3 qubits for more expressive decision space
            qml.RY(theta, wires=0)
            qml.RY(theta * 0.8 + phase_param, wires=1)  # Phase parameter affects rotation
            qml.RY(theta * 0.6, wires=2)
            
            # Create entanglement pattern
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            
            # Additional rotations influenced by market phase
            qml.RZ(phase_param * 0.5, wires=0)
            qml.RX(phase_param * 0.3, wires=1)
            qml.RZ(phase_param * 0.7, wires=2)
            
            # Final entanglement
            qml.CNOT(wires=[2, 0])
            
            # Return full probability distribution over 3 qubits (8 possible states)
            return qml.probs(wires=[0, 1, 2])
        
        # Normalize the intention for quantum circuit
        norm_angle = normalize_signal(weighted_intention, -2, 2)
        
        # Phase parameter depends on market regime
        phase_param = 0.0
        if current_phase == MarketPhase.GROWTH:
            phase_param = 0.4
        elif current_phase == MarketPhase.CONSERVATION:
            phase_param = 0.2
        elif current_phase == MarketPhase.RELEASE:
            phase_param = -0.4
        elif current_phase == MarketPhase.REORGANIZATION:
            phase_param = -0.2
        
        # Get consensus probabilities
        consensus_probs = consensus_circuit(norm_angle, phase_param)
        logger.info(f"[Network] Consensus quantum probabilities: {consensus_probs}")
        
        # Interpret 3-qubit state for decision making
        # We have 8 possible states to map to decisions
        # Use dominant probability states and weighted intention
        if consensus_probs[0] > 0.3:  # 000 state
            consensus = DecisionType.BUY
        elif consensus_probs[7] > 0.3:  # 111 state
            consensus = DecisionType.SELL
        elif np.max(consensus_probs[1:3]) > 0.3:  # 001, 010 states
            consensus = DecisionType.INCREASE if weighted_intention > 0 else DecisionType.DECREASE
        elif np.max(consensus_probs[3:7]) > 0.3:  # 011, 100, 101, 110 states
            consensus = DecisionType.HEDGE if weighted_intention > 0 else DecisionType.EXIT
        else:
            # Count most frequent decision among agents as a fallback
            from collections import Counter
            decision_counts = Counter(decision.name for decision in decisions)
            most_common = decision_counts.most_common(1)[0][0]
            consensus = next(d for d in DecisionType if d.name == most_common)
        
        self.decision_history.append(consensus)
        logger.info(f"[Network] Consensus decision: {consensus.name}, Phase: {current_phase.value}")
        return consensus
    
    def update_agents(self, market_data: Dict[str, float], predicted_return: float, actual_return: float, 
                      learning_rate: float = 0.01):
        """
        Update all agents based on the actual return feedback using cognitive reappraisal.
        Also updates the performance history used for agent weighting.
        
        Args:
            market_data: Dictionary of market factors and signals
            predicted_return: The predicted return from the previous decision
            actual_return: The actual return observed
            learning_rate: Rate of parameter updates
        """
        logger.info(f"[Network] Updating agents based on feedback... Predicted: {predicted_return:.3f}, Actual: {actual_return:.3f}")
        
        # Update performance history for each agent
        error = abs(predicted_return - actual_return)
        normalized_error = 1.0 / (1.0 + error)  # Higher value means better performance
        
        for agent in self.agents:
            # Update the agent
            agent.cognitive_reappraisal(market_data, predicted_return, actual_return, learning_rate)
            
            # Record performance for future weighting
            self.performance_history[agent.name].append(normalized_error)
            
            # Keep history size bounded
            if len(self.performance_history[agent.name]) > 100:
                self.performance_history[agent.name] = self.performance_history[agent.name][-100:]
        
        # Update agent weights based on performance
        self.update_agent_weights()
        
        # Adjust learning rate based on market phase
        if self.market_phase == MarketPhase.RELEASE:
            # Learn faster during volatile periods
            adjusted_lr = learning_rate * 1.5
        elif self.market_phase == MarketPhase.GROWTH:
            # Learn at normal rate during growth
            adjusted_lr = learning_rate
        else:
            # Learn slower during other phases
            adjusted_lr = learning_rate * 0.8
            
        logger.info(f"[Network] Agent update complete. Adjusted learning rate: {adjusted_lr:.4f}")
        
    def get_network_state(self) -> Dict[str, Any]:
        """
        Return the current state of the network for external monitoring
        and visualization.
        
        Returns:
            Dictionary containing network state information
        """
        agent_beliefs = {}
        for agent in self.agents:
            agent_beliefs[agent.name] = agent.compute_beliefs({})
            
        return {
            "agent_count": len(self.agents),
            "agent_weights": self.agent_weights,
            "market_phase": self.market_phase.value,
            "recent_decisions": [d.name for d in self.decision_history[-10:]] if self.decision_history else [],
            "recent_intentions": self.intention_history[-10:] if self.intention_history else [],
            "agent_desires": {agent.name: agent.desire for agent in self.agents},
            "agent_beliefs": agent_beliefs
        }

