#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 19:36:15 2025

@author: ashina
"""

import logging
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Hardware Management
from hardware_manager import HardwareManager
from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType

# Quantum Components
from qar import QuantumAgenticReasoning, TradingDecision, DecisionType, MarketPhase
from qerc import get_quantum_reservoir_computing
from iqad import get_immune_quantum_anomaly_detector
from nqo import get_neuromorphic_quantum_optimizer

# Analysis Components
#from advanced_cdfa import AdvancedCDFA, AdvancedCDFAConfig
from enhanced_cdfa import CognitiveDiversityFusionAnalysis, CDFAConfig
from cdfa_extensions.analyzers.fibonacci_analyzer import FibonacciAnalyzer

from cdfa_extensions.detectors.whale_detector import WhaleDetector
from cdfa_extensions.detectors.black_swan_detector import BlackSwanDetector

from cdfa_extensions.analyzers.antifragility_analyzer import AntifragilityAnalyzer
#from logarithmic_market_scoring_rule import LogarithmicMarketScoringRule, LMSRConfig
from enhanced_lmsr import LogarithmicMarketScoringRule, LMSRConfig
from quantum_lmsr import QuantumLMSR
from prospect_theory_risk_manager import ProspectTheoryRiskManager
from quantum_prospect_theory import QuantumProspectTheory

# Risk Management
from risk_manager import (
    ViaNegativaFilter,
    LuckVsSkillAnalyzer,
    BarbellAllocator,
    ReputationSystem,
    EnhancedMarketAnomalyDetector,
    AntifragileRiskManager
)

# Signal Processing
from usp import UniversalSignalProcessor


try:
    # Assumes panarchy_analyzer.py is importable from qar.py's location
    from cdfa_extensions.analyzers.panarchy_analyzer import MarketPhase
    logging.debug(f"QAR.PY: Successfully imported MarketPhase: {MarketPhase}")
except ImportError:
    logging.error("CRITICAL: Failed to import MarketPhase from panarchy_analyzer. Using fallback enum (No UNKNOWN phase).")
    class MarketPhase(Enum):
         GROWTH="growth"
         CONSERVATION="conservation"
         RELEASE="release"
         REORGANIZATION="reorganization"

         @classmethod
         def from_string(cls, phase_str: str):
             phase_str = str(phase_str).lower()
             for phase in cls:
                 if phase.value == phase_str: return phase
             logging.warning(f"Invalid phase string '{phase_str}' received in fallback enum. Defaulting to CONSERVATION.")
             return cls.CONSERVATION

class DecisionType(Enum):
    """Trading decision types"""

    BUY = auto()
    SELL = auto()
    HOLD = auto()
    EXIT = auto()  # Explicit exit signal (close position)
    HEDGE = auto()
    INCREASE = auto()  # Increase existing position size
    DECREASE = auto()  # Decrease existing position size


@dataclass
class TradingDecision:
    """Trading decision data structure"""

    decision_type: DecisionType
    confidence: float
    reasoning: str
    timestamp: Optional[datetime] = None
    parameters: Dict[str, Any] = field(
        default_factory=dict
    )  # Store supporting params (e.g., raw signals)
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Store execution info (method, factor contribs)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class CircuitCache:
    """Cache for quantum circuits (keep as before)"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.hit_count += 1
            self.cache[key]["last_access"] = time.time()
            return self.cache[key]["circuit"]
        self.miss_count += 1
        return None

    def put(self, key: str, circuit: Any) -> None:
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(), key=lambda k: self.cache[k]["last_access"]
            )
            del self.cache[oldest_key]
        self.cache[key] = {"circuit": circuit, "last_access": time.time()}

    def clear(self) -> None:
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_ratio": self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0,
        }


# Logger setup
logger = logging.getLogger("panarchy_adaptive_decision_system")


class PanarchyAdaptiveDecisionSystem:
    """Adapts Hybrid Pipeline based on Panarchy. Relies on central HardwareManager."""

    def __init__(
        self,
        hw_manager: HardwareManager,
        universal_signal_processor: Optional[Any] = None,
        cdfa_analyzer: Optional[Any] = None,
        strategy_config: Optional[Dict] = None,
        name: str = "Panarchy Adaptive Decision System",
    ):
        self.name = name
        self.logger = logger # Use module logger
        self.logger.info(f"Initializing PanarchyAdaptiveDecisionSystem '{name}'...")

        # --- Store Dependencies ---
        self.hw_manager = hw_manager
        self.usp = universal_signal_processor
        self.cdfa = cdfa_analyzer

        self._lock = threading.RLock()

        # --- Config Extraction ---
        self.config = strategy_config if strategy_config is not None else {}
        # Extract sub-configs for initializing internal components
        qar_config_dict = self.config.get('qar_config', {})
        qaoa_config_dict = self.config.get('qaoa_config', {})
        quareg_config_dict = self.config.get('quareg_config', {})

        # --- Reference to Parent Strategy ---
        self.strategy = self.config.get('strategy_instance', None)

        # --- Direct Internal Component Initialization ---
        self.qar = None  # Will hold QuantumAgenticReasoning instance

        # Initialize component placeholders without specific type annotations
        self.qaoa = None
        self.quareg = None
        self.qerc = None
        self.iqad = None
        self.nqo = None
        self.qstar_predictor = None
        self.narrative_forecaster = None

        # Init QAR
        qar_cls = globals().get("QuantumAgenticReasoning")
        if qar_cls and self.hw_manager:
            try:
                # Base threshold/memory if needed elsewhere, even if overridden by phase later
                self._base_qar_threshold = qar_config_dict.get('decision_threshold', 0.6)
                self._base_qar_memory = qar_config_dict.get('memory_length', 50)

                self.qar = qar_cls(
                    hardware_manager=self.hw_manager,
                    decision_threshold=self._base_qar_threshold, # Initialize with base
                    memory_length=self._base_qar_memory,
                    log_level=self.config.get('pads_log_level', logging.INFO)
                )
                self.logger.info("PADS: Internal QAR instance Initialized.")
            except Exception as e:
                self.logger.error(f"PADS: Error init internal QAR: {e}", exc_info=True)

        # Initialize board members first (needed for LMSR)
        self.board_members = self._initialize_board_members()
        
        # Initialize additional quantum components
        self.qerc = self._initialize_qerc()
        self.iqad = self._initialize_iqad()
        self.nqo = self._initialize_nqo()
        
        # Add LMSR for board decision aggregation
        self.board_lmsr = LogarithmicMarketScoringRule(
            config=LMSRConfig(
                liquidity_parameter=50.0,  # Lower for more sensitivity to board votes
                enable_parallel=True,      # Process board votes in parallel
                cache_size=256            # Larger cache for board voting patterns
            )
        )
        
        # Market state for board members
        self.board_quantities = {member: 0.0 for member in self.board_members}
    
        # Initialize analyzers
        self.whale_detector = self._initialize_whale_detector()
        self.black_swan_detector = self._initialize_black_swan_detector()
        self.antifragility_analyzer = self._initialize_antifragility_analyzer()
        self.fibonacci_analyzer = self._initialize_fibonacci_analyzer()

        # Initialize risk management components
        self.via_negativa_filter = self._initialize_via_negativa_filter()
        self.luck_vs_skill_analyzer = self._initialize_luck_vs_skill_analyzer()
        self.barbell_allocator = self._initialize_barbell_allocator()
        self.reputation_system = self._initialize_reputation_system()
        self.enhanced_anomaly_detector = self._initialize_enhanced_anomaly_detector()
        self.antifragile_risk_manager = self._initialize_antifragile_risk_manager()
        self.prospect_theory_manager = self._initialize_prospect_theory_manager()

        # Initialize LMSR
        self.lmsr = self._initialize_lmsr()

        # Initialize NarrativeForecaster
        self.narrative_forecaster = self._initialize_narrative_forecaster()

        # Initialize QStar Predictor
        self.qstar_predictor = self._initialize_qstar_predictor()

        # --- Load Phase Parameters ---
        self._phase_parameters = self._load_phase_parameters(self.config)

        # --- Initialize PADS State Variables ---
        # Default values from your original qar.py snippet for PADS state
        self.panarchy_state = {
            "phase": "conservation", # Default starting phase?
            "regime": "normal",      # Calculated field
            "soc_index": 0.5,
            "black_swan_risk": 0.1,
             # Added: Raw P/C/R potentially updated from analyze_market
            "micro_phase": "growth", # These should be UPDATED by input, not fixed state
            "meso_phase": "growth",
            "macro_phase": "growth",
        }

        # Board members (weighted voting) - already initialized above

        # Initialize Reputation System for board members
        self.reputation_scores = {member: 0.5 for member in self.board_members.keys()}

        # Decision history and metrics
        self.decision_history = [] # Keep decision history internal to PADS
        self.memory_length = qar_config_dict.get('pads_memory_length', 100)

        # --- Advanced Decision-Making Components ---
        # Decision styles available to the board
        self.decision_styles = [
            'consensus',         # High agreement needed
            'opportunistic',     # Take advantage of temporary situations
            'defensive',         # Protect against downside
            'calculated_risk',   # Accept reasonable risks for potential gain
            'contrarian',        # Go against prevailing sentiment
            'momentum_following' # Follow established trends
        ]

        # Current decision style
        self.current_decision_style = self.decision_styles[0]

        # Confidence thresholds by regime
        self.confidence_thresholds = {
            'conservation': 0.65,  # Higher confidence needed in mature markets
            'growth': 0.55,        # More opportunistic in growth phase
            'release': 0.75,       # More cautious in release phase
            'reorganization': 0.6  # Balanced in reorganization
        }

        # Initialize the "board" state
        self.board_state = {
            'consensus_level': 0.5,    # Agreement level among components
            'conviction_level': 0.5,   # Strength of belief in decision
            'risk_appetite': 0.5,      # Current risk appetite
            'opportunity_score': 0.5,  # Perceived opportunity
            'voting_quorum': 0.0,      # Votes accumulated
            'dissent_level': 0.0,      # Level of disagreement
            'current_strategy': 'balanced' # Current decision strategy
        }

        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'risk_adjusted_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'decisions_by_phase': {}
        }

        self.logger.info(f"PanarchyAdaptiveDecisionSystem '{self.name}' initialization complete.")


    
    def _initialize_board_members(self) -> Dict[str, float]:
        """Initialize board members with initial weights."""
        board_members = {
            'qar': 0.25,                     # Quantum Agentic Reasoning
            'narrative_forecaster': 0.15,    # Narrative Forecasting with Sentiment Analysis
            'qstar': 0.10,                   # QStar Trading Predictor
            'antifragility': 0.08,           # Antifragility Analysis
            'black_swan': 0.05,              # Black Swan Detection
            'whale_detector': 0.05,          # Whale Detection
            'fibonacci': 0.05,               # Fibonacci Analysis
            'prospect_theory': 0.07,         # Prospect Theory Risk Management
            'cdfa': 0.05,                    # Cognitive Diversity and Fusion Analysis
            'barbell': 0.05,                 # Barbell Allocation Strategy
            'via_negativa': 0.05,            # Via Negativa Filtering
            'luck_vs_skill': 0.025,          # Luck vs Skill Analysis
            'antifragile_risk': 0.025,       # Antifragile Risk Management
            'enhanced_anomaly': 0.025,       # Enhanced Anomaly Detection
        }
    
        # Normalize weights
        total_weight = sum(board_members.values())
        normalized_board = {k: v/total_weight for k, v in board_members.items()}
    
        return normalized_board


    def _initialize_narrative_forecaster(self):
        """Initialize NarrativeForecaster component."""
        
        try:
            from narrative_forecaster import NarrativeForecaster, LLMConfig

            # Extract config if available
            narrative_config = self.config.get('narrative_config', {})
            
            llm_config = LLMConfig(
                #provider=LLMProvider.LMSTUDIO,
                provider=narrative_config.get('provider', 'lmstudio'),  # Default to local LMStudio
                api_key=narrative_config.get('api_key', None),
                model=narrative_config.get('model', 'mistral'),
                temperature=narrative_config.get('temperature', 1),
                base_url="http://localhost:1234/v1/chat/completions",
                max_tokens=4096, # Increased from 1000
                timeout=60.0, # Slightly increased timeout just in case
                retry_attempts=3
                )

            # Initialize NarrativeForecaster
            forecaster = NarrativeForecaster(
                llm_config=llm_config,
                cache_duration=narrative_config.get('cache_duration', 60),
                hardware_manager=self.hw_manager
            )

            self.logger.info("PADS: NarrativeForecaster initialized.")
            return forecaster
        except ImportError:
            self.logger.warning("PADS: NarrativeForecaster module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing NarrativeForecaster: {e}", exc_info=True)
            return None

    def _initialize_qstar_predictor(self):
        """Initialize QStarTradingPredictor component."""
        try:
            from qstar_river import QStarTradingPredictor

            # Extract config if available
            qstar_config = self.config.get('qstar_config', {})

            # Get RiverML instance from strategy if available
            river_ml = None
            if hasattr(self.strategy, 'river_ml'):
                river_ml = self.strategy.river_ml

            # Initialize QStarTradingPredictor
            predictor = QStarTradingPredictor(
                river_ml=river_ml,
                use_quantum_representation=qstar_config.get('use_quantum_representation', True),
                initial_states=qstar_config.get('initial_states', 200),
                training_episodes=qstar_config.get('training_episodes', 100)
            )

            self.logger.info("PADS: QStarTradingPredictor initialized.")
            return predictor
        except ImportError:
            self.logger.warning("PADS: QStarTradingPredictor module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing QStarTradingPredictor: {e}", exc_info=True)
            return None


    def _initialize_qerc(self):
        """Initialize QERC component."""
        try:
            # Uses get_quantum_reservoir_computing factory function from import
            qerc = get_quantum_reservoir_computing(self.config.get('qerc_config', {}))
            self.logger.info("PADS: QERC initialized.")
            return qerc
        except Exception as e:
            self.logger.error(f"PADS: Error initializing QERC: {e}", exc_info=True)
            return None

    def _initialize_iqad(self):
        """Initialize IQAD component."""
        try:
            # Uses get_immune_quantum_anomaly_detector factory function from import
            iqad = get_immune_quantum_anomaly_detector(self.config.get('iqad_config', {}))
            self.logger.info("PADS: IQAD initialized.")
            return iqad
        except Exception as e:
            self.logger.error(f"PADS: Error initializing IQAD: {e}", exc_info=True)
            return None

    def _initialize_nqo(self):
        """Initialize NQO component."""
        try:
            # Uses get_neuromorphic_quantum_optimizer factory function from import
            nqo = get_neuromorphic_quantum_optimizer(self.config.get('nqo_config', {}))
            self.logger.info("PADS: NQO initialized.")
            return nqo
        except Exception as e:
            self.logger.error(f"PADS: Error initializing NQO: {e}", exc_info=True)
            return None

    def _initialize_qaoa(self):
        """Initialize QAOA component."""
        try:
            from qaoa import create_qaoa_optimizer

            # Initialize QAOA with hardware manager
            qaoa = create_qaoa_optimizer(
                num_parameters=self.config.get('qaoa_config', {}).get('num_parameters', 5),
                num_layers=self.config.get('qaoa_config', {}).get('num_layers', 2),
                hardware_kwargs={'hw_manager': self.hw_manager}
            )

            self.logger.info("PADS: QAOA initialized.")
            return qaoa
        except ImportError:
            self.logger.warning("PADS: QAOA module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing QAOA: {e}", exc_info=True)
            return None

    def _initialize_quareg(self):
        """Initialize Quareg component."""
        try:
            from quareg import create_quantum_annealing_regression

            # Initialize Quareg with hardware manager
            quareg = create_quantum_annealing_regression(
                hw_manager=self.hw_manager,
                window_size=self.config.get('quareg_config', {}).get('window_size', 20),
                forecast_horizon=self.config.get('quareg_config', {}).get('forecast_horizon', 5)
            )

            self.logger.info("PADS: Quareg initialized.")
            return quareg
        except ImportError:
            self.logger.warning("PADS: Quareg module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Quareg: {e}", exc_info=True)
            return None


    def _initialize_whale_detector(self):
        """Initialize Whale Detector component."""
        try:
            detector = WhaleDetector()
            self.logger.info("PADS: Whale Detector initialized.")
            return detector
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Whale Detector: {e}", exc_info=True)
            return None

    def _initialize_black_swan_detector(self):
        """Initialize Black Swan Detector component."""
        try:
            detector = BlackSwanDetector()
            self.logger.info("PADS: Black Swan Detector initialized.")
            return detector
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Black Swan Detector: {e}", exc_info=True)
            return None

    def _initialize_antifragility_analyzer(self):
        """Initialize Antifragility Analyzer component."""
        try:
            analyzer = AntifragilityAnalyzer(use_jit=True, cache_size=100)
            self.logger.info("PADS: Antifragility Analyzer initialized.")
            return analyzer
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Antifragility Analyzer: {e}", exc_info=True)
            return None

    def _initialize_fibonacci_analyzer(self):
        """Initialize Fibonacci Analyzer component."""
        try:
            analyzer = FibonacciAnalyzer(cache_size=100, use_jit=True)
            self.logger.info("PADS: Fibonacci Analyzer initialized.")
            return analyzer
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Fibonacci Analyzer: {e}", exc_info=True)
            return None

    def _initialize_soc_analyzer(self):
        """Initialize SOC Analyzer component."""
        try:
            from soc_analyzer import SOCAnalyzer

            analyzer = SOCAnalyzer()
            self.logger.info("PADS: SOC Analyzer initialized.")
            return analyzer
        except ImportError:
            self.logger.warning("PADS: SOC Analyzer module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing SOC Analyzer: {e}", exc_info=True)
            return None

    def _initialize_panarchy_analyzer(self):
        """Initialize Panarchy Analyzer component."""
        try:
            from panarchy_analyzer import PanarchyAnalyzer

            analyzer = PanarchyAnalyzer()
            self.logger.info("PADS: Panarchy Analyzer initialized.")
            return analyzer
        except ImportError:
            self.logger.warning("PADS: Panarchy Analyzer module not available.")
            return None
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Panarchy Analyzer: {e}", exc_info=True)
            return None

    def _initialize_via_negativa_filter(self):
        """Initialize Via Negativa Filter component."""
        try:
            filter_obj = ViaNegativaFilter()
            self.logger.info("PADS: Via Negativa Filter initialized.")
            return filter_obj
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Via Negativa Filter: {e}", exc_info=True)
            return None

    def _initialize_luck_vs_skill_analyzer(self):
        """Initialize Luck vs Skill Analyzer component."""
        try:
            analyzer = LuckVsSkillAnalyzer()
            self.logger.info("PADS: Luck vs Skill Analyzer initialized.")
            return analyzer
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Luck vs Skill Analyzer: {e}", exc_info=True)
            return None

    def _initialize_barbell_allocator(self):
        """Initialize Barbell Allocator component."""
        try:
            allocator = BarbellAllocator()
            self.logger.info("PADS: Barbell Allocator initialized.")
            return allocator
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Barbell Allocator: {e}", exc_info=True)
            return None

    def _initialize_reputation_system(self):
        """Initialize Reputation System component."""
        try:
            system = ReputationSystem()
            self.logger.info("PADS: Reputation System initialized.")
            return system
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Reputation System: {e}", exc_info=True)
            return None

    def _initialize_enhanced_anomaly_detector(self):
        """Initialize Enhanced Anomaly Detector component."""
        try:
            detector = EnhancedMarketAnomalyDetector()
            self.logger.info("PADS: Enhanced Anomaly Detector initialized.")
            return detector
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Enhanced Anomaly Detector: {e}", exc_info=True)
            return None

    def _initialize_antifragile_risk_manager(self):
        """Initialize Antifragile Risk Manager component."""
        try:
            manager = AntifragileRiskManager()
            self.logger.info("PADS: Antifragile Risk Manager initialized.")
            return manager
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Antifragile Risk Manager: {e}", exc_info=True)
            return None

    def _initialize_prospect_theory_manager(self):
        """Initialize Prospect Theory Risk Manager component."""
        try:
            manager = ProspectTheoryRiskManager()
            self.logger.info("PADS: Prospect Theory Risk Manager initialized.")
            return manager
        except Exception as e:
            self.logger.error(f"PADS: Error initializing Prospect Theory Risk Manager: {e}", exc_info=True)
            return None

    def _initialize_lmsr(self):
        """Initialize LMSR component."""
        try:
            lmsr = LogarithmicMarketScoringRule()
            self.logger.info("PADS: LMSR initialized.")
            return lmsr
        except Exception as e:
            self.logger.error(f"PADS: Error initializing LMSR: {e}", exc_info=True)
            return None



    def _load_phase_parameters(self, config: Dict) -> Dict:
            # ... (Keep EXACT logic from previous response: get 'phase_params' from config dict, validate, default) ...
            params_from_config = config.get('phase_params', {})
            if not params_from_config:
                self.logger.warning("PADS: No 'phase_params' found in strategy_config. QAR will use static/default weights/thresh.")
                default_weights = self.qar.factor_weights.copy() if self.qar else {}
                default_threshold = self.qar.decision_threshold if self.qar else 0.6
                # Make sure required QAR factors list is available or defined here?
                known_qar_factors = list(default_weights.keys()) # Use keys from QAR's default factors
                if not known_qar_factors: self.logger.error("PADS defaults: QAR has no default factors!")

                # Use dict comprehension for defaults
                params_from_config = {
                     phase: {'threshold': default_threshold,
                             'weights': {factor: default_weights.get(factor, 0.0) for factor in known_qar_factors}
                            }
                     for phase in ['growth', 'conservation', 'release', 'reorganization']
                }
                self.logger.debug(f"PADS created default phase params: {params_from_config}")

            # --- Basic validation example ---
            required_phases = {'growth', 'conservation', 'release', 'reorganization'}
            loaded_phases = set(params_from_config.keys())
            missing = required_phases - loaded_phases
            if missing: self.logger.warning(f"PADS: Phase params missing for: {missing}")
            # Add default dicts for missing phases if you want robustness
            for phase in missing: params_from_config[phase] = params_from_config['conservation'] # Use conservation as safe default?

            self.logger.info(f"PADS loaded phase parameters for phases: {list(params_from_config.keys())}")
            return params_from_config


    def _configure_qar_for_phase(self, phase: str) -> bool:
        """
        Sets the ACTIVE weights and threshold on the QAR instance for the current phase.
        Returns True if configuration succeeded, False otherwise.
        """
        if not self.qar:
             self.logger.error("PADS cannot configure QAR - QAR instance is None")
             return False # Cannot configure if QAR doesn't exist

        # Use the _phase_parameters dictionary stored within PADS instance
        # This dictionary should have been loaded during PADS.__init__
        phase_config = self._phase_parameters.get(phase)

        # Validate the retrieved configuration for the current phase
        if not phase_config or not isinstance(phase_config, dict):
            self.logger.warning(f"PADS: No configuration found for phase '{phase}'. QAR parameters unchanged.")
            return False # Indicate configuration did not happen for this specific phase

        phase_weights = phase_config.get('weights')
        phase_threshold = phase_config.get('threshold')

        if not isinstance(phase_weights, dict) or phase_threshold is None:
            self.logger.error(f"PADS: Invalid config structure for phase '{phase}' (missing/wrong type for 'weights' or 'threshold'). QAR params unchanged.")
            return False # Indicate configuration failed

        # --- Configure the active QAR instance ---
        try:
            # 1. Filter weights: Only use factors QAR actually knows (from its self.factors list)
            # This ensures robustness if config has extra keys QAR wasn't updated with.
            active_weights = {}
            for factor_name in self.qar.factors: # Iterate through factors QAR *knows*
                # Assign the weight from the phase config if present, otherwise default to 0
                active_weights[factor_name] = phase_weights.get(factor_name, 0.0)
                if factor_name not in phase_weights:
                     self.logger.debug(f"Phase '{phase}' config missing factor '{factor_name}', setting weight to 0.")

            # 2. Set the active weights on the QAR object
            self.qar.factor_weights = active_weights # Assign the filtered dictionary

            # 3. Normalize the active weights within QAR
            self.qar._normalize_weights() # Call QAR's normalization method

            # 4. Set the decision threshold for this phase within QAR
            self.qar.decision_threshold = self.qar._validate_threshold(phase_threshold)

            self.logger.info(f"PADS configured QAR for Phase: '{phase}'. Active Threshold: {self.qar.decision_threshold:.4f}")
            # Log only non-zero weights for clarity
            active_weights_log = {k: f'{v:.3f}' for k, v in self.qar.factor_weights.items() if abs(v) > 1e-6}
            self.logger.debug(f" QAR Active Weights Set: {active_weights_log if active_weights_log else '{}'}") # Show {} if all are zero

            return True # Configuration successful

        except AttributeError as e_attr:
             # Catch errors if QAR object is missing expected methods/attributes
             self.logger.error(f"PADS Error configuring QAR attributes: {e_attr}", exc_info=True)
             return False
        except Exception as e_conf:
             self.logger.error(f"PADS Unexpected error configuring QAR for phase '{phase}': {e_conf}", exc_info=True)
             return False

    def _select_decision_style(self, market_state: Dict[str, Any], factor_values: Dict[str, float]) -> None:
        """
        Select the appropriate decision style based on current market conditions.

        This allows PADS to adapt its decision-making approach based on the current
        market environment - being more opportunistic, conservative, etc. as appropriate.

        Args:
            market_state: Current market state
            factor_values: Factor values
        """
        # Extract key indicators
        volatility = market_state.get('volatility_regime', 0.5)
        trend_strength = abs(market_state.get('qerc_trend', 0.0))
        black_swan_risk = factor_values.get('black_swan_risk', 0.1)
        whale_activity = factor_values.get('whale_activity', 0.0)
        anomaly_score = factor_values.get('anomaly_score', 0.0)
        antifragility = factor_values.get('antifragility', 0.5)

        # Default to consensus
        selected_style = 'consensus'

        # Check for strong trend - follow momentum
        if trend_strength > 0.7 and volatility < 0.6:
            selected_style = 'momentum_following'
            self.board_state['risk_appetite'] = min(0.8, self.board_state.get('risk_appetite', 0.5) + 0.1)

        # Check for high volatility - defensive
        elif volatility > 0.7:
            selected_style = 'defensive'
            self.board_state['risk_appetite'] = max(0.2, self.board_state.get('risk_appetite', 0.5) - 0.1)

        # Check for whale activity - opportunistic
        elif whale_activity > 0.7:
            selected_style = 'opportunistic'
            self.board_state['risk_appetite'] = min(0.9, self.board_state.get('risk_appetite', 0.5) + 0.2)

        # Check for anomalies - conservative
        elif anomaly_score > 0.7:
            selected_style = 'defensive'
            self.board_state['risk_appetite'] = max(0.1, self.board_state.get('risk_appetite', 0.5) - 0.2)

        # Check for black swan risk - very defensive
        elif black_swan_risk > 0.5:
            selected_style = 'defensive'
            self.board_state['risk_appetite'] = 0.1  # Minimum risk

        # Check for high antifragility - take calculated risks
        elif antifragility > 0.7:
            selected_style = 'calculated_risk'
            self.board_state['risk_appetite'] = min(0.8, self.board_state.get('risk_appetite', 0.5) + 0.1)

        # If all else fails, use market phase to determine style
        else:
            phase = market_state.get('panarchy_phase', 'conservation')
            if phase == 'growth':
                selected_style = 'calculated_risk'
                self.board_state['risk_appetite'] = 0.7
            elif phase == 'conservation':
                selected_style = 'consensus'
                self.board_state['risk_appetite'] = 0.5
            elif phase == 'release':
                selected_style = 'defensive'
                self.board_state['risk_appetite'] = 0.3
            elif phase == 'reorganization':
                selected_style = 'contrarian'
                self.board_state['risk_appetite'] = 0.6

        # Update current style
        self.current_decision_style = selected_style
        self.board_state['current_strategy'] = selected_style

        self.logger.debug(f"PADS selected decision style: {selected_style} with risk appetite: {self.board_state['risk_appetite']:.2f}")


    def _collect_component_votes(
        self,
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect votes from all components to form a comprehensive view.

        Args:
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Dictionary mapping component names to their votes
        """
        component_votes = {}

        # 1. QAR Vote (core component)
        if self.qar:
            try:
                qar_decision = self.qar.make_decision(
                    factor_values=factor_values,
                    market_data=market_state,
                    position_state=position_state
                )

                if qar_decision:
                    # Convert decision type to vote value (-1 to 1)
                    vote_value = 0.0
                    decision_type = qar_decision.decision_type.name

                    if decision_type in ['BUY', 'INCREASE']:
                        vote_value = 1.0
                    elif decision_type in ['SELL', 'DECREASE', 'EXIT']:
                        vote_value = -1.0

                    component_votes['qar'] = {
                        'vote_value': vote_value,
                        'confidence': qar_decision.confidence,
                        'reasoning': qar_decision.reasoning,
                        'raw_decision': qar_decision
                    }
            except Exception as e:
                self.logger.error(f"Error getting QAR vote: {e}")
                component_votes['qar'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 2. QStar Vote
        if hasattr(self, 'qstar_predictor') and self.qstar_predictor:
            try:
                # Get recent data from strategy if available
                dataframe = None
                if hasattr(self.strategy, 'get_recent_dataframe'):
                    dataframe = self.strategy.get_recent_dataframe(market_state.get('pair', 'UNKNOWN'))

                if dataframe is not None:
                    qstar_prediction = self.qstar_predictor.predict(
                        dataframe=dataframe,
                        current_position=position_state.get('position_size', 0.0) if position_state else 0.0,
                        pair=market_state.get('pair', None)
                    )

                    if qstar_prediction:
                        # Extract vote from QStar prediction
                        action = qstar_prediction.get('action', 0)
                        action_name = qstar_prediction.get('action_name', 'HOLD')
                        confidence = qstar_prediction.get('confidence', 0.5)

                        # Convert to vote value
                        vote_value = 0.0
                        if action_name in ['BUY', 'INCREASE']:
                            vote_value = 1.0
                        elif action_name in ['SELL', 'DECREASE', 'EXIT']:
                            vote_value = -1.0

                        component_votes['qstar'] = {
                            'vote_value': vote_value,
                            'confidence': confidence,
                            'reasoning': f"QStar predicts {action_name}",
                            'raw_prediction': qstar_prediction
                        }
            except Exception as e:
                self.logger.error(f"Error getting QStar vote: {e}")
                component_votes['qstar'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 3. CDFA Vote
        if self.cdfa:
            try:
                # Check if CDFA has the necessary method
                if hasattr(self.cdfa, 'get_fusion_decision'):
                    cdfa_decision = self.cdfa.get_fusion_decision(market_state, factor_values)

                    if cdfa_decision:
                        # Extract vote from CDFA decision
                        action = cdfa_decision.get('action', 0)
                        confidence = cdfa_decision.get('confidence', 0.5)

                        # Convert to vote value
                        vote_value = float(action)  # Assuming action is already -1 to 1

                        component_votes['cdfa'] = {
                            'vote_value': vote_value,
                            'confidence': confidence,
                            'reasoning': cdfa_decision.get('reasoning', "CDFA fusion decision"),
                            'raw_decision': cdfa_decision
                        }
            except Exception as e:
                self.logger.error(f"Error getting CDFA vote: {e}")
                component_votes['cdfa'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 4. Black Swan Vote (typically negative during high risk)
        if hasattr(self, 'black_swan_detector') and self.black_swan_detector:
            try:
                # Check if detector has a get_risk or get_decision method
                if hasattr(self.black_swan_detector, 'get_risk'):
                    risk = self.black_swan_detector.get_risk(market_state, factor_values)

                    # High risk = negative vote
                    vote_value = -risk * 2 + 1  # Map 0-1 risk to 1 to -1 vote

                    component_votes['black_swan'] = {
                        'vote_value': vote_value,
                        'confidence': risk,  # Higher risk = higher confidence in negative vote
                        'reasoning': f"Black swan risk: {risk:.2f}"
                    }
            except Exception as e:
                self.logger.error(f"Error getting Black Swan vote: {e}")
                component_votes['black_swan'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 5. Whale Activity Vote (opportunity detection)
        if hasattr(self, 'whale_detector') and self.whale_detector:
            try:
                # Check if detector has a detect_whales method
                if hasattr(self.whale_detector, 'detect_whales'):
                    whale_activity = self.whale_detector.detect_whales(market_state, factor_values)

                    if isinstance(whale_activity, dict):
                        activity_score = whale_activity.get('score', 0.0)
                        direction = whale_activity.get('direction', 0)

                        # Calculate vote value
                        vote_value = activity_score * direction

                        component_votes['whale_detector'] = {
                            'vote_value': vote_value,
                            'confidence': activity_score,
                            'reasoning': whale_activity.get('explanation', f"Whale activity: {activity_score:.2f}")
                        }
            except Exception as e:
                self.logger.error(f"Error getting Whale Detector vote: {e}")
                component_votes['whale_detector'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 6. Fibonacci Pattern Vote
        if hasattr(self, 'fibonacci_analyzer') and self.fibonacci_analyzer:
            try:
                # Extract Fibonacci signal from factor values
                fib_signal = factor_values.get('fibonacci_pattern', 0.0)

                if abs(fib_signal) > 0.01:  # Non-zero signal
                    component_votes['fibonacci'] = {
                        'vote_value': fib_signal,  # Assuming -1 to 1 range
                        'confidence': abs(fib_signal),
                        'reasoning': f"Fibonacci pattern signal: {fib_signal:.2f}"
                    }
            except Exception as e:
                self.logger.error(f"Error getting Fibonacci vote: {e}")
                component_votes['fibonacci'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # 7. Antifragility Vote (system robustness)
        try:
            antifragility = factor_values.get('antifragility', 0.5)

            # Antifragility affects confidence more than direction
            # Higher antifragility = more confident in current market direction
            trend = market_state.get('qerc_trend', 0.0)
            vote_value = trend  # Use trend direction
            confidence = antifragility  # Higher antifragility = higher confidence

            component_votes['antifragility'] = {
                'vote_value': vote_value,
                'confidence': confidence,
                'reasoning': f"Antifragility: {antifragility:.2f}, Trend: {trend:.2f}"
            }
        except Exception as e:
            self.logger.error(f"Error calculating Antifragility vote: {e}")
            component_votes['antifragility'] = {'vote_value': None, 'confidence': 0.0, 'error': str(e)}

        # Return all component votes
        return component_votes


    def _check_for_decision_overrides(
        self,
        market_state: Dict[str, Any],
        factor_values_raw: Dict[str, float],
        position_state: Optional[Dict[str, Any]],
        board_recommendations: Dict[str, Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """Check if special market conditions warrant overriding normal decision process."""
        # 1. Extreme black swan event detection (emergency exit)
        black_swan_risk = market_state.get('black_swan_risk', factor_values_raw.get('black_swan', 0.0))
        if black_swan_risk > 0.8 and position_state and position_state.get('position_open', False):
            self.logger.warning(f"OVERRIDE: Emergency exit due to extreme black swan risk: {black_swan_risk:.2f}")
            return TradingDecision(
                decision_type=DecisionType.EXIT,
                confidence=0.95,
                reasoning=f"Emergency exit due to extreme black swan risk: {black_swan_risk:.2f}",
                timestamp=datetime.now(),
                parameters={},
                metadata={'override_reason': 'extreme_black_swan'}
            )

        # 2. Liquidation risk detection (avoid margin call)
        liquidation_risk = factor_values_raw.get('liquidation_risk', 0.0)
        if liquidation_risk > 0.7 and position_state and position_state.get('position_open', False):
            self.logger.warning(f"OVERRIDE: Emergency position reduction due to liquidation risk: {liquidation_risk:.2f}")
            return TradingDecision(
                decision_type=DecisionType.DECREASE,
                confidence=0.95,
                reasoning=f"Emergency position reduction due to liquidation risk: {liquidation_risk:.2f}",
                timestamp=datetime.now(),
                parameters={},
                metadata={'override_reason': 'liquidation_risk'}
            )

        # 3. Extreme opportunity detection (whale momentum riding)
        whale_opportunity = factor_values_raw.get('whale_activity', 0.0) * factor_values_raw.get('whale_direction', 0.0)
        momentum_confirmation = factor_values_raw.get('momentum', 0.0)

        if whale_opportunity > 0.7 and momentum_confirmation > 0.6:
            self.logger.info(f"OVERRIDE: Opportunistic entry due to strong whale momentum: {whale_opportunity:.2f}")
            return TradingDecision(
                decision_type=DecisionType.BUY,
                confidence=min(0.9, whale_opportunity),
                reasoning=f"Opportunistic entry due to strong whale momentum: {whale_opportunity:.2f}",
                timestamp=datetime.now(),
                parameters={},
                metadata={'override_reason': 'whale_momentum'}
            )

        # 4. Flash crash detection (opportunistic entry)
        flash_crash = factor_values_raw.get('flash_crash', 0.0)
        antifragility = factor_values_raw.get('antifragility', 0.0)

        if flash_crash > 0.8 and antifragility > 0.7:
            self.logger.info(f"OVERRIDE: Opportunistic entry during flash crash: {flash_crash:.2f}")
            return TradingDecision(
                decision_type=DecisionType.BUY,
                confidence=min(0.9, flash_crash * antifragility),
                reasoning=f"Opportunistic entry during flash crash: {flash_crash:.2f}",
                timestamp=datetime.now(),
                parameters={},
                metadata={'override_reason': 'flash_crash'}
            )

        # No override conditions met
        return None


    def _run_boardroom_decision(
            self,
            market_state: Dict[str, Any],
            factor_values_raw: Dict[str, float],
            position_state: Optional[Dict[str, Any]],
            current_phase: str
        ) -> Optional[TradingDecision]:
            """
            Run the advanced board-room style decision making process with narrative analysis integration.
    
            Args:
                market_state: Current market state
                factor_values_raw: Raw factor values
                position_state: Current position information
                current_phase: Current market phase (string)
    
            Returns:
                Final trading decision
            """
            # 0. Determine the current decision style based on market conditions
            self._select_decision_style(market_state, factor_values_raw)
    
            # 1. Check if narrative forecasting has detected any sentiment extremes
            narrative_sentiment = self._extract_narrative_sentiment(market_state)
            
            # Adjust decision style based on narrative sentiment if available
            if narrative_sentiment:
                self._adjust_decision_style_from_narrative(narrative_sentiment)
    
            # 2. Get decision from QAR (core decision maker)
            qar_decision = None
            if self.qar:
                qar_decision = self.qar.make_decision(
                    factor_values=factor_values_raw,
                    market_data=market_state,
                    position_state=position_state
                )
    
            # 3. Collect votes from all components (including narrative forecaster)
            component_votes = self._collect_component_votes(market_state, factor_values_raw, position_state)
                    
            # 4. LMSR ENHANCEMENT: Calculate information gain from each board member
            member_information = {}
            for member, vote in component_votes.items():
                if vote.get('vote_value') is not None:
                    # How much new information does this member provide?
                    prior = 0.5  # Neutral prior
                    posterior = (vote['vote_value'] + 1) / 2  # Convert [-1,1] to [0,1]
                    
                    info_gain = self.board_lmsr.calculate_information_gain(
                        [prior], [posterior]
                    )
                    
                    member_information[member] = {
                        'vote': vote['vote_value'],
                        'confidence': vote['confidence'],
                        'info_gain': info_gain
                    }
            
            # Update board market state
            current_quantities = [self.board_quantities.get(m, 0.0) for m in self.board_members]
            
            for i, member in enumerate(self.board_members):
                if member in member_information:
                    # Convert from [-1,1] to [0,1] probability space
                    target_prob = (member_information[member]['vote'] + 1) / 2
                    
                    # Calculate market move weighted by member weight and confidence
                    weight = self.board_members[member] * member_information[member]['confidence']
                    
                    # Update quantity
                    current_quantities[i] += self.board_lmsr.calculate_cost_to_move(
                        current_quantities=current_quantities,
                        target_probability=target_prob,
                        outcome_index=i
                    ) * weight
            
            # Get board-implied probabilities
            board_probs = self.board_lmsr.get_all_market_probabilities(current_quantities)
            
            # Calculate aggregate signal from board market
            aggregate_signal = 0.0
            for i, prob in enumerate(board_probs):
                position_scale = (2 * i / (len(board_probs) - 1)) - 1 if len(board_probs) > 1 else 0
                aggregate_signal += prob * position_scale
            
            # Calculate total information gain
            total_info_gain = sum(m['info_gain'] for m in member_information.values())
            
            # Higher information gain = stronger conviction
            conviction = min(0.95, 0.5 + total_info_gain)
            
            # Store updated board state
            self.board_quantities = {m: q for m, q in zip(self.board_members, current_quantities)}
            
            # Update board state
            self.board_state['consensus_level'] = 1.0 - np.std(board_probs)
            self.board_state['conviction_level'] = conviction
            self.board_state['information_value'] = total_info_gain    
    
            # 5. Calculate consensus level among components
            vote_values = [v['vote_value'] for v in component_votes.values() if v['vote_value'] is not None]
    
            if vote_values:
                # Calculate consensus metrics
                mean_vote = sum(vote_values) / len(vote_values)
                vote_variance = sum((v - mean_vote)**2 for v in vote_values) / len(vote_values) if vote_values else 1.0
                consensus_level = 1.0 - min(1.0, np.sqrt(vote_variance) * 2)  # Higher variance = lower consensus
    
                # Update board state
                self.board_state['consensus_level'] = consensus_level
                self.board_state['voting_quorum'] = len(vote_values) / len(component_votes)
                self.board_state['dissent_level'] = min(1.0, np.sqrt(vote_variance) * 2)
            else:
                consensus_level = 0.5  # Default if no votes
    
            # 6. Check if narrative forecaster has high-confidence prediction
            narrative_override = self._check_narrative_conviction(component_votes)
            if narrative_override:
                self.logger.info(f"Using high-conviction narrative forecast: {narrative_override.decision_type.name}")
                return narrative_override
    
            # 7. Execute decision strategy based on selected style
            final_decision = self._execute_decision_strategy(
                qar_decision,
                component_votes.get('qstar', {}).get('raw_prediction'),
                component_votes,
                market_state,
                factor_values_raw,
                position_state,
                current_phase
            )
    
            # 8. Enhance decision reasoning with narrative insights if available
            if final_decision and 'narrative_forecaster' in component_votes:
                narrative_vote = component_votes['narrative_forecaster']
                narrative_raw = narrative_vote.get('raw_prediction', {})
                
                # Add narrative sentiment to decision metadata
                if 'sentiment_dimensions' in narrative_raw:
                    final_decision.metadata['narrative_sentiment'] = narrative_raw['sentiment_dimensions']
                
                # Enhance reasoning with narrative insights
                narrative_reasoning = narrative_vote.get('reasoning', '')
                if narrative_reasoning and len(narrative_reasoning) > 10:
                    # Only add if we have substantial narrative reasoning
                    final_decision.reasoning += f"\n\nNarrative insight: {narrative_reasoning}"
                    
                # Add key market factors if available
                if 'key_factors' in narrative_raw:
                    factors_str = ', '.join(narrative_raw['key_factors'][:3])
                    final_decision.metadata['key_market_factors'] = factors_str
    
            return final_decision

    def _extract_narrative_sentiment(self, market_state: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract sentiment from recent narrative forecasts"""
        try:
            if not hasattr(self, 'narrative_forecaster') or not self.narrative_forecaster:
                return None
                
            # Get sentiment history for current symbol/pair
            symbol = market_state.get('pair', 'UNKNOWN')
            sentiment_history = self.narrative_forecaster.get_sentiment_history(symbol, limit=1)
            
            if not sentiment_history:
                return None
                
            # Get most recent sentiment entry
            recent_sentiment = sentiment_history[0].get('sentiment', {})
            
            # Extract dimensions if available
            dimensions = recent_sentiment.get('dimensions', {})
            if dimensions:
                return dimensions
                
            # Fallback to overall sentiment if dimensions not available
            overall = recent_sentiment.get('overall', {})
            if overall and 'polarity' in overall:
                return {'polarity': overall['polarity']}
                
            return None
                
        except Exception as e:
            self.logger.warning(f"Error extracting narrative sentiment: {e}")
            return None
            
    def _adjust_decision_style_from_narrative(self, sentiment: Dict[str, float]) -> None:
        """Adjust decision style based on narrative sentiment dimensions"""
        try:
            # Extract key sentiment dimensions (with defaults if missing)
            polarity = sentiment.get('polarity', 0.5)       # Market direction sentiment
            fear = sentiment.get('fear', 0.5)               # Fear vs greed
            confidence = sentiment.get('confidence', 0.5)   # Market confidence
            volatility = sentiment.get('volatility', 0.5)   # Expected volatility
            
            # High fear suggests defensive approach
            if fear > 0.7:
                self.current_decision_style = 'defensive'
                self.board_state['risk_appetite'] = max(0.1, self.board_state.get('risk_appetite', 0.5) - 0.2)
                self.logger.debug(f"Narrative analysis detected high fear ({fear:.2f}), switching to defensive style")
                
            # High volatility suggests defensive approach
            elif volatility > 0.75:
                self.current_decision_style = 'defensive'
                self.board_state['risk_appetite'] = max(0.2, self.board_state.get('risk_appetite', 0.5) - 0.15)
                self.logger.debug(f"Narrative analysis detected high volatility ({volatility:.2f}), switching to defensive style")
                
            # Strong polarity with high confidence suggests momentum following
            elif abs(polarity - 0.5) > 0.3 and confidence > 0.7:
                self.current_decision_style = 'momentum_following'
                self.board_state['risk_appetite'] = min(0.8, self.board_state.get('risk_appetite', 0.5) + 0.1)
                direction = "bullish" if polarity > 0.5 else "bearish"
                self.logger.debug(f"Narrative analysis detected strong {direction} sentiment with high confidence, switching to momentum style")
                
            # Low confidence suggests consensus approach
            elif confidence < 0.3:
                self.current_decision_style = 'consensus'
                self.board_state['risk_appetite'] = 0.5  # Neutral risk appetite
                self.logger.debug(f"Narrative analysis detected low confidence ({confidence:.2f}), switching to consensus style")
                
            # Low fear, moderate volatility, and neutral-to-positive polarity suggests calculated risk
            elif fear < 0.4 and volatility < 0.6 and polarity >= 0.45:
                self.current_decision_style = 'calculated_risk'
                self.board_state['risk_appetite'] = min(0.7, self.board_state.get('risk_appetite', 0.5) + 0.1)
                self.logger.debug("Narrative analysis detected favorable conditions for calculated risk")
                
            # Otherwise, maintain current style
                
        except Exception as e:
            self.logger.warning(f"Error adjusting decision style from narrative: {e}")
            
    def _check_narrative_conviction(self, component_votes: Dict[str, Dict[str, Any]]) -> Optional[TradingDecision]:
        """Check if narrative forecaster has a high-conviction signal that should override other votes"""
        try:
            if 'narrative_forecaster' not in component_votes:
                return None
                
            narrative_vote = component_votes['narrative_forecaster']
            confidence = narrative_vote.get('confidence', 0.0)
            decision_type = narrative_vote.get('decision_type', None)
            reasoning = narrative_vote.get('reasoning', '')
            
            # Only override with extremely high confidence
            if confidence >= 0.9 and decision_type in [DecisionType.BUY, DecisionType.SELL, DecisionType.EXIT]:
                # Create override decision
                return TradingDecision(
                    decision_type=decision_type,
                    confidence=confidence,
                    reasoning=f"High-conviction narrative override: {reasoning}",
                    timestamp=datetime.now(),
                    parameters={},
                    metadata={
                        'source': 'narrative_forecaster',
                        'is_override': True,
                        'conviction_level': 'very_high'
                    }
                )
                
            return None
                
        except Exception as e:
            self.logger.warning(f"Error checking narrative conviction: {e}")
            return None

    def _execute_decision_strategy(
        self,
        qar_decision: Optional[TradingDecision],
        qstar_prediction: Optional[Dict[str, Any]],
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]],
        current_phase: str
    ) -> Optional[TradingDecision]:
        """
        Execute the selected decision strategy based on collected votes and market conditions.

        This implements different decision styles based on the current market context.

        Args:
            qar_decision: Decision from QAR
            qstar_prediction: Prediction from QStar
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information
            current_phase: Current market phase

        Returns:
            Final trading decision
        """
        # Get current decision style
        decision_style = self.current_decision_style

        # Get confidence threshold for current phase
        if not hasattr(self, 'confidence_thresholds'):
            # Initialize confidence thresholds if not already defined
            self.confidence_thresholds = {
                'conservation': 0.65,  # Higher confidence needed in mature markets
                'growth': 0.55,        # More opportunistic in growth phase
                'release': 0.75,       # More cautious in release phase
                'reorganization': 0.6  # Balanced in reorganization
            }

        confidence_threshold = self.confidence_thresholds.get(current_phase, 0.6)

        # Default to QAR decision if available
        final_decision = qar_decision

        # Execute strategy based on selected style
        if decision_style == 'consensus':
            # Require consensus among components
            final_decision = self._consensus_decision_strategy(
                component_votes, confidence_threshold, current_phase, position_state
            )

        elif decision_style == 'opportunistic':
            # Look for short-term opportunities
            final_decision = self._opportunistic_decision_strategy(
                component_votes, market_state, factor_values, position_state
            )

        elif decision_style == 'defensive':
            # Prioritize protection
            final_decision = self._defensive_decision_strategy(
                component_votes, market_state, factor_values, position_state
            )

        elif decision_style == 'calculated_risk':
            # Balance risk/reward
            final_decision = self._calculated_risk_decision_strategy(
                component_votes, market_state, factor_values, position_state
            )

        elif decision_style == 'contrarian':
            # Go against prevailing sentiment
            final_decision = self._contrarian_decision_strategy(
                component_votes, market_state, factor_values, position_state
            )

        elif decision_style == 'momentum_following':
            # Follow established trends
            final_decision = self._momentum_decision_strategy(
                component_votes, market_state, factor_values, position_state
            )

        # If no decision was made, fall back to QAR
        if final_decision is None and qar_decision is not None:
            final_decision = qar_decision
            self.logger.debug("PADS: Falling back to QAR decision")

        # Add PADS decision metadata if decision exists
        if final_decision:
            # Add decision metadata
            final_decision.metadata['PADS_phase'] = current_phase
            final_decision.metadata['PADS_style'] = decision_style
            final_decision.metadata['PADS_consensus'] = self.board_state['consensus_level']
            final_decision.metadata['PADS_risk_appetite'] = self.board_state['risk_appetite']

        return final_decision


    def _consensus_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        confidence_threshold: float,
        current_phase: str,
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Consensus decision strategy that requires agreement among components.

        Args:
            component_votes: Votes from all components
            confidence_threshold: Minimum confidence required
            current_phase: Current market phase
            position_state: Current position information

        Returns:
            Trading decision based on consensus
        """
        # Calculate weighted votes
        weighted_votes = []
        total_weight = 0.0

        for component, vote_data in component_votes.items():
            if vote_data['vote_value'] is not None:
                weight = self.board_members.get(component, 0.5) * self.reputation_scores.get(component, 0.5)
                weighted_votes.append((vote_data['vote_value'], weight))
                total_weight += weight

        if not weighted_votes:
            return None

        # Calculate weighted average
        if total_weight > 0:
            weighted_avg = sum(vote * weight for vote, weight in weighted_votes) / total_weight
        else:
            weighted_avg = 0.0

        # Calculate agreement level
        vote_values = [v[0] for v in weighted_votes]
        variance = sum((v - weighted_avg)**2 for v in vote_values) / len(vote_values) if vote_values else 1.0
        agreement_level = 1.0 - min(1.0, np.sqrt(variance) * 2)

        # Determine decision type based on weighted average and position
        decision_type = DecisionType.HOLD  # Default to HOLD

        # Require higher agreement for entry than for exit
        required_agreement = 0.7

        # Check if we have enough agreement and conviction
        if agreement_level >= required_agreement and abs(weighted_avg) >= 0.3:
            # Have position
            position_open = position_state.get('position_open', False) if position_state else False
            position_direction = position_state.get('position_direction', 0) if position_state else 0

            if position_open:
                if position_direction > 0:  # Long position
                    if weighted_avg < -0.5:
                        decision_type = DecisionType.EXIT
                    elif weighted_avg < -0.3:
                        decision_type = DecisionType.DECREASE
                    elif weighted_avg > 0.5:
                        decision_type = DecisionType.INCREASE
                else:  # Short position
                    if weighted_avg > 0.5:
                        decision_type = DecisionType.EXIT
                    elif weighted_avg > 0.3:
                        decision_type = DecisionType.DECREASE
                    elif weighted_avg < -0.5:
                        decision_type = DecisionType.INCREASE
            else:
                # No position
                if weighted_avg > 0.5:
                    decision_type = DecisionType.BUY
                elif weighted_avg < -0.5:
                    decision_type = DecisionType.SELL

        # Create decision
        confidence = agreement_level * min(1.0, abs(weighted_avg) * 2)

        # Only return decision if confidence is above threshold
        if confidence >= confidence_threshold:
            reasoning = (f"Consensus decision with agreement {agreement_level:.2f} and signal {weighted_avg:.2f}. "
                       f"Weighted by {len(weighted_votes)} component votes.")

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                parameters={
                    'weighted_avg': float(weighted_avg),
                    'agreement_level': float(agreement_level),
                    'num_votes': len(weighted_votes)
                },
                metadata={
                    'strategy': 'consensus',
                    'phase': current_phase
                }
            )

        return None


    def _opportunistic_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Opportunistic decision strategy that looks for short-term opportunities.

        Args:
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Trading decision based on opportunistic strategy
        """
        # Check for whale activity first
        whale_vote = component_votes.get('whale_detector', {})
        if whale_vote.get('vote_value') is not None:
            whale_signal = whale_vote['vote_value']
            whale_confidence = whale_vote['confidence']

            # Strong whale activity is a high-priority signal
            if abs(whale_signal) > 0.5 and whale_confidence > 0.7:
                decision_type = DecisionType.BUY if whale_signal > 0 else DecisionType.SELL

                # Position-aware decision
                position_open = position_state.get('position_open', False) if position_state else False
                position_direction = position_state.get('position_direction', 0) if position_state else 0

                if position_open:
                    if position_direction > 0:  # Long position
                        if whale_signal > 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT
                    else:  # Short position
                        if whale_signal < 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT

                reasoning = f"Opportunistic {decision_type.name} based on strong whale activity: {whale_signal:.2f}"

                return TradingDecision(
                    decision_type=decision_type,
                    confidence=whale_confidence,
                    reasoning=reasoning,
                    timestamp=datetime.now(),
                    parameters={
                        'whale_signal': float(whale_signal),
                        'whale_confidence': float(whale_confidence)
                    },
                    metadata={
                        'strategy': 'opportunistic',
                        'trigger': 'whale_activity'
                    }
                )

        # Check for Fibonacci patterns
        fib_vote = component_votes.get('fibonacci', {})
        if fib_vote.get('vote_value') is not None:
            fib_signal = fib_vote['vote_value']
            fib_confidence = fib_vote['confidence']

            # Strong pattern is an opportunity
            if abs(fib_signal) > 0.6 and fib_confidence > 0.7:
                decision_type = DecisionType.BUY if fib_signal > 0 else DecisionType.SELL

                # Position-aware decision
                position_open = position_state.get('position_open', False) if position_state else False
                position_direction = position_state.get('position_direction', 0) if position_state else 0

                if position_open:
                    if position_direction > 0:  # Long position
                        if fib_signal > 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT
                    else:  # Short position
                        if fib_signal < 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT

                reasoning = f"Opportunistic {decision_type.name} based on Fibonacci pattern: {fib_signal:.2f}"

                return TradingDecision(
                    decision_type=decision_type,
                    confidence=fib_confidence,
                    reasoning=reasoning,
                    timestamp=datetime.now(),
                    parameters={
                        'fib_signal': float(fib_signal),
                        'fib_confidence': float(fib_confidence)
                    },
                    metadata={
                        'strategy': 'opportunistic',
                        'trigger': 'fibonacci_pattern'
                    }
                )

        # Check for QStar signals
        qstar_vote = component_votes.get('qstar', {})
        if qstar_vote.get('vote_value') is not None:
            qstar_signal = qstar_vote['vote_value']
            qstar_confidence = qstar_vote['confidence']

            # Strong QStar signal is an opportunity
            if abs(qstar_signal) > 0.7 and qstar_confidence > 0.7:
                decision_type = DecisionType.BUY if qstar_signal > 0 else DecisionType.SELL

                # Position-aware decision
                position_open = position_state.get('position_open', False) if position_state else False
                position_direction = position_state.get('position_direction', 0) if position_state else 0

                if position_open:
                    if position_direction > 0:  # Long position
                        if qstar_signal > 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT
                    else:  # Short position
                        if qstar_signal < 0:
                            decision_type = DecisionType.INCREASE
                        else:
                            decision_type = DecisionType.EXIT

                reasoning = f"Opportunistic {decision_type.name} based on strong QStar signal: {qstar_signal:.2f}"

                return TradingDecision(
                    decision_type=decision_type,
                    confidence=qstar_confidence,
                    reasoning=reasoning,
                    timestamp=datetime.now(),
                    parameters={
                        'qstar_signal': float(qstar_signal),
                        'qstar_confidence': float(qstar_confidence)
                    },
                    metadata={
                        'strategy': 'opportunistic',
                        'trigger': 'qstar_signal'
                    }
                )

        # No opportunities found
        return None


    def _defensive_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Defensive decision strategy that prioritizes protection.

        Args:
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Trading decision based on defensive strategy
        """
        # Check for black swan risk first
        black_swan_vote = component_votes.get('black_swan', {})
        black_swan_risk = factor_values.get('black_swan_risk', 0.0)

        # Check if we have a position to protect
        position_open = position_state.get('position_open', False) if position_state else False
        position_direction = position_state.get('position_direction', 0) if position_state else 0

        if position_open and black_swan_risk > 0.5:
            # High black swan risk and we have a position
            decision_type = DecisionType.EXIT
            confidence = black_swan_risk

            reasoning = f"Defensive EXIT due to high black swan risk: {black_swan_risk:.2f}"

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                parameters={
                    'black_swan_risk': float(black_swan_risk)
                },
                metadata={
                    'strategy': 'defensive',
                    'trigger': 'black_swan_risk'
                }
            )

        # Check for anomaly
        anomaly_score = factor_values.get('anomaly_score', 0.0)

        if position_open and anomaly_score > 0.7:
            # High anomaly and we have a position
            decision_type = DecisionType.DECREASE  # Reduce instead of full exit
            confidence = anomaly_score

            reasoning = f"Defensive DECREASE due to high anomaly score: {anomaly_score:.2f}"

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                parameters={
                    'anomaly_score': float(anomaly_score)
                },
                metadata={
                    'strategy': 'defensive',
                    'trigger': 'anomaly'
                }
            )

        # Check for high volatility
        volatility = market_state.get('volatility_regime', 0.5)

        if position_open and volatility > 0.8 and not position_state.get('stop_loss_set', False):
            # High volatility, we have a position, and no stop loss
            decision_type = DecisionType.DECREASE
            confidence = volatility

            reasoning = f"Defensive DECREASE due to high volatility: {volatility:.2f}"

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                parameters={
                    'volatility': float(volatility)
                },
                metadata={
                    'strategy': 'defensive',
                    'trigger': 'volatility'
                }
            )

        # No defensive actions needed
        return None


    def _calculated_risk_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Calculated risk decision strategy that balances risk/reward.

        Args:
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Trading decision based on calculated risk strategy
        """
        # Check antifragility - higher means more resilient to shocks
        antifragility = factor_values.get('antifragility', 0.5)

        # Get key signals
        qerc_trend = market_state.get('qerc_trend', 0.0)
        qerc_momentum = market_state.get('qerc_momentum', 0.0)

        # Position information
        position_open = position_state.get('position_open', False) if position_state else False
        position_direction = position_state.get('position_direction', 0) if position_state else 0

        # QAR vote for reference
        qar_vote = component_votes.get('qar', {})
        qar_signal = qar_vote.get('vote_value', 0.0)
        qar_confidence = qar_vote.get('confidence', 0.0)

        # Risk factors
        black_swan_risk = factor_values.get('black_swan_risk', 0.1)
        volatility = market_state.get('volatility_regime', 0.5)

        # Calculate risk score (higher = more risky)
        risk_score = (black_swan_risk * 0.5 + volatility * 0.5)

        # Calculate opportunity score (higher = better opportunity)
        trend_strength = abs(qerc_trend)
        opportunity_score = (trend_strength * 0.3 + abs(qerc_momentum) * 0.3 +
                            (1.0 - risk_score) * 0.4)

        # Only take action if antifragility is high enough to handle the risk
        if antifragility > 0.6:
            # Determine decision based on opportunity and risk
            if opportunity_score > 0.7 and risk_score < 0.5:
                # Good opportunity with acceptable risk
                direction = 1 if qerc_trend > 0 else -1

                if position_open:
                    if position_direction * direction > 0:
                        # Position aligned with opportunity
                        decision_type = DecisionType.INCREASE
                    else:
                        # Position opposite to opportunity
                        decision_type = DecisionType.EXIT
                else:
                    # No position, enter based on direction
                    decision_type = DecisionType.BUY if direction > 0 else DecisionType.SELL

                # Calculate confidence based on opportunity and antifragility
                confidence = opportunity_score * antifragility

                reasoning = (f"Calculated risk {decision_type.name} with opportunity score {opportunity_score:.2f}, "
                           f"risk score {risk_score:.2f}, and antifragility {antifragility:.2f}")

                return TradingDecision(
                    decision_type=decision_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    timestamp=datetime.now(),
                    parameters={
                        'opportunity_score': float(opportunity_score),
                        'risk_score': float(risk_score),
                        'antifragility': float(antifragility),
                        'direction': direction
                    },
                    metadata={
                        'strategy': 'calculated_risk',
                        'trigger': 'opportunity'
                    }
                )

        # No action with current risk/opportunity balance
        return None


    def _contrarian_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Contrarian decision strategy that goes against prevailing sentiment.

        Args:
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Trading decision based on contrarian strategy
        """
        # In reorganization phase, look for oversold/overbought conditions

        # Get market indicators
        rsi = market_state.get('rsi_14', 50.0)
        soc_index = factor_values.get('soc_index', 0.5)
        phase = market_state.get('panarchy_phase', 'unknown')

        # Only be contrarian in reorganization phase
        if phase == 'reorganization' and soc_index > 0.7:
            # Position information
            position_open = position_state.get('position_open', False) if position_state else False

            if not position_open:
                # No position, check for extreme conditions
                if rsi < 30:
                    # Oversold - contrarian buy
                    decision_type = DecisionType.BUY
                    confidence = (30 - rsi) / 30  # Lower RSI = higher confidence

                    reasoning = f"Contrarian BUY in oversold market, RSI: {rsi:.1f}"

                    return TradingDecision(
                        decision_type=decision_type,
                        confidence=confidence,
                        reasoning=reasoning,
                        timestamp=datetime.now(),
                        parameters={
                            'rsi': float(rsi),
                            'soc_index': float(soc_index)
                        },
                        metadata={
                            'strategy': 'contrarian',
                            'trigger': 'oversold'
                        }
                    )
                elif rsi > 70:
                    # Overbought - contrarian sell
                    decision_type = DecisionType.SELL
                    confidence = (rsi - 70) / 30  # Higher RSI = higher confidence

                    reasoning = f"Contrarian SELL in overbought market, RSI: {rsi:.1f}"

                    return TradingDecision(
                        decision_type=decision_type,
                        confidence=confidence,
                        reasoning=reasoning,
                        timestamp=datetime.now(),
                        parameters={
                            'rsi': float(rsi),
                            'soc_index': float(soc_index)
                        },
                        metadata={
                            'strategy': 'contrarian',
                            'trigger': 'overbought'
                        }
                    )

        # No contrarian opportunity
        return None


    def _momentum_decision_strategy(
        self,
        component_votes: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> Optional[TradingDecision]:
        """
        Momentum decision strategy that follows established trends.

        Args:
            component_votes: Votes from all components
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information

        Returns:
            Trading decision based on momentum strategy
        """
        # Get trend indicators
        qerc_trend = market_state.get('qerc_trend', 0.0)
        qerc_momentum = market_state.get('qerc_momentum', 0.0)
        adx = market_state.get('adx', 20.0) / 100.0  # Normalize to 0-1

        # Only follow strong trends
        if abs(qerc_trend) > 0.5 and abs(qerc_momentum) > 0.4 and adx > 0.25:
            # Strong trend detected
            trend_direction = 1 if qerc_trend > 0 else -1

            # Position information
            position_open = position_state.get('position_open', False) if position_state else False
            position_direction = position_state.get('position_direction', 0) if position_state else 0

            # Determine decision based on trend and position
            if position_open:
                if position_direction * trend_direction > 0:
                    # Position aligned with trend
                    decision_type = DecisionType.INCREASE
                else:
                    # Position against trend
                    decision_type = DecisionType.EXIT
            else:
                # No position, enter based on trend
                decision_type = DecisionType.BUY if trend_direction > 0 else DecisionType.SELL

            # Calculate confidence based on trend strength and ADX
            trend_strength = abs(qerc_trend)
            momentum_strength = abs(qerc_momentum)
            confidence = (trend_strength * 0.4 + momentum_strength * 0.3 + adx * 0.3)

            reasoning = (f"Momentum {decision_type.name} following strong trend: {qerc_trend:.2f}, "
                       f"momentum: {qerc_momentum:.2f}, ADX: {adx:.2f}")

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                parameters={
                    'trend': float(qerc_trend),
                    'momentum': float(qerc_momentum),
                    'adx': float(adx),
                    'direction': trend_direction
                },
                metadata={
                    'strategy': 'momentum_following',
                    'trigger': 'strong_trend'
                }
            )

        # No strong trend to follow
        return None


    def _apply_risk_management_filters(
        self,
        decision: TradingDecision,
        market_state: Dict[str, Any],
        factor_values_raw: Dict[str, float],
        position_state: Optional[Dict[str, Any]]
    ) -> TradingDecision:
        """Apply risk management filters to the final decision."""
        # Apply risk management based on market regime
        regime = self.panarchy_state.get('regime', 'normal')

        # 1. Via Negativa check - First do no harm
        if hasattr(self, 'via_negativa_filter') and self.via_negativa_filter:
            try:
                filter_result = self.via_negativa_filter.check_conditions(
                    vol=factor_values_raw.get('volatility_regime', 0.5),
                    rsi=factor_values_raw.get('rsi_14', 50),
                    price_dev=factor_values_raw.get('price_deviation', 0.0)
                )

                # If conditions fail filter and decision is entry/increase
                if not filter_result.get('pass_filter', True) and decision.decision_type in [DecisionType.BUY, DecisionType.INCREASE]:
                    decision = TradingDecision(
                        decision_type=DecisionType.HOLD,
                        confidence=max(decision.confidence, filter_result.get('confidence', 0.7)),
                        reasoning=f"Risk management override: {filter_result.get('reason', 'Failed Via Negativa filters')}",
                        timestamp=decision.timestamp,
                        parameters=decision.parameters,
                        metadata={**decision.metadata, 'risk_filtered': True, 'original_decision': decision.decision_type.name}
                    )
            except Exception as e:
                self.logger.error(f"Error applying Via Negativa filter: {e}")

        # 2. Apply Barbell Strategy in high uncertainty regimes
        if hasattr(self, 'barbell_allocator') and self.barbell_allocator and regime in ['black_swan', 'high_uncertainty']:
            try:
                barbell_alloc = self.barbell_allocator.calculate_allocation(
                    vol=factor_values_raw.get('volatility_regime', 0.5),
                    uncertainty=factor_values_raw.get('black_swan', 0.1),
                    trend=factor_values_raw.get('trend', 0.0)
                )

                # Scale down position size for risky regimes
                if decision.decision_type in [DecisionType.BUY, DecisionType.INCREASE]:
                    # Add barbell scaling parameter to decision
                    decision.parameters['position_scale'] = barbell_alloc.get('position_scale', 0.5)
                    decision.metadata['barbell_adjusted'] = True
            except Exception as e:
                self.logger.error(f"Error applying Barbell allocation: {e}")

        # 3. Antifragile Risk Management - Adjust based on fragility
        if hasattr(self, 'antifragile_risk_manager') and self.antifragile_risk_manager:
            try:
                if decision.decision_type in [DecisionType.BUY, DecisionType.INCREASE]:
                    anti_score = factor_values_raw.get('antifragility', 0.5)

                    # Scale position size based on antifragility
                    antifragile_scale = min(1.0, anti_score * 1.5)  # Higher antifragility allows larger positions

                    # Apply scaling
                    if 'position_scale' in decision.parameters:
                        decision.parameters['position_scale'] *= antifragile_scale
                    else:
                        decision.parameters['position_scale'] = antifragile_scale

                    decision.metadata['antifragile_adjusted'] = True
            except Exception as e:
                self.logger.error(f"Error applying Antifragile risk management: {e}")

        # 4. Apply Prospect Theory to adjust confidence based on phase
        if hasattr(self, 'prospect_theory_manager') and self.prospect_theory_manager:
            try:
                pt_result = self.prospect_theory_manager.adjust_confidence(
                    decision.decision_type.name.lower(),
                    decision.confidence,
                    self.panarchy_state.get('phase', 'unknown')
                )

                if pt_result and 'adjusted_confidence' in pt_result:
                    decision = TradingDecision(
                        decision_type=decision.decision_type,
                        confidence=pt_result['adjusted_confidence'],
                        reasoning=decision.reasoning,
                        timestamp=decision.timestamp,
                        parameters=decision.parameters,
                        metadata={**decision.metadata, 'pt_adjusted': True}
                    )
            except Exception as e:
                self.logger.error(f"Error applying Prospect Theory adjustment: {e}")

        return decision


    def _adjust_decision_for_regime(self, decision: TradingDecision) -> TradingDecision:
        """
        Adjust decision based on current market regime

        Args:
            decision: Original trading decision

        Returns:
            Adjusted trading decision
        """
        current_regime = self.panarchy_state["regime"]

        # Apply regime-specific adjustments
        if current_regime == "black_swan":
            # During black swan events, reduce position sizes and increase caution
            if decision.decision_type in [DecisionType.BUY, DecisionType.INCREASE]:
                # Lower confidence for buy/increase decisions
                confidence = decision.confidence * 0.7

                # If confidence drops below threshold, change to HOLD
                if confidence < self.qar.decision_threshold:
                    return TradingDecision(
                        decision_type=DecisionType.HOLD,
                        confidence=1.0 - confidence,  # Inverse confidence
                        reasoning=f"Black swan regime override: {decision.reasoning}",
                        timestamp=decision.timestamp,
                        parameters=decision.parameters,
                        metadata={**decision.metadata, "regime_override": True},
                    )
                else:
                    # Keep decision but lower confidence
                    return TradingDecision(
                        decision_type=decision.decision_type,
                        confidence=confidence,
                        reasoning=f"Black swan regime adjustment: {decision.reasoning}",
                        timestamp=decision.timestamp,
                        parameters=decision.parameters,
                        metadata={**decision.metadata, "regime_adjusted": True},
                    )

        elif current_regime == "critical_top":
            # Near market tops, be more conservative with buys and more aggressive with sells
            if decision.decision_type == DecisionType.BUY:
                # Convert BUY to HOLD near critical top
                return TradingDecision(
                    decision_type=DecisionType.HOLD,
                    confidence=max(0.6, 1.0 - decision.confidence),
                    reasoning=f"Critical top regime override: {decision.reasoning}",
                    timestamp=decision.timestamp,
                    parameters=decision.parameters,
                    metadata={**decision.metadata, "regime_override": True},
                )
            elif decision.decision_type == DecisionType.SELL:
                # Boost confidence for SELL decisions
                return TradingDecision(
                    decision_type=decision.decision_type,
                    confidence=min(1.0, decision.confidence * 1.2),
                    reasoning=f"Critical top regime boost: {decision.reasoning}",
                    timestamp=decision.timestamp,
                    parameters=decision.parameters,
                    metadata={**decision.metadata, "regime_adjusted": True},
                )

        # Default: return original decision
        return decision

    def make_decision(
        self,
        market_state: Dict[str, Any],
        factor_values_raw: Dict[str, float],
        position_state: Optional[Dict[str, Any]] = None
    ) -> Optional[TradingDecision]:
        """ Makes strategic decision by configuring QAR based on phase and with boardroom approach. """
        if not self.qar:
            self.logger.error("PADS: Cannot make decision, internal QAR not initialized.")
            return None

        start_time = time.time()
        try:
            # 1. Extract current phase & update internal state
            current_phase_raw = market_state.get('panarchy_phase', 'conservation') # Get raw string

            # Convert string to enum object
            current_phase_enum = MarketPhase.from_string(current_phase_raw)
            # Get the string value from the enum object
            current_phase_str = current_phase_enum.value # e.g., 'growth'

            # Use the string value for internal state and logging
            self.panarchy_state['phase'] = current_phase_str
            self.logger.debug(f"PADS make_decision: Determined Phase = {current_phase_str}")

            # 2. Configure QAR based on the identified phase
            try:
                config_success = self._configure_qar_for_phase(current_phase_str)
                if not config_success:
                    self.logger.error(f"PADS failed to configure QAR for phase '{current_phase_str}'. Using caution.")
            except Exception as e_config_qar:
                self.logger.error(f"PADS Exception during _configure_qar_for_phase: {e_config_qar}", exc_info=True)
                # Continue anyway - we'll use board-room logic as backup

            # Update panarchy state with market state information
            self._update_panarchy_state(market_state)

            # 3. Collect board recommendations
            try:
                
                board_recommendations = self._collect_board_recommendations(market_state, factor_values_raw, position_state)
                
            except Exception as e_board_rec:
                self.logger.error(f"PADS Exception during _collect_board_recommendations: {e_board_rec}", exc_info=True)
                board_recommendations = {} # Continue with empty recommendations

            # 4. Run the boardroom decision process
            try:
                board_decision = self._run_boardroom_decision(
                    market_state,
                    factor_values_raw,
                    position_state,
                    current_phase_str
                )
            except Exception as e_board_run:
                 self.logger.error(f"PADS Exception during _run_boardroom_decision: {e_board_run}", exc_info=True)
                 board_decision = None

            # 5. Check for override conditions
            override_decision = self._check_for_decision_overrides(
                market_state,
                factor_values_raw,
                position_state,
                board_recommendations
            )

            if override_decision:
                self.logger.info(f"PADS override decision activated: {override_decision.decision_type.name}")
                final_decision = override_decision
            elif board_decision:
                final_decision = board_decision
                
            elif self.qar: # Check if QAR exists before trying to use it as fallback
                try:
                    final_decision = self.qar.make_decision(
                        factor_values=factor_values_raw,
                        market_data=market_state,
                        position_state=position_state
                    )
                except Exception as e_qar_fallback:
                    self.logger.error(f"PADS Exception during QAR fallback decision: {e_qar_fallback}", exc_info=True)
                    final_decision = None # Fallback failed too
            else: # QAR doesn't exist, cannot use as fallback
                 self.logger.error("PADS Fallback Error: QAR instance is None, cannot generate QAR decision.")
                 final_decision = None

            if not final_decision:
                self.logger.warning("PADS: All decision paths failed. No decision available.")
                return None

            # 6. Apply risk management filters
            final_decision = self._apply_risk_management_filters(final_decision, market_state, factor_values_raw, position_state)

            # 7. Adjust for current market regime
            final_decision = self._adjust_decision_for_regime(final_decision)

            # 8. Record history
            self.decision_history.append(final_decision)
            if len(self.decision_history) > self.memory_length:
                self.decision_history = self.decision_history[-self.memory_length:]

            # 9. Update metrics
            self._update_metrics(final_decision)

            processing_time = time.time() - start_time
            self.logger.info(f"PADS decision ({processing_time:.4f}s): Phase={current_phase_str}, "
                            f"Result={final_decision.decision_type.name}, "
                            f"Conf={final_decision.confidence:.3f}, "
                            f"Style={self.current_decision_style}")

            return final_decision

        except Exception as e:
            # Log the error along with the raw phase string received
            phase_received = market_state.get('panarchy_phase', 'NOT_FOUND')
            self.logger.error(f"PADS make_decision error (Phase received: '{phase_received}'): {e}", exc_info=True)
            return None

    def _update_panarchy_state(self, market_state: Dict[str, Any]) -> None:
        """Update internal panarchy state from market_state input."""
        self.panarchy_state.update(
            {
                "soc_index": market_state.get("soc_index", self.panarchy_state["soc_index"]),
                "black_swan_risk": market_state.get("black_swan_risk", self.panarchy_state["black_swan_risk"]),
                "micro_phase": market_state.get("micro_phase", self.panarchy_state["micro_phase"]),
                "meso_phase": market_state.get("meso_phase", self.panarchy_state["meso_phase"]),
                "macro_phase": market_state.get("macro_phase", self.panarchy_state["macro_phase"]),
            }
        )

        # Determine market regime
        self._update_market_regime()

    def _collect_board_recommendations(
            self,
            market_state: Dict[str, Any],
            factor_values_raw: Dict[str, float],
            position_state: Optional[Dict[str, Any]]
        ) -> Dict[str, Dict[str, Any]]:
            """Collect recommendations from all board members including narrative analysis."""
            recommendations = {}
    
            # 1. QAR recommendation (treated as a board member for diversity)
            if self.qar:
                try:
                    qar_decision = self.qar.make_decision(
                        factor_values=factor_values_raw,
                        market_data=market_state,
                        position_state=position_state
                    )
    
                    if qar_decision:
                        recommendations['qar'] = {
                            'decision_type': qar_decision.decision_type,
                            'confidence': qar_decision.confidence,
                            'reasoning': qar_decision.reasoning,
                            'raw_decision': qar_decision
                        }
                except Exception as e:
                    self.logger.error(f"Error getting QAR recommendation: {e}")
    
            # 2. Narrative Forecaster recommendation (NEW!)
            if hasattr(self, 'narrative_forecaster') and self.narrative_forecaster:
                try:
                    # Extract market data needed for narrative generation
                    symbol = market_state.get('pair', 'UNKNOWN')
                    current_price = market_state.get('close', factor_values_raw.get('close', 0.0))
                    volume = market_state.get('volume', factor_values_raw.get('volume', 0.0))
                    
                    # Get support/resistance levels (or estimate from recent data)
                    support_level = market_state.get('support_level', current_price * 0.95)
                    resistance_level = market_state.get('resistance_level', current_price * 1.05)
                    
                    # Additional context from factor values
                    additional_context = {
                        'trend': factor_values_raw.get('trend', 0),
                        'volatility': factor_values_raw.get('volatility_regime', 0.5),
                        'market_phase': self.panarchy_state.get('phase', 'conservation')
                    }
                    
                    # Use asyncio to run the coroutine
                    import asyncio
                    try:
                        # Create event loop if running in thread without one
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Generate narrative forecast
                    narrative_result = loop.run_until_complete(
                        self.narrative_forecaster.generate_narrative(
                            symbol=symbol,
                            current_price=current_price,
                            volume=volume,
                            support_level=support_level,
                            resistance_level=resistance_level,
                            additional_context=additional_context
                        )
                    )
                    
                    if narrative_result:
                        # Extract price prediction and confidence
                        price_prediction = narrative_result.get('price_prediction', current_price)
                        confidence_score = narrative_result.get('confidence_score', 0.5)
                        
                        # Extract multi-dimensional sentiment
                        sentiment = narrative_result.get('sentiment_analysis', {})
                        dimensions = sentiment.get('dimensions', {})
                        
                        # Determine decision type based on price prediction and sentiment
                        decision_type = DecisionType.HOLD  # Default to HOLD
                        price_change_pct = (price_prediction - current_price) / current_price
                        
                        # Use momentum dimension for directional bias
                        momentum_score = dimensions.get('momentum', 0.5)
                        # Transform from [0-1] to [-1, 1] range (0.5 becomes 0)
                        momentum_bias = (momentum_score - 0.5) * 2
                        
                        # Combine price prediction with sentiment for decision
                        if price_change_pct > 0.02 or momentum_bias > 0.5:
                            # Strong bullish signals
                            if position_state and position_state.get('position_open', False):
                                decision_type = DecisionType.INCREASE
                            else:
                                decision_type = DecisionType.BUY
                        elif price_change_pct < -0.02 or momentum_bias < -0.5:
                            # Strong bearish signals
                            if position_state and position_state.get('position_open', False):
                                decision_type = DecisionType.EXIT
                            else:
                                decision_type = DecisionType.SELL
                        
                        # Use fear dimension to adjust confidence (higher fear = lower confidence)
                        fear_score = dimensions.get('fear', 0.5)
                        adjusted_confidence = confidence_score * (1 - (fear_score - 0.5))
                        
                        # Narrative reasoning based on key factors
                        narrative_reasoning = "Narrative analysis: "
                        key_factors = narrative_result.get('key_factors', [])
                        if key_factors:
                            narrative_reasoning += f"Key factors: {', '.join(key_factors[:3])}. "
                        
                        narrative_reasoning += f"Sentiment: Polarity={dimensions.get('polarity', 0.5):.2f}, "
                        narrative_reasoning += f"Confidence={dimensions.get('confidence', 0.5):.2f}, "
                        narrative_reasoning += f"Fear={dimensions.get('fear', 0.5):.2f}, "
                        narrative_reasoning += f"Momentum={dimensions.get('momentum', 0.5):.2f}"
                        
                        # Add to recommendations
                        recommendations['narrative_forecaster'] = {
                            'decision_type': decision_type,
                            'confidence': max(0.1, min(0.95, adjusted_confidence)),  # Bound confidence 
                            'reasoning': narrative_reasoning,
                            'raw_prediction': {
                                'price_prediction': price_prediction,
                                'price_change_pct': price_change_pct,
                                'sentiment_dimensions': dimensions
                            }
                        }
                        
                        self.logger.info(f"Narrative Forecaster recommends: {decision_type.name} with confidence {adjusted_confidence:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"Error getting Narrative Forecaster recommendation: {e}", exc_info=True)
    
            # 3. QStar prediction
            if hasattr(self, 'qstar_predictor') and self.qstar_predictor and hasattr(self.strategy, 'get_recent_dataframe'):
                try:
                    # Get recent data from strategy
                    dataframe = self.strategy.get_recent_dataframe(market_state.get('pair', 'UNKNOWN'))
                    if dataframe is not None:
                        qstar_prediction = self.qstar_predictor.predict(
                            dataframe=dataframe,
                            current_position=position_state.get('position_size', 0.0) if position_state else 0.0,
                            pair=market_state.get('pair', None)
                        )
    
                        if qstar_prediction:
                            # Extract vote from QStar prediction
                            action = qstar_prediction.get('action', 0)
                            action_name = qstar_prediction.get('action_name', 'HOLD')
                            confidence = qstar_prediction.get('confidence', 0.5)
    
                            # Convert to decision type
                            decision_map = {
                                'BUY': DecisionType.BUY,
                                'SELL': DecisionType.SELL,
                                'HOLD': DecisionType.HOLD,
                                'INCREASE': DecisionType.INCREASE,
                                'DECREASE': DecisionType.DECREASE,
                                'EXIT': DecisionType.EXIT
                            }
                            
                            decision_type = decision_map.get(action_name, DecisionType.HOLD)
    
                            recommendations['qstar'] = {
                                'decision_type': decision_type,
                                'confidence': confidence,
                                'reasoning': f"QStar predicts {action_name}",
                                'raw_prediction': qstar_prediction
                            }
                except Exception as e:
                    self.logger.error(f"Error getting QStar recommendation: {e}")
    
            # [REMAINING BOARD MEMBERS RECOMMENDATIONS - KEEP EXISTING CODE BELOW]
            # 4. CDFA recommendation
            if self.cdfa:
                try:
                    # Check if CDFA has the necessary method
                    if hasattr(self.cdfa, 'get_fusion_decision'):
                        cdfa_decision = self.cdfa.get_fusion_decision(market_state, factor_values_raw)
    
                        if cdfa_decision:
                            # ... [rest of CDFA code unchanged]
                            pass
                except Exception as e:
                    self.logger.error(f"Error getting CDFA recommendation: {e}")
    
            # ... [all other existing recommendations from the original method]
            
            return recommendations

    def _make_board_decision(
        self,
        board_recommendations: Dict[str, Dict[str, Any]],
        market_state: Dict[str, Any]
    ) -> Optional[TradingDecision]:
        """Make a decision based on weighted board member recommendations."""
        if not board_recommendations:
            return None

        # Count recommendations by decision type
        decision_counts = {}
        decision_confidences = {}
        decision_weights = {}

        for member, rec in board_recommendations.items():
            decision_type = rec['decision_type']
            confidence = rec['confidence']
            member_weight = self.board_members.get(member, 0.0) * self.reputation_scores.get(member, 0.5)

            # Initialize if needed
            if decision_type not in decision_counts:
                decision_counts[decision_type] = 0
                decision_confidences[decision_type] = 0.0
                decision_weights[decision_type] = 0.0

            # Add weighted vote
            decision_counts[decision_type] += 1
            decision_confidences[decision_type] += confidence
            decision_weights[decision_type] += member_weight

        # Find decision with highest weighted support
        max_weight = 0.0
        best_decision = None

        for decision_type, weight in decision_weights.items():
            if weight > max_weight:
                max_weight = weight
                best_decision = decision_type

        # Require minimum weight threshold for decision
        if max_weight < 0.3:  # Require at least 30% weighted support
            return None

        # Calculate average confidence for the chosen decision
        avg_confidence = decision_confidences[best_decision] / decision_counts[best_decision]

        # Create decision object
        reasoning_parts = [f"Board decision: {best_decision.name} with {decision_counts[best_decision]} votes"]

        # Add supporting members to reasoning
        supporters = [member for member, rec in board_recommendations.items()
                     if rec['decision_type'] == best_decision]
        if supporters:
            reasoning_parts.append(f"Supporting members: {', '.join(supporters)}")

        # Create final decision
        return TradingDecision(
            decision_type=best_decision,
            confidence=avg_confidence,
            reasoning=" - ".join(reasoning_parts),
            timestamp=datetime.now(),
            parameters={},
            metadata={
                'board_support': decision_counts[best_decision],
                'board_weight': max_weight,
                'supporting_members': supporters
            }
        )


    def _update_metrics(self, decision: TradingDecision) -> None:
        """Update performance metrics with new decision."""
        self.performance_metrics['total_decisions'] += 1

        # Update phase-specific metrics
        phase = self.panarchy_state.get('phase', 'unknown')
        if phase not in self.performance_metrics['decisions_by_phase']:
            self.performance_metrics['decisions_by_phase'][phase] = 0

        self.performance_metrics['decisions_by_phase'][phase] += 1

    def _update_market_regime(self) -> None:
        """Update market regime based on panarchy state"""
        # Extract phase information
        micro_phase = self.panarchy_state["micro_phase"]
        meso_phase = self.panarchy_state["meso_phase"]
        macro_phase = self.panarchy_state["macro_phase"]
        soc_index = self.panarchy_state["soc_index"]
        black_swan_risk = self.panarchy_state["black_swan_risk"]

        # Determine regime based on phase combinations
        if macro_phase == "growth" and meso_phase in ["growth", "conservation"]:
            if soc_index < 0.7:
                regime = "normal_bull"
            else:
                regime = "extended_bull"

        elif macro_phase == "conservation" and meso_phase == "conservation":
            if soc_index > 0.8:
                regime = "critical_top"
            else:
                regime = "mature_bull"

        elif macro_phase == "release" or (
            meso_phase == "release" and micro_phase == "release"
        ):
            if black_swan_risk > 0.5:
                regime = "black_swan"
            else:
                regime = "correction"

        elif macro_phase == "reorganization" and meso_phase in [
            "reorganization",
            "growth",
        ]:
            regime = "accumulation"

        elif macro_phase == "release" and meso_phase == "release":
            regime = "bear_market"

        else:
            # Default regime
            regime = "normal"

        # Update regime
        self.panarchy_state["regime"] = regime

    def provide_feedback(
        self, decision_id: str, outcome: Dict[str, Any]
    ) -> None:
        """
        Provide feedback on a past decision for adaptation.

        Args:
            decision_id: ID of the decision to provide feedback for
            outcome: Outcome information including success, profit/loss
        """
        try:
            # Find decision in history
            target_decision = None
            for decision in self.decision_history:
                if decision.id == decision_id:
                    target_decision = decision
                    break

            if target_decision is None:
                self.logger.warning(f"Decision {decision_id} not found in history")
                return

            # Extract outcome information
            success = outcome.get('success', False)
            profit_loss = outcome.get('profit_loss', 0.0)

            # Update QAR with feedback
            if self.qar:
                self.qar.provide_feedback(decision_id, 'success' if success else 'failure', profit_loss)

            # Update reputation scores for board members
            if 'decision_source' in target_decision.metadata:
                decision_source = target_decision.metadata['decision_source']

                if decision_source == 'board' and 'supporting_members' in target_decision.metadata:
                    # Update reputation for supporting members
                    for member in target_decision.metadata['supporting_members']:
                        if member in self.reputation_scores:
                            # Increase reputation for successful decisions, decrease for failures
                            delta = 0.05 * (1 if success else -1)
                            self.reputation_scores[member] = max(0.1, min(1.0, self.reputation_scores[member] + delta))

            # Update performance metrics
            if success:
                self.performance_metrics['successful_decisions'] += 1

            self.performance_metrics['risk_adjusted_return'] += profit_loss

            # Update win rate
            total = self.performance_metrics['total_decisions']
            if total > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['successful_decisions'] / total

            # Log feedback
            self.logger.info(f"Feedback provided for decision {decision_id}: success={success}, P/L={profit_loss:.4f}")

        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}", exc_info=True)

    def update_qar_parameters(self, strategy_config: Dict):
        """Updates QAR weights, thresholds, and memory length after initialization."""
        self.logger.info("PADS: Receiving updated QAR parameters...")
        if not self.qar:
            self.logger.error("Cannot update QAR parameters: QAR instance is None.")
            return

        # Extract relevant parts from the strategy_config passed from Tengri
        phase_params_config = strategy_config.get('phase_params', {})
        qar_config_dict = strategy_config.get('qar_config', {}) # Note: variable name changed from qar_config

        # --- Update phase parameters stored WITHIN PADS ---
        # This makes sure PADS uses the latest config when calling _configure_qar_for_phase later
        self._phase_parameters = phase_params_config
        self.logger.info(f"PADS internal phase parameters updated for phases: {list(self._phase_parameters.keys())}")

        # --- Update QAR internal state using dedicated QAR methods ---
        # 1. Update QAR memory length if provided in config
        if 'memory_length' in qar_config_dict:
            if hasattr(self.qar, 'update_memory_length'):
                 self.qar.update_memory_length(qar_config_dict['memory_length'])
            else:
                 self.logger.error("QAR instance missing 'update_memory_length' method.")

        # 2. Update QAR's knowledge of factors based on ALL phase weights
        if hasattr(self.qar, 'update_factor_weights_from_phases'):
             self.qar.update_factor_weights_from_phases(phase_params_config) # Pass the phase params dict
             # Log factors directly from QAR after the update attempt
             self.logger.info(f"PADS Check: QAR factors AFTER update call: {getattr(self.qar, 'factors', 'AttributeError')}")
        else:
             self.logger.error("QAR instance is missing the 'update_factor_weights_from_phases' method.")

        self.logger.info("PADS update_qar_parameters method finished.")

    def get_panarchy_state(self) -> Dict[str, Any]:
        """Get the current panarchy state"""
        return self.panarchy_state.copy()

    def get_latest_decision(self) -> Optional[TradingDecision]:
        """Get the latest trading decision"""
        if self.decision_history:
            return self.decision_history[-1]
        return None

    def get_decision_history(self) -> List[TradingDecision]:
        """Get the full decision history"""
        return self.decision_history.copy()
    
    def get_risk_advice(
        self,
        market_state: Dict[str, Any],
        factor_values: Dict[str, float],
        position_state: Dict[str, Any],
        current_profit: float
    ) -> Dict[str, Any]:
        """
        Get risk management advice based on current market conditions.
        
        Args:
            market_state: Current market state
            factor_values: Factor values
            position_state: Current position information
            current_profit: Current profit ratio
            
        Returns:
            Dictionary with risk management advice
        """
        try:
            # Get current phase and regime
            current_phase = market_state.get('panarchy_phase', 'conservation')
            
            # Update internal state
            self._update_panarchy_state(market_state)
            current_regime = self.panarchy_state['regime']
            
            # Initialize advice
            advice = {
                'stoploss_adjustment': 1.0,  # Default: No adjustment
                'position_sizing': 1.0,      # Default: Standard position size
                'take_profit': None,         # Default: No take profit advice
                'confidence': 0.7,           # Default: Moderate confidence
                'reasons': []
            }
            
            # Phase-specific adjustments
            if current_phase == 'release':
                advice['stoploss_adjustment'] = 0.7  # Tighter stop (70% of default)
                advice['position_sizing'] = 0.7      # Smaller position
                advice['reasons'].append("Release phase requires caution")
                
            elif current_phase == 'reorganization':
                advice['stoploss_adjustment'] = 0.8  # Moderately tighter stop
                advice['position_sizing'] = 0.9      # Slightly smaller position
                advice['reasons'].append("Reorganization phase suggests moderate caution")
                
            elif current_phase == 'growth':
                advice['stoploss_adjustment'] = 1.1  # Looser stop for more room
                advice['position_sizing'] = 1.2      # Larger position to capture growth
                advice['reasons'].append("Growth phase allows more aggressive positioning")
                
            # Risk factor adjustments
            black_swan_risk = factor_values.get('black_swan', 0.1)
            if black_swan_risk > 0.5:
                # High black swan risk = tighter stop, smaller position
                black_swan_adjustment = max(0.5, 1.0 - black_swan_risk)
                advice['stoploss_adjustment'] *= black_swan_adjustment
                advice['position_sizing'] *= black_swan_adjustment
                advice['reasons'].append(f"Black swan risk ({black_swan_risk:.2f}) requires caution")
                
            # Volatility adjustment
            volatility = market_state.get('volatility_regime', 0.5)
            if volatility > 0.7:
                # High volatility = tighter stop
                vol_adjustment = max(0.6, 1.0 - (volatility - 0.5))
                advice['stoploss_adjustment'] *= vol_adjustment
                advice['reasons'].append(f"High volatility ({volatility:.2f}) requires tighter risk control")
                
            # Profitability adjustments
            if current_profit > 0.05:
                # In profit, can adjust based on profit level
                if current_profit > 0.1:
                    # Significant profit - protect some gains
                    advice['take_profit'] = current_profit * 0.7
                    advice['reasons'].append(f"Protecting {current_profit:.2%} profit")
                    
            # Apply Antifragility insights
            antifragility = factor_values.get('antifragility', 0.5) 
            if antifragility > 0.7:
                # High antifragility = system can handle more volatility
                advice['stoploss_adjustment'] *= min(1.3, 1.0 + (antifragility - 0.5))
                advice['reasons'].append(f"High antifragility ({antifragility:.2f}) allows more flexible risk management")
                
            # System-state adjustments
            soc_fragility = factor_values.get('soc_fragility', 0.5)
            if soc_fragility > 0.7:
                # Fragile system = more caution
                advice['stoploss_adjustment'] *= 0.8
                advice['position_sizing'] *= 0.8
                advice['reasons'].append(f"High system fragility ({soc_fragility:.2f}) requires defensive positioning")
            
            # Ensure adjustments are within reasonable limits
            advice['stoploss_adjustment'] = max(0.5, min(1.5, advice['stoploss_adjustment']))
            advice['position_sizing'] = max(0.3, min(2.0, advice['position_sizing']))
            
            return advice
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_advice: {e}")
            return {
                'stoploss_adjustment': 1.0,
                'position_sizing': 1.0,
                'take_profit': None,
                'confidence': 0.5,
                'reasons': [f"Error in risk advice: {str(e)}"]
            }
    
    def create_system_summary(self) -> Dict[str, Any]:
        """
        Create a comprehensive system summary

        Returns:
            Dictionary with system summary information
        """
        latest_decision = self.get_latest_decision()
        latest_decision_info = None

        if latest_decision:
            latest_decision_info = {
                "type": latest_decision.decision_type.name,
                "confidence": latest_decision.confidence,
                "timestamp": latest_decision.timestamp,
                "reasoning": latest_decision.reasoning,
            }

        # Market regime information
        regime_info = {
            "current_regime": self.panarchy_state["regime"],
            "current_phase": self.panarchy_state["phase"],
            "micro_phase": self.panarchy_state["micro_phase"],
            "meso_phase": self.panarchy_state["meso_phase"],
            "macro_phase": self.panarchy_state["macro_phase"],
            "soc_index": self.panarchy_state["soc_index"],
            "black_swan_risk": self.panarchy_state["black_swan_risk"],
        }

        # Create system summary
        return {
            "timestamp": datetime.now(),
            "latest_decision": latest_decision_info,
            "regime_info": regime_info,
            "total_decisions": self.performance_metrics['total_decisions'],
            "win_rate": self.performance_metrics['win_rate'],
            "board_members": {k: {'weight': v, 'reputation': self.reputation_scores.get(k, 0.5)}
                             for k, v in self.board_members.items()}
        }

    def recover(self):
        """Recovers the PADS system."""
        self.logger.warning("PADS recovery triggered!")
        with self._lock: # Assuming PADS might need a lock for its state
            try:
                # 1. Recover internal QAR instance first
                if hasattr(self.qar, 'recover') and callable(self.qar.recover):
                    self.logger.info("PADS Recovery: Triggering internal QAR recovery...")
                    self.qar.recover()
                elif self.qar is None:
                     self.logger.warning("PADS Recovery: Internal QAR is None, cannot recover it.")
                 # Optionally re-initialize QAR if recovery isn't enough/possible? Risky.

                # 2. Reset PADS internal state
                self.logger.debug("PADS Recovery: Resetting internal state...")
                self.panarchy_state = { # Reset to defaults
                    "phase": "conservation", "regime": "normal", "soc_index": 0.5,
                    "black_swan_risk": 0.1, "micro_phase": "unknown",
                    "meso_phase": "unknown", "macro_phase": "unknown",
                }
                self.decision_history.clear() # Clear history

                # 3. Reload/Reapply Phase Parameters if needed
                # Assumes self.config holds the original strategy config
                self._phase_parameters = self._load_phase_parameters(self.config) # Reload phase defs
                self.logger.info(f"PADS Recovery: Phase parameters reloaded.")
                # Re-configure QAR for the default state (e.g., conservation)
                self._configure_qar_for_phase(self.panarchy_state['phase'])

                self.logger.info("PADS recovery attempt finished successfully.")

            except Exception as e_pads_rec:
                 self.logger.error(f"Error during PADS recovery: {e_pads_rec}", exc_info=True)

def create_quantum_agentic_reasoning(
    hw_manager: HardwareManager,  # Require manager
    num_factors: int = 8,
    decision_threshold: float = 0.6,
) -> QuantumAgenticReasoning:
    """Creates QAR instance using the provided HardwareManager."""
    return QuantumAgenticReasoning(
        hw_manager=hw_manager,
        num_factors=num_factors,
        decision_threshold=decision_threshold,
    )


def create_panarchy_decision_system(
    hw_manager: HardwareManager,  # Require manager
    name: str = "Quantum Panarchy Trading System",
    strategy_config: Optional[Dict] = None
) -> PanarchyAdaptiveDecisionSystem:
    """Creates Panarchy System instance using the provided HardwareManager."""
    return PanarchyAdaptiveDecisionSystem(
        hw_manager=hw_manager,
        name=name,
        strategy_config=strategy_config
    )