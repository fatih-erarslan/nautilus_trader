"""
Enhanced Quantum Whale Defense System with Multi-GPU Support
Supports: GTX 1080, RX 6800XT, RTX 5090
Integration with Tengri Trading System
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import multi-GPU acceleration
from quantum_whale_defense.multi_gpu_acceleration import (
    MultiGPUAccelerator, GPUType, get_accelerator
)

# Import quantum components with graceful fallback
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None
    qnp = np

# Import hardware manager from parent
try:
    from hardware_manager import HardwareManager
except ImportError:
    HardwareManager = None

# Import messaging components
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class WhaleDetectionResult:
    """Result of whale detection analysis"""
    timestamp: datetime
    threat_level: str  # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float
    detection_type: str  # 'oscillation', 'correlation', 'game_theory', 'combined'
    whale_size_estimate: float  # Estimated USD value
    recommended_action: str
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    gpu_used: str = "CPU"


@dataclass
class DefenseStrategy:
    """Recommended defense strategy"""
    strategy_type: str  # 'aggressive', 'balanced', 'conservative', 'stealth'
    position_adjustment: float  # Percentage to adjust position
    order_modifications: List[Dict[str, Any]] = field(default_factory=list)
    hedging_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    estimated_impact: float = 0.0
    confidence: float = 0.0


class EnhancedQuantumWhaleDefense:
    """Enhanced Quantum Whale Defense with Multi-GPU Support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced whale defense system"""
        self.config = config or self._get_default_config()
        
        # Initialize multi-GPU accelerator
        self.accelerator = get_accelerator()
        logger.info(f"Initialized with GPUs: {self.accelerator.get_gpu_status()}")
        
        # Initialize quantum components
        self._initialize_quantum_components()
        
        # Initialize detection components
        self.oscillation_detector = QuantumOscillationDetectorEnhanced(
            self.config['oscillation'], self.accelerator
        )
        self.correlation_engine = QuantumCorrelationEngineEnhanced(
            self.config['correlation'], self.accelerator
        )
        self.game_theory_engine = QuantumGameTheoryEngineEnhanced(
            self.config['game_theory'], self.accelerator
        )
        
        # Performance tracking
        self.detection_history: List[WhaleDetectionResult] = []
        self.performance_metrics = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'avg_detection_time_ms': 0.0,
            'gpu_usage': {}
        }
        
        # Redis connection for real-time updates
        self.redis_client = self._init_redis() if REDIS_AVAILABLE else None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'oscillation': {
                'num_qubits': 8,
                'detection_threshold': 0.7,
                'frequency_bands': [0.001, 0.01, 0.1, 1.0],
                'window_size': 100
            },
            'correlation': {
                'num_qubits': 12,
                'timeframes': ['1m', '5m', '15m', '30m', '1h'],
                'correlation_threshold': 0.8,
                'min_timeframes': 3
            },
            'game_theory': {
                'num_qubits': 10,
                'nash_iterations': 1000,
                'strategy_threshold': 0.6,
                'payoff_matrix_size': 4
            },
            'defense': {
                'max_position_adjustment': 0.5,  # 50% max
                'emergency_exit_threshold': 0.9,
                'stealth_order_ratio': 0.3,
                'hedging_threshold': 0.7
            },
            'performance': {
                'max_latency_ms': 50,
                'min_confidence': 0.7,
                'gpu_preference': 'auto'  # 'auto', 'cuda', 'rocm', 'cpu'
            }
        }
        
    def _initialize_quantum_components(self):
        """Initialize quantum computing components"""
        if not PENNYLANE_AVAILABLE:
            logger.warning("PennyLane not available, using classical fallbacks")
            return
            
        # Determine best quantum device based on available GPUs
        if self.accelerator.primary_gpu.gpu_type == GPUType.GTX_1080:
            # Use lightning.gpu for GTX 1080 if compatible
            try:
                self.quantum_device = qml.device('lightning.kokkos', wires=33)
                logger.info("Using lightning.kokkos backend for GTX 1080")
            except:
                self.quantum_device = qml.device('default.qubit', wires=33)
                logger.info("Using default.qubit backend")
        elif self.accelerator.primary_gpu.gpu_type == GPUType.RTX_5090:
            # RTX 5090 with tensor cores - use lightning.gpu when available
            try:
                self.quantum_device = qml.device('lightning.gpu', wires=33)
                logger.info("Using lightning.gpu backend for RTX 5090")
            except:
                self.quantum_device = qml.device('lightning.kokkos', wires=33)
                logger.info("Using lightning.kokkos backend for RTX 5090")
        else:
            # Default quantum device
            self.quantum_device = qml.device('default.qubit', wires=33)
            logger.info("Using default.qubit backend")
            
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            return None
            
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            client.ping()
            logger.info("Redis connection established")
            return client
        except:
            logger.warning("Redis connection failed, continuing without real-time updates")
            return None
            
    async def detect_whale_activity(self, 
                                  market_data: pd.DataFrame,
                                  order_book: Optional[Dict[str, Any]] = None,
                                  historical_data: Optional[pd.DataFrame] = None) -> WhaleDetectionResult:
        """Main whale detection method with multi-GPU acceleration"""
        start_time = time.time()
        
        # Select optimal GPU for this workload
        data_size = len(market_data) * market_data.shape[1]
        optimal_gpu = self.accelerator.get_optimal_gpu(
            "memory" if data_size > 10_000_000 else "compute"
        )
        logger.debug(f"Using {optimal_gpu.name} for whale detection")
        
        # Run detection components in parallel
        tasks = [
            self._run_oscillation_detection(market_data, optimal_gpu),
            self._run_correlation_analysis(market_data, historical_data, optimal_gpu),
            self._run_game_theory_analysis(market_data, order_book, optimal_gpu)
        ]
        
        results = await asyncio.gather(*tasks)
        oscillation_result, correlation_result, game_theory_result = results
        
        # Combine results using quantum superposition
        combined_result = self._combine_detection_results(
            oscillation_result, correlation_result, game_theory_result
        )
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        combined_result.execution_time_ms = execution_time_ms
        combined_result.gpu_used = optimal_gpu.name
        
        # Update performance metrics
        self._update_performance_metrics(combined_result, optimal_gpu)
        
        # Store detection result
        self.detection_history.append(combined_result)
        
        # Publish to Redis if available
        if self.redis_client:
            self._publish_detection_result(combined_result)
            
        logger.info(f"Whale detection completed in {execution_time_ms:.2f}ms "
                   f"using {optimal_gpu.name}: {combined_result.threat_level}")
        
        return combined_result
        
    async def _run_oscillation_detection(self, 
                                       market_data: pd.DataFrame,
                                       gpu: Any) -> Dict[str, Any]:
        """Run quantum oscillation detection"""
        try:
            return await asyncio.to_thread(
                self.oscillation_detector.detect,
                market_data,
                gpu
            )
        except Exception as e:
            logger.error(f"Oscillation detection failed: {e}")
            return {'confidence': 0.0, 'anomaly_score': 0.0}
            
    async def _run_correlation_analysis(self,
                                      market_data: pd.DataFrame,
                                      historical_data: Optional[pd.DataFrame],
                                      gpu: Any) -> Dict[str, Any]:
        """Run quantum correlation analysis"""
        try:
            return await asyncio.to_thread(
                self.correlation_engine.analyze,
                market_data,
                historical_data,
                gpu
            )
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {'confidence': 0.0, 'manipulation_score': 0.0}
            
    async def _run_game_theory_analysis(self,
                                       market_data: pd.DataFrame,
                                       order_book: Optional[Dict[str, Any]],
                                       gpu: Any) -> Dict[str, Any]:
        """Run quantum game theory analysis"""
        try:
            return await asyncio.to_thread(
                self.game_theory_engine.analyze,
                market_data,
                order_book,
                gpu
            )
        except Exception as e:
            logger.error(f"Game theory analysis failed: {e}")
            return {'confidence': 0.0, 'optimal_strategy': None}
            
    def _combine_detection_results(self,
                                 oscillation: Dict[str, Any],
                                 correlation: Dict[str, Any],
                                 game_theory: Dict[str, Any]) -> WhaleDetectionResult:
        """Combine detection results using quantum superposition"""
        # Extract confidence scores
        osc_conf = oscillation.get('confidence', 0.0)
        corr_conf = correlation.get('confidence', 0.0)
        game_conf = game_theory.get('confidence', 0.0)
        
        # Weighted combination
        weights = np.array([0.4, 0.35, 0.25])  # Oscillation, Correlation, Game Theory
        confidences = np.array([osc_conf, corr_conf, game_conf])
        combined_confidence = np.dot(weights, confidences)
        
        # Determine threat level
        if combined_confidence >= 0.9:
            threat_level = 'CRITICAL'
        elif combined_confidence >= 0.7:
            threat_level = 'HIGH'
        elif combined_confidence >= 0.5:
            threat_level = 'MEDIUM'
        elif combined_confidence >= 0.3:
            threat_level = 'LOW'
        else:
            threat_level = 'NONE'
            
        # Estimate whale size
        whale_size = self._estimate_whale_size(oscillation, correlation, game_theory)
        
        # Determine recommended action
        action = self._determine_recommended_action(threat_level, whale_size)
        
        # Determine primary detection type
        detection_types = ['oscillation', 'correlation', 'game_theory']
        primary_detection = detection_types[np.argmax(confidences)]
        
        return WhaleDetectionResult(
            timestamp=datetime.now(),
            threat_level=threat_level,
            confidence=combined_confidence,
            detection_type=primary_detection,
            whale_size_estimate=whale_size,
            recommended_action=action,
            supporting_evidence={
                'oscillation': oscillation,
                'correlation': correlation,
                'game_theory': game_theory
            }
        )
        
    def _estimate_whale_size(self, 
                           oscillation: Dict[str, Any],
                           correlation: Dict[str, Any],
                           game_theory: Dict[str, Any]) -> float:
        """Estimate whale position size in USD"""
        # Extract size indicators from each component
        osc_size = oscillation.get('estimated_volume', 0.0)
        corr_size = correlation.get('coordinated_volume', 0.0)
        game_size = game_theory.get('opponent_resources', 0.0)
        
        # Use maximum as conservative estimate
        estimated_size = max(osc_size, corr_size, game_size)
        
        # Apply confidence weighting
        confidence = (oscillation.get('confidence', 0.0) + 
                     correlation.get('confidence', 0.0) + 
                     game_theory.get('confidence', 0.0)) / 3
        
        return estimated_size * confidence
        
    def _determine_recommended_action(self, 
                                    threat_level: str,
                                    whale_size: float) -> str:
        """Determine recommended defensive action"""
        if threat_level == 'CRITICAL':
            return "EMERGENCY_EXIT: Close 80% of position immediately"
        elif threat_level == 'HIGH':
            return "DEFENSIVE_HEDGE: Reduce position by 50% and hedge remainder"
        elif threat_level == 'MEDIUM':
            return "CAUTIOUS_MONITOR: Reduce position by 25% and tighten stops"
        elif threat_level == 'LOW':
            return "ENHANCED_MONITORING: Set alerts and prepare defensive orders"
        else:
            return "NORMAL_OPERATIONS: Continue with standard risk management"
            
    def generate_defense_strategy(self, 
                                detection_result: WhaleDetectionResult,
                                current_position: Dict[str, Any]) -> DefenseStrategy:
        """Generate specific defense strategy based on detection"""
        if detection_result.threat_level == 'NONE':
            return DefenseStrategy(
                strategy_type='none',
                position_adjustment=0.0,
                confidence=1.0
            )
            
        # Use game theory engine to calculate optimal response
        optimal_strategy = self.game_theory_engine.calculate_defense_strategy(
            detection_result,
            current_position,
            self.config['defense']
        )
        
        return optimal_strategy
        
    def _update_performance_metrics(self, 
                                  result: WhaleDetectionResult,
                                  gpu: Any):
        """Update performance tracking metrics"""
        self.performance_metrics['total_detections'] += 1
        
        # Update GPU usage stats
        gpu_name = gpu.name
        if gpu_name not in self.performance_metrics['gpu_usage']:
            self.performance_metrics['gpu_usage'][gpu_name] = 0
        self.performance_metrics['gpu_usage'][gpu_name] += 1
        
        # Update average detection time
        n = self.performance_metrics['total_detections']
        prev_avg = self.performance_metrics['avg_detection_time_ms']
        self.performance_metrics['avg_detection_time_ms'] = (
            (prev_avg * (n - 1) + result.execution_time_ms) / n
        )
        
    def _publish_detection_result(self, result: WhaleDetectionResult):
        """Publish detection result to Redis"""
        if not self.redis_client:
            return
            
        try:
            # Publish to whale detection channel
            self.redis_client.publish(
                'whale_detection',
                json.dumps({
                    'timestamp': result.timestamp.isoformat(),
                    'threat_level': result.threat_level,
                    'confidence': result.confidence,
                    'whale_size_estimate': result.whale_size_estimate,
                    'recommended_action': result.recommended_action,
                    'execution_time_ms': result.execution_time_ms,
                    'gpu_used': result.gpu_used
                })
            )
            
            # Store in Redis with TTL
            key = f"whale_detection:{result.timestamp.timestamp()}"
            self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(result.__dict__, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'performance_metrics': self.performance_metrics,
            'gpu_status': self.accelerator.get_gpu_status(),
            'detection_history_summary': {
                'total_detections': len(self.detection_history),
                'threat_distribution': self._get_threat_distribution(),
                'avg_confidence': self._get_average_confidence(),
                'recent_detections': self._get_recent_detections(5)
            },
            'system_health': self._check_system_health()
        }
        
        return report
        
    def _get_threat_distribution(self) -> Dict[str, int]:
        """Get distribution of threat levels"""
        distribution = {'NONE': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for detection in self.detection_history:
            distribution[detection.threat_level] += 1
        return distribution
        
    def _get_average_confidence(self) -> float:
        """Get average detection confidence"""
        if not self.detection_history:
            return 0.0
        return sum(d.confidence for d in self.detection_history) / len(self.detection_history)
        
    def _get_recent_detections(self, n: int) -> List[Dict[str, Any]]:
        """Get n most recent detections"""
        recent = self.detection_history[-n:] if len(self.detection_history) >= n else self.detection_history
        return [
            {
                'timestamp': d.timestamp.isoformat(),
                'threat_level': d.threat_level,
                'confidence': d.confidence,
                'whale_size_estimate': d.whale_size_estimate
            }
            for d in recent
        ]
        
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system health status"""
        health = {
            'quantum_device_available': PENNYLANE_AVAILABLE,
            'redis_connected': self.redis_client is not None,
            'gpu_acceleration_active': len(self.accelerator.available_gpus) > 1,
            'avg_latency_within_target': (
                self.performance_metrics['avg_detection_time_ms'] <= 
                self.config['performance']['max_latency_ms']
            )
        }
        
        health['overall_status'] = 'healthy' if all(health.values()) else 'degraded'
        return health


class QuantumOscillationDetectorEnhanced:
    """Enhanced Quantum Oscillation Detector with Multi-GPU Support"""
    
    def __init__(self, config: Dict[str, Any], accelerator: MultiGPUAccelerator):
        self.config = config
        self.accelerator = accelerator
        self.num_qubits = config['num_qubits']
        
    def detect(self, market_data: pd.DataFrame, gpu: Any) -> Dict[str, Any]:
        """Detect market oscillation anomalies using quantum phase estimation"""
        # Convert market data to numpy array
        prices = market_data['close'].values if 'close' in market_data else market_data.values[:, 0]
        
        # Run quantum phase estimation on GPU
        phases = self.accelerator.execute_kernel(
            'quantum_phase_estimation',
            prices,
            self.num_qubits
        )
        
        # Detect anomalies in phase distribution
        anomaly_score = self._calculate_anomaly_score(phases)
        
        # Estimate volume from oscillation patterns
        estimated_volume = self._estimate_volume_from_oscillations(phases, prices)
        
        return {
            'confidence': min(anomaly_score, 1.0),
            'anomaly_score': anomaly_score,
            'estimated_volume': estimated_volume,
            'dominant_frequencies': self._extract_dominant_frequencies(phases)
        }
        
    def _calculate_anomaly_score(self, phases: np.ndarray) -> float:
        """Calculate anomaly score from phase distribution"""
        # Look for sudden phase shifts
        phase_diff = np.diff(phases)
        anomalies = np.abs(phase_diff) > np.std(phase_diff) * 2
        return np.sum(anomalies) / len(phase_diff)
        
    def _estimate_volume_from_oscillations(self, 
                                         phases: np.ndarray,
                                         prices: np.ndarray) -> float:
        """Estimate whale volume from oscillation patterns"""
        # Larger whales create more pronounced oscillations
        oscillation_amplitude = np.std(phases)
        price_volatility = np.std(np.diff(prices) / prices[:-1])
        
        # Empirical formula relating oscillation to volume
        estimated_volume = oscillation_amplitude * price_volatility * 1e8
        return estimated_volume
        
    def _extract_dominant_frequencies(self, phases: np.ndarray) -> List[float]:
        """Extract dominant frequencies from phase data"""
        fft = np.fft.fft(phases)
        frequencies = np.fft.fftfreq(len(phases))
        
        # Get top 3 frequencies
        magnitude = np.abs(fft)
        top_indices = np.argsort(magnitude)[-3:]
        
        return frequencies[top_indices].tolist()


class QuantumCorrelationEngineEnhanced:
    """Enhanced Quantum Correlation Engine with Multi-GPU Support"""
    
    def __init__(self, config: Dict[str, Any], accelerator: MultiGPUAccelerator):
        self.config = config
        self.accelerator = accelerator
        self.num_qubits = config['num_qubits']
        
    def analyze(self, 
                market_data: pd.DataFrame,
                historical_data: Optional[pd.DataFrame],
                gpu: Any) -> Dict[str, Any]:
        """Analyze multi-timeframe correlations for manipulation detection"""
        # Prepare multi-timeframe data
        timeframe_data = self._prepare_timeframe_data(market_data, historical_data)
        
        # Run correlation analysis on GPU
        correlations = self.accelerator.execute_kernel(
            'correlation_analysis',
            timeframe_data,
            window_size=self.config.get('window_size', 100)
        )
        
        # Detect coordinated manipulation
        manipulation_score = self._detect_manipulation(correlations)
        
        # Estimate coordinated volume
        coordinated_volume = self._estimate_coordinated_volume(correlations, market_data)
        
        return {
            'confidence': min(manipulation_score, 1.0),
            'manipulation_score': manipulation_score,
            'coordinated_volume': coordinated_volume,
            'suspicious_timeframes': self._identify_suspicious_timeframes(correlations)
        }
        
    def _prepare_timeframe_data(self, 
                              market_data: pd.DataFrame,
                              historical_data: Optional[pd.DataFrame]) -> np.ndarray:
        """Prepare data for multiple timeframes"""
        # Simplified: use price data at different resolutions
        prices = market_data['close'].values if 'close' in market_data else market_data.values[:, 0]
        
        timeframes = []
        for window in [1, 5, 15, 30, 60]:  # Different minute windows
            resampled = pd.Series(prices).rolling(window).mean().dropna().values
            timeframes.append(resampled[:min(len(resampled), 1000)])  # Limit size
            
        # Pad to same length
        max_len = max(len(tf) for tf in timeframes)
        padded = np.zeros((len(timeframes), max_len))
        for i, tf in enumerate(timeframes):
            padded[i, :len(tf)] = tf
            
        return padded
        
    def _detect_manipulation(self, correlations: np.ndarray) -> float:
        """Detect coordinated manipulation from correlation patterns"""
        # High correlation across multiple timeframes indicates manipulation
        high_corr_threshold = self.config['correlation_threshold']
        high_correlations = correlations > high_corr_threshold
        
        # Count suspicious patterns
        suspicious_count = np.sum(high_correlations) - len(correlations)  # Exclude diagonal
        total_pairs = len(correlations) * (len(correlations) - 1) / 2
        
        return suspicious_count / total_pairs if total_pairs > 0 else 0.0
        
    def _estimate_coordinated_volume(self, 
                                   correlations: np.ndarray,
                                   market_data: pd.DataFrame) -> float:
        """Estimate volume involved in coordinated activity"""
        # Higher correlation = more coordinated volume
        avg_correlation = np.mean(correlations[correlations < 1.0])  # Exclude diagonal
        
        if 'volume' in market_data:
            total_volume = market_data['volume'].sum()
            coordinated_ratio = avg_correlation ** 2  # Square to emphasize high correlations
            return total_volume * coordinated_ratio
        else:
            # Estimate from price movements
            return avg_correlation * 1e7
            
    def _identify_suspicious_timeframes(self, correlations: np.ndarray) -> List[str]:
        """Identify which timeframes show suspicious correlation"""
        timeframe_names = ['1m', '5m', '15m', '30m', '1h']
        suspicious = []
        
        # Find timeframes with unusually high correlation to others
        for i, tf in enumerate(timeframe_names[:len(correlations)]):
            avg_corr = np.mean(correlations[i, :])
            if avg_corr > self.config['correlation_threshold']:
                suspicious.append(tf)
                
        return suspicious


class QuantumGameTheoryEngineEnhanced:
    """Enhanced Quantum Game Theory Engine with Multi-GPU Support"""
    
    def __init__(self, config: Dict[str, Any], accelerator: MultiGPUAccelerator):
        self.config = config
        self.accelerator = accelerator
        self.num_qubits = config['num_qubits']
        
    def analyze(self,
                market_data: pd.DataFrame,
                order_book: Optional[Dict[str, Any]],
                gpu: Any) -> Dict[str, Any]:
        """Analyze market using quantum game theory"""
        # Build payoff matrix from market conditions
        payoff_matrix = self._build_payoff_matrix(market_data, order_book)
        
        # Calculate Nash equilibrium on GPU
        nash_strategy = self.accelerator.execute_kernel(
            'game_theory_nash',
            payoff_matrix,
            iterations=self.config['nash_iterations']
        )
        
        # Analyze optimal strategy
        strategy_confidence = self._calculate_strategy_confidence(nash_strategy)
        
        # Estimate opponent resources
        opponent_resources = self._estimate_opponent_resources(market_data, order_book)
        
        return {
            'confidence': strategy_confidence,
            'optimal_strategy': self._interpret_strategy(nash_strategy),
            'opponent_resources': opponent_resources,
            'predicted_moves': self._predict_opponent_moves(nash_strategy, payoff_matrix)
        }
        
    def _build_payoff_matrix(self, 
                           market_data: pd.DataFrame,
                           order_book: Optional[Dict[str, Any]]) -> np.ndarray:
        """Build game theory payoff matrix from market conditions"""
        # 4x4 matrix: [Aggressive, Balanced, Conservative, Stealth] for both players
        matrix = np.zeros((4, 4))
        
        # Base payoffs
        base_payoffs = np.array([
            [-0.5, 0.3, 0.8, 0.1],  # Aggressive vs others
            [0.2, 0.5, 0.4, 0.6],   # Balanced vs others
            [0.8, 0.2, 0.1, 0.4],   # Conservative vs others
            [0.3, 0.7, 0.6, 0.9]    # Stealth vs others
        ])
        
        # Adjust based on market conditions
        if 'volume' in market_data:
            volatility = market_data['close'].pct_change().std()
            volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].mean()
            
            # High volatility favors aggressive strategies
            if volatility > 0.02:
                base_payoffs[0, :] *= 1.5
                
            # High volume favors conservative strategies
            if volume_ratio > 2.0:
                base_payoffs[2, :] *= 1.3
                
        return base_payoffs
        
    def _calculate_strategy_confidence(self, strategy: np.ndarray) -> float:
        """Calculate confidence in the optimal strategy"""
        # Higher concentration in one strategy = higher confidence
        max_prob = np.max(strategy)
        entropy = -np.sum(strategy * np.log(strategy + 1e-10))
        
        # Normalize confidence
        confidence = max_prob * (1 - entropy / np.log(len(strategy)))
        return min(confidence, 1.0)
        
    def _estimate_opponent_resources(self,
                                   market_data: pd.DataFrame,
                                   order_book: Optional[Dict[str, Any]]) -> float:
        """Estimate opponent (whale) resources"""
        if order_book and 'asks' in order_book and 'bids' in order_book:
            # Analyze order book imbalance
            total_ask_volume = sum(ask[1] for ask in order_book['asks'][:10])
            total_bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
            
            imbalance = abs(total_ask_volume - total_bid_volume)
            return imbalance * market_data['close'].iloc[-1]
        else:
            # Estimate from price impact
            if 'volume' in market_data:
                price_impact = market_data['close'].pct_change().abs().mean()
                avg_volume = market_data['volume'].mean()
                return price_impact * avg_volume * market_data['close'].iloc[-1] * 100
            else:
                return 1e6  # Default 1M USD
                
    def _interpret_strategy(self, strategy: np.ndarray) -> str:
        """Interpret Nash equilibrium strategy"""
        strategies = ['aggressive', 'balanced', 'conservative', 'stealth']
        return strategies[np.argmax(strategy)]
        
    def _predict_opponent_moves(self,
                              nash_strategy: np.ndarray,
                              payoff_matrix: np.ndarray) -> List[Dict[str, float]]:
        """Predict likely opponent moves"""
        strategies = ['aggressive', 'balanced', 'conservative', 'stealth']
        predictions = []
        
        # Calculate best response to our strategy
        our_strategy_idx = np.argmax(nash_strategy)
        opponent_payoffs = payoff_matrix[:, our_strategy_idx]
        
        # Normalize to probabilities
        opponent_probs = np.exp(opponent_payoffs) / np.sum(np.exp(opponent_payoffs))
        
        for i, (strategy, prob) in enumerate(zip(strategies, opponent_probs)):
            predictions.append({
                'strategy': strategy,
                'probability': float(prob)
            })
            
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)
        
    def calculate_defense_strategy(self,
                                 detection_result: WhaleDetectionResult,
                                 current_position: Dict[str, Any],
                                 defense_config: Dict[str, Any]) -> DefenseStrategy:
        """Calculate optimal defense strategy"""
        # Determine strategy type based on threat level and game theory
        if detection_result.threat_level == 'CRITICAL':
            strategy_type = 'aggressive'
            position_adjustment = -0.8  # Exit 80%
        elif detection_result.threat_level == 'HIGH':
            strategy_type = 'balanced'
            position_adjustment = -0.5  # Exit 50%
        elif detection_result.threat_level == 'MEDIUM':
            strategy_type = 'conservative'
            position_adjustment = -0.25  # Exit 25%
        else:
            strategy_type = 'stealth'
            position_adjustment = 0.0
            
        # Generate specific order modifications
        order_mods = self._generate_order_modifications(
            strategy_type,
            current_position,
            detection_result.whale_size_estimate
        )
        
        # Generate hedging recommendations
        hedging_recs = self._generate_hedging_recommendations(
            strategy_type,
            current_position,
            detection_result.confidence
        )
        
        return DefenseStrategy(
            strategy_type=strategy_type,
            position_adjustment=position_adjustment,
            order_modifications=order_mods,
            hedging_recommendations=hedging_recs,
            estimated_impact=abs(position_adjustment) * 0.002,  # 0.2% per 100% volume
            confidence=detection_result.confidence
        )
        
    def _generate_order_modifications(self,
                                    strategy_type: str,
                                    position: Dict[str, Any],
                                    whale_size: float) -> List[Dict[str, Any]]:
        """Generate specific order modifications"""
        modifications = []
        
        if strategy_type == 'aggressive':
            # Immediate market orders
            modifications.append({
                'action': 'close_position',
                'percentage': 80,
                'order_type': 'market',
                'urgency': 'immediate'
            })
        elif strategy_type == 'balanced':
            # Iceberg orders to reduce position
            modifications.append({
                'action': 'create_iceberg',
                'percentage': 50,
                'chunks': 10,
                'interval_seconds': 30
            })
        elif strategy_type == 'conservative':
            # Adjust stop losses
            modifications.append({
                'action': 'tighten_stops',
                'new_stop_distance': 0.005,  # 0.5%
                'trailing': True
            })
            
        return modifications
        
    def _generate_hedging_recommendations(self,
                                        strategy_type: str,
                                        position: Dict[str, Any],
                                        confidence: float) -> List[Dict[str, Any]]:
        """Generate hedging recommendations"""
        recommendations = []
        
        if confidence > 0.7 and strategy_type in ['aggressive', 'balanced']:
            recommendations.append({
                'instrument': 'options',
                'type': 'protective_put',
                'strike': 'atm',
                'expiry': '1_week',
                'size_percentage': 50
            })
            
        if strategy_type == 'balanced':
            recommendations.append({
                'instrument': 'futures',
                'type': 'short_hedge',
                'size_percentage': 25,
                'expiry': 'front_month'
            })
            
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize enhanced whale defense
    defense = EnhancedQuantumWhaleDefense()
    
    # Print GPU status
    print("GPU Status:", defense.accelerator.get_gpu_status())
    
    # Benchmark GPUs
    print("\nBenchmarking GPUs...")
    benchmark_results = defense.accelerator.benchmark_gpus(1000000)
    for gpu, time_sec in benchmark_results.items():
        print(f"{gpu}: {time_sec:.3f}s")
        
    # Create sample market data
    import numpy as np
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.cumsum(np.random.randn(1000) * 100),
        'volume': np.random.exponential(1000, 1000)
    })
    
    # Run async detection
    async def test_detection():
        result = await defense.detect_whale_activity(market_data)
        print(f"\nDetection Result:")
        print(f"Threat Level: {result.threat_level}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Whale Size Estimate: ${result.whale_size_estimate:,.2f}")
        print(f"Execution Time: {result.execution_time_ms:.2f}ms")
        print(f"GPU Used: {result.gpu_used}")
        print(f"Recommended Action: {result.recommended_action}")
        
        # Generate defense strategy
        current_position = {
            'symbol': 'BTC/USDT',
            'size': 10.0,
            'entry_price': 50000,
            'current_price': market_data['close'].iloc[-1]
        }
        
        strategy = defense.generate_defense_strategy(result, current_position)
        print(f"\nDefense Strategy:")
        print(f"Type: {strategy.strategy_type}")
        print(f"Position Adjustment: {strategy.position_adjustment:.1%}")
        print(f"Estimated Impact: {strategy.estimated_impact:.3%}")
        
        # Get performance report
        report = defense.get_performance_report()
        print(f"\nSystem Health: {report['system_health']['overall_status']}")
        
    # Run test
    asyncio.run(test_detection())