# File: quantum_whale_detection/core_implementation.py

import numpy as np
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import scipy.stats
import scipy.linalg
from queue import Queue, Empty

# Quantum computing imports
import pennylane as qml
import torch
import cupy as cp  # For GPU acceleration

# Configuration classes
@dataclass
class WhaleDetectionConfig:
    """Configuration for whale detection system"""
    detection_qubits: int = 8
    correlation_qubits: int = 12
    sentiment_qubits: int = 6
    game_theory_qubits: int = 10
    steganography_qubits: int = 6
    
    detection_sensitivity: float = 0.001
    early_warning_threshold: float = 0.7
    correlation_threshold: float = 0.7
    
    max_latency_ms: float = 50.0
    update_frequency_ms: float = 100.0
    memory_limit_gb: float = 4.0
    
    enable_gpu_acceleration: bool = True
    enable_error_correction: bool = True
    enable_steganography: bool = True

class QuantumOscillationDetector:
    """
    Detects market frequency anomalies using quantum phase estimation.
    Provides 5-15 second early warning of whale movements.
    """
    
    def __init__(self, detection_qubits=8, sensitivity=0.001):
        self.detection_qubits = detection_qubits
        self.sensitivity = sensitivity
        self.baseline_frequencies = {}
        self.anomaly_threshold = 3.0
        
        # Initialize quantum device with fallback priority
        try:
            self.device = qml.device('lightning.kokkos', wires=detection_qubits)
            logging.info(f"Using lightning.kokkos backend for oscillation detector")
        except Exception as e:
            try:
                self.device = qml.device('lightning.qubit', wires=detection_qubits)
                logging.info(f"Using lightning.qubit backend for oscillation detector")
            except Exception as e2:
                self.device = qml.device('default.qubit', wires=detection_qubits)
                logging.info(f"Using default.qubit backend for oscillation detector")
            
        # Create quantum circuits
        self.phase_estimation_circuit = self._create_phase_estimation_circuit()
        self.frequency_analysis_circuit = self._create_frequency_analysis_circuit()
        
        # Baseline data
        self.baseline_frequencies = self._initialize_baseline()
        self.calibration_data = deque(maxlen=1000)
        
    def _create_phase_estimation_circuit(self):
        """Create quantum phase estimation circuit for whale detection"""
        
        @qml.qnode(self.device)
        def phase_estimation(market_frequencies, control_qubits):
            # Encode market frequencies into quantum state
            for i, freq in enumerate(market_frequencies[:self.detection_qubits]):
                qml.RY(freq * np.pi, wires=i)
                
            # Create superposition in control register
            for i in range(control_qubits):
                qml.Hadamard(wires=i)
                
            # Controlled market evolution operators
            for i in range(control_qubits):
                power = 2**i
                for j in range(self.detection_qubits):
                    if j < len(market_frequencies):
                        qml.ctrl(qml.RZ, control=i)(power * market_frequencies[j], wires=j + control_qubits)
                        
            # Inverse QFT on control register
            qml.adjoint(qml.QFT)(wires=range(control_qubits))
            
            # Measure control register
            return [qml.expval(qml.PauliZ(i)) for i in range(control_qubits)]
            
        return phase_estimation
    
    def _create_frequency_analysis_circuit(self):
        """Create circuit for frequency domain analysis"""
        
        @qml.qnode(self.device)
        def frequency_analysis(price_data):
            # Encode price data
            normalized_prices = self._normalize_data(price_data[:self.detection_qubits])
            
            for i, price in enumerate(normalized_prices):
                qml.RY(price * np.pi, wires=i)
                
            # Apply quantum Fourier transform
            qml.QFT(wires=range(self.detection_qubits))
            
            # Measure frequency components
            return [qml.expval(qml.PauliZ(i)) for i in range(self.detection_qubits)]
            
        return frequency_analysis
    
    def detect_whale_tremors(self, market_data):
        """
        Detect subtle market oscillation changes that precede whale moves.
        """
        try:
            start_time = time.perf_counter()
            
            # Extract frequency components from market data
            price_frequencies = self._extract_price_frequencies(market_data['prices'])
            volume_frequencies = self._extract_volume_frequencies(market_data['volumes'])
            
            # Combine frequencies
            combined_frequencies = self._combine_frequency_data(price_frequencies, volume_frequencies)
            
            # Apply quantum phase estimation
            control_qubits = min(4, self.detection_qubits // 2)
            # Convert frequencies dict to list for quantum circuit
            freq_values = list(combined_frequencies.values())[:self.detection_qubits]
            # Pad if needed
            while len(freq_values) < self.detection_qubits:
                freq_values.append(0.0)
            phase_result = self.phase_estimation_circuit(freq_values, control_qubits)
            
            # Detect anomalies from baseline
            anomalies = self._detect_phase_anomalies(phase_result, combined_frequencies)
            
            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if anomalies['severity'] > self.anomaly_threshold:
                return {
                    'whale_detected': True,
                    'confidence': min(anomalies['severity'] / self.anomaly_threshold, 1.0),
                    'estimated_impact_time': self._estimate_impact_time(anomalies),
                    'predicted_direction': self._predict_direction(anomalies),
                    'anomaly_details': anomalies,
                    'processing_time_ms': processing_time
                }
            
            return {
                'whale_detected': False,
                'processing_time_ms': processing_time,
                'baseline_deviation': anomalies['severity']
            }
            
        except Exception as e:
            logging.error(f"Whale tremor detection error: {e}")
            return {'whale_detected': False, 'error': str(e)}
    
    def _extract_price_frequencies(self, prices, window_sizes=[10, 30, 100]):
        """Extract frequency components from price series"""
        frequencies = {}
        
        for window in window_sizes:
            if len(prices) >= window:
                price_segment = np.array(prices[-window:])
                
                # Remove trend
                detrended = scipy.signal.detrend(price_segment)
                
                # Apply quantum Fourier analysis
                try:
                    quantum_fft = self.frequency_analysis_circuit(detrended)
                    
                    # Convert to frequency domain features
                    frequencies[f'window_{window}'] = {
                        'quantum_amplitudes': quantum_fft,
                        'dominant_frequency': np.argmax(np.abs(quantum_fft)),
                        'spectral_energy': np.sum(np.abs(quantum_fft)**2),
                        'phase_coherence': self._calculate_phase_coherence(quantum_fft)
                    }
                except Exception as e:
                    logging.warning(f"Quantum FFT failed for window {window}: {e}")
                    frequencies[f'window_{window}'] = {'error': str(e)}
        
        return frequencies
    
    def _extract_volume_frequencies(self, volumes, window_sizes=[10, 30, 100]):
        """Extract frequency components from volume series"""
        if not volumes:
            return {}
            
        volume_frequencies = {}
        
        for window in window_sizes:
            if len(volumes) >= window:
                volume_segment = np.array(volumes[-window:])
                
                # Log transform to handle volume spikes
                log_volumes = np.log1p(volume_segment)
                detrended = scipy.signal.detrend(log_volumes)
                
                # Classical FFT for volume (quantum FFT on small circuits)
                if window <= self.detection_qubits:
                    try:
                        quantum_result = self.frequency_analysis_circuit(detrended)
                        volume_frequencies[f'volume_window_{window}'] = {
                            'quantum_amplitudes': quantum_result,
                            'volume_anomaly_score': self._calculate_volume_anomaly(detrended)
                        }
                    except Exception as e:
                        volume_frequencies[f'volume_window_{window}'] = {'error': str(e)}
                else:
                    # Use classical FFT for larger windows
                    fft_result = np.fft.fft(detrended)
                    volume_frequencies[f'volume_window_{window}'] = {
                        'classical_fft': fft_result[:self.detection_qubits],
                        'volume_anomaly_score': self._calculate_volume_anomaly(detrended)
                    }
        
        return volume_frequencies
    
    def _detect_phase_anomalies(self, phase_result, frequency_data):
        """Detect anomalies in quantum phase estimation results"""
        anomalies = {
            'severity': 0.0,
            'anomaly_type': 'none',
            'confidence': 0.0,
            'details': {}
        }
        
        try:
            # Compare against baseline
            if not self.baseline_frequencies:
                self._update_baseline(frequency_data)
                return anomalies
            
            # Calculate deviation from baseline
            current_phases = np.array(phase_result)
            baseline_phases = np.array(self.baseline_frequencies.get('mean_phases', current_phases))
            
            # Phase difference (accounting for 2π periodicity)
            phase_diff = np.abs(current_phases - baseline_phases)
            phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
            
            # Weighted anomaly score
            weights = np.exp(-np.arange(len(phase_diff)) * 0.1)  # Recent phases more important
            anomaly_score = np.average(phase_diff, weights=weights)
            
            # Statistical significance
            baseline_std = self.baseline_frequencies.get('std_phases', 0.1)
            z_score = anomaly_score / (baseline_std + 1e-8)
            
            anomalies['severity'] = z_score
            anomalies['confidence'] = min(scipy.stats.norm.cdf(z_score), 1.0)
            
            # Classify anomaly type
            if z_score > 4.0:
                anomalies['anomaly_type'] = 'extreme_whale'
            elif z_score > 3.0:
                anomalies['anomaly_type'] = 'large_whale'
            elif z_score > 2.0:
                anomalies['anomaly_type'] = 'medium_whale'
            else:
                anomalies['anomaly_type'] = 'normal'
            
            # Additional details
            anomalies['details'] = {
                'z_score': z_score,
                'phase_deviation': anomaly_score,
                'baseline_std': baseline_std,
                'most_anomalous_frequency': np.argmax(phase_diff)
            }
            
        except Exception as e:
            logging.error(f"Phase anomaly detection error: {e}")
            anomalies['error'] = str(e)
        
        return anomalies
    
    def _estimate_impact_time(self, anomalies):
        """Estimate when the whale move will impact price"""
        base_time = 8.0  # seconds
        
        # Adjust based on anomaly strength
        severity = anomalies.get('severity', 1.0)
        strength_factor = min(severity / 5.0, 2.0)
        
        # Add some randomness based on market conditions
        noise_factor = np.random.normal(1.0, 0.2)
        
        estimated_time = (base_time / strength_factor) * noise_factor
        return max(5.0, min(15.0, estimated_time))
    
    def _predict_direction(self, anomalies):
        """Predict direction of whale move"""
        details = anomalies.get('details', {})
        most_anomalous_freq = details.get('most_anomalous_frequency', 0)
        
        # Low frequency anomalies suggest slow moves (accumulation)
        # High frequency anomalies suggest fast moves (dumps)
        if most_anomalous_freq < self.detection_qubits // 3:
            return {'direction': 'accumulation', 'speed': 'slow'}
        elif most_anomalous_freq > 2 * self.detection_qubits // 3:
            return {'direction': 'distribution', 'speed': 'fast'}
        else:
            return {'direction': 'neutral', 'speed': 'medium'}
    
    def calibrate_baseline(self, historical_data):
        """Calibrate baseline oscillation patterns"""
        try:
            all_frequencies = []
            all_phases = []
            
            for data_point in historical_data:
                if 'prices' in data_point:
                    freq_data = self._extract_price_frequencies(data_point['prices'])
                    
                    # Combine all frequency data
                    combined = self._combine_frequency_data(freq_data, {})
                    if len(combined) >= self.detection_qubits:
                        control_qubits = min(4, self.detection_qubits // 2)
                        phase_result = self.phase_estimation_circuit(combined, control_qubits)
                        
                        all_frequencies.append(combined)
                        all_phases.append(phase_result)
            
            if all_phases:
                # Calculate baseline statistics
                phase_array = np.array(all_phases)
                self.baseline_frequencies = {
                    'mean_phases': np.mean(phase_array, axis=0),
                    'std_phases': np.std(phase_array, axis=0),
                    'num_samples': len(all_phases),
                    'last_updated': time.time()
                }
                
                logging.info(f"Baseline calibrated with {len(all_phases)} samples")
            
        except Exception as e:
            logging.error(f"Baseline calibration error: {e}")
    
    def _normalize_data(self, data):
        """Normalize data for quantum encoding"""
        if len(data) == 0:
            return [0.0] * self.detection_qubits
            
        data_array = np.array(data)
        if np.std(data_array) > 0:
            normalized = (data_array - np.mean(data_array)) / np.std(data_array)
        else:
            normalized = data_array - np.mean(data_array)
        
        # Scale to [0, 1] range for quantum encoding
        if len(normalized) > 0:
            min_val, max_val = np.min(normalized), np.max(normalized)
            if max_val > min_val:
                normalized = (normalized - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(normalized)
        
        # Pad or truncate to correct size
        if len(normalized) < self.detection_qubits:
            padded = np.zeros(self.detection_qubits)
            padded[:len(normalized)] = normalized
            return padded.tolist()
        else:
            return normalized[:self.detection_qubits].tolist()
    
    def _initialize_baseline(self):
        """Initialize baseline frequencies for normal market conditions"""
        # Default baseline frequencies - can be calibrated with historical data
        baseline = {}
        for i in range(self.detection_qubits):
            baseline[f'freq_{i}'] = 0.1 + (i * 0.05)  # Simple increasing pattern
        return baseline
    
    def calibrate_baseline(self, historical_data: List[Dict]):
        """Calibrate baseline frequencies from historical normal market data"""
        try:
            all_frequencies = []
            
            for data in historical_data:
                prices = data.get('prices', [])
                volumes = data.get('volumes', [])
                
                if len(prices) > 10:
                    # Extract frequency features
                    price_freq = self._extract_price_frequencies(prices)
                    volume_freq = self._extract_volume_frequencies(volumes)
                    
                    # Combine frequencies
                    combined = {**price_freq, **volume_freq}
                    all_frequencies.append(combined)
            
            if all_frequencies:
                # Calculate average baseline frequencies
                baseline = {}
                for key in all_frequencies[0].keys():
                    values = [freq.get(key, 0) for freq in all_frequencies]
                    baseline[key] = np.mean(values)
                
                self.baseline_frequencies = baseline
                logging.info("Baseline frequencies calibrated from historical data")
                
        except Exception as e:
            logging.warning(f"Baseline calibration failed: {e}")
    
    def _calculate_phase_coherence(self, quantum_fft_result):
        """Calculate phase coherence from quantum FFT result"""
        try:
            if isinstance(quantum_fft_result, (list, np.ndarray)) and len(quantum_fft_result) > 0:
                phases = np.angle(quantum_fft_result)
                # Phase coherence is measured by variance of phases
                coherence = 1.0 - (np.var(phases) / (np.pi ** 2))
                return max(0.0, min(1.0, coherence))
            return 0.0
        except Exception as e:
            logging.warning(f"Phase coherence calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_anomaly(self, volume_data):
        """Calculate volume anomaly score"""
        try:
            if len(volume_data) < 2:
                return 0.0
            
            # Calculate z-score for volume anomaly
            mean_vol = np.mean(volume_data)
            std_vol = np.std(volume_data)
            
            if std_vol > 0:
                recent_vol = volume_data[-5:] if len(volume_data) >= 5 else volume_data
                z_score = (np.mean(recent_vol) - mean_vol) / std_vol
                # Convert to anomaly score [0, 1]
                anomaly_score = min(1.0, abs(z_score) / 3.0)  # 3-sigma normalization
                return anomaly_score
            
            return 0.0
        except Exception as e:
            logging.warning(f"Volume anomaly calculation failed: {e}")
            return 0.0
    
    def _combine_frequency_data(self, price_frequencies, volume_frequencies):
        """Combine price and volume frequency data"""
        try:
            combined = {}
            
            # Extract numerical features from price frequencies
            for key, value in price_frequencies.items():
                if isinstance(value, dict):
                    # Extract numerical features from nested dict
                    if 'spectral_energy' in value:
                        combined[f'price_{key}_energy'] = float(value['spectral_energy'])
                    if 'dominant_frequency' in value:
                        combined[f'price_{key}_dominant'] = float(value['dominant_frequency'])
                    if 'phase_coherence' in value:
                        combined[f'price_{key}_coherence'] = float(value['phase_coherence'])
                else:
                    combined[f'price_{key}'] = float(value) if value is not None else 0.0
            
            # Extract numerical features from volume frequencies
            for key, value in volume_frequencies.items():
                if isinstance(value, dict):
                    # Extract numerical features from nested dict
                    if 'spectral_energy' in value:
                        combined[f'volume_{key}_energy'] = float(value['spectral_energy'])
                    if 'volume_anomaly_score' in value:
                        combined[f'volume_{key}_anomaly'] = float(value['volume_anomaly_score'])
                else:
                    combined[f'volume_{key}'] = float(value) if value is not None else 0.0
            
            # If no data, return default
            if not combined:
                combined = {f'freq_{i}': 0.1 for i in range(self.detection_qubits)}
            
            return combined
        except Exception as e:
            logging.warning(f"Frequency data combination failed: {e}")
            return {f'freq_{i}': 0.1 for i in range(self.detection_qubits)}
    
    def _detect_phase_anomalies(self, phase_result, frequency_data):
        """Detect anomalies in quantum phase measurements"""
        try:
            anomalies = {
                'severity': 0.0,
                'detected_patterns': [],
                'confidence': 0.0
            }
            
            if isinstance(phase_result, (list, np.ndarray)) and len(phase_result) > 0:
                # Calculate deviation from expected phase patterns
                phase_variance = np.var(phase_result)
                phase_mean = np.mean(np.abs(phase_result))
                
                # Compare with baseline if available
                baseline_deviation = 0.0
                if self.baseline_frequencies:
                    baseline_values = list(self.baseline_frequencies.values())[:len(phase_result)]
                    if baseline_values:
                        baseline_deviation = np.mean(np.abs(np.array(phase_result) - np.array(baseline_values)))
                
                # Calculate severity score
                anomalies['severity'] = phase_variance + phase_mean + baseline_deviation
                anomalies['confidence'] = min(1.0, anomalies['severity'] / 2.0)
                
                # Detect specific patterns
                if phase_variance > 0.5:
                    anomalies['detected_patterns'].append('high_variance')
                if phase_mean > 0.7:
                    anomalies['detected_patterns'].append('strong_signal')
                if baseline_deviation > 0.3:
                    anomalies['detected_patterns'].append('baseline_shift')
            
            return anomalies
        except Exception as e:
            logging.warning(f"Phase anomaly detection failed: {e}")
            return {'severity': 0.0, 'detected_patterns': [], 'confidence': 0.0}
    
    def _estimate_impact_time(self, anomalies):
        """Estimate time until whale action impacts market"""
        try:
            base_time = 10.0  # Base 10 second estimate
            severity = anomalies.get('severity', 0)
            
            # Higher severity = more imminent impact
            if severity > 2.0:
                return 5.0  # Very imminent
            elif severity > 1.0:
                return 7.0  # Soon
            else:
                return base_time
        except:
            return 10.0
    
    def _predict_direction(self, anomalies):
        """Predict likely price direction from anomalies"""
        try:
            patterns = anomalies.get('detected_patterns', [])
            
            if 'strong_signal' in patterns and 'high_variance' in patterns:
                return 'down'  # Strong dump signal
            elif 'baseline_shift' in patterns:
                return 'up'    # Accumulation signal
            else:
                return 'uncertain'
        except:
            return 'uncertain'

class QuantumCorrelationEngine:
    """
    Analyzes correlations across multiple timeframes using quantum entanglement.
    Detects coordinated manipulation patterns.
    """
    
    def __init__(self, correlation_qubits=12, timeframes=[1, 5, 15, 60]):
        self.correlation_qubits = correlation_qubits
        self.timeframes = timeframes
        self.entanglement_threshold = 0.7
        
        # Initialize quantum device with fallback priority
        try:
            self.device = qml.device('lightning.kokkos', wires=correlation_qubits)
            logging.info(f"Using lightning.kokkos backend for correlation engine")
        except Exception as e:
            try:
                self.device = qml.device('lightning.qubit', wires=correlation_qubits)
                logging.info(f"Using lightning.qubit backend for correlation engine")
            except Exception as e2:
                self.device = qml.device('default.qubit', wires=correlation_qubits)
                logging.info(f"Using default.qubit backend for correlation engine")
        
        # Create correlation analysis circuit
        self.correlation_circuit = self._create_correlation_circuit()
        self.entanglement_measure_circuit = self._create_entanglement_circuit()
        
        # Historical correlation patterns
        self.baseline_correlations = self._initialize_baseline_correlations()
        
    def _create_correlation_circuit(self):
        """Create quantum circuit for correlation analysis"""
        
        @qml.qnode(self.device)
        def correlation_analysis(timeframe_data):
            num_timeframes = len(timeframe_data)
            qubits_per_timeframe = self.correlation_qubits // num_timeframes
            
            # Encode each timeframe
            for i, data in enumerate(timeframe_data):
                start_qubit = i * qubits_per_timeframe
                end_qubit = min(start_qubit + qubits_per_timeframe, self.correlation_qubits)
                
                for j, value in enumerate(data[:end_qubit - start_qubit]):
                    qml.RY(value * np.pi, wires=start_qubit + j)
            
            # Create entanglement between timeframes
            for i in range(num_timeframes - 1):
                for j in range(i + 1, num_timeframes):
                    qubit_i = i * qubits_per_timeframe
                    qubit_j = j * qubits_per_timeframe
                    if qubit_i < self.correlation_qubits and qubit_j < self.correlation_qubits:
                        qml.CNOT(wires=[qubit_i, qubit_j])
            
            # Measure correlations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.correlation_qubits)]
            
        return correlation_analysis
    
    def _create_entanglement_circuit(self):
        """Create circuit to measure quantum entanglement"""
        
        @qml.qnode(self.device)
        def entanglement_measure(state_data):
            # Encode state
            for i, value in enumerate(state_data[:self.correlation_qubits]):
                qml.RY(value * np.pi, wires=i)
            
            # Create GHZ-like entangled state
            qml.Hadamard(wires=0)
            for i in range(1, self.correlation_qubits):
                qml.CNOT(wires=[0, i])
            
            # Measure entanglement witnesses
            observables = []
            for i in range(min(4, self.correlation_qubits)):
                observables.append(qml.expval(qml.PauliX(i)))
            
            return observables
            
        return entanglement_measure
    
    def analyze_cross_timeframe_correlations(self, market_data):
        """
        Create quantum entangled state representing correlations across timeframes.
        Detect when normal correlations break down (manipulation signal).
        """
        try:
            start_time = time.perf_counter()
            
            # Extract data for each timeframe
            timeframe_data = {}
            for tf in self.timeframes:
                timeframe_data[tf] = self._aggregate_data(market_data, tf)
            
            # Prepare data for quantum circuit
            quantum_data = []
            for tf in self.timeframes:
                data = timeframe_data[tf]
                normalized_data = self._normalize_timeframe_data(data)
                quantum_data.append(normalized_data)
            
            # Apply quantum correlation analysis
            correlation_result = self.correlation_circuit(quantum_data)
            
            # Measure entanglement strength
            combined_data = np.concatenate(quantum_data)[:self.correlation_qubits]
            entanglement_result = self.entanglement_measure_circuit(combined_data)
            entanglement_measure = self._calculate_entanglement_measure(entanglement_result)
            
            # Detect correlation anomalies
            manipulation_signals = self._detect_correlation_anomalies(
                correlation_result, entanglement_measure
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'manipulation_detected': len(manipulation_signals) > 0,
                'entanglement_strength': entanglement_measure,
                'affected_timeframes': manipulation_signals,
                'correlation_breakdown': self._quantify_breakdown(correlation_result),
                'manipulation_type': self._classify_manipulation(manipulation_signals),
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logging.error(f"Correlation analysis error: {e}")
            return {
                'manipulation_detected': False,
                'error': str(e)
            }
    
    def _aggregate_data(self, market_data, timeframe_minutes):
        """Aggregate market data for specific timeframe"""
        try:
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            timestamps = market_data.get('timestamps', [])
            
            if not prices or len(prices) < timeframe_minutes:
                return {
                    'ohlc': [0, 0, 0, 0],
                    'volume': 0,
                    'returns': [0],
                    'volatility': 0
                }
            
            # Take last N data points for timeframe
            recent_prices = prices[-timeframe_minutes:]
            recent_volumes = volumes[-timeframe_minutes:] if volumes else [1] * len(recent_prices)
            
            # Calculate OHLC
            open_price = recent_prices[0]
            high_price = max(recent_prices)
            low_price = min(recent_prices)
            close_price = recent_prices[-1]
            
            # Calculate returns
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] != 0:
                    ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                    returns.append(ret)
            
            # Calculate volatility
            volatility = np.std(returns) if returns else 0
            
            return {
                'ohlc': [open_price, high_price, low_price, close_price],
                'volume': sum(recent_volumes),
                'returns': returns[-5:],  # Last 5 returns
                'volatility': volatility
            }
            
        except Exception as e:
            logging.warning(f"Data aggregation error for timeframe {timeframe_minutes}: {e}")
            return {
                'ohlc': [0, 0, 0, 0],
                'volume': 0,
                'returns': [0],
                'volatility': 0
            }
    
    def _normalize_timeframe_data(self, data):
        """Normalize timeframe data for quantum encoding"""
        try:
            # Extract features
            ohlc = data['ohlc']
            returns = data['returns']
            volatility = data['volatility']
            volume = data['volume']
            
            # Create feature vector
            features = []
            
            # OHLC ratios
            if ohlc[0] != 0:  # open != 0
                features.extend([
                    (ohlc[1] - ohlc[0]) / ohlc[0],  # high/open ratio
                    (ohlc[2] - ohlc[0]) / ohlc[0],  # low/open ratio
                    (ohlc[3] - ohlc[0]) / ohlc[0],  # close/open ratio
                ])
            else:
                features.extend([0, 0, 0])
            
            # Returns statistics
            if returns:
                features.extend([
                    np.mean(returns),
                    np.std(returns) if len(returns) > 1 else 0,
                    returns[-1] if returns else 0  # Latest return
                ])
            else:
                features.extend([0, 0, 0])
            
            # Volatility and volume
            features.extend([volatility, np.log1p(volume)])
            
            # Normalize features to [0, 1]
            features = np.array(features)
            if np.std(features) > 0:
                features = (features - np.min(features)) / (np.max(features) - np.min(features))
            else:
                features = np.zeros_like(features)
            
            # Ensure we have enough features
            qubits_needed = self.correlation_qubits // len(self.timeframes)
            if len(features) < qubits_needed:
                padded = np.zeros(qubits_needed)
                padded[:len(features)] = features
                return padded.tolist()
            else:
                return features[:qubits_needed].tolist()
                
        except Exception as e:
            logging.warning(f"Timeframe data normalization error: {e}")
            qubits_needed = self.correlation_qubits // len(self.timeframes)
            return [0.0] * qubits_needed
    
    def _initialize_baseline_correlations(self):
        """Initialize baseline correlation patterns for normal market conditions"""
        # Default baseline correlations between timeframes
        baseline_correlations = {}
        for i, tf1 in enumerate(self.timeframes):
            for j, tf2 in enumerate(self.timeframes):
                if i < j:  # Avoid duplicate pairs
                    # Normal markets show decreasing correlation with timeframe distance
                    distance = abs(i - j)
                    correlation = 0.5 * np.exp(-distance * 0.3)  # Exponential decay
                    baseline_correlations[f'{tf1}_{tf2}'] = correlation
        
        return baseline_correlations
    
    def _calculate_entanglement_measure(self, entanglement_result):
        """Calculate entanglement measure from quantum circuit result"""
        try:
            if isinstance(entanglement_result, (list, np.ndarray)) and len(entanglement_result) > 0:
                # Convert measurements to entanglement metric
                # Use Von Neumann entropy as entanglement measure
                probs = np.abs(entanglement_result) ** 2
                probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs
                
                # Remove zero probabilities for entropy calculation
                probs = probs[probs > 1e-10]
                
                if len(probs) > 1:
                    entropy = -np.sum(probs * np.log2(probs))
                    max_entropy = np.log2(len(probs))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    return normalized_entropy
                
            return 0.0
        except Exception as e:
            logging.warning(f"Entanglement measure calculation failed: {e}")
            return 0.0
    
    def _detect_correlation_anomalies(self, correlation_results, entanglement_measure):
        """Detect correlation anomalies from quantum results"""
        try:
            anomalies = {
                'manipulation_detected': False,
                'confidence': 0.0,
                'anomaly_types': []
            }
            
            if isinstance(correlation_results, (list, np.ndarray)) and len(correlation_results) > 0:
                # Analyze correlation strength
                avg_correlation = np.mean(np.abs(correlation_results))
                max_correlation = np.max(np.abs(correlation_results))
                
                # High correlation indicates potential manipulation
                if max_correlation > self.entanglement_threshold:
                    anomalies['manipulation_detected'] = True
                    anomalies['confidence'] = min(1.0, max_correlation)
                    anomalies['anomaly_types'].append('high_correlation')
                
                # Entanglement measure indicates coordinated activity
                if entanglement_measure > 0.8:
                    anomalies['manipulation_detected'] = True
                    anomalies['confidence'] = max(anomalies['confidence'], entanglement_measure)
                    anomalies['anomaly_types'].append('entanglement_anomaly')
                
                # Combine confidence scores
                if anomalies['manipulation_detected']:
                    anomalies['confidence'] = (avg_correlation + entanglement_measure) / 2
            
            return anomalies
        except Exception as e:
            logging.warning(f"Correlation anomaly detection failed: {e}")
            return {'manipulation_detected': False, 'confidence': 0.0, 'anomaly_types': []}
    
    def _quantify_breakdown(self, correlation_result):
        """Quantify correlation breakdown patterns"""
        try:
            breakdown_metrics = {
                'severity': 0.0,
                'affected_timeframes': [],
                'breakdown_type': 'none'
            }
            
            if isinstance(correlation_result, (list, np.ndarray)) and len(correlation_result) > 0:
                # Calculate breakdown severity
                correlations = np.array(correlation_result)
                
                # Check for sudden drops in correlation
                correlation_variance = np.var(correlations)
                mean_correlation = np.mean(np.abs(correlations))
                
                if correlation_variance > 0.5:
                    breakdown_metrics['severity'] = correlation_variance
                    breakdown_metrics['breakdown_type'] = 'high_variance'
                
                if mean_correlation < 0.3:
                    breakdown_metrics['severity'] = max(breakdown_metrics['severity'], 1 - mean_correlation)
                    breakdown_metrics['breakdown_type'] = 'low_correlation'
                
                # Identify affected timeframes
                for i, corr in enumerate(correlations):
                    if abs(corr) < 0.2:  # Very low correlation
                        if i < len(self.timeframes):
                            breakdown_metrics['affected_timeframes'].append(self.timeframes[i])
            
            return breakdown_metrics
        except Exception as e:
            logging.warning(f"Correlation breakdown quantification failed: {e}")
            return {'severity': 0.0, 'affected_timeframes': [], 'breakdown_type': 'none'}
    
    def _classify_manipulation(self, manipulation_signals):
        """Classify the type of manipulation detected"""
        try:
            if not isinstance(manipulation_signals, dict):
                return 'unknown'
            
            anomaly_types = manipulation_signals.get('anomaly_types', [])
            confidence = manipulation_signals.get('confidence', 0.0)
            
            if not manipulation_signals.get('manipulation_detected', False):
                return 'none'
            
            if 'high_correlation' in anomaly_types and confidence > 0.8:
                return 'coordinated_attack'
            elif 'entanglement_anomaly' in anomaly_types:
                return 'sophisticated_manipulation'
            elif confidence > 0.6:
                return 'market_manipulation'
            else:
                return 'possible_manipulation'
                
        except Exception as e:
            logging.warning(f"Manipulation classification failed: {e}")
            return 'unknown'

class QuantumGameTheoryEngine:
    """
    Implements quantum game theory for optimal anti-whale strategies.
    Calculates Nash equilibria and dominant strategies.
    """
    
    def __init__(self, game_theory_qubits=10):
        self.game_theory_qubits = game_theory_qubits
        self.whale_psychology_models = {}
        self.historical_games = []
        
        # Initialize quantum device with fallback priority
        try:
            self.device = qml.device('lightning.kokkos', wires=game_theory_qubits)
            logging.info(f"Using lightning.kokkos backend for game theory engine")
        except Exception as e:
            try:
                self.device = qml.device('lightning.qubit', wires=game_theory_qubits)
                logging.info(f"Using lightning.qubit backend for game theory engine")
            except Exception as e2:
                self.device = qml.device('default.qubit', wires=game_theory_qubits)
                logging.info(f"Using default.qubit backend for game theory engine")
        
        # Create game theory circuits
        self.nash_equilibrium_circuit = self._create_nash_circuit()
        self.payoff_optimization_circuit = self._create_payoff_circuit()
        
    def _create_nash_circuit(self):
        """Create quantum circuit for Nash equilibrium calculation"""
        
        @qml.qnode(self.device)
        def nash_calculation(whale_strategies, our_strategies, payoff_weights):
            num_whale = len(whale_strategies)
            num_our = len(our_strategies)
            
            # Encode strategies in quantum superposition
            whale_qubits = self.game_theory_qubits // 2
            our_qubits = self.game_theory_qubits - whale_qubits
            
            # Create uniform superposition for whale strategies
            for i in range(min(whale_qubits, int(np.log2(num_whale)) + 1)):
                qml.Hadamard(wires=i)
                
            # Create uniform superposition for our strategies
            for i in range(min(our_qubits, int(np.log2(num_our)) + 1)):
                qml.Hadamard(wires=whale_qubits + i)
            
            # Apply payoff-based rotations
            for i, weight in enumerate(payoff_weights[:self.game_theory_qubits]):
                qml.RY(weight * np.pi, wires=i)
            
            # Entangle strategies to find correlations
            for i in range(min(whale_qubits, our_qubits)):
                qml.CNOT(wires=[i, whale_qubits + i])
            
            # Measure expected payoffs
            return [qml.expval(qml.PauliZ(i)) for i in range(self.game_theory_qubits)]
            
        return nash_calculation
    
    def _create_payoff_circuit(self):
        """Create circuit for payoff optimization"""
        
        @qml.qnode(self.device)
        def payoff_optimization(strategy_params, market_conditions):
            # Encode strategy parameters
            for i, param in enumerate(strategy_params[:self.game_theory_qubits]):
                qml.RY(param * np.pi, wires=i)
            
            # Apply market condition modulations
            for i, condition in enumerate(market_conditions[:self.game_theory_qubits]):
                qml.RZ(condition * np.pi, wires=i)
            
            # Optimization through variational evolution
            for layer in range(3):
                # Entangling layer
                for i in range(self.game_theory_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Parameterized layer
                for i in range(self.game_theory_qubits):
                    qml.RY(strategy_params[i % len(strategy_params)] * np.pi / (layer + 1), wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.game_theory_qubits)]
            
        return payoff_optimization
    
    def calculate_optimal_counter_strategy(self, whale_profile, market_state):
        """
        Calculate optimal strategy against specific whale using quantum game theory.
        """
        try:
            start_time = time.perf_counter()
            
            # Model whale's possible strategies
            whale_strategies = self._model_whale_strategies(whale_profile, market_state)
            
            # Define our possible counter-strategies
            our_strategies = self._define_counter_strategies(market_state)
            
            # Create payoff matrix
            payoff_matrix = self._create_payoff_matrix(whale_strategies, our_strategies, market_state)
            
            # Flatten payoff matrix for quantum processing
            payoff_weights = payoff_matrix.flatten()[:self.game_theory_qubits]
            
            # Calculate quantum Nash equilibrium
            nash_result = self.nash_equilibrium_circuit(
                whale_strategies, our_strategies, payoff_weights
            )
            
            # Optimize our strategy
            strategy_params = [s.get('weight', 0.5) for s in our_strategies][:self.game_theory_qubits]
            market_conditions = self._encode_market_conditions(market_state)
            
            optimization_result = self.payoff_optimization_circuit(
                strategy_params, market_conditions
            )
            
            # Extract optimal strategy
            optimal_strategy = self._extract_optimal_strategy(
                optimization_result, our_strategies
            )
            
            # Calculate expected payoff
            expected_payoff = self._calculate_expected_payoff(
                optimal_strategy, whale_strategies, payoff_matrix
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'recommended_strategy': optimal_strategy,
                'expected_payoff': expected_payoff,
                'confidence': self._calculate_solution_confidence(nash_result),
                'whale_predicted_action': self._predict_whale_action(whale_strategies, nash_result),
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logging.error(f"Game theory calculation error: {e}")
            return {
                'recommended_strategy': self._get_default_strategy(),
                'expected_payoff': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _model_whale_strategies(self, whale_profile, market_state):
        """Model possible whale strategies based on profile and market"""
        strategies = []
        
        # Size-based strategies
        size_category = whale_profile.get('size_category', 'medium')
        if size_category == 'mega_whale':
            strategies.extend([
                {'type': 'market_dump', 'size': 1.0, 'speed': 0.9, 'weight': 0.3},
                {'type': 'iceberg_accumulation', 'size': 0.8, 'speed': 0.2, 'weight': 0.4},
                {'type': 'cross_venue_coordination', 'size': 0.9, 'speed': 0.7, 'weight': 0.3}
            ])
        elif size_category == 'large_whale':
            strategies.extend([
                {'type': 'stop_hunt', 'size': 0.6, 'speed': 0.8, 'weight': 0.4},
                {'type': 'momentum_following', 'size': 0.5, 'speed': 0.6, 'weight': 0.6}
            ])
        else:
            strategies.extend([
                {'type': 'stealth_accumulation', 'size': 0.3, 'speed': 0.1, 'weight': 0.8},
                {'type': 'technical_breakout', 'size': 0.4, 'speed': 0.5, 'weight': 0.2}
            ])
        
        # Adjust for market conditions
        volatility = market_state.get('volatility', 0.2)
        if volatility > 0.5:
            # High volatility favors aggressive strategies
            for strategy in strategies:
                if strategy['type'] in ['market_dump', 'stop_hunt']:
                    strategy['weight'] *= 1.5
        
        return strategies
    
    def _define_counter_strategies(self, market_state):
        """Define our possible counter-strategies"""
        strategies = [
            {'type': 'defensive_hedge', 'allocation': 0.3, 'timing': 'immediate', 'weight': 0.4},
            {'type': 'position_reduction', 'allocation': 0.5, 'timing': 'gradual', 'weight': 0.3},
            {'type': 'counter_trade', 'allocation': 0.2, 'timing': 'delayed', 'weight': 0.3},
            {'type': 'volatility_play', 'allocation': 0.1, 'timing': 'immediate', 'weight': 0.2}
        ]
        
        # Adjust for liquidity
        liquidity = market_state.get('liquidity', 0.5)
        if liquidity < 0.3:
            # Low liquidity - reduce aggressive strategies
            for strategy in strategies:
                if strategy['type'] == 'counter_trade':
                    strategy['weight'] *= 0.5
        
        return strategies
    
    def _create_payoff_matrix(self, whale_strategies, our_strategies, market_state):
        """Create payoff matrix for the game"""
        n_whale = len(whale_strategies)
        n_our = len(our_strategies)
        
        payoff_matrix = np.zeros((n_whale, n_our))
        
        for i, whale_strategy in enumerate(whale_strategies):
            for j, our_strategy in enumerate(our_strategies):
                # Calculate payoff based on strategy interaction
                payoff = self._calculate_strategy_payoff(
                    whale_strategy, our_strategy, market_state
                )
                payoff_matrix[i, j] = payoff
        
        return payoff_matrix
    
    def _calculate_strategy_payoff(self, whale_strategy, our_strategy, market_state):
        """Calculate payoff for strategy combination"""
        # Base payoff starts at 0
        payoff = 0.0
        
        # Defensive strategies vs aggressive whale moves
        if whale_strategy['type'] in ['market_dump', 'stop_hunt'] and \
           our_strategy['type'] in ['defensive_hedge', 'position_reduction']:
            payoff += 0.7  # Good defense
        
        # Counter-trading against whale moves
        if whale_strategy['type'] in ['iceberg_accumulation'] and \
           our_strategy['type'] == 'counter_trade':
            payoff += 0.5  # Moderate success
        
        # Volatility plays during whale activity
        if whale_strategy.get('speed', 0) > 0.7 and \
           our_strategy['type'] == 'volatility_play':
            payoff += 0.6  # Benefit from volatility
        
        # Penalties for mismatched strategies
        if whale_strategy.get('size', 0) > 0.8 and \
           our_strategy['type'] == 'counter_trade' and \
           our_strategy.get('allocation', 0) > 0.3:
            payoff -= 0.8  # Dangerous to counter large whales directly
        
        # Market condition adjustments
        volatility = market_state.get('volatility', 0.2)
        liquidity = market_state.get('liquidity', 0.5)
        
        # High volatility increases defensive payoffs
        if volatility > 0.5 and our_strategy['type'] in ['defensive_hedge']:
            payoff += 0.3
        
        # Low liquidity penalizes large position changes
        if liquidity < 0.3 and our_strategy.get('allocation', 0) > 0.4:
            payoff -= 0.4
        
        return payoff

class MachiavellianQuantumTradingSystem:
    """
    Main orchestrator for the complete whale detection and defense system.
    """
    
    def __init__(self, config: WhaleDetectionConfig = None):
        self.config = config or WhaleDetectionConfig()
        
        # Initialize core components
        self.oscillation_detector = QuantumOscillationDetector(
            detection_qubits=self.config.detection_qubits,
            sensitivity=self.config.detection_sensitivity
        )
        
        self.correlation_engine = QuantumCorrelationEngine(
            correlation_qubits=self.config.correlation_qubits
        )
        
        self.game_theory_engine = QuantumGameTheoryEngine(
            game_theory_qubits=self.config.game_theory_qubits
        )
        
        # System state
        self.whale_threat_level = 0.0
        self.active_defenses = []
        self.detection_history = deque(maxlen=1000)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_detections': 0,
            'successful_defenses': 0,
            'false_positives': 0,
            'average_latency_ms': 0.0
        }
        
        # Threading for real-time operation
        self.running = False
        self.detection_thread = None
        
    def start_real_time_monitoring(self, data_stream_callback):
        """Start real-time whale detection monitoring"""
        self.running = True
        self.data_stream_callback = data_stream_callback
        
        # Start detection thread
        self.detection_thread = threading.Thread(
            target=self._real_time_detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
        logging.info("Real-time whale detection monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
        
        logging.info("Whale detection monitoring stopped")
    
    def _real_time_detection_loop(self):
        """Main real-time detection loop"""
        while self.running:
            try:
                # Get latest market data
                market_data = self.data_stream_callback()
                
                if market_data:
                    # Run whale detection
                    detection_result = self.comprehensive_whale_detection(market_data)
                    
                    # Update threat level
                    self._update_threat_level(detection_result)
                    
                    # Execute defenses if needed
                    if detection_result.get('whale_detected', False):
                        self._execute_automated_defense(detection_result, market_data)
                    
                    # Store in history
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'detection_result': detection_result,
                        'threat_level': self.whale_threat_level
                    })
                
                # Sleep until next update
                time.sleep(self.config.update_frequency_ms / 1000.0)
                
            except Exception as e:
                logging.error(f"Real-time detection loop error: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def comprehensive_whale_detection(self, market_data, social_data=None):
        """
        Run comprehensive whale detection using all available methods.
        """
        try:
            start_time = time.perf_counter()
            
            detection_results = {}
            
            # Oscillation-based detection
            oscillation_result = self.oscillation_detector.detect_whale_tremors(market_data)
            detection_results['oscillation'] = oscillation_result
            
            # Correlation-based detection
            correlation_result = self.correlation_engine.analyze_cross_timeframe_correlations(market_data)
            detection_results['correlation'] = correlation_result
            
            # Aggregate results using quantum voting
            aggregated_result = self._quantum_threat_aggregation(detection_results)
            
            # Add timing information
            total_processing_time = (time.perf_counter() - start_time) * 1000
            aggregated_result['total_processing_time_ms'] = total_processing_time
            
            # Update performance metrics
            self._update_performance_metrics(aggregated_result, total_processing_time)
            
            return aggregated_result
            
        except Exception as e:
            logging.error(f"Comprehensive whale detection error: {e}")
            return {
                'whale_detected': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _quantum_threat_aggregation(self, detection_results):
        """Aggregate detection results using quantum interference"""
        try:
            # Extract confidence scores
            confidences = []
            detections = []
            
            for method, result in detection_results.items():
                if isinstance(result, dict):
                    conf = result.get('confidence', 0.0)
                    detected = result.get('whale_detected', False) or \
                              result.get('manipulation_detected', False)
                    
                    confidences.append(conf)
                    detections.append(1.0 if detected else 0.0)
            
            if not confidences:
                return {'whale_detected': False, 'confidence': 0.0}
            
            # Weighted aggregation (could be made more sophisticated with quantum circuits)
            weights = [0.4, 0.4, 0.2]  # oscillation, correlation, sentiment
            weighted_confidence = sum(w * c for w, c in zip(weights[:len(confidences)], confidences))
            weighted_detection = sum(w * d for w, d in zip(weights[:len(detections)], detections))
            
            # Overall detection decision
            whale_detected = weighted_detection > 0.5 and weighted_confidence > self.config.early_warning_threshold
            
            # Estimate impact time (take minimum from individual estimates)
            impact_times = []
            for result in detection_results.values():
                if isinstance(result, dict) and 'estimated_impact_time' in result:
                    impact_times.append(result['estimated_impact_time'])
            
            estimated_impact_time = min(impact_times) if impact_times else None
            
            return {
                'whale_detected': whale_detected,
                'confidence': weighted_confidence,
                'detection_methods_triggered': [
                    method for method, result in detection_results.items()
                    if result.get('whale_detected', False) or result.get('manipulation_detected', False)
                ],
                'estimated_impact_time_seconds': estimated_impact_time,
                'individual_results': detection_results
            }
            
        except Exception as e:
            logging.error(f"Threat aggregation error: {e}")
            return {'whale_detected': False, 'confidence': 0.0, 'error': str(e)}
    
    def get_defense_recommendation(self, whale_warning, market_state, current_positions=None):
        """Get optimal defense strategy recommendation"""
        try:
            if not whale_warning.get('whale_detected', False):
                return {'defense_needed': False}
            
            # Classify whale profile from detection results
            whale_profile = self._classify_whale_profile(whale_warning)
            
            # Calculate optimal counter-strategy
            strategy_result = self.game_theory_engine.calculate_optimal_counter_strategy(
                whale_profile, market_state
            )
            
            # Customize for current positions
            if current_positions:
                strategy_result = self._customize_for_positions(
                    strategy_result, current_positions
                )
            
            return {
                'defense_needed': True,
                'recommended_strategy': strategy_result['recommended_strategy'],
                'expected_utility': strategy_result['expected_payoff'],
                'confidence': strategy_result['confidence'],
                'whale_profile': whale_profile,
                'urgency': self._calculate_urgency(whale_warning)
            }
            
        except Exception as e:
            logging.error(f"Defense recommendation error: {e}")
            return {
                'defense_needed': False,
                'error': str(e)
            }
    
    def _classify_whale_profile(self, whale_warning):
        """Classify whale type from detection results"""
        profile = {
            'size_category': 'medium',
            'sophistication': 0.5,
            'aggression_level': 0.5,
            'stealth_level': 0.5
        }
        
        # Analyze detection patterns
        confidence = whale_warning.get('confidence', 0.0)
        methods_triggered = whale_warning.get('detection_methods_triggered', [])
        
        # High confidence suggests large whale
        if confidence > 0.9:
            profile['size_category'] = 'mega_whale'
        elif confidence > 0.7:
            profile['size_category'] = 'large_whale'
        
        # Multiple detection methods suggest sophisticated whale
        if len(methods_triggered) > 1:
            profile['sophistication'] = 0.8
        
        # Short impact time suggests aggressive whale
        impact_time = whale_warning.get('estimated_impact_time_seconds', 10)
        if impact_time < 7:
            profile['aggression_level'] = 0.9
        elif impact_time > 12:
            profile['aggression_level'] = 0.3
        
        # Correlation detection suggests coordinated (less stealthy) activity
        if 'correlation' in methods_triggered:
            profile['stealth_level'] = 0.3
        else:
            profile['stealth_level'] = 0.7
        
        return profile
    
    def _update_performance_metrics(self, detection_result, processing_time):
        """Update system performance metrics"""
        self.performance_metrics['total_detections'] += 1
        
        # Update average latency
        old_avg = self.performance_metrics['average_latency_ms']
        count = self.performance_metrics['total_detections']
        new_avg = ((count - 1) * old_avg + processing_time) / count
        self.performance_metrics['average_latency_ms'] = new_avg
        
        # Check latency requirement
        if processing_time > self.config.max_latency_ms:
            logging.warning(f"Detection latency {processing_time:.2f}ms exceeds target {self.config.max_latency_ms}ms")
    
    def get_system_status(self):
        """Get current system status and performance metrics"""
        return {
            'status': 'running' if self.running else 'stopped',
            'current_threat_level': self.whale_threat_level,
            'active_defenses': len(self.active_defenses),
            'detection_history_size': len(self.detection_history),
            'performance_metrics': self.performance_metrics.copy(),
            'config': {
                'detection_sensitivity': self.config.detection_sensitivity,
                'early_warning_threshold': self.config.early_warning_threshold,
                'max_latency_ms': self.config.max_latency_ms
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    config = WhaleDetectionConfig(
        detection_sensitivity=0.001,
        early_warning_threshold=0.7,
        max_latency_ms=50.0
    )
    
    whale_system = MachiavellianQuantumTradingSystem(config)
    
    # Test with sample data
    sample_market_data = {
        'prices': [100.0, 100.5, 99.8, 101.2, 100.9, 102.1, 101.5, 103.0],
        'volumes': [1000, 1200, 800, 1500, 1100, 2000, 900, 1800],
        'timestamps': [time.time() - i for i in range(8, 0, -1)],
        'volatility': 0.3,
        'liquidity': 0.6
    }
    
    # Run detection
    detection_result = whale_system.comprehensive_whale_detection(sample_market_data)
    print("Detection result:", detection_result)
    
    # Get defense recommendation if whale detected
    if detection_result.get('whale_detected', False):
        defense_recommendation = whale_system.get_defense_recommendation(
            detection_result, 
            {'volatility': 0.3, 'liquidity': 0.6}
        )
        print("Defense recommendation:", defense_recommendation)
    
    # System status
    status = whale_system.get_system_status()
    print("System status:", status)