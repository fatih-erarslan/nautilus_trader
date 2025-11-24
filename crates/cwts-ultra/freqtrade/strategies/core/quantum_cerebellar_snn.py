"""
Quantum Cerebellar Spiking Neural Network (QC-SNN)
Neuromorphic quantum implementation with cerebellar microcircuit architecture
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import cupy as cp

@dataclass
class QuantumSNNConfig:
    """Configuration for Quantum Cerebellar SNN"""
    n_qubits: int = 24
    n_granule_cells: int = 100
    n_purkinje_cells: int = 10
    n_golgi_cells: int = 20
    n_mossy_fibers: int = 50
    backend: str = 'lightning.gpu'
    spike_threshold: float = 0.5
    refractory_period: int = 3
    time_window: int = 10
    learning_rate: float = 0.01
    
@dataclass
class SpikeTrain:
    """Quantum-enhanced spike train representation"""
    times: List[float] = field(default_factory=list)
    amplitudes: List[float] = field(default_factory=list)
    phases: List[float] = field(default_factory=list)  # Quantum phase information
    
class QuantumSpikeEncoder:
    """
    Encodes classical signals into quantum spike trains.
    Uses quantum superposition for efficient spike representation.
    """
    
    def __init__(self, config: QuantumSNNConfig):
        self.config = config
        self.dev = qml.device(
            config.backend,
            wires=config.n_qubits
        )
        
        # Build encoding circuits
        self.amplitude_encoder = self._build_amplitude_encoder()
        self.phase_encoder = self._build_phase_encoder()
        self.frequency_encoder = self._build_frequency_encoder()
        
    def _build_amplitude_encoder(self):
        """Encode signal amplitude into quantum state"""
        @qml.qnode(self.dev, interface='torch')
        def amplitude_circuit(signal, threshold):
            n = min(len(signal), self.config.n_qubits)
            
            # Threshold encoding
            for i in range(n):
                if signal[i] > threshold:
                    # Spike representation
                    angle = 2 * np.arcsin(np.sqrt(min(signal[i], 1.0)))
                    qml.RY(angle, wires=i)
                    qml.PauliZ(wires=i)
                    
            # Entangle neighboring spikes
            for i in range(n-1):
                if signal[i] > threshold and signal[i+1] > threshold:
                    qml.CNOT(wires=[i, i+1])
                    
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]
            
        return amplitude_circuit
        
    def _build_phase_encoder(self):
        """Encode temporal phase information"""
        @qml.qnode(self.dev, interface='torch')
        def phase_circuit(spike_times, time_window):
            n = min(len(spike_times), self.config.n_qubits)
            
            # Phase encoding based on spike timing
            for i in range(n):
                phase = 2 * np.pi * (spike_times[i] % time_window) / time_window
                qml.PhaseShift(phase, wires=i)
                qml.RZ(phase, wires=i)
                
            # Quantum Fourier Transform for frequency analysis
            if n > 1:
                qml.templates.QFT(wires=range(n))
                
            return [qml.expval(qml.PauliX(i)) for i in range(n)]
            
        return phase_circuit
        
    def _build_frequency_encoder(self):
        """Encode spike frequency patterns"""
        @qml.qnode(self.dev, interface='torch')
        def frequency_circuit(spike_counts, max_count):
            n = min(len(spike_counts), self.config.n_qubits // 2)
            
            # Encode counts in amplitude
            normalized_counts = spike_counts[:n] / (max_count + 1)
            qml.AmplitudeEmbedding(
                normalized_counts,
                wires=range(n),
                normalize=True,
                pad_with=0.0
            )
            
            # Encode rate coding in phase
            for i in range(n):
                rate = spike_counts[i] / self.config.time_window
                qml.RZ(rate * np.pi, wires=i + n)
                
            # Correlate count and rate
            for i in range(n):
                qml.CRY(normalized_counts[i] * np.pi, wires=[i, i + n])
                
            return qml.probs(wires=range(2 * n))
            
        return frequency_circuit
        
    def encode(self, signal: torch.Tensor) -> SpikeTrain:
        """
        Convert continuous signal to quantum spike train
        
        Args:
            signal: Input signal tensor
            
        Returns:
            Quantum spike train with times, amplitudes, and phases
        """
        # Classical spike detection
        spikes = (signal > self.config.spike_threshold).float()
        spike_indices = torch.where(spikes)[0]
        
        if len(spike_indices) == 0:
            return SpikeTrain()
            
        # Quantum amplitude encoding
        amp_encoding = self.amplitude_encoder(
            signal[spike_indices],
            self.config.spike_threshold
        )
        
        # Quantum phase encoding
        phase_encoding = self.phase_encoder(
            spike_indices.float(),
            self.config.time_window
        )
        
        # Combine encodings
        spike_train = SpikeTrain(
            times=spike_indices.tolist(),
            amplitudes=[abs(a) for a in amp_encoding],
            phases=phase_encoding
        )
        
        return spike_train
        

class QuantumGranuleCell:
    """
    Quantum implementation of cerebellar granule cells.
    Performs sparse coding of mossy fiber inputs.
    """
    
    def __init__(self, n_inputs: int, config: QuantumSNNConfig):
        self.n_inputs = n_inputs
        self.config = config
        
        # Quantum device for granule cell
        self.dev = qml.device(
            config.backend,
            wires=min(n_inputs + 4, config.n_qubits)
        )
        
        # Build sparse coding circuit
        self.sparse_coder = self._build_sparse_coder()
        
        # Synaptic weights (quantum parameters)
        self.weights = np.random.randn(n_inputs, 3) * 0.1
        
    def _build_sparse_coder(self):
        """Quantum circuit for sparse coding"""
        @qml.qnode(self.dev, interface='torch')
        def sparse_circuit(inputs, weights):
            n = min(len(inputs), self.dev.num_wires - 2)
            
            # Input encoding
            for i in range(n):
                qml.RY(inputs[i] * weights[i, 0], wires=i)
                qml.RZ(inputs[i] * weights[i, 1], wires=i)
                
            # Competitive inhibition (sparse coding)
            ancilla1 = self.dev.num_wires - 2
            ancilla2 = self.dev.num_wires - 1
            
            # Winner-take-all mechanism
            for i in range(n):
                qml.Toffoli(wires=[i, ancilla1, ancilla2])
                qml.CNOT(wires=[ancilla2, ancilla1])
                
            # Sparse activation
            for i in range(n):
                qml.ControlledPhaseShift(weights[i, 2], wires=[ancilla1, i])
                
            # Measure sparse representation
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]
            
        return sparse_circuit
        
    def process(self, mossy_fiber_input: torch.Tensor) -> torch.Tensor:
        """Process mossy fiber inputs with quantum sparse coding"""
        sparse_output = self.sparse_coder(mossy_fiber_input, self.weights)
        return torch.tensor(sparse_output)
        

class QuantumPurkinjeCell(nn.Module):
    """
    Quantum implementation of cerebellar Purkinje cells.
    Integrates parallel fiber and climbing fiber inputs for motor learning.
    """
    
    def __init__(self, n_parallel_fibers: int, config: QuantumSNNConfig):
        super().__init__()
        self.n_parallel_fibers = n_parallel_fibers
        self.config = config
        
        # Quantum device
        self.dev = qml.device(
            config.backend,
            wires=config.n_qubits
        )
        
        # Quantum circuits
        self.integration_circuit = self._build_integration_circuit()
        self.plasticity_circuit = self._build_plasticity_circuit()
        
        # Synaptic weights
        self.pf_weights = nn.Parameter(torch.randn(n_parallel_fibers) * 0.1)
        self.cf_weight = nn.Parameter(torch.tensor(1.0))
        
        # Spike history for STDP
        self.spike_history = deque(maxlen=config.time_window)
        
    def _build_integration_circuit(self):
        """Integrate parallel and climbing fiber inputs"""
        @qml.qnode(self.dev, interface='torch', diff_method='adjoint')
        def integration(pf_inputs, cf_input, weights):
            n_pf = min(len(pf_inputs), self.config.n_qubits - 4)
            cf_qubit = self.config.n_qubits - 1
            
            # Encode parallel fiber inputs
            for i in range(n_pf):
                angle = pf_inputs[i] * weights[i]
                qml.RY(angle, wires=i)
                
            # Encode climbing fiber (teaching signal)
            qml.RX(cf_input * np.pi, wires=cf_qubit)
            
            # Parallel fiber integration
            for i in range(0, n_pf-1, 2):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(weights[i] * weights[i+1], wires=i+1)
                
            # Climbing fiber modulation
            for i in range(n_pf):
                qml.ControlledPhaseShift(
                    cf_input * weights[i],
                    wires=[cf_qubit, i]
                )
                
            # Global integration
            ancilla = self.config.n_qubits - 2
            for i in range(n_pf):
                qml.Toffoli(wires=[i, cf_qubit, ancilla])
                
            # Output measurement
            return qml.expval(qml.PauliZ(ancilla))
            
        return integration
        
    def _build_plasticity_circuit(self):
        """Quantum circuit for synaptic plasticity"""
        @qml.qnode(self.dev, interface='torch')
        def plasticity(pre_spikes, post_spike, time_diff):
            n = min(len(pre_spikes), self.config.n_qubits - 2)
            
            # Encode spike timing differences
            for i in range(n):
                # STDP window function
                angle = np.exp(-abs(time_diff[i]) / 20) * np.sign(time_diff[i])
                qml.RY(angle * np.pi, wires=i)
                
            # Post-synaptic spike encoding
            post_qubit = self.config.n_qubits - 1
            if post_spike > 0:
                qml.PauliX(wires=post_qubit)
                
            # Hebbian learning rule
            for i in range(n):
                if pre_spikes[i] > 0:
                    qml.CRY(0.1 * np.pi, wires=[post_qubit, i])
                    
            # Anti-Hebbian for LTD
            for i in range(n):
                if pre_spikes[i] == 0 and post_spike > 0:
                    qml.CRY(-0.05 * np.pi, wires=[post_qubit, i])
                    
            # Measure weight changes
            return [qml.expval(qml.PauliY(i)) for i in range(n)]
            
        return plasticity
        
    def forward(self, pf_input: torch.Tensor, cf_input: torch.Tensor) -> torch.Tensor:
        """
        Process inputs through Purkinje cell
        
        Args:
            pf_input: Parallel fiber input (from granule cells)
            cf_input: Climbing fiber input (error signal)
            
        Returns:
            Purkinje cell output
        """
        # Quantum integration
        output = self.integration_circuit(pf_input, cf_input, self.pf_weights)
        
        # Apply threshold
        spike = (output > self.config.spike_threshold).float()
        
        # Update spike history
        self.spike_history.append({
            'time': len(self.spike_history),
            'spike': spike.item(),
            'pf_input': pf_input.clone()
        })
        
        return spike * output
        
    def update_weights(self, error_signal: torch.Tensor):
        """Update weights based on climbing fiber error signal"""
        if len(self.spike_history) < 2:
            return
            
        # Get recent spike history
        recent = list(self.spike_history)[-2:]
        pre_spikes = recent[0]['pf_input']
        post_spike = recent[1]['spike']
        time_diff = torch.ones_like(pre_spikes)  # Simplified
        
        # Quantum plasticity computation
        weight_changes = self.plasticity_circuit(
            pre_spikes,
            post_spike,
            time_diff
        )
        
        # Apply weight updates
        with torch.no_grad():
            self.pf_weights += self.config.learning_rate * torch.tensor(weight_changes)
            self.pf_weights.clamp_(-2, 2)  # Bound weights
            

class QuantumCerebellarSNN(nn.Module):
    """
    Complete Quantum Cerebellar Spiking Neural Network.
    Implements full cerebellar microcircuit for adaptive motor control.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 config: QuantumSNNConfig):
        super().__init__()
        self.config = config
        
        # Spike encoder
        self.spike_encoder = QuantumSpikeEncoder(config)
        
        # Mossy fibers (input layer)
        self.mossy_fibers = nn.Linear(input_dim, config.n_mossy_fibers)
        
        # Granule cells (sparse coding)
        self.granule_cells = nn.ModuleList([
            QuantumGranuleCell(config.n_mossy_fibers // 10, config)
            for _ in range(config.n_granule_cells)
        ])
        
        # Golgi cells (inhibitory interneurons)
        self.golgi_cells = self._build_golgi_cells()
        
        # Purkinje cells (output layer)
        self.purkinje_cells = nn.ModuleList([
            QuantumPurkinjeCell(config.n_granule_cells, config)
            for _ in range(config.n_purkinje_cells)
        ])
        
        # Deep cerebellar nuclei (final output)
        self.dcn = nn.Linear(config.n_purkinje_cells, output_dim)
        
        # Inferior olive (error computation)
        self.inferior_olive = self._build_inferior_olive()
        
        # Quantum reservoir for temporal dynamics
        self.quantum_reservoir = self._build_quantum_reservoir()
        
    def _build_golgi_cells(self):
        """Build Golgi cell network for feedback inhibition"""
        dev = qml.device(self.config.backend, wires=8)
        
        @qml.qnode(dev, interface='torch')
        def golgi_circuit(granule_input, mossy_input):
            n = min(4, len(granule_input))
            
            # Feedback from granule cells
            for i in range(n):
                qml.RY(granule_input[i], wires=i)
                
            # Feedforward from mossy fibers
            for i in range(n):
                qml.RY(mossy_input[i], wires=i+4)
                
            # Inhibitory connections
            for i in range(n):
                qml.CNOT(wires=[i, i+4])
                qml.CRZ(-0.5, wires=[i+4, i])  # Inhibition
                
            return [qml.expval(qml.PauliZ(i+4)) for i in range(n)]
            
        return golgi_circuit
        
    def _build_inferior_olive(self):
        """Build inferior olive for error signal generation"""
        dev = qml.device(self.config.backend, wires=12)
        
        @qml.qnode(dev, interface='torch')
        def io_circuit(predicted, actual, history):
            n = min(4, len(predicted))
            
            # Encode prediction error
            for i in range(n):
                error = actual[i] - predicted[i]
                qml.RY(error * np.pi, wires=i)
                
            # Encode error history (temporal difference)
            for i in range(n):
                if i < len(history):
                    qml.RZ(history[i] * np.pi, wires=i+4)
                    
            # Complex spike generation
            for i in range(n):
                # Error threshold
                qml.Toffoli(wires=[i, i+4, i+8])
                
            # Oscillatory dynamics
            for i in range(n):
                qml.RX(0.1 * i, wires=i+8)
                
            return [qml.expval(qml.PauliY(i+8)) for i in range(n)]
            
        return io_circuit
        
    def _build_quantum_reservoir(self):
        """Quantum reservoir for capturing temporal dynamics"""
        dev = qml.device(self.config.backend, wires=16)
        
        @qml.qnode(dev, interface='torch')
        def reservoir_circuit(input_spikes, state):
            n = min(8, len(input_spikes))
            
            # Input injection
            for i in range(n):
                qml.RY(input_spikes[i], wires=i)
                
            # Reservoir state
            for i in range(n):
                qml.RY(state[i], wires=i+8)
                
            # Random unitary evolution
            qml.RandomLayers(
                weights=np.random.randn(2, n, 3),
                wires=range(16)
            )
            
            # Controlled dynamics
            for i in range(8):
                qml.CNOT(wires=[i, i+8])
                qml.CRX(0.1, wires=[i+8, (i+1)%8])
                
            # Extract features
            features = []
            for i in range(16):
                features.append(qml.expval(qml.PauliZ(i)))
                
            return features
            
        return reservoir_circuit
        
    def forward(self, x: torch.Tensor, error_history: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through quantum cerebellar network
        
        Args:
            x: Input tensor [batch_size, input_dim]
            error_history: Previous error signals for learning
            
        Returns:
            Dictionary with output and internal states
        """
        batch_size = x.shape[0]
        outputs = []
        
        for b in range(batch_size):
            # Mossy fiber activation
            mf_activity = torch.relu(self.mossy_fibers(x[b]))
            
            # Encode to spikes
            spike_train = self.spike_encoder.encode(mf_activity)
            
            # Granule cell processing
            gc_outputs = []
            for gc in self.granule_cells:
                # Random subset of mossy fibers (sparse connectivity)
                mf_subset = mf_activity[torch.randperm(len(mf_activity))[:10]]
                gc_out = gc.process(mf_subset)
                gc_outputs.append(gc_out)
                
            gc_activity = torch.stack(gc_outputs)
            
            # Golgi cell inhibition
            golgi_inhibition = self.golgi_cells(
                gc_activity[:4].mean(0),
                mf_activity[:4]
            )
            
            # Apply inhibition to granule cells
            gc_activity = gc_activity * (1 - 0.3 * torch.tensor(golgi_inhibition).unsqueeze(1))
            
            # Purkinje cell processing
            pc_outputs = []
            cf_signals = torch.zeros(len(self.purkinje_cells))
            
            # Generate climbing fiber signals if error history provided
            if error_history is not None:
                cf_signals = self.inferior_olive(
                    x[b, :4],
                    error_history[b, :4],
                    error_history[b, 4:8] if error_history.shape[1] > 4 else torch.zeros(4)
                )
                cf_signals = torch.tensor(cf_signals)
                
            for i, pc in enumerate(self.purkinje_cells):
                pc_out = pc(
                    gc_activity.flatten(),
                    cf_signals[i] if i < len(cf_signals) else 0.0
                )
                pc_outputs.append(pc_out)
                
                # Update weights if error signal present
                if error_history is not None and i < len(cf_signals):
                    pc.update_weights(cf_signals[i])
                    
            pc_activity = torch.stack(pc_outputs)
            
            # Deep cerebellar nuclei output
            dcn_input = -pc_activity  # Purkinje cells are inhibitory
            output = self.dcn(dcn_input)
            
            outputs.append(output)
            
        final_output = torch.stack(outputs)
        
        # Quantum reservoir state
        reservoir_state = self.quantum_reservoir(
            spike_train.amplitudes[:16] if spike_train.amplitudes else torch.zeros(16),
            torch.randn(16) * 0.1
        )
        
        return {
            'output': final_output,
            'granule_activity': gc_activity,
            'purkinje_activity': pc_activity,
            'reservoir_state': torch.tensor(reservoir_state),
            'spike_trains': spike_train
        }
        
    def adapt_online(self, error_signal: torch.Tensor):
        """
        Online adaptation based on error signals
        Implements cerebellar learning for motor control
        """
        # Update all Purkinje cells based on error
        for i, pc in enumerate(self.purkinje_cells):
            if i < error_signal.shape[0]:
                pc.update_weights(error_signal[i])
                
    def reset_state(self):
        """Reset internal states for new sequence"""
        for pc in self.purkinje_cells:
            pc.spike_history.clear()