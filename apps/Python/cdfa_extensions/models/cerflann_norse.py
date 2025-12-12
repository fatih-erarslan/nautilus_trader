import numpy as np
import pandas as pd
import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Callable, Any, Union, Literal, TypeVar
from enum import Enum
from dataclasses import dataclass
from functools import partial

# --- PyTorch & Norse Imports ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import norse.torch as norse
    from norse.torch.functional.lif import LIFParameters
    from norse.torch.functional.leaky_integrator import LIParameters
    from norse.torch.module.lif import LIFCell, LIFRecurrent
    from norse.torch.module.leaky_integrator import LICell, LILinearCell
    from norse.torch.module.encode import ConstantCurrentLIFEncoder
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    torch = nn = F = norse = None

# Configure logging
logger = logging.getLogger("advanced_ml.cerflann_norse")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Neuron Type Enum ---
class NeuronType(Enum):
    LIF = "LIF"  # Leaky Integrate-and-Fire
    ADEX = "AdEx"  # Adaptive Exponential


@dataclass
class LayerConfig:
    """Configuration for a cerebellar layer"""
    size: int
    neuron_type: NeuronType
    tau_mem: float  # Membrane time constant (ms)
    tau_syn_exc: float  # Excitatory synaptic time constant (ms)
    tau_syn_inh: float  # Inhibitory synaptic time constant (ms)
    
    # AdEx specific parameters (ignored for LIF)
    tau_adapt: Optional[float] = None  # Adaptation time constant (ms)
    a: Optional[float] = None  # Subthreshold adaptation
    b: Optional[float] = None  # Spike-triggered adaptation


class CerebellarLayer(nn.Module):
    """
    Norse implementation of a cerebellar layer with configurable neuron types
    """
    def __init__(self, config: LayerConfig, dt: float = 0.001):
        super().__init__()
        self.size = config.size
        self.neuron_type = config.neuron_type
        self.dt = dt
        
        # Convert time constants from ms to seconds
        tau_mem = config.tau_mem * 1e-3
        tau_syn_exc = config.tau_syn_exc * 1e-3
        tau_syn_inh = config.tau_syn_inh * 1e-3
        
        # Create parameters based on neuron type
        if config.neuron_type == NeuronType.LIF:
            # LIF parameters (simpler)
            self.p = LIFParameters(
                tau_mem_inv=1.0/tau_mem,
                tau_syn_inv=1.0/tau_syn_exc,
                v_leak=0.0,
                v_th=1.0,
                v_reset=0.0,
                method="super",  # Use super (or fast) for surrogate gradient
                alpha=100.0  # Surrogate gradient steepness
            )
            # Create LIF cell
            self.cell = LIFCell(p=self.p, dt=dt)
        else:
            # AdEx parameters (more complex, closer to biological neurons)
            # Note: Norse doesn't directly implement AdEx as of my knowledge cutoff
            # We'll use augmented LIF as an approximation
            logger.warning("AdEx not directly available in Norse. Using augmented LIF instead.")
            self.p = LIFParameters(
                tau_mem_inv=1.0/tau_mem,
                tau_syn_inv=1.0/tau_syn_exc,
                v_leak=0.0,
                v_th=1.0,
                v_reset=0.0,
                method="super",
                alpha=100.0
            )
            # Create LIF cell with adaptation simulation
            self.cell = LIFCell(p=self.p, dt=dt)
            
            # Adaptation parameters (manually handled during forward pass)
            if config.tau_adapt is not None:
                self.tau_adapt = config.tau_adapt * 1e-3
            else:
                self.tau_adapt = 100e-3  # Default 100ms
                
            if config.a is not None:
                self.a = config.a  
            else:
                self.a = 0.0  # Default no subthreshold adaptation
                
            if config.b is not None:
                self.b = config.b
            else:
                self.b = 0.0  # Default no spike-triggered adaptation
                
            self.has_adaptation = True
        
        # Initialize state
        self.reset_state(batch_size=1)
    
    def reset_state(self, batch_size=1):
        """Reset the internal state of the neurons"""
        # Initialize membrane potential, synaptic current, and spikes
        self.state = None
        if hasattr(self, 'has_adaptation') and self.has_adaptation:
            # Reset adaptation variable
            self.w = torch.zeros(batch_size, self.size, device=self.device if hasattr(self, 'device') else None)
    
    def forward(self, input_current):
        """
        Forward pass through the neuronal layer
        
        Args:
            input_current: Input current tensor of shape [batch_size, neuron_size]
            
        Returns:
            (spikes, state): Tuple of output spikes and new state
        """
        batch_size = input_current.shape[0]
        
        # Initialize state if needed
        if self.state is None:
            self.reset_state(batch_size)
        
        # Apply adaptation if using AdEx approximation
        if hasattr(self, 'has_adaptation') and self.has_adaptation:
            # Apply subthreshold adaptation current
            if self.a > 0:
                # Subtract adaptation current (w) from input
                input_current = input_current - self.w
        
        # Run neuron dynamics
        spikes, new_state = self.cell(input_current, self.state)
        self.state = new_state
        
        # Update adaptation variable for AdEx approximation
        if hasattr(self, 'has_adaptation') and self.has_adaptation:
            # Subthreshold adaptation (a) increases w based on voltage
            # Spike-triggered adaptation (b) increases w when spike occurs
            if self.a > 0 or self.b > 0:
                # Extract membrane potential from state
                v_mem = self.state[0] if isinstance(self.state, tuple) else self.state
                
                # Update adaptation variable
                # w' = w - dt * w / tau_adapt + a * (v_mem - v_leak) + b * spikes
                self.w = (self.w - self.dt * self.w / self.tau_adapt + 
                         self.a * v_mem + self.b * spikes)
        
        return spikes, new_state


class CerebellarCircuit(nn.Module):
    """
    Complete cerebellar microcircuit with Norse neurons
    Implements the full cerebellar architecture:
    - Mossy Fibers (input) -> Granule Cells -> Purkinje Cells -> Deep Cerebellar Nuclei (output)
    - Mossy Fibers -> Golgi Cells (inhibitory) -> Granule Cells
    - Mossy Fibers -> Deep Cerebellar Nuclei (direct pathway)
    """
    def __init__(self,
                input_dim: int,
                n_granule: int,
                n_purkinje: int,
                n_golgi: int,
                n_dcn: int,
                granule_config: LayerConfig,
                purkinje_config: LayerConfig,
                golgi_config: LayerConfig,
                dcn_config: LayerConfig,
                dt: float = 0.001):
        super().__init__()
        
        self.input_dim = input_dim  # Number of mossy fibers
        self.n_granule = n_granule
        self.n_purkinje = n_purkinje
        self.n_golgi = n_golgi
        self.n_dcn = n_dcn
        self.dt = dt
        
        # Create neuron layers
        self.granule_layer = CerebellarLayer(granule_config, dt=dt)
        self.purkinje_layer = CerebellarLayer(purkinje_config, dt=dt)
        self.golgi_layer = CerebellarLayer(golgi_config, dt=dt)
        self.dcn_layer = CerebellarLayer(dcn_config, dt=dt)
        
        # Create connection weights (using Linear layers for the connections)
        # MF -> GrC connection (excitatory)
        self.conn_mf_grc = nn.Linear(input_dim, n_granule, bias=False)
        
        # MF -> GoC connection (excitatory)
        self.conn_mf_goc = nn.Linear(input_dim, n_golgi, bias=False)
        
        # GoC -> GrC connection (inhibitory)
        self.conn_goc_grc = nn.Linear(n_golgi, n_granule, bias=False)
        
        # GrC -> PC connection (excitatory, learning site)
        self.conn_grc_pc = nn.Linear(n_granule, n_purkinje, bias=False)
        
        # PC -> DCN connection (inhibitory)
        self.conn_pc_dcn = nn.Linear(n_purkinje, n_dcn, bias=False)
        
        # MF -> DCN connection (excitatory, direct pathway)
        self.conn_mf_dcn = nn.Linear(input_dim, n_dcn, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize connection weights with appropriate patterns"""
        # Connection parameters (weight scales and connection probabilities)
        conn_params = {
            'mf_grc': {'w_scale': 0.2, 'p_connect': 0.05, 'is_inhibitory': False},
            'mf_goc': {'w_scale': 0.15, 'p_connect': 0.2, 'is_inhibitory': False},
            'goc_grc': {'w_scale': 0.15, 'p_connect': 0.1, 'is_inhibitory': True},
            'grc_pc': {'w_scale': 0.08, 'p_connect': 0.15, 'is_inhibitory': False},
            'pc_dcn': {'w_scale': 0.25, 'p_connect': 0.8, 'is_inhibitory': True},
            'mf_dcn': {'w_scale': 0.2, 'p_connect': 0.4, 'is_inhibitory': False}
        }
        
        # Helper function to create sparse weight matrices
        def create_sparse_weights(layer, pre_size, post_size, w_scale, p_connect, is_inhibitory):
            # Create mask for sparse connectivity
            mask = torch.rand(post_size, pre_size) < p_connect
            
            # Create weights with normal distribution
            weights = torch.randn(post_size, pre_size) * float(w_scale)
            
            # Make inhibitory if needed
            if is_inhibitory:
                weights = -weights
            
            # Apply mask to weights
            sparse_weights = weights * mask
            
            # Set as layer weights
            with torch.no_grad():
                layer.weight.copy_(sparse_weights)
        
        # Initialize all connection weights
        create_sparse_weights(
            self.conn_mf_grc, self.input_dim, self.n_granule,
            **conn_params['mf_grc']
        )
        
        create_sparse_weights(
            self.conn_mf_goc, self.input_dim, self.n_golgi,
            **conn_params['mf_goc']
        )
        
        create_sparse_weights(
            self.conn_goc_grc, self.n_golgi, self.n_granule,
            **conn_params['goc_grc']
        )
        
        create_sparse_weights(
            self.conn_grc_pc, self.n_granule, self.n_purkinje,
            **conn_params['grc_pc']
        )
        
        create_sparse_weights(
            self.conn_pc_dcn, self.n_purkinje, self.n_dcn,
            **conn_params['pc_dcn']
        )
        
        create_sparse_weights(
            self.conn_mf_dcn, self.input_dim, self.n_dcn,
            **conn_params['mf_dcn']
        )
    
    def reset_state(self):
        """Reset all layer states"""
        self.granule_layer.reset_state()
        self.purkinje_layer.reset_state()
        self.golgi_layer.reset_state()
        self.dcn_layer.reset_state()
    
    def forward(self, x):
        """
        Forward pass through the cerebellar circuit
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Dict containing outputs from all layers and connections
        """
        # MF -> GrC pathway (excitatory)
        mf_grc_current = self.conn_mf_grc(x)
        
        # MF -> GoC pathway (excitatory)
        mf_goc_current = self.conn_mf_goc(x)
        
        # MF -> DCN direct pathway (excitatory)
        mf_dcn_current = self.conn_mf_dcn(x)
        
        # GoC neuron dynamics
        goc_spikes, _ = self.golgi_layer(mf_goc_current)
        
        # GoC -> GrC pathway (inhibitory)
        goc_grc_current = self.conn_goc_grc(goc_spikes)
        
        # GrC neuron dynamics (combining excitatory and inhibitory inputs)
        # Note: in biological systems, inhibition is subtractive
        granule_input = mf_grc_current - goc_grc_current
        grc_spikes, _ = self.granule_layer(granule_input)
        
        # GrC -> PC pathway (excitatory)
        grc_pc_current = self.conn_grc_pc(grc_spikes)
        
        # PC neuron dynamics
        pc_spikes, _ = self.purkinje_layer(grc_pc_current)
        
        # PC -> DCN pathway (inhibitory)
        pc_dcn_current = self.conn_pc_dcn(pc_spikes)
        
        # DCN neuron dynamics (combining excitatory and inhibitory inputs)
        # DCN is excited by MF and inhibited by PC
        dcn_input = mf_dcn_current - pc_dcn_current
        dcn_spikes, _ = self.dcn_layer(dcn_input)
        
        # Return all spikes and currents for analysis
        return {
            "mf_grc_current": mf_grc_current,
            "mf_goc_current": mf_goc_current,
            "mf_dcn_current": mf_dcn_current,
            "goc_spikes": goc_spikes,
            "goc_grc_current": goc_grc_current,
            "grc_spikes": grc_spikes,
            "grc_pc_current": grc_pc_current,
            "pc_spikes": pc_spikes,
            "pc_dcn_current": pc_dcn_current,
            "dcn_spikes": dcn_spikes
        }


class CERFLANN_Norse:
    """
    Cerebellum-inspired Spiking Neural Network using Norse (PyTorch-based) computation.
    
    Features:
    - Hybrid neuron models (LIF for speed, AdEx approximation for biological plausibility)
    - PyTorch/Norse-accelerated computation
    - Flexible configuration of cerebellar microcircuit
    - Biologically-inspired cerebellar architecture
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 1,
                 n_granule: int = 1000,
                 n_purkinje: int = 100,
                 n_golgi: int = 20,
                 n_dcn: int = 10,
                 time_steps: int = 50,
                 dt: float = 1e-3,
                 use_adex: Union[bool, List[str]] = ['purkinje', 'dcn'],
                 seed: int = 42,
                 device: str = None):
        """
        Initialize CERFLANN with Norse implementation.
        
        Args:
            input_dim: Number of input features (Mossy Fibers)
            output_dim: Number of output dimensions
            n_granule: Number of Granule cells (typically large)
            n_purkinje: Number of Purkinje cells
            n_golgi: Number of Golgi cells (inhibitory interneurons)
            n_dcn: Number of Deep Cerebellar Nuclei cells
            time_steps: Number of simulation steps per sample
            dt: Simulation time step in seconds
            use_adex: If True, use AdEx for all layers. If False, use LIF for all.
                      If list, use AdEx only for specified layers ('granule', 'purkinje', 'golgi', 'dcn')
            seed: Random seed for reproducibility
            device: PyTorch device ('cpu', 'cuda', or None for auto-detection)
        """
        if not NORSE_AVAILABLE:
            logger.error("PyTorch and Norse are required for CERFLANN_Norse.")
            raise ImportError("Norse library not found. Install with: pip install norse")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_mf = input_dim
        self.n_grc = n_granule
        self.n_pc = n_purkinje
        self.n_goc = n_golgi
        self.n_dcn = min(n_dcn, output_dim) if n_dcn > 0 else output_dim
        self.time_steps = time_steps
        self.dt = dt
        self.seed = seed
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Determine which layers use AdEx vs LIF neurons
        if isinstance(use_adex, bool):
            self.use_adex = {
                'granule': use_adex,
                'purkinje': use_adex,
                'golgi': use_adex,
                'dcn': use_adex
            }
        else:
            self.use_adex = {
                'granule': 'granule' in use_adex,
                'purkinje': 'purkinje' in use_adex,
                'golgi': 'golgi' in use_adex,
                'dcn': 'dcn' in use_adex
            }
        
        # Initialize layer configurations
        self._create_layer_configs()
        
        # Initialize the network
        self._initialize_network()
        
        # Initialize encoder
        self._initialize_encoder()
        
        # Training status
        self._trained = False
        
        logger.info(
            f"CERFLANN Norse Initialized: InputDim={input_dim}, OutputDim={output_dim}, "
            f"GrC={n_granule}({self.layer_configs['granule'].neuron_type.name}), "
            f"PC={n_purkinje}({self.layer_configs['purkinje'].neuron_type.name}), "
            f"GoC={n_golgi}({self.layer_configs['golgi'].neuron_type.name}), "
            f"DCN={self.n_dcn}({self.layer_configs['dcn'].neuron_type.name})"
        )
    
    def _create_layer_configs(self):
        """Create configuration for each neural layer based on neuron type settings"""
        # Base configurations shared across neuron types
        base_configs = {
            'granule': LayerConfig(
                size=self.n_grc,
                neuron_type=NeuronType.ADEX if self.use_adex['granule'] else NeuronType.LIF,
                tau_mem=10.0,  # ms
                tau_syn_exc=2.0,  # ms
                tau_syn_inh=10.0,  # ms
                tau_adapt=50.0,  # ms
                a=2e-9,  # Subthreshold adaptation
                b=1e-10  # Spike-triggered adaptation
            ),
            'purkinje': LayerConfig(
                size=self.n_pc,
                neuron_type=NeuronType.ADEX if self.use_adex['purkinje'] else NeuronType.LIF,
                tau_mem=15.0,  # ms
                tau_syn_exc=3.0,  # ms
                tau_syn_inh=5.0,  # ms
                tau_adapt=100.0,  # ms
                a=4e-9,  # Subthreshold adaptation
                b=5e-10  # Spike-triggered adaptation
            ),
            'golgi': LayerConfig(
                size=self.n_goc,
                neuron_type=NeuronType.ADEX if self.use_adex['golgi'] else NeuronType.LIF,
                tau_mem=30.0,  # ms
                tau_syn_exc=5.0,  # ms
                tau_syn_inh=10.0,  # ms
                tau_adapt=200.0,  # ms
                a=2e-9,  # Subthreshold adaptation
                b=2e-10  # Spike-triggered adaptation
            ),
            'dcn': LayerConfig(
                size=self.n_dcn,
                neuron_type=NeuronType.ADEX if self.use_adex['dcn'] else NeuronType.LIF,
                tau_mem=25.0,  # ms
                tau_syn_exc=5.0,  # ms
                tau_syn_inh=10.0,  # ms
                tau_adapt=150.0,  # ms
                a=1e-9,  # Subthreshold adaptation
                b=5e-10  # Spike-triggered adaptation
            )
        }
        
        self.layer_configs = base_configs
    
    def _initialize_network(self):
        """Initialize the cerebellar network using Norse"""
        try:
            logger.info("Initializing CERFLANN layers and connections...")
            
            # Create the cerebellar circuit
            self.network = CerebellarCircuit(
                input_dim=self.input_dim,
                n_granule=self.n_grc,
                n_purkinje=self.n_pc,
                n_golgi=self.n_goc,
                n_dcn=self.n_dcn,
                granule_config=self.layer_configs['granule'],
                purkinje_config=self.layer_configs['purkinje'],
                golgi_config=self.layer_configs['golgi'],
                dcn_config=self.layer_configs['dcn'],
                dt=self.dt
            )
            
            # Move network to device
            self.network.to(self.device)
            
            self._initialized = True
            logger.info("CERFLANN Norse network initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CERFLANN Norse network: {e}", exc_info=True)
            self._initialized = False
            raise
    
    def _initialize_encoder(self):
        """Initialize input encoder"""
        # We'll use a simple constant current encoder for now
        # In a more advanced implementation, we could use more sophisticated encoding
        self.encoder = ConstantCurrentLIFEncoder(
            seq_length=self.time_steps,
            p=LIFParameters(
                tau_mem_inv=1.0/10e-3,  # 10ms membrane time constant
                tau_syn_inv=1.0/5e-3,   # 5ms synaptic time constant
                v_leak=0.0,
                v_th=1.0,
                v_reset=0.0
            ),
            dt=self.dt
        )
    
    def _encode_input(self, X: np.ndarray) -> torch.Tensor:
        """
        Encode input data as currents for the network.
        
        Args:
            X: Input data. Can be:
               - Shape (n_samples, time_steps, input_dim): Temporal data
               - Shape (n_samples, input_dim): Static data (will be expanded)
               
        Returns:
            Encoded input currents, shape (time_steps, n_samples, input_dim)
            (Norse expects time-first dimension ordering)
        """
        # Check input dimensions and reshape if needed
        if X.ndim == 2:
            # Static features, expand over time
            n_samples, n_features = X.shape
            if n_features != self.input_dim:
                raise ValueError(f"Input features must match input_dim ({self.input_dim})")
            
            # Expand static features over time
            X_expanded = np.repeat(X[:, np.newaxis, :], self.time_steps, axis=1)
        elif X.ndim == 3:
            # Already temporal data
            n_samples, ts, n_features = X.shape
            if n_features != self.input_dim:
                raise ValueError(f"Input features must match input_dim ({self.input_dim})")
            
            # Ensure correct time steps
            if ts < self.time_steps:
                # Pad with last values
                X_expanded = np.pad(
                    X, 
                    ((0, 0), (0, self.time_steps - ts), (0, 0)),
                    mode='edge'
                )
            elif ts > self.time_steps:
                # Truncate
                X_expanded = X[:, :self.time_steps, :]
            else:
                X_expanded = X
        else:
            raise ValueError(f"Invalid input shape: {X.shape}")
        
        # Normalize input to appropriate range for neurons
        # Z-score normalization with clipping for stability
        X_mean = np.mean(X_expanded, axis=(1, 2), keepdims=True)
        X_std = np.std(X_expanded, axis=(1, 2), keepdims=True) + 1e-6
        X_norm = np.clip((X_expanded - X_mean) / X_std, -3, 3)
        
        # Scale for neuron activation
        X_currents = X_norm * 0.5
        
        # Convert to PyTorch tensor and move to device
        X_tensor = torch.tensor(X_currents, dtype=torch.float32, device=self.device)
        
        # Reorder dimensions to [time_steps, batch_size, input_dim] for Norse
        X_tensor = X_tensor.permute(1, 0, 2)
        
        return X_tensor
    
    def _decode_output(self, outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Decode network outputs to predictions.
        
        Args:
            outputs: Dictionary of network outputs over time
                    
        Returns:
            Decoded predictions, shape (n_samples, output_dim)
        """
        # Extract DCN spikes (output layer)
        dcn_spikes = outputs["dcn_spikes"]  # Shape: [time_steps, batch_size, n_dcn]
        
        # Convert to numpy
        dcn_spikes_np = dcn_spikes.detach().cpu().numpy()
        
        # Reorder dimensions to [batch_size, time_steps, n_dcn]
        dcn_spikes_np = np.transpose(dcn_spikes_np, (1, 0, 2))
        
        n_samples, ts, n_dcn = dcn_spikes_np.shape
        if n_dcn != self.n_dcn:
            raise ValueError(f"Output neurons ({n_dcn}) don't match n_dcn ({self.n_dcn})")
        
        # Use firing rate decoding: average over last 20% of simulation
        window_size = max(1, int(ts * 0.2))
        firing_rates = np.mean(dcn_spikes_np[:, -window_size:, :], axis=1)
        
        # Normalize and clip
        max_rates = np.max(firing_rates, axis=1, keepdims=True)
        max_rates = np.where(max_rates > 0, max_rates, 1.0)
        predictions = firing_rates / max_rates
        
        # Clip to [0, 1] range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        # Return first output_dim neurons
        predictions = predictions[:, :self.output_dim]
        
        # Squeeze if single output dimension
        if self.output_dim == 1:
            predictions = predictions.squeeze(axis=1)
        
        return predictions
    
    def _run_simulation(self, X_tensor):
        """
        Run the network simulation over time
        
        Args:
            X_tensor: Input tensor of shape [time_steps, batch_size, input_dim]
            
        Returns:
            Dictionary of all layer outputs over time
        """
        # Reset network state
        self.network.reset_state()
        
        # Prepare output containers
        time_steps, batch_size, _ = X_tensor.shape
        outputs = {
            "mf_grc_current": [],
            "mf_goc_current": [],
            "mf_dcn_current": [],
            "goc_spikes": [],
            "goc_grc_current": [],
            "grc_spikes": [],
            "grc_pc_current": [],
            "pc_spikes": [],
            "pc_dcn_current": [],
            "dcn_spikes": []
        }
        
        # Run simulation step by step
        for t in range(time_steps):
            # Get current input
            x_t = X_tensor[t]
            
            # Run network for one step
            step_outputs = self.network(x_t)
            
            # Store outputs
            for k, v in step_outputs.items():
                outputs[k].append(v)
        
        # Stack outputs along time dimension
        for k in outputs.keys():
            outputs[k] = torch.stack(outputs[k])
        
        return outputs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the CERFLANN Norse network.
        
        Args:
            X: Input data. Can be:
               - Shape (n_samples, time_steps, input_dim): Temporal data
               - Shape (n_samples, input_dim): Static data (will be expanded)
               - Shape (time_steps, input_dim): Single sample temporal data
               - Shape (input_dim,): Single sample static data
               
        Returns:
            Predictions, shape (n_samples, output_dim) or (output_dim,) for single sample
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized")
        
        # Ensure X has correct dimensions
        if X.ndim == 1:
            # Single sample, static data
            X = X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[0] == self.input_dim:
            # Single sample, transposed
            X = X.T.reshape(1, -1)
        elif X.ndim == 2 and X.shape[1] == self.time_steps:
            # Single sample, temporal data
            X = X.reshape(1, X.shape[0], X.shape[1])
        
        # Remember if this was a single sample
        single_sample = X.ndim == 2 or (X.ndim == 3 and X.shape[0] == 1)
        
        # Encode input
        try:
            encoded_input = self._encode_input(X)
            logger.debug(f"Predicting with input shape {encoded_input.shape}")
        except Exception as e:
            logger.error(f"Error encoding input: {e}", exc_info=True)
            if single_sample:
                return np.zeros(self.output_dim)
            else:
                return np.zeros((X.shape[0], self.output_dim))
        
        # Run simulation
        try:
            with torch.no_grad():
                outputs = self._run_simulation(encoded_input)
            
            # Decode outputs
            predictions = self._decode_output(outputs)
            
            # Return single sample without batch dimension if needed
            if single_sample and predictions.ndim > 1:
                return predictions[0]
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            if single_sample:
                return np.zeros(self.output_dim)
            else:
                return np.zeros((X.shape[0], self.output_dim))
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 5, batch_size: int = 32,
            learning_rate: float = 1e-4):
        """
        Train the CERFLANN Norse network using surrogate gradients for backprop through time.
        
        Args:
            X_train: Training data
            y_train: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized")
        
        logger.info(f"Training CERFLANN Norse for {epochs} epochs with batch size {batch_size}")
        
        # Encode inputs
        encoded_input = self._encode_input(X_train)
        n_samples = encoded_input.shape[1]  # Batch dimension is the second one
        
        # Ensure y_train has correct shape
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Convert targets to PyTorch tensor
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        
        # Define optimizer (Adam)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data indices
            indices = torch.randperm(n_samples, device=self.device)
            
            # Track metrics
            epoch_loss = 0.0
            
            # Loop through batches
            for i in range(0, n_samples, batch_size):
                # Get batch indices
                batch_indices = indices[i:min(i+batch_size, n_samples)]
                
                # Extract batch data
                x_batch = encoded_input[:, batch_indices, :]  # [time_steps, batch_size, input_dim]
                y_batch = y_tensor[batch_indices]  # [batch_size, output_dim]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self._run_simulation(x_batch)
                
                # Decode outputs
                dcn_spikes = outputs["dcn_spikes"]  # [time_steps, batch_size, n_dcn]
                
                # Compute firing rates over last 20% of simulation
                time_steps = dcn_spikes.shape[0]
                window_size = max(1, int(time_steps * 0.2))
                firing_rates = torch.mean(dcn_spikes[-window_size:], dim=0)  # [batch_size, n_dcn]
                
                # Normalize rates for prediction
                max_rates = torch.max(firing_rates, dim=1, keepdim=True)[0]
                max_rates = torch.where(max_rates > 0, max_rates, torch.ones_like(max_rates))
                predictions = firing_rates / max_rates
                
                # Ensure prediction matches target shape
                predictions = predictions[:, :self.output_dim]
                
                # Compute MSE loss
                loss = F.mse_loss(predictions, y_batch)
                
                # Backpropagate
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item() * len(batch_indices)
            
            # Log epoch statistics
            epoch_loss /= n_samples
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
        
        self._trained = True
        logger.info("Training completed")
    
    def save(self, filepath: str):
        """
        Save the CERFLANN Norse model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save network state dict
            torch.save(self.network.state_dict(), filepath + ".pth")
            
            # Save additional parameters
            meta_data = {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'n_grc': self.n_grc,
                'n_pc': self.n_pc,
                'n_goc': self.n_goc,
                'n_dcn': self.n_dcn,
                'time_steps': self.time_steps,
                'dt': self.dt,
                'seed': self.seed,
                'use_adex': self.use_adex,
                'trained': self._trained
            }
            
            # Save metadata
            np.save(filepath + ".meta", meta_data)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False
    
    def load(self, filepath: str):
        """
        Load the CERFLANN Norse model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized")
        
        try:
            # Load network state dict
            state_dict = torch.load(filepath + ".pth", map_location=self.device)
            self.network.load_state_dict(state_dict)
            
            # Load additional parameters
            try:
                meta_data = np.load(filepath + ".meta", allow_pickle=True).item()
                self._trained = meta_data.get('trained', True)
            except FileNotFoundError:
                # No metadata file, assume trained
                self._trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str, device=None):
        """
        Load a CERFLANN Norse model from a file and create a new instance.
        
        Args:
            filepath: Path to the saved model
            device: PyTorch device to use
            
        Returns:
            New CERFLANN_Norse instance with loaded model
        """
        try:
            # Load metadata
            meta_data = np.load(filepath + ".meta", allow_pickle=True).item()
            
            # Create new instance with loaded parameters
            model = cls(
                input_dim=meta_data['input_dim'],
                output_dim=meta_data['output_dim'],
                n_granule=meta_data['n_grc'],
                n_purkinje=meta_data['n_pc'],
                n_golgi=meta_data['n_goc'],
                n_dcn=meta_data['n_dcn'],
                time_steps=meta_data['time_steps'],
                dt=meta_data['dt'],
                use_adex=meta_data['use_adex'],
                seed=meta_data['seed'],
                device=device
            )
            
            # Load network state
            model.load(filepath)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model from file: {e}", exc_info=True)
            return None
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained"""
        return getattr(self, '_trained', False)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the network has been successfully initialized"""
        return getattr(self, '_initialized', False)
