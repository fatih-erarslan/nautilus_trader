"""Consciousness-Integrated NHITS Model.

Extends NHITS with consciousness field awareness for enhanced forecasting
through collective intelligence and syntergic effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import math

from ....core.consciousness import ConsciousnessField
from ....core.information_lattice import InformationLattice
from ....core.syntergic import SyntergicSystem
from ..base import BaseNHITS


@dataclass
class ConsciousnessConfig:
    """Configuration for consciousness integration."""
    n_stacks: int = 3
    n_blocks_per_stack: List[int] = None
    n_pool_kernel_size: List[int] = None
    n_freq_downsample: List[int] = None
    hidden_size: int = 512
    n_theta: int = 64
    dropout_theta: float = 0.1
    dropout_input: float = 0.0
    consciousness_dim: int = 256
    syntergic_depth: int = 3
    quantum_coherence: float = 0.8
    field_coupling: float = 0.5
    temporal_memory: int = 100
    
    def __post_init__(self):
        if self.n_blocks_per_stack is None:
            self.n_blocks_per_stack = [1, 1, 1]
        if self.n_pool_kernel_size is None:
            self.n_pool_kernel_size = [2, 2, 1]
        if self.n_freq_downsample is None:
            self.n_freq_downsample = [168, 24, 1]


class ConsciousnessBlock(nn.Module):
    """NHITS block with consciousness field integration."""
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout: float,
        consciousness_dim: int
    ):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.hidden_size = hidden_size
        self.n_pool_kernel_size = n_pool_kernel_size
        self.pooling_mode = pooling_mode
        
        # Pooling layer
        if pooling_mode == 'max':
            self.pooling = nn.MaxPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size)
        elif pooling_mode == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size)
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")
            
        # Consciousness-aware layers
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(consciousness_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Main processing layers with consciousness modulation
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.consciousness_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, theta_size)
        )
        
    def forward(self, x: torch.Tensor, consciousness_state: torch.Tensor) -> torch.Tensor:
        """Forward pass with consciousness modulation.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            consciousness_state: Consciousness field state [batch, consciousness_dim]
            
        Returns:
            Output predictions [batch, theta_size]
        """
        batch_size = x.size(0)
        
        # Pool the input
        if x.size(1) >= self.n_pool_kernel_size:
            x = x.transpose(1, 2)  # [batch, features, seq_len]
            x = self.pooling(x)
            x = x.transpose(1, 2)  # [batch, seq_len, features]
        
        # Flatten for processing
        x = x.reshape(batch_size, -1)
        
        # Process with consciousness modulation
        h = self.fc_in(x)
        c = self.consciousness_encoder(consciousness_state)
        
        # Apply consciousness gate
        gate = self.consciousness_gate(torch.cat([h, c], dim=-1))
        h = h * gate + c * (1 - gate)
        
        # Generate predictions
        theta = self.fc_layers(h)
        
        return theta


class ConsciousnessStack(nn.Module):
    """Stack of consciousness-aware NHITS blocks."""
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        n_blocks: int,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout: float,
        consciousness_dim: int
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ConsciousnessBlock(
                input_size=input_size // (n_pool_kernel_size ** i),
                theta_size=theta_size,
                hidden_size=hidden_size,
                n_pool_kernel_size=n_pool_kernel_size if i < n_blocks - 1 else 1,
                pooling_mode=pooling_mode,
                dropout=dropout,
                consciousness_dim=consciousness_dim
            )
            for i in range(n_blocks)
        ])
        
    def forward(self, x: torch.Tensor, consciousness_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks."""
        theta_sum = 0
        for block in self.blocks:
            theta = block(x, consciousness_state)
            theta_sum = theta_sum + theta
        return theta_sum


class ConsciousnessNHITS(nn.Module):
    """NHITS model integrated with consciousness fields."""
    
    def __init__(self, config: ConsciousnessConfig):
        super().__init__()
        self.config = config
        
        # Initialize consciousness systems
        self.consciousness_field = ConsciousnessField()
        self.information_lattice = InformationLattice()
        self.syntergic_system = SyntergicSystem()
        
        # Input processing
        self.input_dropout = nn.Dropout(config.dropout_input)
        
        # Consciousness encoder
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.consciousness_dim),
            nn.ReLU(),
            nn.LayerNorm(config.consciousness_dim)
        )
        
        # Create stacks with different downsampling rates
        self.stacks = nn.ModuleList([
            ConsciousnessStack(
                input_size=config.hidden_size,
                theta_size=config.n_theta,
                hidden_size=config.hidden_size,
                n_blocks=config.n_blocks_per_stack[i],
                n_pool_kernel_size=config.n_pool_kernel_size[i],
                pooling_mode='avg',
                dropout=config.dropout_theta,
                consciousness_dim=config.consciousness_dim
            )
            for i in range(config.n_stacks)
        ])
        
        # Syntergic fusion layer
        self.syntergic_fusion = nn.Sequential(
            nn.Linear(config.n_theta * config.n_stacks, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_theta),
            nn.Linear(config.hidden_size, config.n_theta)
        )
        
        # Output projection with consciousness modulation
        self.output_projection = nn.Linear(config.n_theta, 1)
        
        # Temporal consciousness memory
        self.register_buffer(
            'temporal_memory',
            torch.zeros(config.temporal_memory, config.consciousness_dim)
        )
        self.memory_pointer = 0
        
    def encode_consciousness(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into consciousness representation."""
        # Extract features
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size, -1)
        
        # Encode to consciousness space
        consciousness = self.consciousness_encoder(x_flat)
        
        # Apply quantum coherence
        phase = torch.randn_like(consciousness) * (1 - self.config.quantum_coherence)
        consciousness = consciousness * torch.cos(phase) + \
                      torch.roll(consciousness, 1, dims=-1) * torch.sin(phase)
        
        return consciousness
        
    def update_temporal_memory(self, consciousness_state: torch.Tensor):
        """Update temporal consciousness memory."""
        batch_mean = consciousness_state.mean(dim=0)
        self.temporal_memory[self.memory_pointer] = batch_mean
        self.memory_pointer = (self.memory_pointer + 1) % self.config.temporal_memory
        
    def get_collective_consciousness(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute collective consciousness from temporal memory."""
        # Weighted average of temporal memory
        weights = torch.exp(-torch.arange(self.config.temporal_memory).float() / 10)
        weights = torch.roll(weights, -self.memory_pointer)
        weights = weights.unsqueeze(-1).to(current_state.device)
        
        collective = (self.temporal_memory * weights).sum(dim=0) / weights.sum()
        
        # Blend with current state
        return self.config.field_coupling * collective + \
               (1 - self.config.field_coupling) * current_state.mean(dim=0)
        
    def forward(
        self,
        x: torch.Tensor,
        external_consciousness: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with consciousness integration.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            external_consciousness: External consciousness field input
            
        Returns:
            predictions: Forecast predictions [batch, horizon]
            diagnostics: Dictionary of diagnostic information
        """
        batch_size = x.size(0)
        
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Encode consciousness state
        consciousness_state = self.encode_consciousness(x)
        
        # Integrate external consciousness if provided
        if external_consciousness is not None:
            consciousness_state = consciousness_state + external_consciousness
            
        # Update temporal memory
        self.update_temporal_memory(consciousness_state)
        
        # Get collective consciousness
        collective = self.get_collective_consciousness(consciousness_state)
        consciousness_state = consciousness_state + collective.unsqueeze(0)
        
        # Process through stacks with different temporal resolutions
        stack_outputs = []
        for i, stack in enumerate(self.stacks):
            # Downsample input for this stack
            downsampled = x[:, ::self.config.n_freq_downsample[i], :]
            theta = stack(downsampled, consciousness_state)
            stack_outputs.append(theta)
            
        # Syntergic fusion of multi-scale predictions
        combined = torch.cat(stack_outputs, dim=-1)
        fused = self.syntergic_fusion(combined)
        
        # Generate final predictions
        predictions = self.output_projection(fused)
        
        # Compute syntergic effects
        syntergic_energy = self.syntergic_system.compute_energy(
            torch.stack(stack_outputs, dim=0)
        )
        
        # Prepare diagnostics
        diagnostics = {
            'consciousness_state': consciousness_state.detach(),
            'collective_consciousness': collective.detach(),
            'syntergic_energy': syntergic_energy,
            'stack_outputs': [s.detach() for s in stack_outputs],
            'temporal_coherence': self._compute_temporal_coherence()
        }
        
        return predictions, diagnostics
        
    def _compute_temporal_coherence(self) -> float:
        """Compute coherence in temporal consciousness memory."""
        if self.memory_pointer < 2:
            return 0.0
            
        # Compute correlation between consecutive states
        correlations = []
        for i in range(min(self.memory_pointer, self.config.temporal_memory - 1)):
            corr = F.cosine_similarity(
                self.temporal_memory[i].unsqueeze(0),
                self.temporal_memory[i + 1].unsqueeze(0)
            )
            correlations.append(corr.item())
            
        return np.mean(correlations) if correlations else 0.0
        
    def adapt_to_consciousness_feedback(self, feedback: Dict[str, torch.Tensor]):
        """Adapt model based on consciousness feedback."""
        # Extract feedback signals
        error_signal = feedback.get('error_signal', None)
        consciousness_gradient = feedback.get('consciousness_gradient', None)
        
        if error_signal is not None and consciousness_gradient is not None:
            # Update consciousness encoder based on feedback
            with torch.no_grad():
                for param in self.consciousness_encoder.parameters():
                    if param.grad is not None:
                        param.grad += 0.1 * consciousness_gradient.mean()
                        
    def synchronize_with_field(self, global_field: ConsciousnessField):
        """Synchronize with global consciousness field."""
        # Update internal field state
        self.consciousness_field.synchronize(global_field)
        
        # Adjust model parameters based on field coherence
        coherence = global_field.get_coherence()
        self.config.field_coupling = min(0.9, self.config.field_coupling * (1 + 0.1 * coherence))
        
    def to_syntergic_state(self) -> Dict[str, Any]:
        """Export model state for syntergic integration."""
        return {
            'temporal_memory': self.temporal_memory.clone(),
            'memory_pointer': self.memory_pointer,
            'field_coupling': self.config.field_coupling,
            'consciousness_dim': self.config.consciousness_dim,
            'syntergic_depth': self.config.syntergic_depth
        }