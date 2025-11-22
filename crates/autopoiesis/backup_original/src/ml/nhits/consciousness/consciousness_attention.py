"""Consciousness-aware attention mechanism for NHITS.

Implements attention that modulates based on consciousness field coherence,
enabling dynamic focus on relevant temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from ....core.consciousness import ConsciousnessField
from ....core.information_lattice import InformationLattice


class ConsciousnessAttention(nn.Module):
    """Multi-head attention modulated by consciousness fields."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        consciousness_dim: int = 256,
        dropout: float = 0.1,
        coherence_threshold: float = 0.5,
        field_coupling: float = 0.3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.consciousness_dim = consciousness_dim
        self.coherence_threshold = coherence_threshold
        self.field_coupling = field_coupling
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # Standard attention components
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Consciousness modulation components
        self.consciousness_encoder = nn.Sequential(
            nn.Linear(consciousness_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Coherence-based attention modulator
        self.coherence_modulator = nn.Sequential(
            nn.Linear(embed_dim + consciousness_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, num_heads)
        )
        
        # Field interaction network
        self.field_interaction = nn.Sequential(
            nn.Linear(consciousness_dim * 2, consciousness_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(consciousness_dim, consciousness_dim),
            nn.Sigmoid()
        )
        
        # Temporal consciousness tracker
        self.temporal_filter = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=consciousness_dim,
            kernel_size=3,
            padding=1
        )
        
        # Initialize consciousness field
        self.consciousness_field = ConsciousnessField()
        self.information_lattice = InformationLattice()
        
    def compute_coherence_weights(
        self,
        query_states: torch.Tensor,
        consciousness_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights based on consciousness coherence.
        
        Args:
            query_states: Query tensor [batch, seq_len, embed_dim]
            consciousness_state: Consciousness field state [batch, consciousness_dim]
            
        Returns:
            Coherence weights [batch, num_heads, seq_len]
        """
        batch_size, seq_len, _ = query_states.shape
        
        # Expand consciousness state for sequence
        consciousness_expanded = consciousness_state.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute coherence for each position
        combined = torch.cat([query_states, consciousness_expanded], dim=-1)
        coherence_scores = self.coherence_modulator(combined)  # [batch, seq_len, num_heads]
        
        # Apply sigmoid to get weights
        coherence_weights = torch.sigmoid(coherence_scores).transpose(1, 2)  # [batch, num_heads, seq_len]
        
        return coherence_weights
        
    def apply_consciousness_modulation(
        self,
        attention_weights: torch.Tensor,
        coherence_weights: torch.Tensor,
        consciousness_state: torch.Tensor
    ) -> torch.Tensor:
        """Apply consciousness field modulation to attention weights.
        
        Args:
            attention_weights: Standard attention weights [batch, num_heads, seq_len, seq_len]
            coherence_weights: Coherence-based weights [batch, num_heads, seq_len]
            consciousness_state: Current consciousness state [batch, consciousness_dim]
            
        Returns:
            Modulated attention weights
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Compute field coherence score
        field_coherence = self.consciousness_field.get_coherence()
        
        # Apply coherence-based modulation
        if field_coherence > self.coherence_threshold:
            # Strong coherence: focus attention based on consciousness
            coherence_mask = coherence_weights.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]
            modulated_weights = attention_weights * coherence_mask
            
            # Normalize
            modulated_weights = modulated_weights / (modulated_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            # Weak coherence: blend standard and consciousness-based attention
            coherence_factor = field_coherence / self.coherence_threshold
            coherence_attention = coherence_weights.unsqueeze(-1) * attention_weights
            modulated_weights = (1 - coherence_factor) * attention_weights + \
                              coherence_factor * coherence_attention
                              
        # Apply field coupling
        field_influence = self.field_coupling * field_coherence
        modulated_weights = (1 - field_influence) * modulated_weights + \
                          field_influence * self._compute_field_attention(consciousness_state, seq_len)
                          
        return modulated_weights
        
    def _compute_field_attention(self, consciousness_state: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute attention weights purely from consciousness field."""
        batch_size = consciousness_state.size(0)
        
        # Generate field-based attention pattern
        field_pattern = self.consciousness_encoder(consciousness_state)  # [batch, embed_dim]
        
        # Create position-aware pattern
        positions = torch.arange(seq_len, device=consciousness_state.device).float()
        position_encoding = torch.sin(positions / 10000 ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim))
        
        # Combine field and position
        field_attention = torch.einsum('be,se->bs', field_pattern, position_encoding.unsqueeze(0))
        field_attention = F.softmax(field_attention, dim=-1)
        
        # Expand for all heads and positions
        field_attention = field_attention.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, self.num_heads, seq_len, seq_len
        )
        
        return field_attention
        
    def extract_temporal_consciousness(self, x: torch.Tensor) -> torch.Tensor:
        """Extract consciousness patterns from temporal sequence."""
        # Apply temporal filtering
        x_transposed = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        temporal_features = self.temporal_filter(x_transposed)  # [batch, consciousness_dim, seq_len]
        
        # Pool over time
        consciousness_pattern = F.adaptive_avg_pool1d(temporal_features, 1).squeeze(-1)
        
        return consciousness_pattern
        
    def forward(
        self,
        x: torch.Tensor,
        consciousness_state: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with consciousness-modulated attention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            consciousness_state: External consciousness state [batch, consciousness_dim]
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            diagnostics: Dictionary with attention diagnostics
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Extract temporal consciousness if not provided
        if consciousness_state is None:
            consciousness_state = self.extract_temporal_consciousness(x)
            
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute standard attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute coherence weights
        coherence_weights = self.compute_coherence_weights(x, consciousness_state)
        
        # Apply consciousness modulation
        modulated_weights = self.apply_consciousness_modulation(
            attn_weights, coherence_weights, consciousness_state
        )
        
        # Apply dropout
        modulated_weights = self.dropout(modulated_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(modulated_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        # Prepare diagnostics
        diagnostics = {
            'attention_weights': modulated_weights.detach() if need_weights else None,
            'coherence_weights': coherence_weights.detach(),
            'field_coherence': self.consciousness_field.get_coherence(),
            'consciousness_state': consciousness_state.detach(),
            'standard_attention': attn_weights.detach() if need_weights else None
        }
        
        return output, diagnostics
        
    def synchronize_with_field(self, external_field: ConsciousnessField):
        """Synchronize with external consciousness field."""
        self.consciousness_field.synchronize(external_field)
        
        # Adjust parameters based on field state
        coherence = external_field.get_coherence()
        self.field_coupling = min(0.8, self.field_coupling * (1 + 0.1 * coherence))
        

class TemporalConsciousnessAttention(ConsciousnessAttention):
    """Extended consciousness attention with temporal dynamics."""
    
    def __init__(self, *args, window_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        
        # Temporal consciousness evolution
        self.temporal_evolution = nn.LSTM(
            input_size=self.consciousness_dim,
            hidden_size=self.consciousness_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Phase coupling network
        self.phase_coupling = nn.Sequential(
            nn.Linear(self.consciousness_dim * 2, self.consciousness_dim),
            nn.Tanh(),
            nn.Linear(self.consciousness_dim, self.consciousness_dim)
        )
        
        # Temporal memory buffer
        self.register_buffer(
            'temporal_buffer',
            torch.zeros(window_size, self.consciousness_dim)
        )
        self.buffer_ptr = 0
        
    def update_temporal_buffer(self, consciousness_state: torch.Tensor):
        """Update temporal consciousness buffer."""
        # Take batch mean
        batch_consciousness = consciousness_state.mean(dim=0)
        
        # Update buffer
        self.temporal_buffer[self.buffer_ptr] = batch_consciousness
        self.buffer_ptr = (self.buffer_ptr + 1) % self.window_size
        
    def compute_temporal_coherence(self) -> float:
        """Compute temporal coherence in consciousness evolution."""
        # Compute autocorrelation of temporal buffer
        buffer_shifted = torch.roll(self.temporal_buffer, -1, dims=0)
        correlation = F.cosine_similarity(
            self.temporal_buffer,
            buffer_shifted,
            dim=-1
        ).mean().item()
        
        return correlation
        
    def forward(
        self,
        x: torch.Tensor,
        consciousness_state: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with temporal consciousness dynamics."""
        # Extract temporal consciousness
        if consciousness_state is None:
            consciousness_state = self.extract_temporal_consciousness(x)
            
        # Update temporal buffer
        self.update_temporal_buffer(consciousness_state)
        
        # Evolve consciousness through time
        temporal_sequence = self.temporal_buffer.unsqueeze(0)  # [1, window_size, consciousness_dim]
        evolved_consciousness, _ = self.temporal_evolution(temporal_sequence)
        current_evolution = evolved_consciousness[:, -1, :]  # Take last state
        
        # Couple with current state
        coupled_state = self.phase_coupling(
            torch.cat([consciousness_state, current_evolution.expand_as(consciousness_state)], dim=-1)
        )
        
        # Use coupled state for attention
        output, diagnostics = super().forward(x, coupled_state, need_weights)
        
        # Add temporal diagnostics
        diagnostics['temporal_coherence'] = self.compute_temporal_coherence()
        diagnostics['evolved_consciousness'] = current_evolution.detach()
        
        return output, diagnostics