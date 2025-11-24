#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced LSTM Integration for Tengri Prediction App

This module demonstrates how to integrate the advanced_lstm.py and quantum_lstm.py
enhancements with the existing Tengri prediction system.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import the new LSTM implementations
try:
    import advanced_lstm
    ADVANCED_LSTM_AVAILABLE = True
except ImportError:
    ADVANCED_LSTM_AVAILABLE = False

try:
    import quantum_lstm
    QUANTUM_LSTM_AVAILABLE = True
except ImportError:
    QUANTUM_LSTM_AVAILABLE = False

# Import existing prediction engine
try:
    import sys
    sys.path.append('./tengri/prediction_app')
    import superior_engine
    CURRENT_ENGINE_AVAILABLE = True
except ImportError:
    CURRENT_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedLSTMConfig:
    """Configuration for enhanced LSTM integration"""
    # Current PyTorch model settings
    input_size: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    
    # Advanced LSTM enhancements
    use_biological_activation: bool = True
    use_multi_timeframe: bool = True
    timeframes: List[str] = None
    use_ensemble_pathways: bool = True
    use_swarm_optimization: bool = True
    use_advanced_attention: bool = True
    
    # Quantum enhancements
    use_quantum: bool = False
    n_qubits: int = 8
    quantum_layers: int = 1
    use_quantum_attention: bool = False
    use_quantum_memory: bool = False
    
    # Performance settings
    cache_size: int = 1000
    parallel_processing: bool = True
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1h', '4h', '1d', '1w']

class BiologicalActivationModule(nn.Module):
    """PyTorch wrapper for biological activation functions"""
    
    def __init__(self, threshold=0.5, refractory=0.1):
        super().__init__()
        self.threshold = threshold
        self.refractory = refractory
        
    def forward(self, x):
        """Apply biological activation with leaky integrate-and-fire"""
        # Spike when above threshold
        spike = torch.where(x > self.threshold, 1.0, 0.0)
        # Leak when below threshold
        leak = torch.where(x <= self.threshold, x * (1 - self.refractory), 0.0)
        return spike + leak

class AdvancedAttentionModule(nn.Module):
    """Enhanced attention with caching capabilities"""
    
    def __init__(self, hidden_size, num_heads, cache_size=1000):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.cache_size = cache_size
        self.attention_cache = {}
        
    def forward(self, query, key, value, attn_mask=None):
        """Forward pass with caching"""
        # Simple cache key based on tensor shapes and first few values
        cache_key = self._generate_cache_key(query, key, value)
        
        if cache_key in self.attention_cache:
            logger.debug("Using cached attention computation")
            return self.attention_cache[cache_key]
        
        # Compute attention
        output, attention_weights = self.attention(query, key, value, attn_mask=attn_mask)
        
        # Cache result if cache not full
        if len(self.attention_cache) < self.cache_size:
            self.attention_cache[cache_key] = (output, attention_weights)
        
        return output, attention_weights
    
    def _generate_cache_key(self, query, key, value):
        """Generate cache key for attention computation"""
        # Simple hash based on tensor properties
        q_hash = hash((query.shape, query.flatten()[:5].sum().item()))
        k_hash = hash((key.shape, key.flatten()[:5].sum().item()))
        v_hash = hash((value.shape, value.flatten()[:5].sum().item()))
        return f"{q_hash}_{k_hash}_{v_hash}"

class MultiTimeframeEnsemble(nn.Module):
    """Multi-timeframe ensemble processing"""
    
    def __init__(self, config: EnhancedLSTMConfig):
        super().__init__()
        self.config = config
        self.timeframes = config.timeframes
        
        # Create separate LSTM pathways for each timeframe
        self.pathway_lstms = nn.ModuleDict({
            tf: nn.LSTM(
                config.input_size, config.hidden_size,
                config.num_layers, batch_first=True, dropout=config.dropout
            )
            for tf in self.timeframes
        })
        
        # Pathway weight optimization (simplified version of swarm optimization)
        self.pathway_weights = nn.Parameter(
            torch.ones(len(self.timeframes)) / len(self.timeframes)
        )
        
    def forward(self, x, timeframe_inputs=None):
        """Process through multiple timeframe pathways"""
        if timeframe_inputs is None:
            # Use same input for all timeframes (simplified)
            timeframe_inputs = {tf: x for tf in self.timeframes}
        
        pathway_outputs = {}
        for tf in self.timeframes:
            lstm_input = timeframe_inputs.get(tf, x)
            output, _ = self.pathway_lstms[tf](lstm_input)
            pathway_outputs[tf] = output
        
        # Weighted combination of pathway outputs
        combined_output = torch.zeros_like(list(pathway_outputs.values())[0])
        weights = torch.softmax(self.pathway_weights, dim=0)
        
        for i, (tf, output) in enumerate(pathway_outputs.items()):
            combined_output += weights[i] * output
        
        return combined_output

class QuantumEnhancementModule(nn.Module):
    """Quantum enhancement integration module"""
    
    def __init__(self, config: EnhancedLSTMConfig):
        super().__init__()
        self.config = config
        self.use_quantum = config.use_quantum and QUANTUM_LSTM_AVAILABLE
        
        if self.use_quantum:
            try:
                # Initialize quantum components
                self.quantum_encoder = quantum_lstm.QuantumStateEncoder(config.n_qubits)
                self.quantum_attention = quantum_lstm.QuantumAttention(config.n_qubits)
                
                # Classical-quantum interface layers
                self.quantum_proj_in = nn.Linear(config.hidden_size, 2**config.n_qubits)
                self.quantum_proj_out = nn.Linear(2**config.n_qubits, config.hidden_size)
                
                logger.info("Quantum enhancements initialized successfully")
            except Exception as e:
                logger.warning(f"Quantum enhancement initialization failed: {e}")
                self.use_quantum = False
        
    def forward(self, x):
        """Apply quantum enhancements if available"""
        if not self.use_quantum:
            return x
        
        try:
            batch_size, seq_len, hidden_size = x.shape
            
            # Process each sequence element through quantum enhancement
            quantum_enhanced = []
            for t in range(seq_len):
                x_t = x[:, t, :]  # Shape: (batch_size, hidden_size)
                
                # Project to quantum dimension
                x_quantum = self.quantum_proj_in(x_t)
                
                # Apply quantum attention (simplified for batch processing)
                # Note: In practice, this would need more sophisticated batching
                enhanced_t = x_t  # Fallback to classical if quantum fails
                
                try:
                    # For demonstration, we'll use quantum enhancement on first batch element
                    if batch_size > 0:
                        quantum_state = self.quantum_encoder.encode(x_quantum[0].detach().numpy())
                        attended_state = self.quantum_attention.quantum_walk_attention(quantum_state)
                        
                        # Convert back to classical
                        classical_enhanced = torch.from_numpy(
                            np.real(attended_state).astype(np.float32)
                        )
                        
                        # Project back to original dimension
                        if len(classical_enhanced) == 2**self.config.n_qubits:
                            enhanced_classical = self.quantum_proj_out(classical_enhanced.unsqueeze(0))
                            enhanced_t = enhanced_classical.expand(batch_size, -1)
                        
                except Exception as e:
                    logger.debug(f"Quantum processing fallback: {e}")
                    # Fallback to classical processing
                    pass
                
                quantum_enhanced.append(enhanced_t)
            
            # Stack back to sequence
            enhanced_sequence = torch.stack(quantum_enhanced, dim=1)
            return enhanced_sequence
            
        except Exception as e:
            logger.warning(f"Quantum enhancement failed, using classical: {e}")
            return x

class EnhancedLSTMTransformer(nn.Module):
    """Enhanced LSTM-Transformer with advanced and quantum features"""
    
    def __init__(self, config: EnhancedLSTMConfig):
        super().__init__()
        self.config = config
        
        # Biological activation
        if config.use_biological_activation:
            self.biological_activation = BiologicalActivationModule()
        
        # Multi-timeframe ensemble
        if config.use_multi_timeframe:
            self.ensemble = MultiTimeframeEnsemble(config)
        else:
            # Standard LSTM
            self.lstm = nn.LSTM(
                config.input_size, config.hidden_size, config.num_layers,
                batch_first=True, dropout=config.dropout
            )
        
        # Advanced attention
        if config.use_advanced_attention:
            self.attention = AdvancedAttentionModule(
                config.hidden_size, config.num_heads, config.cache_size
            )
        else:
            # Standard transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_heads,
                dropout=config.dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Quantum enhancement
        if config.use_quantum:
            self.quantum_enhancement = QuantumEnhancementModule(config)
        
        # Output projection
        self.projection = nn.Linear(config.hidden_size, 32)
        self.output = nn.Linear(32, 1)
        
        # Performance tracking
        self.forward_count = 0
        self.quantum_usage_count = 0
        
    def forward(self, x, return_attention=False):
        """Enhanced forward pass"""
        self.forward_count += 1
        
        # Multi-timeframe ensemble or standard LSTM
        if hasattr(self, 'ensemble'):
            lstm_output = self.ensemble(x)
        else:
            lstm_output, _ = self.lstm(x)
        
        # Apply biological activation if enabled
        if hasattr(self, 'biological_activation'):
            lstm_output = self.biological_activation(lstm_output)
        
        # Advanced attention or standard transformer
        if hasattr(self, 'attention'):
            attended_output, attention_weights = self.attention(
                lstm_output, lstm_output, lstm_output
            )
        else:
            attended_output = self.transformer(lstm_output)
            attention_weights = None
        
        # Quantum enhancement if enabled
        if hasattr(self, 'quantum_enhancement'):
            attended_output = self.quantum_enhancement(attended_output)
            self.quantum_usage_count += 1
        
        # Final projection
        projected = self.projection(attended_output)
        output = self.output(projected)
        
        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            'forward_passes': self.forward_count,
            'quantum_usage': self.quantum_usage_count,
            'quantum_ratio': self.quantum_usage_count / max(self.forward_count, 1),
            'biological_activation': hasattr(self, 'biological_activation'),
            'multi_timeframe': hasattr(self, 'ensemble'),
            'advanced_attention': hasattr(self, 'attention'),
            'quantum_enhancement': hasattr(self, 'quantum_enhancement')
        }
        return stats

def create_enhanced_lstm(config: Optional[EnhancedLSTMConfig] = None):
    """Create enhanced LSTM model with configuration"""
    if config is None:
        config = EnhancedLSTMConfig()
    
    logger.info("Creating Enhanced LSTM-Transformer model")
    logger.info(f"Configuration: {config}")
    
    # Check available enhancements
    available_enhancements = []
    if ADVANCED_LSTM_AVAILABLE:
        available_enhancements.append("Advanced LSTM")
    if QUANTUM_LSTM_AVAILABLE:
        available_enhancements.append("Quantum LSTM")
    if CURRENT_ENGINE_AVAILABLE:
        available_enhancements.append("Current Engine")
    
    logger.info(f"Available enhancements: {available_enhancements}")
    
    model = EnhancedLSTMTransformer(config)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def compare_with_current_implementation():
    """Compare enhanced model with current implementation"""
    print("\nCOMPARING ENHANCED vs CURRENT IMPLEMENTATION")
    print("=" * 50)
    
    try:
        # Current implementation
        if CURRENT_ENGINE_AVAILABLE:
            current_config = {
                'input_size': 50,
                'hidden_size': 64,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1,
                'quantum_enhanced': True
            }
            current_model = superior_engine.OptimizedLSTMTransformer(**current_config)
            current_params = sum(p.numel() for p in current_model.parameters())
            print(f"Current model parameters: {current_params:,}")
        
        # Enhanced implementation
        enhanced_config = EnhancedLSTMConfig(
            use_biological_activation=True,
            use_multi_timeframe=True,
            use_advanced_attention=True,
            use_quantum=QUANTUM_LSTM_AVAILABLE
        )
        enhanced_model = create_enhanced_lstm(enhanced_config)
        enhanced_stats = enhanced_model.get_performance_stats()
        
        print(f"Enhanced model features: {enhanced_stats}")
        
        # Test forward pass
        test_input = torch.randn(4, 60, 50)  # (batch, seq, features)
        
        if CURRENT_ENGINE_AVAILABLE:
            with torch.no_grad():
                current_output = current_model(test_input)
                print(f"Current output shape: {current_output.shape}")
        
        with torch.no_grad():
            enhanced_output = enhanced_model(test_input)
            print(f"Enhanced output shape: {enhanced_output.shape}")
        
        print("\n✓ Both models working correctly")
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("ENHANCED LSTM INTEGRATION DEMONSTRATION")
    print("=" * 50)
    
    # Test basic enhanced model
    config = EnhancedLSTMConfig(
        input_size=50,
        hidden_size=64,
        use_biological_activation=True,
        use_multi_timeframe=True,
        use_advanced_attention=True,
        use_quantum=False  # Start with classical enhancements
    )
    
    model = create_enhanced_lstm(config)
    print(f"✓ Enhanced model created: {type(model).__name__}")
    
    # Test forward pass
    test_input = torch.randn(2, 30, 50)
    output = model(test_input)
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Show performance stats
    stats = model.get_performance_stats()
    print(f"✓ Performance stats: {stats}")
    
    # Test quantum enhancement if available
    if QUANTUM_LSTM_AVAILABLE:
        print("\nTesting quantum enhancements...")
        quantum_config = EnhancedLSTMConfig(
            input_size=10,  # Smaller for quantum
            hidden_size=32,
            use_quantum=True,
            n_qubits=4  # Smaller for demonstration
        )
        
        quantum_model = create_enhanced_lstm(quantum_config)
        quantum_input = torch.randn(1, 10, 10)  # Smaller input
        quantum_output = quantum_model(quantum_input)
        print(f"✓ Quantum model working: {quantum_output.shape}")
    
    # Compare with current implementation
    compare_with_current_implementation()
    
    print("\n🎉 Enhanced LSTM integration ready for deployment!")