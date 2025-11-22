"""Consciousness-driven adaptive learning for NHITS.

Implements self-adaptive learning that evolves based on consciousness feedback
and collective intelligence patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque
import math

from ....core.consciousness import ConsciousnessField
from ....core.syntergic import SyntergicSystem


class ConsciousnessAdaptiveLearning(nn.Module):
    """Adaptive learning system guided by consciousness feedback."""
    
    def __init__(
        self,
        model_dim: int,
        consciousness_dim: int = 256,
        adaptation_rate: float = 0.01,
        memory_size: int = 1000,
        evolution_depth: int = 3,
        coherence_threshold: float = 0.6
    ):
        super().__init__()
        self.model_dim = model_dim
        self.consciousness_dim = consciousness_dim
        self.adaptation_rate = adaptation_rate
        self.coherence_threshold = coherence_threshold
        
        # Consciousness feedback encoder
        self.feedback_encoder = nn.Sequential(
            nn.Linear(consciousness_dim + model_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.Tanh()
        )
        
        # Parameter evolution network
        self.param_evolution = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim)
            ) for _ in range(evolution_depth)
        ])
        
        # Learning rate modulator
        self.lr_modulator = nn.Sequential(
            nn.Linear(consciousness_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Pattern recognition for adaptive strategies
        self.pattern_recognizer = nn.LSTM(
            input_size=model_dim,
            hidden_size=consciousness_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Meta-learning controller
        self.meta_controller = nn.Sequential(
            nn.Linear(consciousness_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, consciousness_dim),
            nn.LayerNorm(consciousness_dim)
        )
        
        # Experience memory
        self.experience_memory = deque(maxlen=memory_size)
        self.pattern_library = {}
        self.adaptation_history = deque(maxlen=100)
        
        # Initialize systems
        self.consciousness_field = ConsciousnessField()
        self.syntergic_system = SyntergicSystem()
        
    def encode_experience(
        self,
        prediction_error: torch.Tensor,
        consciousness_state: torch.Tensor,
        model_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode experience into learnable representation."""
        # Flatten model state
        model_features = []
        for key, tensor in model_state.items():
            if tensor.numel() > 0:
                model_features.append(tensor.flatten()[:self.model_dim])
                
        if model_features:
            model_encoding = torch.cat(model_features).mean()
        else:
            model_encoding = torch.zeros(self.model_dim)
            
        # Combine with consciousness
        combined = torch.cat([
            consciousness_state.flatten()[:self.consciousness_dim],
            model_encoding.unsqueeze(0).expand(self.consciousness_dim)
        ])
        
        # Encode experience
        experience = self.feedback_encoder(combined)
        
        return experience
        
    def compute_adaptation_delta(
        self,
        experience: torch.Tensor,
        target_param: torch.Tensor
    ) -> torch.Tensor:
        """Compute parameter adaptation based on experience."""
        # Evolve through adaptation layers
        delta = experience
        for evolution_layer in self.param_evolution:
            delta = delta + evolution_layer(delta)
            
        # Scale by target parameter shape
        if target_param.dim() > 1:
            delta = delta.view_as(target_param)
        else:
            delta = delta.mean() * torch.ones_like(target_param)
            
        return delta * self.adaptation_rate
        
    def recognize_learning_pattern(
        self,
        experience_sequence: List[torch.Tensor]
    ) -> Tuple[str, float]:
        """Recognize learning patterns from experience sequence."""
        if len(experience_sequence) < 3:
            return "unknown", 0.0
            
        # Stack experiences
        seq_tensor = torch.stack(experience_sequence[-10:]).unsqueeze(0)
        
        # Process through pattern recognizer
        pattern_features, _ = self.pattern_recognizer(seq_tensor)
        pattern_embedding = pattern_features[:, -1, :].squeeze(0)
        
        # Compare with known patterns
        best_match = "unknown"
        best_score = 0.0
        
        for pattern_name, pattern_data in self.pattern_library.items():
            similarity = F.cosine_similarity(
                pattern_embedding.unsqueeze(0),
                pattern_data['embedding'].unsqueeze(0)
            ).item()
            
            if similarity > best_score:
                best_score = similarity
                best_match = pattern_name
                
        # Identify new patterns
        if best_score < 0.7:  # Low similarity to known patterns
            pattern_variance = torch.var(torch.stack(experience_sequence)).item()
            
            if pattern_variance < 0.1:
                best_match = "convergent"
            elif pattern_variance > 0.5:
                best_match = "divergent"
            else:
                best_match = "oscillatory"
                
            # Store new pattern
            self.pattern_library[f"{best_match}_{len(self.pattern_library)}"] = {
                'embedding': pattern_embedding.detach(),
                'variance': pattern_variance,
                'first_seen': len(self.experience_memory)
            }
            
        return best_match, best_score
        
    def adapt_learning_strategy(
        self,
        pattern_type: str,
        consciousness_coherence: float
    ) -> Dict[str, float]:
        """Adapt learning strategy based on recognized patterns."""
        strategy = {
            'learning_rate_multiplier': 1.0,
            'exploration_rate': 0.1,
            'momentum': 0.9,
            'adaptation_strength': self.adaptation_rate
        }
        
        # Adjust based on pattern type
        if pattern_type == "convergent":
            # Near optimal - reduce learning rate, increase exploitation
            strategy['learning_rate_multiplier'] = 0.5
            strategy['exploration_rate'] = 0.05
            strategy['momentum'] = 0.95
            
        elif pattern_type == "divergent":
            # Far from optimal - increase learning rate and exploration
            strategy['learning_rate_multiplier'] = 2.0
            strategy['exploration_rate'] = 0.3
            strategy['momentum'] = 0.7
            
        elif pattern_type == "oscillatory":
            # Cycling behavior - dampen oscillations
            strategy['learning_rate_multiplier'] = 0.8
            strategy['exploration_rate'] = 0.15
            strategy['momentum'] = 0.85
            
        # Modulate by consciousness coherence
        if consciousness_coherence > self.coherence_threshold:
            # High coherence - trust collective intelligence
            strategy['learning_rate_multiplier'] *= 1.2
            strategy['adaptation_strength'] *= 1.5
        else:
            # Low coherence - be more conservative
            strategy['learning_rate_multiplier'] *= 0.8
            strategy['adaptation_strength'] *= 0.7
            
        return strategy
        
    def generate_exploration_noise(
        self,
        param_shape: torch.Size,
        exploration_rate: float,
        consciousness_state: torch.Tensor
    ) -> torch.Tensor:
        """Generate consciousness-guided exploration noise."""
        # Base noise
        noise = torch.randn(param_shape) * exploration_rate
        
        # Modulate by consciousness patterns
        consciousness_mod = self.lr_modulator(consciousness_state).item()
        noise = noise * consciousness_mod
        
        # Add structured exploration based on consciousness
        if consciousness_state.mean() > 0:
            # Positive consciousness - explore promising directions
            direction = consciousness_state.reshape(-1)[:noise.numel()].reshape(param_shape)
            noise = noise + 0.1 * exploration_rate * direction
            
        return noise
        
    def meta_learn(
        self,
        task_experiences: List[Dict[str, Any]],
        consciousness_evolution: torch.Tensor
    ) -> Dict[str, Any]:
        """Meta-learning from task experiences."""
        if not task_experiences:
            return {}
            
        # Extract learning curves
        learning_curves = []
        for exp in task_experiences:
            if 'error_history' in exp:
                learning_curves.append(exp['error_history'])
                
        if not learning_curves:
            return {}
            
        # Analyze learning efficiency
        avg_convergence_speed = np.mean([
            len(curve) for curve in learning_curves
        ])
        
        final_errors = [curve[-1] if curve else float('inf') for curve in learning_curves]
        avg_final_error = np.mean(final_errors)
        
        # Generate meta-insights
        meta_insights = {
            'convergence_speed': avg_convergence_speed,
            'final_performance': 1.0 / (1.0 + avg_final_error),
            'learning_stability': 1.0 - np.std(final_errors) / (np.mean(final_errors) + 1e-6)
        }
        
        # Update meta-controller
        consciousness_summary = consciousness_evolution.mean(dim=0)
        task_summary = torch.tensor([
            meta_insights['convergence_speed'],
            meta_insights['final_performance'],
            meta_insights['learning_stability']
        ]).repeat(self.consciousness_dim // 3 + 1)[:self.consciousness_dim]
        
        meta_update = self.meta_controller(
            torch.cat([consciousness_summary, task_summary])
        )
        
        meta_insights['meta_update'] = meta_update
        meta_insights['recommended_lr'] = 0.001 * meta_insights['final_performance']
        meta_insights['recommended_batch_size'] = int(32 * meta_insights['learning_stability'])
        
        return meta_insights
        
    def adapt_parameters(
        self,
        model: nn.Module,
        error_signal: torch.Tensor,
        consciousness_state: torch.Tensor,
        training_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main adaptation function for model parameters."""
        # Encode current experience
        model_state = {name: param.data for name, param in model.named_parameters()}
        experience = self.encode_experience(error_signal, consciousness_state, model_state)
        
        # Store experience
        self.experience_memory.append({
            'experience': experience.detach(),
            'error': error_signal.item() if error_signal.numel() == 1 else error_signal.mean().item(),
            'consciousness': consciousness_state.detach(),
            'timestamp': len(self.experience_memory)
        })
        
        # Recognize learning pattern
        recent_experiences = [exp['experience'] for exp in list(self.experience_memory)[-10:]]
        pattern_type, pattern_confidence = self.recognize_learning_pattern(recent_experiences)
        
        # Get consciousness coherence
        coherence = self.consciousness_field.get_coherence()
        
        # Adapt learning strategy
        strategy = self.adapt_learning_strategy(pattern_type, coherence)
        
        # Apply adaptations to model
        adaptations = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Compute adaptation delta
                    delta = self.compute_adaptation_delta(experience, param)
                    
                    # Add exploration noise
                    noise = self.generate_exploration_noise(
                        param.shape,
                        strategy['exploration_rate'],
                        consciousness_state
                    )
                    
                    # Apply momentum
                    if name in training_state.get('momentum_buffer', {}):
                        momentum = training_state['momentum_buffer'][name]
                        momentum = strategy['momentum'] * momentum + (1 - strategy['momentum']) * delta
                        delta = momentum
                        
                    # Apply adaptation
                    param.data += strategy['adaptation_strength'] * (delta + noise)
                    
                    adaptations[name] = {
                        'delta': delta.norm().item(),
                        'noise': noise.norm().item()
                    }
                    
        # Store adaptation history
        self.adaptation_history.append({
            'pattern_type': pattern_type,
            'pattern_confidence': pattern_confidence,
            'strategy': strategy,
            'coherence': coherence,
            'error': error_signal.mean().item(),
            'adaptations': adaptations
        })
        
        # Prepare diagnostics
        diagnostics = {
            'pattern_type': pattern_type,
            'pattern_confidence': pattern_confidence,
            'learning_strategy': strategy,
            'consciousness_coherence': coherence,
            'adaptation_magnitude': np.mean([a['delta'] for a in adaptations.values()]),
            'exploration_magnitude': np.mean([a['noise'] for a in adaptations.values()]),
            'experience_buffer_size': len(self.experience_memory),
            'known_patterns': len(self.pattern_library)
        }
        
        return diagnostics
        
    def consolidate_learning(self) -> Dict[str, Any]:
        """Consolidate learning experiences into long-term patterns."""
        if len(self.experience_memory) < 100:
            return {'status': 'insufficient_experience'}
            
        # Extract experiences
        all_experiences = [exp['experience'] for exp in self.experience_memory]
        all_errors = [exp['error'] for exp in self.experience_memory]
        
        # Identify successful adaptations
        error_reduction_indices = []
        for i in range(1, len(all_errors)):
            if all_errors[i] < all_errors[i-1] * 0.95:  # 5% improvement
                error_reduction_indices.append(i)
                
        # Extract successful patterns
        successful_patterns = []
        for idx in error_reduction_indices:
            if idx >= 5:  # Need context
                pattern_seq = all_experiences[idx-5:idx+1]
                successful_patterns.append(torch.stack(pattern_seq))
                
        # Consolidate into pattern library
        if successful_patterns:
            # Cluster similar patterns
            consolidated = {
                'successful_adaptations': len(successful_patterns),
                'average_improvement': np.mean([
                    (all_errors[i-1] - all_errors[i]) / all_errors[i-1]
                    for i in error_reduction_indices
                ]),
                'pattern_diversity': len(self.pattern_library),
                'consolidation_timestamp': len(self.experience_memory)
            }
            
            # Update pattern library with successful patterns
            for i, pattern in enumerate(successful_patterns[:10]):  # Keep top 10
                pattern_name = f"consolidated_{len(self.pattern_library)}"
                _, (hidden, _) = self.pattern_recognizer(pattern.unsqueeze(0))
                self.pattern_library[pattern_name] = {
                    'embedding': hidden.squeeze(0).detach(),
                    'success_rate': 0.95,
                    'usage_count': 0
                }
                
            return consolidated
            
        return {'status': 'no_successful_patterns'}
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning progress."""
        if not self.adaptation_history:
            return {'status': 'no_adaptation_history'}
            
        recent_history = list(self.adaptation_history)[-20:]
        
        insights = {
            'dominant_pattern': max(
                set(h['pattern_type'] for h in recent_history),
                key=lambda x: sum(1 for h in recent_history if h['pattern_type'] == x)
            ),
            'average_coherence': np.mean([h['coherence'] for h in recent_history]),
            'error_trend': 'improving' if recent_history[-1]['error'] < recent_history[0]['error'] else 'worsening',
            'exploration_trend': np.mean([h['strategy']['exploration_rate'] for h in recent_history]),
            'adaptation_stability': 1.0 - np.std([h['adaptation_magnitude'] for h in recent_history])
        }
        
        return insights