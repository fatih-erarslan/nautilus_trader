"""Syntergic Forecaster using collective consciousness.

Implements forecasting that leverages syntergic effects from distributed
consciousness fields for enhanced prediction accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import deque

from .consciousness_nhits import ConsciousnessNHITS, ConsciousnessConfig
from ....core.syntergic import SyntergicSystem, SyntergicState
from ....core.consciousness import ConsciousnessField


@dataclass
class SyntergicConfig:
    """Configuration for syntergic forecasting."""
    n_agents: int = 5  # Number of syntergic agents
    interaction_depth: int = 3  # Depth of agent interactions
    coherence_threshold: float = 0.7  # Minimum coherence for activation
    emergence_rate: float = 0.1  # Rate of emergent pattern formation
    collective_weight: float = 0.3  # Weight of collective predictions
    diversity_bonus: float = 0.2  # Bonus for diverse predictions
    synchronization_rounds: int = 5  # Rounds of synchronization
    memory_horizon: int = 50  # Horizon for collective memory
    

class SyntergicAgent(nn.Module):
    """Individual agent in syntergic forecasting system."""
    
    def __init__(self, agent_id: int, base_config: ConsciousnessConfig):
        super().__init__()
        self.agent_id = agent_id
        
        # Modify config for diversity
        agent_config = ConsciousnessConfig(
            n_stacks=base_config.n_stacks,
            hidden_size=base_config.hidden_size + agent_id * 32,
            consciousness_dim=base_config.consciousness_dim,
            syntergic_depth=base_config.syntergic_depth + agent_id,
            quantum_coherence=base_config.quantum_coherence * (0.8 + 0.04 * agent_id),
            field_coupling=base_config.field_coupling * (0.9 + 0.02 * agent_id)
        )
        
        self.model = ConsciousnessNHITS(agent_config)
        
        # Agent-specific consciousness modulator
        self.consciousness_modulator = nn.Sequential(
            nn.Linear(base_config.consciousness_dim, base_config.consciousness_dim),
            nn.Tanh(),
            nn.Linear(base_config.consciousness_dim, base_config.consciousness_dim)
        )
        
        # Interaction encoder
        self.interaction_encoder = nn.Linear(
            base_config.consciousness_dim * 2,
            base_config.consciousness_dim
        )
        
        # Local memory
        self.local_memory = deque(maxlen=100)
        self.interaction_history = {}
        
    def forward(
        self,
        x: torch.Tensor,
        collective_consciousness: torch.Tensor,
        peer_states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with syntergic interactions."""
        # Modulate consciousness based on agent personality
        agent_consciousness = self.consciousness_modulator(collective_consciousness)
        
        # Incorporate peer interactions
        if peer_states:
            interactions = []
            for peer_state in peer_states:
                interaction = self.interaction_encoder(
                    torch.cat([agent_consciousness, peer_state], dim=-1)
                )
                interactions.append(interaction)
            
            # Aggregate interactions
            interaction_effect = torch.stack(interactions).mean(dim=0)
            agent_consciousness = agent_consciousness + 0.3 * interaction_effect
        
        # Generate predictions
        predictions, diagnostics = self.model(x, agent_consciousness)
        
        # Extract agent state for sharing
        agent_state = diagnostics['consciousness_state'].mean(dim=0)
        
        # Store in local memory
        self.local_memory.append({
            'state': agent_state.detach(),
            'predictions': predictions.detach(),
            'timestamp': len(self.local_memory)
        })
        
        return predictions, agent_state, diagnostics
        
    def compute_coherence_with_peers(self, peer_states: List[torch.Tensor]) -> float:
        """Compute coherence with peer agents."""
        if not peer_states:
            return 1.0
            
        own_state = self.get_current_state()
        coherences = []
        
        for peer_state in peer_states:
            coherence = torch.nn.functional.cosine_similarity(
                own_state.unsqueeze(0),
                peer_state.unsqueeze(0)
            ).item()
            coherences.append(coherence)
            
        return np.mean(coherences)
        
    def get_current_state(self) -> torch.Tensor:
        """Get current agent state."""
        if self.local_memory:
            return self.local_memory[-1]['state']
        return torch.zeros(self.model.config.consciousness_dim)


class SyntergicForecaster(nn.Module):
    """Syntergic forecasting system using collective consciousness."""
    
    def __init__(
        self,
        base_config: ConsciousnessConfig,
        syntergic_config: Optional[SyntergicConfig] = None
    ):
        super().__init__()
        self.base_config = base_config
        self.syntergic_config = syntergic_config or SyntergicConfig()
        
        # Initialize syntergic system
        self.syntergic_system = SyntergicSystem()
        self.consciousness_field = ConsciousnessField()
        
        # Create diverse agents
        self.agents = nn.ModuleList([
            SyntergicAgent(i, base_config)
            for i in range(self.syntergic_config.n_agents)
        ])
        
        # Collective consciousness aggregator
        self.consciousness_aggregator = nn.Sequential(
            nn.Linear(
                base_config.consciousness_dim * self.syntergic_config.n_agents,
                base_config.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(base_config.hidden_size, base_config.consciousness_dim),
            nn.LayerNorm(base_config.consciousness_dim)
        )
        
        # Syntergic fusion network
        self.syntergic_fusion = nn.Sequential(
            nn.Linear(self.syntergic_config.n_agents, base_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_config.hidden_size, base_config.hidden_size),
            nn.ReLU(),
            nn.Linear(base_config.hidden_size, 1)
        )
        
        # Emergence detector
        self.emergence_detector = nn.Sequential(
            nn.Linear(base_config.consciousness_dim, base_config.hidden_size),
            nn.ReLU(),
            nn.Linear(base_config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Collective memory
        self.collective_memory = deque(maxlen=self.syntergic_config.memory_horizon)
        self.emergence_patterns = {}
        
    def synchronize_agents(self, agent_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Synchronize agent states through multiple rounds."""
        synchronized_states = agent_states.copy()
        
        for round_idx in range(self.syntergic_config.synchronization_rounds):
            new_states = []
            
            for i, state in enumerate(synchronized_states):
                # Get peer states (excluding self)
                peer_states = synchronized_states[:i] + synchronized_states[i+1:]
                
                # Compute syntergic influence
                if peer_states:
                    peer_tensor = torch.stack(peer_states)
                    influence = self.syntergic_system.compute_influence(
                        state.unsqueeze(0),
                        peer_tensor
                    )
                    
                    # Apply influence with decay
                    decay = 0.9 ** round_idx
                    new_state = state + decay * influence.squeeze(0)
                else:
                    new_state = state
                    
                new_states.append(new_state)
                
            synchronized_states = new_states
            
        return synchronized_states
        
    def detect_emergence(self, collective_state: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """Detect emergent patterns in collective consciousness."""
        emergence_score = self.emergence_detector(collective_state).item()
        
        # Analyze emergence patterns
        patterns = {
            'score': emergence_score,
            'threshold_exceeded': emergence_score > self.syntergic_config.coherence_threshold,
            'pattern_type': 'unknown'
        }
        
        if emergence_score > 0.8:
            patterns['pattern_type'] = 'strong_coherence'
        elif emergence_score > 0.6:
            patterns['pattern_type'] = 'moderate_coherence'
        elif emergence_score > 0.4:
            patterns['pattern_type'] = 'weak_coherence'
        else:
            patterns['pattern_type'] = 'chaotic'
            
        return emergence_score, patterns
        
    def compute_diversity_bonus(self, predictions: List[torch.Tensor]) -> float:
        """Compute bonus for prediction diversity."""
        if len(predictions) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                dist = torch.nn.functional.mse_loss(
                    predictions[i],
                    predictions[j]
                ).item()
                distances.append(dist)
                
        # Normalize and return bonus
        avg_distance = np.mean(distances)
        return min(1.0, avg_distance * self.syntergic_config.diversity_bonus)
        
    def forward(
        self,
        x: torch.Tensor,
        external_field: Optional[ConsciousnessField] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with syntergic forecasting.
        
        Args:
            x: Input tensor [batch, seq_len, features]
            external_field: External consciousness field
            
        Returns:
            predictions: Syntergic forecast [batch, horizon]
            diagnostics: Comprehensive diagnostic information
        """
        batch_size = x.size(0)
        
        # Initialize collective consciousness
        if external_field:
            self.consciousness_field.synchronize(external_field)
            
        # Phase 1: Individual agent predictions
        agent_predictions = []
        agent_states = []
        agent_diagnostics = []
        
        # Get current collective consciousness
        if self.collective_memory:
            recent_states = [mem['collective_state'] for mem in list(self.collective_memory)[-5:]]
            collective_consciousness = torch.stack(recent_states).mean(dim=0)
        else:
            collective_consciousness = torch.randn(self.base_config.consciousness_dim)
            
        # Generate predictions from each agent
        for i, agent in enumerate(self.agents):
            # Get peer states from previous iteration
            peer_states = agent_states.copy() if i > 0 else []
            
            predictions, state, diagnostics = agent(
                x,
                collective_consciousness,
                peer_states
            )
            
            agent_predictions.append(predictions)
            agent_states.append(state)
            agent_diagnostics.append(diagnostics)
            
        # Phase 2: Syntergic synchronization
        synchronized_states = self.synchronize_agents(agent_states)
        
        # Phase 3: Collective consciousness aggregation
        all_states = torch.cat(synchronized_states, dim=-1)
        new_collective = self.consciousness_aggregator(all_states)
        
        # Phase 4: Emergence detection
        emergence_score, emergence_patterns = self.detect_emergence(new_collective)
        
        # Phase 5: Syntergic fusion of predictions
        predictions_tensor = torch.stack(agent_predictions, dim=-1)
        
        # Apply diversity bonus
        diversity_bonus = self.compute_diversity_bonus(agent_predictions)
        
        # Weighted fusion based on emergence and diversity
        if emergence_score > self.syntergic_config.coherence_threshold:
            # Strong emergence: use collective fusion
            fusion_weights = self.syntergic_fusion(predictions_tensor)
            final_predictions = (predictions_tensor * fusion_weights).sum(dim=-1)
        else:
            # Weak emergence: average with diversity weighting
            weights = torch.ones(self.syntergic_config.n_agents) / self.syntergic_config.n_agents
            weights = weights * (1 + diversity_bonus)
            weights = weights / weights.sum()
            
            final_predictions = (predictions_tensor * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            
        # Update collective memory
        self.collective_memory.append({
            'collective_state': new_collective.detach(),
            'emergence_score': emergence_score,
            'predictions': final_predictions.detach(),
            'timestamp': len(self.collective_memory)
        })
        
        # Store emergence patterns
        if emergence_patterns['threshold_exceeded']:
            pattern_key = f"pattern_{len(self.emergence_patterns)}"
            self.emergence_patterns[pattern_key] = {
                'pattern': emergence_patterns,
                'state': new_collective.detach(),
                'predictions': final_predictions.detach()
            }
            
        # Prepare comprehensive diagnostics
        diagnostics = {
            'agent_predictions': agent_predictions,
            'agent_states': synchronized_states,
            'collective_consciousness': new_collective,
            'emergence_score': emergence_score,
            'emergence_patterns': emergence_patterns,
            'diversity_bonus': diversity_bonus,
            'coherence_scores': [agent.compute_coherence_with_peers(agent_states[:i] + agent_states[i+1:])
                               for i, agent in enumerate(self.agents)],
            'syntergic_energy': self.syntergic_system.compute_energy(
                torch.stack(synchronized_states)
            ),
            'agent_diagnostics': agent_diagnostics
        }
        
        return final_predictions, diagnostics
        
    def adapt_to_feedback(self, feedback: Dict[str, Any]):
        """Adapt syntergic system based on feedback."""
        error = feedback.get('prediction_error', None)
        
        if error is not None:
            # Adjust agent weights based on individual performance
            for i, agent in enumerate(self.agents):
                agent_error = feedback.get(f'agent_{i}_error', error)
                
                # Adapt agent if performing poorly
                if agent_error > error * 1.2:  # 20% worse than average
                    agent.model.adapt_to_consciousness_feedback({
                        'error_signal': agent_error,
                        'consciousness_gradient': torch.randn_like(
                            agent.get_current_state()
                        ) * 0.1
                    })
                    
        # Adjust syntergic parameters
        emergence_feedback = feedback.get('emergence_quality', 0.5)
        self.syntergic_config.coherence_threshold = \
            0.9 * self.syntergic_config.coherence_threshold + 0.1 * emergence_feedback
            
    def get_emergence_insights(self) -> Dict[str, Any]:
        """Get insights about emergent patterns."""
        if not self.emergence_patterns:
            return {'status': 'no_patterns_detected'}
            
        insights = {
            'total_patterns': len(self.emergence_patterns),
            'pattern_types': {},
            'average_emergence_score': 0.0,
            'strongest_pattern': None
        }
        
        scores = []
        for key, pattern_data in self.emergence_patterns.items():
            pattern_type = pattern_data['pattern']['pattern_type']
            insights['pattern_types'][pattern_type] = \
                insights['pattern_types'].get(pattern_type, 0) + 1
            
            score = pattern_data['pattern']['score']
            scores.append(score)
            
            if insights['strongest_pattern'] is None or \
               score > insights['strongest_pattern']['score']:
                insights['strongest_pattern'] = {
                    'key': key,
                    'score': score,
                    'type': pattern_type
                }
                
        insights['average_emergence_score'] = np.mean(scores)
        
        return insights