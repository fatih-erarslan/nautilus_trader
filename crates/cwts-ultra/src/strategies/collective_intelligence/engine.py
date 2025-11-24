"""
PHASE 9 HIVE MIND ORCHESTRATION - COLLECTIVE INTELLIGENCE ENGINE
Advanced multi-agent intelligence aggregation, emergent behavior detection, and swarm optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from abc import ABC, abstractmethod
import networkx as nx
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class IntelligenceContribution:
    """Individual intelligence contribution from an agent"""
    contributor_id: str
    contribution_type: str
    intelligence_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contributor_id': self.contributor_id,
            'contribution_type': self.contribution_type,
            'intelligence_data': self.intelligence_data,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'validation_score': self.validation_score
        }

@dataclass
class EmergentPattern:
    """Emergent pattern detected in collective behavior"""
    pattern_id: str
    pattern_type: str
    contributing_agents: List[str]
    pattern_description: str
    emergence_strength: float
    detection_confidence: float
    first_detected: datetime
    last_observed: datetime
    pattern_data: Dict[str, Any]
    applications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'contributing_agents': self.contributing_agents,
            'pattern_description': self.pattern_description,
            'emergence_strength': self.emergence_strength,
            'detection_confidence': self.detection_confidence,
            'first_detected': self.first_detected.isoformat(),
            'last_observed': self.last_observed.isoformat(),
            'pattern_data': self.pattern_data,
            'applications': self.applications
        }

@dataclass
class CollectiveInsight:
    """Insight generated from collective intelligence analysis"""
    insight_id: str
    insight_category: str
    insight_description: str
    supporting_evidence: List[Dict[str, Any]]
    confidence_level: float
    novelty_score: float
    actionability_score: float
    contributing_agents: List[str]
    generation_timestamp: datetime
    validation_results: Dict[str, Any] = field(default_factory=dict)

class IntelligenceAggregator(ABC):
    """Abstract base class for intelligence aggregation strategies"""
    
    @abstractmethod
    async def aggregate(self, contributions: List[IntelligenceContribution]) -> Dict[str, Any]:
        """Aggregate intelligence contributions"""
        pass

class WeightedConsensusAggregator(IntelligenceAggregator):
    """Weighted consensus aggregation based on confidence and historical performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.weight_decay = config.get('weight_decay', 0.95)
        
    async def aggregate(self, contributions: List[IntelligenceContribution]) -> Dict[str, Any]:
        """Aggregate using weighted consensus approach"""
        
        if not contributions:
            return {'consensus': {}, 'confidence': 0.0, 'method': 'weighted_consensus'}
            
        # Calculate agent weights based on historical performance and current confidence
        agent_weights = {}
        total_weight = 0
        
        for contrib in contributions:
            agent_id = contrib.contributor_id
            
            # Historical performance weight
            historical_scores = list(self.agent_performance_history[agent_id])
            historical_weight = np.mean(historical_scores) if historical_scores else 0.8
            
            # Current confidence weight
            confidence_weight = contrib.confidence_score
            
            # Validation weight
            validation_weight = contrib.validation_score if contrib.validation_score > 0 else 0.8
            
            # Processing efficiency weight (faster is better, but not at the cost of quality)
            efficiency_weight = min(1.0, 2.0 / max(0.1, contrib.processing_time))
            
            # Combined weight
            agent_weight = (historical_weight * 0.4 + confidence_weight * 0.3 + 
                           validation_weight * 0.2 + efficiency_weight * 0.1)
                           
            agent_weights[agent_id] = agent_weight
            total_weight += agent_weight
            
        # Normalize weights
        if total_weight > 0:
            for agent_id in agent_weights:
                agent_weights[agent_id] /= total_weight
                
        # Aggregate intelligence data
        aggregated_data = {}
        confidence_scores = []
        
        # Group contributions by data type
        data_groups = defaultdict(list)
        for contrib in contributions:
            for data_type, data_value in contrib.intelligence_data.items():
                data_groups[data_type].append({
                    'value': data_value,
                    'weight': agent_weights[contrib.contributor_id],
                    'contributor': contrib.contributor_id
                })
                
        # Aggregate each data type
        for data_type, data_items in data_groups.items():
            aggregated_data[data_type] = await self._aggregate_data_type(data_type, data_items)
            
        # Calculate overall confidence
        weighted_confidences = [contrib.confidence_score * agent_weights[contrib.contributor_id] 
                              for contrib in contributions]
        overall_confidence = sum(weighted_confidences)
        confidence_scores.append(overall_confidence)
        
        return {
            'consensus': aggregated_data,
            'confidence': overall_confidence,
            'method': 'weighted_consensus',
            'agent_weights': agent_weights,
            'contributing_agents': [c.contributor_id for c in contributions],
            'aggregation_timestamp': datetime.now().isoformat()
        }
        
    async def _aggregate_data_type(self, data_type: str, data_items: List[Dict[str, Any]]) -> Any:
        """Aggregate specific data type"""
        
        if not data_items:
            return None
            
        # Numerical aggregation
        if all(isinstance(item['value'], (int, float)) for item in data_items):
            weighted_sum = sum(item['value'] * item['weight'] for item in data_items)
            return weighted_sum
            
        # String/categorical aggregation (most common weighted by confidence)
        elif all(isinstance(item['value'], str) for item in data_items):
            value_weights = defaultdict(float)
            for item in data_items:
                value_weights[item['value']] += item['weight']
            return max(value_weights.items(), key=lambda x: x[1])[0]
            
        # List aggregation
        elif all(isinstance(item['value'], list) for item in data_items):
            all_items = []
            for item in data_items:
                # Weight each list item by agent weight
                weighted_items = [(list_item, item['weight']) for list_item in item['value']]
                all_items.extend(weighted_items)
                
            # Return top items by cumulative weight
            item_weights = defaultdict(float)
            for list_item, weight in all_items:
                item_weights[list_item] += weight
                
            sorted_items = sorted(item_weights.items(), key=lambda x: x[1], reverse=True)
            return [item for item, weight in sorted_items[:10]]  # Top 10 items
            
        # Dictionary aggregation
        elif all(isinstance(item['value'], dict) for item in data_items):
            merged_dict = {}
            for item in data_items:
                for key, value in item['value'].items():
                    if key not in merged_dict:
                        merged_dict[key] = []
                    merged_dict[key].append((value, item['weight']))
                    
            # Aggregate each key
            result_dict = {}
            for key, value_weights in merged_dict.items():
                if all(isinstance(vw[0], (int, float)) for vw in value_weights):
                    result_dict[key] = sum(v * w for v, w in value_weights)
                else:
                    # Take most weighted value
                    result_dict[key] = max(value_weights, key=lambda x: x[1])[0]
                    
            return result_dict
            
        else:
            # Fallback: return most weighted value
            return max(data_items, key=lambda x: x['weight'])['value']

class EmergentPatternDetector:
    """Detects emergent patterns in collective agent behavior"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.behavior_history: deque = deque(maxlen=10000)
        self.detected_patterns: Dict[str, EmergentPattern] = {}
        self.pattern_detection_algorithms = {
            'synchronization': self._detect_synchronization_patterns,
            'convergence': self._detect_convergence_patterns,
            'oscillation': self._detect_oscillation_patterns,
            'cascade': self._detect_cascade_patterns,
            'cooperation': self._detect_cooperation_patterns,
            'competition': self._detect_competition_patterns
        }
        
    async def detect_patterns(self, agent_behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect emergent patterns from agent behaviors"""
        
        # Store behavior in history
        behavior_snapshot = {
            'timestamp': datetime.now(),
            'behaviors': agent_behaviors
        }
        self.behavior_history.append(behavior_snapshot)
        
        detected_patterns = []
        
        # Run each pattern detection algorithm
        for pattern_type, detector in self.pattern_detection_algorithms.items():
            try:
                patterns = await detector(agent_behaviors)
                detected_patterns.extend(patterns)
            except Exception as e:
                logging.error(f"Error detecting {pattern_type} patterns: {str(e)}")
                
        # Store newly detected patterns
        for pattern in detected_patterns:
            self.detected_patterns[pattern.pattern_id] = pattern
            
        return detected_patterns
        
    async def _detect_synchronization_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect synchronization patterns between agents"""
        
        patterns = []
        
        if len(behaviors) < 2:
            return patterns
            
        # Analyze temporal synchronization
        agent_timestamps = {}
        for agent_id, behavior in behaviors.items():
            timestamps = behavior.get('activity_timestamps', [])
            if timestamps:
                agent_timestamps[agent_id] = [
                    datetime.fromisoformat(ts) if isinstance(ts, str) else ts 
                    for ts in timestamps[-10:]  # Last 10 activities
                ]
                
        # Find synchronized activities (within 1-second window)
        if len(agent_timestamps) >= 2:
            sync_groups = self._find_synchronized_groups(agent_timestamps, window_seconds=1.0)
            
            for sync_group in sync_groups:
                if len(sync_group) >= 2:
                    sync_strength = len(sync_group) / len(agent_timestamps)
                    
                    if sync_strength > 0.5:  # At least 50% of agents synchronized
                        pattern = EmergentPattern(
                            pattern_id=f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                            pattern_type='synchronization',
                            contributing_agents=sync_group,
                            pattern_description=f"Synchronized activity detected among {len(sync_group)} agents",
                            emergence_strength=sync_strength,
                            detection_confidence=0.8,
                            first_detected=datetime.now(),
                            last_observed=datetime.now(),
                            pattern_data={
                                'sync_window_seconds': 1.0,
                                'synchronized_agents': sync_group,
                                'synchronization_strength': sync_strength
                            }
                        )
                        patterns.append(pattern)
                        
        return patterns
        
    async def _detect_convergence_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect convergence patterns in agent decisions/outputs"""
        
        patterns = []
        
        # Collect agent decisions/outputs
        agent_outputs = {}
        for agent_id, behavior in behaviors.items():
            outputs = behavior.get('recent_outputs', [])
            if outputs:
                agent_outputs[agent_id] = outputs[-5:]  # Last 5 outputs
                
        if len(agent_outputs) < 3:
            return patterns
            
        # Analyze convergence in numerical outputs
        numerical_convergence = await self._analyze_numerical_convergence(agent_outputs)
        if numerical_convergence['convergence_detected']:
            pattern = EmergentPattern(
                pattern_id=f"conv_num_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                pattern_type='convergence',
                contributing_agents=list(agent_outputs.keys()),
                pattern_description=f"Numerical convergence detected with strength {numerical_convergence['strength']:.3f}",
                emergence_strength=numerical_convergence['strength'],
                detection_confidence=0.85,
                first_detected=datetime.now(),
                last_observed=datetime.now(),
                pattern_data={
                    'convergence_type': 'numerical',
                    'convergence_value': numerical_convergence['convergence_value'],
                    'variance': numerical_convergence['variance']
                }
            )
            patterns.append(pattern)
            
        return patterns
        
    async def _detect_oscillation_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect oscillation patterns in agent behaviors"""
        
        patterns = []
        
        # Need sufficient behavioral history
        if len(self.behavior_history) < 10:
            return patterns
            
        # Analyze oscillations in agent performance metrics
        agent_performance_series = defaultdict(list)
        
        for snapshot in list(self.behavior_history)[-20:]:  # Last 20 snapshots
            for agent_id, behavior in snapshot['behaviors'].items():
                performance = behavior.get('performance_metrics', {}).get('success_rate', 0.5)
                agent_performance_series[agent_id].append(performance)
                
        # Detect oscillatory behavior using autocorrelation
        for agent_id, performance_series in agent_performance_series.items():
            if len(performance_series) >= 10:
                oscillation_detected, period, strength = await self._detect_oscillation_in_series(performance_series)
                
                if oscillation_detected and strength > 0.6:
                    pattern = EmergentPattern(
                        pattern_id=f"osc_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        pattern_type='oscillation',
                        contributing_agents=[agent_id],
                        pattern_description=f"Oscillatory behavior detected in agent {agent_id} with period {period}",
                        emergence_strength=strength,
                        detection_confidence=0.75,
                        first_detected=datetime.now() - timedelta(minutes=period * 2),
                        last_observed=datetime.now(),
                        pattern_data={
                            'oscillation_period': period,
                            'oscillation_strength': strength,
                            'agent_id': agent_id
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_cascade_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect cascade patterns where one agent's behavior influences others"""
        
        patterns = []
        
        # Need behavioral history to detect cascades
        if len(self.behavior_history) < 5:
            return patterns
            
        # Look for patterns where one agent's behavior change precedes others
        recent_snapshots = list(self.behavior_history)[-10:]
        
        # Analyze behavior changes
        behavior_changes = []
        for i in range(1, len(recent_snapshots)):
            prev_snapshot = recent_snapshots[i-1]
            curr_snapshot = recent_snapshots[i]
            
            for agent_id in curr_snapshot['behaviors']:
                if agent_id in prev_snapshot['behaviors']:
                    prev_perf = prev_snapshot['behaviors'][agent_id].get('performance_metrics', {}).get('success_rate', 0.5)
                    curr_perf = curr_snapshot['behaviors'][agent_id].get('performance_metrics', {}).get('success_rate', 0.5)
                    
                    change = abs(curr_perf - prev_perf)
                    if change > 0.1:  # Significant change
                        behavior_changes.append({
                            'agent_id': agent_id,
                            'timestamp': curr_snapshot['timestamp'],
                            'change_magnitude': change,
                            'change_direction': 'increase' if curr_perf > prev_perf else 'decrease'
                        })
                        
        # Look for cascade patterns (changes following each other with short delays)
        if len(behavior_changes) >= 3:
            cascades = await self._identify_cascades(behavior_changes)
            
            for cascade in cascades:
                if len(cascade['agents']) >= 3:
                    pattern = EmergentPattern(
                        pattern_id=f"cascade_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        pattern_type='cascade',
                        contributing_agents=cascade['agents'],
                        pattern_description=f"Cascade pattern detected starting from agent {cascade['initiator']}",
                        emergence_strength=cascade['strength'],
                        detection_confidence=0.7,
                        first_detected=cascade['start_time'],
                        last_observed=cascade['end_time'],
                        pattern_data={
                            'cascade_initiator': cascade['initiator'],
                            'cascade_sequence': cascade['sequence'],
                            'cascade_duration': (cascade['end_time'] - cascade['start_time']).total_seconds()
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_cooperation_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect cooperation patterns between agents"""
        
        patterns = []
        
        # Analyze communication patterns and joint successes
        communication_matrix = defaultdict(lambda: defaultdict(int))
        joint_success_matrix = defaultdict(lambda: defaultdict(list))
        
        for agent_id, behavior in behaviors.items():
            communications = behavior.get('communications', [])
            for comm in communications:
                target = comm.get('target_agent')
                success = comm.get('success', False)
                if target:
                    communication_matrix[agent_id][target] += 1
                    joint_success_matrix[agent_id][target].append(success)
                    
        # Identify cooperative pairs/groups
        cooperative_pairs = []
        for agent1, targets in joint_success_matrix.items():
            for agent2, successes in targets.items():
                if len(successes) >= 3:  # Minimum interactions
                    success_rate = sum(successes) / len(successes)
                    if success_rate > 0.8:  # High cooperation success
                        cooperative_pairs.append({
                            'agents': (agent1, agent2),
                            'success_rate': success_rate,
                            'interaction_count': len(successes)
                        })
                        
        # Find cooperative groups (clusters of cooperative pairs)
        if cooperative_pairs:
            cooperation_groups = await self._cluster_cooperative_agents(cooperative_pairs)
            
            for group in cooperation_groups:
                if len(group['agents']) >= 2:
                    pattern = EmergentPattern(
                        pattern_id=f"coop_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        pattern_type='cooperation',
                        contributing_agents=group['agents'],
                        pattern_description=f"Cooperation pattern detected among {len(group['agents'])} agents",
                        emergence_strength=group['cooperation_strength'],
                        detection_confidence=0.8,
                        first_detected=datetime.now(),
                        last_observed=datetime.now(),
                        pattern_data={
                            'cooperation_type': 'communication_based',
                            'average_success_rate': group['average_success_rate'],
                            'total_interactions': group['total_interactions']
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    async def _detect_competition_patterns(self, behaviors: Dict[str, Dict[str, Any]]) -> List[EmergentPattern]:
        """Detect competition patterns between agents"""
        
        patterns = []
        
        # Analyze resource competition and performance rivalry
        agent_resources = {}
        agent_performance = {}
        
        for agent_id, behavior in behaviors.items():
            resources = behavior.get('resource_usage', {})
            performance = behavior.get('performance_metrics', {}).get('success_rate', 0.5)
            
            agent_resources[agent_id] = resources
            agent_performance[agent_id] = performance
            
        # Detect resource competition
        if len(agent_resources) >= 2:
            resource_competition = await self._analyze_resource_competition(agent_resources)
            
            if resource_competition['competition_detected']:
                pattern = EmergentPattern(
                    pattern_id=f"comp_res_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    pattern_type='competition',
                    contributing_agents=resource_competition['competing_agents'],
                    pattern_description=f"Resource competition detected among {len(resource_competition['competing_agents'])} agents",
                    emergence_strength=resource_competition['competition_intensity'],
                    detection_confidence=0.75,
                    first_detected=datetime.now(),
                    last_observed=datetime.now(),
                    pattern_data={
                        'competition_type': 'resource_based',
                        'contested_resources': resource_competition['contested_resources'],
                        'competition_metrics': resource_competition['metrics']
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    def _find_synchronized_groups(self, agent_timestamps: Dict[str, List[datetime]], 
                                window_seconds: float) -> List[List[str]]:
        """Find groups of agents with synchronized activities"""
        
        sync_groups = []
        
        # Create time windows and find agents active in each window
        all_times = []
        for timestamps in agent_timestamps.values():
            all_times.extend(timestamps)
            
        if not all_times:
            return sync_groups
            
        all_times.sort()
        
        # Create sliding windows
        window_delta = timedelta(seconds=window_seconds)
        
        for i, base_time in enumerate(all_times):
            window_end = base_time + window_delta
            
            # Find agents active in this window
            active_agents = []
            for agent_id, timestamps in agent_timestamps.items():
                for ts in timestamps:
                    if base_time <= ts <= window_end:
                        active_agents.append(agent_id)
                        break
                        
            if len(active_agents) >= 2:
                # Check if this is a new sync group
                active_agents_set = set(active_agents)
                is_new_group = True
                
                for existing_group in sync_groups:
                    if set(existing_group) == active_agents_set:
                        is_new_group = False
                        break
                        
                if is_new_group:
                    sync_groups.append(active_agents)
                    
        return sync_groups
        
    async def _analyze_numerical_convergence(self, agent_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze numerical convergence in agent outputs"""
        
        # Extract numerical values
        numerical_outputs = {}
        for agent_id, outputs in agent_outputs.items():
            numerical_values = []
            for output in outputs:
                if isinstance(output, (int, float)):
                    numerical_values.append(output)
                elif isinstance(output, dict):
                    # Extract numerical values from dict
                    for value in output.values():
                        if isinstance(value, (int, float)):
                            numerical_values.append(value)
                            
            if numerical_values:
                numerical_outputs[agent_id] = numerical_values
                
        if len(numerical_outputs) < 3:
            return {'convergence_detected': False}
            
        # Calculate convergence metrics
        latest_values = []
        for agent_id, values in numerical_outputs.items():
            if values:
                latest_values.append(values[-1])  # Most recent value
                
        if len(latest_values) < 3:
            return {'convergence_detected': False}
            
        # Calculate variance (low variance indicates convergence)
        variance = np.var(latest_values)
        mean_value = np.mean(latest_values)
        
        # Normalized variance (coefficient of variation)
        cv = variance / (abs(mean_value) + 0.001)
        
        # Convergence detected if coefficient of variation is low
        convergence_detected = cv < 0.1  # 10% coefficient of variation threshold
        convergence_strength = max(0.0, 1.0 - cv)
        
        return {
            'convergence_detected': convergence_detected,
            'strength': convergence_strength,
            'convergence_value': mean_value,
            'variance': variance,
            'coefficient_of_variation': cv
        }
        
    async def _detect_oscillation_in_series(self, series: List[float]) -> Tuple[bool, int, float]:
        """Detect oscillation in a time series using autocorrelation"""
        
        if len(series) < 6:
            return False, 0, 0.0
            
        # Calculate autocorrelation
        series_array = np.array(series)
        series_centered = series_array - np.mean(series_array)
        
        autocorr = np.correlate(series_centered, series_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return False, 0, 0.0
            
        # Find peaks in autocorrelation (indicating periodicity)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append((i, autocorr[i]))
                
        if peaks:
            # Find the most significant peak
            best_peak = max(peaks, key=lambda x: x[1])
            period = best_peak[0]
            strength = best_peak[1]
            
            return strength > 0.5, period, strength
        else:
            return False, 0, 0.0
            
    async def _identify_cascades(self, behavior_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cascade patterns from behavior changes"""
        
        cascades = []
        
        # Sort changes by timestamp
        sorted_changes = sorted(behavior_changes, key=lambda x: x['timestamp'])
        
        # Look for sequences of changes
        for i, initial_change in enumerate(sorted_changes):
            cascade_sequence = [initial_change]
            last_timestamp = initial_change['timestamp']
            
            # Look for following changes within time window
            for j in range(i + 1, len(sorted_changes)):
                next_change = sorted_changes[j]
                time_diff = (next_change['timestamp'] - last_timestamp).total_seconds()
                
                if time_diff <= 60:  # Within 1 minute
                    cascade_sequence.append(next_change)
                    last_timestamp = next_change['timestamp']
                elif time_diff > 300:  # More than 5 minutes - stop looking
                    break
                    
            # A cascade needs at least 3 participants
            if len(cascade_sequence) >= 3:
                agents = [change['agent_id'] for change in cascade_sequence]
                strength = len(cascade_sequence) / len(behavior_changes)
                
                cascades.append({
                    'agents': agents,
                    'sequence': cascade_sequence,
                    'initiator': cascade_sequence[0]['agent_id'],
                    'start_time': cascade_sequence[0]['timestamp'],
                    'end_time': cascade_sequence[-1]['timestamp'],
                    'strength': strength
                })
                
        return cascades
        
    async def _cluster_cooperative_agents(self, cooperative_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster cooperative agents into groups"""
        
        # Build cooperation graph
        G = nx.Graph()
        
        for pair in cooperative_pairs:
            agent1, agent2 = pair['agents']
            weight = pair['success_rate'] * pair['interaction_count']
            G.add_edge(agent1, agent2, weight=weight)
            
        # Find connected components (cooperation groups)
        cooperation_groups = []
        
        for component in nx.connected_components(G):
            if len(component) >= 2:
                # Calculate group statistics
                total_interactions = 0
                total_success = 0
                
                for agent1, agent2 in G.subgraph(component).edges():
                    for pair in cooperative_pairs:
                        if set(pair['agents']) == {agent1, agent2}:
                            total_interactions += pair['interaction_count']
                            total_success += pair['success_rate'] * pair['interaction_count']
                            break
                            
                avg_success_rate = total_success / max(1, total_interactions)
                cooperation_strength = min(1.0, total_interactions / (len(component) * 10))
                
                cooperation_groups.append({
                    'agents': list(component),
                    'cooperation_strength': cooperation_strength,
                    'average_success_rate': avg_success_rate,
                    'total_interactions': total_interactions
                })
                
        return cooperation_groups
        
    async def _analyze_resource_competition(self, agent_resources: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource competition between agents"""
        
        # Identify contested resources
        resource_usage = defaultdict(list)
        
        for agent_id, resources in agent_resources.items():
            for resource_type, usage in resources.items():
                if isinstance(usage, (int, float)) and usage > 0:
                    resource_usage[resource_type].append((agent_id, usage))
                    
        # Find resources with high competition
        contested_resources = []
        competing_agents = set()
        
        for resource_type, usage_list in resource_usage.items():
            if len(usage_list) >= 2:  # At least 2 agents using this resource
                total_usage = sum(usage for _, usage in usage_list)
                
                # High usage indicates competition
                if total_usage > 0.8:  # Assuming normalized resource usage
                    contested_resources.append(resource_type)
                    competing_agents.update(agent_id for agent_id, _ in usage_list)
                    
        competition_intensity = len(contested_resources) / max(1, len(resource_usage))
        competition_detected = competition_intensity > 0.3
        
        return {
            'competition_detected': competition_detected,
            'competition_intensity': competition_intensity,
            'competing_agents': list(competing_agents),
            'contested_resources': contested_resources,
            'metrics': {
                'total_resources': len(resource_usage),
                'contested_resources': len(contested_resources),
                'competing_agent_count': len(competing_agents)
            }
        }

class CollectiveIntelligenceEngine:
    """
    Main engine for collective intelligence processing and optimization
    Coordinates intelligence aggregation, pattern detection, and insight generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.aggregator = WeightedConsensusAggregator(config.get('aggregation', {}))
        self.pattern_detector = EmergentPatternDetector(config.get('pattern_detection', {}))
        
        # Intelligence processing
        self.intelligence_buffer: deque = deque(maxlen=10000)
        self.collective_insights: Dict[str, CollectiveInsight] = {}
        self.processing_lock = threading.RLock()
        
        # Optimization metrics
        self.optimization_history: deque = deque(maxlen=1000)
        self.performance_baseline = 0.7
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for collective intelligence engine"""
        logger = logging.getLogger('CollectiveIntelligenceEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def analyze_inputs(self, agent_inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze inputs from multiple agents and generate collective intelligence"""
        
        try:
            # Convert agent inputs to intelligence contributions
            contributions = []
            
            for agent_id, input_data in agent_inputs.items():
                contribution = IntelligenceContribution(
                    contributor_id=agent_id,
                    contribution_type=input_data.get('contribution_type', 'analysis'),
                    intelligence_data=input_data,
                    confidence_score=input_data.get('confidence', 0.8),
                    processing_time=input_data.get('processing_time', 0.5),
                    timestamp=datetime.now(),
                    metadata={
                        'agent_type': input_data.get('agent_type', 'unknown'),
                        'capabilities': input_data.get('capabilities', [])
                    }
                )
                contributions.append(contribution)
                
            # Store contributions
            with self.processing_lock:
                self.intelligence_buffer.extend(contributions)
                
            # Aggregate intelligence
            aggregation_result = await self.aggregator.aggregate(contributions)
            
            # Detect emergent patterns
            agent_behaviors = {
                agent_id: input_data for agent_id, input_data in agent_inputs.items()
            }
            emergent_patterns = await self.pattern_detector.detect_patterns(agent_behaviors)
            
            # Generate collective insights
            insights = await self._generate_collective_insights(
                aggregation_result, emergent_patterns, contributions
            )
            
            # Optimize collective performance
            optimization_recommendations = await self._optimize_collective_performance(
                aggregation_result, emergent_patterns
            )
            
            # Compile final analysis
            analysis_result = {
                'aggregation_result': aggregation_result,
                'emergent_patterns': [pattern.to_dict() for pattern in emergent_patterns],
                'collective_insights': [asdict(insight) for insight in insights],
                'optimization_recommendations': optimization_recommendations,
                'analysis_metadata': {
                    'contributing_agents': len(agent_inputs),
                    'total_contributions': len(contributions),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_quality': await self._assess_processing_quality(aggregation_result)
                }
            }
            
            self.logger.info(f"Analyzed inputs from {len(agent_inputs)} agents, detected {len(emergent_patterns)} patterns")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze agent inputs: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
    async def _generate_collective_insights(self, 
                                          aggregation_result: Dict[str, Any],
                                          emergent_patterns: List[EmergentPattern],
                                          contributions: List[IntelligenceContribution]) -> List[CollectiveInsight]:
        """Generate collective insights from aggregated intelligence and patterns"""
        
        insights = []
        
        try:
            # Insight from aggregation consensus
            consensus_data = aggregation_result.get('consensus', {})
            if consensus_data:
                consensus_insight = await self._create_consensus_insight(aggregation_result, contributions)
                if consensus_insight:
                    insights.append(consensus_insight)
                    
            # Insights from emergent patterns
            for pattern in emergent_patterns:
                pattern_insight = await self._create_pattern_insight(pattern)
                if pattern_insight:
                    insights.append(pattern_insight)
                    
            # Performance optimization insights
            performance_insight = await self._create_performance_insight(contributions)
            if performance_insight:
                insights.append(performance_insight)
                
            # Store insights
            for insight in insights:
                self.collective_insights[insight.insight_id] = insight
                
        except Exception as e:
            self.logger.error(f"Failed to generate collective insights: {str(e)}")
            
        return insights
        
    async def _create_consensus_insight(self, aggregation_result: Dict[str, Any], 
                                      contributions: List[IntelligenceContribution]) -> Optional[CollectiveInsight]:
        """Create insight from consensus aggregation"""
        
        try:
            consensus = aggregation_result.get('consensus', {})
            confidence = aggregation_result.get('confidence', 0.5)
            
            if not consensus or confidence < 0.6:
                return None
                
            # Analyze consensus for actionable insights
            actionable_items = []
            for key, value in consensus.items():
                if isinstance(value, list) and len(value) > 0:
                    actionable_items.extend(value[:3])  # Top 3 items
                elif isinstance(value, str) and len(value) > 0:
                    actionable_items.append(value)
                    
            if not actionable_items:
                return None
                
            insight = CollectiveInsight(
                insight_id=f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                insight_category='consensus_analysis',
                insight_description=f"Collective consensus achieved with {confidence:.1%} confidence across {len(contributions)} agents",
                supporting_evidence=[
                    {
                        'type': 'aggregation_result',
                        'data': aggregation_result,
                        'confidence': confidence
                    }
                ],
                confidence_level=confidence,
                novelty_score=await self._calculate_novelty_score(consensus),
                actionability_score=min(1.0, len(actionable_items) / 5.0),
                contributing_agents=[c.contributor_id for c in contributions],
                generation_timestamp=datetime.now()
            )
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Failed to create consensus insight: {str(e)}")
            return None
            
    async def _create_pattern_insight(self, pattern: EmergentPattern) -> Optional[CollectiveInsight]:
        """Create insight from emergent pattern"""
        
        try:
            # Generate insight based on pattern type
            insight_descriptions = {
                'synchronization': f"Agents are exhibiting synchronized behavior with {pattern.emergence_strength:.1%} coordination",
                'convergence': f"Agent decisions are converging with {pattern.emergence_strength:.1%} alignment",
                'oscillation': f"Oscillatory behavior detected with {pattern.emergence_strength:.1%} regularity",
                'cascade': f"Cascade pattern shows influence propagation across {len(pattern.contributing_agents)} agents",
                'cooperation': f"Cooperative behavior emerging among {len(pattern.contributing_agents)} agents",
                'competition': f"Competitive dynamics detected with {pattern.emergence_strength:.1%} intensity"
            }
            
            description = insight_descriptions.get(
                pattern.pattern_type, 
                f"Emergent {pattern.pattern_type} pattern detected"
            )
            
            # Calculate actionability based on pattern type
            actionability_scores = {
                'cooperation': 0.9,
                'competition': 0.7,
                'convergence': 0.8,
                'synchronization': 0.6,
                'cascade': 0.8,
                'oscillation': 0.5
            }
            
            insight = CollectiveInsight(
                insight_id=f"pattern_{pattern.pattern_id}",
                insight_category='emergent_behavior',
                insight_description=description,
                supporting_evidence=[
                    {
                        'type': 'emergent_pattern',
                        'pattern_data': pattern.to_dict(),
                        'detection_confidence': pattern.detection_confidence
                    }
                ],
                confidence_level=pattern.detection_confidence,
                novelty_score=pattern.emergence_strength,
                actionability_score=actionability_scores.get(pattern.pattern_type, 0.6),
                contributing_agents=pattern.contributing_agents,
                generation_timestamp=datetime.now()
            )
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Failed to create pattern insight: {str(e)}")
            return None
            
    async def _create_performance_insight(self, contributions: List[IntelligenceContribution]) -> Optional[CollectiveInsight]:
        """Create insight from performance analysis"""
        
        try:
            if not contributions:
                return None
                
            # Analyze performance metrics
            confidence_scores = [c.confidence_score for c in contributions]
            processing_times = [c.processing_time for c in contributions]
            
            avg_confidence = np.mean(confidence_scores)
            avg_processing_time = np.mean(processing_times)
            
            # Performance assessment
            performance_level = 'optimal' if avg_confidence > 0.8 else 'good' if avg_confidence > 0.6 else 'suboptimal'
            
            # Generate recommendations
            recommendations = []
            if avg_confidence < 0.7:
                recommendations.append("Consider improving agent training or validation processes")
            if avg_processing_time > 2.0:
                recommendations.append("Optimize processing algorithms for better performance")
            if len(set(c.contributor_id for c in contributions)) < 0.7 * len(contributions):
                recommendations.append("Increase agent participation diversity")
                
            insight = CollectiveInsight(
                insight_id=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                insight_category='performance_analysis',
                insight_description=f"System performance is {performance_level} with {avg_confidence:.1%} average confidence",
                supporting_evidence=[
                    {
                        'type': 'performance_metrics',
                        'average_confidence': avg_confidence,
                        'average_processing_time': avg_processing_time,
                        'total_contributors': len(contributions)
                    }
                ],
                confidence_level=0.9,  # High confidence in performance metrics
                novelty_score=0.3,  # Performance insights are typically not very novel
                actionability_score=0.8 if recommendations else 0.4,
                contributing_agents=list(set(c.contributor_id for c in contributions)),
                generation_timestamp=datetime.now()
            )
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Failed to create performance insight: {str(e)}")
            return None
            
    async def _optimize_collective_performance(self, aggregation_result: Dict[str, Any], 
                                             emergent_patterns: List[EmergentPattern]) -> Dict[str, Any]:
        """Generate optimization recommendations for collective performance"""
        
        try:
            recommendations = {
                'coordination_optimizations': [],
                'performance_improvements': [],
                'pattern_leveraging': [],
                'risk_mitigations': []
            }
            
            # Analyze aggregation quality
            confidence = aggregation_result.get('confidence', 0.5)
            if confidence < 0.7:
                recommendations['performance_improvements'].append({
                    'priority': 'high',
                    'action': 'improve_agent_confidence',
                    'description': 'Enhance agent training and validation processes',
                    'expected_impact': 0.2
                })
                
            # Analyze agent weight distribution
            agent_weights = aggregation_result.get('agent_weights', {})
            if agent_weights:
                weight_variance = np.var(list(agent_weights.values()))
                if weight_variance > 0.1:
                    recommendations['coordination_optimizations'].append({
                        'priority': 'medium',
                        'action': 'balance_agent_contributions',
                        'description': 'Rebalance agent workloads and capabilities',
                        'expected_impact': 0.15
                    })
                    
            # Leverage positive emergent patterns
            for pattern in emergent_patterns:
                if pattern.pattern_type in ['cooperation', 'synchronization', 'convergence']:
                    recommendations['pattern_leveraging'].append({
                        'priority': 'medium',
                        'action': f'amplify_{pattern.pattern_type}',
                        'description': f'Strengthen and expand {pattern.pattern_type} behavior',
                        'pattern_id': pattern.pattern_id,
                        'expected_impact': pattern.emergence_strength * 0.3
                    })
                elif pattern.pattern_type in ['competition', 'oscillation']:
                    recommendations['risk_mitigations'].append({
                        'priority': 'high' if pattern.emergence_strength > 0.7 else 'medium',
                        'action': f'mitigate_{pattern.pattern_type}',
                        'description': f'Address potentially disruptive {pattern.pattern_type} behavior',
                        'pattern_id': pattern.pattern_id,
                        'expected_impact': 0.2
                    })
                    
            # Overall system optimization
            optimization_score = confidence * 0.5 + (1.0 - len(recommendations['risk_mitigations']) * 0.1) * 0.5
            
            return {
                'recommendations': recommendations,
                'optimization_score': optimization_score,
                'priority_actions': self._prioritize_recommendations(recommendations),
                'expected_improvement': sum(rec.get('expected_impact', 0) 
                                          for rec_list in recommendations.values() 
                                          for rec in rec_list)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize collective performance: {str(e)}")
            return {'error': str(e)}
            
    def _prioritize_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact and priority"""
        
        all_recommendations = []
        for category, rec_list in recommendations.items():
            for rec in rec_list:
                rec['category'] = category
                all_recommendations.append(rec)
                
        # Sort by priority (high > medium > low) and then by expected impact
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        sorted_recommendations = sorted(
            all_recommendations,
            key=lambda x: (priority_order.get(x.get('priority', 'low'), 1), x.get('expected_impact', 0)),
            reverse=True
        )
        
        return sorted_recommendations[:10]  # Top 10 priority actions
        
    async def _calculate_novelty_score(self, consensus_data: Dict[str, Any]) -> float:
        """Calculate novelty score for consensus data"""
        
        try:
            # Simple novelty calculation based on historical data
            # In a real implementation, this would compare against historical patterns
            
            # For now, use a heuristic based on data complexity and uniqueness
            complexity_score = min(1.0, len(str(consensus_data)) / 1000)
            uniqueness_score = 0.7  # Placeholder - would be calculated from historical comparison
            
            novelty_score = (complexity_score * 0.4 + uniqueness_score * 0.6)
            return min(1.0, max(0.0, novelty_score))
            
        except Exception:
            return 0.5  # Default moderate novelty
            
    async def _assess_processing_quality(self, aggregation_result: Dict[str, Any]) -> float:
        """Assess the quality of intelligence processing"""
        
        try:
            confidence = aggregation_result.get('confidence', 0.5)
            consensus = aggregation_result.get('consensus', {})
            
            # Quality factors
            confidence_quality = confidence
            data_completeness = min(1.0, len(consensus) / 10.0)  # Assuming 10 ideal data points
            consistency_quality = 0.8  # Would be calculated from data consistency checks
            
            overall_quality = (confidence_quality * 0.4 + data_completeness * 0.3 + consistency_quality * 0.3)
            
            return min(1.0, max(0.0, overall_quality))
            
        except Exception:
            return 0.6  # Default moderate quality
            
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive status of collective intelligence system"""
        
        with self.processing_lock:
            recent_contributions = [c for c in self.intelligence_buffer 
                                  if (datetime.now() - c.timestamp).total_seconds() < 3600]  # Last hour
                                  
            return {
                'total_contributions': len(self.intelligence_buffer),
                'recent_contributions': len(recent_contributions),
                'active_insights': len(self.collective_insights),
                'detected_patterns': len(self.pattern_detector.detected_patterns),
                'average_confidence': np.mean([c.confidence_score for c in recent_contributions]) if recent_contributions else 0,
                'processing_efficiency': np.mean([c.processing_time for c in recent_contributions]) if recent_contributions else 0,
                'unique_contributors': len(set(c.contributor_id for c in recent_contributions)),
                'intelligence_quality': np.mean([c.validation_score for c in recent_contributions if c.validation_score > 0]) if recent_contributions else 0,
                'system_health': min(1.0, len(recent_contributions) / 50.0),  # Assuming 50 contributions/hour is healthy
                'last_updated': datetime.now().isoformat()
            }