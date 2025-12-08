//! Emergent Intellect
//!
//! Records learned behaviors, builds knowledge graphs, and enables
//! the swarm to accumulate and transfer intelligence across runs.
//!
//! ## Concepts
//!
//! - **IntellectRecord**: Captured knowledge from optimization runs
//! - **KnowledgeGraph**: Network of concepts, strategies, and relationships
//! - **Insight**: Extracted patterns and rules
//! - **EmergentIntellect**: The accumulated intelligence of the swarm

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{SwarmResult, SwarmIntelligenceError};
use crate::strategy::StrategyType;
use crate::topology::TopologyType;
use crate::evolution::Genome;

/// A single record of learned behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntellectRecord {
    /// Unique identifier
    pub id: Uuid,
    /// Problem signature (characterizes the problem type)
    pub problem_signature: ProblemSignature,
    /// Strategy that performed best
    pub best_strategy: StrategyType,
    /// Topology that performed best
    pub best_topology: TopologyType,
    /// Optimal parameters discovered
    pub optimal_parameters: HashMap<String, f64>,
    /// Final fitness achieved
    pub best_fitness: f64,
    /// Convergence characteristics
    pub convergence_profile: ConvergenceProfile,
    /// Genome used (if evolved)
    pub genome: Option<Genome>,
    /// Insights extracted
    pub insights: Vec<Insight>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Number of function evaluations
    pub evaluations: usize,
    /// Tags for categorization
    pub tags: HashSet<String>,
}

/// Problem signature for matching similar problems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSignature {
    /// Problem dimensionality
    pub dimensions: usize,
    /// Search space volume (product of bound ranges)
    pub search_volume: f64,
    /// Estimated modality (unimodal, multimodal, highly multimodal)
    pub modality: Modality,
    /// Separability estimate
    pub separability: f64,
    /// Conditioning estimate (ratio of curvatures)
    pub conditioning: f64,
    /// Noise level
    pub noise_level: f64,
    /// Custom features
    pub features: HashMap<String, f64>,
}

impl ProblemSignature {
    /// Compute similarity to another signature
    pub fn similarity(&self, other: &ProblemSignature) -> f64 {
        let mut score = 0.0;
        let mut weights = 0.0;
        
        // Dimension similarity
        let dim_ratio = (self.dimensions as f64).min(other.dimensions as f64) 
            / (self.dimensions as f64).max(other.dimensions as f64);
        score += dim_ratio;
        weights += 1.0;
        
        // Volume similarity (log scale)
        let vol_ratio = self.search_volume.ln().abs().min(other.search_volume.ln().abs())
            / self.search_volume.ln().abs().max(other.search_volume.ln().abs());
        score += vol_ratio;
        weights += 1.0;
        
        // Modality match
        if self.modality == other.modality {
            score += 1.0;
        }
        weights += 1.0;
        
        // Separability similarity
        score += 1.0 - (self.separability - other.separability).abs();
        weights += 1.0;
        
        score / weights
    }
}

/// Problem modality classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Unimodal,
    FewModes,
    Multimodal,
    HighlyMultimodal,
    Unknown,
}

/// Convergence profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceProfile {
    /// Iterations to reach 50% of final fitness
    pub half_life: usize,
    /// Iterations to reach 90% of final fitness
    pub ninety_percent: usize,
    /// Was convergence monotonic?
    pub monotonic: bool,
    /// Number of stagnation periods
    pub stagnation_count: usize,
    /// Average improvement rate
    pub avg_improvement_rate: f64,
    /// Final improvement rate
    pub final_improvement_rate: f64,
}

/// An insight extracted from optimization runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Insight {
    /// Unique identifier
    pub id: Uuid,
    /// Type of insight
    pub insight_type: InsightType,
    /// Natural language description
    pub description: String,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Supporting evidence count
    pub evidence_count: usize,
    /// Related strategies
    pub related_strategies: Vec<StrategyType>,
    /// Related problem features
    pub related_features: HashMap<String, f64>,
    /// Creation time
    pub created_at: DateTime<Utc>,
}

/// Types of insights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Strategy works well for problem type
    StrategyAffinity,
    /// Parameter range is optimal
    ParameterRange,
    /// Topology enhances performance
    TopologyBenefit,
    /// Strategy combination is synergistic
    StrategySynergy,
    /// Problem transformation helps
    ProblemTransformation,
    /// Convergence pattern
    ConvergencePattern,
    /// Failure mode identified
    FailureMode,
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Unique identifier
    pub id: Uuid,
    /// Node type
    pub node_type: NodeType,
    /// Node label
    pub label: String,
    /// Properties
    pub properties: HashMap<String, f64>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Usage count
    pub usage_count: usize,
    /// Success rate when used
    pub success_rate: f64,
}

/// Types of knowledge nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Strategy,
    Topology,
    Parameter,
    ProblemClass,
    Insight,
    Genome,
}

/// An edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    /// Source node ID
    pub from: Uuid,
    /// Target node ID
    pub to: Uuid,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge weight/strength
    pub weight: f64,
    /// Evidence count
    pub evidence_count: usize,
}

/// Types of knowledge edges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Works well together
    Synergy,
    /// One is better than other for context
    Dominates,
    /// Are similar in behavior
    Similar,
    /// One requires the other
    Requires,
    /// One is a variant of another
    Variant,
    /// Are antagonistic
    Conflict,
}

/// Knowledge graph for storing swarm intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All nodes
    nodes: HashMap<Uuid, KnowledgeNode>,
    /// Edges (adjacency list)
    edges: HashMap<Uuid, Vec<KnowledgeEdge>>,
    /// Index by node type
    type_index: HashMap<NodeType, HashSet<Uuid>>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            type_index: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
    
    /// Add a node
    pub fn add_node(&mut self, node: KnowledgeNode) {
        self.type_index
            .entry(node.node_type)
            .or_default()
            .insert(node.id);
        self.nodes.insert(node.id, node);
        self.updated_at = Utc::now();
    }
    
    /// Add an edge
    pub fn add_edge(&mut self, edge: KnowledgeEdge) {
        self.edges.entry(edge.from).or_default().push(edge);
        self.updated_at = Utc::now();
    }
    
    /// Get node by ID
    pub fn get_node(&self, id: Uuid) -> Option<&KnowledgeNode> {
        self.nodes.get(&id)
    }
    
    /// Get nodes by type
    pub fn get_nodes_by_type(&self, node_type: NodeType) -> Vec<&KnowledgeNode> {
        self.type_index
            .get(&node_type)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }
    
    /// Get edges from a node
    pub fn get_edges(&self, from: Uuid) -> &[KnowledgeEdge] {
        self.edges.get(&from).map(|e| e.as_slice()).unwrap_or(&[])
    }
    
    /// Find related nodes
    pub fn find_related(&self, node_id: Uuid, edge_types: &[EdgeType]) -> Vec<&KnowledgeNode> {
        let edges = self.get_edges(node_id);
        edges.iter()
            .filter(|e| edge_types.contains(&e.edge_type))
            .filter_map(|e| self.nodes.get(&e.to))
            .collect()
    }
    
    /// Get strongest connections
    pub fn strongest_connections(&self, node_id: Uuid, limit: usize) -> Vec<(&KnowledgeNode, f64)> {
        let mut edges: Vec<_> = self.get_edges(node_id).iter().collect();
        edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        
        edges.into_iter()
            .take(limit)
            .filter_map(|e| self.nodes.get(&e.to).map(|n| (n, e.weight)))
            .collect()
    }
    
    /// Compute PageRank-like importance scores
    pub fn compute_importance(&self) -> HashMap<Uuid, f64> {
        let damping = 0.85;
        let iterations = 20;
        let n = self.nodes.len() as f64;
        
        let mut scores: HashMap<Uuid, f64> = self.nodes.keys()
            .map(|id| (*id, 1.0 / n))
            .collect();
        
        for _ in 0..iterations {
            let mut new_scores = HashMap::new();
            
            for (&id, _) in &self.nodes {
                let mut incoming = 0.0;
                
                // Find edges pointing to this node
                for (from_id, edges) in &self.edges {
                    let out_degree = edges.len() as f64;
                    for edge in edges {
                        if edge.to == id {
                            incoming += scores.get(from_id).unwrap_or(&0.0) / out_degree;
                        }
                    }
                }
                
                new_scores.insert(id, (1.0 - damping) / n + damping * incoming);
            }
            
            scores = new_scores;
        }
        
        scores
    }
    
    /// Node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Edge count
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }
}

/// The emergent intellect system
pub struct EmergentIntellect {
    /// Knowledge graph
    pub graph: KnowledgeGraph,
    /// Historical records
    records: VecDeque<IntellectRecord>,
    /// Maximum records to keep
    max_records: usize,
    /// Extracted insights
    insights: Vec<Insight>,
    /// Strategy performance statistics
    strategy_stats: HashMap<StrategyType, StrategyStats>,
    /// Problem class models
    problem_models: HashMap<String, ProblemModel>,
}

/// Statistics for a strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyStats {
    pub usage_count: usize,
    pub success_count: usize,
    pub total_fitness: f64,
    pub best_fitness: f64,
    pub avg_convergence_time: f64,
    pub problem_affinities: HashMap<Modality, f64>,
}

/// Model for a problem class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemModel {
    pub name: String,
    pub signature_template: ProblemSignature,
    pub recommended_strategies: Vec<(StrategyType, f64)>,
    pub recommended_topologies: Vec<(TopologyType, f64)>,
    pub parameter_ranges: HashMap<String, (f64, f64)>,
    pub sample_count: usize,
}

impl EmergentIntellect {
    /// Create new emergent intellect
    pub fn new() -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            records: VecDeque::new(),
            max_records: 10000,
            insights: Vec::new(),
            strategy_stats: HashMap::new(),
            problem_models: HashMap::new(),
        }
    }
    
    /// Record a learning experience
    pub fn record(&mut self, record: IntellectRecord) {
        // Update strategy stats
        let stats = self.strategy_stats.entry(record.best_strategy).or_default();
        stats.usage_count += 1;
        stats.total_fitness += record.best_fitness;
        if record.best_fitness < stats.best_fitness || stats.best_fitness == 0.0 {
            stats.best_fitness = record.best_fitness;
        }
        stats.success_count += if record.best_fitness < 1.0 { 1 } else { 0 };
        
        let affinity = stats.problem_affinities
            .entry(record.problem_signature.modality)
            .or_insert(0.0);
        *affinity = (*affinity * (stats.usage_count - 1) as f64 
            + (1.0 / (1.0 + record.best_fitness))) / stats.usage_count as f64;
        
        // Add to knowledge graph
        self.add_to_graph(&record);
        
        // Store record
        self.records.push_back(record);
        while self.records.len() > self.max_records {
            self.records.pop_front();
        }
        
        // Extract insights periodically
        if self.records.len() % 100 == 0 {
            self.extract_insights();
        }
    }
    
    /// Add record to knowledge graph
    fn add_to_graph(&mut self, record: &IntellectRecord) {
        // Add strategy node if not exists
        let strategy_label = format!("{:?}", record.best_strategy);
        let strategy_node_id = self.find_or_create_node(
            NodeType::Strategy,
            &strategy_label,
        );
        
        // Add topology node
        let topology_label = format!("{:?}", record.best_topology);
        let topology_node_id = self.find_or_create_node(
            NodeType::Topology,
            &topology_label,
        );
        
        // Add synergy edge between strategy and topology
        if let (Some(sid), Some(tid)) = (strategy_node_id, topology_node_id) {
            // Check if edge exists
            let existing = self.graph.get_edges(sid)
                .iter()
                .any(|e| e.to == tid && e.edge_type == EdgeType::Synergy);
            
            if !existing {
                self.graph.add_edge(KnowledgeEdge {
                    from: sid,
                    to: tid,
                    edge_type: EdgeType::Synergy,
                    weight: 1.0,
                    evidence_count: 1,
                });
            }
        }
    }
    
    /// Find or create a node
    fn find_or_create_node(&mut self, node_type: NodeType, label: &str) -> Option<Uuid> {
        // Check if exists
        for node in self.graph.get_nodes_by_type(node_type) {
            if node.label == label {
                return Some(node.id);
            }
        }
        
        // Create new
        let node = KnowledgeNode {
            id: Uuid::new_v4(),
            node_type,
            label: label.to_string(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            usage_count: 1,
            success_rate: 0.0,
        };
        
        let id = node.id;
        self.graph.add_node(node);
        Some(id)
    }
    
    /// Extract insights from accumulated records
    fn extract_insights(&mut self) {
        // Strategy affinity insights
        for (strategy, stats) in &self.strategy_stats {
            if stats.usage_count >= 10 {
                let success_rate = stats.success_count as f64 / stats.usage_count as f64;
                
                if success_rate > 0.7 {
                    // Find which problem types this strategy excels at
                    for (modality, &affinity) in &stats.problem_affinities {
                        if affinity > 0.6 {
                            let insight = Insight {
                                id: Uuid::new_v4(),
                                insight_type: InsightType::StrategyAffinity,
                                description: format!(
                                    "{:?} shows strong affinity ({:.1}%) for {:?} problems",
                                    strategy, affinity * 100.0, modality
                                ),
                                confidence: affinity,
                                evidence_count: stats.usage_count,
                                related_strategies: vec![*strategy],
                                related_features: HashMap::new(),
                                created_at: Utc::now(),
                            };
                            
                            // Avoid duplicate insights
                            if !self.insights.iter().any(|i| i.description == insight.description) {
                                self.insights.push(insight);
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Recommend strategy for a problem
    pub fn recommend(&self, signature: &ProblemSignature) -> Vec<(StrategyType, f64)> {
        let mut recommendations: Vec<(StrategyType, f64)> = Vec::new();
        
        // Check strategy stats for this modality
        for (strategy, stats) in &self.strategy_stats {
            if let Some(&affinity) = stats.problem_affinities.get(&signature.modality) {
                let success_rate = if stats.usage_count > 0 {
                    stats.success_count as f64 / stats.usage_count as f64
                } else {
                    0.5
                };
                
                recommendations.push((*strategy, affinity * success_rate));
            }
        }
        
        // Sort by score
        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Default recommendations if no history
        if recommendations.is_empty() {
            recommendations = vec![
                (StrategyType::AdaptiveHybrid, 0.8),
                (StrategyType::GreyWolf, 0.7),
                (StrategyType::ParticleSwarm, 0.6),
            ];
        }
        
        recommendations
    }
    
    /// Get all insights
    pub fn insights(&self) -> &[Insight] {
        &self.insights
    }
    
    /// Get strategy statistics
    pub fn strategy_stats(&self) -> &HashMap<StrategyType, StrategyStats> {
        &self.strategy_stats
    }
    
    /// Get record count
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
    
    /// Save to JSON
    pub fn save(&self) -> SwarmResult<String> {
        let state = IntellectState {
            records: self.records.iter().cloned().collect(),
            insights: self.insights.clone(),
            strategy_stats: self.strategy_stats.clone(),
            graph_nodes: self.graph.nodes.values().cloned().collect(),
            graph_edges: self.graph.edges.values().flatten().cloned().collect(),
        };
        
        serde_json::to_string_pretty(&state)
            .map_err(|e| SwarmIntelligenceError::EvolutionError(e.to_string()))
    }
    
    /// Load from JSON
    pub fn load(json: &str) -> SwarmResult<Self> {
        let state: IntellectState = serde_json::from_str(json)
            .map_err(|e| SwarmIntelligenceError::EvolutionError(e.to_string()))?;
        
        let mut intellect = Self::new();
        intellect.records = state.records.into_iter().collect();
        intellect.insights = state.insights;
        intellect.strategy_stats = state.strategy_stats;
        
        for node in state.graph_nodes {
            intellect.graph.add_node(node);
        }
        for edge in state.graph_edges {
            intellect.graph.add_edge(edge);
        }
        
        Ok(intellect)
    }
}

/// Serializable intellect state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IntellectState {
    records: Vec<IntellectRecord>,
    insights: Vec<Insight>,
    strategy_stats: HashMap<StrategyType, StrategyStats>,
    graph_nodes: Vec<KnowledgeNode>,
    graph_edges: Vec<KnowledgeEdge>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_knowledge_graph() {
        let mut graph = KnowledgeGraph::new();
        
        let node1 = KnowledgeNode {
            id: Uuid::new_v4(),
            node_type: NodeType::Strategy,
            label: "PSO".to_string(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            usage_count: 0,
            success_rate: 0.0,
        };
        
        let node2 = KnowledgeNode {
            id: Uuid::new_v4(),
            node_type: NodeType::Topology,
            label: "Hyperbolic".to_string(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            usage_count: 0,
            success_rate: 0.0,
        };
        
        let id1 = node1.id;
        let id2 = node2.id;
        
        graph.add_node(node1);
        graph.add_node(node2);
        
        graph.add_edge(KnowledgeEdge {
            from: id1,
            to: id2,
            edge_type: EdgeType::Synergy,
            weight: 0.8,
            evidence_count: 5,
        });
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }
    
    #[test]
    fn test_emergent_intellect() {
        let mut intellect = EmergentIntellect::new();
        
        let record = IntellectRecord {
            id: Uuid::new_v4(),
            problem_signature: ProblemSignature {
                dimensions: 10,
                search_volume: 1e10,
                modality: Modality::Multimodal,
                separability: 0.5,
                conditioning: 100.0,
                noise_level: 0.0,
                features: HashMap::new(),
            },
            best_strategy: StrategyType::GreyWolf,
            best_topology: TopologyType::Hyperbolic,
            optimal_parameters: HashMap::new(),
            best_fitness: 0.001,
            convergence_profile: ConvergenceProfile {
                half_life: 50,
                ninety_percent: 150,
                monotonic: true,
                stagnation_count: 0,
                avg_improvement_rate: 0.01,
                final_improvement_rate: 0.001,
            },
            genome: None,
            insights: Vec::new(),
            created_at: Utc::now(),
            evaluations: 5000,
            tags: HashSet::new(),
        };
        
        intellect.record(record);
        
        assert_eq!(intellect.record_count(), 1);
        assert!(intellect.strategy_stats().contains_key(&StrategyType::GreyWolf));
    }
}
