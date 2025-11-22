//! Web of Life based on Fritjof Capra's systems view
//! Life as network pattern, metabolic flows, and cognitive processes

use async_trait::async_trait;
use petgraph::graph::{DiGraph, Graph, NodeIndex, EdgeIndex};
use petgraph::algo::{connected_components, tarjan_scc};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};
use nalgebra as na;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Network topology describing the pattern of life
#[derive(Clone, Debug)]
pub struct NetworkTopology<N> {
    /// Community structure
    pub communities: Vec<Community<N>>,
    /// Hub nodes (high connectivity)
    pub hubs: Vec<NodeIndex>,
    /// Network motifs (recurring patterns)
    pub motifs: Vec<NetworkMotif>,
    /// Fractal dimension of the network
    pub fractality: f64,
}

/// A community within the network
#[derive(Clone, Debug)]
pub struct Community<N> {
    /// Nodes in this community
    pub nodes: Vec<NodeIndex>,
    /// Community identifier
    pub id: String,
    /// Internal coherence
    pub coherence: f64,
    /// Role in the larger network
    pub role: CommunityRole,
    _phantom: std::marker::PhantomData<N>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CommunityRole {
    Producer,
    Consumer,
    Decomposer,
    Regulator,
    Connector,
}

/// Network motif (recurring subgraph pattern)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMotif {
    /// Type of motif
    pub motif_type: MotifType,
    /// Frequency of occurrence
    pub frequency: u32,
    /// Significance score
    pub significance: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MotifType {
    FeedForwardLoop,
    FeedbackLoop,
    BiparallelMotif,
    SingleInputModule,
    DenseOverlappingRegulons,
}

/// Flow network representing metabolic processes
#[derive(Clone, Debug)]
pub struct FlowNetwork<N> {
    /// The directed graph of flows
    pub graph: DiGraph<N, Flow>,
    /// Conservation laws
    pub conservation_laws: Vec<ConservationLaw>,
    /// Flow balance at each node
    pub node_balance: HashMap<NodeIndex, f64>,
}

/// A flow between nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Flow {
    /// Type of flow (energy, matter, information)
    pub flow_type: FlowType,
    /// Flow rate
    pub rate: f64,
    /// Efficiency of transfer
    pub efficiency: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FlowType {
    Energy,
    Matter,
    Information,
    Mixed,
}

/// Conservation law in the flow network
#[derive(Clone, Debug)]
pub struct ConservationLaw {
    /// Name of conserved quantity
    pub quantity: String,
    /// Total amount in system
    pub total: f64,
    /// Tolerance for conservation
    pub tolerance: f64,
}

/// Cognitive map of the system
#[derive(Clone, Debug)]
pub struct CognitionMap {
    /// Perception nodes
    pub perception_nodes: Vec<PerceptionNode>,
    /// Action nodes
    pub action_nodes: Vec<ActionNode>,
    /// Cognitive links
    pub cognitive_links: Vec<CognitiveLink>,
    /// Learning rate
    pub learning_rate: f64,
}

#[derive(Clone, Debug)]
pub struct PerceptionNode {
    pub id: String,
    pub sensitivity: f64,
    pub receptive_field: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ActionNode {
    pub id: String,
    pub activation_threshold: f64,
    pub response_pattern: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CognitiveLink {
    pub from_perception: String,
    pub to_action: String,
    pub weight: f64,
    pub plasticity: f64,
}

/// Relationship between nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relationship {
    pub relationship_type: RelationshipType,
    pub strength: f64,
    pub reciprocity: f64,
    pub stability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RelationshipType {
    Symbiotic,
    Predatory,
    Competitive,
    Neutral,
    Parasitic,
    Mutualistic,
}

/// Core trait for Web of Life systems
pub trait WebOfLife: Send + Sync {
    type Node: Send + Sync;
    type Relationship: Send + Sync;
    
    /// Get the network pattern
    fn network_pattern(&self) -> NetworkTopology<Self::Node>;
    
    /// Get metabolic flow network
    fn metabolic_flows(&self) -> FlowNetwork<Self::Node>;
    
    /// Get cognitive processes
    fn cognitive_processes(&self) -> CognitionMap;
    
    /// Calculate network resilience
    fn network_resilience(&self) -> f64 {
        let topology = self.network_pattern();
        
        // Resilience based on connectivity and redundancy
        let avg_connectivity = topology.communities.iter()
            .map(|c| c.coherence)
            .sum::<f64>() / topology.communities.len().max(1) as f64;
        
        let hub_fraction = topology.hubs.len() as f64 / 
            topology.communities.iter().map(|c| c.nodes.len()).sum::<usize>().max(1) as f64;
        
        // Resilience increases with connectivity but decreases with hub dependence
        avg_connectivity * (1.0 - hub_fraction * 0.5)
    }
    
    /// Detect emergent properties
    fn emergent_properties(&self) -> Vec<EmergentProperty>;
    
    /// Analyze information flow
    fn information_flow_analysis(&self) -> InformationFlowMetrics;
}

/// An emergent property of the network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergentProperty {
    pub name: String,
    pub emergence_level: f64,
    pub contributing_nodes: Vec<String>,
    pub stability: f64,
}

/// Metrics for information flow
#[derive(Clone, Debug)]
pub struct InformationFlowMetrics {
    pub total_information: f64,
    pub flow_efficiency: f64,
    pub bottlenecks: Vec<NodeIndex>,
    pub information_entropy: f64,
}

/// Implementation of ecological network
pub struct EcologicalNetwork {
    /// Species network
    pub species_graph: Graph<Species, EcologicalInteraction>,
    /// Energy flow network
    pub energy_flows: FlowNetwork<Species>,
    /// Nutrient cycles
    pub nutrient_cycles: Vec<NutrientCycle>,
    /// Environmental conditions
    pub environment: EnvironmentalConditions,
}

#[derive(Clone, Debug)]
pub struct Species {
    pub name: String,
    pub biomass: f64,
    pub metabolic_rate: f64,
    pub trophic_level: f64,
    pub population: f64,
}

#[derive(Clone, Debug)]
pub struct EcologicalInteraction {
    pub interaction_type: RelationshipType,
    pub strength: f64,
    pub seasonal_variation: f64,
}

#[derive(Clone, Debug)]
pub struct NutrientCycle {
    pub nutrient: String,
    pub cycle_nodes: Vec<NodeIndex>,
    pub turnover_rate: f64,
    pub storage_capacity: f64,
}

#[derive(Clone, Debug)]
pub struct EnvironmentalConditions {
    pub temperature: f64,
    pub humidity: f64,
    pub light_level: f64,
    pub resource_availability: HashMap<String, f64>,
}

impl WebOfLife for EcologicalNetwork {
    type Node = Species;
    type Relationship = EcologicalInteraction;
    
    fn network_pattern(&self) -> NetworkTopology<Self::Node> {
        // Detect communities using connected components
        let communities = self.detect_communities();
        
        // Find hub species
        let hubs = self.find_hubs();
        
        // Detect network motifs
        let motifs = self.detect_motifs();
        
        // Calculate fractal dimension
        let fractality = self.calculate_fractal_dimension();
        
        NetworkTopology {
            communities,
            hubs,
            motifs,
            fractality,
        }
    }
    
    fn metabolic_flows(&self) -> FlowNetwork<Self::Node> {
        self.energy_flows.clone()
    }
    
    fn cognitive_processes(&self) -> CognitionMap {
        // Ecological cognition through adaptive responses
        let mut perception_nodes = Vec::new();
        let mut action_nodes = Vec::new();
        let mut cognitive_links = Vec::new();
        
        // Each species has perception of resources and threats
        for node in self.species_graph.node_indices() {
            if let Some(species) = self.species_graph.node_weight(node) {
                // Perception of resources
                perception_nodes.push(PerceptionNode {
                    id: format!("{}_resource_perception", species.name),
                    sensitivity: 1.0 / species.metabolic_rate,
                    receptive_field: vec![1.0, 0.5, 0.25], // Spatial decay
                });
                
                // Action: foraging behavior
                action_nodes.push(ActionNode {
                    id: format!("{}_foraging", species.name),
                    activation_threshold: 0.3,
                    response_pattern: vec![0.8, 0.6, 0.4],
                });
                
                // Link perception to action
                cognitive_links.push(CognitiveLink {
                    from_perception: format!("{}_resource_perception", species.name),
                    to_action: format!("{}_foraging", species.name),
                    weight: species.trophic_level,
                    plasticity: 0.1,
                });
            }
        }
        
        CognitionMap {
            perception_nodes,
            action_nodes,
            cognitive_links,
            learning_rate: 0.01,
        }
    }
    
    fn emergent_properties(&self) -> Vec<EmergentProperty> {
        let mut properties = Vec::new();
        
        // Stability emerges from diversity
        let diversity = self.calculate_diversity();
        if diversity > 2.0 {
            properties.push(EmergentProperty {
                name: "Ecosystem Stability".to_string(),
                emergence_level: diversity / 3.0,
                contributing_nodes: self.species_graph.node_weights()
                    .map(|s| s.name.clone())
                    .collect(),
                stability: 0.8,
            });
        }
        
        // Resilience emerges from redundancy
        let redundancy = self.calculate_functional_redundancy();
        if redundancy > 0.5 {
            properties.push(EmergentProperty {
                name: "Ecological Resilience".to_string(),
                emergence_level: redundancy,
                contributing_nodes: vec!["Multiple species".to_string()],
                stability: 0.9,
            });
        }
        
        properties
    }
    
    fn information_flow_analysis(&self) -> InformationFlowMetrics {
        // Information as reduction in uncertainty about resource locations
        let total_info = self.species_graph.node_count() as f64 * 
            self.environment.resource_availability.len() as f64;
        
        // Efficiency based on trophic transfer
        let mut total_transfer = 0.0;
        let mut total_possible = 0.0;
        
        for edge in self.species_graph.edge_indices() {
            if let Some(interaction) = self.species_graph.edge_weight(edge) {
                total_transfer += interaction.strength;
                total_possible += 1.0;
            }
        }
        
        let efficiency = if total_possible > 0.0 {
            total_transfer / total_possible
        } else {
            0.0
        };
        
        // Find bottlenecks (nodes with high betweenness centrality)
        let bottlenecks = self.find_bottlenecks();
        
        // Shannon entropy of the network
        let entropy = self.calculate_network_entropy();
        
        InformationFlowMetrics {
            total_information: total_info,
            flow_efficiency: efficiency,
            bottlenecks,
            information_entropy: entropy,
        }
    }
}

impl EcologicalNetwork {
    pub fn new() -> Self {
        Self {
            species_graph: Graph::new(),
            energy_flows: FlowNetwork {
                graph: DiGraph::new(),
                conservation_laws: vec![
                    ConservationLaw {
                        quantity: "Energy".to_string(),
                        total: 1000.0,
                        tolerance: 0.01,
                    }
                ],
                node_balance: HashMap::new(),
            },
            nutrient_cycles: Vec::new(),
            environment: EnvironmentalConditions {
                temperature: 20.0,
                humidity: 0.6,
                light_level: 0.8,
                resource_availability: HashMap::new(),
            },
        }
    }
    
    /// Add a species to the network
    pub fn add_species(&mut self, species: Species) -> NodeIndex {
        self.species_graph.add_node(species)
    }
    
    /// Add an interaction between species
    pub fn add_interaction(&mut self, from: NodeIndex, to: NodeIndex, interaction: EcologicalInteraction) {
        self.species_graph.add_edge(from, to, interaction);
    }
    
    /// Detect communities using modularity optimization
    fn detect_communities(&self) -> Vec<Community<Species>> {
        let components = tarjan_scc(&self.species_graph);
        
        components.into_iter().enumerate().map(|(i, component)| {
            let coherence = self.calculate_community_coherence(&component);
            
            // Determine role based on average trophic level
            let avg_trophic = component.iter()
                .filter_map(|&idx| self.species_graph.node_weight(idx))
                .map(|s| s.trophic_level)
                .sum::<f64>() / component.len().max(1) as f64;
            
            let role = if avg_trophic < 1.5 {
                CommunityRole::Producer
            } else if avg_trophic < 2.5 {
                CommunityRole::Consumer
            } else if avg_trophic < 3.5 {
                CommunityRole::Regulator
            } else {
                CommunityRole::Decomposer
            };
            
            Community {
                nodes: component,
                id: format!("community_{}", i),
                coherence,
                role,
                _phantom: std::marker::PhantomData,
            }
        }).collect()
    }
    
    /// Calculate coherence of a community
    fn calculate_community_coherence(&self, nodes: &[NodeIndex]) -> f64 {
        if nodes.len() < 2 {
            return 1.0;
        }
        
        let mut internal_edges = 0;
        let mut external_edges = 0;
        
        for &node in nodes {
            for edge in self.species_graph.edges(node) {
                if nodes.contains(&edge.target()) {
                    internal_edges += 1;
                } else {
                    external_edges += 1;
                }
            }
        }
        
        let total = internal_edges + external_edges;
        if total > 0 {
            internal_edges as f64 / total as f64
        } else {
            0.0
        }
    }
    
    /// Find hub species (high degree centrality)
    fn find_hubs(&self) -> Vec<NodeIndex> {
        let mut degree_map: HashMap<NodeIndex, usize> = HashMap::new();
        
        for node in self.species_graph.node_indices() {
            let degree = self.species_graph.edges(node).count();
            degree_map.insert(node, degree);
        }
        
        let avg_degree = degree_map.values().sum::<usize>() as f64 / 
            degree_map.len().max(1) as f64;
        
        degree_map.into_iter()
            .filter(|(_, degree)| *degree as f64 > avg_degree * 2.0)
            .map(|(node, _)| node)
            .collect()
    }
    
    /// Detect network motifs
    fn detect_motifs(&self) -> Vec<NetworkMotif> {
        let mut motifs = Vec::new();
        
        // Detect feed-forward loops (simplified)
        let mut ffl_count = 0;
        for node_a in self.species_graph.node_indices() {
            for edge_ab in self.species_graph.edges(node_a) {
                let node_b = edge_ab.target();
                for edge_bc in self.species_graph.edges(node_b) {
                    let node_c = edge_bc.target();
                    // Check if A also connects to C
                    if self.species_graph.edges(node_a).any(|e| e.target() == node_c) {
                        ffl_count += 1;
                    }
                }
            }
        }
        
        if ffl_count > 0 {
            motifs.push(NetworkMotif {
                motif_type: MotifType::FeedForwardLoop,
                frequency: ffl_count,
                significance: (ffl_count as f64).sqrt(),
            });
        }
        
        motifs
    }
    
    /// Calculate fractal dimension using box-counting
    fn calculate_fractal_dimension(&self) -> f64 {
        // Simplified fractal dimension calculation
        let n = self.species_graph.node_count() as f64;
        let e = self.species_graph.edge_count() as f64;
        
        if n > 1.0 && e > 0.0 {
            // Approximate fractal dimension from network density
            let density = e / (n * (n - 1.0) / 2.0);
            1.0 + density.ln() / n.ln()
        } else {
            1.0
        }
    }
    
    /// Calculate Shannon diversity
    fn calculate_diversity(&self) -> f64 {
        let total_biomass: f64 = self.species_graph.node_weights()
            .map(|s| s.biomass)
            .sum();
        
        if total_biomass <= 0.0 {
            return 0.0;
        }
        
        let mut shannon = 0.0;
        for species in self.species_graph.node_weights() {
            let p = species.biomass / total_biomass;
            if p > 0.0 {
                shannon -= p * p.ln();
            }
        }
        
        shannon
    }
    
    /// Calculate functional redundancy
    fn calculate_functional_redundancy(&self) -> f64 {
        // Group species by trophic level
        let mut trophic_groups: HashMap<i32, Vec<&Species>> = HashMap::new();
        
        for species in self.species_graph.node_weights() {
            let trophic_int = (species.trophic_level * 10.0).round() as i32;
            trophic_groups.entry(trophic_int).or_insert(Vec::new()).push(species);
        }
        
        // Redundancy is high when multiple species per trophic level
        let avg_species_per_level = trophic_groups.values()
            .map(|group| group.len())
            .sum::<usize>() as f64 / trophic_groups.len().max(1) as f64;
        
        (avg_species_per_level - 1.0).max(0.0) / 10.0
    }
    
    /// Find bottleneck nodes
    fn find_bottlenecks(&self) -> Vec<NodeIndex> {
        // Simplified: nodes with high in/out degree ratio
        let mut bottlenecks = Vec::new();
        
        for node in self.species_graph.node_indices() {
            let in_degree = self.species_graph.edges_directed(node, petgraph::Direction::Incoming).count();
            let out_degree = self.species_graph.edges(node).count();
            
            if in_degree > 0 && out_degree > 0 {
                let ratio = in_degree.max(out_degree) as f64 / in_degree.min(out_degree) as f64;
                if ratio > 3.0 {
                    bottlenecks.push(node);
                }
            }
        }
        
        bottlenecks
    }
    
    /// Calculate network entropy
    fn calculate_network_entropy(&self) -> f64 {
        let n = self.species_graph.node_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        
        // Entropy based on degree distribution
        let mut degree_counts: HashMap<usize, usize> = HashMap::new();
        
        for node in self.species_graph.node_indices() {
            let degree = self.species_graph.edges(node).count();
            *degree_counts.entry(degree).or_insert(0) += 1;
        }
        
        let mut entropy = 0.0;
        for count in degree_counts.values() {
            let p = *count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ecological_network() {
        let mut network = EcologicalNetwork::new();
        
        // Add some species
        let grass = network.add_species(Species {
            name: "Grass".to_string(),
            biomass: 1000.0,
            metabolic_rate: 0.1,
            trophic_level: 1.0,
            population: 10000.0,
        });
        
        let rabbit = network.add_species(Species {
            name: "Rabbit".to_string(),
            biomass: 100.0,
            metabolic_rate: 1.0,
            trophic_level: 2.0,
            population: 100.0,
        });
        
        let fox = network.add_species(Species {
            name: "Fox".to_string(),
            biomass: 10.0,
            metabolic_rate: 2.0,
            trophic_level: 3.0,
            population: 10.0,
        });
        
        // Add interactions
        network.add_interaction(rabbit, grass, EcologicalInteraction {
            interaction_type: RelationshipType::Predatory,
            strength: 0.8,
            seasonal_variation: 0.2,
        });
        
        network.add_interaction(fox, rabbit, EcologicalInteraction {
            interaction_type: RelationshipType::Predatory,
            strength: 0.6,
            seasonal_variation: 0.1,
        });
        
        // Test network pattern
        let topology = network.network_pattern();
        assert!(topology.communities.len() > 0);
        
        // Test resilience
        let resilience = network.network_resilience();
        assert!(resilience >= 0.0 && resilience <= 1.0);
    }
}