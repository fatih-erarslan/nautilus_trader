//! Codependent Risk Model based on Pratītyasamutpāda (Dependent Origination)
//!
//! "All phenomena arise through mutual conditions" - Nāgārjuna
//!
//! This module implements risk assessment based on the Buddhist concept of dependent origination,
//! recognizing that no risk exists independently but arises through network conditions.

use nalgebra::{DMatrix, DVector};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodependentRiskError {
    #[error("Invalid asset ID: {0}")]
    InvalidAssetId(usize),
    #[error("Matrix computation failed: {0}")]
    MatrixError(String),
    #[error("Graph operation failed: {0}")]
    GraphError(String),
    #[error("Invalid decay parameter: {0}")]
    InvalidDecay(f64),
}

/// Asset node in the dependency graph
#[derive(Clone, Debug)]
pub struct AssetNode {
    pub id: usize,
    pub symbol: String,
    pub standalone_risk: f64,
    pub sector: String,
}

/// Dependency edge representing relationship between assets
#[derive(Clone, Debug)]
pub struct DependencyEdge {
    /// Correlation-based dependency strength (0.0 to 1.0)
    pub weight: f64,
    /// Type of dependency
    pub dependency_type: DependencyType,
}

/// Types of dependencies between assets
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DependencyType {
    Correlation,    // Statistical correlation
    Supply,         // Supply chain dependency
    Credit,         // Credit exposure
    Contagion,      // Market contagion effect
}

/// Result of codependent risk calculation for a single asset
#[derive(Clone, Debug)]
pub struct CodependentRisk {
    /// Original standalone risk (illusory baseline)
    pub standalone: f64,
    /// Risk arising from conditions (codependent component)
    pub codependent: f64,
    /// Total effective risk (standalone + codependent)
    pub effective: f64,
    /// Top contributing dependencies (asset_id, contribution)
    pub top_contributors: Vec<(usize, f64)>,
}

/// Systemic risk assessment for entire portfolio
#[derive(Clone, Debug)]
pub struct SystemicRisk {
    /// Total system risk considering all dependencies
    pub total: f64,
    /// Risk concentration index (Herfindahl-like)
    pub concentration: f64,
    /// Network centrality scores (eigenvector centrality)
    pub centrality: DVector<f64>,
    /// Critical contagion paths in the network
    pub critical_paths: Vec<Vec<usize>>,
}

/// Codependent Risk Model following Pratītyasamutpāda
///
/// Implements risk propagation through dependency networks using:
/// - Leontief inverse: R = (I - λA)^(-1) * R_0
/// - Eigenvector centrality: Ax = λx
/// - Network flow analysis
pub struct CodependentRiskModel {
    /// Dependency graph: nodes are assets, edges are dependencies
    dependency_graph: Graph<AssetNode, DependencyEdge>,
    /// Map from asset ID to graph NodeIndex
    id_to_node: HashMap<usize, NodeIndex>,
    /// Adjacency matrix for fast computation
    adjacency_matrix: DMatrix<f64>,
    /// Current standalone risks (illusory but useful as priors)
    standalone_risks: DVector<f64>,
    /// Propagation decay factor (0.0 to 1.0)
    decay_lambda: f64,
    /// Maximum propagation depth for path finding
    max_depth: usize,
    /// Number of assets currently in the model
    num_assets: usize,
}

impl CodependentRiskModel {
    /// Create a new CodependentRiskModel
    ///
    /// # Arguments
    /// * `_num_assets` - Expected number of assets (for preallocation, currently unused)
    /// * `decay_lambda` - Risk propagation decay factor (0.0 to 1.0)
    /// * `max_depth` - Maximum depth for contagion path search
    pub fn new(_num_assets: usize, decay_lambda: f64, max_depth: usize) -> Self {
        assert!(
            decay_lambda >= 0.0 && decay_lambda < 1.0,
            "Decay lambda must be in [0, 1)"
        );

        Self {
            dependency_graph: Graph::new(),
            id_to_node: HashMap::new(),
            adjacency_matrix: DMatrix::zeros(0, 0),
            standalone_risks: DVector::zeros(0),
            decay_lambda,
            max_depth,
            num_assets: 0,
        }
    }

    /// Add an asset to the model
    ///
    /// Returns the asset's position in the internal indexing
    pub fn add_asset(&mut self, node: AssetNode) -> usize {
        let asset_id = node.id;
        let node_index = self.dependency_graph.add_node(node);
        self.id_to_node.insert(asset_id, node_index);
        self.num_assets += 1;

        // Resize matrices
        let new_size = self.num_assets;
        let mut new_adjacency = DMatrix::zeros(new_size, new_size);
        let mut new_risks = DVector::zeros(new_size);

        // Copy existing data
        if new_size > 1 {
            for i in 0..new_size - 1 {
                for j in 0..new_size - 1 {
                    new_adjacency[(i, j)] = self.adjacency_matrix[(i, j)];
                }
                new_risks[i] = self.standalone_risks[i];
            }
        }

        // Set new asset's standalone risk
        new_risks[new_size - 1] = self
            .dependency_graph
            .node_weight(node_index)
            .unwrap()
            .standalone_risk;

        self.adjacency_matrix = new_adjacency;
        self.standalone_risks = new_risks;

        asset_id
    }

    /// Add a dependency edge between assets
    ///
    /// # Arguments
    /// * `from` - Source asset ID
    /// * `to` - Target asset ID
    /// * `edge` - Dependency edge with weight and type
    pub fn add_dependency(
        &mut self,
        from: usize,
        to: usize,
        edge: DependencyEdge,
    ) -> Result<(), CodependentRiskError> {
        let from_node = self
            .id_to_node
            .get(&from)
            .ok_or(CodependentRiskError::InvalidAssetId(from))?;
        let to_node = self
            .id_to_node
            .get(&to)
            .ok_or(CodependentRiskError::InvalidAssetId(to))?;

        self.dependency_graph.add_edge(*from_node, *to_node, edge);
        self.rebuild_adjacency_matrix()?;

        Ok(())
    }

    /// Rebuild adjacency matrix from current dependency graph
    fn rebuild_adjacency_matrix(&mut self) -> Result<(), CodependentRiskError> {
        let n = self.num_assets;
        let mut new_adjacency = DMatrix::zeros(n, n);

        // Build reverse mapping from NodeIndex to position
        // Note: asset_id is stored in id_to_node for lookup but position is used for matrix indexing
        let mut node_to_pos: HashMap<NodeIndex, usize> = HashMap::new();
        for (pos, (_asset_id, &node_idx)) in self.id_to_node.iter().enumerate() {
            node_to_pos.insert(node_idx, pos);
        }

        // Fill adjacency matrix from graph edges
        for edge in self.dependency_graph.edge_references() {
            let from_pos = node_to_pos[&edge.source()];
            let to_pos = node_to_pos[&edge.target()];
            let weight = edge.weight().weight;

            new_adjacency[(to_pos, from_pos)] = weight; // Risk flows FROM source TO target
        }

        self.adjacency_matrix = new_adjacency;
        Ok(())
    }

    /// Calculate codependent risk for a single asset
    ///
    /// Uses the formula: R = (I - λA)^(-1) * R_0
    /// where R is effective risk, λ is decay, A is adjacency, R_0 is standalone risk
    pub fn calculate_risk(&self, asset_id: usize) -> Result<CodependentRisk, CodependentRiskError> {
        let node_idx = self
            .id_to_node
            .get(&asset_id)
            .ok_or(CodependentRiskError::InvalidAssetId(asset_id))?;

        // Find position in matrices
        let pos = self
            .id_to_node
            .iter()
            .position(|(_, &idx)| idx == *node_idx)
            .unwrap();

        let standalone = self.standalone_risks[pos];

        // Calculate effective risk using Leontief inverse
        let effective_risks = self.calculate_all_effective_risks()?;
        let effective = effective_risks[pos];
        let codependent = effective - standalone;

        // Find top contributors
        let mut contributors: Vec<(usize, f64)> = Vec::new();
        for i in 0..self.num_assets {
            if i != pos {
                let contribution = self.adjacency_matrix[(pos, i)] * effective_risks[i];
                if contribution.abs() > 1e-6 {
                    // Find asset ID for position i
                    let asset_id = self
                        .id_to_node
                        .iter()
                        .find(|(_, &idx)| {
                            self.id_to_node
                                .iter()
                                .position(|(_, &idx2)| idx2 == idx)
                                == Some(i)
                        })
                        .map(|(id, _)| *id)
                        .unwrap();
                    contributors.push((asset_id, contribution));
                }
            }
        }

        // Sort by contribution magnitude
        contributors.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        contributors.truncate(5); // Keep top 5

        Ok(CodependentRisk {
            standalone,
            codependent,
            effective,
            top_contributors: contributors,
        })
    }

    /// Calculate effective risks for all assets using iterative propagation
    ///
    /// Uses power series approximation: R = R_0 + λAR_0 + (λA)²R_0 + ...
    /// This is more stable than direct Leontief inverse for high decay factors
    fn calculate_all_effective_risks(&self) -> Result<DVector<f64>, CodependentRiskError> {
        let n = self.num_assets;
        if n == 0 {
            return Ok(DVector::zeros(0));
        }

        // Use iterative propagation for stability
        let max_iterations = 100;
        let tolerance = 1e-8;

        let mut effective_risks = self.standalone_risks.clone();
        let lambda_a = &self.adjacency_matrix * self.decay_lambda;

        // Iterate: R^(t+1) = R_0 + λA * R^(t)
        for _ in 0..max_iterations {
            let propagated = &lambda_a * &effective_risks;
            let new_risks = &self.standalone_risks + &propagated;

            // Check convergence
            let diff = (&new_risks - &effective_risks).norm();
            effective_risks = new_risks;

            if diff < tolerance {
                break;
            }
        }

        // Ensure all risks are non-negative
        for i in 0..n {
            if effective_risks[i] < 0.0 {
                effective_risks[i] = self.standalone_risks[i];
            }
        }

        Ok(effective_risks)
    }

    /// Calculate systemic risk for entire portfolio
    pub fn systemic_risk(&self) -> Result<SystemicRisk, CodependentRiskError> {
        if self.num_assets == 0 {
            return Ok(SystemicRisk {
                total: 0.0,
                concentration: 0.0,
                centrality: DVector::zeros(0),
                critical_paths: Vec::new(),
            });
        }

        // Calculate total system risk
        let effective_risks = self.calculate_all_effective_risks()?;
        let total = effective_risks.sum() / self.num_assets as f64;

        // Calculate concentration index (Herfindahl)
        let risk_squared_sum: f64 = effective_risks.iter().map(|r| r * r).sum();
        let concentration = risk_squared_sum / (total * total * self.num_assets as f64);

        // Calculate eigenvector centrality
        let centrality = self.calculate_centrality()?;

        // Find critical contagion paths
        let critical_paths = self.find_critical_paths()?;

        Ok(SystemicRisk {
            total,
            concentration,
            centrality,
            critical_paths,
        })
    }

    /// Calculate network centrality using eigenvector centrality
    ///
    /// Solves: Ax = λx for the largest eigenvalue
    fn calculate_centrality(&self) -> Result<DVector<f64>, CodependentRiskError> {
        let n = self.num_assets;
        if n == 0 {
            return Ok(DVector::zeros(0));
        }

        // Use power iteration to find dominant eigenvector
        let max_iterations = 1000;
        let tolerance = 1e-6;

        let mut x = DVector::from_element(n, 1.0 / (n as f64).sqrt());

        for _ in 0..max_iterations {
            let ax = &self.adjacency_matrix * &x;
            let norm = ax.norm();

            if norm < 1e-10 {
                // Zero matrix or disconnected graph
                return Ok(DVector::from_element(n, 1.0 / (n as f64).sqrt()));
            }

            let x_new = &ax / norm;
            let diff = (&x_new - &x).norm();

            x = x_new;

            if diff < tolerance {
                break;
            }
        }

        // Normalize to sum to 1
        let sum: f64 = x.iter().sum();
        if sum > 1e-10 {
            x /= sum;
        }

        Ok(x)
    }

    /// Find critical contagion paths in the network
    fn find_critical_paths(&self) -> Result<Vec<Vec<usize>>, CodependentRiskError> {
        let mut paths = Vec::new();

        // Find high-risk assets (top 20%)
        let effective_risks = self.calculate_all_effective_risks()?;
        let threshold = effective_risks.max() * 0.8;

        let high_risk_positions: Vec<usize> = effective_risks
            .iter()
            .enumerate()
            .filter(|(_, &risk)| risk >= threshold)
            .map(|(pos, _)| pos)
            .collect();

        // For each pair of high-risk assets, find shortest path
        for &source_pos in &high_risk_positions {
            for &target_pos in &high_risk_positions {
                if source_pos != target_pos {
                    if let Some(path) = self.find_shortest_path(source_pos, target_pos) {
                        if path.len() <= self.max_depth {
                            paths.push(path);
                        }
                    }
                }
            }
        }

        // Sort by path length (shorter = more critical)
        paths.sort_by_key(|p| p.len());
        paths.truncate(10); // Keep top 10 critical paths

        Ok(paths)
    }

    /// Find shortest path between two asset positions using BFS
    fn find_shortest_path(&self, source_pos: usize, target_pos: usize) -> Option<Vec<usize>> {
        // Convert positions to NodeIndex
        let source_node = self
            .id_to_node
            .iter()
            .find(|(_, &idx)| {
                self.id_to_node
                    .iter()
                    .position(|(_, &idx2)| idx2 == idx)
                    == Some(source_pos)
            })
            .map(|(_, &idx)| idx)?;

        let target_node = self
            .id_to_node
            .iter()
            .find(|(_, &idx)| {
                self.id_to_node
                    .iter()
                    .position(|(_, &idx2)| idx2 == idx)
                    == Some(target_pos)
            })
            .map(|(_, &idx)| idx)?;

        // BFS
        let mut queue = VecDeque::new();
        let mut visited = HashMap::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        queue.push_back(source_node);
        visited.insert(source_node, true);

        while let Some(current) = queue.pop_front() {
            if current == target_node {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = target_node;

                while node != source_node {
                    let asset_id = self.dependency_graph.node_weight(node).unwrap().id;
                    path.push(asset_id);
                    node = parent[&node];
                }

                let source_asset_id = self.dependency_graph.node_weight(source_node).unwrap().id;
                path.push(source_asset_id);
                path.reverse();

                return Some(path);
            }

            for neighbor in self.dependency_graph.neighbors_directed(current, Direction::Outgoing)
            {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, true);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Find contagion paths from source to target asset
    pub fn find_contagion_paths(
        &self,
        source: usize,
        target: usize,
    ) -> Result<Vec<Vec<usize>>, CodependentRiskError> {
        let source_node = self
            .id_to_node
            .get(&source)
            .ok_or(CodependentRiskError::InvalidAssetId(source))?;
        let target_node = self
            .id_to_node
            .get(&target)
            .ok_or(CodependentRiskError::InvalidAssetId(target))?;

        let mut all_paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = HashMap::new();

        self.dfs_find_paths(
            *source_node,
            *target_node,
            &mut current_path,
            &mut visited,
            &mut all_paths,
            0,
        );

        Ok(all_paths)
    }

    /// DFS helper for finding all paths
    fn dfs_find_paths(
        &self,
        current: NodeIndex,
        target: NodeIndex,
        current_path: &mut Vec<usize>,
        visited: &mut HashMap<NodeIndex, bool>,
        all_paths: &mut Vec<Vec<usize>>,
        depth: usize,
    ) {
        if depth > self.max_depth {
            return;
        }

        let asset_id = self.dependency_graph.node_weight(current).unwrap().id;
        current_path.push(asset_id);
        visited.insert(current, true);

        if current == target {
            all_paths.push(current_path.clone());
        } else {
            for neighbor in self.dependency_graph.neighbors_directed(current, Direction::Outgoing)
            {
                if !visited.contains_key(&neighbor) {
                    self.dfs_find_paths(
                        neighbor,
                        target,
                        current_path,
                        visited,
                        all_paths,
                        depth + 1,
                    );
                }
            }
        }

        current_path.pop();
        visited.remove(&current);
    }

    /// Update standalone risk for an asset (triggers recalculation)
    pub fn update_standalone_risk(
        &mut self,
        asset_id: usize,
        risk: f64,
    ) -> Result<(), CodependentRiskError> {
        let node_idx = self
            .id_to_node
            .get(&asset_id)
            .ok_or(CodependentRiskError::InvalidAssetId(asset_id))?;

        // Update in graph
        if let Some(node) = self.dependency_graph.node_weight_mut(*node_idx) {
            node.standalone_risk = risk;
        }

        // Update in vector
        let pos = self
            .id_to_node
            .iter()
            .position(|(_, &idx)| idx == *node_idx)
            .unwrap();
        self.standalone_risks[pos] = risk;

        Ok(())
    }

    /// Get number of assets in the model
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }

    /// Get decay lambda parameter
    pub fn decay_lambda(&self) -> f64 {
        self.decay_lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_model_creation() {
        let model = CodependentRiskModel::new(10, 0.8, 5);
        assert_eq!(model.num_assets(), 0);
        assert_eq!(model.decay_lambda(), 0.8);
    }

    #[test]
    fn test_add_assets() {
        let mut model = CodependentRiskModel::new(10, 0.8, 5);

        let asset1 = AssetNode {
            id: 1,
            symbol: "AAPL".to_string(),
            standalone_risk: 0.15,
            sector: "Technology".to_string(),
        };

        let asset2 = AssetNode {
            id: 2,
            symbol: "MSFT".to_string(),
            standalone_risk: 0.12,
            sector: "Technology".to_string(),
        };

        model.add_asset(asset1);
        model.add_asset(asset2);

        assert_eq!(model.num_assets(), 2);
    }

    #[test]
    fn test_simple_dependency() {
        let mut model = CodependentRiskModel::new(10, 0.5, 5);

        let asset1 = AssetNode {
            id: 1,
            symbol: "AAPL".to_string(),
            standalone_risk: 0.20,
            sector: "Technology".to_string(),
        };

        let asset2 = AssetNode {
            id: 2,
            symbol: "SUPPLIER".to_string(),
            standalone_risk: 0.10,
            sector: "Manufacturing".to_string(),
        };

        model.add_asset(asset1);
        model.add_asset(asset2);

        let edge = DependencyEdge {
            weight: 0.8,
            dependency_type: DependencyType::Supply,
        };

        model.add_dependency(2, 1, edge).unwrap();

        let risk = model.calculate_risk(1).unwrap();
        assert!(risk.effective > risk.standalone);
        assert!(risk.codependent > 0.0);
    }

    #[test]
    fn test_financial_network() {
        let mut model = CodependentRiskModel::new(10, 0.6, 5);

        // Create a realistic financial network
        let assets = vec![
            AssetNode {
                id: 1,
                symbol: "JPM".to_string(),
                standalone_risk: 0.25,
                sector: "Finance".to_string(),
            },
            AssetNode {
                id: 2,
                symbol: "BAC".to_string(),
                standalone_risk: 0.28,
                sector: "Finance".to_string(),
            },
            AssetNode {
                id: 3,
                symbol: "AAPL".to_string(),
                standalone_risk: 0.18,
                sector: "Technology".to_string(),
            },
            AssetNode {
                id: 4,
                symbol: "TSLA".to_string(),
                standalone_risk: 0.35,
                sector: "Automotive".to_string(),
            },
        ];

        for asset in assets {
            model.add_asset(asset);
        }

        // Add dependencies
        model
            .add_dependency(
                1,
                2,
                DependencyEdge {
                    weight: 0.7,
                    dependency_type: DependencyType::Correlation,
                },
            )
            .unwrap();

        model
            .add_dependency(
                2,
                1,
                DependencyEdge {
                    weight: 0.6,
                    dependency_type: DependencyType::Credit,
                },
            )
            .unwrap();

        model
            .add_dependency(
                3,
                4,
                DependencyEdge {
                    weight: 0.4,
                    dependency_type: DependencyType::Supply,
                },
            )
            .unwrap();

        // Test risk calculation
        let risk_jpm = model.calculate_risk(1).unwrap();
        assert!(risk_jpm.effective >= risk_jpm.standalone);

        // Test systemic risk
        let systemic = model.systemic_risk().unwrap();
        assert!(systemic.total > 0.0);
        assert!(systemic.concentration > 0.0);
        assert_eq!(systemic.centrality.len(), 4);
    }

    #[test]
    fn test_contagion_paths() {
        let mut model = CodependentRiskModel::new(5, 0.7, 5);

        for i in 1..=5 {
            model.add_asset(AssetNode {
                id: i,
                symbol: format!("ASSET{}", i),
                standalone_risk: 0.15,
                sector: "Test".to_string(),
            });
        }

        // Create chain: 1 -> 2 -> 3 -> 4 -> 5
        for i in 1..=4 {
            model
                .add_dependency(
                    i,
                    i + 1,
                    DependencyEdge {
                        weight: 0.8,
                        dependency_type: DependencyType::Contagion,
                    },
                )
                .unwrap();
        }

        let paths = model.find_contagion_paths(1, 5).unwrap();
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.len() == 5)); // Should find the full chain
    }

    #[test]
    fn test_update_standalone_risk() {
        let mut model = CodependentRiskModel::new(5, 0.5, 5);

        model.add_asset(AssetNode {
            id: 1,
            symbol: "TEST".to_string(),
            standalone_risk: 0.10,
            sector: "Test".to_string(),
        });

        let initial_risk = model.calculate_risk(1).unwrap();
        assert!((initial_risk.standalone - 0.10).abs() < 1e-6);

        model.update_standalone_risk(1, 0.25).unwrap();

        let updated_risk = model.calculate_risk(1).unwrap();
        assert!((updated_risk.standalone - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_systemic_risk_concentration() {
        let mut model = CodependentRiskModel::new(3, 0.5, 5);

        // Create three assets with very different risks
        model.add_asset(AssetNode {
            id: 1,
            symbol: "HIGH".to_string(),
            standalone_risk: 0.50,
            sector: "Risky".to_string(),
        });

        model.add_asset(AssetNode {
            id: 2,
            symbol: "LOW1".to_string(),
            standalone_risk: 0.05,
            sector: "Safe".to_string(),
        });

        model.add_asset(AssetNode {
            id: 3,
            symbol: "LOW2".to_string(),
            standalone_risk: 0.05,
            sector: "Safe".to_string(),
        });

        let systemic = model.systemic_risk().unwrap();

        // Concentration should be high due to one dominant risk
        assert!(systemic.concentration > 1.0);
    }
}
