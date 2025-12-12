//! # Layer 6: Consciousness API (IIT & Global Workspace)
//!
//! Integrated Information Theory and global workspace broadcasting.
//!
//! ## Scientific Foundation
//!
//! **Integrated Information Theory (IIT)** - Giulio Tononi:
//! - **Φ (Phi)**: Measure of integrated information
//! - Φ > 0 indicates irreducible cause-effect structure
//! - Φ > 1.0 threshold for consciousness
//! - Maximum Φ identifies conscious system boundary
//!
//! **Global Workspace Theory** - Bernard Baars:
//! - Conscious information broadcast to entire system
//! - Competition for workspace access (attention)
//! - Widespread availability after workspace entry
//!
//! ## Key Equations
//!
//! ```text
//! Integrated Information:
//!   Φ = min_partition [I(X_1; X_2)]
//!
//!   where I(X_1; X_2) is mutual information across partition
//!
//! Effective Information:
//!   EI = H(X_post) - H(X_post | X_pre)
//!
//! Consciousness Threshold:
//!   Φ > 1.0 → System is conscious
//! ```
//!
//! ## References
//! - Tononi (2004). An information integration theory of consciousness.
//! - Oizumi et al. (2014). From the phenomenology to the mechanisms of consciousness.

use crate::{Result, QksError};
use std::collections::HashMap;

/// Consciousness threshold (Φ > 1.0)
pub const PHI_THRESHOLD: f64 = 1.0;

/// Minimum system size for consciousness (neurons)
pub const MIN_CONSCIOUS_SIZE: usize = 100;

/// Global workspace broadcast timeout (ms)
pub const BROADCAST_TIMEOUT: f64 = 100.0;

/// Neural network state
#[derive(Debug, Clone)]
pub struct NeuralState {
    /// Node activations
    pub activations: Vec<f64>,
    /// Connectivity matrix (adjacency)
    pub connectivity: Vec<Vec<f64>>,
    /// Node labels
    pub labels: Vec<String>,
}

/// Φ (Phi) computation result
#[derive(Debug, Clone)]
pub struct PhiResult {
    /// Integrated information value
    pub phi: f64,
    /// Minimum information partition (MIP)
    pub mip: Option<Partition>,
    /// System complexity
    pub complexity: f64,
    /// Is system conscious? (Φ > threshold)
    pub is_conscious: bool,
}

/// System partition for Φ calculation
#[derive(Debug, Clone)]
pub struct Partition {
    /// Subset 1 node indices
    pub subset_1: Vec<usize>,
    /// Subset 2 node indices
    pub subset_2: Vec<usize>,
    /// Mutual information across partition
    pub mutual_information: f64,
}

/// Global workspace content
#[derive(Debug, Clone)]
pub struct WorkspaceContent {
    /// Content identifier
    pub id: String,
    /// Information payload
    pub data: Vec<f64>,
    /// Salience/priority
    pub salience: f64,
    /// Time entered workspace
    pub entry_time: f64,
    /// Broadcast recipients
    pub recipients: Vec<String>,
}

/// Global workspace state
#[derive(Debug, Clone)]
pub struct GlobalWorkspace {
    /// Current content in workspace
    pub content: Option<WorkspaceContent>,
    /// Competing contents (not yet in workspace)
    pub competing: Vec<WorkspaceContent>,
    /// Broadcast history
    pub history: Vec<WorkspaceContent>,
}

impl GlobalWorkspace {
    /// Create new empty workspace
    pub fn new() -> Self {
        Self {
            content: None,
            competing: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Check if workspace is occupied
    pub fn is_occupied(&self) -> bool {
        self.content.is_some()
    }

    /// Get current content
    pub fn current(&self) -> Option<&WorkspaceContent> {
        self.content.as_ref()
    }
}

impl Default for GlobalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute integrated information Φ
///
/// # Arguments
/// * `network` - Neural network state
///
/// # Returns
/// Φ value and minimum information partition
///
/// # Algorithm
/// 1. Generate all possible bipartitions of the system
/// 2. For each partition, compute mutual information I(X_1; X_2)
/// 3. Φ = minimum I across all partitions (minimum information partition)
///
/// # Example
/// ```rust,ignore
/// let network = NeuralState {
///     activations: vec![1.0, 0.5, 0.8],
///     connectivity: vec![...],
///     labels: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
/// };
///
/// let result = compute_phi(&network)?;
/// if result.is_conscious {
///     println!("System is conscious: Φ = {}", result.phi);
/// }
/// ```
pub fn compute_phi(network: &NeuralState) -> Result<PhiResult> {
    let n = network.activations.len();

    if n < 2 {
        return Ok(PhiResult {
            phi: 0.0,
            mip: None,
            complexity: 0.0,
            is_conscious: false,
        });
    }

    // Generate all non-trivial bipartitions
    let partitions = generate_bipartitions(n);

    // Find minimum information partition (MIP)
    let mut min_mi = f64::INFINITY;
    let mut mip = None;

    for partition in partitions {
        let mi = mutual_information_partition(network, &partition)?;
        if mi < min_mi {
            min_mi = mi;
            mip = Some(partition);
        }
    }

    // Φ is the minimum mutual information
    let phi = min_mi;

    Ok(PhiResult {
        phi,
        mip,
        complexity: phi, // Simplified: complexity ≈ Φ
        is_conscious: phi > PHI_THRESHOLD,
    })
}

/// Generate all bipartitions of n elements
fn generate_bipartitions(n: usize) -> Vec<Partition> {
    let mut partitions = Vec::new();

    // Iterate through all possible subsets (2^n - 2, excluding empty and full)
    for i in 1..(2_usize.pow(n as u32) - 1) {
        let mut subset_1 = Vec::new();
        let mut subset_2 = Vec::new();

        for j in 0..n {
            if (i & (1 << j)) != 0 {
                subset_1.push(j);
            } else {
                subset_2.push(j);
            }
        }

        partitions.push(Partition {
            subset_1,
            subset_2,
            mutual_information: 0.0,
        });
    }

    partitions
}

/// Compute mutual information across a partition
fn mutual_information_partition(
    network: &NeuralState,
    partition: &Partition,
) -> Result<f64> {
    // Simplified MI calculation using correlation
    // Full IIT requires computing effective information

    // Extract states for each subset
    let states_1: Vec<f64> = partition
        .subset_1
        .iter()
        .map(|&i| network.activations[i])
        .collect();

    let states_2: Vec<f64> = partition
        .subset_2
        .iter()
        .map(|&i| network.activations[i])
        .collect();

    // Compute entropies
    let h1 = entropy(&states_1);
    let h2 = entropy(&states_2);
    let h12 = joint_entropy(&states_1, &states_2);

    // MI = H(X1) + H(X2) - H(X1, X2)
    let mi = h1 + h2 - h12;

    Ok(mi.max(0.0))
}

/// Compute Shannon entropy of a state vector
fn entropy(states: &[f64]) -> f64 {
    if states.is_empty() {
        return 0.0;
    }

    // Discretize states into bins for probability estimation
    let bins = 10;
    let min_val = states.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = states.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_val == min_val {
        return 0.0;
    }

    let bin_width = (max_val - min_val) / bins as f64;
    let mut counts = vec![0; bins];

    for &s in states {
        let bin = ((s - min_val) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    // Compute entropy
    let n = states.len() as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            p * p.ln()
        })
        .sum::<f64>()
}

/// Compute joint entropy H(X, Y)
fn joint_entropy(states_x: &[f64], states_y: &[f64]) -> f64 {
    // Simplified: assume independence for now
    // H(X, Y) ≈ H(X) + H(Y) if independent
    entropy(states_x) + entropy(states_y)
}

/// Check if system is conscious
///
/// # Arguments
/// * `phi` - Integrated information value
///
/// # Returns
/// `true` if Φ > threshold
pub fn is_conscious(phi: f64) -> bool {
    phi > PHI_THRESHOLD
}

/// Broadcast to global workspace
///
/// # Arguments
/// * `content` - Content to broadcast
///
/// # Returns
/// Success indicator
///
/// # Example
/// ```rust,ignore
/// let content = CognitiveContent {
///     data: vec![0.5, 0.3, 0.8],
///     salience: 0.9,
/// };
///
/// broadcast_to_workspace(&content)?;
/// ```
pub fn broadcast_to_workspace(content: &WorkspaceContent) -> Result<()> {
    // TODO: Interface with global workspace system
    // 1. Check if workspace is available
    // 2. If occupied and new content has higher salience, replace
    // 3. Broadcast to all registered modules
    Ok(())
}

/// Access global workspace content
///
/// # Returns
/// Current workspace content if available
pub fn access_workspace() -> Result<Option<WorkspaceContent>> {
    // TODO: Interface with workspace
    Ok(None)
}

/// Compete for workspace access
///
/// # Arguments
/// * `contents` - Competing contents
///
/// # Returns
/// Winning content (highest salience)
pub fn compete_for_workspace(contents: &[WorkspaceContent]) -> Option<WorkspaceContent> {
    contents
        .iter()
        .max_by(|a, b| a.salience.partial_cmp(&b.salience).unwrap())
        .cloned()
}

/// Integrate information across modules
///
/// # Arguments
/// * `module_states` - States from different cognitive modules
///
/// # Returns
/// Integrated representation
pub fn integrate_information(module_states: &[Vec<f64>]) -> Result<Vec<f64>> {
    if module_states.is_empty() {
        return Ok(vec![]);
    }

    // Simple concatenation for now
    // Full integration would use attention-weighted combination
    Ok(module_states.iter().flatten().copied().collect())
}

/// Compute effective information (cause-effect power)
///
/// # Arguments
/// * `mechanism` - Neural mechanism (subset of network)
/// * `network` - Full network state
///
/// # Returns
/// Effective information value
pub fn effective_information(mechanism: &[usize], network: &NeuralState) -> Result<f64> {
    // EI = H(X_post) - H(X_post | X_pre)

    let mechanism_states: Vec<f64> = mechanism
        .iter()
        .map(|&i| network.activations[i])
        .collect();

    let h_post = entropy(&mechanism_states);

    // Conditional entropy approximation
    let h_conditional = h_post * 0.5; // Simplified

    let ei = h_post - h_conditional;

    Ok(ei.max(0.0))
}

/// Determine consciousness level (ordinal scale)
///
/// # Arguments
/// * `phi` - Integrated information
///
/// # Returns
/// Consciousness level (0-5)
pub fn consciousness_level(phi: f64) -> u8 {
    if phi < 0.1 {
        0 // Unconscious
    } else if phi < 0.5 {
        1 // Minimally conscious
    } else if phi < 1.0 {
        2 // Low consciousness
    } else if phi < 2.0 {
        3 // Moderate consciousness
    } else if phi < 5.0 {
        4 // High consciousness
    } else {
        5 // Very high consciousness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_threshold() {
        assert_eq!(PHI_THRESHOLD, 1.0);
    }

    #[test]
    fn test_is_conscious() {
        assert!(!is_conscious(0.5));
        assert!(is_conscious(1.5));
    }

    #[test]
    fn test_entropy_uniform() {
        let states = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let h = entropy(&states);
        assert!(h > 0.0); // Uniform distribution has high entropy
    }

    #[test]
    fn test_entropy_deterministic() {
        let states = vec![0.5, 0.5, 0.5, 0.5];
        let h = entropy(&states);
        assert_eq!(h, 0.0); // Deterministic state has zero entropy
    }

    #[test]
    fn test_global_workspace() {
        let workspace = GlobalWorkspace::new();
        assert!(!workspace.is_occupied());
        assert!(workspace.current().is_none());
    }

    #[test]
    fn test_compete_for_workspace() {
        let contents = vec![
            WorkspaceContent {
                id: "c1".to_string(),
                data: vec![],
                salience: 0.5,
                entry_time: 0.0,
                recipients: vec![],
            },
            WorkspaceContent {
                id: "c2".to_string(),
                data: vec![],
                salience: 0.9,
                entry_time: 0.0,
                recipients: vec![],
            },
        ];

        let winner = compete_for_workspace(&contents).unwrap();
        assert_eq!(winner.id, "c2");
    }

    #[test]
    fn test_consciousness_level() {
        assert_eq!(consciousness_level(0.05), 0);
        assert_eq!(consciousness_level(0.3), 1);
        assert_eq!(consciousness_level(0.8), 2);
        assert_eq!(consciousness_level(1.5), 3);
        assert_eq!(consciousness_level(3.0), 4);
        assert_eq!(consciousness_level(10.0), 5);
    }

    #[test]
    fn test_generate_bipartitions() {
        let partitions = generate_bipartitions(3);
        // For 3 nodes: 2^3 - 2 = 6 non-trivial bipartitions
        assert_eq!(partitions.len(), 6);
    }
}
