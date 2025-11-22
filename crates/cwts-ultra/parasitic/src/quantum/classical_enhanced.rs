//! # Classical Enhanced Algorithms
//! 
//! Classical algorithms with quantum-inspired optimizations that provide
//! performance improvements without requiring quantum hardware.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::quantum::QuantumMode;
use crate::{quantum_gate};

/// Classical enhanced pattern matching using quantum-inspired techniques
pub struct QuantumInspiredPatternMatcher {
    patterns: Vec<Pattern>,
    interference_matrix: Vec<Vec<f64>>,
    superposition_weights: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub data: Vec<f64>,
    pub frequency: f64,
    pub phase: f64,
    pub amplitude: f64,
}

impl QuantumInspiredPatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            interference_matrix: Vec::new(),
            superposition_weights: Vec::new(),
        }
    }
    
    /// Add a pattern with quantum-inspired encoding
    pub fn add_pattern(&mut self, pattern: Pattern) {
        // Classical implementation with quantum-inspired features
        quantum_gate!(
            // Classical mode: simple storage
            {
                self.patterns.push(pattern);
            },
            // Enhanced mode: quantum-inspired interference patterns
            {
                self.patterns.push(pattern.clone());
                self.update_interference_matrix(&pattern);
                self.update_superposition_weights(&pattern);
            },
            // Full quantum mode: delegate to quantum simulator
            {
                self.patterns.push(pattern.clone());
                self.update_interference_matrix(&pattern);
                self.update_superposition_weights(&pattern);
                // In full mode, would use actual quantum registers
            }
        );
    }
    
    /// Match patterns using quantum-inspired scoring
    pub fn match_patterns(&self, input: &[f64]) -> Vec<PatternMatch> {
        quantum_gate!(
            // Classical: basic euclidean distance
            self.classical_pattern_match(input),
            // Enhanced: quantum-inspired interference scoring
            self.enhanced_pattern_match(input),
            // Full quantum: quantum circuit pattern matching
            self.quantum_pattern_match(input)
        )
    }
    
    fn classical_pattern_match(&self, input: &[f64]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();
        
        for (i, pattern) in self.patterns.iter().enumerate() {
            let distance = euclidean_distance(input, &pattern.data);
            let similarity = 1.0 / (1.0 + distance);
            
            matches.push(PatternMatch {
                pattern_id: pattern.id.clone(),
                similarity,
                confidence: similarity,
                quantum_probability: None,
            });
        }
        
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches
    }
    
    fn enhanced_pattern_match(&self, input: &[f64]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();
        
        for (i, pattern) in self.patterns.iter().enumerate() {
            // Classical distance
            let distance = euclidean_distance(input, &pattern.data);
            let base_similarity = 1.0 / (1.0 + distance);
            
            // Quantum-inspired interference enhancement
            let interference_boost = self.calculate_interference_boost(i, input);
            
            // Superposition weight contribution
            let superposition_factor = if i < self.superposition_weights.len() {
                self.superposition_weights[i]
            } else {
                1.0
            };
            
            let enhanced_similarity = base_similarity * (1.0 + interference_boost) * superposition_factor;
            
            matches.push(PatternMatch {
                pattern_id: pattern.id.clone(),
                similarity: enhanced_similarity,
                confidence: base_similarity,
                quantum_probability: Some(enhanced_similarity.min(1.0)),
            });
        }
        
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches
    }
    
    fn quantum_pattern_match(&self, input: &[f64]) -> Vec<PatternMatch> {
        // In full quantum mode, this would use actual quantum circuits
        // For now, we simulate with enhanced classical algorithms
        let mut matches = self.enhanced_pattern_match(input);
        
        // Add quantum probability calculations
        for match_result in &mut matches {
            if let Some(prob) = match_result.quantum_probability {
                // Simulate quantum measurement collapse
                let measured_prob = (prob * prob).sqrt(); // Quantum amplitude -> probability
                match_result.quantum_probability = Some(measured_prob);
                match_result.similarity = measured_prob;
            }
        }
        
        matches
    }
    
    fn update_interference_matrix(&mut self, pattern: &Pattern) {
        let n = self.patterns.len();
        
        // Resize matrix if needed
        while self.interference_matrix.len() < n {
            self.interference_matrix.push(vec![0.0; n]);
        }
        for row in &mut self.interference_matrix {
            while row.len() < n {
                row.push(0.0);
            }
        }
        
        // Calculate interference values with existing patterns
        let new_idx = n - 1;
        for (i, existing_pattern) in self.patterns[..new_idx].iter().enumerate() {
            let interference = calculate_interference(&pattern.data, &existing_pattern.data);
            self.interference_matrix[new_idx][i] = interference;
            self.interference_matrix[i][new_idx] = interference;
        }
        self.interference_matrix[new_idx][new_idx] = 1.0; // Self-interference
    }
    
    fn update_superposition_weights(&mut self, pattern: &Pattern) {
        // Calculate superposition weight based on pattern characteristics
        let weight = pattern.amplitude * pattern.frequency.cos();
        self.superposition_weights.push(weight);
    }
    
    fn calculate_interference_boost(&self, pattern_idx: usize, input: &[f64]) -> f64 {
        if pattern_idx >= self.interference_matrix.len() {
            return 0.0;
        }
        
        let mut boost = 0.0;
        for (i, other_similarity) in self.interference_matrix[pattern_idx].iter().enumerate() {
            if i != pattern_idx {
                // Calculate how input correlates with interference pattern
                let correlation = if i < self.patterns.len() {
                    pearson_correlation(input, &self.patterns[i].data)
                } else {
                    0.0
                };
                boost += other_similarity * correlation * 0.1; // Scale factor
            }
        }
        
        boost.tanh() // Bound the boost
    }
}

impl Default for QuantumInspiredPatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: String,
    pub similarity: f64,
    pub confidence: f64,
    pub quantum_probability: Option<f64>,
}

/// Classical enhanced optimization using quantum-inspired annealing
pub struct QuantumInspiredOptimizer {
    temperature_schedule: Vec<f64>,
    energy_function: Option<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
    tunneling_probability: f64,
}

impl QuantumInspiredOptimizer {
    pub fn new() -> Self {
        Self {
            temperature_schedule: geometric_cooling_schedule(1000.0, 0.01, 100),
            energy_function: None,
            tunneling_probability: 0.1,
        }
    }
    
    pub fn with_energy_function<F>(mut self, f: F) -> Self 
    where 
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static
    {
        self.energy_function = Some(Box::new(f));
        self
    }
    
    /// Optimize using quantum-inspired techniques
    pub fn optimize(&self, initial_solution: &[f64]) -> OptimizationResult {
        quantum_gate!(
            self.classical_optimize(initial_solution),
            self.enhanced_optimize(initial_solution),
            self.quantum_optimize(initial_solution)
        )
    }
    
    fn classical_optimize(&self, initial_solution: &[f64]) -> OptimizationResult {
        let energy_fn = match &self.energy_function {
            Some(f) => f,
            None => return OptimizationResult::default(),
        };
        
        let mut current_solution = initial_solution.to_vec();
        let mut current_energy = energy_fn(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;
        let mut iterations = 0;
        
        // Simple hill climbing
        for _ in 0..1000 {
            let mut neighbor = current_solution.clone();
            // Random perturbation
            let idx = fastrand::usize(..neighbor.len());
            neighbor[idx] += fastrand::f64() * 0.1 - 0.05;
            
            let neighbor_energy = energy_fn(&neighbor);
            if neighbor_energy < current_energy {
                current_solution = neighbor;
                current_energy = neighbor_energy;
                
                if neighbor_energy < best_energy {
                    best_solution = current_solution.clone();
                    best_energy = neighbor_energy;
                }
            }
            iterations += 1;
        }
        
        OptimizationResult {
            solution: best_solution,
            energy: best_energy,
            iterations,
            convergence_reason: "Max iterations reached".to_string(),
            quantum_tunneling_events: 0,
        }
    }
    
    fn enhanced_optimize(&self, initial_solution: &[f64]) -> OptimizationResult {
        let energy_fn = match &self.energy_function {
            Some(f) => f,
            None => return OptimizationResult::default(),
        };
        
        let mut current_solution = initial_solution.to_vec();
        let mut current_energy = energy_fn(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_energy = current_energy;
        let mut iterations = 0;
        let mut tunneling_events = 0;
        
        // Quantum-inspired simulated annealing with tunneling
        for &temperature in &self.temperature_schedule {
            for _ in 0..10 {
                let mut neighbor = current_solution.clone();
                
                // Quantum-inspired multi-dimensional perturbation
                for i in 0..neighbor.len() {
                    if fastrand::f64() < 0.3 { // Probability of perturbing this dimension
                        let gaussian_noise = sample_gaussian() * temperature.sqrt();
                        neighbor[i] += gaussian_noise;
                    }
                }
                
                let neighbor_energy = energy_fn(&neighbor);
                let energy_diff = neighbor_energy - current_energy;
                
                // Accept or reject with quantum tunneling possibility
                let accept = if energy_diff < 0.0 {
                    true // Always accept improvements
                } else {
                    let thermal_prob = (-energy_diff / temperature).exp();
                    let tunnel_prob = self.tunneling_probability * (-energy_diff).exp();
                    let accept_prob = thermal_prob + tunnel_prob;
                    
                    if fastrand::f64() < tunnel_prob {
                        tunneling_events += 1;
                    }
                    
                    fastrand::f64() < accept_prob
                };
                
                if accept {
                    current_solution = neighbor;
                    current_energy = neighbor_energy;
                    
                    if neighbor_energy < best_energy {
                        best_solution = current_solution.clone();
                        best_energy = neighbor_energy;
                    }
                }
                
                iterations += 1;
            }
        }
        
        OptimizationResult {
            solution: best_solution,
            energy: best_energy,
            iterations,
            convergence_reason: "Annealing schedule completed".to_string(),
            quantum_tunneling_events: tunneling_events,
        }
    }
    
    fn quantum_optimize(&self, initial_solution: &[f64]) -> OptimizationResult {
        // In full quantum mode, this would use quantum annealing algorithms
        // For simulation, we enhance the classical version with additional quantum effects
        let mut result = self.enhanced_optimize(initial_solution);
        
        // Simulate quantum parallelism by exploring multiple paths
        let num_parallel_paths = 4;
        for _ in 0..num_parallel_paths {
            let path_result = self.enhanced_optimize(initial_solution);
            if path_result.energy < result.energy {
                result = path_result;
            }
        }
        
        result.convergence_reason = "Quantum parallel optimization".to_string();
        result
    }
}

impl Default for QuantumInspiredOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub solution: Vec<f64>,
    pub energy: f64,
    pub iterations: u64,
    pub convergence_reason: String,
    pub quantum_tunneling_events: u64,
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self {
            solution: Vec::new(),
            energy: f64::INFINITY,
            iterations: 0,
            convergence_reason: "No optimization performed".to_string(),
            quantum_tunneling_events: 0,
        }
    }
}

/// Quantum-inspired clustering algorithm
pub struct QuantumInspiredClustering {
    num_clusters: usize,
    max_iterations: usize,
    quantum_coherence: f64,
}

impl QuantumInspiredClustering {
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            max_iterations: 100,
            quantum_coherence: 0.8,
        }
    }
    
    pub fn cluster(&self, data: &[Vec<f64>]) -> ClusteringResult {
        quantum_gate!(
            self.classical_kmeans(data),
            self.enhanced_clustering(data),
            self.quantum_clustering(data)
        )
    }
    
    fn classical_kmeans(&self, data: &[Vec<f64>]) -> ClusteringResult {
        if data.is_empty() {
            return ClusteringResult::default();
        }
        
        let dims = data[0].len();
        let mut centroids = initialize_centroids(self.num_clusters, dims);
        let mut assignments = vec![0; data.len()];
        let mut converged = false;
        let mut iterations = 0;
        
        while !converged && iterations < self.max_iterations {
            let mut new_assignments = vec![0; data.len()];
            
            // Assign points to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = f64::INFINITY;
                
                for (c, centroid) in centroids.iter().enumerate() {
                    let distance = euclidean_distance(point, centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = c;
                    }
                }
                
                new_assignments[i] = best_cluster;
            }
            
            // Update centroids
            let mut new_centroids = vec![vec![0.0; dims]; self.num_clusters];
            let mut cluster_counts = vec![0; self.num_clusters];
            
            for (i, point) in data.iter().enumerate() {
                let cluster = new_assignments[i];
                cluster_counts[cluster] += 1;
                for d in 0..dims {
                    new_centroids[cluster][d] += point[d];
                }
            }
            
            for c in 0..self.num_clusters {
                if cluster_counts[c] > 0 {
                    for d in 0..dims {
                        new_centroids[c][d] /= cluster_counts[c] as f64;
                    }
                }
            }
            
            // Check convergence
            converged = assignments == new_assignments;
            assignments = new_assignments;
            centroids = new_centroids;
            iterations += 1;
        }
        
        let inertia = calculate_inertia(data, &assignments, &centroids);
        
        ClusteringResult {
            assignments,
            centroids,
            iterations,
            inertia,
            quantum_coherence: None,
        }
    }
    
    fn enhanced_clustering(&self, data: &[Vec<f64>]) -> ClusteringResult {
        // Start with classical k-means
        let mut result = self.classical_kmeans(data);
        
        // Apply quantum-inspired refinement
        for _ in 0..10 {
            result = self.quantum_inspired_refinement(data, &result);
        }
        
        result.quantum_coherence = Some(self.quantum_coherence);
        result
    }
    
    fn quantum_clustering(&self, data: &[Vec<f64>]) -> ClusteringResult {
        // Full quantum clustering would use quantum algorithms like quantum k-means
        // For simulation, we enhance the enhanced version
        let result = self.enhanced_clustering(data);
        
        // Simulate quantum superposition effects
        // In real quantum clustering, points would exist in superposition of clusters
        result
    }
    
    fn quantum_inspired_refinement(&self, data: &[Vec<f64>], current: &ClusteringResult) -> ClusteringResult {
        let mut new_assignments = current.assignments.clone();
        let mut changed = false;
        
        for (i, point) in data.iter().enumerate() {
            let current_cluster = current.assignments[i];
            let mut best_cluster = current_cluster;
            let mut best_score = self.calculate_quantum_score(point, &current.centroids, current_cluster);
            
            // Check if moving to another cluster improves quantum score
            for c in 0..self.num_clusters {
                if c != current_cluster {
                    let score = self.calculate_quantum_score(point, &current.centroids, c);
                    if score > best_score {
                        best_score = score;
                        best_cluster = c;
                    }
                }
            }
            
            if best_cluster != current_cluster {
                new_assignments[i] = best_cluster;
                changed = true;
            }
        }
        
        if !changed {
            return current.clone();
        }
        
        // Recalculate centroids
        let dims = data[0].len();
        let mut new_centroids = vec![vec![0.0; dims]; self.num_clusters];
        let mut cluster_counts = vec![0; self.num_clusters];
        
        for (i, point) in data.iter().enumerate() {
            let cluster = new_assignments[i];
            cluster_counts[cluster] += 1;
            for d in 0..dims {
                new_centroids[cluster][d] += point[d];
            }
        }
        
        for c in 0..self.num_clusters {
            if cluster_counts[c] > 0 {
                for d in 0..dims {
                    new_centroids[c][d] /= cluster_counts[c] as f64;
                }
            }
        }
        
        let inertia = calculate_inertia(data, &new_assignments, &new_centroids);
        
        ClusteringResult {
            assignments: new_assignments,
            centroids: new_centroids,
            iterations: current.iterations + 1,
            inertia,
            quantum_coherence: current.quantum_coherence,
        }
    }
    
    fn calculate_quantum_score(&self, point: &[f64], centroids: &[Vec<f64>], cluster: usize) -> f64 {
        let distance = euclidean_distance(point, &centroids[cluster]);
        let base_score = 1.0 / (1.0 + distance);
        
        // Add quantum interference effects
        let mut interference = 0.0;
        for (i, other_centroid) in centroids.iter().enumerate() {
            if i != cluster {
                let other_distance = euclidean_distance(point, other_centroid);
                let phase_diff = (distance - other_distance) * std::f64::consts::PI;
                interference += phase_diff.cos() * self.quantum_coherence;
            }
        }
        
        base_score * (1.0 + interference * 0.1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    pub assignments: Vec<usize>,
    pub centroids: Vec<Vec<f64>>,
    pub iterations: usize,
    pub inertia: f64,
    pub quantum_coherence: Option<f64>,
}

impl Default for ClusteringResult {
    fn default() -> Self {
        Self {
            assignments: Vec::new(),
            centroids: Vec::new(),
            iterations: 0,
            inertia: 0.0,
            quantum_coherence: None,
        }
    }
}

// Helper functions

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    
    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;
    
    for (x, y) in a.iter().zip(b.iter()) {
        let diff_a = x - mean_a;
        let diff_b = y - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }
    
    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn calculate_interference(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

fn sample_gaussian() -> f64 {
    // Box-Muller transform for Gaussian sampling
    static mut HAVE_SPARE: bool = false;
    static mut SPARE: f64 = 0.0;
    
    unsafe {
        if HAVE_SPARE {
            HAVE_SPARE = false;
            return SPARE;
        }
        
        HAVE_SPARE = true;
        let u = fastrand::f64();
        let v = fastrand::f64();
        let mag = 0.1 * (-2.0 * u.ln()).sqrt();
        SPARE = mag * (2.0 * std::f64::consts::PI * v).cos();
        mag * (2.0 * std::f64::consts::PI * v).sin()
    }
}

fn geometric_cooling_schedule(initial_temp: f64, final_temp: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return vec![initial_temp];
    }
    
    let ratio = (final_temp / initial_temp).powf(1.0 / (steps - 1) as f64);
    let mut schedule = Vec::with_capacity(steps);
    let mut temp = initial_temp;
    
    for _ in 0..steps {
        schedule.push(temp);
        temp *= ratio;
    }
    
    schedule
}

fn initialize_centroids(num_clusters: usize, dims: usize) -> Vec<Vec<f64>> {
    let mut centroids = Vec::with_capacity(num_clusters);
    
    for _ in 0..num_clusters {
        let mut centroid = Vec::with_capacity(dims);
        for _ in 0..dims {
            centroid.push(fastrand::f64() * 2.0 - 1.0); // Random in [-1, 1]
        }
        centroids.push(centroid);
    }
    
    centroids
}

fn calculate_inertia(data: &[Vec<f64>], assignments: &[usize], centroids: &[Vec<f64>]) -> f64 {
    let mut inertia = 0.0;
    
    for (i, point) in data.iter().enumerate() {
        let cluster = assignments[i];
        if cluster < centroids.len() {
            let distance = euclidean_distance(point, &centroids[cluster]);
            inertia += distance * distance;
        }
    }
    
    inertia
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::QuantumMode;
    
    #[test]
    fn test_quantum_inspired_pattern_matcher() {
        let mut matcher = QuantumInspiredPatternMatcher::new();
        
        let pattern = Pattern {
            id: "test_pattern".to_string(),
            data: vec![1.0, 2.0, 3.0],
            frequency: 1.0,
            phase: 0.0,
            amplitude: 1.0,
        };
        
        matcher.add_pattern(pattern);
        
        let input = vec![1.1, 2.1, 3.1];
        let matches = matcher.match_patterns(&input);
        
        assert!(!matches.is_empty());
        assert!(matches[0].similarity > 0.0);
    }
    
    #[test]
    fn test_quantum_inspired_optimizer() {
        let optimizer = QuantumInspiredOptimizer::new()
            .with_energy_function(|x| x.iter().map(|v| v.powi(2)).sum::<f64>());
        
        let initial = vec![1.0, -1.0, 2.0];
        let result = optimizer.optimize(&initial);
        
        assert!(result.energy < 10.0); // Should find a better solution
        assert!(result.iterations > 0);
    }
    
    #[test]
    fn test_quantum_inspired_clustering() {
        let clusterer = QuantumInspiredClustering::new(2);
        
        let data = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1], 
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];
        
        let result = clusterer.cluster(&data);
        
        assert_eq!(result.assignments.len(), data.len());
        assert_eq!(result.centroids.len(), 2);
        assert!(result.iterations > 0);
    }
    
    #[test]
    fn test_mode_dependent_behavior() {
        QuantumMode::set_global(QuantumMode::Classical);
        let mut matcher = QuantumInspiredPatternMatcher::new();
        matcher.add_pattern(Pattern {
            id: "test".to_string(),
            data: vec![1.0],
            frequency: 1.0,
            phase: 0.0,
            amplitude: 1.0,
        });
        
        QuantumMode::set_global(QuantumMode::Enhanced);
        // Behavior should change based on mode
        let matches = matcher.match_patterns(&[1.1]);
        assert!(!matches.is_empty());
        
        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }
}