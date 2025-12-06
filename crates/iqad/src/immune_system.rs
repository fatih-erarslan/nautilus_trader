//! Immune system algorithms for anomaly detection

use crate::types::ImmuneDetector;
use ndarray::Array1;
use rand::Rng;
use std::collections::VecDeque;
use tracing::{debug, info};

/// Immune system state
pub struct ImmuneSystem {
    /// Active detector set
    detectors: Vec<ImmuneDetector>,
    /// Normal patterns (self)
    normal_patterns: VecDeque<Array1<f64>>,
    /// Anomaly memory
    anomaly_memory: VecDeque<(Array1<f64>, f64)>,
    /// Configuration
    max_detectors: usize,
    max_normal_patterns: usize,
    max_anomaly_memory: usize,
    mutation_rate: f64,
    affinity_threshold: f64,
}

impl ImmuneSystem {
    /// Create a new immune system
    pub fn new(
        max_detectors: usize,
        max_normal_patterns: usize,
        max_anomaly_memory: usize,
        mutation_rate: f64,
        affinity_threshold: f64,
    ) -> Self {
        Self {
            detectors: Vec::with_capacity(max_detectors),
            normal_patterns: VecDeque::with_capacity(max_normal_patterns),
            anomaly_memory: VecDeque::with_capacity(max_anomaly_memory),
            max_detectors,
            max_normal_patterns,
            max_anomaly_memory,
            mutation_rate,
            affinity_threshold,
        }
    }
    
    /// Initialize detector set
    pub fn initialize_detectors(&mut self, dimension: usize) {
        let mut rng = rand::thread_rng();
        
        self.detectors.clear();
        for _ in 0..self.max_detectors {
            let pattern: Vec<f64> = (0..dimension)
                .map(|_| rng.gen::<f64>())
                .collect();
                
            let detector = ImmuneDetector {
                pattern,
                affinity_threshold: self.affinity_threshold,
                created_at: std::time::SystemTime::now(),
                activation_count: 0,
            };
            
            self.detectors.push(detector);
        }
        
        info!("Initialized {} detectors", self.detectors.len());
    }
    
    /// Learn a normal pattern
    pub fn learn_normal_pattern(&mut self, pattern: Array1<f64>) {
        // Check if similar pattern already exists
        for existing in &self.normal_patterns {
            let similarity = self.calculate_similarity(existing, &pattern);
            if similarity > 0.9 {
                debug!("Skipping duplicate normal pattern");
                return;
            }
        }
        
        // Add to normal patterns
        self.normal_patterns.push_back(pattern);
        
        // Maintain size limit
        if self.normal_patterns.len() > self.max_normal_patterns {
            self.normal_patterns.pop_front();
        }
        
        debug!("Learned normal pattern, total: {}", self.normal_patterns.len());
    }
    
    /// Memorize an anomaly
    pub fn memorize_anomaly(&mut self, pattern: Array1<f64>, score: f64) {
        // Check if similar anomaly already exists
        for (existing, _) in &self.anomaly_memory {
            let similarity = self.calculate_similarity(existing, &pattern);
            if similarity > 0.9 {
                debug!("Skipping duplicate anomaly");
                return;
            }
        }
        
        // Add to anomaly memory
        self.anomaly_memory.push_back((pattern, score));
        
        // Maintain size limit
        if self.anomaly_memory.len() > self.max_anomaly_memory {
            self.anomaly_memory.pop_front();
        }
        
        debug!("Memorized anomaly with score {:.4}, total: {}", score, self.anomaly_memory.len());
    }
    
    /// Update detector set based on new self pattern
    pub fn update_detectors(&mut self, new_self_pattern: &Array1<f64>, quantum_affinity_fn: impl Fn(&[f64], &[f64]) -> f64) {
        let mut rng = rand::thread_rng();
        let mut new_detectors = Vec::new();
        
        // Check each detector against new self pattern
        for detector in self.detectors.drain(..) {
            let affinity = quantum_affinity_fn(
                new_self_pattern.as_slice().unwrap(),
                &detector.pattern,
            );
            
            if affinity < self.affinity_threshold {
                // Keep detector (low affinity to self is good)
                new_detectors.push(detector);
            } else {
                // Generate new detector via mutation
                let mut new_pattern = detector.pattern.clone();
                
                // Apply mutations
                for val in &mut new_pattern {
                    if rng.gen::<f64>() < self.mutation_rate {
                        *val = (*val + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
                    }
                }
                
                let new_detector = ImmuneDetector {
                    pattern: new_pattern,
                    affinity_threshold: self.affinity_threshold,
                    created_at: std::time::SystemTime::now(),
                    activation_count: 0,
                };
                
                new_detectors.push(new_detector);
            }
        }
        
        // Apply additional mutations for diversity
        for detector in &mut new_detectors {
            if rng.gen::<f64>() < self.mutation_rate {
                for val in &mut detector.pattern {
                    *val = (*val + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
                }
            }
        }
        
        self.detectors = new_detectors;
        
        // Ensure we maintain detector count
        while self.detectors.len() < self.max_detectors {
            let pattern: Vec<f64> = (0..new_self_pattern.len())
                .map(|_| rng.gen::<f64>())
                .collect();
                
            let detector = ImmuneDetector {
                pattern,
                affinity_threshold: self.affinity_threshold,
                created_at: std::time::SystemTime::now(),
                activation_count: 0,
            };
            
            self.detectors.push(detector);
        }
    }
    
    /// Calculate detector activation for a pattern
    pub fn calculate_activation(&mut self, pattern: &Array1<f64>, distance_fn: impl Fn(&[f64], &[f64]) -> f64) -> f64 {
        if self.detectors.is_empty() {
            return 0.0;
        }
        
        let mut min_distance = f64::MAX;
        let mut activated_detector_idx = None;
        
        // Find closest detector
        for (idx, detector) in self.detectors.iter().enumerate() {
            let distance = distance_fn(
                pattern.as_slice().unwrap(),
                &detector.pattern,
            );
            
            if distance < min_distance {
                min_distance = distance;
                activated_detector_idx = Some(idx);
            }
        }
        
        // Update activation count
        if let Some(idx) = activated_detector_idx {
            self.detectors[idx].activation_count += 1;
        }
        
        // Convert distance to activation (closer = higher activation)
        let activation = (-min_distance / self.affinity_threshold).exp();
        activation.min(1.0)
    }
    
    /// Get detector affinities for a pattern
    pub fn get_detector_affinities(&self, pattern: &Array1<f64>, affinity_fn: impl Fn(&[f64], &[f64]) -> f64, top_k: usize) -> Vec<f64> {
        let mut affinities: Vec<f64> = self.detectors.iter()
            .map(|detector| affinity_fn(pattern.as_slice().unwrap(), &detector.pattern))
            .collect();
            
        // Sort in descending order and take top k
        affinities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        affinities.truncate(top_k);
        
        affinities
    }
    
    /// Check against anomaly memory
    pub fn check_anomaly_memory(&self, pattern: &Array1<f64>, affinity_fn: impl Fn(&[f64], &[f64]) -> f64) -> Option<f64> {
        let mut max_score: f64 = 0.0;
        
        for (anomaly_pattern, original_score) in &self.anomaly_memory {
            let affinity = affinity_fn(
                pattern.as_slice().unwrap(),
                anomaly_pattern.as_slice().unwrap(),
            );
            
            let weighted_score = affinity * original_score;
            max_score = max_score.max(weighted_score);
        }
        
        if max_score > 0.0 {
            Some(max_score)
        } else {
            None
        }
    }
    
    /// Calculate similarity between patterns
    fn calculate_similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            (dot_product / (norm_a * norm_b)).abs()
        } else {
            0.0
        }
    }
    
    /// Get immune system statistics
    pub fn stats(&self) -> ImmuneSystemStats {
        ImmuneSystemStats {
            num_detectors: self.detectors.len(),
            num_normal_patterns: self.normal_patterns.len(),
            num_anomalies: self.anomaly_memory.len(),
            avg_detector_activations: self.detectors.iter()
                .map(|d| d.activation_count as f64)
                .sum::<f64>() / self.detectors.len().max(1) as f64,
        }
    }
}

/// Immune system statistics
#[derive(Debug, Clone)]
pub struct ImmuneSystemStats {
    pub num_detectors: usize,
    pub num_normal_patterns: usize,
    pub num_anomalies: usize,
    pub avg_detector_activations: f64,
}

/// Affinity maturation for detector optimization
pub struct AffinityMaturation {
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
}

impl AffinityMaturation {
    /// Create new affinity maturation engine
    pub fn new(population_size: usize, mutation_rate: f64, crossover_rate: f64) -> Self {
        Self {
            population_size,
            mutation_rate,
            crossover_rate,
        }
    }
    
    /// Evolve detector population
    pub fn evolve_population(
        &self,
        detectors: &mut Vec<ImmuneDetector>,
        fitness_fn: impl Fn(&ImmuneDetector) -> f64,
    ) {
        let mut rng = rand::thread_rng();
        
        // Calculate fitness for all detectors
        let mut fitness_scores: Vec<(usize, f64)> = detectors.iter()
            .enumerate()
            .map(|(idx, detector)| (idx, fitness_fn(detector)))
            .collect();
            
        // Sort by fitness (descending)
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select top performers
        let elite_count = self.population_size / 4;
        let mut new_population = Vec::new();
        
        // Keep elite detectors
        for &(idx, _) in fitness_scores.iter().take(elite_count) {
            new_population.push(detectors[idx].clone());
        }
        
        // Generate new detectors through crossover and mutation
        while new_population.len() < self.population_size {
            // Tournament selection
            let parent1_idx = self.tournament_selection(&fitness_scores, &mut rng);
            let parent2_idx = self.tournament_selection(&fitness_scores, &mut rng);
            
            let parent1 = &detectors[parent1_idx];
            let parent2 = &detectors[parent2_idx];
            
            // Crossover
            let mut child = if rng.gen::<f64>() < self.crossover_rate {
                self.crossover(parent1, parent2, &mut rng)
            } else {
                parent1.clone()
            };
            
            // Mutation
            if rng.gen::<f64>() < self.mutation_rate {
                self.mutate(&mut child, &mut rng);
            }
            
            new_population.push(child);
        }
        
        // Replace old population
        *detectors = new_population;
    }
    
    /// Tournament selection
    fn tournament_selection(&self, fitness_scores: &[(usize, f64)], rng: &mut impl Rng) -> usize {
        let tournament_size = 3;
        let mut best_idx = fitness_scores[rng.gen_range(0..fitness_scores.len())].0;
        let mut best_fitness = fitness_scores[best_idx].1;
        
        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..fitness_scores.len());
            if fitness_scores[idx].1 > best_fitness {
                best_idx = fitness_scores[idx].0;
                best_fitness = fitness_scores[idx].1;
            }
        }
        
        best_idx
    }
    
    /// Crossover between two detectors
    fn crossover(&self, parent1: &ImmuneDetector, parent2: &ImmuneDetector, rng: &mut impl Rng) -> ImmuneDetector {
        let crossover_point = rng.gen_range(0..parent1.pattern.len());
        let mut child_pattern = Vec::new();
        
        for i in 0..parent1.pattern.len() {
            if i < crossover_point {
                child_pattern.push(parent1.pattern[i]);
            } else {
                child_pattern.push(parent2.pattern[i]);
            }
        }
        
        ImmuneDetector {
            pattern: child_pattern,
            affinity_threshold: (parent1.affinity_threshold + parent2.affinity_threshold) / 2.0,
            created_at: std::time::SystemTime::now(),
            activation_count: 0,
        }
    }
    
    /// Mutate a detector
    fn mutate(&self, detector: &mut ImmuneDetector, rng: &mut impl Rng) {
        for val in &mut detector.pattern {
            if rng.gen::<f64>() < 0.1 { // 10% chance per element
                *val = (*val + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
            }
        }
    }
}