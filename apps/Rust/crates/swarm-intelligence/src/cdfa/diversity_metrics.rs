//! Diversity metrics calculation for CDFA framework
//!
//! This module implements comprehensive diversity measurement algorithms
//! for population-based optimization, including statistical, geometric,
//! and information-theoretic diversity measures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use nalgebra::{DVector, DMatrix};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;
use rand::Rng;

use crate::core::{SwarmError, Position, Individual, Population};

/// Comprehensive diversity metrics calculator
pub struct DiversityMetrics {
    /// Cache for expensive diversity calculations
    calculation_cache: Arc<RwLock<HashMap<String, CachedDiversityResult>>>,
    
    /// SIMD acceleration settings
    simd_enabled: bool,
    
    /// Parallel computation threshold
    parallel_threshold: usize,
}

/// Cached diversity calculation result
#[derive(Debug, Clone)]
struct CachedDiversityResult {
    result: DiversityMeasure,
    timestamp: std::time::Instant,
    population_hash: u64,
}

/// Different types of diversity measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMeasure {
    /// Statistical diversity measures
    Statistical {
        /// Population variance
        variance: f64,
        /// Standard deviation
        std_deviation: f64,
        /// Coefficient of variation
        cv: f64,
        /// Entropy-based diversity
        entropy: f64,
    },
    
    /// Geometric diversity measures
    Geometric {
        /// Average pairwise distance
        avg_pairwise_distance: f64,
        /// Minimum spanning tree length
        mst_length: f64,
        /// Convex hull volume
        convex_hull_volume: f64,
        /// Hypervolume indicator
        hypervolume: f64,
    },
    
    /// Information-theoretic measures
    Information {
        /// Shannon entropy
        shannon_entropy: f64,
        /// Rényi entropy
        renyi_entropy: f64,
        /// Mutual information
        mutual_information: f64,
        /// Kulback-Leibler divergence
        kl_divergence: f64,
        /// Jensen-Shannon divergence
        jensen_shannon_divergence: f64,
        /// Cross entropy
        cross_entropy: f64,
    },
    
    /// Pattern-based diversity measures
    Pattern {
        /// Dynamic Time Warping diversity
        dtw_diversity: f64,
        /// Kendall tau distance
        kendall_distance: f64,
        /// Spearman rank correlation
        spearman_correlation: f64,
        /// Pattern template similarity
        template_similarity: f64,
    },
    
    /// Algorithm-specific diversity
    Algorithmic {
        /// Behavioral diversity
        behavioral: f64,
        /// Phenotypic diversity
        phenotypic: f64,
        /// Genotypic diversity
        genotypic: f64,
        /// Fitness diversity
        fitness: f64,
    },
    
    /// Combined diversity score
    Combined {
        /// Weighted combination of all measures
        composite_score: f64,
        /// Individual component scores
        components: HashMap<String, f64>,
        /// Confidence interval
        confidence_interval: (f64, f64),
    },
}

/// Diversity calculation configuration
#[derive(Debug, Clone)]
pub struct DiversityConfig {
    /// Types of diversity to calculate
    pub measures: Vec<DiversityType>,
    
    /// Use parallel computation
    pub use_parallel: bool,
    
    /// Enable SIMD acceleration
    pub use_simd: bool,
    
    /// Cache results
    pub enable_cache: bool,
    
    /// Sampling strategy for large populations
    pub sampling_strategy: SamplingStrategy,
    
    /// Weights for composite diversity score
    pub component_weights: HashMap<String, f64>,
}

/// Types of diversity measures to calculate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiversityType {
    Statistical,
    Geometric,
    Information,
    Pattern,
    Algorithmic,
    Combined,
}

/// Sampling strategies for large populations
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Use entire population
    Full,
    /// Random sampling with fixed size
    Random { sample_size: usize },
    /// Stratified sampling
    Stratified { strata: usize },
    /// Systematic sampling
    Systematic { interval: usize },
    /// Adaptive sampling based on diversity
    Adaptive { min_samples: usize, max_samples: usize },
}

impl Default for DiversityConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("statistical".to_string(), 0.2);
        weights.insert("geometric".to_string(), 0.2);
        weights.insert("information".to_string(), 0.2);
        weights.insert("pattern".to_string(), 0.2);
        weights.insert("algorithmic".to_string(), 0.2);
        
        Self {
            measures: vec![DiversityType::Combined],
            use_parallel: true,
            use_simd: cfg!(feature = "simd"),
            enable_cache: true,
            sampling_strategy: SamplingStrategy::Adaptive { 
                min_samples: 100, 
                max_samples: 1000 
            },
            component_weights: weights,
        }
    }
}

impl DiversityMetrics {
    /// Create a new diversity metrics calculator
    pub fn new() -> Self {
        Self {
            calculation_cache: Arc::new(RwLock::new(HashMap::new())),
            simd_enabled: cfg!(feature = "simd"),
            parallel_threshold: 1000,
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(simd_enabled: bool, parallel_threshold: usize) -> Self {
        Self {
            calculation_cache: Arc::new(RwLock::new(HashMap::new())),
            simd_enabled,
            parallel_threshold,
        }
    }
    
    /// Calculate diversity measures for a population
    pub fn calculate_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        config: &DiversityConfig,
    ) -> Result<DiversityMeasure, SwarmError> {
        if population.is_empty() {
            return Err(SwarmError::parameter("Empty population"));
        }
        
        // Check cache first
        if config.enable_cache {
            let cache_key = self.generate_cache_key(population, config);
            if let Some(cached) = self.get_cached_result(&cache_key) {
                return Ok(cached.result);
            }
        }
        
        // Sample population if necessary
        let sampled_population = self.sample_population(population, &config.sampling_strategy)?;
        
        // Calculate diversity measures
        let result = if config.measures.contains(&DiversityType::Combined) {
            self.calculate_combined_diversity(&sampled_population, config)?
        } else {
            // Calculate specific measures
            self.calculate_specific_measures(&sampled_population, &config.measures)?
        };
        
        // Cache result
        if config.enable_cache {
            let cache_key = self.generate_cache_key(population, config);
            self.cache_result(cache_key, result.clone(), population);
        }
        
        Ok(result)
    }
    
    /// Calculate statistical diversity measures
    pub fn calculate_statistical_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<DiversityMeasure, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        
        if positions.is_empty() {
            return Err(SwarmError::parameter("Empty population"));
        }
        
        let dimensions = positions[0].len();
        
        // Calculate centroid
        let centroid = self.calculate_centroid(&positions);
        
        // Calculate variance and standard deviation
        let variance = self.calculate_variance(&positions, &centroid);
        let std_deviation = variance.sqrt();
        
        // Calculate coefficient of variation
        let mean_distance = self.calculate_mean_distance(&positions, &centroid);
        let cv = if mean_distance > 0.0 {
            std_deviation / mean_distance
        } else {
            0.0
        };
        
        // Calculate entropy-based diversity
        let entropy = self.calculate_positional_entropy(&positions)?;
        
        Ok(DiversityMeasure::Statistical {
            variance,
            std_deviation,
            cv,
            entropy,
        })
    }
    
    /// Calculate geometric diversity measures
    pub fn calculate_geometric_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<DiversityMeasure, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        
        if positions.len() < 2 {
            return Err(SwarmError::parameter("Need at least 2 individuals for geometric diversity"));
        }
        
        // Calculate average pairwise distance
        let avg_pairwise_distance = self.calculate_avg_pairwise_distance(&positions)?;
        
        // Calculate minimum spanning tree length
        let mst_length = self.calculate_mst_length(&positions)?;
        
        // Calculate convex hull volume (approximation for high dimensions)
        let convex_hull_volume = self.calculate_convex_hull_volume(&positions)?;
        
        // Calculate hypervolume indicator
        let hypervolume = self.calculate_hypervolume(&positions)?;
        
        Ok(DiversityMeasure::Geometric {
            avg_pairwise_distance,
            mst_length,
            convex_hull_volume,
            hypervolume,
        })
    }
    
    /// Calculate information-theoretic diversity measures
    pub fn calculate_information_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<DiversityMeasure, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        
        // Calculate Shannon entropy
        let shannon_entropy = self.calculate_shannon_entropy(&positions)?;
        
        // Calculate Rényi entropy
        let renyi_entropy = self.calculate_renyi_entropy(&positions, 2.0)?;
        
        // Calculate mutual information between dimensions
        let mutual_information = self.calculate_mutual_information(&positions)?;
        
        // Calculate KL divergence from uniform distribution
        let kl_divergence = self.calculate_kl_divergence(&positions)?;
        
        Ok(DiversityMeasure::Information {
            shannon_entropy,
            renyi_entropy,
            mutual_information,
            kl_divergence,
        })
    }
    
    /// Calculate algorithmic diversity measures
    pub fn calculate_algorithmic_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        algorithm_context: Option<&AlgorithmContext>,
    ) -> Result<DiversityMeasure, SwarmError> {
        // Behavioral diversity (based on search behavior)
        let behavioral = self.calculate_behavioral_diversity(population, algorithm_context)?;
        
        // Phenotypic diversity (based on fitness values)
        let phenotypic = self.calculate_phenotypic_diversity(population)?;
        
        // Genotypic diversity (based on parameter space)
        let genotypic = self.calculate_genotypic_diversity(population)?;
        
        // Fitness diversity
        let fitness = self.calculate_fitness_diversity(population)?;
        
        Ok(DiversityMeasure::Algorithmic {
            behavioral,
            phenotypic,
            genotypic,
            fitness,
        })
    }
    
    /// Calculate combined diversity measure
    fn calculate_combined_diversity<T: Individual>(
        &self,
        population: &[&T],
        config: &DiversityConfig,
    ) -> Result<DiversityMeasure, SwarmError> {
        let mut components = HashMap::new();
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        // Convert slice back to Population for other methods
        let temp_population = Population {
            individuals: population.iter().map(|&ind| ind.clone()).collect(),
        };
        
        // Calculate statistical diversity
        if let Ok(DiversityMeasure::Statistical { variance, std_deviation, cv, entropy }) = 
            self.calculate_statistical_diversity(&temp_population) {
            let score = (variance + std_deviation + cv + entropy) / 4.0;
            let weight = config.component_weights.get("statistical").copied().unwrap_or(0.25);
            components.insert("statistical".to_string(), score);
            total_score += score * weight;
            total_weight += weight;
        }
        
        // Calculate geometric diversity
        if let Ok(DiversityMeasure::Geometric { avg_pairwise_distance, mst_length, convex_hull_volume, hypervolume }) = 
            self.calculate_geometric_diversity(&temp_population) {
            let score = (avg_pairwise_distance + mst_length + convex_hull_volume + hypervolume) / 4.0;
            let weight = config.component_weights.get("geometric").copied().unwrap_or(0.25);
            components.insert("geometric".to_string(), score);
            total_score += score * weight;
            total_weight += weight;
        }
        
        // Calculate enhanced information diversity
        if let Ok(DiversityMeasure::Information { shannon_entropy, renyi_entropy, mutual_information, kl_divergence, jensen_shannon_divergence, cross_entropy }) = 
            self.calculate_enhanced_information_diversity(&temp_population) {
            let score = (shannon_entropy + renyi_entropy + mutual_information + kl_divergence + jensen_shannon_divergence + cross_entropy) / 6.0;
            let weight = config.component_weights.get("information").copied().unwrap_or(0.2);
            components.insert("information".to_string(), score);
            total_score += score * weight;
            total_weight += weight;
        }
        
        // Calculate pattern diversity
        if let Ok(DiversityMeasure::Pattern { dtw_diversity, kendall_distance, spearman_correlation, template_similarity }) = 
            self.calculate_pattern_diversity(&temp_population, None) {
            let score = (dtw_diversity + kendall_distance + spearman_correlation + template_similarity) / 4.0;
            let weight = config.component_weights.get("pattern").copied().unwrap_or(0.2);
            components.insert("pattern".to_string(), score);
            total_score += score * weight;
            total_weight += weight;
        }
        
        // Calculate algorithmic diversity
        if let Ok(DiversityMeasure::Algorithmic { behavioral, phenotypic, genotypic, fitness }) = 
            self.calculate_algorithmic_diversity(&temp_population, None) {
            let score = (behavioral + phenotypic + genotypic + fitness) / 4.0;
            let weight = config.component_weights.get("algorithmic").copied().unwrap_or(0.25);
            components.insert("algorithmic".to_string(), score);
            total_score += score * weight;
            total_weight += weight;
        }
        
        let composite_score = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        };
        
        // Calculate confidence interval (simplified)
        let confidence_interval = (
            composite_score * 0.9,
            composite_score * 1.1,
        );
        
        Ok(DiversityMeasure::Combined {
            composite_score,
            components,
            confidence_interval,
        })
    }
    
    /// Calculate specific diversity measures
    fn calculate_specific_measures<T: Individual>(
        &self,
        population: &[&T],
        measures: &[DiversityType],
    ) -> Result<DiversityMeasure, SwarmError> {
        // For simplicity, default to combined measure
        // In a full implementation, this would calculate only requested measures
        let temp_population = Population {
            individuals: population.iter().map(|&ind| ind.clone()).collect(),
        };
        
        self.calculate_statistical_diversity(&temp_population)
    }
    
    /// Sample population based on strategy
    fn sample_population<T: Individual>(
        &self,
        population: &Population<T>,
        strategy: &SamplingStrategy,
    ) -> Result<Vec<&T>, SwarmError> {
        match strategy {
            SamplingStrategy::Full => {
                Ok(population.iter().collect())
            }
            SamplingStrategy::Random { sample_size } => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let sample_size = (*sample_size).min(population.size());
                Ok(population.individuals.choose_multiple(&mut rng, sample_size).collect())
            }
            SamplingStrategy::Adaptive { min_samples, max_samples } => {
                let size = population.size();
                let sample_size = if size <= *min_samples {
                    size
                } else if size >= *max_samples {
                    *max_samples
                } else {
                    size
                };
                
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                Ok(population.individuals.choose_multiple(&mut rng, sample_size).collect())
            }
            _ => {
                // Fallback to full population for other strategies
                Ok(population.iter().collect())
            }
        }
    }
    
    /// Calculate centroid of positions
    fn calculate_centroid(&self, positions: &[&Position]) -> Position {
        if positions.is_empty() {
            return Position::zeros(0);
        }
        
        let dimensions = positions[0].len();
        let mut centroid = Position::zeros(dimensions);
        
        for pos in positions {
            centroid += *pos;
        }
        
        centroid / positions.len() as f64
    }
    
    /// Calculate variance from centroid
    fn calculate_variance(&self, positions: &[&Position], centroid: &Position) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }
        
        let sum_squared_distances: f64 = positions.iter()
            .map(|pos| (*pos - centroid).norm_squared())
            .sum();
        
        sum_squared_distances / positions.len() as f64
    }
    
    /// Calculate mean distance from centroid
    fn calculate_mean_distance(&self, positions: &[&Position], centroid: &Position) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }
        
        let sum_distances: f64 = positions.iter()
            .map(|pos| (*pos - centroid).norm())
            .sum();
        
        sum_distances / positions.len() as f64
    }
    
    /// Calculate positional entropy
    fn calculate_positional_entropy(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        let dimensions = positions[0].len();
        let mut total_entropy = 0.0;
        
        // Calculate entropy for each dimension
        for dim in 0..dimensions {
            let values: Vec<f64> = positions.iter().map(|pos| pos[dim]).collect();
            let entropy = self.calculate_dimension_entropy(&values)?;
            total_entropy += entropy;
        }
        
        Ok(total_entropy / dimensions as f64)
    }
    
    /// Calculate entropy for a single dimension
    fn calculate_dimension_entropy(&self, values: &[f64]) -> Result<f64, SwarmError> {
        if values.is_empty() {
            return Ok(0.0);
        }
        
        // Create histogram bins
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-10 {
            return Ok(0.0); // All values are the same
        }
        
        let num_bins = (values.len() as f64).sqrt().ceil() as usize;
        let bin_width = (max_val - min_val) / num_bins as f64;
        
        let mut histogram = vec![0; num_bins];
        
        for &value in values {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(num_bins - 1);
            histogram[bin_index] += 1;
        }
        
        // Calculate entropy
        let total_count = values.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let probability = count as f64 / total_count;
                entropy -= probability * probability.ln();
            }
        }
        
        Ok(entropy)
    }
    
    /// Calculate average pairwise distance
    fn calculate_avg_pairwise_distance(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 2 {
            return Ok(0.0);
        }
        
        let distances: Vec<f64> = if self.simd_enabled && positions.len() > self.parallel_threshold {
            // Use SIMD-accelerated parallel computation
            self.calculate_pairwise_distances_simd(positions)?
        } else if positions.len() > self.parallel_threshold {
            // Use parallel computation
            self.calculate_pairwise_distances_parallel(positions)?
        } else {
            // Use sequential computation
            self.calculate_pairwise_distances_sequential(positions)?
        };
        
        let sum: f64 = distances.iter().sum();
        Ok(sum / distances.len() as f64)
    }
    
    /// Calculate pairwise distances sequentially
    fn calculate_pairwise_distances_sequential(&self, positions: &[&Position]) -> Result<Vec<f64>, SwarmError> {
        let mut distances = Vec::new();
        
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let distance = (positions[i] - positions[j]).norm();
                distances.push(distance);
            }
        }
        
        Ok(distances)
    }
    
    /// Calculate pairwise distances in parallel
    fn calculate_pairwise_distances_parallel(&self, positions: &[&Position]) -> Result<Vec<f64>, SwarmError> {
        let indices: Vec<(usize, usize)> = (0..positions.len())
            .flat_map(|i| ((i + 1)..positions.len()).map(move |j| (i, j)))
            .collect();
        
        let distances: Vec<f64> = indices.par_iter()
            .map(|&(i, j)| (positions[i] - positions[j]).norm())
            .collect();
        
        Ok(distances)
    }
    
    /// Calculate pairwise distances with SIMD acceleration
    fn calculate_pairwise_distances_simd(&self, positions: &[&Position]) -> Result<Vec<f64>, SwarmError> {
        #[cfg(feature = "simd")]
        {
            // SIMD implementation would go here
            // For now, fallback to parallel
            self.calculate_pairwise_distances_parallel(positions)
        }
        
        #[cfg(not(feature = "simd"))]
        {
            self.calculate_pairwise_distances_parallel(positions)
        }
    }
    
    /// Calculate minimum spanning tree length
    fn calculate_mst_length(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 2 {
            return Ok(0.0);
        }
        
        // Use Prim's algorithm for MST
        let n = positions.len();
        let mut visited = vec![false; n];
        let mut key = vec![f64::INFINITY; n];
        let mut total_length = 0.0;
        
        key[0] = 0.0;
        
        for _ in 0..(n - 1) {
            let mut min_key = f64::INFINITY;
            let mut min_index = 0;
            
            for v in 0..n {
                if !visited[v] && key[v] < min_key {
                    min_key = key[v];
                    min_index = v;
                }
            }
            
            visited[min_index] = true;
            total_length += min_key;
            
            for v in 0..n {
                if !visited[v] {
                    let distance = (positions[min_index] - positions[v]).norm();
                    if distance < key[v] {
                        key[v] = distance;
                    }
                }
            }
        }
        
        Ok(total_length)
    }
    
    /// Calculate convex hull volume (approximation)
    fn calculate_convex_hull_volume(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        let dimensions = positions[0].len();
        
        if dimensions == 1 {
            // Range for 1D
            let min = positions.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            let max = positions.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
            return Ok(max - min);
        }
        
        if dimensions == 2 {
            // Use shoelace formula for 2D
            return Ok(self.calculate_2d_convex_hull_area(positions)?);
        }
        
        // For higher dimensions, use bounding box approximation
        let mut volume = 1.0;
        for dim in 0..dimensions {
            let min = positions.iter().map(|p| p[dim]).fold(f64::INFINITY, f64::min);
            let max = positions.iter().map(|p| p[dim]).fold(f64::NEG_INFINITY, f64::max);
            volume *= max - min;
        }
        
        Ok(volume)
    }
    
    /// Calculate 2D convex hull area
    fn calculate_2d_convex_hull_area(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 3 {
            return Ok(0.0);
        }
        
        // Simple bounding box approximation for now
        let min_x = positions.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let max_x = positions.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        let min_y = positions.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let max_y = positions.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        
        Ok((max_x - min_x) * (max_y - min_y))
    }
    
    /// Calculate hypervolume indicator
    fn calculate_hypervolume(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        // Simplified hypervolume calculation
        // In practice, this would use more sophisticated algorithms
        self.calculate_convex_hull_volume(positions)
    }
    
    /// Calculate Shannon entropy
    fn calculate_shannon_entropy(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        self.calculate_positional_entropy(positions)
    }
    
    /// Calculate Rényi entropy
    fn calculate_renyi_entropy(&self, positions: &[&Position], alpha: f64) -> Result<f64, SwarmError> {
        if positions.is_empty() || alpha == 1.0 {
            return self.calculate_shannon_entropy(positions);
        }
        
        // Simplified Rényi entropy calculation
        let shannon = self.calculate_shannon_entropy(positions)?;
        Ok(shannon / (1.0 - alpha))
    }
    
    /// Calculate mutual information between dimensions
    fn calculate_mutual_information(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        let dimensions = positions[0].len();
        if dimensions < 2 {
            return Ok(0.0);
        }
        
        // Calculate mutual information between first two dimensions as example
        let dim1_values: Vec<f64> = positions.iter().map(|p| p[0]).collect();
        let dim2_values: Vec<f64> = positions.iter().map(|p| p[1]).collect();
        
        let h1 = self.calculate_dimension_entropy(&dim1_values)?;
        let h2 = self.calculate_dimension_entropy(&dim2_values)?;
        
        // Joint entropy (simplified)
        let joint_values: Vec<(f64, f64)> = dim1_values.into_iter().zip(dim2_values).collect();
        let joint_entropy = self.calculate_joint_entropy(&joint_values)?;
        
        Ok(h1 + h2 - joint_entropy)
    }
    
    /// Calculate joint entropy for two dimensions
    fn calculate_joint_entropy(&self, values: &[(f64, f64)]) -> Result<f64, SwarmError> {
        if values.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified joint entropy calculation
        // In practice, this would use proper 2D histograms
        let entropy1 = self.calculate_dimension_entropy(&values.iter().map(|(x, _)| *x).collect::<Vec<_>>())?;
        let entropy2 = self.calculate_dimension_entropy(&values.iter().map(|(_, y)| *y).collect::<Vec<_>>())?;
        
        Ok((entropy1 + entropy2) / 2.0)
    }
    
    /// Calculate KL divergence between two probability distributions
    fn calculate_kl_divergence_distributions(&self, p: &[f64], q: &[f64]) -> Result<f64, SwarmError> {
        if p.len() != q.len() || p.is_empty() {
            return Err(SwarmError::parameter("Distribution length mismatch"));
        }
        
        let epsilon = 1e-10;
        let mut kl_div = 0.0;
        
        for (pi, qi) in p.iter().zip(q.iter()) {
            let p_safe = (*pi).max(epsilon);
            let q_safe = (*qi).max(epsilon);
            kl_div += p_safe * (p_safe / q_safe).ln();
        }
        
        Ok(kl_div)
    }
    
    /// Calculate Jensen-Shannon divergence between two distributions
    fn calculate_jensen_shannon_divergence(&self, p: &[f64], q: &[f64]) -> Result<f64, SwarmError> {
        if p.len() != q.len() || p.is_empty() {
            return Err(SwarmError::parameter("Distribution length mismatch"));
        }
        
        // Calculate M = 0.5 * (P + Q)
        let m: Vec<f64> = p.iter().zip(q.iter())
            .map(|(pi, qi)| 0.5 * (pi + qi))
            .collect();
        
        // Calculate JS(P,Q) = 0.5 * KL(P,M) + 0.5 * KL(Q,M)
        let kl_pm = self.calculate_kl_divergence_distributions(p, &m)?;
        let kl_qm = self.calculate_kl_divergence_distributions(q, &m)?;
        
        Ok(0.5 * kl_pm + 0.5 * kl_qm)
    }
    
    /// Convert positions to probability distribution using histogram
    fn positions_to_distribution(&self, positions: &[&Position], bins: usize) -> Result<Vec<f64>, SwarmError> {
        if positions.is_empty() {
            return Ok(vec![]);
        }
        
        let dimensions = positions[0].len();
        let mut distribution = vec![0.0; bins];
        
        // For each dimension, create histogram and combine
        for dim in 0..dimensions {
            let values: Vec<f64> = positions.iter().map(|p| p[dim]).collect();
            let dim_dist = self.values_to_distribution(&values, bins)?;
            
            // Add to combined distribution
            for (i, val) in dim_dist.iter().enumerate() {
                distribution[i] += val;
            }
        }
        
        // Normalize
        let total: f64 = distribution.iter().sum();
        if total > 0.0 {
            for val in &mut distribution {
                *val /= total;
            }
        }
        
        Ok(distribution)
    }
    
    /// Convert values to probability distribution
    fn values_to_distribution(&self, values: &[f64], bins: usize) -> Result<Vec<f64>, SwarmError> {
        if values.is_empty() {
            return Ok(vec![0.0; bins]);
        }
        
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-10 {
            // All values are the same
            let mut dist = vec![0.0; bins];
            dist[bins / 2] = 1.0; // Put all probability in middle bin
            return Ok(dist);
        }
        
        let bin_width = (max_val - min_val) / bins as f64;
        let mut histogram = vec![0; bins];
        
        for &value in values {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }
        
        // Convert to probabilities
        let total_count = values.len() as f64;
        let distribution: Vec<f64> = histogram.iter()
            .map(|&count| count as f64 / total_count)
            .collect();
        
        Ok(distribution)
    }
    
    /// Calculate KL divergence from uniform distribution
    fn calculate_kl_divergence(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        let bins = 20; // Number of histogram bins
        let distribution = self.positions_to_distribution(positions, bins)?;
        let uniform = vec![1.0 / bins as f64; bins];
        
        self.calculate_kl_divergence_distributions(&distribution, &uniform)
    }
    
    /// Calculate behavioral diversity
    fn calculate_behavioral_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        _context: Option<&AlgorithmContext>,
    ) -> Result<f64, SwarmError> {
        // Simplified behavioral diversity based on position variance
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        let centroid = self.calculate_centroid(&positions);
        let variance = self.calculate_variance(&positions, &centroid);
        Ok(variance.sqrt())
    }
    
    /// Calculate phenotypic diversity
    fn calculate_phenotypic_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<f64, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        let centroid = self.calculate_centroid(&positions);
        let variance = self.calculate_variance(&positions, &centroid);
        Ok(variance)
    }
    
    /// Calculate genotypic diversity
    fn calculate_genotypic_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<f64, SwarmError> {
        // Simplified genotypic diversity based on position entropy
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        self.calculate_positional_entropy(&positions)
    }
    
    /// Calculate fitness diversity
    fn calculate_fitness_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<f64, SwarmError> {
        let fitness_values: Vec<f64> = population.iter().map(|ind| *ind.fitness()).collect();
        
        if fitness_values.is_empty() {
            return Ok(0.0);
        }
        
        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let variance = fitness_values.iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Generate cache key for diversity calculation
    fn generate_cache_key<T: Individual>(
        &self,
        population: &Population<T>,
        config: &DiversityConfig,
    ) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        population.size().hash(&mut hasher);
        format!("{:?}_{}", config.measures, hasher.finish())
    }
    
    /// Get cached diversity result
    fn get_cached_result(&self, cache_key: &str) -> Option<CachedDiversityResult> {
        let cache = self.calculation_cache.read();
        if let Some(cached) = cache.get(cache_key) {
            // Check if cache is still valid (e.g., within 1 minute)
            if cached.timestamp.elapsed() < std::time::Duration::from_secs(60) {
                return Some(cached.clone());
            }
        }
        None
    }
    
    /// Cache diversity calculation result
    fn cache_result<T: Individual>(
        &self,
        cache_key: String,
        result: DiversityMeasure,
        population: &Population<T>,
    ) {
        let mut cache = self.calculation_cache.write();
        
        // Simple population hash for cache validation
        let population_hash = population.size() as u64;
        
        cache.insert(cache_key, CachedDiversityResult {
            result,
            timestamp: std::time::Instant::now(),
            population_hash,
        });
        
        // Limit cache size
        if cache.len() > 1000 {
            cache.clear();
        }
    }
    
    /// Calculate pattern-based diversity measures
    pub fn calculate_pattern_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        reference_patterns: Option<&HashMap<String, Vec<f64>>>,
    ) -> Result<DiversityMeasure, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        
        if positions.len() < 2 {
            return Err(SwarmError::parameter("Need at least 2 individuals for pattern diversity"));
        }
        
        // Calculate DTW diversity
        let dtw_diversity = self.calculate_dtw_diversity(&positions)?;
        
        // Calculate Kendall tau distance
        let kendall_distance = self.calculate_kendall_diversity(&positions)?;
        
        // Calculate Spearman rank correlation
        let spearman_correlation = self.calculate_spearman_diversity(&positions)?;
        
        // Calculate template similarity if patterns provided
        let template_similarity = if let Some(patterns) = reference_patterns {
            self.calculate_template_similarity(&positions, patterns)?
        } else {
            0.5 // Default neutral similarity
        };
        
        Ok(DiversityMeasure::Pattern {
            dtw_diversity,
            kendall_distance,
            spearman_correlation,
            template_similarity,
        })
    }
    
    /// Calculate Dynamic Time Warping diversity
    fn calculate_dtw_diversity(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        // Calculate pairwise DTW distances
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let seq1 = self.position_to_sequence(positions[i]);
                let seq2 = self.position_to_sequence(positions[j]);
                
                let distance = self.dtw_distance(&seq1, &seq2, None)?;
                total_distance += distance;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }
    
    /// Calculate DTW distance between two sequences
    fn dtw_distance(&self, seq1: &[f64], seq2: &[f64], window: Option<usize>) -> Result<f64, SwarmError> {
        let n = seq1.len();
        let m = seq2.len();
        
        if n == 0 || m == 0 {
            return Ok(f64::INFINITY);
        }
        
        // Create DTW matrix
        let mut dtw_matrix = vec![vec![f64::INFINITY; m + 1]; n + 1];
        dtw_matrix[0][0] = 0.0;
        
        // Apply Sakoe-Chiba band constraint if window provided
        let w = window.unwrap_or(n.max(m));
        
        for i in 1..=n {
            let start_j = if w < i { i - w } else { 1 };
            let end_j = (i + w).min(m);
            
            for j in start_j..=end_j {
                let cost = (seq1[i - 1] - seq2[j - 1]).abs();
                let min_prev = dtw_matrix[i - 1][j]
                    .min(dtw_matrix[i][j - 1])
                    .min(dtw_matrix[i - 1][j - 1]);
                    
                dtw_matrix[i][j] = cost + min_prev;
            }
        }
        
        // Normalize by path length
        let distance = dtw_matrix[n][m] / (n + m) as f64;
        Ok(distance)
    }
    
    /// Convert position to time series sequence
    fn position_to_sequence(&self, position: &Position) -> Vec<f64> {
        // For this implementation, we'll use the position coordinates as sequence
        // In practice, this could be a trajectory or behavioral sequence
        position.as_slice().to_vec()
    }
    
    /// Calculate Kendall tau diversity
    fn calculate_kendall_diversity(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        // For each dimension, calculate Kendall tau between position rankings
        let dimensions = positions[0].len();
        
        for dim in 0..dimensions {
            let values: Vec<f64> = positions.iter().map(|p| p[dim]).collect();
            let ranks = self.calculate_ranks(&values);
            
            // Calculate Kendall tau distance between consecutive rankings
            for i in 0..(ranks.len() - 1) {
                let tau_distance = self.kendall_tau_distance(&ranks[i..], &ranks[(i+1)..])?;
                total_distance += tau_distance;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_distance / count as f64 } else { 0.0 })
    }
    
    /// Calculate Kendall tau distance between two ranking sequences
    fn kendall_tau_distance(&self, ranks1: &[f64], ranks2: &[f64]) -> Result<f64, SwarmError> {
        if ranks1.len() != ranks2.len() || ranks1.is_empty() {
            return Ok(0.0);
        }
        
        let n = ranks1.len();
        let mut concordant = 0;
        let mut discordant = 0;
        
        for i in 0..n {
            for j in (i + 1)..n {
                let diff1 = ranks1[i] - ranks1[j];
                let diff2 = ranks2[i] - ranks2[j];
                
                if diff1 * diff2 > 0.0 {
                    concordant += 1;
                } else if diff1 * diff2 < 0.0 {
                    discordant += 1;
                }
            }
        }
        
        let total_pairs = concordant + discordant;
        if total_pairs == 0 {
            return Ok(0.5);
        }
        
        let tau = (concordant as f64 - discordant as f64) / total_pairs as f64;
        Ok((1.0 - tau) / 2.0) // Convert to distance in [0,1]
    }
    
    /// Calculate ranks for a sequence of values
    fn calculate_ranks(&self, values: &[f64]) -> Vec<f64> {
        let mut indexed_values: Vec<(usize, f64)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by value
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut ranks = vec![0.0; values.len()];
        
        // Assign ranks (handling ties)
        let mut i = 0;
        while i < indexed_values.len() {
            let mut j = i;
            let current_value = indexed_values[i].1;
            
            // Find all ties
            while j < indexed_values.len() && (indexed_values[j].1 - current_value).abs() < 1e-10 {
                j += 1;
            }
            
            // Calculate average rank for tied values
            let avg_rank = (i + j + 1) as f64 / 2.0;
            
            // Assign average rank to all tied values
            for k in i..j {
                let original_index = indexed_values[k].0;
                ranks[original_index] = avg_rank;
            }
            
            i = j;
        }
        
        ranks
    }
    
    /// Calculate Spearman rank correlation diversity
    fn calculate_spearman_diversity(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 2 {
            return Ok(0.0);
        }
        
        let dimensions = positions[0].len();
        let mut total_correlation = 0.0;
        
        // Calculate Spearman correlation between dimensions
        for dim1 in 0..dimensions {
            for dim2 in (dim1 + 1)..dimensions {
                let values1: Vec<f64> = positions.iter().map(|p| p[dim1]).collect();
                let values2: Vec<f64> = positions.iter().map(|p| p[dim2]).collect();
                
                let correlation = self.spearman_correlation(&values1, &values2)?;
                total_correlation += correlation.abs(); // Use absolute correlation for diversity
            }
        }
        
        let num_pairs = (dimensions * (dimensions - 1)) / 2;
        let avg_correlation = if num_pairs > 0 {
            total_correlation / num_pairs as f64
        } else {
            0.0
        };
        
        // Convert correlation to diversity (high correlation = low diversity)
        Ok(1.0 - avg_correlation)
    }
    
    /// Calculate Spearman rank correlation coefficient
    fn spearman_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, SwarmError> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }
        
        let ranks_x = self.calculate_ranks(x);
        let ranks_y = self.calculate_ranks(y);
        
        // Calculate Pearson correlation on ranks
        self.pearson_correlation(&ranks_x, &ranks_y)
    }
    
    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, SwarmError> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xx: f64 = x.iter().map(|&val| val * val).sum();
        let sum_yy: f64 = y.iter().map(|&val| val * val).sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator_x = n * sum_xx - sum_x * sum_x;
        let denominator_y = n * sum_yy - sum_y * sum_y;
        
        if denominator_x <= 0.0 || denominator_y <= 0.0 {
            return Ok(0.0);
        }
        
        let correlation = numerator / (denominator_x * denominator_y).sqrt();
        Ok(correlation.clamp(-1.0, 1.0))
    }
    
    /// Calculate template similarity
    fn calculate_template_similarity(
        &self,
        positions: &[&Position],
        templates: &HashMap<String, Vec<f64>>,
    ) -> Result<f64, SwarmError> {
        if positions.is_empty() || templates.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        for position in positions {
            let sequence = self.position_to_sequence(position);
            
            for (_name, template) in templates {
                let distance = self.dtw_distance(&sequence, template, None)?;
                let similarity = 1.0 / (1.0 + distance); // Convert distance to similarity
                total_similarity += similarity;
                count += 1;
            }
        }
        
        Ok(if count > 0 { total_similarity / count as f64 } else { 0.0 })
    }
    
    /// Enhanced information diversity calculation with all measures
    pub fn calculate_enhanced_information_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<DiversityMeasure, SwarmError> {
        let positions: Vec<&Position> = population.iter().map(|ind| ind.position()).collect();
        
        if positions.is_empty() {
            return Err(SwarmError::parameter("Empty population"));
        }
        
        // Calculate Shannon entropy
        let shannon_entropy = self.calculate_shannon_entropy(&positions)?;
        
        // Calculate Rényi entropy
        let renyi_entropy = self.calculate_renyi_entropy(&positions, 2.0)?;
        
        // Calculate mutual information
        let mutual_information = self.calculate_mutual_information(&positions)?;
        
        // Calculate KL divergence from uniform distribution
        let kl_divergence = self.calculate_kl_divergence(&positions)?;
        
        // Calculate Jensen-Shannon divergence between population halves
        let jensen_shannon_divergence = self.calculate_population_js_divergence(&positions)?;
        
        // Calculate cross entropy
        let cross_entropy = self.calculate_cross_entropy(&positions)?;
        
        Ok(DiversityMeasure::Information {
            shannon_entropy,
            renyi_entropy,
            mutual_information,
            kl_divergence,
            jensen_shannon_divergence,
            cross_entropy,
        })
    }
    
    /// Calculate Jensen-Shannon divergence for population
    fn calculate_population_js_divergence(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.len() < 4 {
            return Ok(0.0); // Need at least 4 for two groups
        }
        
        let mid = positions.len() / 2;
        let group1 = &positions[..mid];
        let group2 = &positions[mid..];
        
        let bins = 20;
        let dist1 = self.positions_to_distribution(group1, bins)?;
        let dist2 = self.positions_to_distribution(group2, bins)?;
        
        self.calculate_jensen_shannon_divergence(&dist1, &dist2)
    }
    
    /// Calculate cross entropy
    fn calculate_cross_entropy(&self, positions: &[&Position]) -> Result<f64, SwarmError> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        let bins = 20;
        let distribution = self.positions_to_distribution(positions, bins)?;
        let uniform = vec![1.0 / bins as f64; bins];
        
        // Calculate cross entropy: H(p, q) = -sum(p_i * log(q_i))
        let mut cross_entropy = 0.0;
        let epsilon = 1e-10;
        
        for (pi, qi) in distribution.iter().zip(uniform.iter()) {
            let p_safe = (*pi).max(epsilon);
            let q_safe = (*qi).max(epsilon);
            cross_entropy -= p_safe * q_safe.ln();
        }
        
        Ok(cross_entropy)
    }
    
    /// Clear calculation cache
    pub fn clear_cache(&self) {
        let mut cache = self.calculation_cache.write();
        cache.clear();
    }
}

impl Default for DiversityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Algorithm context for behavioral diversity calculation
#[derive(Debug, Clone)]
pub struct AlgorithmContext {
    pub algorithm_type: String,
    pub parameters: HashMap<String, f64>,
    pub iteration: usize,
    pub search_history: Vec<Position>,
}

/// Diversity calculator trait for different algorithms
pub trait DiversityCalculator {
    fn calculate_population_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<f64, SwarmError>;
    
    fn calculate_dimensional_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        dimension: usize,
    ) -> Result<f64, SwarmError>;
}

impl DiversityCalculator for DiversityMetrics {
    fn calculate_population_diversity<T: Individual>(
        &self,
        population: &Population<T>,
    ) -> Result<f64, SwarmError> {
        let config = DiversityConfig::default();
        match self.calculate_diversity(population, &config)? {
            DiversityMeasure::Combined { composite_score, .. } => Ok(composite_score),
            DiversityMeasure::Statistical { variance, .. } => Ok(variance),
            _ => Ok(0.0),
        }
    }
    
    fn calculate_dimensional_diversity<T: Individual>(
        &self,
        population: &Population<T>,
        dimension: usize,
    ) -> Result<f64, SwarmError> {
        if population.is_empty() {
            return Ok(0.0);
        }
        
        let values: Vec<f64> = population.iter()
            .filter_map(|ind| ind.position().get(dimension).copied())
            .collect();
        
        self.calculate_dimension_entropy(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::BasicIndividual;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_diversity_metrics_creation() {
        let metrics = DiversityMetrics::new();
        assert!(!metrics.simd_enabled || cfg!(feature = "simd"));
        assert_eq!(metrics.parallel_threshold, 1000);
    }
    
    #[test]
    fn test_statistical_diversity() {
        let mut population = Population::new();
        
        // Add some individuals with known positions
        population.add(BasicIndividual::new(Position::from_vec(vec![0.0, 0.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![1.0, 0.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![0.0, 1.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![1.0, 1.0])));
        
        let metrics = DiversityMetrics::new();
        let result = metrics.calculate_statistical_diversity(&population);
        
        assert!(result.is_ok());
        if let Ok(DiversityMeasure::Statistical { variance, std_deviation, cv, entropy }) = result {
            assert!(variance > 0.0);
            assert!(std_deviation > 0.0);
            assert!(cv >= 0.0);
            assert!(entropy >= 0.0);
        }
    }
    
    #[test]
    fn test_geometric_diversity() {
        let mut population = Population::new();
        
        // Create a square of points
        population.add(BasicIndividual::new(Position::from_vec(vec![0.0, 0.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![2.0, 0.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![0.0, 2.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![2.0, 2.0])));
        
        let metrics = DiversityMetrics::new();
        let result = metrics.calculate_geometric_diversity(&population);
        
        assert!(result.is_ok());
        if let Ok(DiversityMeasure::Geometric { avg_pairwise_distance, mst_length, convex_hull_volume, hypervolume }) = result {
            assert!(avg_pairwise_distance > 0.0);
            assert!(mst_length > 0.0);
            assert!(convex_hull_volume > 0.0);
            assert!(hypervolume > 0.0);
        }
    }
    
    #[test]
    fn test_combined_diversity() {
        let mut population = Population::new();
        
        // Add diverse individuals
        for i in 0..10 {
            let x = (i as f64) * 0.5;
            let y = (i as f64 * 0.3).sin();
            population.add(BasicIndividual::new(Position::from_vec(vec![x, y])));
        }
        
        let metrics = DiversityMetrics::new();
        let config = DiversityConfig::default();
        let result = metrics.calculate_diversity(&population, &config);
        
        assert!(result.is_ok());
        if let Ok(DiversityMeasure::Combined { composite_score, components, confidence_interval }) = result {
            assert!(composite_score >= 0.0);
            assert!(!components.is_empty());
            assert!(confidence_interval.0 <= confidence_interval.1);
        }
    }
    
    #[test]
    fn test_diversity_calculator_trait() {
        let mut population = Population::new();
        population.add(BasicIndividual::new(Position::from_vec(vec![1.0, 2.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![3.0, 4.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![5.0, 6.0])));
        
        let calculator = DiversityMetrics::new();
        
        let population_diversity = calculator.calculate_population_diversity(&population);
        assert!(population_diversity.is_ok());
        assert!(population_diversity.unwrap() >= 0.0);
        
        let dimensional_diversity = calculator.calculate_dimensional_diversity(&population, 0);
        assert!(dimensional_diversity.is_ok());
        assert!(dimensional_diversity.unwrap() >= 0.0);
    }
    
    #[test]
    fn test_cache_functionality() {
        let mut population = Population::new();
        population.add(BasicIndividual::new(Position::from_vec(vec![1.0, 1.0])));
        population.add(BasicIndividual::new(Position::from_vec(vec![2.0, 2.0])));
        
        let metrics = DiversityMetrics::new();
        let config = DiversityConfig::default();
        
        // First calculation should populate cache
        let result1 = metrics.calculate_diversity(&population, &config);
        assert!(result1.is_ok());
        
        // Second calculation should use cache
        let result2 = metrics.calculate_diversity(&population, &config);
        assert!(result2.is_ok());
        
        // Clear cache
        metrics.clear_cache();
        
        // Third calculation should recalculate
        let result3 = metrics.calculate_diversity(&population, &config);
        assert!(result3.is_ok());
    }
}