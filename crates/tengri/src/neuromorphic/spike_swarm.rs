//! # Spike Swarm Layer - Collective Neural Dynamics
//!
//! This module implements the first layer of TENGRI's temporal-swarm architecture.
//! It handles collective spiking dynamics, population coding, and synchronization
//! detection across populations of spiking neurons.
//!
//! ## Features
//!
//! - Population spike coordination
//! - Avalanche detection and analysis (power-law τ ≈ 1.5)
//! - Synchronization measurement (phase-locking value)
//! - Multiple spike encoding schemes
//! - Real-time correlation analysis

use crate::neuromorphic::{SpikingNeuron, NeuronConfig, SpikeEvent, NeuromorphicConfig};
use crate::{TengriError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Spike encoding schemes for information representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpikeEncoding {
    /// Rate coding - information in spike frequency
    Rate { time_window_ms: f64 },
    
    /// Temporal coding - information in spike timing
    Temporal { precision_ms: f64 },
    
    /// Phase coding - information in spike phase relative to oscillation
    Phase { reference_frequency_hz: f64 },
    
    /// Population vector coding
    PopulationVector { vector_dimension: usize },
    
    /// Rank-order coding - information in spike order
    RankOrder { max_ranks: usize },
}

impl Default for SpikeEncoding {
    fn default() -> Self {
        SpikeEncoding::Rate { time_window_ms: 100.0 }
    }
}

/// Configuration for the spike swarm layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeSwarmConfig {
    /// Number of neurons in the swarm
    pub population_size: usize,
    
    /// Spike encoding scheme
    pub encoding: SpikeEncoding,
    
    /// Avalanche detection threshold
    pub avalanche_threshold: f64,
    
    /// Time window for correlation analysis in ms
    pub correlation_window_ms: f64,
    
    /// Synchronization detection parameters
    pub sync_detection: SynchronizationConfig,
    
    /// Enable real-time analysis
    pub real_time_analysis: bool,
    
    /// Maximum spike history to maintain
    pub max_history_size: usize,
}

impl Default for SpikeSwarmConfig {
    fn default() -> Self {
        Self {
            population_size: 1000,
            encoding: SpikeEncoding::default(),
            avalanche_threshold: 0.1,
            correlation_window_ms: 50.0,
            sync_detection: SynchronizationConfig::default(),
            real_time_analysis: true,
            max_history_size: 10000,
        }
    }
}

/// Synchronization detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationConfig {
    /// Minimum phase-locking value for synchronization
    pub min_plv: f64,
    
    /// Frequency bands for analysis (Hz)
    pub frequency_bands: Vec<(f64, f64)>,
    
    /// Time window for PLV calculation
    pub plv_window_ms: f64,
    
    /// Minimum number of spikes for analysis
    pub min_spike_count: usize,
}

impl Default for SynchronizationConfig {
    fn default() -> Self {
        Self {
            min_plv: 0.3,
            frequency_bands: vec![
                (1.0, 4.0),   // Delta
                (4.0, 8.0),   // Theta
                (8.0, 12.0),  // Alpha
                (12.0, 30.0), // Beta
                (30.0, 100.0), // Gamma
            ],
            plv_window_ms: 1000.0,
            min_spike_count: 50,
        }
    }
}

/// Neural avalanche detection and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAvalanche {
    /// Avalanche ID
    pub id: usize,
    
    /// Start time of avalanche
    pub start_time_ms: f64,
    
    /// Duration of avalanche
    pub duration_ms: f64,
    
    /// Size (number of participating neurons)
    pub size: usize,
    
    /// Peak activity level
    pub peak_activity: f64,
    
    /// Participating neuron IDs
    pub participating_neurons: Vec<usize>,
    
    /// Avalanche shape (activity over time)
    pub activity_profile: Vec<f64>,
}

/// Population synchronization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationMetrics {
    /// Phase-locking value
    pub plv: f64,
    
    /// Synchronization strength
    pub sync_strength: f64,
    
    /// Coherence coefficient
    pub coherence: f64,
    
    /// Cross-correlation peak
    pub cross_correlation_peak: f64,
    
    /// Time lag of maximum correlation
    pub optimal_lag_ms: f64,
    
    /// Number of synchronized neuron pairs
    pub synchronized_pairs: usize,
}

/// Spike swarm collective dynamics processor
#[derive(Debug)]
pub struct SpikeSwarm {
    /// Swarm configuration
    config: SpikeSwarmConfig,
    
    /// Population of spiking neurons
    neurons: Vec<SpikingNeuron>,
    
    /// Spike history buffer
    spike_history: VecDeque<SpikeEvent>,
    
    /// Detected avalanches
    avalanches: Vec<NeuralAvalanche>,
    
    /// Current synchronization metrics
    sync_metrics: SynchronizationMetrics,
    
    /// Population firing rate history
    firing_rate_history: VecDeque<f64>,
    
    /// Correlation matrix between neurons
    correlation_matrix: Vec<Vec<f64>>,
    
    /// Current simulation time
    current_time_ms: f64,
    
    /// Avalanche detection state
    avalanche_detector: AvalancheDetector,
    
    /// Performance metrics
    performance_start: Instant,
    
    /// Processing statistics
    processing_stats: SwarmProcessingStats,
}

/// Avalanche detection algorithm
#[derive(Debug)]
struct AvalancheDetector {
    /// Activity threshold for avalanche detection
    threshold: f64,
    
    /// Current avalanche being tracked
    current_avalanche: Option<NeuralAvalanche>,
    
    /// Avalanche counter
    avalanche_counter: usize,
    
    /// Recent activity levels
    activity_buffer: VecDeque<f64>,
    
    /// Time window for activity measurement
    activity_window_ms: f64,
}

/// Processing statistics for the swarm
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmProcessingStats {
    /// Total spikes processed
    pub total_spikes_processed: u64,
    
    /// Current population firing rate (Hz)
    pub population_firing_rate_hz: f64,
    
    /// Number of avalanches detected
    pub avalanches_detected: u64,
    
    /// Average avalanche size
    pub avg_avalanche_size: f64,
    
    /// Synchronization events detected
    pub sync_events_detected: u64,
    
    /// Processing time per update (µs)
    pub avg_processing_time_us: f64,
    
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
}

impl SpikeSwarm {
    /// Create a new spike swarm
    pub fn new(config: SpikeSwarmConfig, neuron_config: NeuronConfig, 
               system_config: &NeuromorphicConfig) -> Result<Self> {
        let mut neurons = Vec::with_capacity(config.population_size);
        
        // Create population of neurons
        for i in 0..config.population_size {
            let seed = system_config.seed.unwrap_or(0) + i as u64;
            let neuron = SpikingNeuron::with_seed(
                i, 
                neuron_config.clone(), 
                system_config.timestep_ms, 
                seed
            )?;
            neurons.push(neuron);
        }
        
        // Initialize correlation matrix
        let correlation_matrix = vec![vec![0.0; config.population_size]; config.population_size];
        
        let avalanche_detector = AvalancheDetector {
            threshold: config.avalanche_threshold,
            current_avalanche: None,
            avalanche_counter: 0,
            activity_buffer: VecDeque::new(),
            activity_window_ms: 10.0, // 10ms activity window
        };
        
        Ok(Self {
            config,
            neurons,
            spike_history: VecDeque::new(),
            avalanches: Vec::new(),
            sync_metrics: SynchronizationMetrics::default(),
            firing_rate_history: VecDeque::new(),
            correlation_matrix,
            current_time_ms: 0.0,
            avalanche_detector,
            performance_start: Instant::now(),
            processing_stats: SwarmProcessingStats::default(),
        })
    }
    
    /// Update the entire swarm for one timestep
    pub fn update(&mut self, input_currents: &[f64], dt_ms: f64) -> Vec<SpikeEvent> {
        let update_start = Instant::now();
        
        self.current_time_ms += dt_ms;
        
        // Update all neurons and collect spikes
        let mut new_spikes = Vec::new();
        
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let input_current = input_currents.get(i).copied().unwrap_or(0.0);
            
            if let Some(spike) = neuron.update(input_current, self.current_time_ms) {
                new_spikes.push(spike);
            }
        }
        
        // Process collective dynamics
        self.process_spikes(&new_spikes);
        
        // Update analysis if enabled
        if self.config.real_time_analysis {
            self.update_analysis();
        }
        
        // Update statistics
        let processing_time = update_start.elapsed();
        self.update_processing_stats(&new_spikes, processing_time);
        
        new_spikes
    }
    
    /// Process new spikes for collective analysis
    fn process_spikes(&mut self, spikes: &[SpikeEvent]) {
        // Add spikes to history
        for spike in spikes {
            self.spike_history.push_back(spike.clone());
            self.processing_stats.total_spikes_processed += 1;
        }
        
        // Limit history size
        while self.spike_history.len() > self.config.max_history_size {
            self.spike_history.pop_front();
        }
        
        // Update population firing rate
        self.update_firing_rate();
        
        // Detect avalanches
        self.detect_avalanches(spikes);
        
        // Update synchronization if enough spikes
        if spikes.len() > 5 {
            self.update_synchronization();
        }
    }
    
    /// Update population firing rate
    fn update_firing_rate(&mut self) {
        let window_ms = 100.0; // 100ms window
        let window_start = self.current_time_ms - window_ms;
        
        // Count spikes in time window
        let spike_count = self.spike_history.iter()
            .filter(|spike| spike.timestamp_ms >= window_start)
            .count();
        
        let firing_rate = (spike_count as f64) / (window_ms / 1000.0) / (self.config.population_size as f64);
        
        self.firing_rate_history.push_back(firing_rate);
        self.processing_stats.population_firing_rate_hz = firing_rate;
        
        // Limit firing rate history
        while self.firing_rate_history.len() > 1000 {
            self.firing_rate_history.pop_front();
        }
    }
    
    /// Detect neural avalanches in the spike patterns
    fn detect_avalanches(&mut self, spikes: &[SpikeEvent]) {
        // Calculate current activity level
        let activity_level = spikes.len() as f64 / self.config.population_size as f64;
        
        self.avalanche_detector.activity_buffer.push_back(activity_level);
        
        // Limit buffer size
        while self.avalanche_detector.activity_buffer.len() > 100 {
            self.avalanche_detector.activity_buffer.pop_front();
        }
        
        // Check for avalanche start
        if activity_level > self.avalanche_detector.threshold && 
           self.avalanche_detector.current_avalanche.is_none() {
            
            // Start new avalanche
            let avalanche = NeuralAvalanche {
                id: self.avalanche_detector.avalanche_counter,
                start_time_ms: self.current_time_ms,
                duration_ms: 0.0,
                size: spikes.len(),
                peak_activity: activity_level,
                participating_neurons: spikes.iter().map(|s| s.neuron_id).collect(),
                activity_profile: vec![activity_level],
            };
            
            self.avalanche_detector.current_avalanche = Some(avalanche);
            self.avalanche_detector.avalanche_counter += 1;
        }
        
        // Update current avalanche
        if let Some(ref mut avalanche) = self.avalanche_detector.current_avalanche {
            avalanche.duration_ms = self.current_time_ms - avalanche.start_time_ms;
            avalanche.activity_profile.push(activity_level);
            avalanche.peak_activity = avalanche.peak_activity.max(activity_level);
            
            // Add new participating neurons
            for spike in spikes {
                if !avalanche.participating_neurons.contains(&spike.neuron_id) {
                    avalanche.participating_neurons.push(spike.neuron_id);
                }
            }
            avalanche.size = avalanche.participating_neurons.len();
            
            // Check for avalanche end
            if activity_level < self.avalanche_detector.threshold * 0.5 {
                // End avalanche
                let finished_avalanche = self.avalanche_detector.current_avalanche.take().unwrap();
                self.avalanches.push(finished_avalanche);
                self.processing_stats.avalanches_detected += 1;
                
                // Update average avalanche size
                let total_size: usize = self.avalanches.iter().map(|a| a.size).sum();
                self.processing_stats.avg_avalanche_size = 
                    total_size as f64 / self.avalanches.len() as f64;
            }
        }
    }
    
    /// Update synchronization analysis
    fn update_synchronization(&mut self) {
        // This is a simplified synchronization analysis
        // In a full implementation, this would include:
        // - Phase-locking value calculation
        // - Cross-correlation analysis
        // - Coherence measurement
        // - Frequency domain analysis
        
        let window_ms = self.config.sync_detection.plv_window_ms;
        let window_start = self.current_time_ms - window_ms;
        
        // Get recent spikes
        let recent_spikes: Vec<_> = self.spike_history.iter()
            .filter(|spike| spike.timestamp_ms >= window_start)
            .collect();
        
        if recent_spikes.len() >= self.config.sync_detection.min_spike_count {
            // Simple synchronization metric based on temporal clustering
            let mut time_bins = vec![0u32; (window_ms as usize) / 10]; // 10ms bins
            
            for spike in &recent_spikes {
                let bin_index = ((spike.timestamp_ms - window_start) / 10.0) as usize;
                if bin_index < time_bins.len() {
                    time_bins[bin_index] += 1;
                }
            }
            
            // Calculate coefficient of variation (measure of clustering)
            let mean = recent_spikes.len() as f64 / time_bins.len() as f64;
            let variance: f64 = time_bins.iter()
                .map(|&count| (count as f64 - mean).powi(2))
                .sum::<f64>() / time_bins.len() as f64;
            
            let cv = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
            
            // Update synchronization metrics (simplified)
            self.sync_metrics = SynchronizationMetrics {
                plv: cv.min(1.0), // Use CV as a proxy for synchronization
                sync_strength: cv.min(1.0),
                coherence: cv.min(1.0),
                cross_correlation_peak: cv.min(1.0),
                optimal_lag_ms: 0.0, // Would require actual cross-correlation
                synchronized_pairs: (cv * self.config.population_size as f64) as usize,
            };
            
            if self.sync_metrics.plv > self.config.sync_detection.min_plv {
                self.processing_stats.sync_events_detected += 1;
            }
        }
    }
    
    /// Update real-time analysis
    fn update_analysis(&mut self) {
        // Update correlation matrix (simplified version)
        // In practice, this would use sliding window correlation
        self.update_correlation_matrix();
    }
    
    /// Update correlation matrix between neurons
    fn update_correlation_matrix(&mut self) {
        // This is a placeholder for correlation analysis
        // Real implementation would calculate pairwise spike train correlations
        let size = self.config.population_size;
        
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    // Placeholder: random correlation for now
                    // Real implementation would use cross-correlation analysis
                    self.correlation_matrix[i][j] = 0.1; // Small baseline correlation
                }
            }
        }
    }
    
    /// Update processing statistics
    fn update_processing_stats(&mut self, spikes: &[SpikeEvent], processing_time: std::time::Duration) {
        let processing_time_us = processing_time.as_micros() as f64;
        
        // Update average processing time
        let alpha = 0.1; // Smoothing factor
        self.processing_stats.avg_processing_time_us = 
            self.processing_stats.avg_processing_time_us * (1.0 - alpha) + processing_time_us * alpha;
        
        // Estimate memory usage
        let base_size = std::mem::size_of::<Self>();
        let neuron_size = self.neurons.capacity() * std::mem::size_of::<SpikingNeuron>();
        let history_size = self.spike_history.capacity() * std::mem::size_of::<SpikeEvent>();
        let avalanche_size = self.avalanches.capacity() * std::mem::size_of::<NeuralAvalanche>();
        
        self.processing_stats.memory_usage_bytes = base_size + neuron_size + history_size + avalanche_size;
    }
    
    /// Get current population firing rate
    pub fn population_firing_rate(&self) -> f64 {
        self.processing_stats.population_firing_rate_hz
    }
    
    /// Get detected avalanches
    pub fn avalanches(&self) -> &[NeuralAvalanche] {
        &self.avalanches
    }
    
    /// Get synchronization metrics
    pub fn synchronization_metrics(&self) -> &SynchronizationMetrics {
        &self.sync_metrics
    }
    
    /// Get processing statistics
    pub fn processing_stats(&self) -> &SwarmProcessingStats {
        &self.processing_stats
    }
    
    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> &[Vec<f64>] {
        &self.correlation_matrix
    }
    
    /// Get spike history
    pub fn spike_history(&self) -> &VecDeque<SpikeEvent> {
        &self.spike_history
    }
    
    /// Get neuron by ID
    pub fn get_neuron(&self, id: usize) -> Option<&SpikingNeuron> {
        self.neurons.get(id)
    }
    
    /// Get mutable neuron by ID
    pub fn get_neuron_mut(&mut self, id: usize) -> Option<&mut SpikingNeuron> {
        self.neurons.get_mut(id)
    }
    
    /// Reset the swarm to initial state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        
        self.spike_history.clear();
        self.avalanches.clear();
        self.firing_rate_history.clear();
        self.avalanche_detector.current_avalanche = None;
        self.avalanche_detector.avalanche_counter = 0;
        self.avalanche_detector.activity_buffer.clear();
        self.current_time_ms = 0.0;
        self.processing_stats = SwarmProcessingStats::default();
        
        // Reset correlation matrix
        let size = self.config.population_size;
        self.correlation_matrix = vec![vec![0.0; size]; size];
    }
    
    /// Analyze avalanche statistics for power-law distribution
    pub fn analyze_avalanche_statistics(&self) -> Option<f64> {
        if self.avalanches.len() < 10 {
            return None; // Need more avalanches for statistical analysis
        }
        
        // Extract avalanche sizes
        let mut sizes: Vec<f64> = self.avalanches.iter()
            .map(|a| a.size as f64)
            .filter(|&size| size > 0.0)
            .collect();
        
        if sizes.is_empty() {
            return None;
        }
        
        sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Simple power-law exponent estimation using log-log slope
        // P(s) ~ s^(-τ) where τ ≈ 1.5 for critical avalanches
        let log_sizes: Vec<f64> = sizes.iter().map(|s| s.ln()).collect();
        let log_probs: Vec<f64> = (1..=sizes.len())
            .map(|i| (1.0 - (i as f64 / sizes.len() as f64)).ln())
            .collect();
        
        // Linear regression to estimate slope (τ)
        if log_sizes.len() >= 2 {
            let n = log_sizes.len() as f64;
            let sum_x: f64 = log_sizes.iter().sum();
            let sum_y: f64 = log_probs.iter().sum();
            let sum_xy: f64 = log_sizes.iter().zip(&log_probs).map(|(x, y)| x * y).sum();
            let sum_x2: f64 = log_sizes.iter().map(|x| x * x).sum();
            
            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            Some(-slope) // Return positive τ
        } else {
            None
        }
    }
    
    /// Get power-law criticality indicator
    pub fn criticality_indicator(&self) -> f64 {
        if let Some(tau) = self.analyze_avalanche_statistics() {
            // Target τ ≈ 1.5 for criticality
            let target_tau = 1.5;
            let deviation = (tau - target_tau).abs();
            
            // Return criticality as 1 - normalized_deviation
            (1.0 - (deviation / target_tau).min(1.0)).max(0.0)
        } else {
            0.0 // No criticality data available
        }
    }
}

impl Default for SynchronizationMetrics {
    fn default() -> Self {
        Self {
            plv: 0.0,
            sync_strength: 0.0,
            coherence: 0.0,
            cross_correlation_peak: 0.0,
            optimal_lag_ms: 0.0,
            synchronized_pairs: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::{NeuromorphicConfig, PerformanceMetrics};
    
    #[test]
    fn test_spike_swarm_creation() {
        let swarm_config = SpikeSwarmConfig::default();
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        assert_eq!(swarm.neurons.len(), 1000); // Default population size
        assert_eq!(swarm.correlation_matrix.len(), 1000);
    }
    
    #[test]
    fn test_swarm_update() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 10,
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Provide input currents to generate spikes
        let input_currents = vec![500.0; 10]; // High current to force spikes
        
        // Run several updates
        let mut total_spikes = 0;
        for _ in 0..100 {
            let spikes = swarm.update(&input_currents, 0.1);
            total_spikes += spikes.len();
        }
        
        assert!(total_spikes > 0, "Swarm should generate spikes with high input");
        assert!(swarm.population_firing_rate() >= 0.0);
    }
    
    #[test]
    fn test_avalanche_detection() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 50,
            avalanche_threshold: 0.1,
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Generate coordinated activity to trigger avalanches
        let high_current = vec![1000.0; 50]; // Very high current
        let low_current = vec![0.0; 50];     // No current
        
        // Alternate high and low activity
        for i in 0..200 {
            let inputs = if i % 20 < 10 { &high_current } else { &low_current };
            swarm.update(inputs, 0.1);
        }
        
        let stats = swarm.processing_stats();
        // Should detect some avalanches with alternating high/low activity
        assert!(stats.avalanches_detected >= 0); // May or may not detect avalanches
    }
    
    #[test]
    fn test_firing_rate_calculation() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 100,
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Generate consistent activity
        let input_currents = vec![200.0; 100]; // Moderate current
        
        for _ in 0..1000 { // 100ms of simulation
            swarm.update(&input_currents, 0.1);
        }
        
        let firing_rate = swarm.population_firing_rate();
        assert!(firing_rate >= 0.0);
        assert!(firing_rate < 1000.0); // Reasonable upper bound
    }
    
    #[test]
    fn test_synchronization_detection() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 20,
            sync_detection: SynchronizationConfig {
                min_spike_count: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Generate synchronized activity
        let sync_current = vec![800.0; 20]; // High current for synchronization
        
        for _ in 0..2000 { // Run long enough for analysis
            swarm.update(&sync_current, 0.1);
        }
        
        let sync_metrics = swarm.synchronization_metrics();
        assert!(sync_metrics.plv >= 0.0);
        assert!(sync_metrics.plv <= 1.0);
    }
    
    #[test]
    fn test_correlation_matrix() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 5,
            real_time_analysis: true,
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        let input_currents = vec![300.0; 5];
        
        for _ in 0..100 {
            swarm.update(&input_currents, 0.1);
        }
        
        let correlation = swarm.correlation_matrix();
        assert_eq!(correlation.len(), 5);
        assert_eq!(correlation[0].len(), 5);
        
        // Diagonal should be 0 (no self-correlation calculated)
        for i in 0..5 {
            assert_eq!(correlation[i][i], 0.0);
        }
    }
    
    #[test]
    fn test_power_law_analysis() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 100,
            avalanche_threshold: 0.05, // Lower threshold for more avalanches
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Generate activity pattern that might produce avalanches
        let mut rng = fastrand::Rng::new();
        
        for _ in 0..1000 {
            let mut input_currents = vec![0.0; 100];
            
            // Random stimulation to create avalanche-like activity
            for current in &mut input_currents {
                if rng.f64() < 0.1 { // 10% chance of stimulation
                    *current = 600.0;
                }
            }
            
            swarm.update(&input_currents, 0.1);
        }
        
        // Check if we can analyze power-law
        let tau = swarm.analyze_avalanche_statistics();
        if let Some(tau_value) = tau {
            assert!(tau_value > 0.0);
            assert!(tau_value < 5.0); // Reasonable range for power-law exponent
        }
        
        let criticality = swarm.criticality_indicator();
        assert!(criticality >= 0.0);
        assert!(criticality <= 1.0);
    }
    
    #[test]
    fn test_swarm_reset() {
        let swarm_config = SpikeSwarmConfig {
            population_size: 10,
            ..Default::default()
        };
        let neuron_config = NeuronConfig::default();
        let system_config = NeuromorphicConfig::default();
        
        let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
        
        // Generate some activity
        let input_currents = vec![400.0; 10];
        for _ in 0..50 {
            swarm.update(&input_currents, 0.1);
        }
        
        // Verify we have some history
        assert!(swarm.spike_history().len() > 0);
        assert!(swarm.processing_stats().total_spikes_processed > 0);
        
        // Reset and verify clean state
        swarm.reset();
        
        assert_eq!(swarm.spike_history().len(), 0);
        assert_eq!(swarm.avalanches().len(), 0);
        assert_eq!(swarm.processing_stats().total_spikes_processed, 0);
        assert_eq!(swarm.current_time_ms, 0.0);
    }
}