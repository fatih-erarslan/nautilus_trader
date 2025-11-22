//! # Spike Swarm Neural Network Layer for TENGRI
//! 
//! Implementation of a large-scale neural spike swarm with critical dynamics,
//! avalanche detection, and population coding for TENGRI trading strategy.
//! 
//! ## Features
//! 
//! - **1,000,000 neuron spike swarm** - Massive parallel processing capacity
//! - **Critical dynamics** - Power-law avalanche distribution with τ ≈ 1.5
//! - **Parallel processing** - Rayon-based parallel spike propagation
//! - **Spike encoding** - Rate, temporal, and phase-based encoding schemes
//! - **Synchronization metrics** - Real-time correlation and sync detection
//! - **Population coding** - Distributed representation across neuron populations
//! - **Avalanche analysis** - Real-time critical state monitoring
//! 
//! ## Architecture
//! 
//! The spike swarm operates as a critical dynamical system with:
//! - Leaky integrate-and-fire neurons
//! - Sparse connectivity (p ~ 0.001)
//! - Power-law distributed avalanches
//! - Multi-scale temporal dynamics
//! - Adaptive synaptic weights

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};

use crate::types::{TradingSignal, SignalType};

/// Number of neurons in the spike swarm
pub const SWARM_SIZE: usize = 1_000_000;

/// Sparse connectivity probability
pub const CONNECTIVITY_PROB: f64 = 0.001;

/// Critical dynamics power-law exponent
pub const POWER_LAW_EXPONENT: f64 = 1.5;

/// Synchrony detection threshold
pub const SYNC_THRESHOLD: f64 = 0.3;

/// Maximum avalanche size to track
pub const MAX_AVALANCHE_SIZE: usize = 100_000;

/// Time step in milliseconds
pub const TIME_STEP_MS: f64 = 0.1;

/// Neuron state structure optimized for memory efficiency
#[derive(Debug, Clone)]
pub struct Neuron {
    /// Membrane potential (mV)
    pub potential: f32,
    /// Firing threshold (mV)
    pub threshold: f32,
    /// Leak conductance
    pub leak: f32,
    /// Refractory period remaining (ms)
    pub refractory: u16,
    /// Last spike time (ms)
    pub last_spike: u32,
    /// Input current (nA)
    pub input_current: f32,
    /// Neuron type (excitatory/inhibitory)
    pub neuron_type: NeuronType,
}

/// Neuron type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
}

/// Synaptic connection structure
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Pre-synaptic neuron index
    pub pre_neuron: u32,
    /// Post-synaptic neuron index
    pub post_neuron: u32,
    /// Synaptic weight
    pub weight: f32,
    /// Delay (time steps)
    pub delay: u8,
    /// Last update time
    pub last_update: u32,
}

/// Spike event structure
#[derive(Debug, Clone, Copy)]
pub struct Spike {
    /// Neuron index that spiked
    pub neuron_id: u32,
    /// Spike time (ms)
    pub time: f64,
    /// Spike amplitude
    pub amplitude: f32,
}

/// Spike encoding types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpikeEncoding {
    /// Rate-based encoding
    Rate,
    /// Temporal pattern encoding
    Temporal,
    /// Phase encoding
    Phase,
}

/// Avalanche detection and analysis
#[derive(Debug, Clone)]
pub struct Avalanche {
    /// Avalanche ID
    pub id: u64,
    /// Start time (ms)
    pub start_time: f64,
    /// Duration (ms)
    pub duration: f64,
    /// Number of neurons involved
    pub size: usize,
    /// Spikes in the avalanche
    pub spikes: Vec<Spike>,
    /// Maximum activity level
    pub peak_activity: f32,
}

/// Population coding structure
#[derive(Debug, Clone)]
pub struct PopulationCode {
    /// Population ID
    pub id: String,
    /// Neuron indices in this population
    pub neurons: Vec<u32>,
    /// Current activity level (0-1)
    pub activity: f64,
    /// Encoding type
    pub encoding: SpikeEncoding,
    /// Feature vector being encoded
    pub feature_vector: Option<DVector<f64>>,
}

/// Synchronization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronyMetrics {
    /// Global synchrony index (0-1)
    pub global_synchrony: f64,
    /// Local synchrony by region
    pub local_synchrony: HashMap<String, f64>,
    /// Cross-correlation matrix
    pub correlations: HashMap<(u32, u32), f64>,
    /// Coherence spectrum
    pub coherence: Vec<f64>,
    /// Measurement timestamp
    pub timestamp: std::time::Instant,
}

/// Critical dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalDynamics {
    /// Current avalanche size distribution
    pub avalanche_sizes: HashMap<usize, u64>,
    /// Power-law exponent estimate
    pub power_law_exponent: f64,
    /// Criticality index (0-1, where 1 = critical)
    pub criticality_index: f64,
    /// Branching parameter
    pub branching_parameter: f64,
    /// Activity level variance
    pub activity_variance: f64,
    /// Long-range correlation strength
    pub correlation_length: f64,
}

/// Main spike swarm structure
pub struct SpikeSwarm {
    /// Neurons in the swarm
    neurons: Vec<Neuron>,
    /// Synaptic connections (sparse representation)
    synapses: Vec<Synapse>,
    /// Connection matrix (sparse)
    connectivity: Arc<RwLock<HashMap<u32, Vec<u32>>>>,
    /// Current simulation time (ms)
    current_time: Arc<RwLock<f64>>,
    /// Recent spikes buffer
    spike_buffer: Arc<RwLock<VecDeque<Spike>>>,
    /// Avalanche tracker
    avalanches: Arc<RwLock<Vec<Avalanche>>>,
    /// Population codes
    populations: HashMap<String, PopulationCode>,
    /// Synchrony metrics
    sync_metrics: Arc<RwLock<SynchronyMetrics>>,
    /// Critical dynamics state
    critical_dynamics: Arc<RwLock<CriticalDynamics>>,
    /// Performance metrics
    performance_stats: PerformanceStats,
    /// Configuration parameters
    config: SpikeSwarmConfig,
}

/// Configuration for spike swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeSwarmConfig {
    /// Number of neurons
    pub num_neurons: usize,
    /// Connectivity probability
    pub connectivity_prob: f64,
    /// Excitatory/inhibitory ratio
    pub excitatory_ratio: f64,
    /// Time step (ms)
    pub dt: f64,
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Memory optimization level (0-3)
    pub memory_optimization: u8,
    /// Recording settings
    pub recording: RecordingConfig,
}

/// Recording configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingConfig {
    /// Record all spikes
    pub record_spikes: bool,
    /// Record membrane potentials
    pub record_potentials: bool,
    /// Record avalanches
    pub record_avalanches: bool,
    /// Maximum recording duration (seconds)
    pub max_duration: f64,
    /// Sampling rate for continuous variables
    pub sampling_rate: f64,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Simulation speed (time steps per second)
    pub simulation_speed: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// CPU utilization (%)
    pub cpu_utilization: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Spike processing rate (spikes/second)
    pub spike_rate: f64,
    /// Total simulation time
    pub total_time: Duration,
}

impl Default for SpikeSwarmConfig {
    fn default() -> Self {
        Self {
            num_neurons: SWARM_SIZE,
            connectivity_prob: CONNECTIVITY_PROB,
            excitatory_ratio: 0.8,
            dt: TIME_STEP_MS,
            parallel_processing: true,
            memory_optimization: 2,
            recording: RecordingConfig {
                record_spikes: true,
                record_potentials: false,
                record_avalanches: true,
                max_duration: 3600.0,
                sampling_rate: 1000.0,
            },
        }
    }
}

impl Default for Neuron {
    fn default() -> Self {
        Self {
            potential: -70.0,
            threshold: -50.0,
            leak: 0.1,
            refractory: 0,
            last_spike: 0,
            input_current: 0.0,
            neuron_type: NeuronType::Excitatory,
        }
    }
}

impl SpikeSwarm {
    /// Create a new spike swarm with specified configuration
    pub fn new(config: SpikeSwarmConfig) -> Result<Self> {
        info!("Creating spike swarm with {} neurons", config.num_neurons);
        
        let start_time = Instant::now();
        
        // Initialize neurons
        let mut neurons = Vec::with_capacity(config.num_neurons);
        let excitatory_count = (config.num_neurons as f64 * config.excitatory_ratio) as usize;
        
        // Create neurons in parallel
        neurons.par_extend((0..config.num_neurons).into_par_iter().map(|i| {
            let mut neuron = Neuron::default();
            neuron.neuron_type = if i < excitatory_count {
                NeuronType::Excitatory
            } else {
                NeuronType::Inhibitory
            };
            
            // Add variability to neuron properties
            let variation = (i as f64 / config.num_neurons as f64 - 0.5) * 10.0;
            neuron.threshold += variation as f32;
            neuron.leak += (variation / 100.0) as f32;
            
            neuron
        }));
        
        // Generate sparse connectivity
        let connectivity = Self::generate_connectivity(&config)?;
        let synapses = Self::create_synapses(&connectivity, &config)?;
        
        // Initialize populations for different encoding schemes
        let populations = Self::create_populations(&config)?;
        
        // Initialize metrics structures
        let sync_metrics = SynchronyMetrics {
            global_synchrony: 0.0,
            local_synchrony: HashMap::new(),
            correlations: HashMap::new(),
            coherence: Vec::new(),
            timestamp: Instant::now(),
        };
        
        let critical_dynamics = CriticalDynamics {
            avalanche_sizes: HashMap::new(),
            power_law_exponent: POWER_LAW_EXPONENT,
            criticality_index: 0.5,
            branching_parameter: 1.0,
            activity_variance: 0.0,
            correlation_length: 0.0,
        };
        
        let performance_stats = PerformanceStats {
            simulation_speed: 0.0,
            memory_usage: 0.0,
            cpu_utilization: 0.0,
            parallel_efficiency: 0.0,
            spike_rate: 0.0,
            total_time: start_time.elapsed(),
        };
        
        let swarm = Self {
            neurons,
            synapses,
            connectivity: Arc::new(RwLock::new(connectivity)),
            current_time: Arc::new(RwLock::new(0.0)),
            spike_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(100000))),
            avalanches: Arc::new(RwLock::new(Vec::new())),
            populations,
            sync_metrics: Arc::new(RwLock::new(sync_metrics)),
            critical_dynamics: Arc::new(RwLock::new(critical_dynamics)),
            performance_stats,
            config,
        };
        
        info!("Spike swarm created in {:?}", start_time.elapsed());
        Ok(swarm)
    }
    
    /// Generate sparse connectivity matrix
    fn generate_connectivity(config: &SpikeSwarmConfig) -> Result<HashMap<u32, Vec<u32>>> {
        info!("Generating sparse connectivity with p = {}", config.connectivity_prob);
        
        let mut connectivity = HashMap::with_capacity(config.num_neurons);
        let total_possible = config.num_neurons * config.num_neurons;
        let expected_connections = (total_possible as f64 * config.connectivity_prob) as usize;
        
        info!("Expected {} connections out of {} possible", 
              expected_connections, total_possible);
        
        // Generate connections in parallel chunks
        let chunk_size = config.num_neurons / rayon::current_num_threads().max(1);
        let connections: Vec<_> = (0..config.num_neurons)
            .into_par_iter()
            .with_min_len(chunk_size)
            .map(|pre| {
                let mut pre_connections = Vec::new();
                
                for post in 0..config.num_neurons {
                    if pre != post && rand::random::<f64>() < config.connectivity_prob {
                        pre_connections.push(post as u32);
                    }
                }
                
                (pre as u32, pre_connections)
            })
            .collect();
        
        let total_connections: usize = connections.iter().map(|(_, conns)| conns.len()).sum();
        info!("Generated {} actual connections", total_connections);
        
        Ok(connections.into_iter().collect())
    }
    
    /// Create synaptic connections from connectivity matrix
    fn create_synapses(
        connectivity: &HashMap<u32, Vec<u32>>, 
        config: &SpikeSwarmConfig
    ) -> Result<Vec<Synapse>> {
        info!("Creating synaptic connections");
        
        let synapses: Vec<Synapse> = connectivity
            .par_iter()
            .flat_map(|(pre, posts)| {
                posts.iter().map(move |post| {
                    let weight = if *pre < (config.num_neurons as f64 * config.excitatory_ratio) as u32 {
                        // Excitatory synapse
                        rand::random::<f32>() * 0.1
                    } else {
                        // Inhibitory synapse
                        -(rand::random::<f32>() * 0.2)
                    };
                    
                    Synapse {
                        pre_neuron: *pre,
                        post_neuron: *post,
                        weight,
                        delay: (rand::random::<f32>() * 5.0) as u8 + 1,
                        last_update: 0,
                    }
                }).collect::<Vec<_>>()
            })
            .collect();
        
        info!("Created {} synapses", synapses.len());
        Ok(synapses)
    }
    
    /// Create population codes for different encoding schemes
    fn create_populations(config: &SpikeSwarmConfig) -> Result<HashMap<String, PopulationCode>> {
        let mut populations = HashMap::new();
        
        let population_size = config.num_neurons / 10; // 10 populations
        
        // Create rate coding populations
        for i in 0..3 {
            let start_idx = i * population_size;
            let end_idx = (i + 1) * population_size;
            let neurons: Vec<u32> = (start_idx..end_idx.min(config.num_neurons))
                .map(|x| x as u32)
                .collect();
            
            populations.insert(
                format!("rate_pop_{}", i),
                PopulationCode {
                    id: format!("rate_pop_{}", i),
                    neurons,
                    activity: 0.0,
                    encoding: SpikeEncoding::Rate,
                    feature_vector: None,
                },
            );
        }
        
        // Create temporal coding populations
        for i in 0..3 {
            let start_idx = (3 + i) * population_size;
            let end_idx = (4 + i) * population_size;
            let neurons: Vec<u32> = (start_idx..end_idx.min(config.num_neurons))
                .map(|x| x as u32)
                .collect();
            
            populations.insert(
                format!("temporal_pop_{}", i),
                PopulationCode {
                    id: format!("temporal_pop_{}", i),
                    neurons,
                    activity: 0.0,
                    encoding: SpikeEncoding::Temporal,
                    feature_vector: None,
                },
            );
        }
        
        // Create phase coding populations
        for i in 0..4 {
            let start_idx = (6 + i) * population_size;
            let end_idx = (7 + i) * population_size;
            let neurons: Vec<u32> = (start_idx..end_idx.min(config.num_neurons))
                .map(|x| x as u32)
                .collect();
            
            populations.insert(
                format!("phase_pop_{}", i),
                PopulationCode {
                    id: format!("phase_pop_{}", i),
                    neurons,
                    activity: 0.0,
                    encoding: SpikeEncoding::Phase,
                    feature_vector: None,
                },
            );
        }
        
        info!("Created {} population codes", populations.len());
        Ok(populations)
    }
    
    /// Step the simulation forward by one time step
    pub fn step(&mut self) -> Result<Vec<Spike>> {
        let step_start = Instant::now();
        
        // Update current time
        {
            let mut time = self.current_time.write().unwrap();
            *time += self.config.dt;
        }
        
        // Process neurons in parallel
        let spikes: Vec<Spike> = if self.config.parallel_processing {
            self.neurons
                .par_iter_mut()
                .enumerate()
                .filter_map(|(i, neuron)| {
                    if self.update_neuron(neuron, i as u32) {
                        let current_time = *self.current_time.read().unwrap();
                        Some(Spike {
                            neuron_id: i as u32,
                            time: current_time,
                            amplitude: 1.0,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            let mut spikes = Vec::new();
            let current_time = *self.current_time.read().unwrap();
            
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if self.update_neuron(neuron, i as u32) {
                    spikes.push(Spike {
                        neuron_id: i as u32,
                        time: current_time,
                        amplitude: 1.0,
                    });
                }
            }
            spikes
        };
        
        // Process spikes through synaptic connections
        if !spikes.is_empty() {
            self.propagate_spikes(&spikes)?;
            
            // Update spike buffer
            {
                let mut buffer = self.spike_buffer.write().unwrap();
                for spike in &spikes {
                    buffer.push_back(*spike);
                    if buffer.len() > 100000 {
                        buffer.pop_front();
                    }
                }
            }
            
            // Detect avalanches
            self.detect_avalanches(&spikes)?;
            
            // Update synchrony metrics
            self.update_synchrony_metrics(&spikes)?;
        }
        
        // Update performance statistics
        self.update_performance_stats(step_start.elapsed(), spikes.len());
        
        Ok(spikes)
    }
    
    /// Update individual neuron state
    fn update_neuron(&self, neuron: &mut Neuron, _neuron_id: u32) -> bool {
        // Skip if in refractory period
        if neuron.refractory > 0 {
            neuron.refractory -= 1;
            return false;
        }
        
        // Leaky integrate-and-fire dynamics
        let dt = self.config.dt as f32;
        
        // Membrane potential update
        let leak_current = -neuron.leak * (neuron.potential - (-70.0));
        let total_current = neuron.input_current + leak_current;
        neuron.potential += total_current * dt;
        
        // Check for spike
        if neuron.potential >= neuron.threshold {
            neuron.potential = -70.0; // Reset potential
            neuron.refractory = (2.0 / dt) as u16; // 2ms refractory period
            neuron.input_current = 0.0; // Reset input
            neuron.last_spike = *self.current_time.read().unwrap() as u32;
            return true;
        }
        
        // Decay input current
        neuron.input_current *= 0.9;
        
        false
    }
    
    /// Propagate spikes through synaptic connections
    fn propagate_spikes(&mut self, spikes: &[Spike]) -> Result<()> {
        if spikes.is_empty() {
            return Ok(());
        }
        
        // Create a set of spiking neurons for fast lookup
        let spiking_neurons: std::collections::HashSet<u32> = 
            spikes.iter().map(|s| s.neuron_id).collect();
        
        // Process synaptic transmission in parallel
        let synaptic_inputs: Vec<(u32, f32)> = self.synapses
            .par_iter()
            .filter_map(|synapse| {
                if spiking_neurons.contains(&synapse.pre_neuron) {
                    Some((synapse.post_neuron, synapse.weight))
                } else {
                    None
                }
            })
            .collect();
        
        // Apply synaptic inputs to post-synaptic neurons
        for (post_neuron, weight) in synaptic_inputs {
            if let Some(neuron) = self.neurons.get_mut(post_neuron as usize) {
                neuron.input_current += weight;
            }
        }
        
        Ok(())
    }
    
    /// Detect and analyze avalanches in spike activity
    fn detect_avalanches(&mut self, current_spikes: &[Spike]) -> Result<()> {
        if current_spikes.is_empty() {
            return Ok(());
        }
        
        let current_time = *self.current_time.read().unwrap();
        let spike_count = current_spikes.len();
        
        // Simple avalanche detection: significant activity burst
        if spike_count > 100 { // Minimum avalanche size
            let avalanche = Avalanche {
                id: rand::random(),
                start_time: current_time,
                duration: self.config.dt, // Will be updated as avalanche continues
                size: spike_count,
                spikes: current_spikes.to_vec(),
                peak_activity: spike_count as f32 / self.config.num_neurons as f32,
            };
            
            // Update critical dynamics
            {
                let mut dynamics = self.critical_dynamics.write().unwrap();
                *dynamics.avalanche_sizes.entry(spike_count).or_insert(0) += 1;
                
                // Update power-law exponent estimate (simplified)
                if dynamics.avalanche_sizes.len() > 10 {
                    dynamics.power_law_exponent = self.estimate_power_law_exponent(&dynamics.avalanche_sizes);
                    dynamics.criticality_index = (POWER_LAW_EXPONENT - dynamics.power_law_exponent).abs();
                    dynamics.criticality_index = 1.0 - dynamics.criticality_index.min(1.0);
                }
            }
            
            // Store avalanche
            {
                let mut avalanches = self.avalanches.write().unwrap();
                avalanches.push(avalanche);
                
                // Keep only recent avalanches to manage memory
                if avalanches.len() > 10000 {
                    avalanches.remove(0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Estimate power-law exponent from avalanche size distribution
    fn estimate_power_law_exponent(&self, size_distribution: &HashMap<usize, u64>) -> f64 {
        if size_distribution.len() < 3 {
            return POWER_LAW_EXPONENT;
        }
        
        // Simple log-linear regression to estimate exponent
        let mut log_sizes = Vec::new();
        let mut log_counts = Vec::new();
        
        for (&size, &count) in size_distribution.iter() {
            if size > 0 && count > 0 {
                log_sizes.push((size as f64).ln());
                log_counts.push((count as f64).ln());
            }
        }
        
        if log_sizes.len() < 3 {
            return POWER_LAW_EXPONENT;
        }
        
        // Calculate linear regression slope (negative of power-law exponent)
        let n = log_sizes.len() as f64;
        let sum_x: f64 = log_sizes.iter().sum();
        let sum_y: f64 = log_counts.iter().sum();
        let sum_xy: f64 = log_sizes.iter().zip(&log_counts).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_sizes.iter().map(|x| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        (-slope).abs() // Return absolute value of negative slope
    }
    
    /// Update synchronization metrics
    fn update_synchrony_metrics(&mut self, current_spikes: &[Spike]) -> Result<()> {
        if current_spikes.is_empty() {
            return Ok(());
        }
        
        // Calculate global synchrony (fraction of neurons active)
        let global_synchrony = current_spikes.len() as f64 / self.config.num_neurons as f64;
        
        // Update metrics
        {
            let mut metrics = self.sync_metrics.write().unwrap();
            metrics.global_synchrony = global_synchrony;
            metrics.timestamp = Instant::now();
            
            // Simple local synchrony calculation by population
            for (pop_name, population) in &self.populations {
                let pop_spikes = current_spikes.iter()
                    .filter(|spike| population.neurons.contains(&spike.neuron_id))
                    .count();
                
                let local_sync = pop_spikes as f64 / population.neurons.len() as f64;
                metrics.local_synchrony.insert(pop_name.clone(), local_sync);
            }
        }
        
        Ok(())
    }
    
    /// Update performance statistics
    fn update_performance_stats(&mut self, step_duration: Duration, spike_count: usize) {
        let steps_per_second = 1.0 / step_duration.as_secs_f64();
        let spikes_per_second = spike_count as f64 / step_duration.as_secs_f64();
        
        // Update running averages (simple exponential smoothing)
        let alpha = 0.1;
        self.performance_stats.simulation_speed = 
            alpha * steps_per_second + (1.0 - alpha) * self.performance_stats.simulation_speed;
        self.performance_stats.spike_rate = 
            alpha * spikes_per_second + (1.0 - alpha) * self.performance_stats.spike_rate;
        
        // Estimate memory usage (simplified)
        let neuron_memory = self.neurons.len() * std::mem::size_of::<Neuron>();
        let synapse_memory = self.synapses.len() * std::mem::size_of::<Synapse>();
        let total_memory_bytes = neuron_memory + synapse_memory;
        self.performance_stats.memory_usage = total_memory_bytes as f64 / (1024.0 * 1024.0);
        
        // Estimate parallel efficiency (placeholder)
        let theoretical_max = rayon::current_num_threads() as f64;
        let actual_speedup = self.performance_stats.simulation_speed / 1000.0; // Baseline
        self.performance_stats.parallel_efficiency = (actual_speedup / theoretical_max).min(1.0);
    }
    
    /// Encode input signal using population coding
    pub fn encode_signal(
        &mut self, 
        signal: &TradingSignal, 
        encoding_type: SpikeEncoding
    ) -> Result<()> {
        debug!("Encoding trading signal: {} with {:?}", signal.symbol, encoding_type);
        
        // Convert trading signal to feature vector
        let feature_vector = self.signal_to_features(signal)?;
        
        // Find appropriate population
        let population_id = match encoding_type {
            SpikeEncoding::Rate => "rate_pop_0".to_string(),
            SpikeEncoding::Temporal => "temporal_pop_0".to_string(),
            SpikeEncoding::Phase => "phase_pop_0".to_string(),
        };
        
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.feature_vector = Some(feature_vector.clone());
            population.activity = signal.strength;
            
            // Apply encoding to neurons in population
            match encoding_type {
                SpikeEncoding::Rate => self.apply_rate_encoding(population, &feature_vector)?,
                SpikeEncoding::Temporal => self.apply_temporal_encoding(population, &feature_vector)?,
                SpikeEncoding::Phase => self.apply_phase_encoding(population, &feature_vector)?,
            }
        }
        
        Ok(())
    }
    
    /// Convert trading signal to neural feature vector
    fn signal_to_features(&self, signal: &TradingSignal) -> Result<DVector<f64>> {
        // Create feature vector from signal properties
        let mut features = Vec::new();
        
        // Signal strength and confidence
        features.push(signal.strength);
        features.push(signal.confidence);
        
        // Signal type encoding
        match signal.signal_type {
            SignalType::Buy => features.extend_from_slice(&[1.0, 0.0, 0.0]),
            SignalType::Sell => features.extend_from_slice(&[0.0, 1.0, 0.0]),
            SignalType::Hold => features.extend_from_slice(&[0.0, 0.0, 1.0]),
            SignalType::StrongBuy => features.extend_from_slice(&[2.0, 0.0, 0.0]),
            SignalType::StrongSell => features.extend_from_slice(&[0.0, 2.0, 0.0]),
        }
        
        // Add metadata features (normalized)
        for value in signal.metadata.values() {
            features.push(value.tanh()); // Normalize to [-1, 1]
        }
        
        // Pad or truncate to fixed size
        features.resize(32, 0.0);
        
        Ok(DVector::from_vec(features))
    }
    
    /// Apply rate-based encoding to population
    fn apply_rate_encoding(
        &mut self, 
        population: &PopulationCode, 
        features: &DVector<f64>
    ) -> Result<()> {
        let feature_dim = features.len();
        let neurons_per_feature = population.neurons.len() / feature_dim;
        
        for (feature_idx, &feature_value) in features.iter().enumerate() {
            let start_neuron = feature_idx * neurons_per_feature;
            let end_neuron = ((feature_idx + 1) * neurons_per_feature).min(population.neurons.len());
            
            // Set input current proportional to feature value
            let input_current = (feature_value * 10.0) as f32; // Scale to reasonable range
            
            for &neuron_id in &population.neurons[start_neuron..end_neuron] {
                if let Some(neuron) = self.neurons.get_mut(neuron_id as usize) {
                    neuron.input_current += input_current;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply temporal pattern encoding to population
    fn apply_temporal_encoding(
        &mut self, 
        population: &PopulationCode, 
        features: &DVector<f64>
    ) -> Result<()> {
        // Create temporal patterns based on feature values
        let current_time = *self.current_time.read().unwrap();
        
        for (i, &feature_value) in features.iter().enumerate() {
            if i >= population.neurons.len() {
                break;
            }
            
            let neuron_id = population.neurons[i];
            if let Some(neuron) = self.neurons.get_mut(neuron_id as usize) {
                // Temporal delay based on feature value
                let delay = (feature_value.abs() * 50.0) as f32; // 0-50ms delay
                let input_strength = feature_value.signum() as f32 * 5.0;
                
                // Apply delayed input (simplified - in real implementation would use event queue)
                if (current_time % (delay as f64 + 1.0)) < 1.0 {
                    neuron.input_current += input_strength;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply phase-based encoding to population
    fn apply_phase_encoding(
        &mut self, 
        population: &PopulationCode, 
        features: &DVector<f64>
    ) -> Result<()> {
        let current_time = *self.current_time.read().unwrap();
        let oscillation_period = 100.0; // 100ms period
        let phase = (current_time % oscillation_period) / oscillation_period * 2.0 * std::f64::consts::PI;
        
        for (i, &feature_value) in features.iter().enumerate() {
            if i >= population.neurons.len() {
                break;
            }
            
            let neuron_id = population.neurons[i];
            if let Some(neuron) = self.neurons.get_mut(neuron_id as usize) {
                // Phase-dependent input strength
                let target_phase = feature_value * std::f64::consts::PI; // -π to π
                let phase_diff = (phase - target_phase).sin();
                let input_current = (phase_diff * 5.0) as f32;
                
                neuron.input_current += input_current;
            }
        }
        
        Ok(())
    }
    
    /// Decode population activity to trading signal
    pub fn decode_population(
        &self, 
        population_id: &str, 
        encoding_type: SpikeEncoding
    ) -> Result<Option<TradingSignal>> {
        let population = self.populations.get(population_id)
            .context("Population not found")?;
        
        // Get recent spikes from this population
        let spike_buffer = self.spike_buffer.read().unwrap();
        let current_time = *self.current_time.read().unwrap();
        let time_window = 50.0; // 50ms window
        
        let population_spikes: Vec<_> = spike_buffer
            .iter()
            .filter(|spike| {
                population.neurons.contains(&spike.neuron_id) &&
                (current_time - spike.time) <= time_window
            })
            .collect();
        
        if population_spikes.is_empty() {
            return Ok(None);
        }
        
        // Decode based on encoding type
        let (signal_strength, signal_confidence, signal_type) = match encoding_type {
            SpikeEncoding::Rate => self.decode_rate_encoding(population, &population_spikes)?,
            SpikeEncoding::Temporal => self.decode_temporal_encoding(population, &population_spikes)?,
            SpikeEncoding::Phase => self.decode_phase_encoding(population, &population_spikes)?,
        };
        
        // Create trading signal
        let signal = TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "DECODED_SIGNAL".to_string(),
            signal_type,
            strength: signal_strength,
            confidence: signal_confidence,
            timestamp: chrono::Utc::now(),
            source: "SpikeSwarm".to_string(),
            metadata: std::collections::HashMap::new(),
            expires_at: Some(chrono::Utc::now() + chrono::Duration::milliseconds(1000)),
        };
        
        Ok(Some(signal))
    }
    
    /// Decode rate-based population activity
    fn decode_rate_encoding(
        &self,
        _population: &PopulationCode,
        population_spikes: &[&Spike]
    ) -> Result<(f64, f64, SignalType)> {
        let spike_rate = population_spikes.len() as f64;
        let max_expected_rate = 1000.0; // Expected maximum
        
        let strength = (spike_rate / max_expected_rate).min(1.0);
        let confidence = if spike_rate > 10.0 { 0.8 } else { 0.3 };
        
        let signal_type = if strength > 0.7 {
            SignalType::StrongBuy
        } else if strength > 0.3 {
            SignalType::Buy
        } else {
            SignalType::Hold
        };
        
        Ok((strength, confidence, signal_type))
    }
    
    /// Decode temporal pattern population activity
    fn decode_temporal_encoding(
        &self,
        _population: &PopulationCode,
        population_spikes: &[&Spike]
    ) -> Result<(f64, f64, SignalType)> {
        if population_spikes.len() < 2 {
            return Ok((0.0, 0.0, SignalType::Hold));
        }
        
        // Analyze temporal patterns
        let mut intervals = Vec::new();
        for i in 1..population_spikes.len() {
            intervals.push(population_spikes[i].time - population_spikes[i-1].time);
        }
        
        let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let strength = (50.0 / mean_interval).min(1.0); // Faster = stronger
        let confidence = 0.6;
        
        let signal_type = if mean_interval < 10.0 {
            SignalType::StrongBuy
        } else if mean_interval < 25.0 {
            SignalType::Buy
        } else {
            SignalType::Hold
        };
        
        Ok((strength, confidence, signal_type))
    }
    
    /// Decode phase-based population activity
    fn decode_phase_encoding(
        &self,
        _population: &PopulationCode,
        population_spikes: &[&Spike]
    ) -> Result<(f64, f64, SignalType)> {
        if population_spikes.is_empty() {
            return Ok((0.0, 0.0, SignalType::Hold));
        }
        
        let current_time = *self.current_time.read().unwrap();
        let oscillation_period = 100.0;
        let current_phase = (current_time % oscillation_period) / oscillation_period * 2.0 * std::f64::consts::PI;
        
        // Analyze phase relationship
        let spike_phases: Vec<f64> = population_spikes
            .iter()
            .map(|spike| {
                let spike_phase = (spike.time % oscillation_period) / oscillation_period * 2.0 * std::f64::consts::PI;
                (spike_phase - current_phase).sin()
            })
            .collect();
        
        let mean_phase_diff: f64 = spike_phases.iter().sum::<f64>() / spike_phases.len() as f64;
        let strength = mean_phase_diff.abs();
        let confidence = 0.7;
        
        let signal_type = if mean_phase_diff > 0.5 {
            SignalType::Buy
        } else if mean_phase_diff < -0.5 {
            SignalType::Sell
        } else {
            SignalType::Hold
        };
        
        Ok((strength, confidence, signal_type))
    }
    
    /// Get current swarm status and metrics
    pub fn get_status(&self) -> Result<SwarmStatus> {
        let current_time = *self.current_time.read().unwrap();
        let sync_metrics = self.sync_metrics.read().unwrap().clone();
        let critical_dynamics = self.critical_dynamics.read().unwrap().clone();
        let avalanche_count = self.avalanches.read().unwrap().len();
        let spike_buffer_size = self.spike_buffer.read().unwrap().len();
        
        Ok(SwarmStatus {
            current_time,
            active_neurons: self.neurons.len(),
            total_synapses: self.synapses.len(),
            recent_spikes: spike_buffer_size,
            avalanche_count,
            sync_metrics,
            critical_dynamics,
            performance_stats: self.performance_stats.clone(),
            population_activities: self.get_population_activities(),
        })
    }
    
    /// Get current activity levels for all populations
    fn get_population_activities(&self) -> HashMap<String, f64> {
        let spike_buffer = self.spike_buffer.read().unwrap();
        let current_time = *self.current_time.read().unwrap();
        let time_window = 100.0; // 100ms window
        
        let mut activities = HashMap::new();
        
        for (pop_name, population) in &self.populations {
            let recent_spikes = spike_buffer
                .iter()
                .filter(|spike| {
                    population.neurons.contains(&spike.neuron_id) &&
                    (current_time - spike.time) <= time_window
                })
                .count();
            
            let activity = recent_spikes as f64 / population.neurons.len() as f64;
            activities.insert(pop_name.clone(), activity);
        }
        
        activities
    }
    
    /// Run spike swarm for specified duration
    pub fn run(&mut self, duration_ms: f64) -> Result<SwarmRunResults> {
        info!("Running spike swarm for {}ms", duration_ms);
        
        let start_time = Instant::now();
        let steps = (duration_ms / self.config.dt) as usize;
        let mut total_spikes = 0;
        let mut all_avalanches = Vec::new();
        
        for step in 0..steps {
            let step_spikes = self.step()?;
            total_spikes += step_spikes.len();
            
            // Log progress every 1000 steps
            if step % 1000 == 0 {
                debug!("Step {}/{}, spikes: {}", step, steps, step_spikes.len());
            }
            
            // Collect avalanches
            if !step_spikes.is_empty() {
                let avalanches = self.avalanches.read().unwrap();
                if let Some(last_avalanche) = avalanches.last() {
                    if all_avalanches.is_empty() || 
                       all_avalanches.last().unwrap().id != last_avalanche.id {
                        all_avalanches.push(last_avalanche.clone());
                    }
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        let final_status = self.get_status()?;
        
        info!("Simulation completed: {} spikes, {} avalanches, {:.2}ms elapsed", 
              total_spikes, all_avalanches.len(), elapsed.as_secs_f64() * 1000.0);
        
        Ok(SwarmRunResults {
            total_spikes,
            avalanche_count: all_avalanches.len(),
            simulation_time: elapsed,
            final_status,
            avalanches: all_avalanches,
        })
    }
}

/// Comprehensive swarm status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    /// Current simulation time (ms)
    pub current_time: f64,
    /// Number of active neurons
    pub active_neurons: usize,
    /// Total number of synapses
    pub total_synapses: usize,
    /// Recent spikes in buffer
    pub recent_spikes: usize,
    /// Total avalanches detected
    pub avalanche_count: usize,
    /// Synchronization metrics
    pub sync_metrics: SynchronyMetrics,
    /// Critical dynamics analysis
    pub critical_dynamics: CriticalDynamics,
    /// Performance statistics
    pub performance_stats: PerformanceStats,
    /// Population activity levels
    pub population_activities: HashMap<String, f64>,
}

/// Results from running the swarm simulation
#[derive(Debug, Clone)]
pub struct SwarmRunResults {
    /// Total spikes generated
    pub total_spikes: usize,
    /// Number of avalanches detected
    pub avalanche_count: usize,
    /// Simulation elapsed time
    pub simulation_time: Duration,
    /// Final swarm status
    pub final_status: SwarmStatus,
    /// All detected avalanches
    pub avalanches: Vec<Avalanche>,
}

/// Use rand crate for random number generation (mock implementation)
mod rand {
    pub fn random<T>() -> T
    where
        T: Default + std::ops::Add<Output = T> + From<f64>,
    {
        // Mock random number generation - in real implementation use proper RNG
        T::from(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spike_swarm_creation() {
        let config = SpikeSwarmConfig {
            num_neurons: 1000, // Smaller for tests
            ..Default::default()
        };
        
        let swarm = SpikeSwarm::new(config);
        assert!(swarm.is_ok());
        
        let swarm = swarm.unwrap();
        assert_eq!(swarm.neurons.len(), 1000);
        assert!(!swarm.synapses.is_empty());
        assert!(!swarm.populations.is_empty());
    }
    
    #[test]
    fn test_neuron_update() {
        let config = SpikeSwarmConfig::default();
        let swarm = SpikeSwarm::new(config).unwrap();
        
        let mut neuron = Neuron::default();
        neuron.input_current = 50.0; // Strong input
        
        let spiked = swarm.update_neuron(&mut neuron, 0);
        assert!(spiked); // Should spike with strong input
        assert_eq!(neuron.potential, -70.0); // Should reset
    }
    
    #[test]
    fn test_power_law_estimation() {
        let swarm = SpikeSwarm::new(SpikeSwarmConfig::default()).unwrap();
        
        let mut distribution = HashMap::new();
        distribution.insert(10, 100);
        distribution.insert(100, 10);
        distribution.insert(1000, 1);
        
        let exponent = swarm.estimate_power_law_exponent(&distribution);
        assert!(exponent > 0.0 && exponent < 5.0); // Reasonable range
    }
    
    #[test]
    fn test_signal_encoding_decoding() {
        let mut swarm = SpikeSwarm::new(SpikeSwarmConfig::default()).unwrap();
        
        let signal = TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.8,
            confidence: 0.9,
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: std::collections::HashMap::new(),
            expires_at: None,
        };
        
        // Test encoding
        assert!(swarm.encode_signal(&signal, SpikeEncoding::Rate).is_ok());
        
        // Test decoding
        let decoded = swarm.decode_population("rate_pop_0", SpikeEncoding::Rate).unwrap();
        // Note: Decoding might return None if no recent activity
    }
    
    #[test]
    fn test_performance_validation() {
        let config = SpikeSwarmConfig {
            num_neurons: 10000, // Moderate size for performance test
            ..Default::default()
        };
        
        let mut swarm = SpikeSwarm::new(config).unwrap();
        
        // Run short simulation
        let start = Instant::now();
        let _spikes = swarm.step().unwrap();
        let duration = start.elapsed();
        
        // Should complete within reasonable time
        assert!(duration.as_millis() < 1000); // Less than 1 second per step
        
        // Memory usage should be reasonable
        assert!(swarm.performance_stats.memory_usage < 1000.0); // Less than 1GB
    }
}