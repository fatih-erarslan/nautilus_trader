use std::collections::{HashMap, VecDeque};
use nalgebra::{DVector, DMatrix};
use crate::dynamics::{
    soc::{SelfOrganizedCriticality, AvalancheEvent},
    cas::{ComplexAdaptiveSystem, AdaptiveAgent, FitnessSnapshot},
    phase_space::{PhaseSpaceReconstructor, AttractorAnalysis, AttractorType},
    lattice_dynamics::{LatticeFieldDynamics, LatticeState},
};

/// Emergence detector for identifying emergent behaviors in swarm systems
/// Implements multiple detection algorithms for different types of emergence
pub struct EmergenceDetector {
    /// Detection parameters
    params: DetectionParameters,
    /// Historical data for analysis
    history: EmergenceHistory,
    /// Detection algorithms
    algorithms: DetectionAlgorithms,
    /// Current emergence state
    current_state: EmergenceState,
    /// Alert system
    alerts: Vec<EmergenceAlert>,
}

#[derive(Clone, Debug)]
pub struct DetectionParameters {
    /// Window size for temporal analysis
    pub temporal_window: usize,
    /// Threshold for significance detection
    pub significance_threshold: f64,
    /// Minimum duration for persistent emergence
    pub min_persistence_duration: usize,
    /// Maximum noise level allowed
    pub max_noise_level: f64,
    /// Correlation threshold for coupling detection
    pub correlation_threshold: f64,
    /// Complexity threshold
    pub complexity_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct EmergenceHistory {
    /// Time series of system metrics
    pub metrics_history: VecDeque<SystemMetrics>,
    /// Phase space trajectories
    pub phase_trajectories: VecDeque<DVector<f64>>,
    /// Avalanche events from SOC systems
    pub avalanche_events: VecDeque<AvalancheEvent>,
    /// Agent fitness evolution
    pub fitness_evolution: VecDeque<FitnessSnapshot>,
    /// Lattice state evolution
    pub lattice_states: VecDeque<LatticeState>,
}

#[derive(Clone, Debug)]
pub struct SystemMetrics {
    /// Timestamp
    pub timestamp: f64,
    /// System size/scale
    pub system_size: usize,
    /// Total energy
    pub total_energy: f64,
    /// Entropy measure
    pub entropy: f64,
    /// Information content
    pub information: f64,
    /// Complexity measure
    pub complexity: f64,
    /// Coherence measure
    pub coherence: f64,
    /// Coupling strength
    pub coupling: f64,
}

/// Different algorithms for emergence detection
#[derive(Clone, Debug)]
pub struct DetectionAlgorithms {
    /// Information-theoretic measures
    pub information_theory: InformationTheoryDetector,
    /// Statistical measures
    pub statistical: StatisticalDetector,
    /// Dynamical systems measures
    pub dynamical: DynamicalSystemsDetector,
    /// Network-based measures
    pub network: NetworkEmergenceDetector,
}

/// Current state of emergence detection
#[derive(Clone, Debug)]
pub struct EmergenceState {
    /// Overall emergence score (0-1)
    pub emergence_score: f64,
    /// Detected emergence types
    pub emergence_types: Vec<EmergenceType>,
    /// Confidence level
    pub confidence: f64,
    /// Persistence duration
    pub persistence_duration: usize,
    /// Spatial extent
    pub spatial_extent: f64,
    /// Temporal stability
    pub temporal_stability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmergenceType {
    /// Weak emergence - predictable from components
    Weak,
    /// Strong emergence - irreducible to components
    Strong,
    /// Synchronization emergence
    Synchronization,
    /// Phase transition emergence
    PhaseTransition,
    /// Self-organization emergence
    SelfOrganization,
    /// Collective intelligence emergence
    CollectiveIntelligence,
    /// Pattern formation emergence
    PatternFormation,
    /// Critical behavior emergence
    CriticalBehavior,
}

#[derive(Clone, Debug)]
pub struct EmergenceAlert {
    pub timestamp: f64,
    pub emergence_type: EmergenceType,
    pub intensity: f64,
    pub location: Option<DVector<f64>>,
    pub description: String,
    pub confidence: f64,
}

impl Default for DetectionParameters {
    fn default() -> Self {
        Self {
            temporal_window: 1000,
            significance_threshold: 0.05,
            min_persistence_duration: 50,
            max_noise_level: 0.1,
            correlation_threshold: 0.8,
            complexity_threshold: 0.7,
        }
    }
}

impl EmergenceDetector {
    /// Create new emergence detector
    pub fn new(params: DetectionParameters) -> Self {
        let history = EmergenceHistory {
            metrics_history: VecDeque::with_capacity(params.temporal_window),
            phase_trajectories: VecDeque::with_capacity(params.temporal_window),
            avalanche_events: VecDeque::with_capacity(params.temporal_window),
            fitness_evolution: VecDeque::with_capacity(params.temporal_window),
            lattice_states: VecDeque::with_capacity(params.temporal_window),
        };

        let algorithms = DetectionAlgorithms {
            information_theory: InformationTheoryDetector::new(),
            statistical: StatisticalDetector::new(),
            dynamical: DynamicalSystemsDetector::new(),
            network: NetworkEmergenceDetector::new(),
        };

        let current_state = EmergenceState {
            emergence_score: 0.0,
            emergence_types: Vec::new(),
            confidence: 0.0,
            persistence_duration: 0,
            spatial_extent: 0.0,
            temporal_stability: 0.0,
        };

        Self {
            params,
            history,
            algorithms,
            current_state,
            alerts: Vec::new(),
        }
    }

    /// Update detector with SOC system data
    pub fn update_from_soc(&mut self, soc: &SelfOrganizedCriticality, timestamp: f64) {
        let soc_state = soc.get_state();
        
        let metrics = SystemMetrics {
            timestamp,
            system_size: soc_state.grid_snapshot.len(),
            total_energy: soc_state.total_energy,
            entropy: self.calculate_entropy_from_grid(&soc_state.grid_snapshot),
            information: self.calculate_information_content(&soc_state),
            complexity: self.calculate_complexity(&soc_state),
            coherence: self.calculate_coherence(&soc_state),
            coupling: self.calculate_coupling_strength(&soc_state),
        };

        self.add_metrics(metrics);

        // Add avalanche events
        for event in soc.get_avalanche_history() {
            if event.timestamp >= timestamp - 1.0 { // Recent events
                self.history.avalanche_events.push_back(event.clone());
                if self.history.avalanche_events.len() > self.params.temporal_window {
                    self.history.avalanche_events.pop_front();
                }
            }
        }

        self.detect_emergence();
    }

    /// Update detector with CAS system data
    pub fn update_from_cas(&mut self, cas: &ComplexAdaptiveSystem, timestamp: f64) {
        let cas_state = cas.get_state();
        
        let metrics = SystemMetrics {
            timestamp,
            system_size: cas_state.population_size,
            total_energy: cas_state.fitness_stats.as_ref()
                .map(|s| s.mean_fitness)
                .unwrap_or(0.0),
            entropy: self.calculate_agent_entropy(cas.get_agents()),
            information: self.calculate_information_from_agents(cas.get_agents()),
            complexity: cas_state.fitness_stats.as_ref()
                .map(|s| s.diversity_index)
                .unwrap_or(0.0),
            coherence: cas_state.fitness_stats.as_ref()
                .map(|s| s.cooperation_level)
                .unwrap_or(0.0),
            coupling: cas_state.network_density,
        };

        self.add_metrics(metrics);

        // Add fitness evolution
        if let Some(fitness_stats) = cas_state.fitness_stats {
            self.history.fitness_evolution.push_back(fitness_stats);
            if self.history.fitness_evolution.len() > self.params.temporal_window {
                self.history.fitness_evolution.pop_front();
            }
        }

        self.detect_emergence();
    }

    /// Update detector with phase space data
    pub fn update_from_phase_space(&mut self, reconstructor: &PhaseSpaceReconstructor, timestamp: f64) {
        let phase_space = reconstructor.get_phase_space();
        
        if !phase_space.is_empty() {
            // Add latest phase space point
            self.history.phase_trajectories.push_back(phase_space.last().unwrap().clone());
            if self.history.phase_trajectories.len() > self.params.temporal_window {
                self.history.phase_trajectories.pop_front();
            }

            // Analyze attractor
            let attractor_analysis = reconstructor.analyze_attractor();
            
            let metrics = SystemMetrics {
                timestamp,
                system_size: phase_space.len(),
                total_energy: 0.0, // Phase space doesn't have explicit energy
                entropy: attractor_analysis.entropy,
                information: -attractor_analysis.entropy, // Information ~ negative entropy
                complexity: attractor_analysis.correlation_dimension,
                coherence: attractor_analysis.predictability,
                coupling: if attractor_analysis.largest_lyapunov_exponent > 0.0 { 0.5 } else { 0.8 },
            };

            self.add_metrics(metrics);
        }

        self.detect_emergence();
    }

    /// Update detector with lattice dynamics data
    pub fn update_from_lattice(&mut self, lattice: &LatticeFieldDynamics, timestamp: f64) {
        let lattice_state = lattice.get_state();
        
        let metrics = SystemMetrics {
            timestamp,
            system_size: lattice_state.num_sites,
            total_energy: lattice_state.total_energy,
            entropy: self.calculate_lattice_entropy(&lattice_state),
            information: self.calculate_lattice_information(&lattice_state),
            complexity: lattice_state.topology_changes as f64 / lattice_state.time_step as f64,
            coherence: lattice_state.total_topological_charge.abs(),
            coupling: self.calculate_lattice_coupling(&lattice_state),
        };

        self.add_metrics(metrics);

        // Store lattice state
        self.history.lattice_states.push_back(lattice_state);
        if self.history.lattice_states.len() > self.params.temporal_window {
            self.history.lattice_states.pop_front();
        }

        self.detect_emergence();
    }

    /// Add metrics to history
    fn add_metrics(&mut self, metrics: SystemMetrics) {
        self.history.metrics_history.push_back(metrics);
        if self.history.metrics_history.len() > self.params.temporal_window {
            self.history.metrics_history.pop_front();
        }
    }

    /// Main emergence detection routine
    fn detect_emergence(&mut self) {
        if self.history.metrics_history.len() < 10 {
            return; // Not enough data
        }

        // Run all detection algorithms
        let info_score = self.algorithms.information_theory.detect(&self.history);
        let stat_score = self.algorithms.statistical.detect(&self.history);
        let dyn_score = self.algorithms.dynamical.detect(&self.history);
        let net_score = self.algorithms.network.detect(&self.history);

        // Combine scores
        let combined_score = (info_score + stat_score + dyn_score + net_score) / 4.0;

        // Update emergence state
        self.current_state.emergence_score = combined_score;
        self.current_state.confidence = self.calculate_confidence();

        // Classify emergence types
        self.current_state.emergence_types = self.classify_emergence_types();

        // Update persistence
        if combined_score > self.params.significance_threshold {
            self.current_state.persistence_duration += 1;
        } else {
            self.current_state.persistence_duration = 0;
        }

        // Generate alerts if needed
        self.check_for_alerts();
    }

    /// Calculate confidence in emergence detection
    fn calculate_confidence(&self) -> f64 {
        if self.history.metrics_history.len() < 2 {
            return 0.0;
        }

        // Confidence based on consistency across different measures
        let recent_scores: Vec<f64> = self.history.metrics_history.iter()
            .rev()
            .take(10)
            .map(|m| m.complexity + m.coherence + m.information)
            .collect();

        if recent_scores.is_empty() {
            return 0.0;
        }

        let mean: f64 = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
        let variance: f64 = recent_scores.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_scores.len() as f64;

        let stability = if variance > 0.0 { 1.0 / (1.0 + variance) } else { 1.0 };
        
        // Higher confidence for stable, significant measurements
        stability * self.current_state.emergence_score
    }

    /// Classify types of emergence based on current state
    fn classify_emergence_types(&self) -> Vec<EmergenceType> {
        let mut types = Vec::new();

        if let Some(latest_metrics) = self.history.metrics_history.back() {
            // Strong emergence: high complexity, low predictability
            if latest_metrics.complexity > self.params.complexity_threshold && 
               latest_metrics.coherence < 0.5 {
                types.push(EmergenceType::Strong);
            }

            // Weak emergence: moderate complexity, high predictability
            if latest_metrics.complexity > 0.3 && 
               latest_metrics.coherence > 0.7 {
                types.push(EmergenceType::Weak);
            }

            // Synchronization: high coherence
            if latest_metrics.coherence > 0.9 {
                types.push(EmergenceType::Synchronization);
            }

            // Self-organization: increasing complexity over time
            if self.is_complexity_increasing() {
                types.push(EmergenceType::SelfOrganization);
            }

            // Phase transition: sudden changes in metrics
            if self.detect_phase_transition() {
                types.push(EmergenceType::PhaseTransition);
            }

            // Critical behavior: power-law distributions in avalanches
            if self.detect_critical_behavior() {
                types.push(EmergenceType::CriticalBehavior);
            }

            // Pattern formation: spatial coherence
            if latest_metrics.coherence > 0.7 && latest_metrics.coupling > 0.6 {
                types.push(EmergenceType::PatternFormation);
            }

            // Collective intelligence: increasing fitness with diversity
            if self.detect_collective_intelligence() {
                types.push(EmergenceType::CollectiveIntelligence);
            }
        }

        types
    }

    /// Check if complexity is increasing over time
    fn is_complexity_increasing(&self) -> bool {
        if self.history.metrics_history.len() < 20 {
            return false;
        }

        let recent: f64 = self.history.metrics_history.iter()
            .rev()
            .take(10)
            .map(|m| m.complexity)
            .sum::<f64>() / 10.0;

        let older: f64 = self.history.metrics_history.iter()
            .rev()
            .skip(10)
            .take(10)
            .map(|m| m.complexity)
            .sum::<f64>() / 10.0;

        recent > older * 1.1 // 10% increase
    }

    /// Detect phase transitions
    fn detect_phase_transition(&self) -> bool {
        if self.history.metrics_history.len() < 10 {
            return false;
        }

        // Look for sudden changes in any metric
        let recent_metrics: Vec<_> = self.history.metrics_history.iter()
            .rev()
            .take(5)
            .collect();

        let older_metrics: Vec<_> = self.history.metrics_history.iter()
            .rev()
            .skip(5)
            .take(5)
            .collect();

        if recent_metrics.len() != 5 || older_metrics.len() != 5 {
            return false;
        }

        // Check for sudden changes in key metrics
        let recent_complexity: f64 = recent_metrics.iter().map(|m| m.complexity).sum::<f64>() / 5.0;
        let older_complexity: f64 = older_metrics.iter().map(|m| m.complexity).sum::<f64>() / 5.0;

        let recent_coherence: f64 = recent_metrics.iter().map(|m| m.coherence).sum::<f64>() / 5.0;
        let older_coherence: f64 = older_metrics.iter().map(|m| m.coherence).sum::<f64>() / 5.0;

        // Phase transition if significant change in either metric
        (recent_complexity - older_complexity).abs() > 0.3 ||
        (recent_coherence - older_coherence).abs() > 0.3
    }

    /// Detect critical behavior from avalanche statistics
    fn detect_critical_behavior(&self) -> bool {
        if self.history.avalanche_events.len() < 50 {
            return false;
        }

        // Check for power-law distribution in avalanche sizes
        let mut sizes: Vec<usize> = self.history.avalanche_events.iter()
            .map(|e| e.size)
            .collect();
        sizes.sort();

        // Simple power-law test: check if log-log plot is roughly linear
        let mut log_sizes = Vec::new();
        let mut log_ranks = Vec::new();

        for (i, &size) in sizes.iter().enumerate() {
            if size > 0 {
                log_sizes.push((size as f64).ln());
                log_ranks.push(((i + 1) as f64).ln());
            }
        }

        if log_sizes.len() < 10 {
            return false;
        }

        // Linear regression to check for power-law
        let correlation = self.calculate_correlation(&log_sizes, &log_ranks);
        correlation.abs() > 0.8 // Strong negative correlation indicates power-law
    }

    /// Detect collective intelligence
    fn detect_collective_intelligence(&self) -> bool {
        if self.history.fitness_evolution.len() < 10 {
            return false;
        }

        // Collective intelligence: improving mean fitness with maintained diversity
        let recent_fitness: Vec<_> = self.history.fitness_evolution.iter()
            .rev()
            .take(5)
            .collect();

        let older_fitness: Vec<_> = self.history.fitness_evolution.iter()
            .rev()
            .skip(5)
            .take(5)
            .collect();

        if recent_fitness.len() != 5 || older_fitness.len() != 5 {
            return false;
        }

        let recent_mean: f64 = recent_fitness.iter().map(|f| f.mean_fitness).sum::<f64>() / 5.0;
        let older_mean: f64 = older_fitness.iter().map(|f| f.mean_fitness).sum::<f64>() / 5.0;

        let recent_diversity: f64 = recent_fitness.iter().map(|f| f.diversity_index).sum::<f64>() / 5.0;

        // Improving fitness with maintained diversity
        recent_mean > older_mean * 1.05 && recent_diversity > 0.3
    }

    /// Check for alerts and generate them
    fn check_for_alerts(&mut self) {
        let current_time = self.history.metrics_history.back()
            .map(|m| m.timestamp)
            .unwrap_or(0.0);

        // Generate alert for persistent strong emergence
        if self.current_state.emergence_score > 0.8 && 
           self.current_state.persistence_duration > self.params.min_persistence_duration {
            
            let alert = EmergenceAlert {
                timestamp: current_time,
                emergence_type: EmergenceType::Strong,
                intensity: self.current_state.emergence_score,
                location: None,
                description: format!(
                    "Strong emergence detected with score {:.3} persisting for {} steps",
                    self.current_state.emergence_score,
                    self.current_state.persistence_duration
                ),
                confidence: self.current_state.confidence,
            };

            self.alerts.push(alert);
        }

        // Generate alerts for specific emergence types
        for emergence_type in &self.current_state.emergence_types {
            if matches!(emergence_type, EmergenceType::PhaseTransition | EmergenceType::CriticalBehavior) {
                let alert = EmergenceAlert {
                    timestamp: current_time,
                    emergence_type: emergence_type.clone(),
                    intensity: self.current_state.emergence_score,
                    location: None,
                    description: format!("Detected emergence type: {:?}", emergence_type),
                    confidence: self.current_state.confidence,
                };

                self.alerts.push(alert);
            }
        }

        // Keep only recent alerts
        self.alerts.retain(|alert| current_time - alert.timestamp < 100.0);
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
        let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate entropy from SOC grid
    fn calculate_entropy_from_grid(&self, grid: &nalgebra::DMatrix<f64>) -> f64 {
        // Histogram-based entropy calculation
        let values: Vec<f64> = grid.iter().cloned().collect();
        if values.is_empty() {
            return 0.0;
        }

        let min_val = values.iter().fold(f64::INFINITY, |min, &val| val.min(min));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |max, &val| val.max(max));

        if max_val <= min_val {
            return 0.0;
        }

        let bins = 50;
        let bin_width = (max_val - min_val) / bins as f64;
        let mut histogram = vec![0; bins];

        for &value in &values {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            histogram[bin_idx] += 1;
        }

        let total = values.len() as f64;
        let mut entropy = 0.0;

        for &count in &histogram {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Helper methods for various entropy and information calculations
    fn calculate_information_content(&self, soc_state: &crate::dynamics::soc::SocState) -> f64 {
        // Information content based on critical sites and energy distribution
        let critical_ratio = soc_state.critical_sites as f64 / soc_state.grid_snapshot.len() as f64;
        -critical_ratio * critical_ratio.ln() - (1.0 - critical_ratio) * (1.0 - critical_ratio).ln()
    }

    fn calculate_complexity(&self, soc_state: &crate::dynamics::soc::SocState) -> f64 {
        // Complexity based on energy distribution and criticality
        let energy_ratio = soc_state.max_energy / (soc_state.total_energy + 1e-10);
        1.0 - energy_ratio // Higher complexity when energy is more distributed
    }

    fn calculate_coherence(&self, soc_state: &crate::dynamics::soc::SocState) -> f64 {
        // Coherence based on avalanche patterns
        soc_state.critical_sites as f64 / (soc_state.grid_snapshot.len() as f64).sqrt()
    }

    fn calculate_coupling_strength(&self, soc_state: &crate::dynamics::soc::SocState) -> f64 {
        // Coupling strength based on total energy and criticality
        (soc_state.total_energy / soc_state.grid_snapshot.len() as f64).tanh()
    }

    fn calculate_agent_entropy(&self, agents: &[AdaptiveAgent]) -> f64 {
        if agents.is_empty() {
            return 0.0;
        }

        // Strategy diversity entropy
        let mut strategy_sum = DVector::zeros(agents[0].strategy.len());
        for agent in agents {
            strategy_sum += &agent.strategy;
        }
        strategy_sum /= agents.len() as f64;

        let mut entropy = 0.0;
        for agent in agents {
            let diff = &agent.strategy - &strategy_sum;
            entropy += diff.norm_squared();
        }

        entropy / agents.len() as f64
    }

    fn calculate_information_from_agents(&self, agents: &[AdaptiveAgent]) -> f64 {
        if agents.is_empty() {
            return 0.0;
        }

        // Information based on agent type distribution
        let mut type_counts = std::collections::HashMap::new();
        for agent in agents {
            let type_name = format!("{:?}", agent.agent_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        let total = agents.len() as f64;
        let mut information = 0.0;

        for &count in type_counts.values() {
            if count > 0 {
                let p = count as f64 / total;
                information -= p * p.ln();
            }
        }

        information
    }

    fn calculate_lattice_entropy(&self, lattice_state: &LatticeState) -> f64 {
        // Entropy based on level distribution
        let total_sites = lattice_state.num_sites as f64;
        let mut entropy = 0.0;

        for &count in lattice_state.lattice_levels.values() {
            if count > 0 {
                let p = count as f64 / total_sites;
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    fn calculate_lattice_information(&self, lattice_state: &LatticeState) -> f64 {
        // Information based on topological changes
        if lattice_state.time_step > 0 {
            lattice_state.topology_changes as f64 / lattice_state.time_step as f64
        } else {
            0.0
        }
    }

    fn calculate_lattice_coupling(&self, lattice_state: &LatticeState) -> f64 {
        // Coupling based on energy density and topological charge
        lattice_state.total_topological_charge.abs() / (lattice_state.total_energy + 1.0)
    }

    /// Get current emergence state
    pub fn get_emergence_state(&self) -> &EmergenceState {
        &self.current_state
    }

    /// Get recent alerts
    pub fn get_alerts(&self) -> &[EmergenceAlert] {
        &self.alerts
    }

    /// Get detection history
    pub fn get_history(&self) -> &EmergenceHistory {
        &self.history
    }

    /// Clear all history and reset
    pub fn reset(&mut self) {
        self.history.metrics_history.clear();
        self.history.phase_trajectories.clear();
        self.history.avalanche_events.clear();
        self.history.fitness_evolution.clear();
        self.history.lattice_states.clear();
        self.alerts.clear();
        
        self.current_state = EmergenceState {
            emergence_score: 0.0,
            emergence_types: Vec::new(),
            confidence: 0.0,
            persistence_duration: 0,
            spatial_extent: 0.0,
            temporal_stability: 0.0,
        };
    }
}

/// Information theory based emergence detection
#[derive(Clone, Debug)]
pub struct InformationTheoryDetector {}

impl InformationTheoryDetector {
    fn new() -> Self {
        Self {}
    }

    fn detect(&self, history: &EmergenceHistory) -> f64 {
        if history.metrics_history.len() < 2 {
            return 0.0;
        }

        // Information-theoretic emergence based on mutual information
        let recent_info: f64 = history.metrics_history.iter()
            .rev()
            .take(10)
            .map(|m| m.information)
            .sum::<f64>() / 10.0;

        recent_info.tanh() // Normalize to [0,1]
    }
}

/// Statistical emergence detection
#[derive(Clone, Debug)]
pub struct StatisticalDetector {}

impl StatisticalDetector {
    fn new() -> Self {
        Self {}
    }

    fn detect(&self, history: &EmergenceHistory) -> f64 {
        if history.metrics_history.len() < 10 {
            return 0.0;
        }

        // Statistical significance of complexity increase
        let complexities: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.complexity)
            .collect();

        if complexities.len() < 10 {
            return 0.0;
        }

        let recent: f64 = complexities.iter().rev().take(5).sum::<f64>() / 5.0;
        let older: f64 = complexities.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

        ((recent - older) / (older + 1e-10)).tanh().max(0.0)
    }
}

/// Dynamical systems emergence detection
#[derive(Clone, Debug)]
pub struct DynamicalSystemsDetector {}

impl DynamicalSystemsDetector {
    fn new() -> Self {
        Self {}
    }

    fn detect(&self, history: &EmergenceHistory) -> f64 {
        // Based on phase space trajectories
        if history.phase_trajectories.len() < 10 {
            return 0.0;
        }

        // Measure trajectory complexity
        let mut total_distance = 0.0;
        let trajectories: Vec<_> = history.phase_trajectories.iter().collect();

        for i in 1..trajectories.len() {
            total_distance += (trajectories[i] - trajectories[i-1]).norm();
        }

        let avg_distance = total_distance / (trajectories.len() - 1) as f64;
        (avg_distance / 10.0).tanh() // Normalize
    }
}

/// Network-based emergence detection
#[derive(Clone, Debug)]
pub struct NetworkEmergenceDetector {}

impl NetworkEmergenceDetector {
    fn new() -> Self {
        Self {}
    }

    fn detect(&self, history: &EmergenceHistory) -> f64 {
        if history.metrics_history.len() < 2 {
            return 0.0;
        }

        // Network emergence based on coupling strength
        let recent_coupling: f64 = history.metrics_history.iter()
            .rev()
            .take(10)
            .map(|m| m.coupling)
            .sum::<f64>() / 10.0;

        recent_coupling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergence_detector_creation() {
        let params = DetectionParameters::default();
        let detector = EmergenceDetector::new(params);
        
        assert_eq!(detector.current_state.emergence_score, 0.0);
        assert!(detector.current_state.emergence_types.is_empty());
    }

    #[test]
    fn test_metrics_addition() {
        let params = DetectionParameters::default();
        let mut detector = EmergenceDetector::new(params);
        
        let metrics = SystemMetrics {
            timestamp: 1.0,
            system_size: 100,
            total_energy: 50.0,
            entropy: 2.5,
            information: 1.8,
            complexity: 0.7,
            coherence: 0.6,
            coupling: 0.5,
        };
        
        detector.add_metrics(metrics);
        
        assert_eq!(detector.history.metrics_history.len(), 1);
        assert_eq!(detector.history.metrics_history[0].timestamp, 1.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let params = DetectionParameters::default();
        let detector = EmergenceDetector::new(params);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation
        
        let correlation = detector.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_emergence_type_classification() {
        let params = DetectionParameters::default();
        let mut detector = EmergenceDetector::new(params);
        
        // Add metrics indicating strong emergence
        let metrics = SystemMetrics {
            timestamp: 1.0,
            system_size: 100,
            total_energy: 50.0,
            entropy: 3.0,
            information: 2.5,
            complexity: 0.8, // High complexity
            coherence: 0.3,   // Low coherence
            coupling: 0.7,
        };
        
        detector.add_metrics(metrics);
        
        let types = detector.classify_emergence_types();
        assert!(types.contains(&EmergenceType::Strong));
    }
}