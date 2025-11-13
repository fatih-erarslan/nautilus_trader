//! Real-time consciousness and performance dashboard
//!
//! This module implements the enterprise-grade monitoring dashboard
//! as specified in Layer 7 of the blueprint architecture.

use hyperphysics_core::Result;
use hyperphysics_consciousness::{EmergenceEvent, EmergenceLevel, HierarchicalResult};
use hyperphysics_geometry::HyperbolicTessellation;
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};

/// Consciousness state for a single node or region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Node or region identifier
    pub id: usize,
    /// Integrated information (Φ) value
    pub phi: f64,
    /// Resonance complexity (CI) value
    pub ci: f64,
    /// Spatial position
    pub position: [f64; 3],
    /// Timestamp of measurement
    pub timestamp: SystemTime,
}

/// Real-time consciousness and performance dashboard
pub struct Dashboard {
    phi_monitor: PhiMonitor,
    ci_monitor: CIMonitor,
    energy_monitor: EnergyMonitor,
    performance_monitor: PerformanceMonitor,
    update_interval: Duration,
    last_update: Instant,
}

/// Φ (Integrated Information) monitoring
pub struct PhiMonitor {
    phi_history: VecDeque<PhiDataPoint>,
    max_history_size: usize,
    current_phi: f64,
    phi_trend: PhiTrend,
    emergence_events: Vec<EmergenceEvent>,
}

/// CI (Resonance Complexity Index) monitoring
pub struct CIMonitor {
    ci_history: VecDeque<CIDataPoint>,
    max_history_size: usize,
    current_ci: f64,
    fractal_dimension: f64,
    gain_factor: f64,
    coherence_level: f64,
    dwell_time: f64,
}

/// Energy flow monitoring
pub struct EnergyMonitor {
    energy_history: VecDeque<EnergyDataPoint>,
    max_history_size: usize,
    current_energy: f64,
    entropy: f64,
    negentropy: f64,
    landauer_violations: u32,
    heat_dissipation_rate: f64,
}

/// Performance metrics monitoring
pub struct PerformanceMonitor {
    performance_history: VecDeque<PerformanceDataPoint>,
    max_history_size: usize,
    current_fps: f64,
    gpu_utilization: f64,
    memory_usage: f64,
    computation_rate: f64,
    bottlenecks: Vec<BottleneckType>,
}

/// Φ data point for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiDataPoint {
    pub timestamp: SystemTime,
    pub phi_value: f64,
    pub node_count: usize,
    pub emergence_level: EmergenceLevel,
    pub integration_strength: f64,
}

/// CI data point for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIDataPoint {
    pub timestamp: SystemTime,
    pub ci_value: f64,
    pub fractal_dimension: f64,
    pub gain: f64,
    pub coherence: f64,
    pub dwell_time: f64,
}

/// Energy data point for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyDataPoint {
    pub timestamp: SystemTime,
    pub total_energy: f64,
    pub entropy: f64,
    pub negentropy: f64,
    pub heat_dissipated: f64,
    pub landauer_bound_satisfied: bool,
}

/// Performance data point for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    pub timestamp: SystemTime,
    pub fps: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub computation_rate_hz: f64,
    pub latency_ms: f64,
}

/// Φ trend analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhiTrend {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Emergent,
}

/// Performance bottleneck types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckType {
    GPU,
    Memory,
    CPU,
    NetworkIO,
    Computation,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub update_rate_hz: f64,
    pub history_duration_seconds: f64,
    pub phi_alert_threshold: f64,
    pub energy_alert_threshold: f64,
    pub performance_alert_threshold: f64,
    pub enable_emergence_detection: bool,
    pub enable_real_time_alerts: bool,
}

impl Dashboard {
    /// Create new dashboard with specified configuration
    pub fn new(config: DashboardConfig) -> Self {
        let update_interval = Duration::from_secs_f64(1.0 / config.update_rate_hz);
        let max_history_size = (config.history_duration_seconds * config.update_rate_hz) as usize;
        
        Self {
            phi_monitor: PhiMonitor::new(max_history_size),
            ci_monitor: CIMonitor::new(max_history_size),
            energy_monitor: EnergyMonitor::new(max_history_size),
            performance_monitor: PerformanceMonitor::new(max_history_size),
            update_interval,
            last_update: Instant::now(),
        }
    }
    
    /// Update dashboard with new consciousness and performance data
    pub fn update(
        &mut self,
        consciousness_states: &[ConsciousnessState],
        hierarchical_result: &HierarchicalResult,
        energy_data: &EnergyData,
        performance_data: &PerformanceData,
    ) -> Result<DashboardUpdate> {
        let now = Instant::now();
        
        // Check if update interval has elapsed
        if now.duration_since(self.last_update) < self.update_interval {
            return Ok(DashboardUpdate::NoUpdate);
        }
        
        self.last_update = now;
        let timestamp = SystemTime::now();
        
        // Update Φ monitoring
        let phi_update = self.phi_monitor.update(
            consciousness_states,
            hierarchical_result,
            timestamp,
        )?;
        
        // Update CI monitoring
        let ci_update = self.ci_monitor.update(
            consciousness_states,
            timestamp,
        )?;
        
        // Update energy monitoring
        let energy_update = self.energy_monitor.update(
            energy_data,
            timestamp,
        )?;
        
        // Update performance monitoring
        let performance_update = self.performance_monitor.update(
            performance_data,
            timestamp,
        )?;
        
        // Generate alerts if necessary
        let alerts = self.generate_alerts(&phi_update, &energy_update, &performance_update);
        
        Ok(DashboardUpdate::Updated {
            phi_update,
            ci_update,
            energy_update,
            performance_update,
            alerts,
            timestamp,
        })
    }
    
    /// Get current dashboard state
    pub fn get_current_state(&self) -> DashboardState {
        DashboardState {
            phi_state: self.phi_monitor.get_current_state(),
            ci_state: self.ci_monitor.get_current_state(),
            energy_state: self.energy_monitor.get_current_state(),
            performance_state: self.performance_monitor.get_current_state(),
            last_update: self.last_update,
        }
    }
    
    /// Get time series data for visualization
    pub fn get_time_series(&self, duration: Duration) -> TimeSeriesData {
        let cutoff_time = SystemTime::now() - duration;
        
        TimeSeriesData {
            phi_series: self.phi_monitor.get_time_series(cutoff_time),
            ci_series: self.ci_monitor.get_time_series(cutoff_time),
            energy_series: self.energy_monitor.get_time_series(cutoff_time),
            performance_series: self.performance_monitor.get_time_series(cutoff_time),
        }
    }
    
    /// Generate alerts based on current data
    fn generate_alerts(
        &self,
        phi_update: &PhiUpdate,
        energy_update: &EnergyUpdate,
        performance_update: &PerformanceUpdate,
    ) -> Vec<DashboardAlert> {
        let mut alerts = Vec::new();
        
        // Φ alerts
        if phi_update.emergence_detected {
            alerts.push(DashboardAlert {
                alert_type: AlertType::EmergenceDetected,
                severity: AlertSeverity::High,
                message: format!(
                    "Consciousness emergence detected: Φ = {:.3}",
                    phi_update.current_phi
                ),
                timestamp: SystemTime::now(),
            });
        }
        
        // Energy alerts
        if energy_update.landauer_violations > 0 {
            alerts.push(DashboardAlert {
                alert_type: AlertType::LandauerViolation,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Landauer bound violated {} times",
                    energy_update.landauer_violations
                ),
                timestamp: SystemTime::now(),
            });
        }
        
        // Performance alerts
        if performance_update.fps < 30.0 {
            alerts.push(DashboardAlert {
                alert_type: AlertType::PerformanceDegradation,
                severity: AlertSeverity::Medium,
                message: format!(
                    "Low frame rate: {:.1} FPS",
                    performance_update.fps
                ),
                timestamp: SystemTime::now(),
            });
        }
        
        alerts
    }
}

impl PhiMonitor {
    fn new(max_history_size: usize) -> Self {
        Self {
            phi_history: VecDeque::new(),
            max_history_size,
            current_phi: 0.0,
            phi_trend: PhiTrend::Stable,
            emergence_events: Vec::new(),
        }
    }
    
    fn update(
        &mut self,
        consciousness_states: &[ConsciousnessState],
        hierarchical_result: &HierarchicalResult,
        timestamp: SystemTime,
    ) -> Result<PhiUpdate> {
        // Calculate current Φ
        let current_phi = hierarchical_result.total_phi;
        
        // Detect emergence
        let emergence_detected = !hierarchical_result.emergence_events.is_empty();
        
        // Create data point
        let data_point = PhiDataPoint {
            timestamp,
            phi_value: current_phi,
            node_count: consciousness_states.len(),
            emergence_level: hierarchical_result.overall_emergence,
            integration_strength: hierarchical_result.total_integration(),
        };
        
        // Add to history
        self.phi_history.push_back(data_point);
        while self.phi_history.len() > self.max_history_size {
            self.phi_history.pop_front();
        }
        
        // Update trend analysis
        self.phi_trend = self.analyze_phi_trend();
        
        // Store emergence events
        self.emergence_events.extend(hierarchical_result.emergence_events.clone());
        
        // Update current value
        let previous_phi = self.current_phi;
        self.current_phi = current_phi;
        
        Ok(PhiUpdate {
            current_phi,
            previous_phi,
            phi_change: current_phi - previous_phi,
            trend: self.phi_trend.clone(),
            emergence_detected,
            node_count: consciousness_states.len(),
        })
    }
    
    fn analyze_phi_trend(&self) -> PhiTrend {
        if self.phi_history.len() < 5 {
            return PhiTrend::Stable;
        }
        
        let recent_values: Vec<f64> = self.phi_history
            .iter()
            .rev()
            .take(5)
            .map(|dp| dp.phi_value)
            .collect();
        
        // Simple trend analysis
        let mut increasing_count = 0;
        let mut decreasing_count = 0;
        
        for i in 1..recent_values.len() {
            if recent_values[i] > recent_values[i-1] {
                increasing_count += 1;
            } else if recent_values[i] < recent_values[i-1] {
                decreasing_count += 1;
            }
        }
        
        // Check for emergence (rapid increase)
        if increasing_count >= 3 && recent_values[0] > recent_values[4] * 2.0 {
            PhiTrend::Emergent
        } else if increasing_count >= 3 {
            PhiTrend::Increasing
        } else if decreasing_count >= 3 {
            PhiTrend::Decreasing
        } else if increasing_count > 0 && decreasing_count > 0 {
            PhiTrend::Oscillating
        } else {
            PhiTrend::Stable
        }
    }
    
    fn get_current_state(&self) -> PhiState {
        PhiState {
            current_phi: self.current_phi,
            trend: self.phi_trend.clone(),
            recent_emergence_events: self.emergence_events.len(),
            history_length: self.phi_history.len(),
        }
    }
    
    fn get_time_series(&self, cutoff_time: SystemTime) -> Vec<PhiDataPoint> {
        self.phi_history
            .iter()
            .filter(|dp| dp.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
}

impl CIMonitor {
    fn new(max_history_size: usize) -> Self {
        Self {
            ci_history: VecDeque::new(),
            max_history_size,
            current_ci: 0.0,
            fractal_dimension: 0.0,
            gain_factor: 0.0,
            coherence_level: 0.0,
            dwell_time: 0.0,
        }
    }
    
    fn update(
        &mut self,
        consciousness_states: &[ConsciousnessState],
        timestamp: SystemTime,
    ) -> Result<CIUpdate> {
        // Calculate CI components (simplified)
        let fractal_dimension = self.calculate_fractal_dimension(consciousness_states);
        let gain_factor = self.calculate_gain_factor(consciousness_states);
        let coherence_level = self.calculate_coherence(consciousness_states);
        let dwell_time = self.calculate_dwell_time(consciousness_states);
        
        // Calculate CI using RCT formula: CI = D^α * G^β * C^γ * τ^δ
        let alpha = 1.0; // Empirically determined exponents
        let beta = 0.5;
        let gamma = 0.7;
        let delta = 0.3;
        
        let current_ci = fractal_dimension.powf(alpha) *
                        gain_factor.powf(beta) *
                        coherence_level.powf(gamma) *
                        dwell_time.powf(delta);
        
        // Create data point
        let data_point = CIDataPoint {
            timestamp,
            ci_value: current_ci,
            fractal_dimension,
            gain: gain_factor,
            coherence: coherence_level,
            dwell_time,
        };
        
        // Add to history
        self.ci_history.push_back(data_point);
        while self.ci_history.len() > self.max_history_size {
            self.ci_history.pop_front();
        }
        
        // Update current values
        let previous_ci = self.current_ci;
        self.current_ci = current_ci;
        self.fractal_dimension = fractal_dimension;
        self.gain_factor = gain_factor;
        self.coherence_level = coherence_level;
        self.dwell_time = dwell_time;
        
        Ok(CIUpdate {
            current_ci,
            previous_ci,
            ci_change: current_ci - previous_ci,
            fractal_dimension,
            gain_factor,
            coherence_level,
            dwell_time,
        })
    }
    
    fn calculate_fractal_dimension(&self, states: &[ConsciousnessState]) -> f64 {
        // Simplified fractal dimension calculation using box-counting
        // Real implementation would use proper fractal analysis
        if states.is_empty() {
            return 0.0;
        }
        
        let active_states = states.iter().filter(|s| s.phi > 0.0).count();
        let total_states = states.len();
        
        if active_states == 0 {
            0.0
        } else {
            (active_states as f64 / total_states as f64).ln() / (1.0f64 / total_states as f64).ln()
        }
    }
    
    fn calculate_gain_factor(&self, states: &[ConsciousnessState]) -> f64 {
        // Simplified gain calculation
        if states.is_empty() {
            return 1.0;
        }
        
        let max_phi = states.iter().map(|s| s.phi).fold(0.0, f64::max);
        let mean_phi = states.iter().map(|s| s.phi).sum::<f64>() / states.len() as f64;
        
        if mean_phi > 0.0 {
            max_phi / mean_phi
        } else {
            1.0
        }
    }
    
    fn calculate_coherence(&self, states: &[ConsciousnessState]) -> f64 {
        // Simplified coherence calculation
        if states.len() < 2 {
            return 1.0;
        }
        
        let mean_phi = states.iter().map(|s| s.phi).sum::<f64>() / states.len() as f64;
        let variance = states.iter()
            .map(|s| (s.phi - mean_phi).powi(2))
            .sum::<f64>() / states.len() as f64;
        
        // Higher coherence = lower variance
        1.0 / (1.0 + variance)
    }
    
    fn calculate_dwell_time(&self, _states: &[ConsciousnessState]) -> f64 {
        // Simplified dwell time - would need temporal analysis in real implementation
        1.0
    }
    
    fn get_current_state(&self) -> CIState {
        CIState {
            current_ci: self.current_ci,
            fractal_dimension: self.fractal_dimension,
            gain_factor: self.gain_factor,
            coherence_level: self.coherence_level,
            dwell_time: self.dwell_time,
            history_length: self.ci_history.len(),
        }
    }
    
    fn get_time_series(&self, cutoff_time: SystemTime) -> Vec<CIDataPoint> {
        self.ci_history
            .iter()
            .filter(|dp| dp.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
}

impl EnergyMonitor {
    fn new(max_history_size: usize) -> Self {
        Self {
            energy_history: VecDeque::new(),
            max_history_size,
            current_energy: 0.0,
            entropy: 0.0,
            negentropy: 0.0,
            landauer_violations: 0,
            heat_dissipation_rate: 0.0,
        }
    }
    
    fn update(
        &mut self,
        energy_data: &EnergyData,
        timestamp: SystemTime,
    ) -> Result<EnergyUpdate> {
        // Create data point
        let data_point = EnergyDataPoint {
            timestamp,
            total_energy: energy_data.total_energy,
            entropy: energy_data.entropy,
            negentropy: energy_data.negentropy,
            heat_dissipated: energy_data.heat_dissipated,
            landauer_bound_satisfied: energy_data.landauer_bound_satisfied,
        };
        
        // Add to history
        self.energy_history.push_back(data_point);
        while self.energy_history.len() > self.max_history_size {
            self.energy_history.pop_front();
        }
        
        // Update current values
        let previous_energy = self.current_energy;
        self.current_energy = energy_data.total_energy;
        self.entropy = energy_data.entropy;
        self.negentropy = energy_data.negentropy;
        self.heat_dissipation_rate = energy_data.heat_dissipation_rate;
        
        // Count Landauer violations
        if !energy_data.landauer_bound_satisfied {
            self.landauer_violations += 1;
        }
        
        Ok(EnergyUpdate {
            current_energy: self.current_energy,
            previous_energy,
            energy_change: self.current_energy - previous_energy,
            entropy: self.entropy,
            negentropy: self.negentropy,
            landauer_violations: if !energy_data.landauer_bound_satisfied { 1 } else { 0 },
            heat_dissipation_rate: self.heat_dissipation_rate,
        })
    }
    
    fn get_current_state(&self) -> EnergyState {
        EnergyState {
            current_energy: self.current_energy,
            entropy: self.entropy,
            negentropy: self.negentropy,
            total_landauer_violations: self.landauer_violations,
            heat_dissipation_rate: self.heat_dissipation_rate,
            history_length: self.energy_history.len(),
        }
    }
    
    fn get_time_series(&self, cutoff_time: SystemTime) -> Vec<EnergyDataPoint> {
        self.energy_history
            .iter()
            .filter(|dp| dp.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
}

impl PerformanceMonitor {
    fn new(max_history_size: usize) -> Self {
        Self {
            performance_history: VecDeque::new(),
            max_history_size,
            current_fps: 0.0,
            gpu_utilization: 0.0,
            memory_usage: 0.0,
            computation_rate: 0.0,
            bottlenecks: Vec::new(),
        }
    }
    
    fn update(
        &mut self,
        performance_data: &PerformanceData,
        timestamp: SystemTime,
    ) -> Result<PerformanceUpdate> {
        // Create data point
        let data_point = PerformanceDataPoint {
            timestamp,
            fps: performance_data.fps,
            gpu_utilization: performance_data.gpu_utilization,
            memory_usage_mb: performance_data.memory_usage_mb,
            computation_rate_hz: performance_data.computation_rate_hz,
            latency_ms: performance_data.latency_ms,
        };
        
        // Add to history
        self.performance_history.push_back(data_point);
        while self.performance_history.len() > self.max_history_size {
            self.performance_history.pop_front();
        }
        
        // Update current values
        let previous_fps = self.current_fps;
        self.current_fps = performance_data.fps;
        self.gpu_utilization = performance_data.gpu_utilization;
        self.memory_usage = performance_data.memory_usage_mb;
        self.computation_rate = performance_data.computation_rate_hz;
        
        // Detect bottlenecks
        self.bottlenecks = self.detect_bottlenecks(performance_data);
        
        Ok(PerformanceUpdate {
            fps: self.current_fps,
            previous_fps,
            fps_change: self.current_fps - previous_fps,
            gpu_utilization: self.gpu_utilization,
            memory_usage: self.memory_usage,
            computation_rate: self.computation_rate,
            bottlenecks: self.bottlenecks.clone(),
        })
    }
    
    fn detect_bottlenecks(&self, data: &PerformanceData) -> Vec<BottleneckType> {
        let mut bottlenecks = Vec::new();
        
        if data.gpu_utilization > 90.0 {
            bottlenecks.push(BottleneckType::GPU);
        }
        
        if data.memory_usage_mb > 7000.0 { // > 7GB
            bottlenecks.push(BottleneckType::Memory);
        }
        
        if data.fps < 30.0 {
            bottlenecks.push(BottleneckType::Computation);
        }
        
        bottlenecks
    }
    
    fn get_current_state(&self) -> PerformanceState {
        PerformanceState {
            current_fps: self.current_fps,
            gpu_utilization: self.gpu_utilization,
            memory_usage: self.memory_usage,
            computation_rate: self.computation_rate,
            active_bottlenecks: self.bottlenecks.clone(),
            history_length: self.performance_history.len(),
        }
    }
    
    fn get_time_series(&self, cutoff_time: SystemTime) -> Vec<PerformanceDataPoint> {
        self.performance_history
            .iter()
            .filter(|dp| dp.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct EnergyData {
    pub total_energy: f64,
    pub entropy: f64,
    pub negentropy: f64,
    pub heat_dissipated: f64,
    pub heat_dissipation_rate: f64,
    pub landauer_bound_satisfied: bool,
}

#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub fps: f64,
    pub gpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub computation_rate_hz: f64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub enum DashboardUpdate {
    NoUpdate,
    Updated {
        phi_update: PhiUpdate,
        ci_update: CIUpdate,
        energy_update: EnergyUpdate,
        performance_update: PerformanceUpdate,
        alerts: Vec<DashboardAlert>,
        timestamp: SystemTime,
    },
}

#[derive(Debug, Clone)]
pub struct PhiUpdate {
    pub current_phi: f64,
    pub previous_phi: f64,
    pub phi_change: f64,
    pub trend: PhiTrend,
    pub emergence_detected: bool,
    pub node_count: usize,
}

#[derive(Debug, Clone)]
pub struct CIUpdate {
    pub current_ci: f64,
    pub previous_ci: f64,
    pub ci_change: f64,
    pub fractal_dimension: f64,
    pub gain_factor: f64,
    pub coherence_level: f64,
    pub dwell_time: f64,
}

#[derive(Debug, Clone)]
pub struct EnergyUpdate {
    pub current_energy: f64,
    pub previous_energy: f64,
    pub energy_change: f64,
    pub entropy: f64,
    pub negentropy: f64,
    pub landauer_violations: u32,
    pub heat_dissipation_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceUpdate {
    pub fps: f64,
    pub previous_fps: f64,
    pub fps_change: f64,
    pub gpu_utilization: f64,
    pub memory_usage: f64,
    pub computation_rate: f64,
    pub bottlenecks: Vec<BottleneckType>,
}

#[derive(Debug, Clone)]
pub struct DashboardAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    EmergenceDetected,
    LandauerViolation,
    PerformanceDegradation,
    MemoryExhaustion,
    ComputationTimeout,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DashboardState {
    pub phi_state: PhiState,
    pub ci_state: CIState,
    pub energy_state: EnergyState,
    pub performance_state: PerformanceState,
    pub last_update: Instant,
}

#[derive(Debug, Clone)]
pub struct PhiState {
    pub current_phi: f64,
    pub trend: PhiTrend,
    pub recent_emergence_events: usize,
    pub history_length: usize,
}

#[derive(Debug, Clone)]
pub struct CIState {
    pub current_ci: f64,
    pub fractal_dimension: f64,
    pub gain_factor: f64,
    pub coherence_level: f64,
    pub dwell_time: f64,
    pub history_length: usize,
}

#[derive(Debug, Clone)]
pub struct EnergyState {
    pub current_energy: f64,
    pub entropy: f64,
    pub negentropy: f64,
    pub total_landauer_violations: u32,
    pub heat_dissipation_rate: f64,
    pub history_length: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceState {
    pub current_fps: f64,
    pub gpu_utilization: f64,
    pub memory_usage: f64,
    pub computation_rate: f64,
    pub active_bottlenecks: Vec<BottleneckType>,
    pub history_length: usize,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    pub phi_series: Vec<PhiDataPoint>,
    pub ci_series: Vec<CIDataPoint>,
    pub energy_series: Vec<EnergyDataPoint>,
    pub performance_series: Vec<PerformanceDataPoint>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_rate_hz: 60.0, // 60 FPS monitoring
            history_duration_seconds: 300.0, // 5 minutes of history
            phi_alert_threshold: 1.0,
            energy_alert_threshold: 1000.0,
            performance_alert_threshold: 30.0, // 30 FPS minimum
            enable_emergence_detection: true,
            enable_real_time_alerts: true,
        }
    }
}
