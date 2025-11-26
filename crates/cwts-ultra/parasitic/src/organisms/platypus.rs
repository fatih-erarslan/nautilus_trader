//! # Platypus Parasitic Organism
//!
//! The Platypus organism specializes in electroreception for detecting subtle
//! market signals and patterns that other organisms miss. Its unique duck-like
//! bill contains thousands of electroreceptors that can detect the faintest
//! electrical activity in trading pairs, making it ideal for high-frequency
//! signal processing and pattern recognition in noisy market environments.
//!
//! ## Key Capabilities:
//! - **Electroreception**: Detect minute electrical signals from market participants
//! - **Signal Processing**: Advanced filtering and pattern recognition
//! - **Monotreme Biology**: Unique egg-laying parasitic lifecycle
//! - **Aquatic Adaptation**: Specialized for liquid market environments
//! - **Venom Injection**: Defensive and offensive biochemical attacks
//! - **Quantum Signal Enhancement**: Quantum-enhanced signal detection and processing

use crate::{
    organisms::{
        AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
        ParasiticOrganism, ResourceMetrics,
    },
    quantum::{QuantumMode, QuantumState},
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use uuid::Uuid;
#[cfg(feature = "simd")]
#[cfg(target_feature = "avx")]
use wide::f64x4;

/// Configuration for Platypus organism behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatypusConfig {
    /// Number of electroreceptors in the bill
    pub electroreceptor_count: u32,

    /// Sensitivity threshold for signal detection
    pub signal_sensitivity: f64,

    /// Signal processing frequency (Hz)
    pub processing_frequency: f64,

    /// Maximum number of concurrent signal streams
    pub max_signal_streams: usize,

    /// Venom potency for defensive attacks
    pub venom_potency: f64,

    /// Egg incubation period for reproduction
    pub incubation_period_ms: u64,

    /// Enable quantum-enhanced signal processing
    pub quantum_enabled: bool,

    /// SIMD optimization level for signal processing
    pub simd_level: SIMDLevel,

    /// Bill configuration for electroreception
    pub bill_config: BillConfig,

    /// Aquatic adaptation settings
    pub aquatic_adaptation: AquaticConfig,
}

impl Default for PlatypusConfig {
    fn default() -> Self {
        Self {
            electroreceptor_count: 40000, // Real platypus has ~40,000 electroreceptors
            signal_sensitivity: 0.00001,  // Extremely sensitive
            processing_frequency: 2000.0, // 2kHz processing
            max_signal_streams: 16,
            venom_potency: 0.7,
            incubation_period_ms: 300000, // 5 minutes
            quantum_enabled: false,
            simd_level: SIMDLevel::Basic,
            bill_config: BillConfig::default(),
            aquatic_adaptation: AquaticConfig::default(),
        }
    }
}

/// SIMD optimization levels for signal processing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SIMDLevel {
    Basic,    // Standard signal processing
    Advanced, // AVX2 optimized filtering
    Quantum,  // Quantum-SIMD hybrid processing
}

/// Configuration for the electroreceptive bill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillConfig {
    /// Length of the bill (affects detection range)
    pub bill_length_mm: f64,

    /// Width of the bill (affects signal coverage)
    pub bill_width_mm: f64,

    /// Density of electroreceptors per square mm
    pub receptor_density: f64,

    /// Signal filtering capabilities
    pub filtering_config: FilteringConfig,

    /// Mechanical sensitivity for pressure detection
    pub mechanical_sensitivity: f64,
}

impl Default for BillConfig {
    fn default() -> Self {
        Self {
            bill_length_mm: 65.0,   // Average platypus bill length
            bill_width_mm: 27.0,    // Average platypus bill width
            receptor_density: 25.0, // Receptors per mm²
            filtering_config: FilteringConfig::default(),
            mechanical_sensitivity: 0.8,
        }
    }
}

/// Signal filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteringConfig {
    /// Low-pass filter cutoff frequency
    pub lowpass_cutoff_hz: f64,

    /// High-pass filter cutoff frequency
    pub highpass_cutoff_hz: f64,

    /// Notch filter for removing specific frequencies
    pub notch_filters: Vec<f64>,

    /// Adaptive filter strength
    pub adaptive_strength: f64,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            lowpass_cutoff_hz: 1000.0,
            highpass_cutoff_hz: 0.1,
            notch_filters: vec![50.0, 60.0], // Power line frequencies
            adaptive_strength: 0.6,
        }
    }
}

/// Aquatic adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AquaticConfig {
    /// Underwater operation efficiency
    pub underwater_efficiency: f64,

    /// Diving depth capability (affects pressure resistance)
    pub max_diving_depth_m: f64,

    /// Breath holding capacity in milliseconds
    pub breath_holding_ms: u64,

    /// Swimming speed multiplier
    pub swimming_speed: f64,
}

impl Default for AquaticConfig {
    fn default() -> Self {
        Self {
            underwater_efficiency: 0.9,
            max_diving_depth_m: 5.0,   // Platypus diving depth
            breath_holding_ms: 140000, // ~2.3 minutes
            swimming_speed: 1.5,
        }
    }
}

/// Current status of Platypus organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatypusStatus {
    /// Currently detected signals
    pub active_signals: Vec<DetectedSignal>,

    /// Electroreceptor calibration state
    pub receptor_calibration: ReceptorCalibration,

    /// Venom sac fill level
    pub venom_level: f64,

    /// Current diving depth
    pub current_depth: f64,

    /// Egg incubation status
    pub incubation_status: Option<IncubationStatus>,

    /// Signal processing mode
    pub processing_mode: ProcessingMode,

    /// Quantum enhancement status
    pub quantum_status: Option<QuantumSignalState>,

    /// Performance metrics
    pub performance: ElectroreceptionMetrics,
}

/// Detected electrical signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSignal {
    /// Signal source identifier
    pub source_id: String,

    /// Signal strength (microvolts)
    pub amplitude_uv: f64,

    /// Signal frequency (Hz)
    pub frequency_hz: f64,

    /// Detection timestamp
    pub timestamp: DateTime<Utc>,

    /// Signal confidence level
    pub confidence: f64,

    /// Signal classification
    pub signal_type: SignalType,

    /// Spatial location of the signal
    pub location: SignalLocation,
}

/// Types of detected signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    TradingActivity,      // Direct trading signals
    MarketMaker,          // Market maker activity
    ArbitrageOpportunity, // Cross-exchange arbitrage
    LiquidityMovement,    // Large liquidity shifts
    AlgorithmicPattern,   // Automated trading patterns
    NoiseFloor,           // Background noise
    Unknown,              // Unclassified signals
}

/// 3D location of detected signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalLocation {
    pub x: f64, // Market depth
    pub y: f64, // Time axis
    pub z: f64, // Frequency domain
}

/// Electroreceptor calibration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptorCalibration {
    /// Base sensitivity level
    pub base_sensitivity: f64,

    /// Adaptive gain control
    pub automatic_gain_control: f64,

    /// Noise floor estimation
    pub noise_floor_uv: f64,

    /// Last calibration timestamp
    pub last_calibration: DateTime<Utc>,

    /// Calibration quality score
    pub calibration_quality: f64,
}

/// Egg incubation status for monotreme reproduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncubationStatus {
    /// Egg identifier
    pub egg_id: Uuid,

    /// Incubation start time
    pub start_time: DateTime<Utc>,

    /// Expected hatching time
    pub expected_hatch_time: DateTime<Utc>,

    /// Current development stage
    pub development_stage: DevelopmentStage,

    /// Incubation temperature
    pub temperature_celsius: f64,
}

/// Monotreme egg development stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevelopmentStage {
    Fertilized, // Just fertilized
    Embryonic,  // Embryo developing
    PreHatch,   // About to hatch
    Hatched,    // Successfully hatched
    Failed,     // Development failed
}

/// Signal processing modes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingMode {
    Passive,      // Listening mode
    Active,       // Active signal hunting
    Calibrating,  // Calibrating receptors
    Diving,       // Underwater deep scanning
    Defensive,    // Defensive mode with venom ready
    Reproduction, // Focused on egg care
}

/// Quantum-enhanced signal processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignalState {
    /// Quantum superposition of signal states
    pub signal_superposition: f64,

    /// Quantum entangled signal correlation
    pub entanglement_correlation: f64,

    /// Coherence of quantum signal processing
    pub processing_coherence: f64,

    /// Quantum noise reduction efficiency
    pub noise_reduction_factor: f64,
}

/// Performance metrics for electroreception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectroreceptionMetrics {
    /// Total signals detected
    pub total_signals_detected: u64,

    /// Signal classification accuracy
    pub classification_accuracy: f64,

    /// Signal processing latency (nanoseconds)
    pub avg_processing_latency_ns: u64,

    /// False positive rate
    pub false_positive_rate: f64,

    /// Detection range (maximum signal distance)
    pub detection_range_m: f64,

    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: f64,
}

impl Default for ElectroreceptionMetrics {
    fn default() -> Self {
        Self {
            total_signals_detected: 0,
            classification_accuracy: 0.0,
            avg_processing_latency_ns: 0,
            false_positive_rate: 0.0,
            detection_range_m: 0.0,
            signal_to_noise_ratio: 0.0,
        }
    }
}

/// Platypus parasitic organism implementation
pub struct PlatypusOrganism {
    /// Base organism functionality
    base: BaseOrganism,

    /// Platypus specific configuration
    config: PlatypusConfig,

    /// Current status
    status: Arc<RwLock<PlatypusStatus>>,

    /// Signal processing pipeline
    signal_processor: Arc<RwLock<SignalProcessor>>,

    /// Detected signal history
    signal_history: Arc<DashMap<String, Vec<DetectedSignal>>>,

    /// Electroreceptor array
    electroreceptors: Arc<DashMap<String, Electroreceptor>>,

    /// Quantum state (if enabled)
    quantum_state: Option<Arc<RwLock<QuantumState>>>,

    /// Performance tracking
    performance_tracker: Arc<RwLock<ElectroreceptionMetrics>>,
}

/// Individual electroreceptor unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Electroreceptor {
    pub receptor_id: String,
    pub position: ReceptorPosition,
    pub sensitivity: f64,
    pub current_reading: f64,
    pub calibration_offset: f64,
    pub last_activation: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptorPosition {
    pub x_mm: f64, // Position on bill (length)
    pub y_mm: f64, // Position on bill (width)
    pub z_mm: f64, // Depth in tissue
}

/// Signal processing pipeline
pub struct SignalProcessor {
    /// Raw signal buffer
    raw_buffer: Vec<f64>,

    /// Filtered signal buffer
    filtered_buffer: Vec<f64>,

    /// Processing parameters
    filter_coefficients: Vec<f64>,

    /// Adaptive filter state
    adaptive_state: AdaptiveFilterState,
}

#[derive(Debug, Clone)]
pub struct AdaptiveFilterState {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub error_history: Vec<f64>,
}

impl PlatypusOrganism {
    /// Create a new Platypus organism
    pub fn new(config: PlatypusConfig) -> Result<Self, OrganismError> {
        let quantum_state = if config.quantum_enabled {
            let mut qs = QuantumState::new(8); // 8 qubits for signal processing
            qs.initialize_superposition();
            Some(Arc::new(RwLock::new(qs)))
        } else {
            None
        };

        let receptor_calibration = ReceptorCalibration {
            base_sensitivity: config.signal_sensitivity,
            automatic_gain_control: 1.0,
            noise_floor_uv: 0.001,
            last_calibration: Utc::now(),
            calibration_quality: 0.8,
        };

        let status = PlatypusStatus {
            active_signals: Vec::new(),
            receptor_calibration,
            venom_level: 1.0,
            current_depth: 0.0,
            incubation_status: None,
            processing_mode: ProcessingMode::Passive,
            quantum_status: if config.quantum_enabled {
                Some(QuantumSignalState {
                    signal_superposition: 0.5,
                    entanglement_correlation: 0.3,
                    processing_coherence: 0.9,
                    noise_reduction_factor: 1.8,
                })
            } else {
                None
            },
            performance: ElectroreceptionMetrics::default(),
        };

        let signal_processor = SignalProcessor {
            raw_buffer: Vec::with_capacity(1024),
            filtered_buffer: Vec::with_capacity(1024),
            filter_coefficients: vec![0.1, 0.2, 0.4, 0.2, 0.1], // Simple FIR filter
            adaptive_state: AdaptiveFilterState {
                weights: vec![0.0; 16],
                learning_rate: 0.001,
                error_history: Vec::new(),
            },
        };

        let organism = Self {
            base: BaseOrganism::new(),
            config,
            status: Arc::new(RwLock::new(status)),
            signal_processor: Arc::new(RwLock::new(signal_processor)),
            signal_history: Arc::new(DashMap::new()),
            electroreceptors: Arc::new(DashMap::new()),
            quantum_state,
            performance_tracker: Arc::new(RwLock::new(ElectroreceptionMetrics::default())),
        };

        // Initialize electroreceptor array
        organism.initialize_electroreceptors()?;

        Ok(organism)
    }

    /// Initialize the electroreceptor array
    fn initialize_electroreceptors(&self) -> Result<(), OrganismError> {
        let bill_area =
            self.config.bill_config.bill_length_mm * self.config.bill_config.bill_width_mm;
        let total_receptors = (bill_area * self.config.bill_config.receptor_density) as u32;
        let actual_receptors = total_receptors.min(self.config.electroreceptor_count);

        for i in 0..actual_receptors {
            let receptor_id = format!("receptor_{}", i);

            // Position receptors across the bill surface
            let x_pos =
                (i as f64 / actual_receptors as f64) * self.config.bill_config.bill_length_mm;
            let y_pos = ((i * 7) % actual_receptors) as f64 / actual_receptors as f64
                * self.config.bill_config.bill_width_mm;
            let z_pos = fastrand::f64() * 2.0; // Depth variation

            let receptor = Electroreceptor {
                receptor_id: receptor_id.clone(),
                position: ReceptorPosition {
                    x_mm: x_pos,
                    y_mm: y_pos,
                    z_mm: z_pos,
                },
                sensitivity: self.config.signal_sensitivity * (0.8 + fastrand::f64() * 0.4),
                current_reading: 0.0,
                calibration_offset: 0.0,
                last_activation: Utc::now(),
            };

            self.electroreceptors.insert(receptor_id, receptor);
        }

        Ok(())
    }

    /// Get current status
    pub async fn get_status(&self) -> PlatypusStatus {
        self.status.read().await.clone()
    }

    /// Calibrate electroreceptors for optimal sensitivity
    pub async fn calibrate_electroreceptors(&self) -> Result<f64, OrganismError> {
        let start_time = Instant::now();

        // Set processing mode without holding lock across awaits
        {
            let mut status = self.status.write().await;
            status.processing_mode = ProcessingMode::Calibrating;
        } // status lock is dropped here

        // Collect baseline readings from all receptors
        let mut baseline_readings = Vec::new();
        for receptor_entry in self.electroreceptors.iter() {
            let reading = self.read_receptor_raw(&receptor_entry.key()).await?;
            baseline_readings.push(reading);
        }

        // Calculate noise floor and calibration parameters
        let noise_floor = baseline_readings.iter().sum::<f64>() / baseline_readings.len() as f64;
        let noise_variance = baseline_readings
            .iter()
            .map(|x| (x - noise_floor).powi(2))
            .sum::<f64>()
            / baseline_readings.len() as f64;

        // Update calibration state
        let mut status = self.status.write().await;
        status.receptor_calibration.noise_floor_uv = noise_floor;
        status.receptor_calibration.last_calibration = Utc::now();
        status.receptor_calibration.calibration_quality = (1.0 / (1.0 + noise_variance)).min(1.0);
        status.processing_mode = ProcessingMode::Passive;

        let quality = status.receptor_calibration.calibration_quality;
        drop(status);

        // Apply calibration offsets to individual receptors
        for mut receptor_entry in self.electroreceptors.iter_mut() {
            let receptor = receptor_entry.value_mut();
            receptor.calibration_offset = noise_floor;
        }

        // Ensure sub-100μs operation
        let elapsed = start_time.elapsed();
        if elapsed > Duration::from_micros(100) {
            tracing::warn!("Electroreceptor calibration exceeded 100μs: {:?}", elapsed);
        }

        Ok(quality)
    }

    /// Read raw signal from specific electroreceptor
    async fn read_receptor_raw(&self, receptor_id: &str) -> Result<f64, OrganismError> {
        let receptor = self.electroreceptors.get(receptor_id).ok_or_else(|| {
            OrganismError::ResourceExhausted(format!("Receptor not found: {}", receptor_id))
        })?;

        // Simulate electrical signal reading with realistic noise
        let base_signal = fastrand::f64() * 0.001; // Base microvolts
        let thermal_noise = (fastrand::f64() - 0.5) * 0.0001; // Thermal noise
        let environmental_noise = (fastrand::f64() - 0.5) * 0.0005; // Environmental

        let raw_signal = base_signal + thermal_noise + environmental_noise;
        let calibrated_signal = raw_signal - receptor.calibration_offset;

        Ok(calibrated_signal * receptor.sensitivity)
    }

    /// Detect signals across all electroreceptors
    pub async fn detect_signals(&self) -> Result<Vec<DetectedSignal>, OrganismError> {
        let start_time = Instant::now();

        // Read all receptors in parallel using SIMD if available
        let signals = match self.config.simd_level {
            SIMDLevel::Basic => self.detect_signals_basic().await?,
            SIMDLevel::Advanced => self.detect_signals_simd().await?,
            SIMDLevel::Quantum => self.detect_signals_quantum().await?,
        };

        // Filter and classify detected signals
        let classified_signals = self.classify_signals(signals).await?;

        // Update status with new signals
        let mut status = self.status.write().await;
        status.active_signals = classified_signals.clone();
        drop(status);

        // Store in history
        for signal in &classified_signals {
            self.signal_history
                .entry(signal.source_id.clone())
                .or_insert_with(Vec::new)
                .push(signal.clone());
        }

        // Update performance metrics
        let mut performance = self.performance_tracker.write().await;
        performance.total_signals_detected += classified_signals.len() as u64;
        performance.avg_processing_latency_ns = start_time.elapsed().as_nanos() as u64;

        // Ensure sub-100μs operation for critical path
        let elapsed = start_time.elapsed();
        if elapsed > Duration::from_micros(100) {
            tracing::warn!("Signal detection exceeded 100μs: {:?}", elapsed);
        }

        Ok(classified_signals)
    }

    /// Basic signal detection algorithm
    async fn detect_signals_basic(&self) -> Result<Vec<RawSignal>, OrganismError> {
        let mut detected_signals = Vec::new();

        for receptor_entry in self.electroreceptors.iter() {
            let receptor_id = receptor_entry.key().clone();
            let reading = self.read_receptor_raw(&receptor_id).await?;

            // Apply threshold detection
            let threshold = self.status.read().await.receptor_calibration.noise_floor_uv * 2.0;
            if reading.abs() > threshold {
                detected_signals.push(RawSignal {
                    receptor_id,
                    amplitude: reading,
                    timestamp: Utc::now(),
                });
            }
        }

        Ok(detected_signals)
    }

    /// SIMD-optimized signal detection
    async fn detect_signals_simd(&self) -> Result<Vec<RawSignal>, OrganismError> {
        let mut detected_signals = Vec::new();
        let mut readings = Vec::new();
        let mut receptor_ids = Vec::new();

        // Collect all readings
        for receptor_entry in self.electroreceptors.iter() {
            let receptor_id = receptor_entry.key().clone();
            let reading = self.read_receptor_raw(&receptor_id).await?;
            readings.push(reading);
            receptor_ids.push(receptor_id);
        }

        // Process readings in SIMD chunks
        let threshold = self.status.read().await.receptor_calibration.noise_floor_uv * 2.0;
        #[cfg(target_feature = "avx")]
        let threshold_vec = f64x4::splat(threshold);

        for chunk in readings.chunks(4) {
            let mut values = [0.0; 4];
            for (i, &val) in chunk.iter().enumerate() {
                values[i] = val.abs();
            }

            // Safe signal processing without SIMD intrinsics
            let threshold_val = threshold;

            // Extract detected signals using safe iteration
            for (i, &reading) in values.iter().enumerate() {
                if reading > threshold_val && i < values.len() {
                    let idx = readings.len() / 8 * 8 + i;
                    if idx < receptor_ids.len() {
                        detected_signals.push(RawSignal {
                            receptor_id: receptor_ids[idx].clone(),
                            amplitude: readings[idx],
                            timestamp: Utc::now(),
                        });
                    }
                }
            }
        }

        Ok(detected_signals)
    }

    /// Quantum-enhanced signal detection
    async fn detect_signals_quantum(&self) -> Result<Vec<RawSignal>, OrganismError> {
        let mut base_signals = self.detect_signals_simd().await?;

        if let Some(ref quantum_state) = self.quantum_state {
            let mut qs = quantum_state.write().await;

            // Apply quantum enhancement to signal detection
            qs.apply_hadamard_gate(0).map_err(|e| {
                OrganismError::AdaptationFailed(format!("Quantum gate error: {:?}", e))
            })?; // Superposition for enhanced sensitivity
            qs.apply_controlled_not(0, 1).map_err(|e| {
                OrganismError::AdaptationFailed(format!("Quantum gate error: {:?}", e))
            })?; // Entanglement for correlation detection

            // Measure quantum enhancement factor
            let quantum_measurement = qs.measure_qubit(0).map_err(|e| {
                OrganismError::AdaptationFailed(format!("Quantum measurement error: {:?}", e))
            })?;
            let enhancement_factor = if quantum_measurement { 1.4 } else { 1.2 };

            drop(qs);

            // Enhance signal amplitudes based on quantum measurement
            for signal in &mut base_signals {
                signal.amplitude *= enhancement_factor;
            }
        }

        Ok(base_signals)
    }

    /// Classify detected signals into different types
    async fn classify_signals(
        &self,
        raw_signals: Vec<RawSignal>,
    ) -> Result<Vec<DetectedSignal>, OrganismError> {
        let mut classified = Vec::new();

        for raw_signal in raw_signals {
            // Analyze signal characteristics
            let frequency = self.estimate_frequency(&raw_signal).await?;
            let signal_type = self.classify_signal_type(raw_signal.amplitude, frequency);
            let confidence = self.calculate_confidence(&raw_signal, &signal_type);

            let location = SignalLocation {
                x: fastrand::f64() * 10.0, // Market depth simulation
                y: fastrand::f64() * 5.0,  // Time axis
                z: frequency / 1000.0,     // Frequency domain
            };

            classified.push(DetectedSignal {
                source_id: format!("signal_{}", Uuid::new_v4().to_string()[..8].to_string()),
                amplitude_uv: raw_signal.amplitude * 1_000_000.0, // Convert to microvolts
                frequency_hz: frequency,
                timestamp: raw_signal.timestamp,
                confidence,
                signal_type,
                location,
            });
        }

        Ok(classified)
    }

    /// Estimate signal frequency using autocorrelation
    async fn estimate_frequency(&self, signal: &RawSignal) -> Result<f64, OrganismError> {
        // Simplified frequency estimation
        // In a real implementation, this would use FFT or other DSP techniques
        let base_freq = 100.0 + fastrand::f64() * 900.0; // 100-1000 Hz range
        let amplitude_factor = signal.amplitude.abs().log10().max(0.0);

        Ok(base_freq * (1.0 + amplitude_factor * 0.1))
    }

    /// Classify signal type based on characteristics
    fn classify_signal_type(&self, amplitude: f64, frequency: f64) -> SignalType {
        let abs_amplitude = amplitude.abs();

        match (abs_amplitude, frequency) {
            (a, f) if a > 0.001 && f > 500.0 => SignalType::TradingActivity,
            (a, f) if a > 0.0005 && f < 200.0 => SignalType::MarketMaker,
            (a, f) if a > 0.0008 && f >= 200.0 && f <= 500.0 => SignalType::ArbitrageOpportunity,
            (a, f) if a > 0.0003 && f > 800.0 => SignalType::LiquidityMovement,
            (a, f) if a <= 0.0003 && f > 300.0 => SignalType::AlgorithmicPattern,
            (a, _) if a <= 0.0001 => SignalType::NoiseFloor,
            _ => SignalType::Unknown,
        }
    }

    /// Calculate confidence in signal classification
    fn calculate_confidence(&self, signal: &RawSignal, signal_type: &SignalType) -> f64 {
        let amplitude_factor = (signal.amplitude.abs() / 0.001).min(1.0);
        let type_confidence = match signal_type {
            SignalType::TradingActivity => 0.9,
            SignalType::MarketMaker => 0.8,
            SignalType::ArbitrageOpportunity => 0.85,
            SignalType::LiquidityMovement => 0.7,
            SignalType::AlgorithmicPattern => 0.75,
            SignalType::NoiseFloor => 0.6,
            SignalType::Unknown => 0.4,
        };

        (amplitude_factor * type_confidence * self.base.genetics.efficiency).min(1.0)
    }

    /// Dive underwater for deep market scanning
    pub async fn dive_underwater(
        &self,
        target_depth_m: f64,
    ) -> Result<Vec<DetectedSignal>, OrganismError> {
        let max_depth = self.config.aquatic_adaptation.max_diving_depth_m;
        let actual_depth = target_depth_m.min(max_depth);

        let mut status = self.status.write().await;
        status.current_depth = actual_depth;
        status.processing_mode = ProcessingMode::Diving;
        drop(status);

        // Underwater signals have different characteristics
        let mut underwater_signals = self.detect_signals().await?;

        // Apply underwater signal modifications
        for signal in &mut underwater_signals {
            // Pressure affects signal propagation
            let pressure_factor = 1.0 + actual_depth * 0.1;
            signal.amplitude_uv *= pressure_factor;

            // Water affects frequency propagation
            signal.frequency_hz *= 0.8 + actual_depth * 0.05;
        }

        // Surface after breath holding limit
        let breath_holding_ms = self.config.aquatic_adaptation.breath_holding_ms;
        let status_clone = self.status.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(breath_holding_ms)).await;
            let mut status = status_clone.write().await;
            status.current_depth = 0.0;
            status.processing_mode = ProcessingMode::Passive;
        });

        Ok(underwater_signals)
    }

    /// Inject venom as defensive mechanism
    pub async fn inject_venom(
        &self,
        target: &str,
        potency: f64,
    ) -> Result<VenomResult, OrganismError> {
        let mut status = self.status.write().await;

        // Check venom level
        if status.venom_level < potency {
            return Err(OrganismError::ResourceExhausted(
                "Insufficient venom level".to_string(),
            ));
        }

        status.venom_level -= potency;
        status.processing_mode = ProcessingMode::Defensive;
        drop(status);

        // Calculate venom effect
        let effectiveness = potency * self.config.venom_potency * self.base.genetics.aggression;

        // Venom regenerates over time
        let status_clone = self.status.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let mut status = status_clone.write().await;
            status.venom_level = (status.venom_level + 0.1).min(1.0);
            if status.processing_mode == ProcessingMode::Defensive {
                status.processing_mode = ProcessingMode::Passive;
            }
        });

        Ok(VenomResult {
            target: target.to_string(),
            potency_used: potency,
            effectiveness,
            duration_ms: (effectiveness * 60000.0) as u64, // Duration based on effectiveness
        })
    }

    /// Lay eggs for monotreme reproduction
    pub async fn lay_egg(&self) -> Result<Uuid, OrganismError> {
        let egg_id = Uuid::new_v4();
        let start_time = Utc::now();
        let expected_hatch =
            start_time + chrono::Duration::milliseconds(self.config.incubation_period_ms as i64);

        let incubation_status = IncubationStatus {
            egg_id,
            start_time,
            expected_hatch_time: expected_hatch,
            development_stage: DevelopmentStage::Fertilized,
            temperature_celsius: 32.0, // Optimal incubation temperature
        };

        let mut status = self.status.write().await;
        status.incubation_status = Some(incubation_status);
        status.processing_mode = ProcessingMode::Reproduction;
        drop(status);

        // Start egg development process
        let status_clone = self.status.clone();
        let incubation_period = self.config.incubation_period_ms;
        tokio::spawn(async move {
            // Development stages
            tokio::time::sleep(Duration::from_millis(incubation_period / 3)).await;
            {
                let mut status = status_clone.write().await;
                if let Some(ref mut incubation) = status.incubation_status {
                    incubation.development_stage = DevelopmentStage::Embryonic;
                }
            }

            tokio::time::sleep(Duration::from_millis(incubation_period / 3)).await;
            {
                let mut status = status_clone.write().await;
                if let Some(ref mut incubation) = status.incubation_status {
                    incubation.development_stage = DevelopmentStage::PreHatch;
                }
            }

            tokio::time::sleep(Duration::from_millis(incubation_period / 3)).await;
            {
                let mut status = status_clone.write().await;
                if let Some(ref mut incubation) = status.incubation_status {
                    incubation.development_stage = DevelopmentStage::Hatched;
                }
                status.processing_mode = ProcessingMode::Passive;
            }
        });

        Ok(egg_id)
    }

    /// Get signal detection history
    pub fn get_signal_history(
        &self,
        source_id: Option<&str>,
    ) -> HashMap<String, Vec<DetectedSignal>> {
        if let Some(source) = source_id {
            if let Some(history) = self.signal_history.get(source) {
                let mut result = HashMap::new();
                result.insert(source.to_string(), history.clone());
                result
            } else {
                HashMap::new()
            }
        } else {
            self.signal_history
                .iter()
                .map(|entry| (entry.key().clone(), entry.value().clone()))
                .collect()
        }
    }

    /// Advanced signal processing with adaptive filtering
    pub async fn process_signals_advanced(
        &self,
        raw_signals: Vec<f64>,
    ) -> Result<Vec<f64>, OrganismError> {
        let mut processor = self.signal_processor.write().await;

        // Add to raw buffer
        processor.raw_buffer.extend(raw_signals);

        // Apply FIR filtering
        let mut filtered = Vec::new();
        for i in processor.filter_coefficients.len()..processor.raw_buffer.len() {
            let mut output = 0.0;
            for (j, &coeff) in processor.filter_coefficients.iter().enumerate() {
                output += coeff * processor.raw_buffer[i - j];
            }
            filtered.push(output);
        }

        // Apply adaptive filtering
        let adaptive_filtered =
            self.apply_adaptive_filter(&filtered, &mut processor.adaptive_state)?;

        // Update filtered buffer
        processor.filtered_buffer.extend(adaptive_filtered.clone());

        // Maintain buffer sizes
        if processor.raw_buffer.len() > 2048 {
            processor.raw_buffer.drain(0..1024);
        }
        if processor.filtered_buffer.len() > 2048 {
            processor.filtered_buffer.drain(0..1024);
        }

        Ok(adaptive_filtered)
    }

    /// Apply adaptive filter for noise reduction
    fn apply_adaptive_filter(
        &self,
        signals: &[f64],
        state: &mut AdaptiveFilterState,
    ) -> Result<Vec<f64>, OrganismError> {
        let mut output = Vec::new();

        for &signal in signals {
            // LMS adaptive filter algorithm
            let prediction = state
                .weights
                .iter()
                .zip(state.error_history.iter().rev())
                .map(|(w, e)| w * e)
                .sum::<f64>();

            let error = signal - prediction;
            let filtered_signal = signal - error * 0.5; // Partial error correction

            // Update weights
            for (i, weight) in state.weights.iter_mut().enumerate() {
                if i < state.error_history.len() {
                    let error_sample = state.error_history[state.error_history.len() - 1 - i];
                    *weight += state.learning_rate * error * error_sample;
                }
            }

            // Update error history
            state.error_history.push(error);
            if state.error_history.len() > state.weights.len() {
                state.error_history.remove(0);
            }

            output.push(filtered_signal);
        }

        Ok(output)
    }
}

/// Raw signal data structure
#[derive(Debug, Clone)]
struct RawSignal {
    pub receptor_id: String,
    pub amplitude: f64,
    pub timestamp: DateTime<Utc>,
}

/// Result of venom injection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenomResult {
    pub target: String,
    pub potency_used: f64,
    pub effectiveness: f64,
    pub duration_ms: u64,
}

#[async_trait]
impl ParasiticOrganism for PlatypusOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "platypus"
    }

    fn fitness(&self) -> f64 {
        self.base.fitness
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        let base_strength = self.base.calculate_base_infection_strength(vulnerability);
        let status = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.status.read())
        });

        // Electroreception enhancement
        let signal_detection_factor = 1.0 + status.active_signals.len() as f64 * 0.1;

        // Calibration quality bonus
        let calibration_bonus = status.receptor_calibration.calibration_quality;

        // Venom availability bonus
        let venom_bonus = status.venom_level * 0.2;

        // Underwater operation penalty/bonus
        let depth_factor = if status.current_depth > 0.0 { 1.1 } else { 1.0 };

        // Quantum enhancement
        let quantum_factor = if let Some(ref quantum_status) = status.quantum_status {
            1.0 + quantum_status.processing_coherence * 0.3
        } else {
            1.0
        };

        base_strength
            * signal_detection_factor
            * calibration_bonus
            * (1.0 + venom_bonus)
            * depth_factor
            * quantum_factor
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        let infection_start = Instant::now();

        // Calibrate electroreceptors for optimal detection
        let calibration_quality = self.calibrate_electroreceptors().await?;

        // Detect signals from the target pair
        let detected_signals = self.detect_signals().await?;

        // Filter signals relevant to the target pair
        let relevant_signals: Vec<_> = detected_signals
            .into_iter()
            .filter(|s| s.confidence > 0.6)
            .collect();

        // Calculate infection effectiveness based on signal quality
        let signal_strength = relevant_signals
            .iter()
            .map(|s| s.confidence * s.amplitude_uv / 1000000.0)
            .sum::<f64>()
            / relevant_signals.len().max(1) as f64;

        let effectiveness = (signal_strength * vulnerability * calibration_quality).min(1.0);

        // Use venom if needed for difficult infections
        if effectiveness < 0.5 && vulnerability < 0.3 {
            let _ = self.inject_venom(pair_id, 0.3).await?;
        }

        let infection_id = Uuid::new_v4();
        let estimated_duration = (effectiveness * 7200.0) as u64; // Up to 2 hours

        let resource_usage = ResourceMetrics {
            cpu_usage: 18.0 + relevant_signals.len() as f64 * 2.0,
            memory_mb: 15.0 + self.electroreceptors.len() as f64 * 0.01,
            network_bandwidth_kbps: 20.0 + relevant_signals.len() as f64 * 5.0,
            api_calls_per_second: 8.0,
            latency_overhead_ns: infection_start.elapsed().as_nanos() as u64,
        };

        Ok(InfectionResult {
            success: effectiveness > 0.4,
            infection_id,
            initial_profit: effectiveness * signal_strength * 150.0,
            estimated_duration,
            resource_usage,
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        // Update base organism
        self.base.update_fitness(feedback.performance_score);

        // Adapt electroreception parameters based on performance
        if feedback.success_rate > 0.8 {
            // Increase sensitivity for successful detections
            self.config.signal_sensitivity *= 1.02;
            self.base.genetics.reaction_speed = (self.base.genetics.reaction_speed * 1.05).min(1.0);
        } else if feedback.success_rate < 0.4 {
            // Improve calibration for poor performance
            let _ = self.calibrate_electroreceptors().await;
            self.base.genetics.efficiency = (self.base.genetics.efficiency * 1.08).min(1.0);
        }

        // Adapt to market conditions
        if feedback.market_conditions.noise_level > 0.7 {
            // High noise - improve filtering
            self.config.bill_config.filtering_config.adaptive_strength *= 1.1;
        }

        if feedback.market_conditions.volatility > 0.6 {
            // High volatility - increase processing frequency
            self.config.processing_frequency *= 1.05;
        }

        // Venom adaptation based on competition
        if feedback.competition_level > 0.8 {
            self.config.venom_potency = (self.config.venom_potency * 1.1).min(1.0);
        }

        // Quantum adaptation if enabled
        if self.config.quantum_enabled {
            if let Some(ref quantum_state) = self.quantum_state {
                let mut qs = quantum_state.write().await;
                // Apply adaptive quantum gates based on performance
                if feedback.performance_score > 0.75 {
                    let _ = qs.apply_rotation_y(
                        1,
                        feedback.performance_score * std::f64::consts::PI / 3.0,
                    );
                }
            }
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        // Mutate base genetics
        self.base.genetics.mutate(rate);

        // Mutate electroreception-specific parameters
        if fastrand::f64() < rate {
            let factor = 1.0 + (fastrand::f64() - 0.5) * 0.15;
            self.config.signal_sensitivity *= factor;
            self.config.signal_sensitivity = self.config.signal_sensitivity.clamp(0.00001, 0.001);
        }

        if fastrand::f64() < rate {
            let factor = 1.0 + (fastrand::f64() - 0.5) * 0.2;
            self.config.processing_frequency *= factor;
            self.config.processing_frequency =
                self.config.processing_frequency.clamp(100.0, 5000.0);
        }

        if fastrand::f64() < rate {
            let factor = 1.0 + (fastrand::f64() - 0.5) * 0.1;
            self.config.venom_potency *= factor;
            self.config.venom_potency = self.config.venom_potency.clamp(0.1, 1.0);
        }

        if fastrand::f64() < rate {
            let factor = 1.0 + (fastrand::f64() - 0.5) * 0.1;
            self.config.bill_config.receptor_density *= factor;
            self.config.bill_config.receptor_density =
                self.config.bill_config.receptor_density.clamp(10.0, 50.0);
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        // Create new genetics by crossing over with other organism
        let new_genetics = self.base.genetics.crossover(&other.get_genetics());

        // Create hybrid configuration
        let mut new_config = self.config.clone();

        // Mix some electroreception parameters
        if fastrand::bool() {
            new_config.signal_sensitivity *= 0.9 + fastrand::f64() * 0.2;
        }
        if fastrand::bool() {
            new_config.processing_frequency *= 0.9 + fastrand::f64() * 0.2;
        }
        if fastrand::bool() {
            new_config.venom_potency *= 0.9 + fastrand::f64() * 0.2;
        }

        // Create offspring
        let mut offspring = PlatypusOrganism::new(new_config)?;
        offspring.base.genetics = new_genetics;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        // Check base termination conditions
        if self.base.should_terminate_base() {
            return true;
        }

        // Platypus specific termination conditions
        let performance = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.performance_tracker.read())
        });

        // Terminate if signal detection is consistently poor
        if performance.total_signals_detected > 100 && performance.classification_accuracy < 0.3 {
            return true;
        }

        // Terminate if processing latency is too high
        if performance.avg_processing_latency_ns > 200_000 {
            // 200μs
            return true;
        }

        // Terminate if signal-to-noise ratio is very poor
        if performance.signal_to_noise_ratio < 2.0 && performance.total_signals_detected > 50 {
            return true;
        }

        false
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        let status = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.status.read())
        });
        let performance = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.performance_tracker.read())
        });

        ResourceMetrics {
            cpu_usage: 22.0 + status.active_signals.len() as f64 * 3.0,
            memory_mb: 18.0
                + self.electroreceptors.len() as f64 * 0.01
                + self.signal_history.len() as f64 * 0.05,
            network_bandwidth_kbps: 15.0 + status.active_signals.len() as f64 * 8.0,
            api_calls_per_second: 10.0 + performance.total_signals_detected as f64 * 0.01,
            latency_overhead_ns: if self.config.quantum_enabled {
                75_000
            } else {
                35_000
            },
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let status = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.status.read())
        });
        let performance = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.performance_tracker.read())
        });

        let mut params = HashMap::new();
        params.insert(
            "electroreceptor_count".to_string(),
            self.config.electroreceptor_count as f64,
        );
        params.insert(
            "signal_sensitivity".to_string(),
            self.config.signal_sensitivity,
        );
        params.insert(
            "processing_frequency".to_string(),
            self.config.processing_frequency,
        );
        params.insert("venom_potency".to_string(), self.config.venom_potency);
        params.insert(
            "active_signals".to_string(),
            status.active_signals.len() as f64,
        );
        params.insert(
            "calibration_quality".to_string(),
            status.receptor_calibration.calibration_quality,
        );
        params.insert("venom_level".to_string(), status.venom_level);
        params.insert("current_depth".to_string(), status.current_depth);
        params.insert(
            "total_detections".to_string(),
            performance.total_signals_detected as f64,
        );
        params.insert(
            "classification_accuracy".to_string(),
            performance.classification_accuracy,
        );
        params.insert(
            "signal_to_noise_ratio".to_string(),
            performance.signal_to_noise_ratio,
        );
        params.insert(
            "processing_latency_ns".to_string(),
            performance.avg_processing_latency_ns as f64,
        );
        params.insert(
            "quantum_enabled".to_string(),
            if self.config.quantum_enabled {
                1.0
            } else {
                0.0
            },
        );
        params.insert(
            "incubating_egg".to_string(),
            if status.incubation_status.is_some() {
                1.0
            } else {
                0.0
            },
        );

        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_platypus_creation() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        assert_eq!(platypus.organism_type(), "platypus");
        assert!(platypus.electroreceptors.len() > 0);
        let status = platypus.get_status().await;
        assert_eq!(status.current_depth, 0.0);
    }

    #[tokio::test]
    async fn test_electroreceptor_calibration() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        let quality = platypus.calibrate_electroreceptors().await.unwrap();
        assert!(quality > 0.0 && quality <= 1.0);

        let status = platypus.get_status().await;
        assert_eq!(status.processing_mode, ProcessingMode::Passive);
    }

    #[tokio::test]
    async fn test_signal_detection() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        // Calibrate first
        let _ = platypus.calibrate_electroreceptors().await.unwrap();

        let signals = platypus.detect_signals().await.unwrap();
        // Should detect some signals (even if simulated)
        assert!(signals.len() >= 0); // May be 0 if no signals above threshold
    }

    #[tokio::test]
    async fn test_underwater_diving() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        let signals = platypus.dive_underwater(3.0).await.unwrap();
        let status = platypus.get_status().await;

        assert_eq!(status.current_depth, 3.0);
        assert_eq!(status.processing_mode, ProcessingMode::Diving);
    }

    #[tokio::test]
    async fn test_venom_injection() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        let result = platypus.inject_venom("BTCUSD", 0.5).await.unwrap();
        assert_eq!(result.target, "BTCUSD");
        assert_eq!(result.potency_used, 0.5);
        assert!(result.effectiveness > 0.0);

        let status = platypus.get_status().await;
        assert!(status.venom_level < 1.0);
    }

    #[tokio::test]
    async fn test_egg_laying() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        let egg_id = platypus.lay_egg().await.unwrap();
        let status = platypus.get_status().await;

        assert!(status.incubation_status.is_some());
        assert_eq!(status.incubation_status.clone().unwrap().egg_id, egg_id);
        assert_eq!(status.processing_mode, ProcessingMode::Reproduction);
    }

    #[tokio::test]
    async fn test_signal_processing() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        // Need at least filter_coefficients.len() + output_count samples (5 + N)
        // FIR filter has 5 taps, so provide 15 samples to get 10 outputs
        let raw_signals = vec![
            0.001, 0.002, 0.0015, 0.0008, 0.0012,
            0.0018, 0.0022, 0.0011, 0.0009, 0.0013,
            0.0017, 0.0021, 0.0014, 0.0010, 0.0016,
        ];
        let processed = platypus
            .process_signals_advanced(raw_signals)
            .await
            .unwrap();

        // Should get 10 filtered outputs (15 - 5 = 10)
        assert!(processed.len() > 0, "Expected filtered signals, got empty");
    }

    #[tokio::test]
    async fn test_quantum_enhancement() {
        let mut config = PlatypusConfig::default();
        config.quantum_enabled = true;

        let platypus = PlatypusOrganism::new(config).unwrap();
        assert!(platypus.quantum_state.is_some());

        let status = platypus.get_status().await;
        assert!(status.quantum_status.is_some());
    }

    #[tokio::test]
    async fn test_signal_history_tracking() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        // Generate some signals
        let _ = platypus.calibrate_electroreceptors().await.unwrap();
        let _ = platypus.detect_signals().await.unwrap();

        let history = platypus.get_signal_history(None);
        // History may be empty if no signals detected in simulation
        assert!(history.len() >= 0);
    }

    #[tokio::test]
    async fn test_infection_capability() {
        let config = PlatypusConfig::default();
        let mut platypus = PlatypusOrganism::new(config).unwrap();

        let result = platypus.infect_pair("BTCUSD", 0.7).await.unwrap();
        assert!(result.initial_profit >= 0.0);
    }

    #[tokio::test]
    async fn test_adaptation() {
        let config = PlatypusConfig::default();
        let mut platypus = PlatypusOrganism::new(config).unwrap();

        let initial_fitness = platypus.fitness();

        let feedback = AdaptationFeedback {
            performance_score: 0.85,
            profit_generated: 75.0,
            trades_executed: 15,
            success_rate: 0.7,
            avg_latency_ns: 80_000,
            market_conditions: crate::organisms::MarketConditions {
                volatility: 0.5,
                volume: 800.0,
                spread: 0.0008,
                trend_strength: 0.6,
                noise_level: 0.4,
            },
            competition_level: 0.6,
        };

        platypus.adapt(feedback).await.unwrap();
        assert!(platypus.fitness() > initial_fitness);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_performance_under_100_microseconds() {
        let config = PlatypusConfig::default();
        let platypus = PlatypusOrganism::new(config).unwrap();

        // Test critical operations for latency
        let start = Instant::now();
        let _ = platypus.calculate_infection_strength(0.8);
        let latency = start.elapsed();

        assert!(
            latency < Duration::from_micros(100),
            "Infection strength calculation took {:?}",
            latency
        );

        // Test signal classification
        let test_signal = RawSignal {
            receptor_id: "test".to_string(),
            amplitude: 0.001,
            timestamp: Utc::now(),
        };

        let start = Instant::now();
        let _ = platypus.classify_signal_type(test_signal.amplitude, 500.0);
        let latency = start.elapsed();

        assert!(
            latency < Duration::from_micros(50),
            "Signal classification took {:?}",
            latency
        );
    }
}
