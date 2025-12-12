//! # Quantum Feature Extraction
//!
//! This module implements superposition-based feature extraction for quantum-enhanced
//! uncertainty quantification in trading systems.

use std::collections::HashMap;
use std::f64::consts::PI;

use anyhow::anyhow;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    QuantumState, QuantumConfig, QuantumGate, QuantumCircuit, QuantumCircuitSimulator, Result,
};

/// Quantum coherence measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceMeasures {
    /// Relative entropy of coherence
    pub relative_entropy: f64,
    /// L1 norm of coherence
    pub l1_norm: f64,
    /// Robustness of coherence
    pub robustness: f64,
    /// Formation coherence
    pub formation: f64,
}

impl Default for QuantumCoherenceMeasures {
    fn default() -> Self {
        Self {
            relative_entropy: 0.0,
            l1_norm: 0.0,
            robustness: 0.0,
            formation: 0.0,
        }
    }
}

/// Quantum features extracted from classical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatures {
    /// Original classical features
    pub classical_features: Vec<f64>,
    /// Quantum superposition features
    pub superposition_features: Vec<Complex64>,
    /// Quantum entanglement features
    pub entanglement_features: Vec<f64>,
    /// Quantum interference features
    pub interference_features: Vec<f64>,
    /// Quantum phase features
    pub phase_features: Vec<f64>,
    /// Quantum amplitude features
    pub amplitude_features: Vec<f64>,
    /// Quantum coherence features
    pub coherence_features: Vec<f64>,
    /// Feature importance scores
    pub feature_importance: Vec<f64>,
    /// Quantum fidelity measures
    pub fidelity_measures: Vec<f64>,
    /// Feature extraction metadata
    pub metadata: QuantumFeatureMetadata,
    /// Superposition coherence measure
    pub superposition_coherence: f64,
    /// Entanglement strength measure
    pub entanglement_strength: f64,
    /// Feature dimension
    pub feature_dimension: usize,
    /// Quantum information content
    pub quantum_information_content: f64,
    /// Coherence measures
    pub coherence_measures: QuantumCoherenceMeasures,
}

impl QuantumFeatures {
    /// Create new quantum features
    pub fn new(classical_features: Vec<f64>) -> Self {
        let n_features = classical_features.len();
        
        Self {
            classical_features,
            superposition_features: vec![Complex64::new(0.0, 0.0); n_features],
            entanglement_features: vec![0.0; n_features],
            interference_features: vec![0.0; n_features],
            phase_features: vec![0.0; n_features],
            amplitude_features: vec![0.0; n_features],
            coherence_features: vec![0.0; n_features],
            feature_importance: vec![0.0; n_features],
            fidelity_measures: vec![0.0; n_features],
            metadata: QuantumFeatureMetadata::new(),
            superposition_coherence: 0.0,
            entanglement_strength: 0.0,
            feature_dimension: n_features,
            quantum_information_content: 0.0,
            coherence_measures: QuantumCoherenceMeasures::default(),
        }
    }

    /// Get total number of features
    pub fn total_features(&self) -> usize {
        self.classical_features.len() + 
        self.superposition_features.len() + 
        self.entanglement_features.len() + 
        self.interference_features.len() + 
        self.phase_features.len() + 
        self.amplitude_features.len() + 
        self.coherence_features.len()
    }

    /// Get quantum feature vector
    pub fn quantum_feature_vector(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Add classical features
        features.extend_from_slice(&self.classical_features);
        
        // Add superposition features (real and imaginary parts)
        for complex_feature in &self.superposition_features {
            features.push(complex_feature.re);
            features.push(complex_feature.im);
        }
        
        // Add other quantum features
        features.extend_from_slice(&self.entanglement_features);
        features.extend_from_slice(&self.interference_features);
        features.extend_from_slice(&self.phase_features);
        features.extend_from_slice(&self.amplitude_features);
        features.extend_from_slice(&self.coherence_features);
        
        features
    }

    /// Get most important features
    pub fn most_important_features(&self, n_features: usize) -> Vec<(usize, f64)> {
        let mut indexed_importance: Vec<(usize, f64)> = self.feature_importance
            .iter()
            .enumerate()
            .map(|(i, &importance)| (i, importance))
            .collect();
        
        indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_importance.truncate(n_features);
        indexed_importance
    }

    /// Calculate feature statistics
    pub fn calculate_statistics(&self) -> QuantumFeatureStatistics {
        QuantumFeatureStatistics {
            classical_mean: self.classical_features.iter().sum::<f64>() / self.classical_features.len() as f64,
            classical_std: self.calculate_std(&self.classical_features),
            superposition_magnitude: self.superposition_features.iter().map(|c| c.norm()).sum::<f64>() / self.superposition_features.len() as f64,
            entanglement_mean: self.entanglement_features.iter().sum::<f64>() / self.entanglement_features.len() as f64,
            interference_mean: self.interference_features.iter().sum::<f64>() / self.interference_features.len() as f64,
            phase_coherence: self.calculate_phase_coherence(),
            amplitude_variance: self.calculate_std(&self.amplitude_features),
            coherence_persistence: self.coherence_features.iter().sum::<f64>() / self.coherence_features.len() as f64,
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, features: &[f64]) -> f64 {
        let mean = features.iter().sum::<f64>() / features.len() as f64;
        let variance = features.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / features.len() as f64;
        variance.sqrt()
    }

    /// Calculate phase coherence
    fn calculate_phase_coherence(&self) -> f64 {
        let phase_sum: Complex64 = self.superposition_features.iter()
            .map(|c| c / c.norm())
            .sum();
        
        phase_sum.norm() / self.superposition_features.len() as f64
    }
}

/// Quantum feature extraction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatureMetadata {
    /// Extraction timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Number of qubits used
    pub n_qubits: usize,
    /// Extraction method
    pub extraction_method: String,
    /// Quantum circuit depth
    pub circuit_depth: usize,
    /// Fidelity of extraction
    pub extraction_fidelity: f64,
    /// Quantum advantage measure
    pub quantum_advantage: f64,
    /// Feature encoding type
    pub encoding_type: String,
    /// Number of measurements
    pub n_measurements: usize,
}

impl QuantumFeatureMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            n_qubits: 0,
            extraction_method: "superposition".to_string(),
            circuit_depth: 0,
            extraction_fidelity: 0.0,
            quantum_advantage: 0.0,
            encoding_type: "amplitude".to_string(),
            n_measurements: 0,
        }
    }
}

/// Quantum feature extractor engine
#[derive(Debug)]
pub struct QuantumFeatureExtractor {
    /// Configuration
    pub config: QuantumConfig,
    /// Quantum circuit simulator
    pub simulator: QuantumCircuitSimulator,
    /// Feature extraction circuits
    pub extraction_circuits: Vec<QuantumCircuit>,
    /// Superposition generators
    pub superposition_generators: Vec<SuperpositionGenerator>,
    /// Entanglement analyzers
    pub entanglement_analyzers: Vec<EntanglementAnalyzer>,
    /// Interference detectors
    pub interference_detectors: Vec<InterferenceDetector>,
    /// Phase extractors
    pub phase_extractors: Vec<PhaseExtractor>,
    /// Amplitude analyzers
    pub amplitude_analyzers: Vec<AmplitudeAnalyzer>,
    /// Coherence meters
    pub coherence_meters: Vec<CoherenceMeter>,
    /// Feature importance calculator
    pub importance_calculator: FeatureImportanceCalculator,
}

impl QuantumFeatureExtractor {
    /// Create new quantum feature extractor
    pub fn new(config: QuantumConfig) -> Result<Self> {
        let simulator = QuantumCircuitSimulator::new(config.n_qubits)?;
        
        let extraction_circuits = Self::create_extraction_circuits(&config)?;
        let superposition_generators = Self::create_superposition_generators(&config)?;
        let entanglement_analyzers = Self::create_entanglement_analyzers(&config)?;
        let interference_detectors = Self::create_interference_detectors(&config)?;
        let phase_extractors = Self::create_phase_extractors(&config)?;
        let amplitude_analyzers = Self::create_amplitude_analyzers(&config)?;
        let coherence_meters = Self::create_coherence_meters(&config)?;
        let importance_calculator = FeatureImportanceCalculator::new(config.clone())?;

        Ok(Self {
            config,
            simulator,
            extraction_circuits,
            superposition_generators,
            entanglement_analyzers,
            interference_detectors,
            phase_extractors,
            amplitude_analyzers,
            coherence_meters,
            importance_calculator,
        })
    }

    /// Extract quantum features from classical data
    pub async fn extract_features(&self, data: &Array2<f64>) -> Result<QuantumFeatures> {
        info!("Extracting quantum features from classical data");
        
        let mut features = QuantumFeatures::new(data.column(0).to_vec());
        
        // Extract superposition features
        for generator in &self.superposition_generators {
            let superposition_features = generator.generate_superposition_features(data).await?;
            features.superposition_features = superposition_features;
        }
        
        // Extract entanglement features
        for analyzer in &self.entanglement_analyzers {
            let entanglement_features = analyzer.analyze_entanglement(data).await?;
            features.entanglement_features = entanglement_features;
        }
        
        // Calculate feature importance
        features.feature_importance = self.importance_calculator.calculate_importance(&features).await?;
        
        Ok(features)
    }

    /// Reset the feature extractor
    pub fn reset(&mut self) -> Result<()> {
        self.simulator.reset()?;
        for generator in &mut self.superposition_generators {
            generator.reset()?;
        }
        Ok(())
    }

    /// Create extraction circuits
    fn create_extraction_circuits(config: &QuantumConfig) -> Result<Vec<QuantumCircuit>> {
        let mut circuits = Vec::new();
        
        // Create hardware-efficient ansatz
        let hw_efficient = QuantumCircuit::hardware_efficient_ansatz(
            config.n_qubits,
            config.n_layers,
            "hardware_efficient".to_string(),
        );
        circuits.push(hw_efficient);
        
        // Create QAOA ansatz
        let qaoa = QuantumCircuit::qaoa_ansatz(
            config.n_qubits,
            config.n_layers,
            "qaoa".to_string(),
        );
        circuits.push(qaoa);
        
        Ok(circuits)
    }

    /// Create superposition generators
    fn create_superposition_generators(config: &QuantumConfig) -> Result<Vec<SuperpositionGenerator>> {
        let mut generators = Vec::new();
        
        // Uniform superposition generator
        let uniform_generator = SuperpositionGenerator::new(
            "uniform".to_string(),
            SuperpositionType::Uniform,
            config.n_qubits,
        )?;
        generators.push(uniform_generator);
        
        // Weighted superposition generator
        let weighted_generator = SuperpositionGenerator::new(
            "weighted".to_string(),
            SuperpositionType::Weighted,
            config.n_qubits,
        )?;
        generators.push(weighted_generator);
        
        Ok(generators)
    }

    /// Create entanglement analyzers
    fn create_entanglement_analyzers(config: &QuantumConfig) -> Result<Vec<EntanglementAnalyzer>> {
        let mut analyzers = Vec::new();
        
        // Bipartite entanglement analyzer
        let bipartite_analyzer = EntanglementAnalyzer::new(
            "bipartite".to_string(),
            EntanglementType::Bipartite,
            config.n_qubits,
        )?;
        analyzers.push(bipartite_analyzer);
        
        // Multipartite entanglement analyzer
        let multipartite_analyzer = EntanglementAnalyzer::new(
            "multipartite".to_string(),
            EntanglementType::Multipartite,
            config.n_qubits,
        )?;
        analyzers.push(multipartite_analyzer);
        
        Ok(analyzers)
    }

    /// Create interference detectors
    fn create_interference_detectors(config: &QuantumConfig) -> Result<Vec<InterferenceDetector>> {
        let mut detectors = Vec::new();
        
        let amplitude_detector = InterferenceDetector::new(
            "amplitude".to_string(),
            InterferenceType::Amplitude,
            config.n_qubits,
        )?;
        detectors.push(amplitude_detector);
        
        let phase_detector = InterferenceDetector::new(
            "phase".to_string(),
            InterferenceType::Phase,
            config.n_qubits,
        )?;
        detectors.push(phase_detector);
        
        Ok(detectors)
    }

    /// Create phase extractors
    fn create_phase_extractors(config: &QuantumConfig) -> Result<Vec<PhaseExtractor>> {
        let mut extractors = Vec::new();
        
        let qft_extractor = PhaseExtractor::new(
            "qft".to_string(),
            PhaseExtractionType::QuantumFourierTransform,
            config.n_qubits,
        )?;
        extractors.push(qft_extractor);
        
        let ramsey_extractor = PhaseExtractor::new(
            "ramsey".to_string(),
            PhaseExtractionType::Ramsey,
            config.n_qubits,
        )?;
        extractors.push(ramsey_extractor);
        
        Ok(extractors)
    }

    /// Create amplitude analyzers
    fn create_amplitude_analyzers(config: &QuantumConfig) -> Result<Vec<AmplitudeAnalyzer>> {
        let mut analyzers = Vec::new();
        
        let direct_analyzer = AmplitudeAnalyzer::new(
            "direct".to_string(),
            AmplitudeAnalysisType::Direct,
            config.n_qubits,
        )?;
        analyzers.push(direct_analyzer);
        
        let tomographic_analyzer = AmplitudeAnalyzer::new(
            "tomographic".to_string(),
            AmplitudeAnalysisType::Tomographic,
            config.n_qubits,
        )?;
        analyzers.push(tomographic_analyzer);
        
        Ok(analyzers)
    }

    /// Create coherence meters
    fn create_coherence_meters(config: &QuantumConfig) -> Result<Vec<CoherenceMeter>> {
        let mut meters = Vec::new();
        
        let l1_meter = CoherenceMeter::new(
            "l1_norm".to_string(),
            CoherenceType::L1Norm,
            config.n_qubits,
        )?;
        meters.push(l1_meter);
        
        let entropy_meter = CoherenceMeter::new(
            "relative_entropy".to_string(),
            CoherenceType::RelativeEntropy,
            config.n_qubits,
        )?;
        meters.push(entropy_meter);
        
        Ok(meters)
    }
}

/// Quantum feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFeatureStatistics {
    /// Mean of classical features
    pub classical_mean: f64,
    /// Standard deviation of classical features
    pub classical_std: f64,
    /// Average magnitude of superposition features
    pub superposition_magnitude: f64,
    /// Mean entanglement feature
    pub entanglement_mean: f64,
    /// Mean interference feature
    pub interference_mean: f64,
    /// Phase coherence measure
    pub phase_coherence: f64,
    /// Amplitude variance
    pub amplitude_variance: f64,
    /// Coherence persistence
    pub coherence_persistence: f64,
}

/// Superposition generator for quantum feature extraction
#[derive(Debug, Clone)]
pub struct SuperpositionGenerator {
    /// Generator name
    pub name: String,
    /// Superposition type
    pub superposition_type: SuperpositionType,
    /// Number of qubits
    pub n_qubits: usize,
    /// Generator parameters
    pub parameters: Vec<f64>,
    /// Generation statistics
    pub stats: GenerationStats,
}

impl SuperpositionGenerator {
    /// Create new superposition generator
    pub fn new(name: String, superposition_type: SuperpositionType, n_qubits: usize) -> Result<Self> {
        let parameters = vec![0.0; n_qubits * 2]; // RY and RZ angles
        
        Ok(Self {
            name,
            superposition_type,
            n_qubits,
            parameters,
            stats: GenerationStats::new(),
        })
    }

    /// Generate superposition features
    pub async fn generate_superposition_features(&self, data: &Array2<f64>) -> Result<Vec<Complex64>> {
        let mut features = Vec::new();
        
        match self.superposition_type {
            SuperpositionType::Uniform => {
                features = self.generate_uniform_superposition(data).await?;
            }
            SuperpositionType::Weighted => {
                features = self.generate_weighted_superposition(data).await?;
            }
            SuperpositionType::Adaptive => {
                features = self.generate_adaptive_superposition(data).await?;
            }
        }
        
        Ok(features)
    }

    /// Generate uniform superposition
    async fn generate_uniform_superposition(&self, data: &Array2<f64>) -> Result<Vec<Complex64>> {
        let n_features = data.ncols();
        let n_states = 2_usize.pow(self.n_qubits as u32);
        let amplitude = Complex64::new(1.0 / (n_states as f64).sqrt(), 0.0);
        
        Ok(vec![amplitude; n_features])
    }

    /// Generate weighted superposition
    async fn generate_weighted_superposition(&self, data: &Array2<f64>) -> Result<Vec<Complex64>> {
        let n_features = data.ncols();
        let mut features = Vec::new();
        
        for i in 0..n_features {
            let weight = data.column(i).iter().sum::<f64>() / data.nrows() as f64;
            let amplitude = Complex64::new(weight, 0.0);
            features.push(amplitude);
        }
        
        // Normalize
        let norm_sq: f64 = features.iter().map(|amp| amp.norm_sqr()).sum();
        let norm = norm_sq.sqrt();
        
        if norm > 0.0 {
            for amp in &mut features {
                *amp /= norm;
            }
        }
        
        Ok(features)
    }

    /// Generate adaptive superposition
    async fn generate_adaptive_superposition(&self, data: &Array2<f64>) -> Result<Vec<Complex64>> {
        let n_features = data.ncols();
        let mut features = Vec::new();
        
        for i in 0..n_features {
            let column = data.column(i);
            let mean = column.iter().sum::<f64>() / column.len() as f64;
            let variance = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
            
            let real_part = mean;
            let imag_part = variance.sqrt();
            let amplitude = Complex64::new(real_part, imag_part);
            features.push(amplitude);
        }
        
        // Normalize
        let norm_sq: f64 = features.iter().map(|amp| amp.norm_sqr()).sum();
        let norm = norm_sq.sqrt();
        
        if norm > 0.0 {
            for amp in &mut features {
                *amp /= norm;
            }
        }
        
        Ok(features)
    }

    /// Reset generator
    pub fn reset(&mut self) -> Result<()> {
        self.parameters.fill(0.0);
        self.stats.reset();
        Ok(())
    }
}

/// Superposition types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperpositionType {
    /// Uniform superposition
    Uniform,
    /// Weighted superposition
    Weighted,
    /// Adaptive superposition
    Adaptive,
}

/// Entanglement analyzer
#[derive(Debug, Clone)]
pub struct EntanglementAnalyzer {
    /// Analyzer name
    pub name: String,
    /// Entanglement type
    pub entanglement_type: EntanglementType,
    /// Number of qubits
    pub n_qubits: usize,
    /// Analysis parameters
    pub parameters: Vec<f64>,
    /// Analysis statistics
    pub stats: AnalysisStats,
}

impl EntanglementAnalyzer {
    /// Create new entanglement analyzer
    pub fn new(name: String, entanglement_type: EntanglementType, n_qubits: usize) -> Result<Self> {
        Ok(Self {
            name,
            entanglement_type,
            n_qubits,
            parameters: Vec::new(),
            stats: AnalysisStats::new(),
        })
    }

    /// Analyze entanglement in data
    pub async fn analyze_entanglement(&self, data: &Array2<f64>) -> Result<Vec<f64>> {
        match self.entanglement_type {
            EntanglementType::Bipartite => self.analyze_bipartite_entanglement(data).await,
            EntanglementType::Multipartite => self.analyze_multipartite_entanglement(data).await,
        }
    }

    /// Analyze bipartite entanglement
    async fn analyze_bipartite_entanglement(&self, data: &Array2<f64>) -> Result<Vec<f64>> {
        let n_features = data.ncols();
        let mut entanglement_features = Vec::new();
        
        for i in 0..n_features {
            let column = data.column(i);
            let entanglement = self.calculate_bipartite_entanglement(&column.to_vec()).await?;
            entanglement_features.push(entanglement);
        }
        
        Ok(entanglement_features)
    }

    /// Analyze multipartite entanglement
    async fn analyze_multipartite_entanglement(&self, data: &Array2<f64>) -> Result<Vec<f64>> {
        let n_features = data.ncols();
        let mut entanglement_features = Vec::new();
        
        for i in 0..n_features {
            let column = data.column(i);
            let entanglement = self.calculate_multipartite_entanglement(&column.to_vec()).await?;
            entanglement_features.push(entanglement);
        }
        
        Ok(entanglement_features)
    }

    /// Calculate bipartite entanglement
    async fn calculate_bipartite_entanglement(&self, feature_data: &[f64]) -> Result<f64> {
        // Simplified bipartite entanglement calculation using von Neumann entropy
        let mean = feature_data.iter().sum::<f64>() / feature_data.len() as f64;
        let variance = feature_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / feature_data.len() as f64;
        
        let entanglement = if variance > 0.0 {
            -variance.ln().abs()
        } else {
            0.0
        };
        
        Ok(entanglement)
    }

    /// Calculate multipartite entanglement
    async fn calculate_multipartite_entanglement(&self, feature_data: &[f64]) -> Result<f64> {
        // Simplified multipartite entanglement using mutual information
        let mut mutual_info = 0.0;
        
        for i in 0..feature_data.len() {
            for j in i+1..feature_data.len() {
                let correlation = feature_data[i] * feature_data[j];
                if correlation > 0.0 {
                    mutual_info += correlation.ln();
                }
            }
        }
        
        Ok(mutual_info)
    }

    /// Reset analyzer
    pub fn reset(&mut self) -> Result<()> {
        self.parameters.clear();
        self.stats.reset();
        Ok(())
    }
}

/// Entanglement types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglementType {
    /// Bipartite entanglement
    Bipartite,
    /// Multipartite entanglement
    Multipartite,
}

/// Placeholder implementations for other analyzers
macro_rules! impl_analyzer {
    ($name:ident, $type:ident, $method:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            pub name: String,
            pub analysis_type: $type,
            pub n_qubits: usize,
            pub parameters: Vec<f64>,
            pub stats: AnalysisStats,
        }

        impl $name {
            pub fn new(name: String, analysis_type: $type, n_qubits: usize) -> Result<Self> {
                Ok(Self {
                    name,
                    analysis_type,
                    n_qubits,
                    parameters: Vec::new(),
                    stats: AnalysisStats::new(),
                })
            }

            pub async fn $method(&self, data: &Array2<f64>) -> Result<Vec<f64>> {
                // Placeholder implementation
                Ok(vec![0.0; data.ncols()])
            }

            pub fn reset(&mut self) -> Result<()> {
                self.parameters.clear();
                self.stats.reset();
                Ok(())
            }
        }
    };
}

impl_analyzer!(InterferenceDetector, InterferenceType, detect_interference);
impl_analyzer!(PhaseExtractor, PhaseExtractionType, extract_phases);
impl_analyzer!(AmplitudeAnalyzer, AmplitudeAnalysisType, analyze_amplitudes);
impl_analyzer!(CoherenceMeter, CoherenceType, measure_coherence);

/// Analysis type enums
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceType {
    Amplitude,
    Phase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseExtractionType {
    QuantumFourierTransform,
    Ramsey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmplitudeAnalysisType {
    Direct,
    Tomographic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceType {
    L1Norm,
    RelativeEntropy,
}

/// Feature importance calculator
#[derive(Debug, Clone)]
pub struct FeatureImportanceCalculator {
    /// Configuration
    pub config: QuantumConfig,
    /// Importance calculation method
    pub method: ImportanceMethod,
    /// Calculation parameters
    pub parameters: Vec<f64>,
}

impl FeatureImportanceCalculator {
    /// Create new importance calculator
    pub fn new(config: QuantumConfig) -> Result<Self> {
        Ok(Self {
            config,
            method: ImportanceMethod::QuantumFisherInformation,
            parameters: Vec::new(),
        })
    }

    /// Calculate feature importance
    pub async fn calculate_importance(&self, features: &QuantumFeatures) -> Result<Vec<f64>> {
        match self.method {
            ImportanceMethod::QuantumFisherInformation => {
                self.calculate_quantum_fisher_importance(features).await
            }
            ImportanceMethod::QuantumGradient => {
                self.calculate_quantum_gradient_importance(features).await
            }
        }
    }

    /// Calculate quantum Fisher information importance
    async fn calculate_quantum_fisher_importance(&self, features: &QuantumFeatures) -> Result<Vec<f64>> {
        let mut importance = Vec::new();
        
        for i in 0..features.classical_features.len() {
            let fisher_info = self.calculate_fisher_information(features, i).await?;
            importance.push(fisher_info);
        }
        
        Ok(importance)
    }

    /// Calculate Fisher information for a feature
    async fn calculate_fisher_information(&self, features: &QuantumFeatures, feature_idx: usize) -> Result<f64> {
        if feature_idx >= features.classical_features.len() {
            return Ok(0.0);
        }
        
        let feature_value = features.classical_features[feature_idx];
        let fisher_info = feature_value.abs(); // Simplified calculation
        
        Ok(fisher_info)
    }

    /// Calculate quantum gradient importance
    async fn calculate_quantum_gradient_importance(&self, features: &QuantumFeatures) -> Result<Vec<f64>> {
        let mut importance = Vec::new();
        
        for i in 0..features.classical_features.len() {
            let gradient = self.calculate_quantum_gradient(features, i).await?;
            importance.push(gradient.abs());
        }
        
        Ok(importance)
    }

    /// Calculate quantum gradient for a feature
    async fn calculate_quantum_gradient(&self, features: &QuantumFeatures, feature_idx: usize) -> Result<f64> {
        if feature_idx >= features.classical_features.len() {
            return Ok(0.0);
        }
        
        let feature_value = features.classical_features[feature_idx];
        let gradient = feature_value * 2.0; // Simplified calculation
        
        Ok(gradient)
    }

    /// Reset calculator
    pub fn reset(&mut self) -> Result<()> {
        self.parameters.clear();
        Ok(())
    }
}

/// Importance calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceMethod {
    /// Quantum Fisher information
    QuantumFisherInformation,
    /// Quantum gradient
    QuantumGradient,
}

/// Statistics structures
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub total_generations: u64,
    pub average_fidelity: f64,
    pub generation_time: f64,
}

impl GenerationStats {
    pub fn new() -> Self {
        Self {
            total_generations: 0,
            average_fidelity: 0.0,
            generation_time: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total_generations = 0;
        self.average_fidelity = 0.0;
        self.generation_time = 0.0;
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisStats {
    pub total_analyses: u64,
    pub average_accuracy: f64,
    pub analysis_time: f64,
}

impl AnalysisStats {
    pub fn new() -> Self {
        Self {
            total_analyses: 0,
            average_accuracy: 0.0,
            analysis_time: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total_analyses = 0;
        self.average_accuracy = 0.0;
        self.analysis_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_features_creation() {
        let classical_features = vec![0.1, 0.2, 0.3];
        let features = QuantumFeatures::new(classical_features.clone());
        
        assert_eq!(features.classical_features, classical_features);
        assert_eq!(features.superposition_features.len(), 3);
        assert_eq!(features.entanglement_features.len(), 3);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = QuantumConfig::default();
        let extractor = QuantumFeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_superposition_generator() {
        let generator = SuperpositionGenerator::new(
            "test".to_string(),
            SuperpositionType::Uniform,
            2,
        ).unwrap();
        
        assert_eq!(generator.name, "test");
        assert_eq!(generator.superposition_type, SuperpositionType::Uniform);
        assert_eq!(generator.n_qubits, 2);
    }

    #[test]
    fn test_entanglement_analyzer() {
        let analyzer = EntanglementAnalyzer::new(
            "test".to_string(),
            EntanglementType::Bipartite,
            2,
        ).unwrap();
        
        assert_eq!(analyzer.name, "test");
        assert_eq!(analyzer.entanglement_type, EntanglementType::Bipartite);
        assert_eq!(analyzer.n_qubits, 2);
    }

    #[test]
    fn test_feature_importance_calculator() {
        let config = QuantumConfig::default();
        let calculator = FeatureImportanceCalculator::new(config).unwrap();
        assert_eq!(calculator.method, ImportanceMethod::QuantumFisherInformation);
    }

    #[test]
    fn test_quantum_features_statistics() {
        let classical_features = vec![0.1, 0.2, 0.3];
        let features = QuantumFeatures::new(classical_features);
        
        let stats = features.calculate_statistics();
        assert_abs_diff_eq!(stats.classical_mean, 0.2, epsilon = 1e-10);
        assert!(stats.classical_std > 0.0);
    }

    #[test]
    fn test_most_important_features() {
        let classical_features = vec![0.1, 0.2, 0.3];
        let mut features = QuantumFeatures::new(classical_features);
        features.feature_importance = vec![0.5, 0.8, 0.3];
        
        let most_important = features.most_important_features(2);
        assert_eq!(most_important.len(), 2);
        assert_eq!(most_important[0], (1, 0.8));
        assert_eq!(most_important[1], (0, 0.5));
    }

    #[tokio::test]
    async fn test_uniform_superposition_generation() {
        let generator = SuperpositionGenerator::new(
            "uniform".to_string(),
            SuperpositionType::Uniform,
            2,
        ).unwrap();
        
        let data = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let features = generator.generate_uniform_superposition(&data).await.unwrap();
        
        assert_eq!(features.len(), 2);
        for feature in features {
            assert_abs_diff_eq!(feature.norm(), 0.5, epsilon = 1e-10);
        }
    }
}