use std::collections::{HashMap, VecDeque};
use nalgebra::{DVector, DMatrix, Complex};
use crate::emergence::detector::{EmergenceHistory, SystemMetrics, EmergenceType};

/// Temporal pattern recognition system for identifying recurring patterns in emergent behaviors
/// Implements various pattern detection algorithms including spectral analysis, template matching, and grammar induction
pub struct TemporalPatternRecognizer {
    /// Pattern detection parameters
    params: PatternParameters,
    /// Discovered patterns
    patterns: Vec<TemporalPattern>,
    /// Pattern matching engine
    matcher: PatternMatcher,
    /// Spectral analyzer for frequency domain patterns
    spectral_analyzer: SpectralAnalyzer,
    /// Grammar induction system
    grammar_inducer: GrammarInducer,
    /// Pattern history and statistics
    pattern_history: PatternHistory,
}

#[derive(Clone, Debug)]
pub struct PatternParameters {
    /// Minimum pattern length
    pub min_pattern_length: usize,
    /// Maximum pattern length
    pub max_pattern_length: usize,
    /// Pattern similarity threshold
    pub similarity_threshold: f64,
    /// Minimum pattern frequency for significance
    pub min_frequency: usize,
    /// Window size for analysis
    pub analysis_window: usize,
    /// Noise tolerance level
    pub noise_tolerance: f64,
    /// Spectral resolution
    pub spectral_resolution: usize,
}

#[derive(Clone, Debug)]
pub struct TemporalPattern {
    /// Unique pattern identifier
    pub id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern template (sequence of values)
    pub template: Vec<f64>,
    /// Pattern duration
    pub duration: usize,
    /// Pattern frequency (how often it occurs)
    pub frequency: usize,
    /// Pattern strength/confidence
    pub strength: f64,
    /// Spectral signature
    pub spectral_signature: Vec<Complex<f64>>,
    /// Associated emergence types
    pub emergence_types: Vec<EmergenceType>,
    /// Statistical properties
    pub statistics: PatternStatistics,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PatternType {
    /// Periodic/cyclic patterns
    Periodic,
    /// Quasi-periodic patterns
    QuasiPeriodic,
    /// Chaotic patterns with structure
    Chaotic,
    /// Transient patterns
    Transient,
    /// Critical scaling patterns
    Critical,
    /// Synchronization patterns
    Synchronization,
    /// Cascade patterns
    Cascade,
    /// Self-similar fractal patterns
    Fractal,
}

#[derive(Clone, Debug)]
pub struct PatternStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Hurst exponent (for self-similarity)
    pub hurst_exponent: f64,
    /// Lyapunov exponent (for chaos)
    pub lyapunov_exponent: f64,
    /// Fractal dimension
    pub fractal_dimension: f64,
}

#[derive(Clone, Debug)]
pub struct PatternHistory {
    /// Recent pattern occurrences
    pub recent_occurrences: VecDeque<PatternOccurrence>,
    /// Pattern evolution over time
    pub pattern_evolution: HashMap<String, Vec<PatternSnapshot>>,
    /// Pattern interaction matrix
    pub interaction_matrix: DMatrix<f64>,
    /// Prediction accuracy history
    pub prediction_accuracy: VecDeque<f64>,
}

#[derive(Clone, Debug)]
pub struct PatternOccurrence {
    pub pattern_id: String,
    pub timestamp: f64,
    pub strength: f64,
    pub location: Option<DVector<f64>>,
    pub context: PatternContext,
}

#[derive(Clone, Debug)]
pub struct PatternContext {
    /// System state when pattern occurred
    pub system_metrics: SystemMetrics,
    /// Concurrent patterns
    pub concurrent_patterns: Vec<String>,
    /// Environmental conditions
    pub environment: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct PatternSnapshot {
    pub timestamp: f64,
    pub template: Vec<f64>,
    pub strength: f64,
    pub frequency: usize,
}

impl Default for PatternParameters {
    fn default() -> Self {
        Self {
            min_pattern_length: 5,
            max_pattern_length: 100,
            similarity_threshold: 0.8,
            min_frequency: 3,
            analysis_window: 1000,
            noise_tolerance: 0.1,
            spectral_resolution: 256,
        }
    }
}

impl TemporalPatternRecognizer {
    /// Create new temporal pattern recognizer
    pub fn new(params: PatternParameters) -> Self {
        let matcher = PatternMatcher::new(&params);
        let spectral_analyzer = SpectralAnalyzer::new(params.spectral_resolution);
        let grammar_inducer = GrammarInducer::new();
        
        let pattern_history = PatternHistory {
            recent_occurrences: VecDeque::with_capacity(params.analysis_window),
            pattern_evolution: HashMap::new(),
            interaction_matrix: DMatrix::zeros(0, 0),
            prediction_accuracy: VecDeque::with_capacity(100),
        };

        Self {
            params,
            patterns: Vec::new(),
            matcher,
            spectral_analyzer,
            grammar_inducer,
            pattern_history,
        }
    }

    /// Analyze emergence history for temporal patterns
    pub fn analyze_patterns(&mut self, history: &EmergenceHistory) -> Vec<TemporalPattern> {
        if history.metrics_history.len() < self.params.min_pattern_length {
            return Vec::new();
        }

        // Extract time series for different metrics
        let complexity_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.complexity)
            .collect();
        
        let coherence_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.coherence)
            .collect();
        
        let energy_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.total_energy)
            .collect();
        
        let coupling_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.coupling)
            .collect();

        let mut discovered_patterns = Vec::new();

        // Discover patterns in each metric
        discovered_patterns.extend(self.discover_patterns_in_series(&complexity_series, "complexity"));
        discovered_patterns.extend(self.discover_patterns_in_series(&coherence_series, "coherence"));
        discovered_patterns.extend(self.discover_patterns_in_series(&energy_series, "energy"));
        discovered_patterns.extend(self.discover_patterns_in_series(&coupling_series, "coupling"));

        // Cross-metric pattern analysis
        discovered_patterns.extend(self.discover_cross_metric_patterns(history));

        // Update pattern database
        self.update_pattern_database(discovered_patterns.clone());

        // Analyze pattern interactions
        self.analyze_pattern_interactions(&discovered_patterns);

        discovered_patterns
    }

    /// Discover patterns in a single time series
    fn discover_patterns_in_series(&mut self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        // Template matching approach
        patterns.extend(self.matcher.find_recurring_templates(series, metric_name));

        // Spectral analysis approach
        patterns.extend(self.spectral_analyzer.find_spectral_patterns(series, metric_name));

        // Grammar induction approach
        patterns.extend(self.grammar_inducer.induce_patterns(series, metric_name));

        // Statistical pattern detection
        patterns.extend(self.find_statistical_patterns(series, metric_name));

        patterns
    }

    /// Discover cross-metric patterns
    fn discover_cross_metric_patterns(&mut self, history: &EmergenceHistory) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        if history.metrics_history.len() < self.params.min_pattern_length {
            return patterns;
        }

        // Analyze correlations between metrics
        let metrics_matrix = self.build_metrics_matrix(history);
        patterns.extend(self.find_correlation_patterns(&metrics_matrix));

        // Phase relationship patterns
        patterns.extend(self.find_phase_patterns(history));

        // Cascade patterns (one metric triggering changes in others)
        patterns.extend(self.find_cascade_patterns(history));

        patterns
    }

    /// Build matrix of all metrics over time
    fn build_metrics_matrix(&self, history: &EmergenceHistory) -> DMatrix<f64> {
        let num_metrics = 7; // complexity, coherence, energy, coupling, entropy, information, system_size
        let num_timesteps = history.metrics_history.len();
        
        let mut matrix = DMatrix::zeros(num_timesteps, num_metrics);

        for (i, metrics) in history.metrics_history.iter().enumerate() {
            matrix[(i, 0)] = metrics.complexity;
            matrix[(i, 1)] = metrics.coherence;
            matrix[(i, 2)] = metrics.total_energy;
            matrix[(i, 3)] = metrics.coupling;
            matrix[(i, 4)] = metrics.entropy;
            matrix[(i, 5)] = metrics.information;
            matrix[(i, 6)] = metrics.system_size as f64;
        }

        matrix
    }

    /// Find correlation patterns between metrics
    fn find_correlation_patterns(&self, metrics_matrix: &DMatrix<f64>) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        let num_metrics = metrics_matrix.ncols();

        // Calculate cross-correlations
        for i in 0..num_metrics {
            for j in (i + 1)..num_metrics {
                let series_i: Vec<f64> = metrics_matrix.column(i).iter().cloned().collect();
                let series_j: Vec<f64> = metrics_matrix.column(j).iter().cloned().collect();

                let correlation = self.calculate_correlation(&series_i, &series_j);
                
                if correlation.abs() > self.params.similarity_threshold {
                    let pattern_id = format!("correlation_{}_{}", i, j);
                    
                    let pattern = TemporalPattern {
                        id: pattern_id,
                        pattern_type: if correlation > 0.0 { 
                            PatternType::Synchronization 
                        } else { 
                            PatternType::QuasiPeriodic 
                        },
                        template: vec![correlation],
                        duration: series_i.len(),
                        frequency: 1, // Correlation patterns are persistent
                        strength: correlation.abs(),
                        spectral_signature: Vec::new(),
                        emergence_types: vec![EmergenceType::Synchronization],
                        statistics: self.calculate_pattern_statistics(&series_i),
                    };

                    patterns.push(pattern);
                }
            }
        }

        patterns
    }

    /// Find phase relationship patterns
    fn find_phase_patterns(&self, history: &EmergenceHistory) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        if history.metrics_history.len() < 20 {
            return patterns;
        }

        // Analyze phase relationships using Hilbert transform approximation
        let complexity_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.complexity)
            .collect();
        
        let coherence_series: Vec<f64> = history.metrics_history.iter()
            .map(|m| m.coherence)
            .collect();

        // Simple phase difference calculation (approximation)
        let mut phase_diffs = Vec::new();
        for i in 1..complexity_series.len() {
            let complex_phase = (complexity_series[i] - complexity_series[i-1]).atan2(1.0);
            let coherence_phase = (coherence_series[i] - coherence_series[i-1]).atan2(1.0);
            phase_diffs.push(complex_phase - coherence_phase);
        }

        // Look for consistent phase relationships
        if !phase_diffs.is_empty() {
            let mean_phase_diff: f64 = phase_diffs.iter().sum::<f64>() / phase_diffs.len() as f64;
            let phase_stability: f64 = 1.0 - (phase_diffs.iter()
                .map(|&pd| (pd - mean_phase_diff).abs())
                .sum::<f64>() / phase_diffs.len() as f64) / std::f64::consts::PI;

            if phase_stability > 0.7 {
                let pattern = TemporalPattern {
                    id: "phase_relationship".to_string(),
                    pattern_type: PatternType::Synchronization,
                    template: vec![mean_phase_diff],
                    duration: phase_diffs.len(),
                    frequency: 1,
                    strength: phase_stability,
                    spectral_signature: Vec::new(),
                    emergence_types: vec![EmergenceType::Synchronization],
                    statistics: self.calculate_pattern_statistics(&phase_diffs),
                };

                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Find cascade patterns (one event triggering others)
    fn find_cascade_patterns(&self, history: &EmergenceHistory) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        if history.metrics_history.len() < 10 {
            return patterns;
        }

        // Look for avalanche events followed by complexity changes
        if !history.avalanche_events.is_empty() {
            let mut cascade_events = Vec::new();

            for avalanche in &history.avalanche_events {
                // Find metrics around the avalanche time
                let avalanche_time = avalanche.timestamp;
                
                let before_metrics = history.metrics_history.iter()
                    .find(|m| (m.timestamp - avalanche_time).abs() < 0.5 && m.timestamp <= avalanche_time);
                
                let after_metrics = history.metrics_history.iter()
                    .find(|m| (m.timestamp - avalanche_time).abs() < 0.5 && m.timestamp > avalanche_time);

                if let (Some(before), Some(after)) = (before_metrics, after_metrics) {
                    let complexity_change = after.complexity - before.complexity;
                    let coherence_change = after.coherence - before.coherence;
                    
                    cascade_events.push(vec![
                        avalanche.size as f64,
                        complexity_change,
                        coherence_change,
                    ]);
                }
            }

            if cascade_events.len() >= self.params.min_frequency {
                let pattern = TemporalPattern {
                    id: "avalanche_cascade".to_string(),
                    pattern_type: PatternType::Cascade,
                    template: cascade_events.iter()
                        .map(|event| event.iter().sum::<f64>() / event.len() as f64)
                        .collect(),
                    duration: 3, // avalanche -> complexity -> coherence
                    frequency: cascade_events.len(),
                    strength: 0.8, // High strength for clear cascade patterns
                    spectral_signature: Vec::new(),
                    emergence_types: vec![EmergenceType::CriticalBehavior, EmergenceType::SelfOrganization],
                    statistics: PatternStatistics {
                        mean: 0.0,
                        std_dev: 0.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                        hurst_exponent: 0.5,
                        lyapunov_exponent: 0.0,
                        fractal_dimension: 1.0,
                    },
                };

                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Find statistical patterns (trends, periodicities, etc.)
    fn find_statistical_patterns(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        if series.len() < self.params.min_pattern_length {
            return patterns;
        }

        // Trend analysis
        if let Some(trend_pattern) = self.detect_trend_pattern(series, metric_name) {
            patterns.push(trend_pattern);
        }

        // Oscillation detection
        patterns.extend(self.detect_oscillation_patterns(series, metric_name));

        // Jump/discontinuity detection
        patterns.extend(self.detect_jump_patterns(series, metric_name));

        // Self-similarity analysis
        if let Some(fractal_pattern) = self.detect_fractal_pattern(series, metric_name) {
            patterns.push(fractal_pattern);
        }

        patterns
    }

    /// Detect trend patterns
    fn detect_trend_pattern(&self, series: &[f64], metric_name: &str) -> Option<TemporalPattern> {
        if series.len() < 10 {
            return None;
        }

        // Linear regression to detect trends
        let x: Vec<f64> = (0..series.len()).map(|i| i as f64).collect();
        let slope = self.calculate_linear_regression_slope(&x, series);

        if slope.abs() > 0.001 { // Significant trend
            let pattern_type = if slope > 0.0 {
                PatternType::Transient // Growing trend
            } else {
                PatternType::Critical // Declining trend
            };

            Some(TemporalPattern {
                id: format!("trend_{}", metric_name),
                pattern_type,
                template: vec![slope],
                duration: series.len(),
                frequency: 1,
                strength: slope.abs() * 100.0, // Scale for visibility
                spectral_signature: Vec::new(),
                emergence_types: vec![EmergenceType::SelfOrganization],
                statistics: self.calculate_pattern_statistics(series),
            })
        } else {
            None
        }
    }

    /// Detect oscillation patterns
    fn detect_oscillation_patterns(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        // Simple peak detection for oscillations
        let peaks = self.find_peaks(series);
        
        if peaks.len() >= 3 {
            // Calculate period from peak intervals
            let mut intervals = Vec::new();
            for i in 1..peaks.len() {
                intervals.push(peaks[i] - peaks[i-1]);
            }

            let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let interval_std: f64 = {
                let variance: f64 = intervals.iter()
                    .map(|&x| (x - mean_interval).powi(2))
                    .sum::<f64>() / intervals.len() as f64;
                variance.sqrt()
            };

            // Regular oscillation if intervals are consistent
            if interval_std < mean_interval * 0.2 { // Less than 20% variation
                let pattern = TemporalPattern {
                    id: format!("oscillation_{}", metric_name),
                    pattern_type: PatternType::Periodic,
                    template: vec![mean_interval, interval_std],
                    duration: mean_interval as usize,
                    frequency: peaks.len(),
                    strength: 1.0 - (interval_std / mean_interval),
                    spectral_signature: Vec::new(),
                    emergence_types: vec![EmergenceType::Synchronization],
                    statistics: self.calculate_pattern_statistics(series),
                };

                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Find peaks in time series
    fn find_peaks(&self, series: &[f64]) -> Vec<usize> {
        let mut peaks = Vec::new();

        for i in 1..(series.len() - 1) {
            if series[i] > series[i-1] && series[i] > series[i+1] {
                // Additional check for significance
                let local_max = series[i];
                let local_avg = (series[i-1] + series[i+1]) / 2.0;
                
                if local_max > local_avg * 1.1 { // At least 10% higher
                    peaks.push(i);
                }
            }
        }

        peaks
    }

    /// Detect jump/discontinuity patterns
    fn detect_jump_patterns(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        // Calculate differences
        let mut diffs = Vec::new();
        for i in 1..series.len() {
            diffs.push((series[i] - series[i-1]).abs());
        }

        if diffs.is_empty() {
            return patterns;
        }

        // Find large jumps
        let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_diff: f64 = {
            let variance: f64 = diffs.iter()
                .map(|&x| (x - mean_diff).powi(2))
                .sum::<f64>() / diffs.len() as f64;
            variance.sqrt()
        };

        let jump_threshold = mean_diff + 3.0 * std_diff; // 3-sigma threshold
        let mut jumps = Vec::new();

        for (i, &diff) in diffs.iter().enumerate() {
            if diff > jump_threshold {
                jumps.push((i + 1, diff)); // +1 because diffs is offset by 1
            }
        }

        if jumps.len() >= 2 {
            let jump_pattern = TemporalPattern {
                id: format!("jumps_{}", metric_name),
                pattern_type: PatternType::Critical,
                template: jumps.iter().map(|(_, magnitude)| *magnitude).collect(),
                duration: 1, // Jumps are instantaneous
                frequency: jumps.len(),
                strength: jump_threshold / mean_diff,
                spectral_signature: Vec::new(),
                emergence_types: vec![EmergenceType::PhaseTransition],
                statistics: self.calculate_pattern_statistics(&jumps.iter().map(|(_, m)| *m).collect::<Vec<_>>()),
            };

            patterns.push(jump_pattern);
        }

        patterns
    }

    /// Detect fractal/self-similar patterns
    fn detect_fractal_pattern(&self, series: &[f64], metric_name: &str) -> Option<TemporalPattern> {
        if series.len() < 50 {
            return None;
        }

        // Estimate Hurst exponent using R/S analysis
        let hurst_exponent = self.calculate_hurst_exponent(series);
        
        // Fractal if significantly different from 0.5 (random walk)
        if (hurst_exponent - 0.5).abs() > 0.1 {
            Some(TemporalPattern {
                id: format!("fractal_{}", metric_name),
                pattern_type: PatternType::Fractal,
                template: vec![hurst_exponent],
                duration: series.len(),
                frequency: 1,
                strength: (hurst_exponent - 0.5).abs() * 2.0, // Scale to [0,1]
                spectral_signature: Vec::new(),
                emergence_types: vec![EmergenceType::SelfOrganization],
                statistics: PatternStatistics {
                    mean: series.iter().sum::<f64>() / series.len() as f64,
                    std_dev: {
                        let mean = series.iter().sum::<f64>() / series.len() as f64;
                        let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
                        variance.sqrt()
                    },
                    skewness: 0.0, // Would need more complex calculation
                    kurtosis: 0.0,
                    hurst_exponent,
                    lyapunov_exponent: 0.0,
                    fractal_dimension: 2.0 - hurst_exponent,
                },
            })
        } else {
            None
        }
    }

    /// Calculate Hurst exponent using R/S analysis
    fn calculate_hurst_exponent(&self, series: &[f64]) -> f64 {
        if series.len() < 10 {
            return 0.5; // Default for insufficient data
        }

        let n = series.len();
        let mean: f64 = series.iter().sum::<f64>() / n as f64;
        
        // Calculate cumulative deviations from mean
        let mut cumulative_devs = Vec::with_capacity(n);
        let mut sum = 0.0;
        for &value in series {
            sum += value - mean;
            cumulative_devs.push(sum);
        }
        
        // Calculate range
        let max_dev = cumulative_devs.iter().fold(f64::NEG_INFINITY, |max, &val| val.max(max));
        let min_dev = cumulative_devs.iter().fold(f64::INFINITY, |min, &val| val.min(min));
        let range = max_dev - min_dev;
        
        // Calculate standard deviation
        let variance: f64 = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 || range == 0.0 {
            return 0.5;
        }
        
        // R/S statistic
        let rs = range / std_dev;
        
        // Hurst exponent approximation: H ≈ log(R/S) / log(n)
        // This is a simplified version; full R/S analysis would use multiple window sizes
        let hurst = rs.ln() / (n as f64).ln();
        
        // Clamp to reasonable range
        hurst.max(0.0).min(1.0)
    }

    /// Update pattern database with new discoveries
    fn update_pattern_database(&mut self, new_patterns: Vec<TemporalPattern>) {
        for new_pattern in new_patterns {
            // Check if similar pattern already exists
            let mut found_similar = false;
            
            for existing_pattern in &mut self.patterns {
                if self.patterns_are_similar(&new_pattern, existing_pattern) {
                    // Update existing pattern
                    existing_pattern.frequency += 1;
                    existing_pattern.strength = (existing_pattern.strength + new_pattern.strength) / 2.0;
                    found_similar = true;
                    break;
                }
            }
            
            // Add new pattern if no similar one found
            if !found_similar {
                self.patterns.push(new_pattern);
            }
        }
        
        // Remove weak patterns
        self.patterns.retain(|p| p.frequency >= self.params.min_frequency);
    }

    /// Check if two patterns are similar
    fn patterns_are_similar(&self, pattern1: &TemporalPattern, pattern2: &TemporalPattern) -> bool {
        // Same type and similar template
        pattern1.pattern_type == pattern2.pattern_type &&
        pattern1.id.contains(&pattern2.id.split('_').next().unwrap_or("")) &&
        self.template_similarity(&pattern1.template, &pattern2.template) > self.params.similarity_threshold
    }

    /// Calculate template similarity
    fn template_similarity(&self, template1: &[f64], template2: &[f64]) -> f64 {
        if template1.is_empty() || template2.is_empty() {
            return 0.0;
        }
        
        // Use correlation as similarity measure
        self.calculate_correlation(template1, template2).abs()
    }

    /// Analyze interactions between patterns
    fn analyze_pattern_interactions(&mut self, patterns: &[TemporalPattern]) {
        if patterns.len() < 2 {
            return;
        }

        // Build interaction matrix
        let n = patterns.len();
        let mut interaction_matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let interaction_strength = self.calculate_pattern_interaction(&patterns[i], &patterns[j]);
                    interaction_matrix[(i, j)] = interaction_strength;
                }
            }
        }

        self.pattern_history.interaction_matrix = interaction_matrix;
    }

    /// Calculate interaction strength between two patterns
    fn calculate_pattern_interaction(&self, pattern1: &TemporalPattern, pattern2: &TemporalPattern) -> f64 {
        // Simple interaction based on temporal overlap and emergence type compatibility
        let type_compatibility = match (&pattern1.pattern_type, &pattern2.pattern_type) {
            (PatternType::Periodic, PatternType::Periodic) => 0.8,
            (PatternType::Chaotic, PatternType::Critical) => 0.7,
            (PatternType::Synchronization, PatternType::Synchronization) => 0.9,
            (PatternType::Cascade, PatternType::Critical) => 0.8,
            _ => 0.3,
        };

        // Temporal compatibility (similar durations interact more)
        let duration_similarity = if pattern1.duration > 0 && pattern2.duration > 0 {
            let ratio = pattern1.duration.min(pattern2.duration) as f64 / 
                       pattern1.duration.max(pattern2.duration) as f64;
            ratio
        } else {
            0.5
        };

        (type_compatibility + duration_similarity) / 2.0
    }

    /// Predict future patterns based on current state
    pub fn predict_patterns(&self, horizon: usize) -> Vec<PatternPrediction> {
        let mut predictions = Vec::new();

        for pattern in &self.patterns {
            if pattern.pattern_type == PatternType::Periodic || 
               pattern.pattern_type == PatternType::QuasiPeriodic {
                
                let prediction = PatternPrediction {
                    pattern_id: pattern.id.clone(),
                    predicted_occurrence_time: horizon as f64, // Simplified
                    confidence: pattern.strength * (pattern.frequency as f64 / 100.0).min(1.0),
                    expected_strength: pattern.strength,
                    emergence_types: pattern.emergence_types.clone(),
                };
                
                predictions.push(prediction);
            }
        }

        predictions
    }

    /// Helper functions for calculations
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

    fn calculate_linear_regression_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }

    fn calculate_pattern_statistics(&self, data: &[f64]) -> PatternStatistics {
        if data.is_empty() {
            return PatternStatistics {
                mean: 0.0,
                std_dev: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                hurst_exponent: 0.5,
                lyapunov_exponent: 0.0,
                fractal_dimension: 1.0,
            };
        }

        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        PatternStatistics {
            mean,
            std_dev,
            skewness: 0.0, // Simplified - would need proper calculation
            kurtosis: 0.0, // Simplified - would need proper calculation
            hurst_exponent: self.calculate_hurst_exponent(data),
            lyapunov_exponent: 0.0, // Would need more complex calculation
            fractal_dimension: 1.5, // Simplified estimate
        }
    }

    /// Get all discovered patterns
    pub fn get_patterns(&self) -> &[TemporalPattern] {
        &self.patterns
    }

    /// Get pattern history
    pub fn get_pattern_history(&self) -> &PatternHistory {
        &self.pattern_history
    }
}

#[derive(Clone, Debug)]
pub struct PatternPrediction {
    pub pattern_id: String,
    pub predicted_occurrence_time: f64,
    pub confidence: f64,
    pub expected_strength: f64,
    pub emergence_types: Vec<EmergenceType>,
}

/// Pattern matching engine for template-based detection
#[derive(Clone, Debug)]
pub struct PatternMatcher {
    similarity_threshold: f64,
    min_length: usize,
    max_length: usize,
}

impl PatternMatcher {
    fn new(params: &PatternParameters) -> Self {
        Self {
            similarity_threshold: params.similarity_threshold,
            min_length: params.min_pattern_length,
            max_length: params.max_pattern_length,
        }
    }

    fn find_recurring_templates(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        // Simple sliding window template matching
        for window_size in self.min_length..=self.max_length.min(series.len() / 2) {
            let templates =  self.extract_templates(series, window_size);
            patterns.extend(self.find_matching_templates(templates, metric_name, window_size));
        }
        
        patterns
    }

    fn extract_templates(&self, series: &[f64], window_size: usize) -> Vec<Vec<f64>> {
        let mut templates = Vec::new();
        
        for i in 0..=(series.len().saturating_sub(window_size)) {
            let template: Vec<f64> = series[i..i + window_size].to_vec();
            templates.push(template);
        }
        
        templates
    }

    fn find_matching_templates(&self, templates: Vec<Vec<f64>>, metric_name: &str, window_size: usize) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        let mut used_templates = vec![false; templates.len()];
        
        for i in 0..templates.len() {
            if used_templates[i] {
                continue;
            }
            
            let mut matches = vec![i];
            used_templates[i] = true;
            
            for j in (i + 1)..templates.len() {
                if used_templates[j] {
                    continue;
                }
                
                let similarity = self.calculate_template_similarity(&templates[i], &templates[j]);
                if similarity > self.similarity_threshold {
                    matches.push(j);
                    used_templates[j] = true;
                }
            }
            
            if matches.len() >= 2 { // At least 2 occurrences
                let pattern = TemporalPattern {
                    id: format!("template_{}_{}", metric_name, window_size),
                    pattern_type: PatternType::QuasiPeriodic,
                    template: templates[i].clone(),
                    duration: window_size,
                    frequency: matches.len(),
                    strength: 0.7, // Default strength for template matches
                    spectral_signature: Vec::new(),
                    emergence_types: vec![EmergenceType::PatternFormation],
                    statistics: PatternStatistics {
                        mean: templates[i].iter().sum::<f64>() / templates[i].len() as f64,
                        std_dev: 0.0, // Simplified
                        skewness: 0.0,
                        kurtosis: 0.0,
                        hurst_exponent: 0.5,
                        lyapunov_exponent: 0.0,
                        fractal_dimension: 1.0,
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        patterns
    }

    fn calculate_template_similarity(&self, template1: &[f64], template2: &[f64]) -> f64 {
        if template1.len() != template2.len() {
            return 0.0;
        }
        
        // Normalized cross-correlation
        let mean1: f64 = template1.iter().sum::<f64>() / template1.len() as f64;
        let mean2: f64 = template2.iter().sum::<f64>() / template2.len() as f64;
        
        let numerator: f64 = template1.iter().zip(template2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum();
        
        let norm1: f64 = template1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = template2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>().sqrt();
        
        if norm1 * norm2 == 0.0 {
            0.0
        } else {
            numerator / (norm1 * norm2)
        }
    }
}

/// Spectral analyzer for frequency domain patterns
#[derive(Clone, Debug)]
pub struct SpectralAnalyzer {
    resolution: usize,
}

impl SpectralAnalyzer {
    fn new(resolution: usize) -> Self {
        Self { resolution }
    }

    fn find_spectral_patterns(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        // Simplified spectral analysis - in practice would use FFT
        let mut patterns = Vec::new();
        
        if series.len() < 16 {
            return patterns;
        }
        
        // Find dominant frequencies using simple periodogram
        let dominant_frequencies = self.find_dominant_frequencies(series);
        
        for (frequency, strength) in dominant_frequencies {
            if strength > 0.1 { // Threshold for significance
                let pattern = TemporalPattern {
                    id: format!("spectral_{}_{:.3}", metric_name, frequency),
                    pattern_type: PatternType::Periodic,
                    template: vec![frequency, strength],
                    duration: (1.0 / frequency) as usize,
                    frequency: 1,
                    strength,
                    spectral_signature: vec![Complex::new(strength, 0.0)],
                    emergence_types: vec![EmergenceType::Synchronization],
                    statistics: PatternStatistics {
                        mean: series.iter().sum::<f64>() / series.len() as f64,
                        std_dev: 0.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                        hurst_exponent: 0.5,
                        lyapunov_exponent: 0.0,
                        fractal_dimension: 1.0,
                    },
                };
                
                patterns.push(pattern);
            }
        }
        
        patterns
    }

    fn find_dominant_frequencies(&self, series: &[f64]) -> Vec<(f64, f64)> {
        let mut frequencies = Vec::new();
        
        // Simple autocorrelation-based frequency detection
        for lag in 2..=series.len() / 4 {
            let correlation = self.calculate_autocorrelation(series, lag);
            if correlation > 0.5 { // Strong autocorrelation
                let frequency = 1.0 / lag as f64;
                frequencies.push((frequency, correlation));
            }
        }
        
        frequencies
    }

    fn calculate_autocorrelation(&self, series: &[f64], lag: usize) -> f64 {
        if lag >= series.len() {
            return 0.0;
        }
        
        let n = series.len() - lag;
        let mut correlation = 0.0;
        
        for i in 0..n {
            correlation += series[i] * series[i + lag];
        }
        
        correlation / n as f64
    }
}

/// Grammar induction system for symbolic patterns
#[derive(Clone, Debug)]
pub struct GrammarInducer {}

impl GrammarInducer {
    fn new() -> Self {
        Self {}
    }

    fn induce_patterns(&self, series: &[f64], metric_name: &str) -> Vec<TemporalPattern> {
        // Simplified grammar induction - convert to symbols and find rules
        let symbols = self.discretize_series(series);
        self.find_symbolic_patterns(&symbols, metric_name)
    }

    fn discretize_series(&self, series: &[f64]) -> Vec<char> {
        if series.is_empty() {
            return Vec::new();
        }
        
        let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;
        let std_dev: f64 = {
            let variance: f64 = series.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / series.len() as f64;
            variance.sqrt()
        };
        
        let mut symbols = Vec::new();
        
        for &value in series {
            let symbol = if value > mean + std_dev {
                'H' // High
            } else if value < mean - std_dev {
                'L' // Low
            } else {
                'M' // Medium
            };
            symbols.push(symbol);
        }
        
        symbols
    }

    fn find_symbolic_patterns(&self, symbols: &[char], metric_name: &str) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        // Find repeating subsequences
        for len in 2..=10.min(symbols.len() / 2) {
            let repeats = self.find_repeating_subsequences(symbols, len);
            
            for (subsequence, count) in repeats {
                if count >= 2 {
                    // Convert back to numeric for template
                    let template: Vec<f64> = subsequence.iter()
                        .map(|&c| match c {
                            'H' => 1.0,
                            'L' => -1.0,
                            'M' => 0.0,
                            _ => 0.0,
                        })
                        .collect();
                    
                    let pattern = TemporalPattern {
                        id: format!("grammar_{}_{}", metric_name, subsequence.iter().collect::<String>()),
                        pattern_type: PatternType::QuasiPeriodic,
                        template,
                        duration: len,
                        frequency: count,
                        strength: count as f64 / (symbols.len() / len) as f64,
                        spectral_signature: Vec::new(),
                        emergence_types: vec![EmergenceType::PatternFormation],
                        statistics: PatternStatistics {
                            mean: 0.0,
                            std_dev: 1.0,
                            skewness: 0.0,
                            kurtosis: 0.0,
                            hurst_exponent: 0.5,
                            lyapunov_exponent: 0.0,
                            fractal_dimension: 1.0,
                        },
                    };
                    
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }

    fn find_repeating_subsequences(&self, symbols: &[char], length: usize) -> HashMap<Vec<char>, usize> {
        let mut subsequences = HashMap::new();
        
        for i in 0..=(symbols.len().saturating_sub(length)) {
            let subsequence: Vec<char> = symbols[i..i + length].to_vec();
            *subsequences.entry(subsequence).or_insert(0) += 1;
        }
        
        // Only return subsequences that appear multiple times
        subsequences.into_iter()
            .filter(|(_, count)| *count > 1)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_recognizer_creation() {
        let params = PatternParameters::default();
        let recognizer = TemporalPatternRecognizer::new(params);
        
        assert!(recognizer.patterns.is_empty());
        assert_eq!(recognizer.params.min_pattern_length, 5);
    }

    #[test]
    fn test_trend_detection() {
        let params = PatternParameters::default();
        let recognizer = TemporalPatternRecognizer::new(params);
        
        // Create upward trend
        let series: Vec<f64> = (0..20).map(|i| i as f64).collect();
        
        let trend_pattern = recognizer.detect_trend_pattern(&series, "test");
        assert!(trend_pattern.is_some());
        
        let pattern = trend_pattern.unwrap();
        assert!(pattern.template[0] > 0.0); // Positive slope
    }

    #[test]
    fn test_oscillation_detection() {
        let params = PatternParameters::default();
        let recognizer = TemporalPatternRecognizer::new(params);
        
        // Create sine wave
        let series: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.2).sin())
            .collect();
        
        let oscillation_patterns = recognizer.detect_oscillation_patterns(&series, "test");
        assert!(!oscillation_patterns.is_empty());
    }

    #[test]
    fn test_correlation_calculation() {
        let params = PatternParameters::default();
        let recognizer = TemporalPatternRecognizer::new(params);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation
        
        let correlation = recognizer.calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hurst_exponent_calculation() {
        let params = PatternParameters::default();
        let recognizer = TemporalPatternRecognizer::new(params);
        
        // Random walk should have Hurst exponent around 0.5
        let series: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 0.1).collect();
        
        let hurst = recognizer.calculate_hurst_exponent(&series);
        assert!(hurst >= 0.0 && hurst <= 1.0);
    }

    #[test]
    fn test_template_similarity() {
        let params = PatternParameters::default();
        let matcher = PatternMatcher::new(&params);
        
        let template1 = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let template2 = vec![1.1, 2.1, 2.9, 2.1, 1.1]; // Similar
        let template3 = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Different
        
        let sim1 = matcher.calculate_template_similarity(&template1, &template2);
        let sim2 = matcher.calculate_template_similarity(&template1, &template3);
        
        assert!(sim1 > sim2);
        assert!(sim1 > 0.8);
    }
}