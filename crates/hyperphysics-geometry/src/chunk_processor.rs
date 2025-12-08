//! # Chunk-and-Pass Processor for Hyperbolic SNNs
//!
//! Implementation of the "Now-or-Never" bottleneck and Chunk-and-Pass processing
//! from Christiansen & Chater's "Creating Language" framework, adapted for
//! spiking neural networks on hyperbolic lattice topology.
//!
//! ## Theoretical Foundation
//!
//! The Now-or-Never bottleneck forces hierarchical temporal chunking:
//! - Raw sensory input must be rapidly compressed into chunks
//! - Chunks are passed to higher levels for integration
//! - This creates emergent hierarchical structure without explicit design
//!
//! ## Multi-Timescale Mapping
//!
//! | Language Level | Neural Level | Timescale |
//! |----------------|--------------|-----------|
//! | Phonemes       | Spike trains | ~10ms     |
//! | Words          | Spike packets| ~100ms    |
//! | Phrases        | Avalanches   | ~1s       |
//! | Sentences      | SOC events   | ~10s      |
//!
//! ## References
//!
//! - Christiansen & Chater (2016) "Creating Language" MIT Press
//! - Kiebel et al. (2008) "A hierarchy of time-scales" PLoS Comput Biol
//! - Murray et al. (2014) "A hierarchy of intrinsic timescales" Nature Neurosci

use std::collections::VecDeque;

use crate::hyperbolic_snn::LorentzVec;
use crate::free_energy::{Precision, PrecisionWeightedError, HierarchicalErrorAggregator};

/// Temporal resolution constants (in seconds)
pub const SPIKE_RESOLUTION: f64 = 0.001;      // 1ms - spike timing
pub const PACKET_RESOLUTION: f64 = 0.010;     // 10ms - spike packets
pub const CHUNK_RESOLUTION: f64 = 0.100;      // 100ms - basic chunks
pub const PHRASE_RESOLUTION: f64 = 1.0;       // 1s - phrase-level
pub const SENTENCE_RESOLUTION: f64 = 10.0;    // 10s - sentence-level

/// Spike event with precise timing and spatial position
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Time of spike in seconds
    pub time: f64,
    /// Neuron ID that fired
    pub neuron_id: usize,
    /// Position on hyperboloid (Lorentz coordinates)
    pub position: LorentzVec,
    /// Spike amplitude (for graded potentials)
    pub amplitude: f64,
}

/// Spike packet: collection of spikes within a temporal window
#[derive(Debug, Clone)]
pub struct SpikePacket {
    /// Start time of packet window
    pub start_time: f64,
    /// End time of packet window
    pub end_time: f64,
    /// Spikes in this packet
    pub spikes: Vec<SpikeEvent>,
    /// Centroid position on hyperboloid
    pub centroid: LorentzVec,
    /// Total spike count (for rate coding)
    pub spike_count: usize,
    /// Information content estimate (bits)
    pub information: f64,
}

impl SpikePacket {
    /// Create new spike packet for time window
    pub fn new(start_time: f64, end_time: f64) -> Self {
        Self {
            start_time,
            end_time,
            spikes: Vec::new(),
            centroid: LorentzVec::origin(),
            spike_count: 0,
            information: 0.0,
        }
    }

    /// Add spike to packet
    pub fn add_spike(&mut self, spike: SpikeEvent) {
        self.spikes.push(spike);
        self.spike_count += 1;
        self.update_centroid();
        self.update_information();
    }

    /// Update centroid using hyperbolic weighted average
    fn update_centroid(&mut self) {
        if self.spikes.is_empty() {
            self.centroid = LorentzVec::origin();
            return;
        }

        // Weighted average in tangent space at origin, then project back
        let mut _sum_t = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        let mut total_weight = 0.0;

        for spike in &self.spikes {
            let w = spike.amplitude;
            _sum_t += w * spike.position.t;
            sum_x += w * spike.position.x;
            sum_y += w * spike.position.y;
            sum_z += w * spike.position.z;
            total_weight += w;
        }

        if total_weight > 1e-10 {
            // Normalize and project to hyperboloid
            let avg_x = sum_x / total_weight;
            let avg_y = sum_y / total_weight;
            let avg_z = sum_z / total_weight;

            // Project to hyperboloid: t² - x² - y² - z² = 1
            let spatial_sq = avg_x * avg_x + avg_y * avg_y + avg_z * avg_z;
            let t = (1.0 + spatial_sq).sqrt();

            self.centroid = LorentzVec::new(t, avg_x, avg_y, avg_z);
        }
    }

    /// Estimate information content using spike timing precision
    fn update_information(&mut self) {
        if self.spikes.len() < 2 {
            self.information = 0.0;
            return;
        }

        // Shannon entropy from spike time distribution
        let dt = self.end_time - self.start_time;
        let n_bins = 10;
        let bin_width = dt / n_bins as f64;

        let mut bin_counts = vec![0usize; n_bins];
        for spike in &self.spikes {
            let bin = ((spike.time - self.start_time) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            bin_counts[bin] += 1;
        }

        let total = self.spikes.len() as f64;
        self.information = 0.0;

        for count in bin_counts {
            if count > 0 {
                let p = count as f64 / total;
                self.information -= p * p.log2();
            }
        }
    }

    /// Get packet duration
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Temporal chunk: higher-level aggregation of spike packets
#[derive(Debug, Clone)]
pub struct TemporalChunk {
    /// Chunk level in hierarchy (0=packets, 1=chunks, 2=phrases, etc.)
    pub level: usize,
    /// Start time
    pub start_time: f64,
    /// End time
    pub end_time: f64,
    /// Child packets or chunks
    pub children: Vec<ChunkChild>,
    /// Compressed representation
    pub representation: ChunkRepresentation,
    /// Chunk quality score (0-1)
    pub quality: f64,
}

/// Child element of a chunk (either packet or sub-chunk)
#[derive(Debug, Clone)]
pub enum ChunkChild {
    Packet(SpikePacket),
    Chunk(Box<TemporalChunk>),
}

/// Compressed representation of chunk content
#[derive(Debug, Clone)]
pub struct ChunkRepresentation {
    /// Position centroid on hyperboloid
    pub centroid: LorentzVec,
    /// Temporal signature (eigenvalues of spike correlation matrix)
    pub temporal_signature: Vec<f64>,
    /// Spatial signature (principal directions)
    pub spatial_signature: [LorentzVec; 3],
    /// Activity level (normalized spike count)
    pub activity: f64,
    /// Complexity measure (approximate entropy)
    pub complexity: f64,
    /// Confidence in chunk representation (0-1)
    pub confidence: f64,
    /// Precision (inverse variance) of the representation
    pub precision: Precision,
    /// Prediction error from higher level
    pub prediction_error: Option<ChunkPredictionError>,
}

/// Prediction error for a chunk (predictive coding)
#[derive(Debug, Clone)]
pub struct ChunkPredictionError {
    /// Spatial prediction error (hyperbolic distance from predicted centroid)
    pub spatial_error: f64,
    /// Temporal prediction error (timing deviation)
    pub temporal_error: f64,
    /// Activity prediction error
    pub activity_error: f64,
    /// Combined precision-weighted error
    pub weighted_error: f64,
    /// Precision of the prediction
    pub precision: Precision,
}

impl Default for ChunkRepresentation {
    fn default() -> Self {
        Self {
            centroid: LorentzVec::origin(),
            temporal_signature: vec![0.0; 4],
            spatial_signature: [LorentzVec::origin(); 3],
            activity: 0.0,
            complexity: 0.0,
            confidence: 0.0,
            precision: 1.0,
            prediction_error: None,
        }
    }
}

impl ChunkPredictionError {
    /// Create new prediction error
    pub fn new(
        spatial_error: f64,
        temporal_error: f64,
        activity_error: f64,
        precision: Precision,
    ) -> Self {
        // Combined error: weighted sum with spatial dominating
        let weighted_error = precision * (
            0.5 * spatial_error.powi(2) +
            0.3 * temporal_error.powi(2) +
            0.2 * activity_error.powi(2)
        ).sqrt();

        Self {
            spatial_error,
            temporal_error,
            activity_error,
            weighted_error,
            precision,
        }
    }

    /// Convert to PrecisionWeightedError for hierarchical aggregation
    pub fn to_precision_weighted(&self, level: usize, source_id: usize) -> PrecisionWeightedError {
        PrecisionWeightedError::new(
            self.weighted_error,
            self.precision,
            level,
            source_id,
        )
    }
}

use serde::{Deserialize, Serialize};

/// Chunk-and-Pass processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProcessorConfig {
    /// Number of hierarchical levels
    pub num_levels: usize,
    /// Temporal window for each level (in seconds)
    pub level_windows: Vec<f64>,
    /// Minimum spikes to form a chunk
    pub min_spikes_per_chunk: usize,
    /// Quality threshold for chunk acceptance
    pub quality_threshold: f64,
    /// Whether to use predictive chunking
    pub predictive: bool,
    /// Prediction horizon (in units of chunk window)
    pub prediction_horizon: f64,
}

impl Default for ChunkProcessorConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            level_windows: vec![
                PACKET_RESOLUTION,   // Level 0: 10ms packets
                CHUNK_RESOLUTION,    // Level 1: 100ms chunks
                PHRASE_RESOLUTION,   // Level 2: 1s phrases
                SENTENCE_RESOLUTION, // Level 3: 10s sentences
            ],
            min_spikes_per_chunk: 3,
            quality_threshold: 0.3,
            predictive: true,
            prediction_horizon: 2.0,
        }
    }
}

/// Chunk-and-Pass processor implementing hierarchical temporal chunking
pub struct ChunkProcessor {
    /// Configuration
    config: ChunkProcessorConfig,
    /// Spike buffer at each level
    level_buffers: Vec<VecDeque<SpikeEvent>>,
    /// Current packets at each level
    current_packets: Vec<Option<SpikePacket>>,
    /// Completed chunks at each level
    completed_chunks: Vec<Vec<TemporalChunk>>,
    /// Current time
    current_time: f64,
    /// Prediction state
    prediction_state: PredictionState,
    /// Statistics
    stats: ProcessorStats,
    /// Hierarchical prediction error aggregator
    error_aggregator: HierarchicalErrorAggregator,
    /// Predictions for each level (what we expect next)
    level_predictions: Vec<Option<ChunkPrediction>>,
}

/// State for predictive chunking
#[derive(Debug, Clone, Default)]
pub struct PredictionState {
    /// Predicted chunk boundaries
    predicted_boundaries: Vec<f64>,
    /// Prediction confidence
    confidence: f64,
    /// History of chunk durations for prediction
    duration_history: VecDeque<f64>,
    /// Bayesian prior for chunk duration (mean, variance)
    duration_prior: (f64, f64),
    /// Total prediction error accumulated
    total_prediction_error: f64,
    /// Number of predictions made
    prediction_count: usize,
}

/// Prediction for upcoming chunk at a level
#[derive(Debug, Clone)]
pub struct ChunkPrediction {
    /// Predicted centroid position
    pub centroid: LorentzVec,
    /// Predicted activity level
    pub activity: f64,
    /// Predicted start time
    pub start_time: f64,
    /// Predicted duration
    pub duration: f64,
    /// Precision of the prediction (higher = more confident)
    pub precision: Precision,
}

impl Default for ChunkPrediction {
    fn default() -> Self {
        Self {
            centroid: LorentzVec::origin(),
            activity: 0.5,
            start_time: 0.0,
            duration: CHUNK_RESOLUTION,
            precision: 1.0,
        }
    }
}

/// Processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessorStats {
    /// Total spikes processed
    pub total_spikes: usize,
    /// Chunks formed at each level
    pub chunks_per_level: Vec<usize>,
    /// Average chunk quality at each level
    pub avg_quality_per_level: Vec<f64>,
    /// Information throughput (bits/second)
    pub information_rate: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Total chunks formed across all levels
    pub chunks_formed: usize,
}

impl ChunkProcessor {
    /// Create new chunk processor
    pub fn new(config: ChunkProcessorConfig) -> Self {
        let num_levels = config.num_levels;

        Self {
            config,
            level_buffers: (0..num_levels).map(|_| VecDeque::new()).collect(),
            current_packets: (0..num_levels).map(|_| None).collect(),
            completed_chunks: (0..num_levels).map(|_| Vec::new()).collect(),
            current_time: 0.0,
            prediction_state: PredictionState::default(),
            stats: ProcessorStats {
                chunks_per_level: vec![0; num_levels],
                avg_quality_per_level: vec![0.0; num_levels],
                ..Default::default()
            },
            error_aggregator: HierarchicalErrorAggregator::new(num_levels),
            level_predictions: (0..num_levels).map(|_| None).collect(),
        }
    }

    /// Process incoming spike
    pub fn process_spike(&mut self, spike: SpikeEvent) {
        self.current_time = self.current_time.max(spike.time);
        self.stats.total_spikes += 1;

        // Add to level 0 buffer
        self.level_buffers[0].push_back(spike);

        // Process all levels bottom-up
        self.process_all_levels();
    }

    /// Process batch of spikes
    pub fn process_spikes(&mut self, spikes: &[SpikeEvent]) {
        for spike in spikes {
            self.process_spike(*spike);
        }
    }

    /// Process all hierarchical levels
    fn process_all_levels(&mut self) {
        for level in 0..self.config.num_levels {
            self.process_level(level);
        }
    }

    /// Process single level
    fn process_level(&mut self, level: usize) {
        let window = self.config.level_windows[level];

        // Initialize packet if needed
        if self.current_packets[level].is_none() {
            let start = if level == 0 {
                self.current_time - window
            } else {
                // Align with parent level
                let parent_window = self.config.level_windows.get(level - 1)
                    .copied().unwrap_or(window / 10.0);
                (self.current_time / parent_window).floor() * parent_window
            };
            self.current_packets[level] = Some(SpikePacket::new(start, start + window));
        }

        // Check if window has passed - extract values to avoid borrow issues
        let (should_process, packet_end_time, spike_count) = {
            let packet = self.current_packets[level].as_ref().unwrap();
            (
                self.current_time > packet.end_time,
                packet.end_time,
                packet.spike_count,
            )
        };

        if should_process {
            // Window complete - form chunk if enough data
            if spike_count >= self.config.min_spikes_per_chunk {
                // Clone packet for processing
                let packet_clone = self.current_packets[level].as_ref().unwrap().clone();
                let chunk = self.form_chunk(level, packet_clone);

                if chunk.quality >= self.config.quality_threshold {
                    self.completed_chunks[level].push(chunk.clone());
                    self.stats.chunks_per_level[level] += 1;

                    // Pass to higher level
                    if level + 1 < self.config.num_levels {
                        self.pass_chunk_up(level + 1, chunk);
                    }
                }
            }

            // Start new packet
            self.current_packets[level] = Some(SpikePacket::new(
                packet_end_time,
                packet_end_time + window,
            ));
        }

        // Add buffered spikes to current packet
        if level == 0 {
            // Get current packet end time
            let current_end_time = self.current_packets[0].as_ref()
                .map(|p| p.end_time)
                .unwrap_or(0.0);

            while let Some(spike) = self.level_buffers[0].front() {
                if spike.time <= current_end_time {
                    let spike = self.level_buffers[0].pop_front().unwrap();
                    if let Some(ref mut packet) = self.current_packets[0] {
                        packet.add_spike(spike);
                    }
                } else {
                    break;
                }
            }
        }
    }

    /// Form chunk from spike packet
    fn form_chunk(&self, level: usize, packet: SpikePacket) -> TemporalChunk {
        let representation = self.compute_representation(&packet);
        let quality = self.compute_quality(&packet, &representation);

        TemporalChunk {
            level,
            start_time: packet.start_time,
            end_time: packet.end_time,
            children: vec![ChunkChild::Packet(packet)],
            representation,
            quality,
        }
    }

    /// Compute compressed representation
    fn compute_representation(&self, packet: &SpikePacket) -> ChunkRepresentation {
        if packet.spikes.is_empty() {
            return ChunkRepresentation::default();
        }

        // Compute temporal signature via autocorrelation
        let temporal_signature = self.compute_temporal_signature(&packet.spikes);

        // Compute spatial signature via PCA on positions
        let spatial_signature = self.compute_spatial_signature(&packet.spikes);

        // Activity level
        let duration = packet.duration().max(1e-6);
        let activity = (packet.spike_count as f64 / duration).min(1000.0) / 1000.0;

        // Complexity via approximate entropy
        let complexity = self.compute_complexity(&packet.spikes);

        // Compute confidence from spike count and activity
        let confidence = self.compute_confidence(packet);

        // Compute precision as inverse variance of spike positions
        let precision = self.compute_precision(packet);

        ChunkRepresentation {
            centroid: packet.centroid,
            temporal_signature,
            spatial_signature,
            activity,
            complexity,
            confidence,
            precision,
            prediction_error: None, // Set later during prediction comparison
        }
    }

    /// Compute precision (inverse variance) from spike distribution
    fn compute_precision(&self, packet: &SpikePacket) -> Precision {
        if packet.spikes.len() < 2 {
            return 1.0; // Default precision for insufficient data
        }

        // Compute variance of distances from centroid
        let centroid = packet.centroid;
        let mut total_sq_dist = 0.0;

        for spike in &packet.spikes {
            let dist = centroid.hyperbolic_distance(&spike.position);
            total_sq_dist += dist * dist;
        }

        let variance = total_sq_dist / packet.spikes.len() as f64;

        // Precision = 1 / variance, with floor to avoid infinities
        if variance < 1e-10 {
            100.0 // Maximum precision for tightly clustered spikes
        } else {
            (1.0 / variance).min(100.0).max(0.01)
        }
    }

    /// Compute confidence in chunk representation
    fn compute_confidence(&self, packet: &SpikePacket) -> f64 {
        if packet.spike_count < self.config.min_spikes_per_chunk {
            return 0.0;
        }
        // Confidence based on spike count and duration
        let count_factor = (packet.spike_count as f64 / 10.0).min(1.0);
        let duration = packet.duration().max(1e-6);
        let rate_factor = ((packet.spike_count as f64 / duration) / 100.0).min(1.0);
        (count_factor * 0.6 + rate_factor * 0.4).clamp(0.0, 1.0)
    }

    /// Compute temporal signature from spike times
    fn compute_temporal_signature(&self, spikes: &[SpikeEvent]) -> Vec<f64> {
        if spikes.len() < 2 {
            return vec![0.0; 4];
        }

        let times: Vec<f64> = spikes.iter().map(|s| s.time).collect();
        let n = times.len();
        let mean = times.iter().sum::<f64>() / n as f64;

        // Compute first 4 moments
        let mut moments = vec![0.0f64; 4];
        for &t in &times {
            let d = t - mean;
            moments[0] += d;
            moments[1] += d * d;
            moments[2] += d * d * d;
            moments[3] += d * d * d * d;
        }

        for m in &mut moments {
            *m /= n as f64;
        }

        // Normalize
        let std = moments[1].sqrt().max(1e-10);
        moments[2] /= std * std * std;  // Skewness
        moments[3] /= std * std * std * std;  // Kurtosis

        moments
    }

    /// Compute spatial signature via hyperbolic PCA
    fn compute_spatial_signature(&self, spikes: &[SpikeEvent]) -> [LorentzVec; 3] {
        let mut signature = [LorentzVec::origin(); 3];

        if spikes.len() < 3 {
            return signature;
        }

        // Compute covariance in tangent space at centroid
        // Simplified: use spatial components directly
        let n = spikes.len() as f64;
        let mut mean = [0.0; 3];

        for spike in spikes {
            mean[0] += spike.position.x;
            mean[1] += spike.position.y;
            mean[2] += spike.position.z;
        }

        for m in &mut mean {
            *m /= n;
        }

        // Covariance matrix
        let mut cov = [[0.0f64; 3]; 3];
        for spike in spikes {
            let d = [
                spike.position.x - mean[0],
                spike.position.y - mean[1],
                spike.position.z - mean[2],
            ];
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }

        for row in &mut cov {
            for val in row {
                *val /= n;
            }
        }

        // Power iteration for top 3 eigenvectors (simplified)
        for k in 0..3 {
            let mut v = [1.0, 0.0, 0.0];
            if k == 1 { v = [0.0, 1.0, 0.0]; }
            if k == 2 { v = [0.0, 0.0, 1.0]; }

            for _ in 0..10 {
                let mut new_v = [0.0; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        new_v[i] += cov[i][j] * v[j];
                    }
                }
                let norm = (new_v[0]*new_v[0] + new_v[1]*new_v[1] + new_v[2]*new_v[2]).sqrt();
                if norm > 1e-10 {
                    for i in 0..3 {
                        v[i] = new_v[i] / norm;
                    }
                }
            }

            // Project to hyperboloid
            let spatial_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
            let t = (1.0 + spatial_sq).sqrt();
            signature[k] = LorentzVec::new(t, v[0], v[1], v[2]);
        }

        signature
    }

    /// Compute complexity via approximate entropy
    fn compute_complexity(&self, spikes: &[SpikeEvent]) -> f64 {
        if spikes.len() < 4 {
            return 0.0;
        }

        // Simplified approximate entropy
        let times: Vec<f64> = spikes.iter().map(|s| s.time).collect();
        let n = times.len();

        // Inter-spike intervals
        let mut isis: Vec<f64> = Vec::with_capacity(n - 1);
        for i in 1..n {
            isis.push(times[i] - times[i-1]);
        }

        if isis.is_empty() {
            return 0.0;
        }

        // Normalize
        let mean_isi = isis.iter().sum::<f64>() / isis.len() as f64;
        let std_isi = (isis.iter().map(|x| (x - mean_isi).powi(2)).sum::<f64>()
            / isis.len() as f64).sqrt();

        if std_isi < 1e-10 {
            return 0.0;  // Perfectly regular
        }

        // Coefficient of variation as complexity proxy
        let cv = std_isi / mean_isi.max(1e-10);

        // Map to [0, 1] - higher CV = more complex
        1.0 - (-cv).exp()
    }

    /// Compute chunk quality
    fn compute_quality(&self, packet: &SpikePacket, repr: &ChunkRepresentation) -> f64 {
        // Quality based on:
        // 1. Sufficient activity
        // 2. Spatial coherence
        // 3. Temporal structure

        let activity_score = (repr.activity * 10.0).min(1.0);

        // Spatial coherence: how clustered are spikes?
        let spatial_coherence = if packet.spikes.len() > 1 {
            let centroid = packet.centroid;
            let mut total_dist = 0.0;
            for spike in &packet.spikes {
                total_dist += centroid.hyperbolic_distance(&spike.position);
            }
            let avg_dist = total_dist / packet.spikes.len() as f64;
            (-avg_dist / 2.0).exp()  // Decay with distance
        } else {
            0.5
        };

        // Temporal structure: not too regular, not too random
        let temporal_score = if repr.complexity > 0.1 && repr.complexity < 0.9 {
            1.0 - (repr.complexity - 0.5).abs() * 2.0
        } else {
            0.3
        };

        // Weighted combination
        0.4 * activity_score + 0.3 * spatial_coherence + 0.3 * temporal_score
    }

    /// Pass chunk to higher level
    fn pass_chunk_up(&mut self, target_level: usize, chunk: TemporalChunk) {
        // Convert chunk to "spike" for higher level
        let spike = SpikeEvent {
            time: (chunk.start_time + chunk.end_time) / 2.0,
            neuron_id: target_level * 10000 + self.stats.chunks_per_level[target_level - 1],
            position: chunk.representation.centroid,
            amplitude: chunk.quality,
        };

        // Add to higher level buffer
        if target_level < self.level_buffers.len() {
            self.level_buffers[target_level].push_back(spike);
        }
    }

    /// Get completed chunks at specified level
    pub fn get_chunks(&self, level: usize) -> &[TemporalChunk] {
        if level < self.completed_chunks.len() {
            &self.completed_chunks[level]
        } else {
            &[]
        }
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ProcessorStats {
        &self.stats
    }

    /// Update prediction state
    pub fn update_predictions(&mut self) {
        if !self.config.predictive {
            return;
        }

        // Update duration history
        if let Some(chunk) = self.completed_chunks[0].last() {
            let duration = chunk.end_time - chunk.start_time;
            self.prediction_state.duration_history.push_back(duration);

            // Keep limited history
            while self.prediction_state.duration_history.len() > 100 {
                self.prediction_state.duration_history.pop_front();
            }
        }

        // Update Bayesian prior
        if !self.prediction_state.duration_history.is_empty() {
            let durations: Vec<f64> = self.prediction_state.duration_history.iter().copied().collect();
            let mean = durations.iter().sum::<f64>() / durations.len() as f64;
            let variance = durations.iter()
                .map(|d| (d - mean).powi(2))
                .sum::<f64>() / durations.len() as f64;

            self.prediction_state.duration_prior = (mean, variance);
            self.prediction_state.confidence = 1.0 / (1.0 + variance.sqrt() / mean.max(1e-10));
        }

        // Predict next boundary
        let (mean_dur, _) = self.prediction_state.duration_prior;
        let next_boundary = self.current_time + mean_dur * self.config.prediction_horizon;
        self.prediction_state.predicted_boundaries = vec![next_boundary];
    }

    /// Get predicted chunk boundaries
    pub fn predicted_boundaries(&self) -> &[f64] {
        &self.prediction_state.predicted_boundaries
    }

    /// Get prediction confidence
    pub fn prediction_confidence(&self) -> f64 {
        self.prediction_state.confidence
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        for buffer in &mut self.level_buffers {
            buffer.clear();
        }
        for packet in &mut self.current_packets {
            *packet = None;
        }
        for chunks in &mut self.completed_chunks {
            chunks.clear();
        }
        self.current_time = 0.0;
        self.prediction_state = PredictionState::default();
        self.stats = ProcessorStats {
            chunks_per_level: vec![0; self.config.num_levels],
            avg_quality_per_level: vec![0.0; self.config.num_levels],
            ..Default::default()
        };
        self.error_aggregator.clear();
        for pred in &mut self.level_predictions {
            *pred = None;
        }
    }

    // ========================================================================
    // Precision-Weighted Prediction Error Methods
    // ========================================================================

    /// Compute prediction error for a newly formed chunk
    ///
    /// Compares the actual chunk to the prediction at that level,
    /// producing precision-weighted error signals for hierarchical propagation.
    pub fn compute_prediction_error(
        &mut self,
        level: usize,
        chunk: &TemporalChunk,
    ) -> Option<ChunkPredictionError> {
        // Get prediction for this level
        let prediction = self.level_predictions[level].as_ref()?;

        // Spatial error: hyperbolic distance from predicted centroid
        let spatial_error = prediction.centroid.hyperbolic_distance(&chunk.representation.centroid);

        // Temporal error: deviation from predicted timing
        let actual_start = chunk.start_time;
        let temporal_error = (actual_start - prediction.start_time).abs() / prediction.duration.max(1e-6);

        // Activity error: deviation from predicted activity
        let activity_error = (chunk.representation.activity - prediction.activity).abs();

        // Combine into prediction error
        let error = ChunkPredictionError::new(
            spatial_error,
            temporal_error,
            activity_error,
            prediction.precision,
        );

        // Add to error aggregator
        let pw_error = error.to_precision_weighted(level, self.stats.chunks_per_level[level]);
        self.error_aggregator.add_error(pw_error);

        // Update prediction state statistics
        self.prediction_state.total_prediction_error += error.weighted_error;
        self.prediction_state.prediction_count += 1;

        Some(error)
    }

    /// Generate prediction for next chunk at a level
    ///
    /// Uses completed chunks to predict the next chunk's properties.
    /// Precision of prediction increases with more consistent history.
    pub fn generate_prediction(&mut self, level: usize) {
        let chunks = &self.completed_chunks[level];

        if chunks.is_empty() {
            // Default prediction: origin with low precision
            self.level_predictions[level] = Some(ChunkPrediction::default());
            return;
        }

        // Use recent chunks to predict next
        let recent: Vec<&TemporalChunk> = chunks.iter().rev().take(10).collect();

        // Predict centroid: exponentially weighted average
        let mut pred_x = 0.0_f64;
        let mut pred_y = 0.0_f64;
        let mut pred_z = 0.0_f64;
        let mut total_weight = 0.0_f64;
        let decay: f64 = 0.8;

        for (i, chunk) in recent.iter().enumerate() {
            let w = decay.powi(i as i32);
            pred_x += w * chunk.representation.centroid.x;
            pred_y += w * chunk.representation.centroid.y;
            pred_z += w * chunk.representation.centroid.z;
            total_weight += w;
        }

        let pred_centroid = if total_weight > 1e-10 {
            let x = pred_x / total_weight;
            let y = pred_y / total_weight;
            let z = pred_z / total_weight;
            let t = (1.0_f64 + x*x + y*y + z*z).sqrt();
            LorentzVec::new(t, x, y, z)
        } else {
            LorentzVec::origin()
        };

        // Predict activity: weighted average
        let pred_activity: f64 = recent.iter()
            .enumerate()
            .map(|(i, c)| decay.powi(i as i32) * c.representation.activity)
            .sum::<f64>() / total_weight.max(1e-10);

        // Predict timing
        let last_chunk = chunks.last().unwrap();
        let pred_start = last_chunk.end_time;
        let pred_duration = self.config.level_windows[level];

        // Compute precision from consistency of recent chunks
        let precision = self.compute_prediction_precision(&recent);

        self.level_predictions[level] = Some(ChunkPrediction {
            centroid: pred_centroid,
            activity: pred_activity,
            start_time: pred_start,
            duration: pred_duration,
            precision,
        });
    }

    /// Compute prediction precision from chunk history
    fn compute_prediction_precision(&self, chunks: &[&TemporalChunk]) -> Precision {
        if chunks.len() < 2 {
            return 1.0;
        }

        // Compute variance of centroids
        let n = chunks.len() as f64;
        let mut mean_x = 0.0;
        let mut mean_y = 0.0;
        let mut mean_z = 0.0;

        for chunk in chunks {
            mean_x += chunk.representation.centroid.x;
            mean_y += chunk.representation.centroid.y;
            mean_z += chunk.representation.centroid.z;
        }
        mean_x /= n;
        mean_y /= n;
        mean_z /= n;

        let mut variance = 0.0;
        for chunk in chunks {
            let dx = chunk.representation.centroid.x - mean_x;
            let dy = chunk.representation.centroid.y - mean_y;
            let dz = chunk.representation.centroid.z - mean_z;
            variance += dx*dx + dy*dy + dz*dz;
        }
        variance /= n;

        // Precision = 1/variance, bounded
        if variance < 1e-10 {
            10.0
        } else {
            (1.0 / variance).min(10.0).max(0.1)
        }
    }

    /// Propagate prediction errors through hierarchy
    ///
    /// Bottom-up: aggregate errors from lower levels
    /// Top-down: adjust predictions based on higher-level context
    pub fn propagate_errors(&mut self) {
        self.error_aggregator.propagate();
    }

    /// Get total prediction error at a level
    pub fn total_error_at_level(&self, level: usize) -> f64 {
        self.error_aggregator.total_error_at_level(level)
    }

    /// Get average precision at a level
    pub fn average_precision_at_level(&self, level: usize) -> Precision {
        self.error_aggregator.average_precision_at_level(level)
    }

    /// Get mean prediction error across all levels
    pub fn mean_prediction_error(&self) -> f64 {
        if self.prediction_state.prediction_count == 0 {
            return 0.0;
        }
        self.prediction_state.total_prediction_error / self.prediction_state.prediction_count as f64
    }

    /// Get prediction for a level
    pub fn get_prediction(&self, level: usize) -> Option<&ChunkPrediction> {
        self.level_predictions.get(level).and_then(|p| p.as_ref())
    }

    /// Get the hierarchical error aggregator
    pub fn error_aggregator(&self) -> &HierarchicalErrorAggregator {
        &self.error_aggregator
    }

    // ========================================================================
    // Phase 5: Language Creation Integration Methods
    // ========================================================================

    /// Scale all temporal windows by a factor
    ///
    /// Used by acquisition-processing constraint bridge to adjust processing
    /// windows based on learned construction properties.
    pub fn scale_windows(&mut self, factor: f64) {
        for window in self.config.level_windows.iter_mut() {
            *window *= factor.clamp(0.1, 10.0); // Bounded scaling
        }
    }

    /// Drain all completed chunks from all levels
    ///
    /// Returns all completed chunks and clears the internal buffers.
    /// Used for processing-to-acquisition constraint flow.
    pub fn drain_completed_chunks(&mut self) -> Vec<TemporalChunk> {
        let mut all_chunks = Vec::new();
        for level_chunks in &mut self.completed_chunks {
            all_chunks.extend(level_chunks.drain(..));
        }
        all_chunks
    }

    /// Advance time without processing new spikes
    ///
    /// Triggers level processing based on time passage.
    /// Used when stepping the tri-directional system without new input.
    pub fn advance_time(&mut self, dt: f64) {
        self.current_time += dt;
        self.process_all_levels();
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get reference to configuration
    pub fn config(&self) -> &ChunkProcessorConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut ChunkProcessorConfig {
        &mut self.config
    }
}

/// Now-or-Never bottleneck enforcer
/// Forces rapid compression of input to prevent information loss
pub struct NowOrNeverBottleneck {
    /// Maximum time input can wait (in seconds)
    max_delay: f64,
    /// Compression ratio target
    target_compression: f64,
    /// Input buffer
    input_buffer: VecDeque<(f64, LorentzVec, f64)>,  // (time, position, value)
    /// Output queue
    output_queue: VecDeque<CompressedUnit>,
}

/// Compressed unit after bottleneck processing
#[derive(Debug, Clone)]
pub struct CompressedUnit {
    /// Time of compression
    pub time: f64,
    /// Compressed position
    pub position: LorentzVec,
    /// Compressed value
    pub value: f64,
    /// Compression loss estimate
    pub loss: f64,
    /// Number of inputs compressed
    pub input_count: usize,
}

impl NowOrNeverBottleneck {
    /// Create new bottleneck
    pub fn new(max_delay: f64, target_compression: f64) -> Self {
        Self {
            max_delay,
            target_compression,
            input_buffer: VecDeque::new(),
            output_queue: VecDeque::new(),
        }
    }

    /// Process input through bottleneck
    pub fn process(&mut self, time: f64, position: LorentzVec, value: f64) {
        self.input_buffer.push_back((time, position, value));

        // Force compression if buffer too old
        while let Some(&(oldest_time, _, _)) = self.input_buffer.front() {
            if time - oldest_time > self.max_delay {
                self.compress();
            } else {
                break;
            }
        }

        // Also compress if buffer too large
        let target_size = (1.0 / self.target_compression).ceil() as usize;
        while self.input_buffer.len() > target_size * 2 {
            self.compress();
        }
    }

    /// Force compression of buffer
    fn compress(&mut self) {
        if self.input_buffer.is_empty() {
            return;
        }

        let target_size = (1.0 / self.target_compression).ceil() as usize;
        let compress_count = self.input_buffer.len().min(target_size).max(1);

        // Extract items to compress
        let mut items = Vec::with_capacity(compress_count);
        for _ in 0..compress_count {
            if let Some(item) = self.input_buffer.pop_front() {
                items.push(item);
            }
        }

        if items.is_empty() {
            return;
        }

        // Compute compressed representation
        let mut _sum_t = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        let mut sum_val = 0.0;
        let mut max_time = 0.0f64;

        for (t, pos, val) in &items {
            max_time = max_time.max(*t);
            sum_val += val;
            _sum_t += pos.t * val;
            sum_x += pos.x * val;
            sum_y += pos.y * val;
            sum_z += pos.z * val;
        }

        let total_val = sum_val.max(1e-10);

        // Weighted average position
        let avg_x = sum_x / total_val;
        let avg_y = sum_y / total_val;
        let avg_z = sum_z / total_val;
        let spatial_sq = avg_x * avg_x + avg_y * avg_y + avg_z * avg_z;
        let t_coord = (1.0 + spatial_sq).sqrt();

        let position = LorentzVec::new(t_coord, avg_x, avg_y, avg_z);

        // Estimate compression loss
        let mut loss = 0.0;
        for (_, pos, val) in &items {
            loss += val * position.hyperbolic_distance(pos);
        }
        loss /= total_val;

        let unit = CompressedUnit {
            time: max_time,
            position,
            value: sum_val / items.len() as f64,
            loss,
            input_count: items.len(),
        };

        self.output_queue.push_back(unit);
    }

    /// Get next compressed unit
    pub fn pop(&mut self) -> Option<CompressedUnit> {
        self.output_queue.pop_front()
    }

    /// Check if output available
    pub fn has_output(&self) -> bool {
        !self.output_queue.is_empty()
    }

    /// Flush all remaining input
    pub fn flush(&mut self) {
        while !self.input_buffer.is_empty() {
            self.compress();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_spike(time: f64, neuron_id: usize) -> SpikeEvent {
        SpikeEvent {
            time,
            neuron_id,
            position: LorentzVec::origin(),
            amplitude: 1.0,
        }
    }

    #[test]
    fn test_spike_packet_creation() {
        let mut packet = SpikePacket::new(0.0, 0.01);

        packet.add_spike(create_test_spike(0.001, 0));
        packet.add_spike(create_test_spike(0.005, 1));
        packet.add_spike(create_test_spike(0.008, 2));

        assert_eq!(packet.spike_count, 3);
        assert!(packet.information >= 0.0);
    }

    #[test]
    fn test_chunk_processor_basic() {
        let config = ChunkProcessorConfig {
            num_levels: 2,
            level_windows: vec![0.01, 0.1],
            min_spikes_per_chunk: 2,
            quality_threshold: 0.0,
            predictive: false,
            prediction_horizon: 1.0,
        };

        let mut processor = ChunkProcessor::new(config);

        // Generate spikes
        for i in 0..20 {
            let spike = SpikeEvent {
                time: i as f64 * 0.005,
                neuron_id: i % 5,
                position: LorentzVec::new(1.1, 0.1 * (i as f64).sin(), 0.1 * (i as f64).cos(), 0.0),
                amplitude: 1.0,
            };
            processor.process_spike(spike);
        }

        assert!(processor.stats().total_spikes == 20);
    }

    #[test]
    fn test_now_or_never_bottleneck() {
        let mut bottleneck = NowOrNeverBottleneck::new(0.1, 0.5);

        // Add inputs
        for i in 0..10 {
            let t = i as f64 * 0.02;
            let pos = LorentzVec::new(1.1, 0.1 * i as f64, 0.0, 0.0);
            bottleneck.process(t, pos, 1.0);
        }

        bottleneck.flush();

        // Should have compressed outputs
        let mut output_count = 0;
        while bottleneck.pop().is_some() {
            output_count += 1;
        }

        assert!(output_count > 0);
        assert!(output_count < 10);  // Should be compressed
    }

    #[test]
    fn test_chunk_quality() {
        let mut packet = SpikePacket::new(0.0, 0.1);

        // Add coherent spikes (clustered in space and time)
        for i in 0..10 {
            packet.add_spike(SpikeEvent {
                time: 0.05 + 0.001 * i as f64,  // Clustered around 50ms
                neuron_id: i,
                position: LorentzVec::new(1.01, 0.05, 0.05, 0.0),  // Close together
                amplitude: 1.0,
            });
        }

        let processor = ChunkProcessor::new(ChunkProcessorConfig::default());
        let repr = processor.compute_representation(&packet);
        let quality = processor.compute_quality(&packet, &repr);

        // Coherent packet should have reasonable quality
        assert!(quality > 0.2, "Quality was {}", quality);
    }

    #[test]
    fn test_hierarchical_chunking() {
        let config = ChunkProcessorConfig {
            num_levels: 3,
            level_windows: vec![0.01, 0.05, 0.25],
            min_spikes_per_chunk: 3,
            quality_threshold: 0.1,
            predictive: true,
            prediction_horizon: 2.0,
        };

        let mut processor = ChunkProcessor::new(config);

        // Generate spikes over longer period
        for i in 0..100 {
            let spike = SpikeEvent {
                time: i as f64 * 0.005,
                neuron_id: i % 10,
                position: LorentzVec::new(
                    1.0 + 0.1 * (i as f64 / 10.0).sin(),
                    0.2 * (i as f64 / 5.0).sin(),
                    0.2 * (i as f64 / 7.0).cos(),
                    0.0
                ),
                amplitude: 0.5 + 0.5 * (i as f64 / 20.0).sin().abs(),
            };
            processor.process_spike(spike);
        }

        // Should have chunks at multiple levels
        let stats = processor.stats();
        assert!(stats.total_spikes == 100);

        // Level 0 should have most chunks
        // (exact numbers depend on parameters)
    }
}
