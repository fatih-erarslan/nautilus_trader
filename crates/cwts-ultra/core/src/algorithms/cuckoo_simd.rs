//! SIMD-optimized whale detection using Cuckoo Hashing algorithm
//!
//! This module implements a high-performance whale detection system using SIMD
//! operations and cuckoo hashing for O(1) lookups with minimal collision handling.

use std::arch::x86_64::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
// Removed unused rayon import - not needed for current SIMD implementation

/// SIMD vector size for AVX2 operations
const SIMD_WIDTH: usize = 8; // 8 x f64 elements per AVX2 vector
const CUCKOO_TABLE_SIZE: usize = 4096; // Power of 2 for efficient modulo
const MAX_CUCKOO_ITERATIONS: u32 = 8;
const WHALE_THRESHOLD_MULTIPLIER: f64 = 5.0; // 5x average volume
const VOLUME_DECAY_FACTOR: f64 = 0.95;

/// Cuckoo hash table entry for whale tracking
#[repr(C, align(32))] // AVX2 alignment
#[derive(Clone)]
pub struct WhaleEntry {
    pub symbol_hash: u64,       // Symbol hash for identification
    pub cumulative_volume: f64, // Total volume accumulated
    pub last_timestamp: u64,    // Last activity timestamp
    pub transaction_count: u32, // Number of transactions
    pub velocity_score: f32,    // Volume velocity indicator
    pub risk_score: f32,        // Whale risk assessment
    pub is_active: bool,        // Entry validity flag
    pub padding: [u8; 7],       // Alignment padding
}

impl Default for WhaleEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl WhaleEntry {
    pub fn new() -> Self {
        Self {
            symbol_hash: 0,
            cumulative_volume: 0.0,
            last_timestamp: 0,
            transaction_count: 0,
            velocity_score: 0.0,
            risk_score: 0.0,
            is_active: false,
            padding: [0; 7],
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// SIMD-optimized whale detector with cuckoo hashing
pub struct SimdWhaleDetector {
    // Dual cuckoo hash tables for efficient lookups
    table1: Vec<WhaleEntry>,
    table2: Vec<WhaleEntry>,

    // Statistical tracking
    volume_history: VecDeque<f64>,
    timestamp_history: VecDeque<u64>,
    moving_average: f64,
    volume_std_dev: f64,

    // SIMD-aligned buffers for batch processing
    volume_buffer: Vec<f64>,
    timestamp_buffer: Vec<u64>,
    hash_buffer: Vec<u64>,

    // Performance counters
    total_detections: AtomicU64,
    false_positives: AtomicU64,
    hash_collisions: AtomicU64,

    // Configuration
    min_whale_volume: f64,
    detection_window: usize,
    hash_seed1: u64,
    hash_seed2: u64,
}

impl SimdWhaleDetector {
    /// Create new SIMD whale detector with specified parameters
    pub fn new(min_whale_volume: f64, detection_window: usize) -> Self {
        let mut detector = Self {
            table1: vec![WhaleEntry::new(); CUCKOO_TABLE_SIZE],
            table2: vec![WhaleEntry::new(); CUCKOO_TABLE_SIZE],
            volume_history: VecDeque::with_capacity(detection_window),
            timestamp_history: VecDeque::with_capacity(detection_window),
            moving_average: 0.0,
            volume_std_dev: 0.0,
            volume_buffer: Vec::with_capacity(SIMD_WIDTH * 16),
            timestamp_buffer: Vec::with_capacity(SIMD_WIDTH * 16),
            hash_buffer: Vec::with_capacity(SIMD_WIDTH * 16),
            total_detections: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            hash_collisions: AtomicU64::new(0),
            min_whale_volume,
            detection_window,
            hash_seed1: 0x517cc1b727220a95, // Random seed for hash function 1
            hash_seed2: 0x85ebca6b63b3f5c9, // Random seed for hash function 2
        };

        detector.initialize_simd_buffers();
        detector
    }

    /// Detect whale movements using SIMD-optimized algorithms
    pub fn detect_whales_simd(
        &mut self,
        symbol: &str,
        transactions: &[(f64, u64)],
    ) -> Vec<WhaleMovement> {
        if transactions.is_empty() {
            return Vec::new();
        }

        let symbol_hash = self.hash_symbol(symbol);
        let mut whale_movements = Vec::new();

        // Update statistical baseline
        self.update_volume_statistics(transactions);

        // Process transactions in SIMD-optimized batches
        for chunk in transactions.chunks(SIMD_WIDTH) {
            let batch_whales = self.process_transaction_batch_simd(symbol, symbol_hash, chunk);
            whale_movements.extend(batch_whales);
        }

        // Age out old entries
        self.age_whale_entries();

        whale_movements
    }

    /// Process transaction batch using SIMD operations
    fn process_transaction_batch_simd(
        &mut self,
        symbol: &str,
        symbol_hash: u64,
        transactions: &[(f64, u64)],
    ) -> Vec<WhaleMovement> {
        if transactions.is_empty() {
            return Vec::new();
        }

        // Prepare SIMD-aligned data
        self.volume_buffer.clear();
        self.timestamp_buffer.clear();

        for (volume, timestamp) in transactions {
            self.volume_buffer.push(*volume);
            self.timestamp_buffer.push(*timestamp);
        }

        // Pad to SIMD width
        while self.volume_buffer.len() % SIMD_WIDTH != 0 {
            self.volume_buffer.push(0.0_f64);
            self.timestamp_buffer.push(0_u64);
        }

        let mut whale_movements = Vec::new();

        // Process using SIMD intrinsics (when available)
        if is_x86_feature_detected!("avx2") {
            unsafe {
                whale_movements.extend(self.simd_whale_detection_avx2(symbol, symbol_hash));
            }
        } else {
            // Fallback to scalar processing
            whale_movements.extend(self.scalar_whale_detection(symbol, symbol_hash, transactions));
        }

        whale_movements
    }

    /// SIMD whale detection using AVX2 instructions
    #[target_feature(enable = "avx2")]
    unsafe fn simd_whale_detection_avx2(
        &mut self,
        symbol: &str,
        symbol_hash: u64,
    ) -> Vec<WhaleMovement> {
        let mut whale_movements = Vec::new();

        // Load threshold values into SIMD registers
        let threshold_vec = _mm256_set1_pd(self.min_whale_volume);
        let avg_vec = _mm256_set1_pd(self.moving_average);
        let multiplier_vec = _mm256_set1_pd(WHALE_THRESHOLD_MULTIPLIER);

        // Process volumes in chunks of 4 (AVX2 can process 4 double-precision floats)
        for i in (0..self.volume_buffer.len()).step_by(4) {
            if i + 4 > self.volume_buffer.len() {
                break;
            }

            // Load 4 volumes into SIMD register
            let volumes = _mm256_loadu_pd(self.volume_buffer[i..].as_ptr());

            // Calculate dynamic threshold: avg * multiplier
            let dynamic_threshold = _mm256_mul_pd(avg_vec, multiplier_vec);

            // Compare volumes against both static and dynamic thresholds
            let static_mask = _mm256_cmp_pd(volumes, threshold_vec, _CMP_GT_OQ);
            let dynamic_mask = _mm256_cmp_pd(volumes, dynamic_threshold, _CMP_GT_OQ);

            // Combine masks (whale if above either threshold)
            let whale_mask = _mm256_or_pd(static_mask, dynamic_mask);

            // Extract mask to check which elements are whales
            let mask_bits = _mm256_movemask_pd(whale_mask);

            for j in 0..4 {
                if (mask_bits & (1 << j)) != 0 {
                    let volume = self.volume_buffer[i + j];
                    let timestamp = self.timestamp_buffer[i + j];

                    if let Some(whale_movement) =
                        self.validate_and_create_whale(symbol, symbol_hash, volume, timestamp)
                    {
                        whale_movements.push(whale_movement);
                    }
                }
            }
        }

        whale_movements
    }

    /// Scalar fallback whale detection
    fn scalar_whale_detection(
        &mut self,
        symbol: &str,
        symbol_hash: u64,
        transactions: &[(f64, u64)],
    ) -> Vec<WhaleMovement> {
        let mut whale_movements = Vec::new();
        let dynamic_threshold = self.moving_average * WHALE_THRESHOLD_MULTIPLIER;

        for &(volume, timestamp) in transactions {
            let is_whale = volume > self.min_whale_volume || volume > dynamic_threshold;

            if is_whale {
                if let Some(whale_movement) =
                    self.validate_and_create_whale(symbol, symbol_hash, volume, timestamp)
                {
                    whale_movements.push(whale_movement);
                }
            }
        }

        whale_movements
    }

    /// Validate whale candidate and create WhaleMovement if confirmed
    fn validate_and_create_whale(
        &mut self,
        symbol: &str,
        symbol_hash: u64,
        volume: f64,
        timestamp: u64,
    ) -> Option<WhaleMovement> {
        // Calculate velocity score
        let velocity_score = self.calculate_velocity_score(volume, timestamp);

        // Calculate risk score based on historical patterns
        let risk_score = self.calculate_risk_score(symbol_hash, volume, velocity_score);

        // Apply additional filters to reduce false positives
        if self.apply_whale_filters(volume, velocity_score, risk_score) {
            // Update or insert whale entry in cuckoo tables
            self.update_whale_entry(symbol_hash, volume, timestamp, velocity_score, risk_score);

            self.total_detections.fetch_add(1, Ordering::Relaxed);

            Some(WhaleMovement {
                symbol: symbol.to_string(),
                volume,
                timestamp,
                velocity_score,
                risk_score,
                detection_confidence: self
                    .calculate_detection_confidence(volume, velocity_score as f64)
                    as f32,
                pattern_type: self.classify_whale_pattern(
                    volume,
                    velocity_score as f64,
                    risk_score as f64,
                ),
            })
        } else {
            self.false_positives.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Update whale entry using cuckoo hashing
    fn update_whale_entry(
        &mut self,
        symbol_hash: u64,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        let hash1 = self.cuckoo_hash1(symbol_hash);
        let hash2 = self.cuckoo_hash2(symbol_hash);

        let entry1 = &mut self.table1[hash1];
        let entry2 = &mut self.table2[hash2];

        // Try to update existing entry first
        if entry1.symbol_hash == symbol_hash && entry1.is_active {
            Self::update_existing_entry_static(
                entry1,
                volume,
                timestamp,
                velocity_score,
                risk_score,
            );
            return;
        }

        if entry2.symbol_hash == symbol_hash && entry2.is_active {
            Self::update_existing_entry_static(
                entry2,
                volume,
                timestamp,
                velocity_score,
                risk_score,
            );
            return;
        }

        // Insert new entry
        if !entry1.is_active {
            Self::create_new_entry_static(
                entry1,
                symbol_hash,
                volume,
                timestamp,
                velocity_score,
                risk_score,
            );
        } else if !entry2.is_active {
            Self::create_new_entry_static(
                entry2,
                symbol_hash,
                volume,
                timestamp,
                velocity_score,
                risk_score,
            );
        } else {
            // Both slots occupied - perform cuckoo eviction
            self.perform_cuckoo_eviction(
                symbol_hash,
                volume,
                timestamp,
                velocity_score,
                risk_score,
            );
        }
    }

    /// Perform cuckoo eviction when both hash positions are occupied
    fn perform_cuckoo_eviction(
        &mut self,
        symbol_hash: u64,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        let mut current_hash = symbol_hash;
        let mut current_volume = volume;
        let mut current_timestamp = timestamp;
        let mut current_velocity = velocity_score;
        let mut current_risk = risk_score;

        for _ in 0..MAX_CUCKOO_ITERATIONS {
            let hash1 = self.cuckoo_hash1(current_hash);
            let evicted_entry = self.table1[hash1].clone();

            // Insert current item into table1
            Self::create_new_entry_static(
                &mut self.table1[hash1],
                current_hash,
                current_volume,
                current_timestamp,
                current_velocity,
                current_risk,
            );

            if !evicted_entry.is_active {
                return; // Successfully inserted
            }

            // Move evicted item to table2
            let hash2 = self.cuckoo_hash2(evicted_entry.symbol_hash);
            let evicted_entry2 = self.table2[hash2].clone();

            self.table2[hash2] = evicted_entry;

            if !evicted_entry2.is_active {
                return; // Successfully inserted
            }

            // Prepare for next iteration
            current_hash = evicted_entry2.symbol_hash;
            current_volume = evicted_entry2.cumulative_volume;
            current_timestamp = evicted_entry2.last_timestamp;
            current_velocity = evicted_entry2.velocity_score;
            current_risk = evicted_entry2.risk_score;
        }

        // Failed to insert after max iterations
        self.hash_collisions.fetch_add(1, Ordering::Relaxed);
    }

    /// Create new whale entry - wrapper for compatibility
    #[allow(dead_code)]
    fn create_new_entry(
        &mut self,
        entry: &mut WhaleEntry,
        symbol_hash: u64,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        Self::create_new_entry_static(
            entry,
            symbol_hash,
            volume,
            timestamp,
            velocity_score,
            risk_score,
        );
    }

    /// Static helper for creating new whale entry (to avoid borrow checker issues)
    fn create_new_entry_static(
        entry: &mut WhaleEntry,
        symbol_hash: u64,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        entry.symbol_hash = symbol_hash;
        entry.cumulative_volume = volume;
        entry.last_timestamp = timestamp;
        entry.transaction_count = 1;
        entry.velocity_score = velocity_score;
        entry.risk_score = risk_score;
        entry.is_active = true;
    }

    /// Update existing whale entry - wrapper for compatibility
    #[allow(dead_code)]
    fn update_existing_entry(
        &mut self,
        entry: &mut WhaleEntry,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        Self::update_existing_entry_static(entry, volume, timestamp, velocity_score, risk_score);
    }

    /// Static helper for updating existing whale entry (to avoid borrow checker issues)
    fn update_existing_entry_static(
        entry: &mut WhaleEntry,
        volume: f64,
        timestamp: u64,
        velocity_score: f32,
        risk_score: f32,
    ) {
        entry.cumulative_volume += volume;
        entry.last_timestamp = timestamp;
        entry.transaction_count += 1;
        entry.velocity_score = (entry.velocity_score + velocity_score) / 2.0; // Moving average
        entry.risk_score = entry.risk_score.max(risk_score); // Take maximum risk
    }

    /// Calculate velocity score based on transaction timing
    fn calculate_velocity_score(&self, volume: f64, timestamp: u64) -> f32 {
        if self.timestamp_history.is_empty() {
            return 1.0;
        }

        let last_timestamp = *self.timestamp_history.back().unwrap();
        let time_diff = if timestamp > last_timestamp {
            timestamp - last_timestamp
        } else {
            1 // Avoid division by zero
        };

        let velocity = volume / (time_diff as f64 / 1_000_000.0); // Volume per second
        let normalized_velocity = (velocity / self.moving_average).min(10.0); // Cap at 10x

        normalized_velocity as f32
    }

    /// Calculate risk score based on patterns
    fn calculate_risk_score(&self, symbol_hash: u64, volume: f64, velocity_score: f32) -> f32 {
        let volume_risk = (volume / self.min_whale_volume).min(5.0) / 5.0; // 0-1 scale
        let velocity_risk = (velocity_score / 5.0).min(1.0) as f64; // 0-1 scale
        let frequency_risk = self.get_frequency_risk(symbol_hash);

        // Weighted combination
        (volume_risk * 0.4 + velocity_risk * 0.4 + frequency_risk * 0.2) as f32
    }

    /// Get frequency risk based on historical activity
    fn get_frequency_risk(&self, symbol_hash: u64) -> f64 {
        let hash1 = self.cuckoo_hash1(symbol_hash);
        let hash2 = self.cuckoo_hash2(symbol_hash);

        let entry1 = &self.table1[hash1];
        let entry2 = &self.table2[hash2];

        let transaction_count = if entry1.symbol_hash == symbol_hash && entry1.is_active {
            entry1.transaction_count
        } else if entry2.symbol_hash == symbol_hash && entry2.is_active {
            entry2.transaction_count
        } else {
            0
        };

        (transaction_count as f64 / 100.0).min(1.0) // Normalize to 0-1
    }

    /// Apply filters to reduce false positives
    fn apply_whale_filters(&self, volume: f64, velocity_score: f32, risk_score: f32) -> bool {
        // Minimum volume threshold
        if volume < self.min_whale_volume {
            return false;
        }

        // Velocity sanity check
        if velocity_score > 100.0 {
            return false; // Unrealistic velocity
        }

        // Risk score threshold
        if risk_score < 0.1 {
            return false; // Too low risk to be significant
        }

        // Statistical significance check
        let z_score = (volume - self.moving_average) / self.volume_std_dev.max(1.0);
        if z_score < 2.0 {
            return false; // Not statistically significant
        }

        true
    }

    /// Calculate detection confidence
    fn calculate_detection_confidence(&self, volume: f64, velocity_score: f64) -> f64 {
        let volume_confidence = ((volume / self.moving_average) / 10.0).min(1.0);
        let velocity_confidence = (velocity_score / 5.0).min(1.0);
        let statistical_confidence = if self.volume_std_dev > 0.0 {
            let z_score = (volume - self.moving_average) / self.volume_std_dev;
            (z_score / 5.0).min(1.0).max(0.0)
        } else {
            0.5
        };

        (volume_confidence + velocity_confidence + statistical_confidence) / 3.0_f64
    }

    /// Classify whale pattern type
    fn classify_whale_pattern(
        &self,
        volume: f64,
        velocity_score: f64,
        risk_score: f64,
    ) -> WhalePatternType {
        if velocity_score > 3.0 && risk_score > 0.7 {
            WhalePatternType::Aggressive
        } else if volume > self.moving_average * 10.0 {
            WhalePatternType::Accumulation
        } else if velocity_score > 2.0 {
            WhalePatternType::Momentum
        } else {
            WhalePatternType::Gradual
        }
    }

    /// Update volume statistics for baseline calculation
    fn update_volume_statistics(&mut self, transactions: &[(f64, u64)]) {
        for &(volume, timestamp) in transactions {
            // Add to history
            self.volume_history.push_back(volume);
            self.timestamp_history.push_back(timestamp);

            // Maintain window size
            if self.volume_history.len() > self.detection_window {
                self.volume_history.pop_front();
                self.timestamp_history.pop_front();
            }
        }

        // Recalculate statistics
        if !self.volume_history.is_empty() {
            self.moving_average =
                self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;

            let variance = self
                .volume_history
                .iter()
                .map(|v| (v - self.moving_average).powi(2))
                .sum::<f64>()
                / self.volume_history.len() as f64;

            self.volume_std_dev = variance.sqrt();
        }
    }

    /// Age out old whale entries
    fn age_whale_entries(&mut self) {
        let current_time = self.get_timestamp_us();
        let age_threshold = 300_000_000; // 5 minutes in microseconds

        for entry in self.table1.iter_mut().chain(self.table2.iter_mut()) {
            if entry.is_active && current_time - entry.last_timestamp > age_threshold {
                // Apply decay to volume
                entry.cumulative_volume *= VOLUME_DECAY_FACTOR;

                // Deactivate if volume becomes too small
                if entry.cumulative_volume < self.min_whale_volume * 0.1 {
                    entry.reset();
                }
            }
        }
    }

    /// Hash functions for cuckoo hashing
    fn cuckoo_hash1(&self, key: u64) -> usize {
        let hash = key.wrapping_mul(self.hash_seed1);
        (hash as usize) & (CUCKOO_TABLE_SIZE - 1)
    }

    fn cuckoo_hash2(&self, key: u64) -> usize {
        let hash = key.wrapping_mul(self.hash_seed2);
        (hash as usize) & (CUCKOO_TABLE_SIZE - 1)
    }

    /// Hash symbol string to u64
    fn hash_symbol(&self, symbol: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        symbol.hash(&mut hasher);
        hasher.finish()
    }

    /// Initialize SIMD buffers with proper alignment
    fn initialize_simd_buffers(&mut self) {
        self.volume_buffer = Vec::with_capacity(SIMD_WIDTH * 16);
        self.timestamp_buffer = Vec::with_capacity(SIMD_WIDTH * 16);
        self.hash_buffer = Vec::with_capacity(SIMD_WIDTH * 16);
    }

    fn get_timestamp_us(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }

    /// Get detector statistics
    pub fn get_statistics(&self) -> WhaleDetectorStats {
        WhaleDetectorStats {
            total_detections: self.total_detections.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            hash_collisions: self.hash_collisions.load(Ordering::Relaxed),
            current_moving_average: self.moving_average,
            current_std_dev: self.volume_std_dev,
            active_whales: self.count_active_whales(),
            table_utilization: self.calculate_table_utilization(),
        }
    }

    /// Count currently active whale entries
    fn count_active_whales(&self) -> usize {
        self.table1
            .iter()
            .chain(self.table2.iter())
            .filter(|entry| entry.is_active)
            .count()
    }

    /// Calculate hash table utilization
    fn calculate_table_utilization(&self) -> f32 {
        let active_count = self.count_active_whales();
        active_count as f32 / (CUCKOO_TABLE_SIZE * 2) as f32
    }
}

/// Enhanced whale movement with additional metrics
#[derive(Debug, Clone)]
pub struct WhaleMovement {
    pub symbol: String,
    pub volume: f64,
    pub timestamp: u64,
    pub velocity_score: f32,
    pub risk_score: f32,
    pub detection_confidence: f32,
    pub pattern_type: WhalePatternType,
}

/// Whale pattern classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhalePatternType {
    Aggressive,   // High velocity, high risk
    Accumulation, // Large volume, gradual
    Momentum,     // Medium velocity, sustained
    Gradual,      // Low velocity, low risk
}

/// Detector performance statistics
#[derive(Debug, Clone)]
pub struct WhaleDetectorStats {
    pub total_detections: u64,
    pub false_positives: u64,
    pub hash_collisions: u64,
    pub current_moving_average: f64,
    pub current_std_dev: f64,
    pub active_whales: usize,
    pub table_utilization: f32,
}

// Thread safety
unsafe impl Send for SimdWhaleDetector {}
unsafe impl Sync for SimdWhaleDetector {}
