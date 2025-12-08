//! Core data structures for the CDFA system
//!
//! This module provides high-performance, cache-aligned data structures
//! designed for efficient financial signal processing and analysis.

use core::fmt;
use core::ops::{Deref, DerefMut, Index, IndexMut};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a financial signal with time-series data
///
/// Signals are the fundamental data unit in CDFA, representing any time-series
/// financial data such as price, volume, or derived indicators.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Signal {
    /// Unique identifier for this signal
    pub id: SignalId,
    
    /// Timestamp in nanoseconds since Unix epoch
    pub timestamp_ns: u64,
    
    /// Signal values (cache-aligned for SIMD operations)
    pub values: AlignedVec<f64>,
    
    /// Optional metadata
    pub metadata: SignalMetadata,
}

impl Signal {
    /// Creates a new signal with the given values
    pub fn new(id: SignalId, timestamp_ns: u64, values: Vec<f64>) -> Self {
        Self {
            id,
            timestamp_ns,
            values: AlignedVec::from_vec(values),
            metadata: SignalMetadata::default(),
        }
    }

    /// Creates an empty signal with specified capacity
    pub fn with_capacity(id: SignalId, timestamp_ns: u64, capacity: usize) -> Self {
        Self {
            id,
            timestamp_ns,
            values: AlignedVec::with_capacity(capacity),
            metadata: SignalMetadata::default(),
        }
    }

    /// Returns the number of values in this signal
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the signal has no values
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns a slice of the signal values
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Returns a mutable slice of the signal values
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

/// Unique identifier for signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalId(pub u64);

impl fmt::Display for SignalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signal#{}", self.0)
    }
}

/// Metadata associated with a signal
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalMetadata {
    /// Source of the signal (e.g., "price", "volume", "rsi")
    pub source: Option<String>,
    
    /// Trading symbol (e.g., "BTC/USDT")
    pub symbol: Option<String>,
    
    /// Timeframe in seconds (e.g., 60 for 1-minute candles)
    pub timeframe: Option<u32>,
    
    /// Additional properties
    pub properties: Vec<(String, String)>,
}

/// Result of cognitive diversity analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalysisResult {
    /// Unique identifier for this result
    pub id: AnalysisId,
    
    /// Timestamp when analysis was completed (nanoseconds)
    pub timestamp_ns: u64,
    
    /// Analyzer that produced this result
    pub analyzer_id: String,
    
    /// Primary prediction or signal
    pub prediction: f64,
    
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f64,
    
    /// Diversity metrics computed
    pub diversity_scores: Vec<(String, f64)>,
    
    /// Feature importances or contributions
    pub features: Vec<Feature>,
    
    /// Processing latency in nanoseconds
    pub latency_ns: u64,
}

impl AnalysisResult {
    /// Creates a new analysis result
    pub fn new(analyzer_id: String, prediction: f64, confidence: f64) -> Self {
        Self {
            id: AnalysisId::new(),
            timestamp_ns: timestamp_now_ns(),
            analyzer_id,
            prediction,
            confidence,
            diversity_scores: Vec::new(),
            features: Vec::new(),
            latency_ns: 0,
        }
    }

    /// Adds a diversity score
    pub fn add_diversity_score(&mut self, metric: String, score: f64) {
        self.diversity_scores.push((metric, score));
    }

    /// Adds a feature contribution
    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }
}

/// Unique identifier for analysis results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalysisId(pub u64);

impl AnalysisId {
    /// Creates a new unique analysis ID
    pub fn new() -> Self {
        // In production, use a proper ID generator (UUID, atomic counter, etc.)
        Self(timestamp_now_ns())
    }
}

impl Default for AnalysisId {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a feature contribution in analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Feature {
    /// Feature name
    pub name: String,
    
    /// Feature value
    pub value: f64,
    
    /// Importance or weight (0.0 to 1.0)
    pub importance: f64,
}

/// Matrix for storing pairwise diversity scores
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiversityMatrix {
    /// Flattened matrix data (row-major order)
    data: AlignedVec<f64>,
    
    /// Matrix dimension (square matrix)
    dimension: usize,
}

impl DiversityMatrix {
    /// Creates a new diversity matrix filled with zeros
    pub fn zeros(dimension: usize) -> Self {
        Self {
            data: AlignedVec::from_vec(vec![0.0; dimension * dimension]),
            dimension,
        }
    }

    /// Gets the value at position (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.dimension && col < self.dimension);
        self.data[row * self.dimension + col]
    }

    /// Sets the value at position (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.dimension && col < self.dimension);
        self.data[row * self.dimension + col] = value;
    }

    /// Returns the matrix dimension
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Computes the average diversity score
    pub fn average_diversity(&self) -> f64 {
        if self.dimension <= 1 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.dimension {
            for j in i + 1..self.dimension {
                sum += self.get(i, j);
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

/// Output of signal fusion
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FusionOutput {
    /// Fused signal
    pub signal: Signal,
    
    /// Confidence in the fusion (0.0 to 1.0)
    pub confidence: f64,
    
    /// Weights applied to each input signal
    pub weights: Vec<f64>,
    
    /// Individual contributions from each analyzer
    pub contributions: Vec<(String, f64)>,
    
    /// Fusion strategy used
    pub strategy_id: String,
    
    /// Processing latency in nanoseconds
    pub latency_ns: u64,
}

/// Cache-aligned vector for SIMD operations
///
/// This type ensures that data is aligned to cache line boundaries
/// for optimal performance with SIMD instructions.
#[derive(Clone)]
#[repr(C, align(64))] // Align to typical cache line size
pub struct AlignedVec<T> {
    data: Vec<T>,
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for AlignedVec<T> {
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for AlignedVec<T> {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            data: Vec::deserialize(deserializer)?,
        })
    }
}

impl<T> AlignedVec<T> {
    /// Creates a new empty aligned vector
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates an aligned vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates an aligned vector from a regular vector
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { data: vec }
    }

    /// Returns the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Pushes an element to the vector
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Returns a pointer to the aligned data
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the aligned data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Index<usize> for AlignedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for AlignedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}

impl<T: PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

/// Gets current timestamp in nanoseconds
///
/// In production, use a high-precision clock source
fn timestamp_now_ns() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    #[cfg(not(feature = "std"))]
    {
        // In no_std environment, this would come from a hardware timer
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]);
        assert_eq!(signal.len(), 3);
        assert_eq!(signal.id, SignalId(1));
        assert_eq!(signal.timestamp_ns, 1000);
    }

    #[test]
    fn test_aligned_vec() {
        let mut vec = AlignedVec::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);
        
        vec.push(4.0);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[3], 4.0);
    }

    #[test]
    fn test_diversity_matrix() {
        let mut matrix = DiversityMatrix::zeros(3);
        matrix.set(0, 1, 0.5);
        matrix.set(1, 0, 0.5);
        matrix.set(0, 2, 0.8);
        matrix.set(2, 0, 0.8);
        matrix.set(1, 2, 0.3);
        matrix.set(2, 1, 0.3);
        
        assert_eq!(matrix.get(0, 1), 0.5);
        assert_eq!(matrix.get(1, 0), 0.5);
        
        let avg = matrix.average_diversity();
        assert!((avg - 0.533).abs() < 0.01); // (0.5 + 0.8 + 0.3) / 3
    }

    #[test]
    fn test_analysis_result() {
        let mut result = AnalysisResult::new("test_analyzer".to_string(), 0.75, 0.9);
        result.add_diversity_score("kendall_tau".to_string(), 0.6);
        result.add_feature(Feature {
            name: "rsi".to_string(),
            value: 65.0,
            importance: 0.8,
        });
        
        assert_eq!(result.analyzer_id, "test_analyzer");
        assert_eq!(result.prediction, 0.75);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.diversity_scores.len(), 1);
        assert_eq!(result.features.len(), 1);
    }
}