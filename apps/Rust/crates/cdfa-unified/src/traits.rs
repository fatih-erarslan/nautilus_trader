//! Common Traits for CDFA Unified
//!
//! This module defines common traits used across the CDFA ecosystem.

use ndarray::Array2;
#[cfg(feature = "ml")]
use crate::ml::MLResult;
#[cfg(not(feature = "ml"))]
type MLResult<T> = crate::error::Result<T>;

/// Feature extractor trait
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from input data
    fn extract(&self, data: &Array2<f32>) -> MLResult<Array2<f32>>;
    
    /// Get extractor name
    fn name(&self) -> &str;
}

/// Analyzer trait for different analysis types
pub trait Analyzer: Send + Sync {
    /// Input data type
    type Input;
    /// Output result type
    type Output;
    /// Configuration type
    type Config;
    
    /// Create new analyzer with configuration
    fn new(config: Self::Config) -> MLResult<Self> where Self: Sized;
    
    /// Analyze input data
    fn analyze(&self, input: &Self::Input) -> MLResult<Self::Output>;
    
    /// Get analyzer name
    fn name(&self) -> &str;
}

/// System analyzer trait for complex system analysis
pub trait SystemAnalyzer: Send + Sync {
    /// Analyze data and return metrics
    fn analyze(&self, data: &crate::types::FloatArrayView2, _scores: &crate::types::FloatArrayView1) -> crate::error::Result<std::collections::HashMap<String, crate::types::Float>>;
    
    /// Get analyzer name
    fn name(&self) -> &'static str;
    
    /// Get metric names provided by this analyzer
    fn metric_names(&self) -> Vec<String>;
}

/// Diversity method trait for calculating diversity between data sources
pub trait DiversityMethod: Send + Sync {
    /// Calculate diversity scores for the given data
    fn calculate(&self, data: &crate::types::FloatArrayView2) -> crate::error::Result<crate::types::FloatArray1>;
    
    /// Get the name of this diversity method
    fn name(&self) -> &'static str {
        "diversity_method"
    }
}

/// Fusion method trait for combining multiple scores or rankings
pub trait FusionMethod: Send + Sync {
    /// Fuse multiple inputs into a single result
    fn fuse(&self, data: &crate::types::FloatArrayView2) -> crate::error::Result<crate::types::FloatArray1>;
    
    /// Get the name of this fusion method
    fn name(&self) -> &'static str {
        "fusion_method"
    }
}

/// Pattern detector trait for identifying specific patterns in data
pub trait PatternDetector: Send + Sync {
    /// Detect patterns in the given data
    fn detect(&self, data: &crate::types::FloatArrayView1) -> crate::error::Result<bool>;
    
    /// Get the name of this pattern detector
    fn name(&self) -> &'static str {
        "pattern_detector"
    }
}

/// Signal algorithm trait for processing signals
pub trait SignalAlgorithm: Send + Sync {
    /// Process the signal and return the result
    fn process(&self, signal: &crate::types::FloatArrayView1) -> crate::error::Result<crate::types::FloatArray1>;
    
    /// Get the name of this signal algorithm
    fn name(&self) -> &'static str {
        "signal_algorithm"
    }
}