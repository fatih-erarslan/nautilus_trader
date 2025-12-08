//! Parallel processing utilities for SOC analysis

use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::{Result, SOCError};

/// Initialize the thread pool for parallel processing
pub fn init_thread_pool() -> Result<()> {
    // Rayon automatically initializes its global thread pool
    // We can configure it if needed
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .map_err(|e| SOCError::ComputationError { 
            message: format!("Failed to initialize thread pool: {}", e) 
        })?;
    Ok(())
}

/// Parallel SOC analyzer for processing multiple time series
pub struct ParallelSOCAnalyzer {
    num_threads: usize,
}

impl ParallelSOCAnalyzer {
    pub fn new(num_threads: Option<usize>) -> Self {
        Self {
            num_threads: num_threads.unwrap_or_else(|| rayon::current_num_threads()),
        }
    }
    
    /// Process multiple time series in parallel
    pub fn analyze_batch(&self, series: &[Vec<f64>]) -> Result<Vec<f64>> {
        series.par_iter()
            .map(|s| self.analyze_single(s))
            .collect()
    }
    
    fn analyze_single(&self, series: &[f64]) -> Result<f64> {
        // Placeholder for actual SOC analysis
        if series.is_empty() {
            return Err(SOCError::InvalidParameters { message: "Empty series".to_string() });
        }
        
        // Simple complexity measure for now
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let variance = series.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / series.len() as f64;
        
        Ok(variance.sqrt() / mean.abs().max(1.0))
    }
}

/// Parallel avalanche detector
pub struct ParallelAvalancheDetector {
    threshold: f64,
}

impl ParallelAvalancheDetector {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
    
    /// Detect avalanches in parallel across multiple series
    pub fn detect_batch(&self, series: &[Vec<f64>]) -> Vec<Vec<bool>> {
        series.par_iter()
            .map(|s| self.detect_single(s))
            .collect()
    }
    
    fn detect_single(&self, series: &[f64]) -> Vec<bool> {
        series.windows(2)
            .map(|w| (w[1] - w[0]).abs() > self.threshold)
            .collect()
    }
}