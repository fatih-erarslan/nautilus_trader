//! Python bindings for CDFA Unified Library using PyO3
//!
//! This module provides comprehensive Python bindings with:
//! - NumPy array integration for zero-copy operations
//! - Pythonic API design with proper error handling
//! - Financial data validation and safety
//! - Performance optimization with parallel processing
//! - Memory-efficient operations

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::unified::UnifiedCdfa;
use crate::builder::UnifiedCdfaBuilder;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

/// Python wrapper for UnifiedCdfa
#[pyclass(name = "CdfaUnified")]
pub struct PyCdfaUnified {
    inner: UnifiedCdfa,
}

/// Python wrapper for CdfaConfig
#[pyclass(name = "CdfaConfig")]
#[derive(Clone)]
pub struct PyCdfaConfig {
    inner: CdfaConfig,
}

/// Python wrapper for AnalysisResult
#[pyclass(name = "AnalysisResult")]
pub struct PyAnalysisResult {
    inner: AnalysisResult,
}

/// Python wrapper for Pattern
#[pyclass(name = "Pattern")]
#[derive(Clone)]
pub struct PyPattern {
    inner: Pattern,
}

/// Python wrapper for PerformanceMetrics
#[pyclass(name = "PerformanceMetrics")]
#[derive(Clone)]
pub struct PyPerformanceMetrics {
    inner: PerformanceMetrics,
}

/// Python wrapper for DataQuality
#[pyclass(name = "DataQuality")]
#[derive(Clone)]
pub struct PyDataQuality {
    inner: DataQuality,
}

/// Python wrapper for TimeSeries
#[pyclass(name = "TimeSeries")]
pub struct PyTimeSeries {
    inner: TimeSeries,
}

/// Python wrapper for MultiTimeSeries
#[pyclass(name = "MultiTimeSeries")]
pub struct PyMultiTimeSeries {
    inner: MultiTimeSeries,
}

// Helper function to convert Rust errors to Python exceptions
fn to_py_err(error: CdfaError) -> PyErr {
    match error {
        CdfaError::InvalidInput { message } => PyValueError::new_err(message),
        CdfaError::DimensionMismatch { expected, actual } => {
            PyValueError::new_err(format!("Dimension mismatch: expected {}, got {}", expected, actual))
        }
        CdfaError::ConfigError { message } => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

// Convert Result<T> to PyResult<T>
fn to_py_result<T>(result: Result<T>) -> PyResult<T> {
    result.map_err(to_py_err)
}

#[pymethods]
impl PyCdfaConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: CdfaConfig::default(),
        }
    }

    /// Create config from Python dictionary
    #[classmethod]
    fn from_dict(_cls: &PyType, py: Python, dict: &PyDict) -> PyResult<Self> {
        let mut config = CdfaConfig::default();

        if let Some(num_threads) = dict.get_item("num_threads")? {
            config.num_threads = num_threads.extract::<usize>()?;
        }

        if let Some(enable_simd) = dict.get_item("enable_simd")? {
            config.enable_simd = enable_simd.extract::<bool>()?;
        }

        if let Some(enable_gpu) = dict.get_item("enable_gpu")? {
            config.enable_gpu = enable_gpu.extract::<bool>()?;
        }

        if let Some(tolerance) = dict.get_item("tolerance")? {
            config.tolerance = tolerance.extract::<f64>()?;
        }

        if let Some(max_iterations) = dict.get_item("max_iterations")? {
            config.max_iterations = max_iterations.extract::<usize>()?;
        }

        if let Some(convergence_threshold) = dict.get_item("convergence_threshold")? {
            config.convergence_threshold = convergence_threshold.extract::<f64>()?;
        }

        if let Some(cache_size_mb) = dict.get_item("cache_size_mb")? {
            config.cache_size_mb = cache_size_mb.extract::<usize>()?;
        }

        if let Some(enable_distributed) = dict.get_item("enable_distributed")? {
            config.enable_distributed = enable_distributed.extract::<bool>()?;
        }

        if let Some(diversity_method) = dict.get_item("diversity_method")? {
            config.diversity_method = Some(diversity_method.extract::<String>()?);
        }

        if let Some(fusion_method) = dict.get_item("fusion_method")? {
            config.fusion_method = Some(fusion_method.extract::<String>()?);
        }

        Ok(Self { inner: config })
    }

    /// Convert to Python dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("num_threads", self.inner.num_threads)?;
        dict.set_item("enable_simd", self.inner.enable_simd)?;
        dict.set_item("enable_gpu", self.inner.enable_gpu)?;
        dict.set_item("tolerance", self.inner.tolerance)?;
        dict.set_item("max_iterations", self.inner.max_iterations)?;
        dict.set_item("convergence_threshold", self.inner.convergence_threshold)?;
        dict.set_item("cache_size_mb", self.inner.cache_size_mb)?;
        dict.set_item("enable_distributed", self.inner.enable_distributed)?;

        if let Some(ref method) = self.inner.diversity_method {
            dict.set_item("diversity_method", method)?;
        }

        if let Some(ref method) = self.inner.fusion_method {
            dict.set_item("fusion_method", method)?;
        }

        Ok(dict.into())
    }

    // Property getters and setters
    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads
    }

    #[setter]
    fn set_num_threads(&mut self, value: usize) {
        self.inner.num_threads = value;
    }

    #[getter]
    fn enable_simd(&self) -> bool {
        self.inner.enable_simd
    }

    #[setter]
    fn set_enable_simd(&mut self, value: bool) {
        self.inner.enable_simd = value;
    }

    #[getter]
    fn enable_gpu(&self) -> bool {
        self.inner.enable_gpu
    }

    #[setter]
    fn set_enable_gpu(&mut self, value: bool) {
        self.inner.enable_gpu = value;
    }

    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    #[setter]
    fn set_tolerance(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(PyValueError::new_err("Tolerance must be positive"));
        }
        self.inner.tolerance = value;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("CdfaConfig(num_threads={}, enable_simd={}, enable_gpu={}, tolerance={})",
            self.inner.num_threads, self.inner.enable_simd, self.inner.enable_gpu, self.inner.tolerance)
    }
}

#[pymethods]
impl PyPattern {
    #[getter]
    fn pattern_type(&self) -> &str {
        &self.inner.pattern_type
    }

    #[getter]
    fn start_index(&self) -> usize {
        self.inner.start_index
    }

    #[getter]
    fn end_index(&self) -> usize {
        self.inner.end_index
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn timestamp(&self) -> i64 {
        self.inner.timestamp
    }

    #[getter]
    fn parameters(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.parameters {
            dict.set_item(key, *value)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("Pattern(type={}, start={}, end={}, confidence={:.4})",
            self.inner.pattern_type, self.inner.start_index, 
            self.inner.end_index, self.inner.confidence)
    }
}

#[pymethods]
impl PyPerformanceMetrics {
    #[getter]
    fn execution_time_us(&self) -> u64 {
        self.inner.execution_time_us
    }

    #[getter]
    fn memory_used_bytes(&self) -> usize {
        self.inner.memory_used_bytes
    }

    #[getter]
    fn simd_operations(&self) -> usize {
        self.inner.simd_operations
    }

    #[getter]
    fn parallel_tasks(&self) -> usize {
        self.inner.parallel_tasks
    }

    #[getter]
    fn cache_hit_rate(&self) -> f64 {
        self.inner.cache_hit_rate
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn convergence_error(&self) -> f64 {
        self.inner.convergence_error
    }

    fn __repr__(&self) -> String {
        format!("PerformanceMetrics(time_us={}, memory_bytes={}, cache_hit_rate={:.4})",
            self.inner.execution_time_us, self.inner.memory_used_bytes, self.inner.cache_hit_rate)
    }
}

#[pymethods]
impl PyDataQuality {
    #[getter]
    fn missing_percentage(&self) -> f64 {
        self.inner.missing_percentage
    }

    #[getter]
    fn outlier_count(&self) -> usize {
        self.inner.outlier_count
    }

    #[getter]
    fn snr_db(&self) -> f64 {
        self.inner.snr_db
    }

    #[getter]
    fn stationarity_pvalue(&self) -> f64 {
        self.inner.stationarity_pvalue
    }

    #[getter]
    fn skewness(&self) -> f64 {
        self.inner.skewness
    }

    #[getter]
    fn kurtosis(&self) -> f64 {
        self.inner.kurtosis
    }

    #[getter]
    fn autocorrelation_lag1(&self) -> f64 {
        self.inner.autocorrelation_lag1
    }

    fn __repr__(&self) -> String {
        format!("DataQuality(missing_pct={:.2}, outliers={}, snr_db={:.2})",
            self.inner.missing_percentage, self.inner.outlier_count, self.inner.snr_db)
    }
}

#[pymethods]
impl PyAnalysisResult {
    /// Get main result data as NumPy array
    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.inner.data.to_pyarray(py).to_object(py))
    }

    /// Get secondary data by name as NumPy array
    fn get_secondary_data(&self, py: Python, name: &str) -> PyResult<Option<PyObject>> {
        match self.inner.secondary_data.get(name) {
            Some(data) => Ok(Some(data.to_pyarray(py).to_object(py))),
            None => Ok(None),
        }
    }

    /// Get all secondary data as dictionary
    fn get_all_secondary_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (name, data) in &self.inner.secondary_data {
            dict.set_item(name, data.to_pyarray(py))?;
        }
        Ok(dict.into())
    }

    /// Get metric by name
    fn get_metric(&self, name: &str) -> Option<f64> {
        self.inner.get_metric(name)
    }

    /// Get all metrics as dictionary
    fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (name, value) in &self.inner.metrics {
            dict.set_item(name, *value)?;
        }
        Ok(dict.into())
    }

    /// Get detected patterns
    fn get_patterns(&self) -> Vec<PyPattern> {
        self.inner.patterns.iter()
            .map(|p| PyPattern { inner: p.clone() })
            .collect()
    }

    /// Get performance metrics
    fn get_performance(&self) -> PyPerformanceMetrics {
        PyPerformanceMetrics {
            inner: self.inner.performance.clone(),
        }
    }

    /// Get data quality metrics
    fn get_data_quality(&self) -> PyDataQuality {
        PyDataQuality {
            inner: self.inner.data_quality.clone(),
        }
    }

    /// Get analysis timestamp
    fn get_timestamp(&self) -> i64 {
        self.inner.timestamp
    }

    fn __repr__(&self) -> String {
        format!("AnalysisResult(data_len={}, metrics={}, patterns={})",
            self.inner.data.len(), self.inner.metrics.len(), self.inner.patterns.len())
    }
}

#[pymethods]
impl PyTimeSeries {
    #[new]
    fn new(timestamps: Vec<i64>, values: PyReadonlyArray1<f64>) -> Self {
        let values_array = Array1::from_vec(values.as_array().to_vec());
        Self {
            inner: TimeSeries::new(timestamps, values_array),
        }
    }

    /// Get timestamps
    fn get_timestamps(&self) -> Vec<i64> {
        self.inner.timestamps.clone()
    }

    /// Get values as NumPy array
    fn get_values(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.inner.values.to_pyarray(py).to_object(py))
    }

    /// Get length
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Slice the time series
    fn slice(&self, start: usize, end: usize) -> PyResult<Self> {
        if start >= end || end > self.inner.len() {
            return Err(PyValueError::new_err("Invalid slice indices"));
        }
        
        Ok(Self {
            inner: self.inner.slice(start, end),
        })
    }

    /// Get metadata
    fn get_metadata(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.metadata {
            dict.set_item(key, value)?;
        }
        Ok(dict.into())
    }

    /// Set metadata
    fn set_metadata(&mut self, key: String, value: String) {
        self.inner.metadata.insert(key, value);
    }

    fn __repr__(&self) -> String {
        format!("TimeSeries(length={})", self.inner.len())
    }
}

#[pymethods]
impl PyMultiTimeSeries {
    #[new]
    fn new(
        timestamps: Vec<i64>,
        values: PyReadonlyArray2<f64>,
        feature_names: Vec<String>,
    ) -> PyResult<Self> {
        let values_array = Array2::from_shape_vec(
            values.as_array().dim(),
            values.as_array().to_vec(),
        ).map_err(|e| PyValueError::new_err(format!("Invalid array shape: {}", e)))?;

        Ok(Self {
            inner: MultiTimeSeries::new(timestamps, values_array, feature_names),
        })
    }

    /// Get timestamps
    fn get_timestamps(&self) -> Vec<i64> {
        self.inner.timestamps.clone()
    }

    /// Get values as NumPy array
    fn get_values(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.inner.values.to_pyarray(py).to_object(py))
    }

    /// Get feature names
    fn get_feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }

    /// Get length (number of time points)
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get number of features
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Get a specific feature as TimeSeries
    fn get_feature(&self, feature_index: usize) -> PyResult<Option<PyTimeSeries>> {
        match self.inner.get_feature(feature_index) {
            Some(ts) => Ok(Some(PyTimeSeries { inner: ts })),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("MultiTimeSeries(length={}, features={})", 
                self.inner.len(), self.inner.num_features())
    }
}

#[pymethods]
impl PyCdfaUnified {
    #[new]
    fn new(config: Option<PyCdfaConfig>) -> PyResult<Self> {
        let cdfa = match config {
            Some(cfg) => UnifiedCdfa::builder().with_config(cfg.inner).build(),
            None => UnifiedCdfa::new(),
        };

        to_py_result(cdfa).map(|inner| Self { inner })
    }

    /// Create with builder pattern
    #[classmethod]
    fn builder(_cls: &PyType) -> PyCdfaBuilder {
        PyCdfaBuilder::new()
    }

    /// Perform comprehensive CDFA analysis
    /// 
    /// Args:
    ///     data: Input data matrix as NumPy array (rows=observations, cols=features)
    ///     
    /// Returns:
    ///     AnalysisResult: Comprehensive analysis results
    ///     
    /// Raises:
    ///     ValueError: If input data is invalid
    ///     RuntimeError: If analysis fails
    fn analyze(&self, py: Python, data: PyReadonlyArray2<f64>) -> PyResult<PyAnalysisResult> {
        // Validate input for financial safety
        self.validate_financial_data(data.as_array())?;

        // Convert to ndarray view for zero-copy operation
        let data_view = data.as_array();
        
        // Perform analysis
        let result = to_py_result(self.inner.analyze(&data_view))?;
        
        Ok(PyAnalysisResult { inner: result })
    }

    /// Calculate diversity metrics only
    /// 
    /// Args:
    ///     data: Input data matrix as NumPy array
    ///     
    /// Returns:
    ///     numpy.ndarray: Diversity scores
    fn calculate_diversity(&self, py: Python, data: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let data_view = data.as_array();
        let result = to_py_result(self.inner.calculate_diversity(&data_view))?;
        Ok(result.into_pyarray(py).to_object(py))
    }

    /// Apply fusion algorithms
    /// 
    /// Args:
    ///     scores: Input diversity scores as NumPy array
    ///     data: Original data matrix as NumPy array
    ///     
    /// Returns:
    ///     numpy.ndarray: Fused scores
    fn apply_fusion(
        &self,
        py: Python,
        scores: PyReadonlyArray1<f64>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<PyObject> {
        let scores_view = scores.as_array();
        let data_view = data.as_array();
        let result = to_py_result(self.inner.apply_fusion(&scores_view, &data_view))?;
        Ok(result.into_pyarray(py).to_object(py))
    }

    /// Detect patterns in data
    /// 
    /// Args:
    ///     data: Input data matrix as NumPy array
    ///     scores: Optional scores array
    ///     
    /// Returns:
    ///     List[Pattern]: Detected patterns
    fn detect_patterns(
        &self,
        data: PyReadonlyArray2<f64>,
        scores: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Vec<PyPattern>> {
        let data_view = data.as_array();
        let scores_array = match scores {
            Some(s) => s.as_array().to_owned(),
            None => {
                // Calculate diversity scores if not provided
                to_py_result(self.inner.calculate_diversity(&data_view))?
            }
        };
        
        let patterns = to_py_result(self.inner.detect_patterns(&data_view, &scores_array.view()))?;
        Ok(patterns.into_iter().map(|p| PyPattern { inner: p }).collect())
    }

    /// Get current configuration
    fn get_config(&self) -> PyCdfaConfig {
        PyCdfaConfig {
            inner: self.inner.config(),
        }
    }

    /// Update configuration
    fn update_config(&self, config: PyCdfaConfig) -> PyResult<()> {
        to_py_result(self.inner.update_config(|cfg| *cfg = config.inner))
    }

    /// Get performance metrics
    fn get_performance_metrics(&self) -> PyPerformanceMetrics {
        PyPerformanceMetrics {
            inner: self.inner.performance_metrics(),
        }
    }

    /// Clear all caches
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    /// Validate data for financial system safety
    fn validate_data(&self, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.validate_financial_data(data.as_array())
    }

    /// Batch process multiple datasets
    fn batch_analyze(
        &self,
        py: Python,
        datasets: &PyList,
    ) -> PyResult<Vec<PyAnalysisResult>> {
        let mut results = Vec::new();
        
        for item in datasets.iter() {
            let data: PyReadonlyArray2<f64> = item.extract()?;
            let data_view = data.as_array();
            
            self.validate_financial_data(&data_view)?;
            let result = to_py_result(self.inner.analyze(&data_view))?;
            results.push(PyAnalysisResult { inner: result });
        }
        
        Ok(results)
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
        format!("CdfaUnified(threads={}, simd={}, gpu={})",
            config.num_threads, config.enable_simd, config.enable_gpu)
    }
}

impl PyCdfaUnified {
    /// Validate input data for financial system safety
    fn validate_financial_data(&self, data: &ArrayView2<f64>) -> PyResult<()> {
        // Basic shape validation
        if data.nrows() < 2 || data.ncols() < 2 {
            return Err(PyValueError::new_err(
                "Data must have at least 2 rows and 2 columns"
            ));
        }

        // Check for NaN, infinity, and unreasonable financial values
        for &value in data.iter() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(
                    "Data contains NaN or infinite values"
                ));
            }
            
            // Financial data validation
            if value.abs() > 1e15 {
                return Err(PyValueError::new_err(
                    "Data contains unreasonably large values for financial analysis"
                ));
            }
        }

        // Additional financial data checks
        let mut has_variance = false;
        for col in 0..data.ncols() {
            let column = data.column(col);
            let mean = column.sum() / column.len() as f64;
            let variance = column.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / column.len() as f64;
            
            if variance > 1e-12 {
                has_variance = true;
                break;
            }
        }
        
        if !has_variance {
            return Err(PyValueError::new_err(
                "Data appears to have no variance (all constant values)"
            ));
        }

        Ok(())
    }
}

/// Python builder for CdfaUnified
#[pyclass(name = "CdfaBuilder")]
pub struct PyCdfaBuilder {
    inner: UnifiedCdfaBuilder,
}

#[pymethods]
impl PyCdfaBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: UnifiedCdfa::builder(),
        }
    }

    /// Set number of threads
    fn with_threads(mut self_: PyRefMut<Self>, threads: usize) -> PyRefMut<Self> {
        self_.inner = self_.inner.with_threads(threads);
        self_
    }

    /// Enable SIMD processing
    fn with_simd(mut self_: PyRefMut<Self>, enable: bool) -> PyRefMut<Self> {
        self_.inner = self_.inner.with_simd(enable);
        self_
    }

    /// Enable GPU processing
    fn with_gpu(mut self_: PyRefMut<Self>, enable: bool) -> PyRefMut<Self> {
        self_.inner = self_.inner.with_gpu(enable);
        self_
    }

    /// Set tolerance
    fn with_tolerance(mut self_: PyRefMut<Self>, tolerance: f64) -> PyResult<PyRefMut<Self>> {
        if tolerance <= 0.0 {
            return Err(PyValueError::new_err("Tolerance must be positive"));
        }
        self_.inner = self_.inner.with_tolerance(tolerance);
        Ok(self_)
    }

    /// Set configuration
    fn with_config(mut self_: PyRefMut<Self>, config: PyCdfaConfig) -> PyRefMut<Self> {
        self_.inner = self_.inner.with_config(config.inner);
        self_
    }

    /// Build the CdfaUnified instance
    fn build(&self) -> PyResult<PyCdfaUnified> {
        let cdfa = to_py_result(self.inner.build())?;
        Ok(PyCdfaUnified { inner: cdfa })
    }
}

/// Utility functions
#[pyfunction]
fn validate_financial_data(data: PyReadonlyArray2<f64>) -> PyResult<()> {
    let data_view = data.as_array();
    
    // Same validation as in PyCdfaUnified::validate_financial_data
    if data_view.nrows() < 2 || data_view.ncols() < 2 {
        return Err(PyValueError::new_err(
            "Data must have at least 2 rows and 2 columns"
        ));
    }

    for &value in data_view.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(
                "Data contains NaN or infinite values"
            ));
        }
        if value.abs() > 1e15 {
            return Err(PyValueError::new_err(
                "Data contains unreasonably large values"
            ));
        }
    }

    Ok(())
}

#[pyfunction]
fn get_version() -> &'static str {
    crate::VERSION
}

#[pyfunction]
fn create_sample_data(py: Python, rows: usize, cols: usize, seed: Option<u64>) -> PyResult<PyObject> {
    use rand::prelude::*;
    use rand_distr::Normal;

    if rows < 2 || cols < 2 {
        return Err(PyValueError::new_err("Need at least 2 rows and 2 columns"));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = Normal::new(100.0, 15.0).unwrap(); // Stock price-like distribution
    let mut data = Array2::zeros((rows, cols));
    
    for mut row in data.rows_mut() {
        let base_price: f64 = rng.sample(normal);
        row[0] = base_price;
        
        for i in 1..cols {
            // Random walk with small steps
            let change: f64 = rng.sample(Normal::new(0.0, 0.02).unwrap());
            row[i] = row[i-1] * (1.0 + change);
        }
    }

    Ok(data.into_pyarray(py).to_object(py))
}

/// Python module definition
#[pymodule]
fn cdfa_unified(py: Python, m: &PyModule) -> PyResult<()> {
    // Add classes
    m.add_class::<PyCdfaUnified>()?;
    m.add_class::<PyCdfaConfig>()?;
    m.add_class::<PyAnalysisResult>()?;
    m.add_class::<PyPattern>()?;
    m.add_class::<PyPerformanceMetrics>()?;
    m.add_class::<PyDataQuality>()?;
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyMultiTimeSeries>()?;
    m.add_class::<PyCdfaBuilder>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(validate_financial_data, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(create_sample_data, m)?)?;

    // Add constants
    m.add("VERSION", crate::VERSION)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use numpy::PyArray2;

    #[test]
    fn test_config_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let config = PyCdfaConfig::new();
            assert_eq!(config.inner.num_threads, 0);
            assert!(config.inner.enable_simd);
        });
    }

    #[test]
    fn test_data_validation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Valid data
            let valid_data = array![[1.0, 2.0, 3.0], [1.1, 2.1, 2.9]];
            let py_array = PyArray2::from_array(py, &valid_data);
            let readonly = py_array.readonly();
            
            let cdfa = PyCdfaUnified::new(None).unwrap();
            assert!(cdfa.validate_financial_data(readonly.as_array()).is_ok());

            // Invalid data with NaN
            let invalid_data = array![[1.0, f64::NAN, 3.0], [1.1, 2.1, 2.9]];
            let py_array = PyArray2::from_array(py, &invalid_data);
            let readonly = py_array.readonly();
            
            assert!(cdfa.validate_financial_data(readonly.as_array()).is_err());
        });
    }
}