//! PyO3 Python bindings for prospect theory

use crate::{ValueFunction, ValueFunctionParams, ProbabilityWeighting, WeightingParams};
use crate::probability_weighting::WeightingFunction;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
use std::collections::HashMap;

/// Convert our Result type to PyResult
fn to_py_result<T>(result: crate::Result<T>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Python wrapper for ValueFunctionParams
#[pyclass(name = "ValueFunctionParams")]
#[derive(Clone)]
pub struct PyValueFunctionParams {
    inner: ValueFunctionParams,
}

#[pymethods]
impl PyValueFunctionParams {
    #[new]
    #[pyo3(signature = (alpha = 0.88, beta = 0.88, lambda = 2.25, reference_point = 0.0))]
    fn new(alpha: f64, beta: f64, lambda: f64, reference_point: f64) -> PyResult<Self> {
        let inner = to_py_result(ValueFunctionParams::new(alpha, beta, lambda, reference_point))?;
        Ok(Self { inner })
    }

    /// Create default Kahneman-Tversky parameters
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ValueFunctionParams::default(),
        }
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    #[getter]
    fn lambda(&self) -> f64 {
        self.inner.lambda
    }

    #[getter]
    fn reference_point(&self) -> f64 {
        self.inner.reference_point
    }

    fn __repr__(&self) -> String {
        format!(
            "ValueFunctionParams(alpha={}, beta={}, lambda={}, reference_point={})",
            self.inner.alpha, self.inner.beta, self.inner.lambda, self.inner.reference_point
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Convert to dictionary
    fn to_dict(&self) -> HashMap<String, f64> {
        let mut dict = HashMap::new();
        dict.insert("alpha".to_string(), self.inner.alpha);
        dict.insert("beta".to_string(), self.inner.beta);
        dict.insert("lambda".to_string(), self.inner.lambda);
        dict.insert("reference_point".to_string(), self.inner.reference_point);
        dict
    }

    /// Create from dictionary
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, dict: HashMap<String, f64>) -> PyResult<Self> {
        let alpha = dict.get("alpha").copied().unwrap_or(0.88);
        let beta = dict.get("beta").copied().unwrap_or(0.88);
        let lambda = dict.get("lambda").copied().unwrap_or(2.25);
        let reference_point = dict.get("reference_point").copied().unwrap_or(0.0);
        
        Self::new(alpha, beta, lambda, reference_point)
    }
}

/// Python wrapper for WeightingParams
#[pyclass(name = "WeightingParams")]
#[derive(Clone)]
pub struct PyWeightingParams {
    inner: WeightingParams,
}

#[pymethods]
impl PyWeightingParams {
    #[new]
    #[pyo3(signature = (gamma_gains = 0.61, gamma_losses = 0.69, delta_gains = 1.0, delta_losses = 1.0))]
    fn new(gamma_gains: f64, gamma_losses: f64, delta_gains: f64, delta_losses: f64) -> PyResult<Self> {
        let inner = to_py_result(WeightingParams::new(gamma_gains, gamma_losses, delta_gains, delta_losses))?;
        Ok(Self { inner })
    }

    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: WeightingParams::default(),
        }
    }

    #[getter]
    fn gamma_gains(&self) -> f64 {
        self.inner.gamma_gains
    }

    #[getter]
    fn gamma_losses(&self) -> f64 {
        self.inner.gamma_losses
    }

    #[getter]
    fn delta_gains(&self) -> f64 {
        self.inner.delta_gains
    }

    #[getter]
    fn delta_losses(&self) -> f64 {
        self.inner.delta_losses
    }

    fn __repr__(&self) -> String {
        format!(
            "WeightingParams(gamma_gains={}, gamma_losses={}, delta_gains={}, delta_losses={})",
            self.inner.gamma_gains, self.inner.gamma_losses, self.inner.delta_gains, self.inner.delta_losses
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn to_dict(&self) -> HashMap<String, f64> {
        let mut dict = HashMap::new();
        dict.insert("gamma_gains".to_string(), self.inner.gamma_gains);
        dict.insert("gamma_losses".to_string(), self.inner.gamma_losses);
        dict.insert("delta_gains".to_string(), self.inner.delta_gains);
        dict.insert("delta_losses".to_string(), self.inner.delta_losses);
        dict
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, dict: HashMap<String, f64>) -> PyResult<Self> {
        let gamma_gains = dict.get("gamma_gains").copied().unwrap_or(0.61);
        let gamma_losses = dict.get("gamma_losses").copied().unwrap_or(0.69);
        let delta_gains = dict.get("delta_gains").copied().unwrap_or(1.0);
        let delta_losses = dict.get("delta_losses").copied().unwrap_or(1.0);
        
        Self::new(gamma_gains, gamma_losses, delta_gains, delta_losses)
    }
}

/// Python wrapper for ValueFunction
#[pyclass(name = "ValueFunction")]
pub struct PyValueFunction {
    inner: ValueFunction,
}

#[pymethods]
impl PyValueFunction {
    #[new]
    fn new(params: PyValueFunctionParams) -> PyResult<Self> {
        let inner = to_py_result(ValueFunction::new(params.inner))?;
        Ok(Self { inner })
    }

    /// Create with default Kahneman-Tversky parameters
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ValueFunction::default_kt(),
        }
    }

    /// Calculate value for a single outcome
    fn value(&self, outcome: f64) -> PyResult<f64> {
        to_py_result(self.inner.value(outcome))
    }

    /// Calculate values for multiple outcomes
    fn values(&self, outcomes: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.values(&outcomes))
    }

    /// Calculate values with parallel processing
    fn values_parallel(&self, outcomes: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.values_parallel(&outcomes))
    }

    /// Calculate marginal value (derivative)
    fn marginal_value(&self, outcome: f64) -> PyResult<f64> {
        to_py_result(self.inner.marginal_value(outcome))
    }

    /// Calculate risk premium for a lottery
    fn risk_premium(&self, outcomes: Vec<f64>, probabilities: Vec<f64>) -> PyResult<f64> {
        to_py_result(self.inner.risk_premium(&outcomes, &probabilities))
    }

    /// Calculate certainty equivalent (inverse value function)
    fn certainty_equivalent(&self, value: f64) -> PyResult<f64> {
        to_py_result(self.inner.inverse_value(value))
    }

    /// Calculate loss aversion ratio
    fn loss_aversion_ratio(&self, gain: f64, loss: f64) -> PyResult<f64> {
        to_py_result(self.inner.loss_aversion_ratio(gain, loss))
    }

    /// Get parameters as dictionary
    fn get_params(&self) -> HashMap<String, f64> {
        let params = self.inner.params();
        let mut dict = HashMap::new();
        dict.insert("alpha".to_string(), params.alpha);
        dict.insert("beta".to_string(), params.beta);
        dict.insert("lambda".to_string(), params.lambda);
        dict.insert("reference_point".to_string(), params.reference_point);
        dict
    }

    fn __repr__(&self) -> String {
        format!("ValueFunction(params={:?})", self.inner.params())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Batch processing method for financial applications
    fn batch_calculate(&self, py: Python, outcomes: Vec<f64>) -> PyResult<PyObject> {
        let values = to_py_result(self.inner.values_parallel(&outcomes))?;
        let result_dict = HashMap::from([
            ("outcomes".to_string(), outcomes.into_py(py)),
            ("values".to_string(), values.into_py(py)),
        ]);
        Ok(result_dict.into_py(py))
    }
}

/// Python wrapper for ProbabilityWeighting
#[pyclass(name = "ProbabilityWeighting")]
pub struct PyProbabilityWeighting {
    inner: ProbabilityWeighting,
}

#[pymethods]
impl PyProbabilityWeighting {
    #[new]
    #[pyo3(signature = (params, function_type = "tversky_kahneman"))]
    fn new(params: PyWeightingParams, function_type: &str) -> PyResult<Self> {
        let function_type = match function_type.to_lowercase().as_str() {
            "tversky_kahneman" | "tk" => WeightingFunction::TverskyKahneman,
            "prelec" => WeightingFunction::Prelec,
            "linear" => WeightingFunction::Linear,
            _ => return Err(PyValueError::new_err(format!(
                "Unknown function type: {}. Use 'tversky_kahneman', 'prelec', or 'linear'",
                function_type
            ))),
        };

        let inner = to_py_result(ProbabilityWeighting::new(params.inner, function_type))?;
        Ok(Self { inner })
    }

    /// Create with default Tversky-Kahneman parameters
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ProbabilityWeighting::default_tk(),
        }
    }

    /// Create Prelec weighting function
    #[classmethod]
    fn prelec(_cls: &Bound<'_, PyType>, params: PyWeightingParams) -> PyResult<Self> {
        let inner = to_py_result(ProbabilityWeighting::prelec(params.inner))?;
        Ok(Self { inner })
    }

    /// Create linear weighting function
    #[classmethod]
    fn linear(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: ProbabilityWeighting::linear(),
        }
    }

    /// Calculate probability weight for gains
    fn weight_gains(&self, probability: f64) -> PyResult<f64> {
        to_py_result(self.inner.weight_gains(probability))
    }

    /// Calculate probability weight for losses
    fn weight_losses(&self, probability: f64) -> PyResult<f64> {
        to_py_result(self.inner.weight_losses(probability))
    }

    /// Calculate weights for multiple probabilities (gains)
    fn weights_gains(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.weights_gains(&probabilities))
    }

    /// Calculate weights for multiple probabilities (losses)
    fn weights_losses(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.weights_losses(&probabilities))
    }

    /// Calculate weights with parallel processing (gains)
    fn weights_gains_parallel(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.weights_gains_parallel(&probabilities))
    }

    /// Calculate weights with parallel processing (losses)
    fn weights_losses_parallel(&self, probabilities: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.weights_losses_parallel(&probabilities))
    }

    /// Calculate decision weights for a probability distribution
    fn decision_weights(&self, probabilities: Vec<f64>, outcomes: Vec<f64>) -> PyResult<Vec<f64>> {
        to_py_result(self.inner.decision_weights(&probabilities, &outcomes))
    }

    /// Calculate attractiveness measure
    fn attractiveness(&self, probability: f64) -> PyResult<f64> {
        to_py_result(self.inner.attractiveness(probability))
    }

    /// Get parameters as dictionary
    fn get_params(&self) -> HashMap<String, f64> {
        let params = self.inner.params();
        let mut dict = HashMap::new();
        dict.insert("gamma_gains".to_string(), params.gamma_gains);
        dict.insert("gamma_losses".to_string(), params.gamma_losses);
        dict.insert("delta_gains".to_string(), params.delta_gains);
        dict.insert("delta_losses".to_string(), params.delta_losses);
        dict
    }

    /// Get function type as string
    fn get_function_type(&self) -> String {
        match self.inner.function_type() {
            WeightingFunction::TverskyKahneman => "tversky_kahneman".to_string(),
            WeightingFunction::Prelec => "prelec".to_string(),
            WeightingFunction::Linear => "linear".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProbabilityWeighting(function_type={}, params={:?})",
            self.get_function_type(),
            self.inner.params()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Batch processing method for financial applications
    fn batch_calculate(&self, py: Python, probabilities: Vec<f64>, domain: &str) -> PyResult<PyObject> {
        let weights = match domain.to_lowercase().as_str() {
            "gains" => to_py_result(self.inner.weights_gains_parallel(&probabilities))?,
            "losses" => to_py_result(self.inner.weights_losses_parallel(&probabilities))?,
            _ => return Err(PyValueError::new_err(
                "Domain must be 'gains' or 'losses'",
            )),
        };

        let result_dict = HashMap::from([
            ("probabilities".to_string(), probabilities.into_py(py)),
            ("weights".to_string(), weights.into_py(py)),
            ("domain".to_string(), domain.into_py(py)),
        ]);
        Ok(result_dict.into_py(py))
    }
}

/// High-level prospect theory calculator
#[pyclass(name = "ProspectTheory")]
pub struct PyProspectTheory {
    value_function: ValueFunction,
    probability_weighting: ProbabilityWeighting,
}

#[pymethods]
impl PyProspectTheory {
    #[new]
    fn new(
        value_params: Option<PyValueFunctionParams>,
        weighting_params: Option<PyWeightingParams>,
        weighting_function: Option<&str>,
    ) -> PyResult<Self> {
        let value_params = value_params
            .map(|p| p.inner)
            .unwrap_or_else(ValueFunctionParams::default);
        
        let weighting_params = weighting_params
            .map(|p| p.inner)
            .unwrap_or_else(WeightingParams::default);

        let function_type = match weighting_function.unwrap_or("tversky_kahneman").to_lowercase().as_str() {
            "tversky_kahneman" | "tk" => WeightingFunction::TverskyKahneman,
            "prelec" => WeightingFunction::Prelec,
            "linear" => WeightingFunction::Linear,
            f => return Err(PyValueError::new_err(format!(
                "Unknown function type: {}. Use 'tversky_kahneman', 'prelec', or 'linear'", f
            ))),
        };

        let value_function = to_py_result(ValueFunction::new(value_params))?;
        let probability_weighting = to_py_result(ProbabilityWeighting::new(weighting_params, function_type))?;

        Ok(Self {
            value_function,
            probability_weighting,
        })
    }

    /// Create with default parameters
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            value_function: ValueFunction::default_kt(),
            probability_weighting: ProbabilityWeighting::default_tk(),
        }
    }

    /// Calculate prospect value for a lottery
    fn prospect_value(&self, outcomes: Vec<f64>, probabilities: Vec<f64>) -> PyResult<f64> {
        if outcomes.len() != probabilities.len() {
            return Err(PyValueError::new_err(
                "Outcomes and probabilities must have the same length",
            ));
        }

        // Calculate values
        let values = to_py_result(self.value_function.values(&outcomes))?;
        
        // Calculate decision weights
        let decision_weights = to_py_result(
            self.probability_weighting.decision_weights(&probabilities, &outcomes)
        )?;

        // Calculate prospect value
        let prospect_value: f64 = values
            .iter()
            .zip(decision_weights.iter())
            .map(|(&value, &weight)| value * weight)
            .sum();

        Ok(prospect_value)
    }

    /// Batch calculate prospect values for multiple lotteries
    fn batch_prospect_values(
        &self,
        py: Python,
        lotteries: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> PyResult<PyObject> {
        let mut prospect_values = Vec::with_capacity(lotteries.len());
        
        for (outcomes, probabilities) in lotteries.iter() {
            let value = self.prospect_value(outcomes.clone(), probabilities.clone())?;
            prospect_values.push(value);
        }

        Ok(prospect_values.into_py(py))
    }

    /// Compare two lotteries and return preference
    fn compare_lotteries(
        &self,
        outcomes_a: Vec<f64>,
        probabilities_a: Vec<f64>,
        outcomes_b: Vec<f64>,
        probabilities_b: Vec<f64>,
    ) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
        
        let value_a = self.prospect_value(outcomes_a.clone(), probabilities_a.clone())?;
        let value_b = self.prospect_value(outcomes_b.clone(), probabilities_b.clone())?;
        
        let preference = if value_a > value_b {
            "A"
        } else if value_b > value_a {
            "B"
        } else {
            "Indifferent"
        };

        let mut result = HashMap::new();
        result.insert("lottery_a_value".to_string(), value_a.into_py(py));
        result.insert("lottery_b_value".to_string(), value_b.into_py(py));
        result.insert("preference".to_string(), preference.into_py(py));
        result.insert("value_difference".to_string(), (value_a - value_b).into_py(py));

        Ok(result)
        })
    }

    fn __repr__(&self) -> String {
        "ProspectTheory(value_function=ValueFunction, probability_weighting=ProbabilityWeighting)".to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}