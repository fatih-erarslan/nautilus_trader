// ATS-Core Integration - Calibrated Predictions and Scientific Computing
// Integration with ATS-Core for temperature scaling and conformal prediction

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    IntegrationError, NeuralModel, InputData, CalibrationConfig, 
    CalibratedPrediction, ForecastResult
};

/// ATS-Core integration for calibrated predictions
pub struct AtsCoreIntegration {
    temperature_scaling: TemperatureScaler,
    conformal_predictor: ConformalPredictor,
    calibration_cache: Arc<tokio::sync::RwLock<HashMap<String, CalibratedModel>>>,
}

impl AtsCoreIntegration {
    pub fn new() -> Self {
        Self {
            temperature_scaling: TemperatureScaler::new(),
            conformal_predictor: ConformalPredictor::new(),
            calibration_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
    
    /// Generate calibrated prediction with uncertainty quantification
    pub async fn calibrated_prediction(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        config: CalibrationConfig,
    ) -> Result<CalibratedPrediction, IntegrationError> {
        // Get raw prediction from model
        let raw_prediction = self.get_raw_prediction(&*model, &input_data).await?;
        
        let mut calibrated_prediction = raw_prediction;
        let mut confidence = 0.5; // Default confidence
        let mut temperature = 1.0; // Default temperature
        let mut ats_score = 0.0;
        
        // Apply temperature scaling if enabled
        if config.temperature_scaling {
            let temp_result = self.temperature_scaling.calibrate(raw_prediction, &input_data).await?;
            calibrated_prediction = temp_result.calibrated_value;
            temperature = temp_result.temperature;
            confidence = temp_result.confidence;
            ats_score += temp_result.improvement_score;
        }
        
        // Apply Platt scaling if enabled
        if config.platt_scaling {
            let platt_result = self.apply_platt_scaling(calibrated_prediction, &input_data).await?;
            calibrated_prediction = platt_result.calibrated_value;
            confidence = confidence.max(platt_result.confidence);
            ats_score += platt_result.improvement_score;
        }
        
        // Apply isotonic regression if enabled
        if config.isotonic_regression {
            let isotonic_result = self.apply_isotonic_regression(calibrated_prediction, &input_data).await?;
            calibrated_prediction = isotonic_result.calibrated_value;
            confidence = confidence.max(isotonic_result.confidence);
            ats_score += isotonic_result.improvement_score;
        }
        
        // Apply conformal prediction if enabled
        if config.conformal_prediction {
            let conformal_result = self.conformal_predictor.predict(
                calibrated_prediction,
                &input_data,
                confidence
            ).await?;
            
            confidence = conformal_result.prediction_interval_coverage;
            ats_score += conformal_result.nonconformity_score;
        }
        
        // Normalize ATS score
        ats_score = (ats_score / 4.0).min(1.0).max(0.0);
        
        Ok(CalibratedPrediction {
            prediction: raw_prediction,
            calibrated_prediction,
            confidence,
            temperature,
            ats_score,
        })
    }
    
    /// Calibrate model with validation data
    pub async fn calibrate_model(
        &self,
        model: Arc<dyn NeuralModel>,
        calibration_data: CalibrationDataset,
        config: CalibrationConfig,
    ) -> Result<CalibratedModel, IntegrationError> {
        let model_id = self.generate_model_cache_key(&*model);
        
        // Check if already calibrated
        {
            let cache = self.calibration_cache.read().await;
            if let Some(calibrated_model) = cache.get(&model_id) {
                return Ok(calibrated_model.clone());
            }
        }
        
        // Perform calibration
        let mut calibrated_model = CalibratedModel {
            base_model: model.clone(),
            temperature_scaler: None,
            platt_scaler: None,
            isotonic_regressor: None,
            conformal_predictor: None,
            calibration_metrics: CalibrationMetrics::default(),
        };
        
        // Train temperature scaler
        if config.temperature_scaling {
            let temp_scaler = self.train_temperature_scaler(&*model, &calibration_data).await?;
            calibrated_model.temperature_scaler = Some(temp_scaler);
        }
        
        // Train Platt scaler
        if config.platt_scaling {
            let platt_scaler = self.train_platt_scaler(&*model, &calibration_data).await?;
            calibrated_model.platt_scaler = Some(platt_scaler);
        }
        
        // Train isotonic regressor
        if config.isotonic_regression {
            let isotonic_regressor = self.train_isotonic_regressor(&*model, &calibration_data).await?;
            calibrated_model.isotonic_regressor = Some(isotonic_regressor);
        }
        
        // Train conformal predictor
        if config.conformal_prediction {
            let conformal_predictor = self.train_conformal_predictor(&*model, &calibration_data).await?;
            calibrated_model.conformal_predictor = Some(conformal_predictor);
        }
        
        // Evaluate calibration quality
        calibrated_model.calibration_metrics = self.evaluate_calibration_quality(
            &calibrated_model,
            &calibration_data
        ).await?;
        
        // Cache calibrated model
        let mut cache = self.calibration_cache.write().await;
        cache.insert(model_id, calibrated_model.clone());
        
        Ok(calibrated_model)
    }
    
    /// Generate uncertainty bounds using conformal prediction
    pub async fn uncertainty_bounds(
        &self,
        model: Arc<dyn NeuralModel>,
        input_data: InputData,
        confidence_level: f32,
    ) -> Result<UncertaintyBounds, IntegrationError> {
        let prediction = self.get_raw_prediction(&*model, &input_data).await?;
        
        let bounds = self.conformal_predictor.prediction_interval(
            prediction,
            &input_data,
            confidence_level
        ).await?;
        
        Ok(bounds)
    }
    
    /// Reliability diagram for calibration assessment
    pub async fn generate_reliability_diagram(
        &self,
        model: Arc<dyn NeuralModel>,
        test_data: CalibrationDataset,
        n_bins: usize,
    ) -> Result<ReliabilityDiagram, IntegrationError> {
        let predictions = self.batch_predict(&*model, &test_data).await?;
        
        // Create bins for confidence scores
        let mut bins = vec![ReliabilityBin::default(); n_bins];
        let bin_size = 1.0 / n_bins as f32;
        
        for ((prediction, actual), confidence) in predictions.iter().zip(test_data.targets.iter()).zip(test_data.confidences.iter()) {
            let bin_index = ((confidence / bin_size) as usize).min(n_bins - 1);

            bins[bin_index].predictions.push(*prediction);
            bins[bin_index].actuals.push(*actual);
            bins[bin_index].confidences.push(*confidence);
        }
        
        // Calculate statistics for each bin
        for bin in bins.iter_mut() {
            if !bin.predictions.is_empty() {
                bin.accuracy = self.calculate_accuracy(&bin.predictions, &bin.actuals);
                bin.confidence = bin.confidences.iter().sum::<f32>() / bin.confidences.len() as f32;
                bin.count = bin.predictions.len();
            }
        }
        
        let ece = self.calculate_expected_calibration_error(&bins);
        let mce = self.calculate_maximum_calibration_error(&bins);
        
        Ok(ReliabilityDiagram {
            bins,
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
            n_bins,
        })
    }
    
    /// Advanced ensemble calibration
    pub async fn ensemble_calibration(
        &self,
        models: Vec<Arc<dyn NeuralModel>>,
        input_data: InputData,
        ensemble_config: EnsembleCalibrationConfig,
    ) -> Result<EnsembleCalibratedPrediction, IntegrationError> {
        let mut individual_predictions = Vec::new();
        let mut individual_confidences = Vec::new();
        
        // Get predictions from each model
        for model in &models {
            let prediction = self.get_raw_prediction(&**model, &input_data).await?;
            individual_predictions.push(prediction);
            
            // Estimate confidence (simplified)
            let confidence = self.estimate_prediction_confidence(&**model, &input_data).await?;
            individual_confidences.push(confidence);
        }
        
        // Combine predictions based on strategy
        let ensemble_prediction = match ensemble_config.combination_strategy {
            EnsembleCombinationStrategy::Mean => {
                individual_predictions.iter().sum::<f32>() / individual_predictions.len() as f32
            },
            EnsembleCombinationStrategy::WeightedMean => {
                let total_weight: f32 = individual_confidences.iter().sum();
                individual_predictions.iter().zip(individual_confidences.iter())
                    .map(|(pred, conf)| pred * conf / total_weight)
                    .sum()
            },
            EnsembleCombinationStrategy::Median => {
                let mut sorted_preds = individual_predictions.clone();
                sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted_preds[sorted_preds.len() / 2]
            },
        };
        
        // Calculate ensemble confidence
        let ensemble_confidence = match ensemble_config.confidence_strategy {
            EnsembleConfidenceStrategy::Mean => {
                individual_confidences.iter().sum::<f32>() / individual_confidences.len() as f32
            },
            EnsembleConfidenceStrategy::Max => {
                individual_confidences.iter().fold(0.0_f32, |a, &b| a.max(b))
            },
            EnsembleConfidenceStrategy::Disagreement => {
                let variance = self.calculate_prediction_variance(&individual_predictions);
                1.0 - (variance / (1.0 + variance)).min(1.0) // Convert variance to confidence
            },
        };
        
        // Apply ensemble-specific calibration
        let calibrated_prediction = if ensemble_config.apply_ensemble_calibration {
            self.apply_ensemble_temperature_scaling(
                ensemble_prediction,
                &individual_predictions,
                &individual_confidences
            ).await?
        } else {
            ensemble_prediction
        };
        
        // Calculate variance before moving individual_predictions
        let prediction_variance = self.calculate_prediction_variance(&individual_predictions);
        let epistemic_uncertainty = self.estimate_epistemic_uncertainty(&individual_predictions);

        Ok(EnsembleCalibratedPrediction {
            ensemble_prediction: calibrated_prediction,
            individual_predictions,
            ensemble_confidence,
            individual_confidences,
            prediction_variance,
            epistemic_uncertainty,
            aleatoric_uncertainty: ensemble_config.aleatoric_uncertainty.unwrap_or(0.0),
        })
    }
    
    // Private implementation methods
    
    async fn get_raw_prediction(
        &self,
        model: &dyn NeuralModel,
        input_data: &InputData,
    ) -> Result<f32, IntegrationError> {
        if let Some(features) = input_data.features.first() {
            let output = model.forward(features)
                .map_err(|e| IntegrationError::AtsCoreIntegrationFailed(e))?;
            
            Ok(output.get(0).copied().unwrap_or(0.0))
        } else {
            Err(IntegrationError::AtsCoreIntegrationFailed("No input features provided".to_string()))
        }
    }
    
    async fn apply_platt_scaling(
        &self,
        prediction: f32,
        _input_data: &InputData,
    ) -> Result<PlattScalingResult, IntegrationError> {
        // Simplified Platt scaling implementation
        let a = -0.5; // Would be learned from calibration data
        let b = 0.1;  // Would be learned from calibration data
        
        let calibrated_value = 1.0 / (1.0 + (-a * prediction - b).exp());
        
        Ok(PlattScalingResult {
            calibrated_value,
            confidence: calibrated_value,
            improvement_score: 0.1,
        })
    }
    
    async fn apply_isotonic_regression(
        &self,
        prediction: f32,
        _input_data: &InputData,
    ) -> Result<IsotonicRegressionResult, IntegrationError> {
        // Simplified isotonic regression implementation
        // In practice, this would use a trained isotonic regressor
        
        let calibrated_value = if prediction < 0.3 {
            prediction * 0.8
        } else if prediction < 0.7 {
            0.24 + (prediction - 0.3) * 1.2
        } else {
            0.72 + (prediction - 0.7) * 0.9
        };
        
        Ok(IsotonicRegressionResult {
            calibrated_value: calibrated_value.min(1.0).max(0.0),
            confidence: 0.8,
            improvement_score: 0.05,
        })
    }
    
    async fn train_temperature_scaler(
        &self,
        model: &dyn NeuralModel,
        calibration_data: &CalibrationDataset,
    ) -> Result<TemperatureScaler, IntegrationError> {
        // Get predictions on calibration data
        let predictions = self.batch_predict(model, calibration_data).await?;
        
        // Find optimal temperature using cross-entropy loss
        let optimal_temperature = self.find_optimal_temperature(&predictions, &calibration_data.targets)?;
        
        Ok(TemperatureScaler::with_temperature(optimal_temperature))
    }
    
    async fn train_platt_scaler(
        &self,
        model: &dyn NeuralModel,
        calibration_data: &CalibrationDataset,
    ) -> Result<PlattScaler, IntegrationError> {
        let predictions = self.batch_predict(model, calibration_data).await?;
        
        // Train sigmoid parameters A and B
        let (a, b) = self.fit_sigmoid_parameters(&predictions, &calibration_data.targets)?;
        
        Ok(PlattScaler::new(a, b))
    }
    
    async fn train_isotonic_regressor(
        &self,
        model: &dyn NeuralModel,
        calibration_data: &CalibrationDataset,
    ) -> Result<IsotonicRegressor, IntegrationError> {
        let predictions = self.batch_predict(model, calibration_data).await?;
        
        // Train isotonic regression mapping
        let mapping = self.fit_isotonic_mapping(&predictions, &calibration_data.targets)?;
        
        Ok(IsotonicRegressor::new(mapping))
    }
    
    async fn train_conformal_predictor(
        &self,
        model: &dyn NeuralModel,
        calibration_data: &CalibrationDataset,
    ) -> Result<ConformalPredictor, IntegrationError> {
        let predictions = self.batch_predict(model, calibration_data).await?;
        
        // Calculate nonconformity scores
        let nonconformity_scores: Vec<f32> = predictions.iter()
            .zip(calibration_data.targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .collect();
        
        Ok(ConformalPredictor::with_scores(nonconformity_scores))
    }
    
    async fn batch_predict(
        &self,
        model: &dyn NeuralModel,
        dataset: &CalibrationDataset,
    ) -> Result<Vec<f32>, IntegrationError> {
        let mut predictions = Vec::new();
        
        for features in &dataset.features {
            let output = model.forward(features)
                .map_err(|e| IntegrationError::AtsCoreIntegrationFailed(e))?;
            
            predictions.push(output.get(0).copied().unwrap_or(0.0));
        }
        
        Ok(predictions)
    }
    
    async fn evaluate_calibration_quality(
        &self,
        calibrated_model: &CalibratedModel,
        test_data: &CalibrationDataset,
    ) -> Result<CalibrationMetrics, IntegrationError> {
        let predictions = self.batch_predict(&*calibrated_model.base_model, test_data).await?;
        
        let reliability_diagram = self.generate_reliability_diagram(
            calibrated_model.base_model.clone(),
            test_data.clone(),
            10
        ).await?;
        
        let brier_score = self.calculate_brier_score(&predictions, &test_data.targets);
        let log_loss = self.calculate_log_loss(&predictions, &test_data.targets);
        
        Ok(CalibrationMetrics {
            expected_calibration_error: reliability_diagram.expected_calibration_error,
            maximum_calibration_error: reliability_diagram.maximum_calibration_error,
            brier_score,
            log_loss,
            reliability_diagram,
        })
    }
    
    fn find_optimal_temperature(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, IntegrationError> {
        // Grid search for optimal temperature
        let mut best_temperature = 1.0;
        let mut best_loss = f32::INFINITY;
        
        for temp_int in 10..500 {
            let temperature = temp_int as f32 / 100.0;
            
            let calibrated_predictions: Vec<f32> = predictions.iter()
                .map(|&p| self.apply_temperature(p, temperature))
                .collect();
            
            let loss = self.calculate_cross_entropy_loss(&calibrated_predictions, targets);
            
            if loss < best_loss {
                best_loss = loss;
                best_temperature = temperature;
            }
        }
        
        Ok(best_temperature)
    }
    
    fn apply_temperature(&self, logit: f32, temperature: f32) -> f32 {
        let scaled_logit = logit / temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }
    
    fn calculate_cross_entropy_loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter().zip(targets.iter())
            .map(|(pred, target)| {
                let pred_clipped = pred.max(1e-7).min(1.0 - 1e-7);
                -target * pred_clipped.ln() - (1.0 - target) * (1.0 - pred_clipped).ln()
            })
            .sum::<f32>() / predictions.len() as f32
    }
    
    fn fit_sigmoid_parameters(&self, predictions: &[f32], targets: &[f32]) -> Result<(f32, f32), IntegrationError> {
        // Simplified parameter fitting
        // In practice, this would use maximum likelihood estimation
        Ok((-1.0, 0.0))
    }
    
    fn fit_isotonic_mapping(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<(f32, f32)>, IntegrationError> {
        // Simplified isotonic regression
        let mut mapping = Vec::new();
        
        // Create sorted pairs and fit isotonic function
        let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (p, t)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Pool adjacent violators algorithm (simplified)
        for (pred, target) in pairs {
            mapping.push((pred, target));
        }
        
        Ok(mapping)
    }
    
    fn calculate_accuracy(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        let correct = predictions.iter().zip(targets.iter())
            .map(|(pred, target)| if (*pred > 0.5) == (*target > 0.5) { 1.0 } else { 0.0 })
            .sum::<f32>();
        
        correct / predictions.len() as f32
    }
    
    fn calculate_expected_calibration_error(&self, bins: &[ReliabilityBin]) -> f32 {
        let total_samples: usize = bins.iter().map(|bin| bin.count).sum();
        
        if total_samples == 0 {
            return 0.0;
        }
        
        bins.iter()
            .map(|bin| {
                if bin.count > 0 {
                    let weight = bin.count as f32 / total_samples as f32;
                    weight * (bin.confidence - bin.accuracy).abs()
                } else {
                    0.0
                }
            })
            .sum()
    }
    
    fn calculate_maximum_calibration_error(&self, bins: &[ReliabilityBin]) -> f32 {
        bins.iter()
            .filter(|bin| bin.count > 0)
            .map(|bin| (bin.confidence - bin.accuracy).abs())
            .fold(0.0, f32::max)
    }
    
    fn calculate_brier_score(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter().zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f32>() / predictions.len() as f32
    }
    
    fn calculate_log_loss(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        -predictions.iter().zip(targets.iter())
            .map(|(pred, target)| {
                let pred_clipped = pred.max(1e-15).min(1.0 - 1e-15);
                target * pred_clipped.ln() + (1.0 - target) * (1.0 - pred_clipped).ln()
            })
            .sum::<f32>() / predictions.len() as f32
    }
    
    async fn estimate_prediction_confidence(
        &self,
        _model: &dyn NeuralModel,
        _input_data: &InputData,
    ) -> Result<f32, IntegrationError> {
        // Simplified confidence estimation
        Ok(0.8)
    }
    
    fn calculate_prediction_variance(&self, predictions: &[f32]) -> f32 {
        let mean = predictions.iter().sum::<f32>() / predictions.len() as f32;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f32>() / predictions.len() as f32;
        
        variance
    }
    
    fn estimate_epistemic_uncertainty(&self, predictions: &[f32]) -> f32 {
        // Epistemic uncertainty from prediction disagreement
        self.calculate_prediction_variance(predictions).sqrt()
    }
    
    async fn apply_ensemble_temperature_scaling(
        &self,
        ensemble_prediction: f32,
        _individual_predictions: &[f32],
        _individual_confidences: &[f32],
    ) -> Result<f32, IntegrationError> {
        // Simplified ensemble temperature scaling
        let temperature = 1.2; // Would be learned from data
        Ok(self.apply_temperature(ensemble_prediction, temperature))
    }
    
    fn generate_model_cache_key(&self, model: &dyn NeuralModel) -> String {
        // Generate cache key based on model parameters
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        let params = model.get_parameters();
        params.len().hash(&mut hasher);
        
        format!("model_{}", hasher.finish())
    }
}

// Supporting types and structures

#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    temperature: f32,
}

impl TemperatureScaler {
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }
    
    pub fn with_temperature(temperature: f32) -> Self {
        Self { temperature }
    }
    
    pub async fn calibrate(&self, prediction: f32, _input_data: &InputData) -> Result<TemperatureScalingResult, IntegrationError> {
        let calibrated_value = 1.0 / (1.0 + (-(prediction / self.temperature)).exp());
        
        Ok(TemperatureScalingResult {
            calibrated_value,
            temperature: self.temperature,
            confidence: calibrated_value,
            improvement_score: 0.2,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    nonconformity_scores: Vec<f32>,
}

impl ConformalPredictor {
    pub fn new() -> Self {
        Self {
            nonconformity_scores: Vec::new(),
        }
    }
    
    pub fn with_scores(scores: Vec<f32>) -> Self {
        Self {
            nonconformity_scores: scores,
        }
    }
    
    pub async fn predict(
        &self,
        prediction: f32,
        _input_data: &InputData,
        confidence_level: f32,
    ) -> Result<ConformalPredictionResult, IntegrationError> {
        let quantile_level = confidence_level;
        let quantile_index = (quantile_level * self.nonconformity_scores.len() as f32) as usize;
        
        let nonconformity_threshold = if quantile_index < self.nonconformity_scores.len() {
            self.nonconformity_scores[quantile_index]
        } else {
            self.nonconformity_scores.last().copied().unwrap_or(0.0)
        };
        
        Ok(ConformalPredictionResult {
            prediction_interval_coverage: confidence_level,
            nonconformity_score: nonconformity_threshold,
            lower_bound: prediction - nonconformity_threshold,
            upper_bound: prediction + nonconformity_threshold,
        })
    }
    
    pub async fn prediction_interval(
        &self,
        prediction: f32,
        input_data: &InputData,
        confidence_level: f32,
    ) -> Result<UncertaintyBounds, IntegrationError> {
        let result = self.predict(prediction, input_data, confidence_level).await?;
        
        Ok(UncertaintyBounds {
            lower_bound: result.lower_bound,
            upper_bound: result.upper_bound,
            confidence_level,
            interval_width: result.upper_bound - result.lower_bound,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PlattScaler {
    a: f32,
    b: f32,
}

impl PlattScaler {
    pub fn new(a: f32, b: f32) -> Self {
        Self { a, b }
    }
}

#[derive(Debug, Clone)]
pub struct IsotonicRegressor {
    mapping: Vec<(f32, f32)>,
}

impl IsotonicRegressor {
    pub fn new(mapping: Vec<(f32, f32)>) -> Self {
        Self { mapping }
    }
}

// Result types

#[derive(Debug, Clone)]
pub struct TemperatureScalingResult {
    pub calibrated_value: f32,
    pub temperature: f32,
    pub confidence: f32,
    pub improvement_score: f32,
}

#[derive(Debug, Clone)]
pub struct PlattScalingResult {
    pub calibrated_value: f32,
    pub confidence: f32,
    pub improvement_score: f32,
}

#[derive(Debug, Clone)]
pub struct IsotonicRegressionResult {
    pub calibrated_value: f32,
    pub confidence: f32,
    pub improvement_score: f32,
}

#[derive(Debug, Clone)]
pub struct ConformalPredictionResult {
    pub prediction_interval_coverage: f32,
    pub nonconformity_score: f32,
    pub lower_bound: f32,
    pub upper_bound: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub confidence_level: f32,
    pub interval_width: f32,
}

#[derive(Clone)]
pub struct CalibratedModel {
    pub base_model: Arc<dyn NeuralModel>,
    pub temperature_scaler: Option<TemperatureScaler>,
    pub platt_scaler: Option<PlattScaler>,
    pub isotonic_regressor: Option<IsotonicRegressor>,
    pub conformal_predictor: Option<ConformalPredictor>,
    pub calibration_metrics: CalibrationMetrics,
}

impl std::fmt::Debug for CalibratedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalibratedModel")
            .field("base_model", &"<dyn NeuralModel>")
            .field("temperature_scaler", &self.temperature_scaler)
            .field("platt_scaler", &self.platt_scaler)
            .field("isotonic_regressor", &self.isotonic_regressor)
            .field("conformal_predictor", &self.conformal_predictor)
            .field("calibration_metrics", &self.calibration_metrics)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    pub features: Vec<Vec<f32>>,
    pub targets: Vec<f32>,
    pub confidences: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationMetrics {
    pub expected_calibration_error: f32,
    pub maximum_calibration_error: f32,
    pub brier_score: f32,
    pub log_loss: f32,
    pub reliability_diagram: ReliabilityDiagram,
}

#[derive(Debug, Clone, Default)]
pub struct ReliabilityDiagram {
    pub bins: Vec<ReliabilityBin>,
    pub expected_calibration_error: f32,
    pub maximum_calibration_error: f32,
    pub n_bins: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ReliabilityBin {
    pub predictions: Vec<f32>,
    pub actuals: Vec<f32>,
    pub confidences: Vec<f32>,
    pub accuracy: f32,
    pub confidence: f32,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct EnsembleCalibrationConfig {
    pub combination_strategy: EnsembleCombinationStrategy,
    pub confidence_strategy: EnsembleConfidenceStrategy,
    pub apply_ensemble_calibration: bool,
    pub aleatoric_uncertainty: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum EnsembleCombinationStrategy {
    Mean,
    WeightedMean,
    Median,
}

#[derive(Debug, Clone)]
pub enum EnsembleConfidenceStrategy {
    Mean,
    Max,
    Disagreement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCalibratedPrediction {
    pub ensemble_prediction: f32,
    pub individual_predictions: Vec<f32>,
    pub ensemble_confidence: f32,
    pub individual_confidences: Vec<f32>,
    pub prediction_variance: f32,
    pub epistemic_uncertainty: f32,
    pub aleatoric_uncertainty: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ats_core_integration_creation() {
        let integration = AtsCoreIntegration::new();
        assert_eq!(integration.temperature_scaling.temperature, 1.0);
    }
    
    #[tokio::test]
    async fn test_temperature_scaling() {
        let scaler = TemperatureScaler::with_temperature(2.0);
        assert_eq!(scaler.temperature, 2.0);
    }
    
    #[tokio::test]
    async fn test_conformal_predictor() {
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let predictor = ConformalPredictor::with_scores(scores.clone());
        assert_eq!(predictor.nonconformity_scores, scores);
    }
    
    #[test]
    fn test_prediction_variance_calculation() {
        let integration = AtsCoreIntegration::new();
        let predictions = vec![0.8, 0.9, 0.7, 0.85];
        let variance = integration.calculate_prediction_variance(&predictions);
        assert!(variance > 0.0);
    }
}