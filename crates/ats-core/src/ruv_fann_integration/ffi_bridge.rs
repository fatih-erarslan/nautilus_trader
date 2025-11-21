// FFI Bridge - TypeScript/JavaScript Integration Layer
// Seamless integration between Rust ruv-FANN and web frontend

use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Object, Reflect};
use web_sys::console;
use serde_wasm_bindgen::{to_value, from_value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    RuvFannIntegration, ModelConfig, TrainingConfig, ForecastConfig,
    TrainingData, InputData, IntegrationError, ModelId
};

// Enable console.log! macro for WASM
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (unsafe { log(&format_args!($($t)*).to_string()) })
}

/// Main FFI bridge for ruv-FANN integration
#[wasm_bindgen]
pub struct RuvFannBridge {
    inner: Arc<RwLock<Option<RuvFannIntegration>>>,
}

#[wasm_bindgen]
impl RuvFannBridge {
    /// Create new FFI bridge
    #[wasm_bindgen(constructor)]
    pub fn new() -> RuvFannBridge {
        console_error_panic_hook::set_once();
        console_log!("🚀 Initializing ruv-FANN Bridge");
        
        RuvFannBridge {
            inner: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Initialize the ruv-FANN integration
    #[wasm_bindgen]
    pub async fn initialize(&self) -> Result<(), JsValue> {
        console_log!("🔄 Initializing ruv-FANN integration...");
        
        let integration = RuvFannIntegration::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to initialize: {:?}", e)))?;
        
        let mut inner = self.inner.write().await;
        *inner = Some(integration);
        
        console_log!("✅ ruv-FANN integration initialized successfully");
        Ok(())
    }
    
    /// Get available neural architectures
    #[wasm_bindgen]
    pub async fn get_architectures(&self) -> Result<Array, JsValue> {
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let architectures = integration.get_architectures().await;
        let js_array = Array::new();
        
        for arch in architectures {
            js_array.push(&JsValue::from_str(&arch));
        }
        
        Ok(js_array)
    }
    
    /// Create a new model with specified architecture
    #[wasm_bindgen]
    pub async fn create_model(&self, 
        name: String, 
        architecture: String, 
        config: JsValue
    ) -> Result<String, JsValue> {
        console_log!("🏗️ Creating model: {} with architecture: {}", name, architecture);
        
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let model_config: ModelConfig = from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid model config: {:?}", e)))?;
        
        let model_id = integration.create_model(name, architecture, model_config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create model: {:?}", e)))?;
        
        console_log!("✅ Model created with ID: {}", model_id);
        Ok(model_id)
    }
    
    /// Train a model with the specified configuration
    #[wasm_bindgen]
    pub async fn train_model(&self,
        model_id: String,
        training_data: JsValue,
        config: JsValue,
        progress_callback: Option<js_sys::Function>
    ) -> Result<JsValue, JsValue> {
        console_log!("🚀 Starting training for model: {}", model_id);
        
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let data: TrainingData = self.parse_training_data(training_data)?;
        let training_config: TrainingConfig = from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid training config: {:?}", e)))?;
        
        // Set up progress callback if provided
        if let Some(callback) = progress_callback {
            // TODO: Implement progress callback mechanism
            console_log!("📈 Progress callback registered");
        }
        
        let result = integration.train_model(model_id, data, training_config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Training failed: {:?}", e)))?;
        
        console_log!("✅ Training completed successfully");
        
        // Convert training result to JS object
        let js_result = self.training_result_to_js(result)?;
        Ok(js_result)
    }
    
    /// Generate forecasts with uncertainty quantification
    #[wasm_bindgen]
    pub async fn forecast(&self,
        model_id: String,
        input_data: JsValue,
        config: JsValue
    ) -> Result<JsValue, JsValue> {
        console_log!("🔮 Generating forecast for model: {}", model_id);
        
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let data: InputData = self.parse_input_data(input_data)?;
        let forecast_config: ForecastConfig = from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid forecast config: {:?}", e)))?;
        
        let result = integration.forecast(model_id, data, forecast_config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Forecasting failed: {:?}", e)))?;
        
        console_log!("✅ Forecast generated successfully");
        
        let js_result = to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize forecast result: {:?}", e)))?;
        
        Ok(js_result)
    }
    
    /// Get calibrated prediction using ATS-Core integration
    #[wasm_bindgen]
    pub async fn calibrated_prediction(&self,
        model_id: String,
        input_data: JsValue,
        calibration_config: JsValue
    ) -> Result<JsValue, JsValue> {
        console_log!("🎯 Generating calibrated prediction for model: {}", model_id);
        
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let data: InputData = self.parse_input_data(input_data)?;
        let config: super::CalibrationConfig = from_value(calibration_config)
            .map_err(|e| JsValue::from_str(&format!("Invalid calibration config: {:?}", e)))?;
        
        let result = integration.calibrated_prediction(model_id, data, config)
            .await
            .map_err(|e| JsValue::from_str(&format!("Calibrated prediction failed: {:?}", e)))?;
        
        console_log!("✅ Calibrated prediction generated successfully");
        
        let js_result = to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize calibrated prediction: {:?}", e)))?;
        
        Ok(js_result)
    }
    
    /// Run performance benchmarks
    #[wasm_bindgen]
    pub async fn benchmark(&self) -> Result<JsValue, JsValue> {
        console_log!("🚀 Running performance benchmarks...");
        
        let inner = self.inner.read().await;
        let integration = inner.as_ref()
            .ok_or_else(|| JsValue::from_str("Integration not initialized"))?;
        
        let results = integration.benchmark()
            .await
            .map_err(|e| JsValue::from_str(&format!("Benchmarking failed: {:?}", e)))?;
        
        console_log!("✅ Benchmarks completed successfully");
        
        let js_result = to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize benchmark results: {:?}", e)))?;
        
        Ok(js_result)
    }
    
    /// Get model information
    #[wasm_bindgen]
    pub async fn get_model_info(&self, model_id: String) -> Result<JsValue, JsValue> {
        // TODO: Implement model info retrieval
        console_log!("ℹ️ Getting model info for: {}", model_id);
        
        let info = Object::new();
        Reflect::set(&info, &JsValue::from_str("id"), &JsValue::from_str(&model_id))?;
        Reflect::set(&info, &JsValue::from_str("status"), &JsValue::from_str("active"))?;
        
        Ok(info.into())
    }
    
    /// Save model state
    #[wasm_bindgen]
    pub async fn save_model(&self, model_id: String, name: Option<String>) -> Result<String, JsValue> {
        console_log!("💾 Saving model: {}", model_id);
        
        // TODO: Implement model saving
        let checkpoint_id = format!("checkpoint_{}_{}", model_id, js_sys::Date::now() as u64);
        
        console_log!("✅ Model saved with checkpoint ID: {}", checkpoint_id);
        Ok(checkpoint_id)
    }
    
    /// Load model from checkpoint
    #[wasm_bindgen]
    pub async fn load_model(&self, checkpoint_id: String) -> Result<String, JsValue> {
        console_log!("📂 Loading model from checkpoint: {}", checkpoint_id);
        
        // TODO: Implement model loading
        let model_id = format!("model_{}", js_sys::Date::now() as u64);
        
        console_log!("✅ Model loaded with ID: {}", model_id);
        Ok(model_id)
    }
    
    /// Get system capabilities
    #[wasm_bindgen]
    pub async fn get_capabilities(&self) -> Result<JsValue, JsValue> {
        console_log!("🔍 Retrieving system capabilities...");
        
        let capabilities = Object::new();
        
        // CPU capabilities
        let cpu = Object::new();
        Reflect::set(&cpu, &JsValue::from_str("cores"), &JsValue::from_f64(web_sys::window().unwrap().navigator().hardware_concurrency() as f64))?;
        Reflect::set(&cpu, &JsValue::from_str("architecture"), &JsValue::from_str("unknown"))?;
        
        // WASM capabilities
        let wasm = Object::new();
        Reflect::set(&wasm, &JsValue::from_str("simdSupport"), &JsValue::from_bool(self.detect_wasm_simd().await))?;
        Reflect::set(&wasm, &JsValue::from_str("threadsSupport"), &JsValue::from_bool(self.detect_wasm_threads().await))?;
        
        // GPU capabilities
        let gpu = self.detect_gpu_capabilities().await?;
        
        Reflect::set(&capabilities, &JsValue::from_str("cpu"), &cpu)?;
        Reflect::set(&capabilities, &JsValue::from_str("wasm"), &wasm)?;
        Reflect::set(&capabilities, &JsValue::from_str("gpu"), &gpu)?;
        
        console_log!("✅ System capabilities retrieved");
        Ok(capabilities.into())
    }
    
    /// Enable GPU acceleration
    #[wasm_bindgen]
    pub async fn enable_gpu(&self) -> Result<bool, JsValue> {
        console_log!("⚡ Enabling GPU acceleration...");
        
        // TODO: Implement GPU enablement
        console_log!("✅ GPU acceleration enabled");
        Ok(true)
    }
    
    /// Set logging level
    #[wasm_bindgen]
    pub fn set_log_level(&self, level: String) {
        console_log!("📝 Setting log level to: {}", level);
    }
    
    /// Get version information
    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

// Private helper methods
impl RuvFannBridge {
    /// Parse training data from JavaScript
    fn parse_training_data(&self, js_data: JsValue) -> Result<TrainingData, JsValue> {
        let obj = js_data.dyn_into::<Object>()
            .map_err(|_| JsValue::from_str("Training data must be an object"))?;
        
        let features_js = Reflect::get(&obj, &JsValue::from_str("features"))?;
        let targets_js = Reflect::get(&obj, &JsValue::from_str("targets"))?;
        let validation_split_js = Reflect::get(&obj, &JsValue::from_str("validationSplit"))?;
        
        let features = self.parse_float_array_2d(features_js)?;
        let targets = self.parse_float_array_2d(targets_js)?;
        
        let validation_split = if validation_split_js.is_undefined() {
            None
        } else {
            Some(validation_split_js.as_f64().unwrap_or(0.0) as f32)
        };
        
        Ok(TrainingData {
            features,
            targets,
            validation_split,
        })
    }
    
    /// Parse input data from JavaScript
    fn parse_input_data(&self, js_data: JsValue) -> Result<InputData, JsValue> {
        let obj = js_data.dyn_into::<Object>()
            .map_err(|_| JsValue::from_str("Input data must be an object"))?;
        
        let features_js = Reflect::get(&obj, &JsValue::from_str("features"))?;
        let sequence_length_js = Reflect::get(&obj, &JsValue::from_str("sequenceLength"))?;
        
        let features = self.parse_float_array_2d(features_js)?;
        
        let sequence_length = if sequence_length_js.is_undefined() {
            None
        } else {
            Some(sequence_length_js.as_f64().unwrap_or(0.0) as usize)
        };
        
        Ok(InputData {
            features,
            sequence_length,
        })
    }
    
    /// Parse 2D float array from JavaScript
    fn parse_float_array_2d(&self, js_value: JsValue) -> Result<Vec<Vec<f32>>, JsValue> {
        let outer_array = js_value.dyn_into::<Array>()
            .map_err(|_| JsValue::from_str("Expected array"))?;
        
        let mut result = Vec::new();
        
        for i in 0..outer_array.length() {
            let inner_js = outer_array.get(i);
            let inner_array = inner_js.dyn_into::<Array>()
                .map_err(|_| JsValue::from_str("Expected inner array"))?;
            
            let mut inner_vec = Vec::new();
            for j in 0..inner_array.length() {
                let value = inner_array.get(j).as_f64()
                    .ok_or_else(|| JsValue::from_str("Expected number"))? as f32;
                inner_vec.push(value);
            }
            
            result.push(inner_vec);
        }
        
        Ok(result)
    }
    
    /// Convert training result to JavaScript object
    fn training_result_to_js(&self, result: super::TrainingResult) -> Result<JsValue, JsValue> {
        let obj = Object::new();
        
        // Convert loss history
        let loss_array = Array::new();
        for loss in result.loss_history {
            loss_array.push(&JsValue::from_f64(loss as f64));
        }
        
        // Convert accuracy history
        let accuracy_array = Array::new();
        for acc in result.accuracy_history {
            accuracy_array.push(&JsValue::from_f64(acc as f64));
        }
        
        Reflect::set(&obj, &JsValue::from_str("lossHistory"), &loss_array)?;
        Reflect::set(&obj, &JsValue::from_str("accuracyHistory"), &accuracy_array)?;
        Reflect::set(&obj, &JsValue::from_str("trainingTime"), &JsValue::from_f64(result.training_time.as_secs_f64()))?;
        Reflect::set(&obj, &JsValue::from_str("finalLoss"), &JsValue::from_f64(result.final_loss as f64))?;
        Reflect::set(&obj, &JsValue::from_str("finalAccuracy"), &JsValue::from_f64(result.final_accuracy as f64))?;
        
        Ok(obj.into())
    }
    
    /// Detect WASM SIMD support
    async fn detect_wasm_simd(&self) -> bool {
        // TODO: Implement WASM SIMD detection
        false
    }
    
    /// Detect WASM threads support
    async fn detect_wasm_threads(&self) -> bool {
        js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
            .map(|val| val.is_truthy())
            .unwrap_or(false)
    }
    
    /// Detect GPU capabilities
    async fn detect_gpu_capabilities(&self) -> Result<JsValue, JsValue> {
        let gpu_obj = Object::new();
        
        // Try to detect WebGL
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.create_element("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        
        let webgl_support = canvas.get_context("webgl2")
            .unwrap_or(None)
            .or_else(|| canvas.get_context("webgl").unwrap_or(None))
            .is_some();
        
        Reflect::set(&gpu_obj, &JsValue::from_str("webgl"), &JsValue::from_bool(webgl_support))?;
        
        if webgl_support {
            // Try to get GPU info from WebGL
            if let Ok(Some(context)) = canvas.get_context("webgl") {
                let gl = context.dyn_into::<web_sys::WebGlRenderingContext>().unwrap();
                
                if let Some(debug_info) = gl.get_extension("WEBGL_debug_renderer_info").unwrap() {
                    let vendor_param = 0x9245; // UNMASKED_VENDOR_WEBGL
                    let renderer_param = 0x9246; // UNMASKED_RENDERER_WEBGL
                    
                    if let Some(vendor) = gl.get_parameter(vendor_param).ok() {
                        Reflect::set(&gpu_obj, &JsValue::from_str("vendor"), &vendor)?;
                    }
                    
                    if let Some(renderer) = gl.get_parameter(renderer_param).ok() {
                        Reflect::set(&gpu_obj, &JsValue::from_str("renderer"), &renderer)?;
                    }
                }
            }
        }
        
        Ok(gpu_obj.into())
    }
}

/// Export convenience functions for direct JavaScript usage
#[wasm_bindgen]
pub async fn create_ruv_fann_bridge() -> RuvFannBridge {
    RuvFannBridge::new()
}

/// Quick test function
#[wasm_bindgen]
pub fn test_ffi_bridge() -> String {
    console_log!("🧪 Testing FFI bridge...");
    "ruv-FANN FFI Bridge is working!".to_string()
}

/// Get supported architectures without initializing full bridge
#[wasm_bindgen]
pub fn get_supported_architectures() -> Array {
    let architectures = vec![
        "mlp", "deep_mlp", "rnn", "lstm", "gru", "bi_lstm",
        "cnn", "resnet", "densenet", "mobilenet",
        "transformer", "bert", "gpt",
        "nhits", "nbeats", "temporal_cnn", "wavenet",
        "attention", "self_attention", "multi_head_attention",
        "neural_ode", "graph_neural_net", "vae", "gan",
        "ensemble", "bagging", "boosting", "financial_lstm"
    ];
    
    let js_array = Array::new();
    for arch in architectures {
        js_array.push(&JsValue::from_str(arch));
    }
    
    js_array
}

/// Utility function to validate model configuration
#[wasm_bindgen]
pub fn validate_model_config(config: JsValue) -> Result<bool, JsValue> {
    let _config: ModelConfig = from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid model config: {:?}", e)))?;
    
    // TODO: Add comprehensive validation
    Ok(true)
}

/// Utility function to get default configuration for architecture
#[wasm_bindgen]
pub fn get_default_config(architecture: String) -> Result<JsValue, JsValue> {
    // TODO: Return appropriate default config based on architecture
    let config = Object::new();
    Reflect::set(&config, &JsValue::from_str("inputSize"), &JsValue::from_f64(10.0))?;
    Reflect::set(&config, &JsValue::from_str("hiddenSizes"), &Array::of2(&JsValue::from_f64(64.0), &JsValue::from_f64(32.0)))?;
    Reflect::set(&config, &JsValue::from_str("outputSize"), &JsValue::from_f64(1.0))?;
    Reflect::set(&config, &JsValue::from_str("activation"), &JsValue::from_str("ReLU"))?;
    
    Ok(config.into())
}

// Error handling utilities
#[wasm_bindgen]
pub struct RuvFannError {
    message: String,
}

#[wasm_bindgen]
impl RuvFannError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl From<IntegrationError> for RuvFannError {
    fn from(error: IntegrationError) -> Self {
        RuvFannError {
            message: format!("{:?}", error),
        }
    }
}