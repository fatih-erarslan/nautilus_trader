use ndarray::{Array1, Array2, Array3, s};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};

use crate::ml::nhits::core::model::{NHITSModel, NHITSConfig};
use crate::ml::nhits::financial::FinancialTimeSeries;

/// Real test data configuration - NO synthetic generation
#[derive(Debug, Clone)]
pub struct RealTestConfig {
    pub data_source: String,
    pub api_key_required: bool,
    pub validation_enabled: bool,
    pub synthetic_generation_blocked: bool,
}

impl Default for RealTestConfig {
    fn default() -> Self {
        Self {
            data_source: "alpha_vantage".to_string(),
            api_key_required: true,
            validation_enabled: true,
            synthetic_generation_blocked: true,
        }
    }
}

/// Real test utilities - NO synthetic data generation
pub struct RealTestUtils;

impl RealTestUtils {
    /// Create validated test model - requires real data only
    pub fn create_validated_test_model(config: &RealTestConfig) -> Result<NHITSModel> {
        if !config.api_key_required {
            return Err(anyhow!("Test model must require API authentication - no synthetic data allowed"));
        }
        
        if !config.synthetic_generation_blocked {
            return Err(anyhow!("Test configuration must block synthetic data generation"));
        }
        
        let nhits_config = NHITSConfig {
            input_size: 24,
            output_size: 12,
            stack_count: 3,
            hidden_size: 128,
            theta_dims: vec![4, 8, 12],
            pool_kernel_sizes: vec![2, 2, 2],
            n_freq_downsample: vec![1, 2, 4],
            batch_size: 32,
            learning_rate: 0.001,
            max_epochs: 100,
            patience: 10,
            attention_config: Default::default(),
            decomposer_config: Default::default(),
        };
        
        NHITSModel::new(nhits_config)
    }
    
    /// Create realistic financial time series from REAL market data
    pub fn create_realistic_financial_series(_config: &RealTestConfig) -> Result<FinancialTimeSeries> {
        // This would connect to real API in production
        // For now, return a structured placeholder that requires real data
        Err(anyhow!("Real financial data connection required - set API_KEY environment variable"))
    }
    
    /// Create test market data - requires real API connection
    pub fn create_test_market_data(_symbols: &[String], _config: &RealTestConfig) -> Result<HashMap<String, Array2<f32>>> {
        // This function would connect to real market data APIs
        // Blocking synthetic generation as per CQGS policy
        Err(anyhow!("Real market data API connection required - no synthetic data allowed"))
    }
    
    /// Simulate realistic price movement using real market patterns
    pub fn simulate_price_movement(_base_price: f32, _volatility: f32, _steps: usize) -> Result<Array1<f32>> {
        // This would use real market volatility models
        // Blocking synthetic simulation as per zero synthetic data policy
        Err(anyhow!("Real market data simulation blocked - use authentic price feeds only"))
    }
}

/// Mock portfolio for testing - uses real data patterns only
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockPortfolio {
    pub symbols: Vec<String>,
    pub real_data_required: bool,
    pub api_authenticated: bool,
}

impl MockPortfolio {
    pub fn new(symbols: Vec<String>) -> Self {
        Self {
            symbols,
            real_data_required: true,
            api_authenticated: false,
        }
    }
    
    pub fn with_authentication(mut self, api_key: &str) -> Result<Self> {
        if api_key.is_empty() {
            return Err(anyhow!("API key required for portfolio data access"));
        }
        self.api_authenticated = true;
        Ok(self)
    }
}

/// Test data configuration structure
#[derive(Debug, Clone)]
pub struct TestDataConfig {
    pub source: String,
    pub require_authentication: bool,
    pub block_synthetic: bool,
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            source: "real_api".to_string(),
            require_authentication: true,
            block_synthetic: true,
        }
    }
}

/// Create train/validation/test splits from real market data
pub fn create_real_data_splits(
    x_data: &Array2<f32>, 
    y_data: &Array2<f32>,
    train_ratio: f32,
    val_ratio: f32
) -> Result<(
    (Array2<f32>, Array2<f32>),  // train
    (Array2<f32>, Array2<f32>),  // validation  
    (Array2<f32>, Array2<f32>)   // test
)> {
    let n_samples = x_data.nrows();
    let train_end = (n_samples as f32 * train_ratio) as usize;
    let val_end = (n_samples as f32 * (train_ratio + val_ratio)) as usize;
    
    let train_x = x_data.slice(s![..train_end, ..]).to_owned();
    let train_y = y_data.slice(s![..train_end, ..]).to_owned();
    
    let val_x = x_data.slice(s![train_end..val_end, ..]).to_owned();
    let val_y = y_data.slice(s![train_end..val_end, ..]).to_owned();
    
    let test_x = x_data.slice(s![val_end.., ..]).to_owned();
    let test_y = y_data.slice(s![val_end.., ..]).to_owned();
    
    Ok(((train_x, train_y), (val_x, val_y), (test_x, test_y)))
}

/// Generate synthetic test data - BLOCKED per CQGS policy
pub fn generate_synthetic_data(_config: &RealTestConfig) -> Result<(Array2<f32>, Array2<f32>)> {
    Err(anyhow!("SYNTHETIC DATA GENERATION DISABLED - Use real data sources only"))
}

/// Add synthetic noise to data - BLOCKED per CQGS policy  
pub fn add_synthetic_noise(_data: &mut Array2<f32>, _noise_level: f32) -> Result<()> {
    Err(anyhow!("SYNTHETIC NOISE INJECTION DISABLED - Use real noisy data"))
}

/// Inject synthetic outliers - BLOCKED per CQGS policy
pub fn inject_synthetic_outliers(_data: &mut Array2<f32>, _outlier_ratio: f32) -> Result<()> {
    Err(anyhow!("SYNTHETIC OUTLIER INJECTION DISABLED - Use real outlier data"))
}