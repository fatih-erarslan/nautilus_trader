//! Configuration system for Neural Forge
//! 
//! Provides comprehensive, type-safe configuration for all training aspects.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use crate::error::{Result, NeuralForgeError};

pub mod model;
pub mod training;
pub mod optimizer;
pub mod scheduler;
pub mod calibration;
pub mod data;
pub mod distributed;

pub use model::*;
pub use training::*;
pub use optimizer::*;
pub use scheduler::*;
pub use calibration::*;
pub use data::*;
pub use distributed::*;

/// Main configuration container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralForgeConfig {
    /// Model configuration
    pub model: ModelConfig,
    
    /// Training configuration
    pub training: TrainingConfig,
    
    /// Data configuration
    pub data: DataConfig,
    
    /// Optimizer configuration
    pub optimizer: OptimizerConfig,
    
    /// Learning rate scheduler configuration
    pub scheduler: Option<SchedulerConfig>,
    
    /// Calibration configuration
    pub calibration: Option<CalibrationConfig>,
    
    /// Distributed training configuration
    pub distributed: Option<DistributedConfig>,
    
    /// Hardware configuration
    pub hardware: HardwareConfig,
    
    /// Logging and monitoring configuration
    pub logging: LoggingConfig,
    
    /// Custom hyperparameters
    pub hyperparams: HashMap<String, serde_json::Value>,
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Device type preference
    pub device: DeviceConfig,
    
    /// Memory management
    pub memory: MemoryConfig,
    
    /// Parallelization settings
    pub parallel: ParallelConfig,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    /// Automatic device selection
    Auto,
    
    /// CPU only
    Cpu { threads: Option<usize> },
    
    /// CUDA GPU
    Cuda { 
        device_id: Option<usize>,
        memory_fraction: Option<f32>,
        allow_growth: bool,
    },
    
    /// Metal GPU (Apple)
    Metal { device_id: Option<usize> },
    
    /// Multiple devices
    Multi { devices: Vec<DeviceConfig> },
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage (bytes)
    pub max_memory: Option<u64>,
    
    /// Memory pool size
    pub pool_size: Option<u64>,
    
    /// Enable memory mapping for large datasets
    pub memory_mapping: bool,
    
    /// Garbage collection frequency
    pub gc_frequency: Option<u32>,
    
    /// Cache size for preprocessed data
    pub cache_size: Option<u64>,
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of data loading threads
    pub data_threads: Option<usize>,
    
    /// Number of preprocessing threads
    pub preprocessing_threads: Option<usize>,
    
    /// Enable NUMA awareness
    pub numa_aware: bool,
    
    /// Thread affinity settings
    pub thread_affinity: Option<Vec<usize>>,
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Output directory
    pub output_dir: PathBuf,
    
    /// Enable TensorBoard logging
    pub tensorboard: bool,
    
    /// Enable Weights & Biases
    pub wandb: Option<WandbConfig>,
    
    /// Enable MLflow
    pub mlflow: Option<MlflowConfig>,
    
    /// Enable metrics collection
    pub metrics: MetricsConfig,
    
    /// Checkpoint frequency
    pub checkpoint_frequency: u32,
    
    /// Save best model only
    pub save_best_only: bool,
}

/// Log level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Weights & Biases configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    pub project: String,
    pub entity: Option<String>,
    pub tags: Vec<String>,
    pub notes: Option<String>,
}

/// MLflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlflowConfig {
    pub tracking_uri: String,
    pub experiment_name: String,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable hardware metrics
    pub hardware: bool,
    
    /// Enable training metrics
    pub training: bool,
    
    /// Enable custom metrics
    pub custom: HashMap<String, bool>,
    
    /// Metrics collection frequency
    pub frequency: u32,
}

impl Default for NeuralForgeConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            data: DataConfig::default(),
            optimizer: OptimizerConfig::default(),
            scheduler: Some(SchedulerConfig::default()),
            calibration: Some(CalibrationConfig::default()),
            distributed: None,
            hardware: HardwareConfig::default(),
            logging: LoggingConfig::default(),
            hyperparams: HashMap::new(),
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Auto,
            memory: MemoryConfig::default(),
            parallel: ParallelConfig::default(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: None,
            pool_size: None,
            memory_mapping: true,
            gc_frequency: Some(100),
            cache_size: Some(1024 * 1024 * 1024), // 1GB
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            data_threads: None, // Auto-detect
            preprocessing_threads: None, // Auto-detect
            numa_aware: false,
            thread_affinity: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            output_dir: PathBuf::from("./logs"),
            tensorboard: true,
            wandb: None,
            mlflow: None,
            metrics: MetricsConfig::default(),
            checkpoint_frequency: 1000,
            save_best_only: true,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            hardware: true,
            training: true,
            custom: HashMap::new(),
            frequency: 100,
        }
    }
}

impl NeuralForgeConfig {
    /// Load configuration from file
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = std::fs::read_to_string(&path)?;
        
        match path.extension().and_then(|s| s.to_str()) {
            Some("json") => Ok(serde_json::from_str(&content)?),
            Some("yaml") | Some("yml") => Ok(serde_yaml::from_str(&content)
                .map_err(|e| NeuralForgeError::config(format!("YAML parse error: {}", e)))?),
            Some("toml") => Ok(toml::from_str(&content)
                .map_err(|e| NeuralForgeError::config(format!("TOML parse error: {}", e)))?),
            _ => Err(NeuralForgeError::config("Unsupported config file format")),
        }
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: impl Into<PathBuf>) -> Result<()> {
        let path = path.into();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("json") => serde_json::to_string_pretty(self)?,
            Some("yaml") | Some("yml") => serde_yaml::to_string(self)
                .map_err(|e| NeuralForgeError::config(format!("YAML serialize error: {}", e)))?,
            Some("toml") => toml::to_string_pretty(self)
                .map_err(|e| NeuralForgeError::config(format!("TOML serialize error: {}", e)))?,
            _ => return Err(NeuralForgeError::config("Unsupported config file format")),
        };
        
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.training.validate()?;
        self.data.validate()?;
        self.optimizer.validate()?;
        
        if let Some(ref scheduler) = self.scheduler {
            scheduler.validate()?;
        }
        
        if let Some(ref calibration) = self.calibration {
            calibration.validate()?;
        }
        
        if let Some(ref distributed) = self.distributed {
            distributed.validate()?;
        }
        
        Ok(())
    }
    
    /// Merge with another configuration (other takes precedence)
    pub fn merge(mut self, other: Self) -> Self {
        // Implement sophisticated merging logic
        self.model = other.model;
        self.training = other.training;
        self.data = other.data;
        self.optimizer = other.optimizer;
        
        if other.scheduler.is_some() {
            self.scheduler = other.scheduler;
        }
        
        if other.calibration.is_some() {
            self.calibration = other.calibration;
        }
        
        if other.distributed.is_some() {
            self.distributed = other.distributed;
        }
        
        self.hardware = other.hardware;
        self.logging = other.logging;
        
        // Merge hyperparams
        for (key, value) in other.hyperparams {
            self.hyperparams.insert(key, value);
        }
        
        self
    }
    
    /// Builder pattern methods
    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.model = model;
        self
    }
    
    pub fn with_training(mut self, training: TrainingConfig) -> Self {
        self.training = training;
        self
    }
    
    pub fn with_data(mut self, data: DataConfig) -> Self {
        self.data = data;
        self
    }
    
    pub fn with_optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.optimizer = optimizer;
        self
    }
    
    pub fn with_scheduler(mut self, scheduler: SchedulerConfig) -> Self {
        self.scheduler = Some(scheduler);
        self
    }
    
    pub fn with_calibration(mut self, calibration: CalibrationConfig) -> Self {
        self.calibration = Some(calibration);
        self
    }
    
    pub fn with_distributed(mut self, distributed: DistributedConfig) -> Self {
        self.distributed = Some(distributed);
        self
    }
    
    pub fn with_hardware(mut self, hardware: HardwareConfig) -> Self {
        self.hardware = hardware;
        self
    }
    
    pub fn with_logging(mut self, logging: LoggingConfig) -> Self {
        self.logging = logging;
        self
    }
    
    pub fn with_hyperparam<T: serde::Serialize>(mut self, key: impl Into<String>, value: T) -> Self {
        self.hyperparams.insert(
            key.into(), 
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
        );
        self
    }
}