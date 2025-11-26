//! Model registry and factory pattern

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::{Result, NeuroDivergentError, NeuralModel, ModelConfig};

/// Model factory function type
pub type ModelFactoryFn = Box<dyn Fn(&ModelConfig) -> Result<Box<dyn NeuralModel>> + Send + Sync>;

/// Model registry for managing available models
pub struct ModelRegistry {
    factories: Arc<RwLock<HashMap<String, ModelFactoryFn>>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            factories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model factory
    pub fn register<F>(&self, name: &str, factory: F) -> Result<()>
    where
        F: Fn(&ModelConfig) -> Result<Box<dyn NeuralModel>> + Send + Sync + 'static,
    {
        let mut factories = self.factories.write()
            .map_err(|e| NeuroDivergentError::Unknown(format!("Lock error: {}", e)))?;

        factories.insert(name.to_string(), Box::new(factory));
        Ok(())
    }

    /// Create a model by name
    pub fn create(&self, name: &str, config: &ModelConfig) -> Result<Box<dyn NeuralModel>> {
        let factories = self.factories.read()
            .map_err(|e| NeuroDivergentError::Unknown(format!("Lock error: {}", e)))?;

        let factory = factories.get(name)
            .ok_or_else(|| NeuroDivergentError::ModelError(
                format!("Model '{}' not registered", name)
            ))?;

        factory(config)
    }

    /// List all registered models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let factories = self.factories.read()
            .map_err(|e| NeuroDivergentError::Unknown(format!("Lock error: {}", e)))?;

        Ok(factories.keys().cloned().collect())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global model factory
pub struct ModelFactory;

impl ModelFactory {
    /// Get the global model registry
    pub fn registry() -> &'static ModelRegistry {
        static INSTANCE: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(ModelRegistry::new)
    }

    /// Create a model by name using the global registry
    pub fn create(name: &str, config: &ModelConfig) -> Result<Box<dyn NeuralModel>> {
        Self::registry().create(name, config)
    }

    /// List all available models
    pub fn list_models() -> Result<Vec<String>> {
        Self::registry().list_models()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModel {
        config: ModelConfig,
    }

    impl NeuralModel for DummyModel {
        fn fit(&mut self, _data: &crate::TimeSeriesDataFrame) -> Result<()> {
            Ok(())
        }

        fn predict(&self, horizon: usize) -> Result<Vec<f64>> {
            Ok(vec![0.0; horizon])
        }

        fn predict_intervals(
            &self,
            _horizon: usize,
            _levels: &[f64],
        ) -> Result<crate::inference::PredictionIntervals> {
            Err(NeuroDivergentError::NotImplemented("Not implemented".to_string()))
        }

        fn name(&self) -> &str {
            "dummy"
        }

        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn save(&self, _path: &std::path::Path) -> Result<()> {
            Ok(())
        }

        fn load(_path: &std::path::Path) -> Result<Self> {
            Err(NeuroDivergentError::NotImplemented("Not implemented".to_string()))
        }
    }

    #[test]
    fn test_registry() {
        let registry = ModelRegistry::new();

        registry.register("dummy", |config| {
            Ok(Box::new(DummyModel { config: config.clone() }))
        }).unwrap();

        let models = registry.list_models().unwrap();
        assert!(models.contains(&"dummy".to_string()));

        let config = ModelConfig::default();
        let model = registry.create("dummy", &config).unwrap();
        assert_eq!(model.name(), "dummy");
    }
}
