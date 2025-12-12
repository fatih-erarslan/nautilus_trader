//! Component registry for managing CDFA analyzers, detectors, and algorithms
//!
//! This module provides a centralized registry system for managing all CDFA components.
//! It supports dynamic component registration, discovery, configuration, and lifecycle
//! management with plugin-like architecture.

use crate::error::{CdfaError, Result};
use crate::traits::*;
use crate::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Central registry for all CDFA components
/// 
/// The ComponentRegistry provides a plugin-like architecture where different
/// types of components (diversity methods, fusion algorithms, detectors, etc.)
/// can be dynamically registered, discovered, and managed.
/// 
/// # Example
/// 
/// ```rust
/// use cdfa_unified::registry::ComponentRegistry;
/// 
/// let mut registry = ComponentRegistry::new();
/// 
/// // Register a custom component
/// registry.register_diversity_method("custom_kendall", my_custom_method)?;
/// 
/// // List available components
/// let available = registry.list_diversity_methods();
/// println!("Available diversity methods: {:?}", available);
/// 
/// // Get a component for use
/// let method = registry.get_diversity_method("kendall")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ComponentRegistry {
    /// Registered diversity calculation methods
    diversity_methods: Arc<RwLock<HashMap<String, Box<dyn DiversityMethod + Send + Sync>>>>,
    
    /// Registered fusion algorithms
    fusion_methods: Arc<RwLock<HashMap<String, Box<dyn FusionMethod + Send + Sync>>>>,
    
    /// Registered pattern detectors
    detectors: Arc<RwLock<HashMap<String, Box<dyn PatternDetector + Send + Sync>>>>,
    
    /// Registered system analyzers
    analyzers: Arc<RwLock<HashMap<String, Box<dyn SystemAnalyzer + Send + Sync>>>>,
    
    /// Registered signal processing algorithms
    algorithms: Arc<RwLock<HashMap<String, Box<dyn SignalAlgorithm + Send + Sync>>>>,
    
    /// Component metadata and configuration
    metadata: Arc<RwLock<HashMap<String, ComponentMetadata>>>,
    
    /// Default configurations for components
    default_configs: Arc<RwLock<HashMap<String, AlgorithmParams>>>,
    
    /// Component dependencies and relationships
    dependencies: Arc<RwLock<HashMap<String, Vec<String>>>>,
    
    /// Component performance metrics
    performance: Arc<RwLock<HashMap<String, ComponentPerformance>>>,
}

/// Metadata about a registered component
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    
    /// Component type (diversity, fusion, detector, etc.)
    pub component_type: ComponentType,
    
    /// Version information
    pub version: String,
    
    /// Author/source information
    pub author: String,
    
    /// Description of the component
    pub description: String,
    
    /// Supported features and capabilities
    pub features: Vec<String>,
    
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    
    /// Minimum data requirements
    pub min_data_length: usize,
    
    /// Component complexity (1=simple, 5=complex)
    pub complexity: u8,
    
    /// Whether component supports real-time processing
    pub realtime_capable: bool,
    
    /// Required features for this component to work
    pub required_features: Vec<String>,
}

/// Types of components that can be registered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    DiversityMethod,
    FusionMethod,
    PatternDetector,
    SystemAnalyzer,
    SignalAlgorithm,
    HardwareBackend,
    CacheBackend,
}

/// Performance profile for a component
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Typical execution time for standard data sizes
    pub typical_time_us: u64,
    
    /// Memory usage characteristics
    pub memory_usage: MemoryUsage,
    
    /// Scalability with data size
    pub scalability: Scalability,
    
    /// Whether the component benefits from parallelization
    pub parallel_benefit: bool,
    
    /// Whether the component benefits from SIMD
    pub simd_benefit: bool,
}

/// Memory usage characteristics
#[derive(Debug, Clone)]
pub enum MemoryUsage {
    Constant,
    Linear,
    Quadratic,
    Exponential,
}

/// Scalability characteristics
#[derive(Debug, Clone)]
pub enum Scalability {
    Excellent,
    Good,
    Moderate,
    Poor,
}

/// Performance tracking for individual components
#[derive(Debug, Clone, Default)]
pub struct ComponentPerformance {
    /// Number of times this component has been used
    pub usage_count: u64,
    
    /// Total execution time
    pub total_time_us: u64,
    
    /// Average execution time
    pub avg_time_us: f64,
    
    /// Best execution time
    pub best_time_us: u64,
    
    /// Worst execution time
    pub worst_time_us: u64,
    
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    
    /// Last used timestamp
    pub last_used: Option<i64>,
}

impl ComponentRegistry {
    /// Create a new component registry
    pub fn new() -> Self {
        let registry = Self {
            diversity_methods: Arc::new(RwLock::new(HashMap::new())),
            fusion_methods: Arc::new(RwLock::new(HashMap::new())),
            detectors: Arc::new(RwLock::new(HashMap::new())),
            analyzers: Arc::new(RwLock::new(HashMap::new())),
            algorithms: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            default_configs: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            performance: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Register default components
        registry.register_default_components();
        
        registry
    }
    
    /// Register default built-in components
    fn register_default_components(&self) {
        // Register core diversity methods
        #[cfg(feature = "core")]
        {
            self.register_diversity_method_internal(
                "kendall".to_string(),
                Box::new(crate::core::diversity::KendallTauDiversity::new()),
                ComponentMetadata {
                    name: "kendall".to_string(),
                    component_type: ComponentType::DiversityMethod,
                    version: "1.0.0".to_string(),
                    author: "CDFA Core Team".to_string(),
                    description: "Kendall tau rank correlation diversity measure".to_string(),
                    features: vec!["rank_correlation".to_string(), "robust".to_string()],
                    performance_profile: PerformanceProfile {
                        typical_time_us: 100,
                        memory_usage: MemoryUsage::Quadratic,
                        scalability: Scalability::Good,
                        parallel_benefit: true,
                        simd_benefit: true,
                    },
                    min_data_length: 3,
                    complexity: 2,
                    realtime_capable: true,
                    required_features: vec!["core".to_string()],
                },
            ).expect("Failed to register Kendall diversity method");
            
            self.register_diversity_method_internal(
                "pearson".to_string(),
                Box::new(crate::core::diversity::PearsonDiversity::new()),
                ComponentMetadata {
                    name: "pearson".to_string(),
                    component_type: ComponentType::DiversityMethod,
                    version: "1.0.0".to_string(),
                    author: "CDFA Core Team".to_string(),
                    description: "Pearson correlation diversity measure".to_string(),
                    features: vec!["linear_correlation".to_string(), "fast".to_string()],
                    performance_profile: PerformanceProfile {
                        typical_time_us: 50,
                        memory_usage: MemoryUsage::Linear,
                        scalability: Scalability::Excellent,
                        parallel_benefit: true,
                        simd_benefit: true,
                    },
                    min_data_length: 2,
                    complexity: 1,
                    realtime_capable: true,
                    required_features: vec!["core".to_string()],
                },
            ).expect("Failed to register Pearson diversity method");
        }
        
        // Register fusion methods
        #[cfg(feature = "core")]
        {
            self.register_fusion_method_internal(
                "weighted_average".to_string(),
                Box::new(crate::core::fusion::WeightedAverageFusion::new()),
                ComponentMetadata {
                    name: "weighted_average".to_string(),
                    component_type: ComponentType::FusionMethod,
                    version: "1.0.0".to_string(),
                    author: "CDFA Core Team".to_string(),
                    description: "Weighted average fusion of multiple signals".to_string(),
                    features: vec!["adaptive_weights".to_string(), "realtime".to_string()],
                    performance_profile: PerformanceProfile {
                        typical_time_us: 20,
                        memory_usage: MemoryUsage::Linear,
                        scalability: Scalability::Excellent,
                        parallel_benefit: true,
                        simd_benefit: true,
                    },
                    min_data_length: 1,
                    complexity: 1,
                    realtime_capable: true,
                    required_features: vec!["core".to_string()],
                },
            ).expect("Failed to register weighted average fusion");
        }
        
        // Register pattern detectors
        #[cfg(feature = "detectors")]
        {
            self.register_detector_internal(
                "fibonacci".to_string(),
                Box::new(crate::detectors::fibonacci::FibonacciPatternDetector::new()),
                ComponentMetadata {
                    name: "fibonacci".to_string(),
                    component_type: ComponentType::PatternDetector,
                    version: "1.0.0".to_string(),
                    author: "CDFA Detectors Team".to_string(),
                    description: "Fibonacci retracement and extension pattern detector".to_string(),
                    features: vec!["harmonic_patterns".to_string(), "price_action".to_string()],
                    performance_profile: PerformanceProfile {
                        typical_time_us: 500,
                        memory_usage: MemoryUsage::Linear,
                        scalability: Scalability::Good,
                        parallel_benefit: false,
                        simd_benefit: false,
                    },
                    min_data_length: 50,
                    complexity: 3,
                    realtime_capable: true,
                    required_features: vec!["detectors".to_string()],
                },
            ).expect("Failed to register Fibonacci detector");
        }
    }
    
    // Public API methods for registering components
    
    /// Register a diversity method
    pub fn register_diversity_method<T>(&self, name: String, method: T) -> Result<()>
    where
        T: DiversityMethod + Send + Sync + 'static,
    {
        let metadata = self.create_default_metadata(name.clone(), ComponentType::DiversityMethod);
        self.register_diversity_method_internal(name, Box::new(method), metadata)
    }
    
    /// Register a fusion method
    pub fn register_fusion_method<T>(&self, name: String, method: T) -> Result<()>
    where
        T: FusionMethod + Send + Sync + 'static,
    {
        let metadata = self.create_default_metadata(name.clone(), ComponentType::FusionMethod);
        self.register_fusion_method_internal(name, Box::new(method), metadata)
    }
    
    /// Register a pattern detector
    pub fn register_detector<T>(&self, name: String, detector: T) -> Result<()>
    where
        T: PatternDetector + Send + Sync + 'static,
    {
        let metadata = self.create_default_metadata(name.clone(), ComponentType::PatternDetector);
        self.register_detector_internal(name, Box::new(detector), metadata)
    }
    
    /// Register a system analyzer
    pub fn register_analyzer<T>(&self, name: String, analyzer: T) -> Result<()>
    where
        T: SystemAnalyzer + Send + Sync + 'static,
    {
        let metadata = self.create_default_metadata(name.clone(), ComponentType::SystemAnalyzer);
        self.register_analyzer_internal(name, Box::new(analyzer), metadata)
    }
    
    /// Register a signal algorithm
    pub fn register_algorithm<T>(&self, name: String, algorithm: T) -> Result<()>
    where
        T: SignalAlgorithm + Send + Sync + 'static,
    {
        let metadata = self.create_default_metadata(name.clone(), ComponentType::SignalAlgorithm);
        self.register_algorithm_internal(name, Box::new(algorithm), metadata)
    }
    
    // Component retrieval methods
    
    /// Get a diversity method by name
    pub fn get_diversity_method(&self, name: &str) -> Result<Arc<dyn DiversityMethod + Send + Sync>> {
        let methods = self.diversity_methods.read();
        let _method = methods.get(name)
            .ok_or_else(|| CdfaError::config_error(format!("Unknown diversity method: {}", name)))?;
        
        // Update usage statistics
        self.update_component_usage(name);
        
        // Note: This is a simplified approach. In a real implementation,
        // you might want to use Arc<dyn Trait> throughout or implement cloning.
        Err(CdfaError::unsupported_operation("Component sharing not implemented"))
    }
    
    /// List all available diversity methods
    pub fn list_diversity_methods(&self) -> Vec<String> {
        self.diversity_methods.read().keys().cloned().collect()
    }
    
    /// List all available fusion methods
    pub fn list_fusion_methods(&self) -> Vec<String> {
        self.fusion_methods.read().keys().cloned().collect()
    }
    
    /// List all available detectors
    pub fn list_detectors(&self) -> Vec<String> {
        self.detectors.read().keys().cloned().collect()
    }
    
    /// List all available analyzers
    pub fn list_analyzers(&self) -> Vec<String> {
        self.analyzers.read().keys().cloned().collect()
    }
    
    /// List all available algorithms
    pub fn list_algorithms(&self) -> Vec<String> {
        self.algorithms.read().keys().cloned().collect()
    }
    
    /// Get component metadata
    pub fn get_metadata(&self, name: &str) -> Option<ComponentMetadata> {
        self.metadata.read().get(name).cloned()
    }
    
    /// Get all component metadata
    pub fn list_all_components(&self) -> HashMap<String, ComponentMetadata> {
        self.metadata.read().clone()
    }
    
    /// Get components by type
    pub fn get_components_by_type(&self, component_type: ComponentType) -> Vec<ComponentMetadata> {
        self.metadata.read()
            .values()
            .filter(|meta| meta.component_type == component_type)
            .cloned()
            .collect()
    }
    
    /// Get components with specific features
    pub fn get_components_with_features(&self, required_features: &[String]) -> Vec<ComponentMetadata> {
        self.metadata.read()
            .values()
            .filter(|meta| {
                required_features.iter()
                    .all(|feature| meta.features.contains(feature))
            })
            .cloned()
            .collect()
    }
    
    /// Get performance statistics for a component
    pub fn get_component_performance(&self, name: &str) -> Option<ComponentPerformance> {
        self.performance.read().get(name).cloned()
    }
    
    /// Get all performance statistics
    pub fn get_all_performance_stats(&self) -> HashMap<String, ComponentPerformance> {
        self.performance.read().clone()
    }
    
    /// Check if a component exists
    pub fn contains(&self, name: &str) -> bool {
        self.metadata.read().contains_key(name)
    }
    
    /// Remove a component
    pub fn remove_component(&self, name: &str) -> Result<()> {
        // Check dependencies first
        let dependents = self.get_dependents(name);
        if !dependents.is_empty() {
            return Err(CdfaError::config_error(format!(
                "Cannot remove component '{}' - it has dependents: {:?}", 
                name, dependents
            )));
        }
        
        // Remove from all registries
        {
            let mut diversity = self.diversity_methods.write();
            diversity.remove(name);
        }
        {
            let mut fusion = self.fusion_methods.write();
            fusion.remove(name);
        }
        {
            let mut detectors = self.detectors.write();
            detectors.remove(name);
        }
        {
            let mut analyzers = self.analyzers.write();
            analyzers.remove(name);
        }
        {
            let mut algorithms = self.algorithms.write();
            algorithms.remove(name);
        }
        {
            let mut metadata = self.metadata.write();
            metadata.remove(name);
        }
        {
            let mut performance = self.performance.write();
            performance.remove(name);
        }
        {
            let mut dependencies = self.dependencies.write();
            dependencies.remove(name);
        }
        
        Ok(())
    }
    
    // Internal implementation methods
    
    fn register_diversity_method_internal(
        &self,
        name: String,
        method: Box<dyn DiversityMethod + Send + Sync>,
        metadata: ComponentMetadata,
    ) -> Result<()> {
        {
            let mut methods = self.diversity_methods.write();
            if methods.contains_key(&name) {
                return Err(CdfaError::config_error(format!(
                    "Diversity method '{}' already registered", name
                )));
            }
            methods.insert(name.clone(), method);
        }
        
        self.register_metadata(name, metadata)?;
        Ok(())
    }
    
    fn register_fusion_method_internal(
        &self,
        name: String,
        method: Box<dyn FusionMethod + Send + Sync>,
        metadata: ComponentMetadata,
    ) -> Result<()> {
        {
            let mut methods = self.fusion_methods.write();
            if methods.contains_key(&name) {
                return Err(CdfaError::config_error(format!(
                    "Fusion method '{}' already registered", name
                )));
            }
            methods.insert(name.clone(), method);
        }
        
        self.register_metadata(name, metadata)?;
        Ok(())
    }
    
    fn register_detector_internal(
        &self,
        name: String,
        detector: Box<dyn PatternDetector + Send + Sync>,
        metadata: ComponentMetadata,
    ) -> Result<()> {
        {
            let mut detectors = self.detectors.write();
            if detectors.contains_key(&name) {
                return Err(CdfaError::config_error(format!(
                    "Detector '{}' already registered", name
                )));
            }
            detectors.insert(name.clone(), detector);
        }
        
        self.register_metadata(name, metadata)?;
        Ok(())
    }
    
    fn register_analyzer_internal(
        &self,
        name: String,
        analyzer: Box<dyn SystemAnalyzer + Send + Sync>,
        metadata: ComponentMetadata,
    ) -> Result<()> {
        {
            let mut analyzers = self.analyzers.write();
            if analyzers.contains_key(&name) {
                return Err(CdfaError::config_error(format!(
                    "Analyzer '{}' already registered", name
                )));
            }
            analyzers.insert(name.clone(), analyzer);
        }
        
        self.register_metadata(name, metadata)?;
        Ok(())
    }
    
    fn register_algorithm_internal(
        &self,
        name: String,
        algorithm: Box<dyn SignalAlgorithm + Send + Sync>,
        metadata: ComponentMetadata,
    ) -> Result<()> {
        {
            let mut algorithms = self.algorithms.write();
            if algorithms.contains_key(&name) {
                return Err(CdfaError::config_error(format!(
                    "Algorithm '{}' already registered", name
                )));
            }
            algorithms.insert(name.clone(), algorithm);
        }
        
        self.register_metadata(name, metadata)?;
        Ok(())
    }
    
    fn register_metadata(&self, name: String, metadata: ComponentMetadata) -> Result<()> {
        let mut meta_map = self.metadata.write();
        meta_map.insert(name.clone(), metadata);
        
        // Initialize performance tracking
        let mut perf_map = self.performance.write();
        perf_map.insert(name, ComponentPerformance::default());
        
        Ok(())
    }
    
    fn create_default_metadata(&self, name: String, component_type: ComponentType) -> ComponentMetadata {
        ComponentMetadata {
            name: name.clone(),
            component_type,
            version: "1.0.0".to_string(),
            author: "User".to_string(),
            description: format!("Custom {} component", match component_type {
                ComponentType::DiversityMethod => "diversity method",
                ComponentType::FusionMethod => "fusion method",
                ComponentType::PatternDetector => "pattern detector",
                ComponentType::SystemAnalyzer => "system analyzer",
                ComponentType::SignalAlgorithm => "signal algorithm",
                ComponentType::HardwareBackend => "hardware backend",
                ComponentType::CacheBackend => "cache backend",
            }),
            features: vec![],
            performance_profile: PerformanceProfile {
                typical_time_us: 1000,
                memory_usage: MemoryUsage::Linear,
                scalability: Scalability::Moderate,
                parallel_benefit: false,
                simd_benefit: false,
            },
            min_data_length: 10,
            complexity: 3,
            realtime_capable: false,
            required_features: vec![],
        }
    }
    
    fn update_component_usage(&self, name: &str) {
        let mut perf_map = self.performance.write();
        if let Some(perf) = perf_map.get_mut(name) {
            perf.usage_count += 1;
            perf.last_used = Some(chrono::Utc::now().timestamp());
        }
    }
    
    fn get_dependents(&self, name: &str) -> Vec<String> {
        let deps = self.dependencies.read();
        deps.iter()
            .filter_map(|(component, dependencies)| {
                if dependencies.contains(&name.to_string()) {
                    Some(component.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock implementations for testing
    struct MockDiversityMethod;
    impl DiversityMethod for MockDiversityMethod {
        fn calculate(&self, data: &FloatArrayView2) -> Result<FloatArray1> {
            Ok(FloatArray1::ones(data.ncols()))
        }
    }
    
    #[test]
    fn test_registry_creation() {
        let registry = ComponentRegistry::new();
        
        // Should have some default components
        #[cfg(feature = "core")]
        {
            let diversity_methods = registry.list_diversity_methods();
            assert!(!diversity_methods.is_empty());
            assert!(diversity_methods.contains(&"kendall".to_string()));
            assert!(diversity_methods.contains(&"pearson".to_string()));
        }
    }
    
    #[test]
    fn test_component_registration() {
        let registry = ComponentRegistry::new();
        
        // Register a custom component
        let result = registry.register_diversity_method(
            "mock".to_string(),
            MockDiversityMethod,
        );
        assert!(result.is_ok());
        
        // Check it appears in the list
        let methods = registry.list_diversity_methods();
        assert!(methods.contains(&"mock".to_string()));
        
        // Check metadata was created
        let metadata = registry.get_metadata("mock");
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().component_type, ComponentType::DiversityMethod);
    }
    
    #[test]
    fn test_duplicate_registration() {
        let registry = ComponentRegistry::new();
        
        // Register a component
        let result1 = registry.register_diversity_method(
            "mock".to_string(),
            MockDiversityMethod,
        );
        assert!(result1.is_ok());
        
        // Try to register again with same name
        let result2 = registry.register_diversity_method(
            "mock".to_string(),
            MockDiversityMethod,
        );
        assert!(result2.is_err());
        assert!(result2.unwrap_err().to_string().contains("already registered"));
    }
    
    #[test]
    fn test_component_filtering() {
        let registry = ComponentRegistry::new();
        
        // Get components by type
        let diversity_components = registry.get_components_by_type(ComponentType::DiversityMethod);
        
        #[cfg(feature = "core")]
        {
            assert!(!diversity_components.is_empty());
            assert!(diversity_components.iter().any(|c| c.name == "kendall"));
        }
        
        // Get components with features
        let fast_components = registry.get_components_with_features(&["fast".to_string()]);
        
        #[cfg(feature = "core")]
        {
            assert!(fast_components.iter().any(|c| c.name == "pearson"));
        }
    }
    
    #[test]
    fn test_component_removal() {
        let registry = ComponentRegistry::new();
        
        // Register a component
        registry.register_diversity_method(
            "test_removal".to_string(),
            MockDiversityMethod,
        ).unwrap();
        
        // Verify it exists
        assert!(registry.contains("test_removal"));
        
        // Remove it
        let result = registry.remove_component("test_removal");
        assert!(result.is_ok());
        
        // Verify it's gone
        assert!(!registry.contains("test_removal"));
        assert!(!registry.list_diversity_methods().contains(&"test_removal".to_string()));
    }
}
