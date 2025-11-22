//! System environment module for autopoietic systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::Result;

/// The environment in which an autopoietic system operates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemEnvironment {
    /// Environmental conditions
    pub conditions: HashMap<String, EnvironmentalCondition>,
    
    /// External signals from the environment
    pub signals: Vec<EnvironmentalSignal>,
    
    /// Resources available in the environment
    pub resources: HashMap<String, Resource>,
    
    /// Constraints imposed by the environment
    pub constraints: Vec<Constraint>,
    
    /// Environmental stability (0.0 = chaotic, 1.0 = stable)
    pub stability: f64,
    
    /// Environmental complexity
    pub complexity: f64,
    
    /// Rate of environmental change
    pub change_rate: f64,
    
    /// Environmental pressure on the system
    pub pressure: f64,
    
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// An environmental condition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentalCondition {
    /// Condition name
    pub name: String,
    
    /// Current value
    pub value: f64,
    
    /// Optimal range for the system
    pub optimal_range: (f64, f64),
    
    /// Critical thresholds
    pub critical_low: f64,
    pub critical_high: f64,
    
    /// Rate of change
    pub rate_of_change: f64,
    
    /// Condition volatility
    pub volatility: f64,
    
    /// Impact factor on system performance
    pub impact_factor: f64,
}

/// External signal from the environment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentalSignal {
    /// Signal identifier
    pub id: String,
    
    /// Signal type
    pub signal_type: SignalType,
    
    /// Signal strength (0.0 - 1.0)
    pub strength: f64,
    
    /// Signal frequency
    pub frequency: f64,
    
    /// Signal source
    pub source: String,
    
    /// Signal data
    pub data: serde_json::Value,
    
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of environmental signals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SignalType {
    /// Information signal
    Information,
    
    /// Energy signal
    Energy,
    
    /// Threat signal
    Threat,
    
    /// Opportunity signal
    Opportunity,
    
    /// Feedback signal
    Feedback,
    
    /// Noise
    Noise,
    
    /// Custom signal type
    Custom(String),
}

/// Resource in the environment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resource {
    /// Resource name
    pub name: String,
    
    /// Resource type
    pub resource_type: ResourceType,
    
    /// Available quantity
    pub quantity: f64,
    
    /// Maximum capacity
    pub capacity: f64,
    
    /// Renewal rate (units per time)
    pub renewal_rate: f64,
    
    /// Current consumption rate
    pub consumption_rate: f64,
    
    /// Resource quality (0.0 - 1.0)
    pub quality: f64,
    
    /// Accessibility (0.0 - 1.0)
    pub accessibility: f64,
}

/// Types of resources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResourceType {
    /// Energy resource
    Energy,
    
    /// Information resource
    Information,
    
    /// Material resource
    Material,
    
    /// Time resource
    Time,
    
    /// Space resource
    Space,
    
    /// Computational resource
    Computational,
    
    /// Custom resource type
    Custom(String),
}

/// Environmental constraint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint name
    pub name: String,
    
    /// Constraint type
    pub constraint_type: ConstraintType,
    
    /// Severity (0.0 = soft, 1.0 = hard)
    pub severity: f64,
    
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
    
    /// Is the constraint active?
    pub active: bool,
    
    /// Impact on system operations
    pub impact: String,
}

/// Types of constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Physical limitation
    Physical,
    
    /// Regulatory constraint
    Regulatory,
    
    /// Resource limitation
    ResourceLimit,
    
    /// Time constraint
    Temporal,
    
    /// Spatial constraint
    Spatial,
    
    /// Performance constraint
    Performance,
    
    /// Custom constraint type
    Custom(String),
}

impl Default for SystemEnvironment {
    fn default() -> Self {
        Self {
            conditions: HashMap::new(),
            signals: Vec::new(),
            resources: HashMap::new(),
            constraints: Vec::new(),
            stability: 0.5,
            complexity: 0.5,
            change_rate: 0.1,
            pressure: 0.5,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl SystemEnvironment {
    /// Create a new environment
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an environmental condition
    pub fn add_condition(&mut self, condition: EnvironmentalCondition) {
        self.conditions.insert(condition.name.clone(), condition);
        self.update_metrics();
    }
    
    /// Add a resource
    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.name.clone(), resource);
        self.update_metrics();
    }
    
    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
        self.update_metrics();
    }
    
    /// Add an environmental signal
    pub fn add_signal(&mut self, signal: EnvironmentalSignal) {
        self.signals.push(signal);
        
        // Keep signal history limited
        if self.signals.len() > 1000 {
            self.signals.remove(0);
        }
    }
    
    /// Update environmental metrics
    fn update_metrics(&mut self) {
        // Calculate stability based on condition volatility
        let total_volatility: f64 = self.conditions.values()
            .map(|c| c.volatility * c.impact_factor)
            .sum();
        self.stability = 1.0 / (1.0 + total_volatility);
        
        // Calculate complexity
        self.complexity = (self.conditions.len() as f64 * 0.1 +
                         self.resources.len() as f64 * 0.1 +
                         self.constraints.len() as f64 * 0.2)
                         .min(1.0);
        
        // Calculate change rate
        self.change_rate = self.conditions.values()
            .map(|c| c.rate_of_change.abs())
            .sum::<f64>() / self.conditions.len().max(1) as f64;
        
        // Calculate pressure
        let resource_pressure = self.resources.values()
            .map(|r| {
                if r.capacity > 0.0 {
                    (r.consumption_rate / r.renewal_rate.max(0.1)).min(2.0) / 2.0
                } else {
                    0.0
                }
            })
            .sum::<f64>() / self.resources.len().max(1) as f64;
        
        let constraint_pressure = self.constraints.iter()
            .filter(|c| c.active)
            .map(|c| c.severity)
            .sum::<f64>() / self.constraints.len().max(1) as f64;
        
        self.pressure = (resource_pressure + constraint_pressure) / 2.0;
        
        self.last_updated = chrono::Utc::now();
    }
    
    /// Check if environment is favorable
    pub fn is_favorable(&self) -> bool {
        self.stability > 0.3 && self.pressure < 0.7
    }
    
    /// Get environmental health score
    pub fn health_score(&self) -> f64 {
        let condition_score = self.conditions.values()
            .map(|c| {
                let value = c.value;
                if value >= c.optimal_range.0 && value <= c.optimal_range.1 {
                    1.0
                } else if value < c.critical_low || value > c.critical_high {
                    0.0
                } else {
                    0.5
                }
            })
            .sum::<f64>() / self.conditions.len().max(1) as f64;
        
        let resource_score = self.resources.values()
            .map(|r| (r.quantity / r.capacity.max(1.0)) * r.quality * r.accessibility)
            .sum::<f64>() / self.resources.len().max(1) as f64;
        
        (self.stability + (1.0 - self.pressure) + condition_score + resource_score) / 4.0
    }
    
    /// Get resource availability
    pub fn resource_availability(&self, resource_name: &str) -> f64 {
        self.resources.get(resource_name)
            .map(|r| (r.quantity / r.capacity.max(1.0)) * r.accessibility)
            .unwrap_or(0.0)
    }
    
    /// Check if a constraint is violated
    pub fn is_constraint_violated(&self, constraint_name: &str) -> bool {
        self.constraints.iter()
            .find(|c| c.name == constraint_name)
            .map(|c| c.active && c.severity > 0.8)
            .unwrap_or(false)
    }
}

/// Environment sensor for monitoring
#[async_trait]
pub trait EnvironmentSensor: Send + Sync {
    /// Sense current environmental conditions
    async fn sense(&self) -> Result<SystemEnvironment>;
    
    /// Monitor specific condition
    async fn monitor_condition(&self, condition_name: &str) -> Result<EnvironmentalCondition>;
    
    /// Detect environmental signals
    async fn detect_signals(&self) -> Result<Vec<EnvironmentalSignal>>;
    
    /// Predict environmental changes
    async fn predict_changes(&self, time_horizon: chrono::Duration) -> Result<SystemEnvironment>;
}

/// Environment adapter for system adjustment
#[async_trait]
pub trait EnvironmentAdapter: Send + Sync {
    /// Adapt to environmental changes
    async fn adapt(&mut self, environment: &SystemEnvironment) -> Result<()>;
    
    /// Optimize for current conditions
    async fn optimize(&mut self, target_metric: &str) -> Result<()>;
    
    /// Respond to environmental signal
    async fn respond_to_signal(&mut self, signal: &EnvironmentalSignal) -> Result<()>;
    
    /// Allocate resources optimally
    async fn allocate_resources(&mut self, resources: &HashMap<String, Resource>) -> Result<()>;
}

/// Shared environment container
pub type SharedEnvironment = Arc<RwLock<SystemEnvironment>>;

/// Create a new shared environment
pub fn shared_environment() -> SharedEnvironment {
    Arc::new(RwLock::new(SystemEnvironment::new()))
}