//! # Resource Handlers
//! 
//! Comprehensive resource management for parasitic operations

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};

/// Resource manager for parasitic trading operations
pub struct ParasiticResourceManager {
    /// Available resources by type
    pub resources: Arc<RwLock<HashMap<ResourceType, ResourcePool>>>,
    /// Resource allocation history
    pub allocation_history: Arc<RwLock<Vec<ResourceAllocation>>>,
    /// Resource constraints
    pub constraints: ResourceConstraints,
    /// Performance metrics
    pub metrics: Arc<RwLock<ResourceMetrics>>,
}

/// Types of resources managed by the system
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// Computing power allocation
    ComputingPower,
    /// Memory allocation
    Memory,
    /// Network bandwidth
    NetworkBandwidth,
    /// Trading capital
    TradingCapital,
    /// API rate limits
    ApiQuota,
    /// Data feeds
    DataFeeds,
    /// Analysis threads
    AnalysisThreads,
    /// Storage capacity
    Storage,
}

/// Resource pool for a specific resource type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    pub resource_type: ResourceType,
    pub total_capacity: f64,
    pub available_capacity: f64,
    pub allocated_capacity: f64,
    pub reserved_capacity: f64,
    pub utilization_rate: f64,
    pub last_update: DateTime<Utc>,
    pub allocations: Vec<ActiveAllocation>,
}

/// Active resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAllocation {
    pub allocation_id: String,
    pub organism_id: String,
    pub amount: f64,
    pub priority: AllocationPriority,
    pub start_time: DateTime<Utc>,
    pub expected_duration: chrono::Duration,
    pub actual_usage: f64,
    pub efficiency_score: f64,
}

/// Priority levels for resource allocation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Hash)]
pub enum AllocationPriority {
    Critical = 4,
    High = 3,
    Medium = 2,
    Low = 1,
    Background = 0,
}

/// Historical resource allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub allocation_id: String,
    pub resource_type: ResourceType,
    pub organism_id: String,
    pub requested_amount: f64,
    pub allocated_amount: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub efficiency: f64,
    pub cost: f64,
    pub outcome: AllocationOutcome,
}

/// Outcome of resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationOutcome {
    Successful,
    PartialSuccess,
    Failed,
    Cancelled,
    Expired,
}

/// Resource constraints and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum allocations per organism
    pub max_allocations_per_organism: HashMap<String, u32>,
    /// Resource limits by type
    pub resource_limits: HashMap<ResourceType, ResourceLimit>,
    /// Time-based constraints
    pub time_constraints: Vec<TimeConstraint>,
    /// Priority-based constraints
    pub priority_constraints: PriorityConstraints,
}

/// Resource limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    pub max_single_allocation: f64,
    pub max_total_allocation: f64,
    pub max_concurrent_allocations: u32,
    pub cooldown_period: chrono::Duration,
    pub burst_capacity: f64,
}

/// Time-based resource constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraint {
    pub constraint_id: String,
    pub resource_type: ResourceType,
    pub time_window: chrono::Duration,
    pub max_allocations: u32,
    pub max_total_amount: f64,
    pub active_period: Option<TimeRange>,
}

/// Time range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_hour: u8,
    pub end_hour: u8,
    pub days_of_week: Vec<chrono::Weekday>,
}

/// Priority-based constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityConstraints {
    /// Reserved capacity by priority
    pub priority_reserves: HashMap<AllocationPriority, f64>,
    /// Preemption rules
    pub preemption_rules: Vec<PreemptionRule>,
    /// Priority escalation rules
    pub escalation_rules: Vec<EscalationRule>,
}

/// Rule for resource preemption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionRule {
    pub rule_id: String,
    pub preempting_priority: AllocationPriority,
    pub preempted_priority: AllocationPriority,
    pub resource_type: Option<ResourceType>,
    pub conditions: Vec<PreemptionCondition>,
}

/// Condition for resource preemption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionCondition {
    pub condition_type: ConditionType,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
}

/// Types of preemption conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    ResourceUtilization,
    WaitTime,
    EfficiencyScore,
    SystemLoad,
    Priority,
}

/// Comparison operators for conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterOrEqual,
    LessOrEqual,
}

/// Priority escalation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRule {
    pub rule_id: String,
    pub trigger_condition: EscalationTrigger,
    pub escalation_amount: u8,
    pub max_escalation_level: AllocationPriority,
    pub cooldown: chrono::Duration,
}

/// Trigger for priority escalation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationTrigger {
    pub wait_time_threshold: chrono::Duration,
    pub resource_scarcity_threshold: f64,
    pub system_conditions: Vec<String>,
}

/// Resource performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub allocation_success_rate: f64,
    pub average_allocation_time: chrono::Duration,
    pub resource_utilization: HashMap<ResourceType, f64>,
    pub efficiency_scores: HashMap<String, f64>, // organism_id -> efficiency
    pub cost_effectiveness: f64,
    pub waste_percentage: f64,
    pub contention_events: u64,
    pub preemption_events: u64,
}

impl ParasiticResourceManager {
    pub fn new(constraints: ResourceConstraints) -> Self {
        Self {
            resources: Arc::new(RwLock::new(Self::initialize_resource_pools())),
            allocation_history: Arc::new(RwLock::new(Vec::new())),
            constraints,
            metrics: Arc::new(RwLock::new(ResourceMetrics::default())),
        }
    }
    
    /// Initialize resource pools with default capacities
    fn initialize_resource_pools() -> HashMap<ResourceType, ResourcePool> {
        let mut pools = HashMap::new();
        
        let resource_configs = vec![
            (ResourceType::ComputingPower, 100.0),
            (ResourceType::Memory, 16384.0), // MB
            (ResourceType::NetworkBandwidth, 1000.0), // Mbps
            (ResourceType::TradingCapital, 1000000.0), // USD
            (ResourceType::ApiQuota, 10000.0), // requests/hour
            (ResourceType::DataFeeds, 50.0), // concurrent feeds
            (ResourceType::AnalysisThreads, 16.0), // thread count
            (ResourceType::Storage, 1024.0), // GB
        ];
        
        for (resource_type, capacity) in resource_configs {
            pools.insert(resource_type.clone(), ResourcePool {
                resource_type: resource_type.clone(),
                total_capacity: capacity,
                available_capacity: capacity,
                allocated_capacity: 0.0,
                reserved_capacity: capacity * 0.1, // 10% reserved
                utilization_rate: 0.0,
                last_update: Utc::now(),
                allocations: Vec::new(),
            });
        }
        
        pools
    }
    
    /// Allocate resources for an organism
    pub async fn allocate_resources(
        &self,
        organism_id: &str,
        resource_type: ResourceType,
        amount: f64,
        priority: AllocationPriority,
        expected_duration: chrono::Duration,
    ) -> Result<String, ResourceError> {
        // Validate allocation request
        self.validate_allocation_request(&resource_type, amount, priority)?;
        
        // Check availability
        let allocation_id = {
            let mut resources = self.resources.write();
            let pool = resources.get_mut(&resource_type)
                .ok_or(ResourceError::ResourceNotFound)?;
            
            if pool.available_capacity < amount {
                return Err(ResourceError::InsufficientResources);
            }
            
            // Create allocation
            let allocation_id = uuid::Uuid::new_v4().to_string();
            let allocation = ActiveAllocation {
                allocation_id: allocation_id.clone(),
                organism_id: organism_id.to_string(),
                amount,
                priority,
                start_time: Utc::now(),
                expected_duration,
                actual_usage: 0.0,
                efficiency_score: 0.0,
            };
            
            // Update pool
            pool.available_capacity -= amount;
            pool.allocated_capacity += amount;
            pool.utilization_rate = pool.allocated_capacity / pool.total_capacity;
            pool.last_update = Utc::now();
            pool.allocations.push(allocation);
            
            allocation_id
        };
        
        // Record allocation in history
        let allocation_record = ResourceAllocation {
            allocation_id: allocation_id.clone(),
            resource_type,
            organism_id: organism_id.to_string(),
            requested_amount: amount,
            allocated_amount: amount,
            start_time: Utc::now(),
            end_time: None,
            efficiency: 0.0,
            cost: self.calculate_resource_cost(&resource_type, amount),
            outcome: AllocationOutcome::Successful,
        };
        
        self.allocation_history.write().push(allocation_record);
        
        Ok(allocation_id)
    }
    
    /// Release allocated resources
    pub async fn release_resources(&self, allocation_id: &str) -> Result<(), ResourceError> {
        let mut resources = self.resources.write();
        
        // Find and remove the allocation
        for pool in resources.values_mut() {
            if let Some(pos) = pool.allocations.iter().position(|a| a.allocation_id == allocation_id) {
                let allocation = pool.allocations.remove(pos);
                
                // Update pool
                pool.available_capacity += allocation.amount;
                pool.allocated_capacity -= allocation.amount;
                pool.utilization_rate = pool.allocated_capacity / pool.total_capacity;
                pool.last_update = Utc::now();
                
                // Update allocation history
                let mut history = self.allocation_history.write();
                if let Some(record) = history.iter_mut().find(|r| r.allocation_id == allocation_id) {
                    record.end_time = Some(Utc::now());
                    record.efficiency = allocation.efficiency_score;
                }
                
                return Ok(());
            }
        }
        
        Err(ResourceError::AllocationNotFound)
    }
    
    /// Validate allocation request against constraints
    fn validate_allocation_request(
        &self,
        resource_type: &ResourceType,
        amount: f64,
        _priority: AllocationPriority,
    ) -> Result<(), ResourceError> {
        if let Some(limit) = self.constraints.resource_limits.get(resource_type) {
            if amount > limit.max_single_allocation {
                return Err(ResourceError::AllocationTooLarge);
            }
        }
        
        Ok(())
    }
    
    /// Calculate cost of resource allocation
    fn calculate_resource_cost(&self, resource_type: &ResourceType, amount: f64) -> f64 {
        let base_cost = match resource_type {
            ResourceType::ComputingPower => 0.01 * amount,
            ResourceType::Memory => 0.001 * amount,
            ResourceType::NetworkBandwidth => 0.1 * amount,
            ResourceType::TradingCapital => 0.0001 * amount,
            ResourceType::ApiQuota => 0.001 * amount,
            ResourceType::DataFeeds => 0.1 * amount,
            ResourceType::AnalysisThreads => 0.01 * amount,
            ResourceType::Storage => 0.001 * amount,
        };
        
        // Apply dynamic pricing based on scarcity
        let resources = self.resources.read();
        if let Some(pool) = resources.get(resource_type) {
            let scarcity_multiplier = 1.0 + (1.0 - pool.utilization_rate);
            base_cost * scarcity_multiplier
        } else {
            base_cost
        }
    }
    
    /// Get resource utilization statistics
    pub fn get_resource_utilization(&self) -> HashMap<ResourceType, f64> {
        let resources = self.resources.read();
        resources.iter()
            .map(|(rt, pool)| (rt.clone(), pool.utilization_rate))
            .collect()
    }
    
    /// Get resource metrics
    pub fn get_metrics(&self) -> ResourceMetrics {
        self.metrics.read().clone()
    }
    
    /// Update resource metrics
    pub async fn update_metrics(&self) {
        let mut metrics = self.metrics.write();
        let resources = self.resources.read();
        let history = self.allocation_history.read();
        
        // Update utilization rates
        metrics.resource_utilization = resources.iter()
            .map(|(rt, pool)| (rt.clone(), pool.utilization_rate))
            .collect();
        
        // Calculate success rate
        let successful_allocations = history.iter()
            .filter(|a| matches!(a.outcome, AllocationOutcome::Successful))
            .count();
        metrics.allocation_success_rate = if !history.is_empty() {
            successful_allocations as f64 / history.len() as f64
        } else {
            1.0
        };
        
        // Calculate average efficiency
        let efficiency_sum: f64 = history.iter().map(|a| a.efficiency).sum();
        metrics.cost_effectiveness = if !history.is_empty() {
            efficiency_sum / history.len() as f64
        } else {
            1.0
        };
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            allocation_success_rate: 1.0,
            average_allocation_time: chrono::Duration::milliseconds(100),
            resource_utilization: HashMap::new(),
            efficiency_scores: HashMap::new(),
            cost_effectiveness: 1.0,
            waste_percentage: 0.0,
            contention_events: 0,
            preemption_events: 0,
        }
    }
}

/// Resource management errors
#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    #[error("Resource not found")]
    ResourceNotFound,
    #[error("Insufficient resources available")]
    InsufficientResources,
    #[error("Allocation not found")]
    AllocationNotFound,
    #[error("Allocation too large")]
    AllocationTooLarge,
    #[error("Resource constraints violated")]
    ConstraintViolation,
    #[error("Priority conflict")]
    PriorityConflict,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resource_manager_creation() {
        let constraints = ResourceConstraints {
            max_allocations_per_organism: HashMap::new(),
            resource_limits: HashMap::new(),
            time_constraints: Vec::new(),
            priority_constraints: PriorityConstraints {
                priority_reserves: HashMap::new(),
                preemption_rules: Vec::new(),
                escalation_rules: Vec::new(),
            },
        };
        
        let manager = ParasiticResourceManager::new(constraints);
        let utilization = manager.get_resource_utilization();
        assert!(!utilization.is_empty());
    }
    
    #[test]
    fn test_resource_pool_initialization() {
        let pools = ParasiticResourceManager::initialize_resource_pools();
        assert!(pools.contains_key(&ResourceType::ComputingPower));
        assert!(pools.contains_key(&ResourceType::Memory));
        assert!(pools.contains_key(&ResourceType::TradingCapital));
    }
    
    #[tokio::test]
    async fn test_resource_allocation() {
        let constraints = ResourceConstraints {
            max_allocations_per_organism: HashMap::new(),
            resource_limits: HashMap::new(),
            time_constraints: Vec::new(),
            priority_constraints: PriorityConstraints {
                priority_reserves: HashMap::new(),
                preemption_rules: Vec::new(),
                escalation_rules: Vec::new(),
            },
        };
        
        let manager = ParasiticResourceManager::new(constraints);
        let result = manager.allocate_resources(
            "test_organism",
            ResourceType::ComputingPower,
            10.0,
            AllocationPriority::Medium,
            chrono::Duration::minutes(5),
        ).await;
        
        assert!(result.is_ok());
        let allocation_id = result.unwrap();
        assert!(!allocation_id.is_empty());
    }
}