//! Resource Manager Module
//!
//! Advanced resource management for quantum trading operations with dynamic allocation and monitoring.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Resource types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Cpu,
    Memory,
    GpuMemory,
    Bandwidth,
    Storage,
    QuantumProcessors,
    WasmRuntime,
}

/// Resource status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceStatus {
    Available,
    Allocated,
    Reserved,
    Maintenance,
    Offline,
}

/// Resource allocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub id: String,
    pub requester: String,
    pub component: String,
    pub resources: HashMap<ResourceType, u64>,
    pub priority: AllocationPriority,
    pub duration_seconds: Option<u64>,
    pub deadline: Option<DateTime<Utc>>,
    pub requirements: Vec<ResourceRequirement>,
    pub preferences: Vec<ResourcePreference>,
    pub metadata: HashMap<String, String>,
}

/// Allocation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    Low,
    Medium,
    High,
    Critical,
    System,
}

/// Resource requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub requirement_type: String,
    pub value: String,
    pub is_mandatory: bool,
}

/// Resource preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePreference {
    pub preference_type: String,
    pub value: String,
    pub weight: f64,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub id: String,
    pub request_id: String,
    pub allocated_resources: HashMap<ResourceType, u64>,
    pub resource_nodes: Vec<String>,
    pub allocation_time: DateTime<Utc>,
    pub expiry_time: Option<DateTime<Utc>>,
    pub status: AllocationStatus,
    pub performance_metrics: HashMap<String, f64>,
    pub cost: f64,
}

/// Allocation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AllocationStatus {
    Active,
    Expired,
    Released,
    Failed,
}

/// Resource node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: String,
    pub name: String,
    pub node_type: String,
    pub location: String,
    pub status: ResourceStatus,
    pub total_resources: HashMap<ResourceType, u64>,
    pub available_resources: HashMap<ResourceType, u64>,
    pub allocated_resources: HashMap<ResourceType, u64>,
    pub reservations: Vec<ResourceReservation>,
    pub performance_score: f64,
    pub cost_per_unit: HashMap<ResourceType, f64>,
    pub last_updated: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Resource reservation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReservation {
    pub id: String,
    pub allocation_id: String,
    pub resources: HashMap<ResourceType, u64>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub node_id: String,
    pub resource_type: ResourceType,
    pub total_capacity: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub average_usage: f64,
    pub utilization_percentage: f64,
    pub efficiency_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Resource manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagerConfig {
    pub allocation_strategy: AllocationStrategy,
    pub load_balancing_enabled: bool,
    pub auto_scaling_enabled: bool,
    pub resource_pooling_enabled: bool,
    pub cost_optimization_enabled: bool,
    pub performance_monitoring_interval_seconds: u64,
    pub allocation_timeout_seconds: u64,
    pub cleanup_interval_seconds: u64,
    pub over_allocation_threshold: f64,
    pub reservation_buffer_percentage: f64,
}

/// Allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    RoundRobin,
    LeastLoaded,
    HighestPerformance,
    LowestCost,
    Quantum,
}

/// Resource manager implementation
#[derive(Debug)]
pub struct ResourceManager {
    config: ResourceManagerConfig,
    resource_nodes: Arc<RwLock<HashMap<String, ResourceNode>>>,
    active_allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    allocation_requests: Arc<RwLock<Vec<ResourceRequest>>>,
    utilization_metrics: Arc<RwLock<Vec<ResourceUtilization>>>,
    performance_monitor: Arc<Mutex<ResourcePerformanceMonitor>>,
    cost_tracker: Arc<Mutex<ResourceCostTracker>>,
}

/// Performance monitoring
#[derive(Debug)]
pub struct ResourcePerformanceMonitor {
    pub total_allocations: u64,
    pub successful_allocations: u64,
    pub failed_allocations: u64,
    pub average_allocation_time_ms: f64,
    pub total_resource_hours: f64,
    pub efficiency_score: f64,
    pub utilization_scores: HashMap<ResourceType, f64>,
}

/// Cost tracking
#[derive(Debug)]
pub struct ResourceCostTracker {
    pub total_cost: f64,
    pub cost_by_resource_type: HashMap<ResourceType, f64>,
    pub cost_by_component: HashMap<String, f64>,
    pub cost_per_hour: f64,
    pub budget_remaining: f64,
    pub cost_optimization_savings: f64,
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new(config: ResourceManagerConfig) -> Self {
        Self {
            config,
            resource_nodes: Arc::new(RwLock::new(HashMap::new())),
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            allocation_requests: Arc::new(RwLock::new(Vec::new())),
            utilization_metrics: Arc::new(RwLock::new(Vec::new())),
            performance_monitor: Arc::new(Mutex::new(ResourcePerformanceMonitor {
                total_allocations: 0,
                successful_allocations: 0,
                failed_allocations: 0,
                average_allocation_time_ms: 0.0,
                total_resource_hours: 0.0,
                efficiency_score: 0.0,
                utilization_scores: HashMap::new(),
            })),
            cost_tracker: Arc::new(Mutex::new(ResourceCostTracker {
                total_cost: 0.0,
                cost_by_resource_type: HashMap::new(),
                cost_by_component: HashMap::new(),
                cost_per_hour: 0.0,
                budget_remaining: 100000.0, // Default budget
                cost_optimization_savings: 0.0,
            })),
        }
    }

    /// Register resource node
    pub async fn register_node(&self, node: ResourceNode) -> QarResult<()> {
        let mut nodes = self.resource_nodes.write().await;
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Allocate resources
    pub async fn allocate_resources(&self, request: ResourceRequest) -> QarResult<ResourceAllocation> {
        let start_time = std::time::Instant::now();
        
        // Validate request
        self.validate_request(&request).await?;
        
        // Find suitable nodes
        let suitable_nodes = self.find_suitable_nodes(&request).await?;
        
        if suitable_nodes.is_empty() {
            return Err(QarError::ResourceError("No suitable nodes available".to_string()));
        }
        
        // Select optimal allocation
        let selected_allocation = self.select_optimal_allocation(&request, suitable_nodes).await?;
        
        // Reserve resources
        self.reserve_resources(&selected_allocation).await?;
        
        // Create allocation record
        let allocation = ResourceAllocation {
            id: Uuid::new_v4().to_string(),
            request_id: request.id.clone(),
            allocated_resources: selected_allocation.resources.clone(),
            resource_nodes: selected_allocation.node_ids,
            allocation_time: Utc::now(),
            expiry_time: request.duration_seconds.map(|d| Utc::now() + chrono::Duration::seconds(d as i64)),
            status: AllocationStatus::Active,
            performance_metrics: HashMap::new(),
            cost: selected_allocation.cost,
        };
        
        // Store allocation
        {
            let mut allocations = self.active_allocations.write().await;
            allocations.insert(allocation.id.clone(), allocation.clone());
        }
        
        // Update performance metrics
        let allocation_time = start_time.elapsed().as_millis() as f64;
        self.update_performance_metrics(true, allocation_time).await?;
        self.update_cost_tracking(&allocation).await?;
        
        Ok(allocation)
    }

    /// Release resources
    pub async fn release_resources(&self, allocation_id: &str) -> QarResult<()> {
        let allocation = {
            let mut allocations = self.active_allocations.write().await;
            allocations.remove(allocation_id)
                .ok_or_else(|| QarError::ResourceError(format!("Allocation not found: {}", allocation_id)))?
        };
        
        // Release from nodes
        self.release_from_nodes(&allocation).await?;
        
        Ok(())
    }

    /// Find suitable nodes for allocation
    async fn find_suitable_nodes(&self, request: &ResourceRequest) -> QarResult<Vec<String>> {
        let nodes = self.resource_nodes.read().await;
        let mut suitable_nodes = Vec::new();
        
        for (node_id, node) in nodes.iter() {
            if node.status != ResourceStatus::Available {
                continue;
            }
            
            // Check if node can satisfy resource requirements
            let mut can_satisfy = true;
            for (resource_type, required_amount) in &request.resources {
                let available = node.available_resources.get(resource_type).unwrap_or(&0);
                if *available < *required_amount {
                    can_satisfy = false;
                    break;
                }
            }
            
            if can_satisfy && self.check_requirements(node, &request.requirements).await? {
                suitable_nodes.push(node_id.clone());
            }
        }
        
        Ok(suitable_nodes)
    }

    /// Select optimal allocation
    async fn select_optimal_allocation(
        &self,
        request: &ResourceRequest,
        suitable_nodes: Vec<String>,
    ) -> QarResult<AllocationPlan> {
        let nodes = self.resource_nodes.read().await;
        let mut best_plan: Option<AllocationPlan> = None;
        let mut best_score = f64::NEGATIVE_INFINITY;
        
        // For simplicity, we'll use single-node allocation
        // In practice, this could distribute across multiple nodes
        for node_id in suitable_nodes {
            if let Some(node) = nodes.get(&node_id) {
                let cost = self.calculate_allocation_cost(request, node).await?;
                let performance_score = node.performance_score;
                let utilization_score = self.calculate_utilization_score(node, request).await?;
                
                // Calculate overall score based on strategy
                let score = match self.config.allocation_strategy {
                    AllocationStrategy::HighestPerformance => performance_score,
                    AllocationStrategy::LowestCost => 1.0 / (cost + 1.0),
                    AllocationStrategy::LeastLoaded => utilization_score,
                    _ => performance_score * 0.4 + utilization_score * 0.4 + (1.0 / (cost + 1.0)) * 0.2,
                };
                
                if score > best_score {
                    best_score = score;
                    best_plan = Some(AllocationPlan {
                        node_ids: vec![node_id],
                        resources: request.resources.clone(),
                        cost,
                    });
                }
            }
        }
        
        best_plan.ok_or_else(|| QarError::ResourceError("No optimal allocation found".to_string()))
    }

    /// Reserve resources on nodes
    async fn reserve_resources(&self, plan: &AllocationPlan) -> QarResult<()> {
        let mut nodes = self.resource_nodes.write().await;
        
        for node_id in &plan.node_ids {
            if let Some(node) = nodes.get_mut(node_id) {
                for (resource_type, amount) in &plan.resources {
                    let available = node.available_resources.get_mut(resource_type)
                        .ok_or_else(|| QarError::ResourceError("Resource type not available".to_string()))?;
                    
                    if *available < *amount {
                        return Err(QarError::ResourceError("Insufficient resources".to_string()));
                    }
                    
                    *available -= amount;
                    
                    let allocated = node.allocated_resources.entry(resource_type.clone()).or_insert(0);
                    *allocated += amount;
                }
            }
        }
        
        Ok(())
    }

    /// Release resources from nodes
    async fn release_from_nodes(&self, allocation: &ResourceAllocation) -> QarResult<()> {
        let mut nodes = self.resource_nodes.write().await;
        
        for node_id in &allocation.resource_nodes {
            if let Some(node) = nodes.get_mut(node_id) {
                for (resource_type, amount) in &allocation.allocated_resources {
                    let available = node.available_resources.entry(resource_type.clone()).or_insert(0);
                    *available += amount;
                    
                    let allocated = node.allocated_resources.get_mut(resource_type)
                        .ok_or_else(|| QarError::ResourceError("Resource not allocated".to_string()))?;
                    
                    if *allocated >= *amount {
                        *allocated -= amount;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Validate resource request
    async fn validate_request(&self, request: &ResourceRequest) -> QarResult<()> {
        if request.resources.is_empty() {
            return Err(QarError::ResourceError("No resources requested".to_string()));
        }
        
        for (_, amount) in &request.resources {
            if *amount == 0 {
                return Err(QarError::ResourceError("Resource amount cannot be zero".to_string()));
            }
        }
        
        Ok(())
    }

    /// Check if node meets requirements
    async fn check_requirements(&self, _node: &ResourceNode, _requirements: &[ResourceRequirement]) -> QarResult<bool> {
        // Implement requirement checking logic
        Ok(true)
    }

    /// Calculate allocation cost
    async fn calculate_allocation_cost(&self, request: &ResourceRequest, node: &ResourceNode) -> QarResult<f64> {
        let mut total_cost = 0.0;
        
        for (resource_type, amount) in &request.resources {
            let cost_per_unit = node.cost_per_unit.get(resource_type).unwrap_or(&1.0);
            total_cost += (*amount as f64) * cost_per_unit;
        }
        
        // Apply duration multiplier
        if let Some(duration) = request.duration_seconds {
            total_cost *= (duration as f64) / 3600.0; // Convert to hours
        }
        
        Ok(total_cost)
    }

    /// Calculate utilization score
    async fn calculate_utilization_score(&self, node: &ResourceNode, request: &ResourceRequest) -> QarResult<f64> {
        let mut utilization_sum = 0.0;
        let mut resource_count = 0;
        
        for (resource_type, total) in &node.total_resources {
            let allocated = node.allocated_resources.get(resource_type).unwrap_or(&0);
            let utilization = (*allocated as f64) / (*total as f64);
            utilization_sum += utilization;
            resource_count += 1;
        }
        
        let current_utilization = if resource_count > 0 {
            utilization_sum / (resource_count as f64)
        } else {
            0.0
        };
        
        // Return inverse utilization (prefer less loaded nodes)
        Ok(1.0 - current_utilization)
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, success: bool, allocation_time_ms: f64) -> QarResult<()> {
        let mut monitor = self.performance_monitor.lock().await;
        
        monitor.total_allocations += 1;
        
        if success {
            monitor.successful_allocations += 1;
            monitor.average_allocation_time_ms = 
                (monitor.average_allocation_time_ms * (monitor.successful_allocations - 1) as f64 + allocation_time_ms) 
                / monitor.successful_allocations as f64;
        } else {
            monitor.failed_allocations += 1;
        }
        
        Ok(())
    }

    /// Update cost tracking
    async fn update_cost_tracking(&self, allocation: &ResourceAllocation) -> QarResult<()> {
        let mut tracker = self.cost_tracker.lock().await;
        
        tracker.total_cost += allocation.cost;
        
        for (resource_type, _) in &allocation.allocated_resources {
            let cost = tracker.cost_by_resource_type.entry(resource_type.clone()).or_insert(0.0);
            *cost += allocation.cost / allocation.allocated_resources.len() as f64;
        }
        
        Ok(())
    }

    /// Get resource utilization
    pub async fn get_resource_utilization(&self) -> QarResult<Vec<ResourceUtilization>> {
        let nodes = self.resource_nodes.read().await;
        let mut utilizations = Vec::new();
        
        for (node_id, node) in nodes.iter() {
            for (resource_type, total_capacity) in &node.total_resources {
                let current_usage = node.allocated_resources.get(resource_type).unwrap_or(&0);
                let utilization_percentage = (*current_usage as f64 / *total_capacity as f64) * 100.0;
                
                utilizations.push(ResourceUtilization {
                    node_id: node_id.clone(),
                    resource_type: resource_type.clone(),
                    total_capacity: *total_capacity,
                    current_usage: *current_usage,
                    peak_usage: *current_usage, // Simplified
                    average_usage: *current_usage as f64, // Simplified
                    utilization_percentage,
                    efficiency_score: node.performance_score,
                    timestamp: Utc::now(),
                });
            }
        }
        
        Ok(utilizations)
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> QarResult<ResourcePerformanceMonitor> {
        let monitor = self.performance_monitor.lock().await;
        Ok(ResourcePerformanceMonitor {
            total_allocations: monitor.total_allocations,
            successful_allocations: monitor.successful_allocations,
            failed_allocations: monitor.failed_allocations,
            average_allocation_time_ms: monitor.average_allocation_time_ms,
            total_resource_hours: monitor.total_resource_hours,
            efficiency_score: monitor.efficiency_score,
            utilization_scores: monitor.utilization_scores.clone(),
        })
    }

    /// List active allocations
    pub async fn list_active_allocations(&self) -> QarResult<Vec<ResourceAllocation>> {
        let allocations = self.active_allocations.read().await;
        Ok(allocations.values().cloned().collect())
    }

    /// Cleanup expired allocations
    pub async fn cleanup_expired_allocations(&self) -> QarResult<usize> {
        let now = Utc::now();
        let mut expired_ids = Vec::new();
        
        {
            let allocations = self.active_allocations.read().await;
            for (id, allocation) in allocations.iter() {
                if let Some(expiry) = allocation.expiry_time {
                    if now > expiry {
                        expired_ids.push(id.clone());
                    }
                }
            }
        }
        
        for id in &expired_ids {
            self.release_resources(id).await?;
        }
        
        Ok(expired_ids.len())
    }
}

/// Allocation plan helper
#[derive(Debug, Clone)]
struct AllocationPlan {
    node_ids: Vec<String>,
    resources: HashMap<ResourceType, u64>,
    cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_resource_manager() -> ResourceManager {
        let config = ResourceManagerConfig {
            allocation_strategy: AllocationStrategy::BestFit,
            load_balancing_enabled: true,
            auto_scaling_enabled: false,
            resource_pooling_enabled: true,
            cost_optimization_enabled: true,
            performance_monitoring_interval_seconds: 60,
            allocation_timeout_seconds: 30,
            cleanup_interval_seconds: 300,
            over_allocation_threshold: 0.8,
            reservation_buffer_percentage: 10.0,
        };

        ResourceManager::new(config)
    }

    fn create_test_node() -> ResourceNode {
        let mut total_resources = HashMap::new();
        total_resources.insert(ResourceType::Cpu, 16);
        total_resources.insert(ResourceType::Memory, 32768);
        total_resources.insert(ResourceType::GpuMemory, 8192);

        let mut available_resources = total_resources.clone();
        let allocated_resources = HashMap::new();

        let mut cost_per_unit = HashMap::new();
        cost_per_unit.insert(ResourceType::Cpu, 0.1);
        cost_per_unit.insert(ResourceType::Memory, 0.001);
        cost_per_unit.insert(ResourceType::GpuMemory, 0.01);

        ResourceNode {
            id: "node-1".to_string(),
            name: "Test Node 1".to_string(),
            node_type: "compute".to_string(),
            location: "datacenter-1".to_string(),
            status: ResourceStatus::Available,
            total_resources,
            available_resources,
            allocated_resources,
            reservations: Vec::new(),
            performance_score: 0.95,
            cost_per_unit,
            last_updated: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn create_test_request() -> ResourceRequest {
        let mut resources = HashMap::new();
        resources.insert(ResourceType::Cpu, 4);
        resources.insert(ResourceType::Memory, 8192);

        ResourceRequest {
            id: "req-1".to_string(),
            requester: "test_component".to_string(),
            component: "portfolio_optimizer".to_string(),
            resources,
            priority: AllocationPriority::Medium,
            duration_seconds: Some(3600),
            deadline: None,
            requirements: Vec::new(),
            preferences: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_register_node() {
        let manager = create_test_resource_manager();
        let node = create_test_node();

        manager.register_node(node).await.unwrap();

        let nodes = manager.resource_nodes.read().await;
        assert!(nodes.contains_key("node-1"));
    }

    #[tokio::test]
    async fn test_allocate_resources() {
        let manager = create_test_resource_manager();
        let node = create_test_node();
        let request = create_test_request();

        manager.register_node(node).await.unwrap();

        let allocation = manager.allocate_resources(request).await.unwrap();
        assert_eq!(allocation.status, AllocationStatus::Active);
        assert!(!allocation.id.is_empty());
    }

    #[tokio::test]
    async fn test_release_resources() {
        let manager = create_test_resource_manager();
        let node = create_test_node();
        let request = create_test_request();

        manager.register_node(node).await.unwrap();

        let allocation = manager.allocate_resources(request).await.unwrap();
        manager.release_resources(&allocation.id).await.unwrap();

        let allocations = manager.active_allocations.read().await;
        assert!(!allocations.contains_key(&allocation.id));
    }

    #[tokio::test]
    async fn test_resource_utilization() {
        let manager = create_test_resource_manager();
        let node = create_test_node();

        manager.register_node(node).await.unwrap();

        let utilization = manager.get_resource_utilization().await.unwrap();
        assert!(!utilization.is_empty());
    }

    #[tokio::test]
    async fn test_insufficient_resources() {
        let manager = create_test_resource_manager();
        let node = create_test_node();
        let mut request = create_test_request();

        // Request more CPU than available
        request.resources.insert(ResourceType::Cpu, 32);

        manager.register_node(node).await.unwrap();

        let result = manager.allocate_resources(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let manager = create_test_resource_manager();
        let node = create_test_node();
        let request = create_test_request();

        manager.register_node(node).await.unwrap();
        let _allocation = manager.allocate_resources(request).await.unwrap();

        let metrics = manager.get_performance_metrics().await.unwrap();
        assert!(metrics.total_allocations > 0);
        assert!(metrics.successful_allocations > 0);
    }
}