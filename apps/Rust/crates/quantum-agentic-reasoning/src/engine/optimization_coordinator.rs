//! Optimization Coordinator Module
//!
//! Coordinates optimization processes across quantum trading components with intelligent resource allocation.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Optimization task priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Optimization task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

/// Optimization task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    Portfolio,
    Strategy,
    Risk,
    Execution,
    Circuit,
    Parameters,
    Performance,
    Resource,
}

/// Optimization task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTask {
    pub id: String,
    pub name: String,
    pub optimization_type: OptimizationType,
    pub priority: OptimizationPriority,
    pub status: OptimizationStatus,
    pub component: String,
    pub parameters: HashMap<String, String>,
    pub constraints: Vec<OptimizationConstraint>,
    pub objectives: Vec<OptimizationObjective>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub progress: f64,
    pub result: Option<OptimizationResult>,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Optimization constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    pub name: String,
    pub constraint_type: String,
    pub value: f64,
    pub operator: String, // <=, >=, ==, !=
    pub weight: f64,
}

/// Optimization objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationObjective {
    pub name: String,
    pub objective_type: String, // minimize, maximize
    pub weight: f64,
    pub target_value: Option<f64>,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub id: String,
    pub task_id: String,
    pub optimized_parameters: HashMap<String, f64>,
    pub objective_values: HashMap<String, f64>,
    pub constraint_violations: Vec<String>,
    pub optimization_time_ms: u64,
    pub iterations: u32,
    pub convergence_achieved: bool,
    pub confidence_score: f64,
    pub metadata: HashMap<String, String>,
}

/// Resource allocation for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: usize,
    pub memory_mb: usize,
    pub gpu_memory_mb: Option<usize>,
    pub max_runtime_seconds: u64,
    pub priority_boost: f64,
}

/// Optimization coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCoordinatorConfig {
    pub max_concurrent_tasks: usize,
    pub default_timeout_seconds: u64,
    pub enable_parallel_optimization: bool,
    pub enable_quantum_acceleration: bool,
    pub resource_allocation_strategy: ResourceAllocationStrategy,
    pub optimization_algorithms: Vec<String>,
    pub convergence_threshold: f64,
    pub max_iterations: u32,
}

/// Resource allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceAllocationStrategy {
    Fair,
    Priority,
    Dynamic,
    Quantum,
}

/// Optimization coordinator implementation
#[derive(Debug)]
pub struct OptimizationCoordinator {
    config: OptimizationCoordinatorConfig,
    active_tasks: Arc<RwLock<HashMap<String, OptimizationTask>>>,
    task_queue: Arc<RwLock<Vec<OptimizationTask>>>,
    completed_tasks: Arc<RwLock<Vec<OptimizationTask>>>,
    optimization_semaphore: Arc<Semaphore>,
    resource_manager: Arc<dyn ResourceManager + Send + Sync>,
    optimization_engines: HashMap<OptimizationType, Arc<dyn OptimizationEngine + Send + Sync>>,
    performance_monitor: Arc<Mutex<OptimizationPerformanceMonitor>>,
}

/// Resource manager trait
#[async_trait::async_trait]
pub trait ResourceManager {
    async fn allocate_resources(&self, task: &OptimizationTask) -> QarResult<ResourceAllocation>;
    async fn release_resources(&self, allocation: &ResourceAllocation) -> QarResult<()>;
    async fn get_available_resources(&self) -> QarResult<ResourceAllocation>;
    async fn estimate_resource_requirements(&self, task: &OptimizationTask) -> QarResult<ResourceAllocation>;
}

/// Optimization engine trait
#[async_trait::async_trait]
pub trait OptimizationEngine {
    async fn optimize(&self, task: &OptimizationTask) -> QarResult<OptimizationResult>;
    async fn validate_constraints(&self, task: &OptimizationTask) -> QarResult<Vec<String>>;
    async fn estimate_completion_time(&self, task: &OptimizationTask) -> QarResult<u64>;
    fn supports_parallel_execution(&self) -> bool;
}

/// Performance monitoring
#[derive(Debug)]
pub struct OptimizationPerformanceMonitor {
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_completion_time_ms: f64,
    pub resource_utilization: f64,
    pub success_rate: f64,
    pub throughput_per_hour: f64,
}

impl OptimizationCoordinator {
    /// Create new optimization coordinator
    pub fn new(
        config: OptimizationCoordinatorConfig,
        resource_manager: Arc<dyn ResourceManager + Send + Sync>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));
        
        Self {
            config,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            completed_tasks: Arc::new(RwLock::new(Vec::new())),
            optimization_semaphore: semaphore,
            resource_manager,
            optimization_engines: HashMap::new(),
            performance_monitor: Arc::new(Mutex::new(OptimizationPerformanceMonitor {
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                average_completion_time_ms: 0.0,
                resource_utilization: 0.0,
                success_rate: 0.0,
                throughput_per_hour: 0.0,
            })),
        }
    }

    /// Register optimization engine
    pub fn register_engine(
        &mut self,
        optimization_type: OptimizationType,
        engine: Arc<dyn OptimizationEngine + Send + Sync>,
    ) {
        self.optimization_engines.insert(optimization_type, engine);
    }

    /// Submit optimization task
    pub async fn submit_task(&self, mut task: OptimizationTask) -> QarResult<String> {
        task.id = Uuid::new_v4().to_string();
        task.status = OptimizationStatus::Pending;
        task.created_at = Utc::now();

        // Validate task
        self.validate_task(&task).await?;

        // Add to queue
        {
            let mut queue = self.task_queue.write().await;
            queue.push(task.clone());
            queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        // Start processing if resources available
        self.try_start_next_task().await?;

        Ok(task.id)
    }

    /// Start next task from queue
    async fn try_start_next_task(&self) -> QarResult<()> {
        if self.optimization_semaphore.try_acquire().is_ok() {
            let task = {
                let mut queue = self.task_queue.write().await;
                queue.pop()
            };

            if let Some(task) = task {
                self.start_optimization_task(task).await?;
            }
        }

        Ok(())
    }

    /// Start optimization task
    async fn start_optimization_task(&self, mut task: OptimizationTask) -> QarResult<()> {
        task.status = OptimizationStatus::Running;
        task.started_at = Some(Utc::now());

        // Add to active tasks
        {
            let mut active = self.active_tasks.write().await;
            active.insert(task.id.clone(), task.clone());
        }

        // Allocate resources
        let resource_allocation = self.resource_manager.allocate_resources(&task).await?;

        // Get optimization engine
        let engine = self.optimization_engines.get(&task.optimization_type)
            .ok_or_else(|| QarError::OptimizationError(format!("No engine for type: {:?}", task.optimization_type)))?
            .clone();

        // Run optimization in background
        let coordinator = self.clone_for_task();
        let task_id = task.id.clone();
        
        tokio::spawn(async move {
            let result = coordinator.run_optimization_task(task, engine, resource_allocation).await;
            if let Err(e) = result {
                log::error!("Optimization task {} failed: {}", task_id, e);
            }
        });

        Ok(())
    }

    /// Run optimization task
    async fn run_optimization_task(
        &self,
        mut task: OptimizationTask,
        engine: Arc<dyn OptimizationEngine + Send + Sync>,
        resource_allocation: ResourceAllocation,
    ) -> QarResult<()> {
        let start_time = std::time::Instant::now();

        let result = engine.optimize(&task).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(optimization_result) => {
                task.status = OptimizationStatus::Completed;
                task.completed_at = Some(Utc::now());
                task.progress = 1.0;
                task.result = Some(optimization_result);
            }
            Err(e) => {
                task.status = OptimizationStatus::Failed;
                task.completed_at = Some(Utc::now());
                task.error_message = Some(e.to_string());
            }
        }

        // Update task
        self.complete_optimization_task(task, resource_allocation, execution_time).await?;

        // Release semaphore and try to start next task
        self.optimization_semaphore.add_permits(1);
        self.try_start_next_task().await?;

        Ok(())
    }

    /// Complete optimization task
    async fn complete_optimization_task(
        &self,
        task: OptimizationTask,
        resource_allocation: ResourceAllocation,
        execution_time: u64,
    ) -> QarResult<()> {
        // Remove from active tasks
        {
            let mut active = self.active_tasks.write().await;
            active.remove(&task.id);
        }

        // Add to completed tasks
        {
            let mut completed = self.completed_tasks.write().await;
            completed.push(task.clone());
            
            // Limit completed tasks history
            if completed.len() > 10000 {
                completed.drain(0..1000);
            }
        }

        // Release resources
        self.resource_manager.release_resources(&resource_allocation).await?;

        // Update performance metrics
        self.update_performance_metrics(&task, execution_time).await?;

        Ok(())
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self, task: &OptimizationTask, execution_time: u64) -> QarResult<()> {
        let mut monitor = self.performance_monitor.lock().await;
        
        monitor.total_tasks += 1;
        
        if task.status == OptimizationStatus::Completed {
            monitor.completed_tasks += 1;
            
            // Update average completion time
            monitor.average_completion_time_ms = 
                (monitor.average_completion_time_ms * (monitor.completed_tasks - 1) as f64 + execution_time as f64) 
                / monitor.completed_tasks as f64;
        } else if task.status == OptimizationStatus::Failed {
            monitor.failed_tasks += 1;
        }
        
        monitor.success_rate = monitor.completed_tasks as f64 / monitor.total_tasks as f64;
        
        // Calculate throughput (tasks per hour)
        if monitor.average_completion_time_ms > 0.0 {
            monitor.throughput_per_hour = 3600000.0 / monitor.average_completion_time_ms;
        }

        Ok(())
    }

    /// Validate optimization task
    async fn validate_task(&self, task: &OptimizationTask) -> QarResult<()> {
        // Check if engine exists
        if !self.optimization_engines.contains_key(&task.optimization_type) {
            return Err(QarError::OptimizationError(
                format!("No optimization engine for type: {:?}", task.optimization_type)
            ));
        }

        // Validate objectives
        if task.objectives.is_empty() {
            return Err(QarError::OptimizationError("No optimization objectives specified".to_string()));
        }

        // Validate constraints
        for constraint in &task.constraints {
            if constraint.weight < 0.0 || constraint.weight > 1.0 {
                return Err(QarError::OptimizationError("Constraint weight must be between 0 and 1".to_string()));
            }
        }

        Ok(())
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> QarResult<Option<OptimizationTask>> {
        // Check active tasks
        {
            let active = self.active_tasks.read().await;
            if let Some(task) = active.get(task_id) {
                return Ok(Some(task.clone()));
            }
        }

        // Check completed tasks
        {
            let completed = self.completed_tasks.read().await;
            for task in completed.iter().rev() {
                if task.id == task_id {
                    return Ok(Some(task.clone()));
                }
            }
        }

        // Check queue
        {
            let queue = self.task_queue.read().await;
            for task in queue.iter() {
                if task.id == task_id {
                    return Ok(Some(task.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Cancel optimization task
    pub async fn cancel_task(&self, task_id: &str) -> QarResult<()> {
        // Remove from queue
        {
            let mut queue = self.task_queue.write().await;
            queue.retain(|task| task.id != task_id);
        }

        // Update active task status
        {
            let mut active = self.active_tasks.write().await;
            if let Some(task) = active.get_mut(task_id) {
                task.status = OptimizationStatus::Cancelled;
                task.completed_at = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Get optimization statistics
    pub async fn get_optimization_stats(&self) -> QarResult<OptimizationPerformanceMonitor> {
        let monitor = self.performance_monitor.lock().await;
        Ok(OptimizationPerformanceMonitor {
            total_tasks: monitor.total_tasks,
            completed_tasks: monitor.completed_tasks,
            failed_tasks: monitor.failed_tasks,
            average_completion_time_ms: monitor.average_completion_time_ms,
            resource_utilization: monitor.resource_utilization,
            success_rate: monitor.success_rate,
            throughput_per_hour: monitor.throughput_per_hour,
        })
    }

    /// List active tasks
    pub async fn list_active_tasks(&self) -> QarResult<Vec<OptimizationTask>> {
        let active = self.active_tasks.read().await;
        Ok(active.values().cloned().collect())
    }

    /// Clone for task execution
    fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_tasks: self.active_tasks.clone(),
            task_queue: self.task_queue.clone(),
            completed_tasks: self.completed_tasks.clone(),
            optimization_semaphore: self.optimization_semaphore.clone(),
            resource_manager: self.resource_manager.clone(),
            optimization_engines: self.optimization_engines.clone(),
            performance_monitor: self.performance_monitor.clone(),
        }
    }
}

/// Mock implementations for testing
pub struct MockResourceManager;

#[async_trait::async_trait]
impl ResourceManager for MockResourceManager {
    async fn allocate_resources(&self, _task: &OptimizationTask) -> QarResult<ResourceAllocation> {
        Ok(ResourceAllocation {
            cpu_cores: 4,
            memory_mb: 1024,
            gpu_memory_mb: Some(512),
            max_runtime_seconds: 3600,
            priority_boost: 1.0,
        })
    }

    async fn release_resources(&self, _allocation: &ResourceAllocation) -> QarResult<()> {
        Ok(())
    }

    async fn get_available_resources(&self) -> QarResult<ResourceAllocation> {
        Ok(ResourceAllocation {
            cpu_cores: 16,
            memory_mb: 16384,
            gpu_memory_mb: Some(8192),
            max_runtime_seconds: 86400,
            priority_boost: 1.0,
        })
    }

    async fn estimate_resource_requirements(&self, _task: &OptimizationTask) -> QarResult<ResourceAllocation> {
        Ok(ResourceAllocation {
            cpu_cores: 2,
            memory_mb: 512,
            gpu_memory_mb: Some(256),
            max_runtime_seconds: 1800,
            priority_boost: 1.0,
        })
    }
}

pub struct MockOptimizationEngine;

#[async_trait::async_trait]
impl OptimizationEngine for MockOptimizationEngine {
    async fn optimize(&self, task: &OptimizationTask) -> QarResult<OptimizationResult> {
        // Simulate optimization
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(OptimizationResult {
            id: Uuid::new_v4().to_string(),
            task_id: task.id.clone(),
            optimized_parameters: HashMap::new(),
            objective_values: HashMap::new(),
            constraint_violations: Vec::new(),
            optimization_time_ms: 100,
            iterations: 50,
            convergence_achieved: true,
            confidence_score: 0.95,
            metadata: HashMap::new(),
        })
    }

    async fn validate_constraints(&self, _task: &OptimizationTask) -> QarResult<Vec<String>> {
        Ok(Vec::new())
    }

    async fn estimate_completion_time(&self, _task: &OptimizationTask) -> QarResult<u64> {
        Ok(1000) // 1 second
    }

    fn supports_parallel_execution(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_coordinator() -> OptimizationCoordinator {
        let config = OptimizationCoordinatorConfig {
            max_concurrent_tasks: 4,
            default_timeout_seconds: 3600,
            enable_parallel_optimization: true,
            enable_quantum_acceleration: true,
            resource_allocation_strategy: ResourceAllocationStrategy::Dynamic,
            optimization_algorithms: vec!["genetic".to_string(), "gradient".to_string()],
            convergence_threshold: 1e-6,
            max_iterations: 1000,
        };

        let mut coordinator = OptimizationCoordinator::new(
            config,
            Arc::new(MockResourceManager),
        );

        coordinator.register_engine(
            OptimizationType::Portfolio,
            Arc::new(MockOptimizationEngine),
        );

        coordinator
    }

    fn create_test_task() -> OptimizationTask {
        OptimizationTask {
            id: String::new(),
            name: "Test Optimization".to_string(),
            optimization_type: OptimizationType::Portfolio,
            priority: OptimizationPriority::Medium,
            status: OptimizationStatus::Pending,
            component: "portfolio".to_string(),
            parameters: HashMap::new(),
            constraints: vec![
                OptimizationConstraint {
                    name: "max_risk".to_string(),
                    constraint_type: "risk".to_string(),
                    value: 0.1,
                    operator: "<=".to_string(),
                    weight: 1.0,
                }
            ],
            objectives: vec![
                OptimizationObjective {
                    name: "maximize_return".to_string(),
                    objective_type: "maximize".to_string(),
                    weight: 1.0,
                    target_value: None,
                }
            ],
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            result: None,
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_submit_task() {
        let coordinator = create_test_coordinator();
        let task = create_test_task();

        let task_id = coordinator.submit_task(task).await.unwrap();
        assert!(!task_id.is_empty());

        // Wait for task to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let status = coordinator.get_task_status(&task_id).await.unwrap();
        assert!(status.is_some());
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let coordinator = create_test_coordinator();
        let task = create_test_task();

        let task_id = coordinator.submit_task(task).await.unwrap();
        coordinator.cancel_task(&task_id).await.unwrap();

        let status = coordinator.get_task_status(&task_id).await.unwrap();
        if let Some(task) = status {
            assert_eq!(task.status, OptimizationStatus::Cancelled);
        }
    }

    #[tokio::test]
    async fn test_optimization_stats() {
        let coordinator = create_test_coordinator();
        let task = create_test_task();

        let _task_id = coordinator.submit_task(task).await.unwrap();
        
        // Wait for task to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let stats = coordinator.get_optimization_stats().await.unwrap();
        assert!(stats.total_tasks > 0);
    }

    #[tokio::test]
    async fn test_task_validation() {
        let coordinator = create_test_coordinator();
        let mut task = create_test_task();
        
        // Remove objectives to make invalid
        task.objectives.clear();

        let result = coordinator.submit_task(task).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_active_tasks() {
        let coordinator = create_test_coordinator();
        let task = create_test_task();

        let _task_id = coordinator.submit_task(task).await.unwrap();
        
        let active_tasks = coordinator.list_active_tasks().await.unwrap();
        assert!(!active_tasks.is_empty());
    }
}