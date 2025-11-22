//! System process module for autopoietic systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::Result;

/// A process within an autopoietic system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemProcess {
    /// Process identifier
    pub id: String,
    
    /// Process name
    pub name: String,
    
    /// Process type
    pub process_type: ProcessType,
    
    /// Current state of the process
    pub state: ProcessState,
    
    /// Process priority (0.0 = lowest, 1.0 = highest)
    pub priority: f64,
    
    /// Resource consumption
    pub resource_usage: ResourceUsage,
    
    /// Dependencies on other processes
    pub dependencies: Vec<String>,
    
    /// Processes that depend on this one
    pub dependents: Vec<String>,
    
    /// Process metrics
    pub metrics: ProcessMetrics,
    
    /// Process configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Types of processes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProcessType {
    /// Core system process
    Core,
    
    /// Auxiliary support process
    Auxiliary,
    
    /// Adaptive process
    Adaptive,
    
    /// Maintenance process
    Maintenance,
    
    /// Communication process
    Communication,
    
    /// Learning process
    Learning,
    
    /// Custom process type
    Custom(String),
}

/// Process state
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProcessState {
    /// Process is initializing
    Initializing,
    
    /// Process is running
    Running,
    
    /// Process is paused
    Paused,
    
    /// Process is waiting for dependencies
    Waiting,
    
    /// Process completed successfully
    Completed,
    
    /// Process failed
    Failed(String),
    
    /// Process is terminating
    Terminating,
}

/// Resource usage tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (0.0 - 1.0)
    pub cpu: f64,
    
    /// Memory usage in bytes
    pub memory: u64,
    
    /// Network bandwidth in bytes/sec
    pub network: u64,
    
    /// Energy consumption (abstract units)
    pub energy: f64,
    
    /// Time consumed in milliseconds
    pub time_ms: u64,
}

/// Process performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessMetrics {
    /// Number of successful executions
    pub executions: u64,
    
    /// Number of failures
    pub failures: u64,
    
    /// Average execution time in milliseconds
    pub avg_execution_time: f64,
    
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    
    /// Efficiency score (0.0 - 1.0)
    pub efficiency: f64,
    
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for SystemProcess {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: "unnamed_process".to_string(),
            process_type: ProcessType::Core,
            state: ProcessState::Initializing,
            priority: 0.5,
            resource_usage: ResourceUsage::default(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
            metrics: ProcessMetrics::default(),
            config: HashMap::new(),
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu: 0.0,
            memory: 0,
            network: 0,
            energy: 0.0,
            time_ms: 0,
        }
    }
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self {
            executions: 0,
            failures: 0,
            avg_execution_time: 0.0,
            success_rate: 1.0,
            efficiency: 1.0,
            last_execution: None,
        }
    }
}

impl SystemProcess {
    /// Create a new process
    pub fn new(name: String, process_type: ProcessType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            process_type,
            ..Default::default()
        }
    }
    
    /// Check if process can run (dependencies satisfied)
    pub fn can_run(&self) -> bool {
        self.state == ProcessState::Waiting || self.state == ProcessState::Initializing
    }
    
    /// Update process state
    pub fn set_state(&mut self, state: ProcessState) {
        self.state = state;
    }
    
    /// Add a dependency
    pub fn add_dependency(&mut self, process_id: String) {
        if !self.dependencies.contains(&process_id) {
            self.dependencies.push(process_id);
        }
    }
    
    /// Add a dependent
    pub fn add_dependent(&mut self, process_id: String) {
        if !self.dependents.contains(&process_id) {
            self.dependents.push(process_id);
        }
    }
    
    /// Update metrics after execution
    pub fn update_metrics(&mut self, success: bool, execution_time_ms: u64) {
        self.metrics.executions += 1;
        if !success {
            self.metrics.failures += 1;
        }
        
        // Update average execution time
        let total_time = self.metrics.avg_execution_time * (self.metrics.executions - 1) as f64;
        self.metrics.avg_execution_time = (total_time + execution_time_ms as f64) / self.metrics.executions as f64;
        
        // Update success rate
        self.metrics.success_rate = (self.metrics.executions - self.metrics.failures) as f64 / self.metrics.executions as f64;
        
        // Update efficiency (simplified calculation)
        let expected_time = 1000.0; // Expected execution time in ms
        self.metrics.efficiency = (expected_time / self.metrics.avg_execution_time).min(1.0);
        
        self.metrics.last_execution = Some(chrono::Utc::now());
    }
    
    /// Calculate process health score
    pub fn health_score(&self) -> f64 {
        let state_score = match &self.state {
            ProcessState::Running => 1.0,
            ProcessState::Completed => 1.0,
            ProcessState::Paused => 0.5,
            ProcessState::Waiting => 0.7,
            ProcessState::Initializing => 0.8,
            ProcessState::Failed(_) => 0.0,
            ProcessState::Terminating => 0.3,
        };
        
        (state_score + self.metrics.success_rate + self.metrics.efficiency) / 3.0
    }
}

/// Process manager for coordinating system processes
#[async_trait]
pub trait ProcessManager: Send + Sync {
    /// Start a process
    async fn start_process(&mut self, process_id: &str) -> Result<()>;
    
    /// Stop a process
    async fn stop_process(&mut self, process_id: &str) -> Result<()>;
    
    /// Pause a process
    async fn pause_process(&mut self, process_id: &str) -> Result<()>;
    
    /// Resume a process
    async fn resume_process(&mut self, process_id: &str) -> Result<()>;
    
    /// Get process status
    async fn get_process(&self, process_id: &str) -> Result<SystemProcess>;
    
    /// List all processes
    async fn list_processes(&self) -> Result<Vec<SystemProcess>>;
    
    /// Execute a process step
    async fn execute_step(&mut self, process_id: &str) -> Result<()>;
}

/// Process execution context
#[derive(Clone)]
pub struct ProcessContext {
    /// The process being executed
    pub process: Arc<RwLock<SystemProcess>>,
    
    /// Shared state across processes
    pub shared_state: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    
    /// Message passing channel
    pub message_tx: tokio::sync::mpsc::Sender<ProcessMessage>,
    
    /// Message receiving channel
    pub message_rx: Arc<RwLock<tokio::sync::mpsc::Receiver<ProcessMessage>>>,
}

/// Messages between processes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessMessage {
    /// Source process ID
    pub from: String,
    
    /// Target process ID (None for broadcast)
    pub to: Option<String>,
    
    /// Message type
    pub message_type: MessageType,
    
    /// Message payload
    pub payload: serde_json::Value,
    
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of inter-process messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MessageType {
    /// Data message
    Data,
    
    /// Control message
    Control,
    
    /// Status update
    Status,
    
    /// Request
    Request,
    
    /// Response
    Response,
    
    /// Error
    Error,
}

/// Process orchestrator for complex workflows
pub struct ProcessOrchestrator {
    /// All managed processes
    processes: Arc<RwLock<HashMap<String, SystemProcess>>>,
    
    /// Execution order based on dependencies
    execution_order: Arc<RwLock<Vec<String>>>,
    
    /// Process contexts
    contexts: Arc<RwLock<HashMap<String, ProcessContext>>>,
}

impl ProcessOrchestrator {
    /// Create a new orchestrator
    pub fn new() -> Self {
        Self {
            processes: Arc::new(RwLock::new(HashMap::new())),
            execution_order: Arc::new(RwLock::new(Vec::new())),
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Add a process to the orchestrator
    pub async fn add_process(&self, process: SystemProcess) -> Result<()> {
        let mut processes = self.processes.write().await;
        processes.insert(process.id.clone(), process);
        
        // Recalculate execution order
        self.calculate_execution_order().await?;
        
        Ok(())
    }
    
    /// Calculate execution order based on dependencies
    async fn calculate_execution_order(&self) -> Result<()> {
        let processes = self.processes.read().await;
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();
        
        // Topological sort
        for process_id in processes.keys() {
            if !visited.contains(process_id) {
                self.visit_process(process_id, &processes, &mut visited, &mut visiting, &mut order)?;
            }
        }
        
        let mut execution_order = self.execution_order.write().await;
        *execution_order = order;
        
        Ok(())
    }
    
    /// DFS visit for topological sort
    fn visit_process(
        &self,
        process_id: &str,
        processes: &HashMap<String, SystemProcess>,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        if visiting.contains(process_id) {
            return Err(crate::Error::Config("Circular dependency detected".to_string()));
        }
        
        if visited.contains(process_id) {
            return Ok(());
        }
        
        visiting.insert(process_id.to_string());
        
        if let Some(process) = processes.get(process_id) {
            for dep in &process.dependencies {
                self.visit_process(dep, processes, visited, visiting, order)?;
            }
        }
        
        visiting.remove(process_id);
        visited.insert(process_id.to_string());
        order.push(process_id.to_string());
        
        Ok(())
    }
}