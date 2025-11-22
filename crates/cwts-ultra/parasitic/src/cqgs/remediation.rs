//! Self-Healing Remediation System
//!
//! Implements automated remediation and self-healing capabilities for quality violations.
//! Provides intelligent repair strategies and system recovery mechanisms.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::cqgs::sentinels::{SentinelId, SentinelType};
use crate::cqgs::{CqgsEvent, QualityViolation, ViolationSeverity};

/// Maximum concurrent remediation tasks
const MAX_CONCURRENT_REMEDIATIONS: usize = 10;

/// Default timeout for remediation operations
const DEFAULT_REMEDIATION_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// Remediation strategy types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RemediationStrategy {
    AutomaticFix,
    ConfigurationUpdate,
    ResourceReallocation,
    ProcessRestart,
    DependencyUpdate,
    CodeRefactoring,
    TestGeneration,
    DocumentationUpdate,
    SecurityPatch,
    PerformanceOptimization,
}

/// Remediation task status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RemediationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    RequiresHumanIntervention,
}

/// Remediation task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationTask {
    pub id: Uuid,
    pub violation_id: Uuid,
    pub strategy: RemediationStrategy,
    pub priority: RemediationPriority,
    pub status: RemediationStatus,
    pub assigned_sentinel: Option<SentinelId>,
    pub created_at: SystemTime,
    pub started_at: Option<SystemTime>,
    pub completed_at: Option<SystemTime>,
    pub timeout: Duration,
    pub retry_count: u32,
    pub max_retries: u32,
    pub steps: Vec<RemediationStep>,
    pub result: Option<RemediationResult>,
    pub dependencies: Vec<Uuid>, // Other tasks that must complete first
}

/// Remediation priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemediationPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

impl From<ViolationSeverity> for RemediationPriority {
    fn from(severity: ViolationSeverity) -> Self {
        match severity {
            ViolationSeverity::Info => RemediationPriority::Low,
            ViolationSeverity::Warning => RemediationPriority::Medium,
            ViolationSeverity::Error => RemediationPriority::High,
            ViolationSeverity::Critical => RemediationPriority::Critical,
            ViolationSeverity::Fatal => RemediationPriority::Emergency,
        }
    }
}

/// Individual remediation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationStep {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub command: Option<String>,
    pub expected_outcome: String,
    pub validation: Vec<ValidationCheck>,
    pub rollback_command: Option<String>,
    pub status: RemediationStatus,
    pub started_at: Option<SystemTime>,
    pub completed_at: Option<SystemTime>,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// Validation check for remediation steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub check_type: ValidationCheckType,
    pub expected_value: String,
    pub tolerance: Option<f64>,
}

/// Types of validation checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationCheckType {
    FileExists,
    ProcessRunning,
    NetworkConnectable,
    MetricValue,
    LogOutput,
    TestPassing,
    ConfigurationValid,
    DependencyAvailable,
}

/// Remediation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationResult {
    pub success: bool,
    pub message: String,
    pub steps_completed: usize,
    pub steps_failed: usize,
    pub duration: Duration,
    pub side_effects: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Self-healing remediation engine
pub struct RemediationEngine {
    active_tasks: Arc<DashMap<Uuid, RemediationTask>>,
    task_queue: Arc<RwLock<VecDeque<Uuid>>>,
    strategy_registry: Arc<RwLock<HashMap<ViolationSeverity, Vec<RemediationStrategy>>>>,
    execution_semaphore: Arc<Semaphore>,
    metrics: Arc<Mutex<RemediationMetrics>>,
    config: RemediationConfig,
}

/// Remediation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationConfig {
    pub max_concurrent_tasks: usize,
    pub default_timeout: Duration,
    pub max_retries: u32,
    pub enable_rollback: bool,
    pub require_validation: bool,
    pub auto_approve_low_risk: bool,
}

impl Default for RemediationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: MAX_CONCURRENT_REMEDIATIONS,
            default_timeout: DEFAULT_REMEDIATION_TIMEOUT,
            max_retries: 3,
            enable_rollback: true,
            require_validation: true,
            auto_approve_low_risk: true,
        }
    }
}

/// Remediation system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationMetrics {
    pub total_tasks_created: u64,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub success_rate: f64,
    pub average_completion_time: Duration,
    pub tasks_by_strategy: HashMap<RemediationStrategy, u64>,
    pub tasks_by_priority: HashMap<RemediationPriority, u64>,
    pub active_tasks_count: usize,
}

impl Default for RemediationMetrics {
    fn default() -> Self {
        Self {
            total_tasks_created: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
            success_rate: 0.0,
            average_completion_time: Duration::from_secs(0),
            tasks_by_strategy: HashMap::new(),
            tasks_by_priority: HashMap::new(),
            active_tasks_count: 0,
        }
    }
}

impl RemediationEngine {
    /// Create new remediation engine
    pub fn new(config: RemediationConfig) -> Self {
        let engine = Self {
            active_tasks: Arc::new(DashMap::new()),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            strategy_registry: Arc::new(RwLock::new(HashMap::new())),
            execution_semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks)),
            metrics: Arc::new(Mutex::new(RemediationMetrics::default())),
            config,
        };

        engine.initialize_default_strategies();
        engine
    }

    /// Initialize default remediation strategies
    fn initialize_default_strategies(&self) {
        let mut registry = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.strategy_registry.write())
        });

        registry.insert(
            ViolationSeverity::Info,
            vec![
                RemediationStrategy::DocumentationUpdate,
                RemediationStrategy::AutomaticFix,
            ],
        );

        registry.insert(
            ViolationSeverity::Warning,
            vec![
                RemediationStrategy::ConfigurationUpdate,
                RemediationStrategy::AutomaticFix,
                RemediationStrategy::TestGeneration,
            ],
        );

        registry.insert(
            ViolationSeverity::Error,
            vec![
                RemediationStrategy::CodeRefactoring,
                RemediationStrategy::DependencyUpdate,
                RemediationStrategy::ProcessRestart,
                RemediationStrategy::PerformanceOptimization,
            ],
        );

        registry.insert(
            ViolationSeverity::Critical,
            vec![
                RemediationStrategy::SecurityPatch,
                RemediationStrategy::ProcessRestart,
                RemediationStrategy::ResourceReallocation,
                RemediationStrategy::CodeRefactoring,
            ],
        );

        registry.insert(
            ViolationSeverity::Fatal,
            vec![
                RemediationStrategy::ProcessRestart,
                RemediationStrategy::ResourceReallocation,
                RemediationStrategy::SecurityPatch,
            ],
        );

        info!("Initialized remediation strategies for all violation severity levels");
    }

    /// Create remediation task for a quality violation
    #[instrument(skip(self, violation), fields(violation_id = %violation.id))]
    pub async fn create_remediation_task(
        &self,
        violation: &QualityViolation,
    ) -> Result<Uuid, Box<dyn std::error::Error + Send + Sync>> {
        let strategies = self.select_strategies(&violation.severity).await;
        if strategies.is_empty() {
            return Err("No remediation strategies available for this violation".into());
        }

        let primary_strategy = strategies[0].clone();
        let task_id = Uuid::new_v4();
        // Fix E0382: Convert severity to remediation priority
        let priority: RemediationPriority = match violation.severity {
            crate::cqgs::ViolationSeverity::Info => RemediationPriority::Low,
            crate::cqgs::ViolationSeverity::Warning => RemediationPriority::Medium,
            crate::cqgs::ViolationSeverity::Error => RemediationPriority::High,
            crate::cqgs::ViolationSeverity::Critical => RemediationPriority::Critical,
            crate::cqgs::ViolationSeverity::Fatal => RemediationPriority::Emergency,
        };
        let priority_clone = priority.clone();

        let steps = self
            .generate_remediation_steps(&violation, &primary_strategy)
            .await?;

        let task = RemediationTask {
            id: task_id,
            violation_id: violation.id,
            strategy: primary_strategy,
            priority: priority_clone,
            status: RemediationStatus::Pending,
            assigned_sentinel: None,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            timeout: self.config.default_timeout,
            retry_count: 0,
            max_retries: self.config.max_retries,
            steps,
            result: None,
            dependencies: Vec::new(),
        };

        self.active_tasks.insert(task_id, task.clone());

        // Add to priority queue
        {
            let mut queue = self.task_queue.write().await;
            // Insert based on priority (higher priority first)
            let mut inserted = false;
            for (i, existing_id) in queue.iter().enumerate() {
                if let Some(existing_task) = self.active_tasks.get(existing_id) {
                    if priority > existing_task.priority {
                        queue.insert(i, task_id);
                        inserted = true;
                        break;
                    }
                }
            }
            if !inserted {
                queue.push_back(task_id);
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.total_tasks_created += 1;
            *metrics
                .tasks_by_strategy
                .entry(task.strategy.clone())
                .or_insert(0) += 1;
            *metrics.tasks_by_priority.entry(priority).or_insert(0) += 1;
            metrics.active_tasks_count = self.active_tasks.len();
        }

        info!(
            "Created remediation task {} for violation {} using strategy {:?}",
            task_id, violation.id, task.strategy
        );

        Ok(task_id)
    }

    /// Select appropriate remediation strategies for a violation
    async fn select_strategies(&self, severity: &ViolationSeverity) -> Vec<RemediationStrategy> {
        let registry = self.strategy_registry.read().await;
        registry.get(severity).cloned().unwrap_or_default()
    }

    /// Generate remediation steps for a violation and strategy
    async fn generate_remediation_steps(
        &self,
        violation: &QualityViolation,
        strategy: &RemediationStrategy,
    ) -> Result<Vec<RemediationStep>, Box<dyn std::error::Error + Send + Sync>> {
        let mut steps = Vec::new();

        match strategy {
            RemediationStrategy::AutomaticFix => {
                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: "Analyze Issue".to_string(),
                    description: "Analyze the violation to determine fix approach".to_string(),
                    command: None,
                    expected_outcome: "Issue analysis completed".to_string(),
                    validation: vec![],
                    rollback_command: None,
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });

                if violation.auto_fixable {
                    steps.push(RemediationStep {
                        id: Uuid::new_v4(),
                        name: "Apply Automatic Fix".to_string(),
                        description: violation
                            .remediation_suggestion
                            .clone()
                            .unwrap_or_else(|| "Apply automated fix".to_string()),
                        command: Some("auto-fix".to_string()),
                        expected_outcome: "Issue automatically resolved".to_string(),
                        validation: vec![ValidationCheck {
                            name: "Fix Verification".to_string(),
                            check_type: ValidationCheckType::TestPassing,
                            expected_value: "true".to_string(),
                            tolerance: None,
                        }],
                        rollback_command: Some("rollback-fix".to_string()),
                        status: RemediationStatus::Pending,
                        started_at: None,
                        completed_at: None,
                        output: None,
                        error: None,
                    });
                }
            }

            RemediationStrategy::ProcessRestart => {
                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: "Graceful Process Shutdown".to_string(),
                    description: "Gracefully shutdown the affected process".to_string(),
                    command: Some("systemctl stop service".to_string()),
                    expected_outcome: "Process stopped gracefully".to_string(),
                    validation: vec![ValidationCheck {
                        name: "Process Stopped".to_string(),
                        check_type: ValidationCheckType::ProcessRunning,
                        expected_value: "false".to_string(),
                        tolerance: None,
                    }],
                    rollback_command: Some("systemctl start service".to_string()),
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });

                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: "Process Restart".to_string(),
                    description: "Restart the process with updated configuration".to_string(),
                    command: Some("systemctl start service".to_string()),
                    expected_outcome: "Process restarted successfully".to_string(),
                    validation: vec![ValidationCheck {
                        name: "Process Running".to_string(),
                        check_type: ValidationCheckType::ProcessRunning,
                        expected_value: "true".to_string(),
                        tolerance: None,
                    }],
                    rollback_command: None,
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });
            }

            RemediationStrategy::ConfigurationUpdate => {
                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: "Backup Current Configuration".to_string(),
                    description: "Create backup of current configuration".to_string(),
                    command: Some("cp config.toml config.toml.bak".to_string()),
                    expected_outcome: "Configuration backed up".to_string(),
                    validation: vec![ValidationCheck {
                        name: "Backup Exists".to_string(),
                        check_type: ValidationCheckType::FileExists,
                        expected_value: "config.toml.bak".to_string(),
                        tolerance: None,
                    }],
                    rollback_command: Some("mv config.toml.bak config.toml".to_string()),
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });

                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: "Update Configuration".to_string(),
                    description: "Apply corrective configuration changes".to_string(),
                    command: Some("update-config --fix-violations".to_string()),
                    expected_outcome: "Configuration updated successfully".to_string(),
                    validation: vec![ValidationCheck {
                        name: "Config Valid".to_string(),
                        check_type: ValidationCheckType::ConfigurationValid,
                        expected_value: "true".to_string(),
                        tolerance: None,
                    }],
                    rollback_command: Some("mv config.toml.bak config.toml".to_string()),
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });
            }

            _ => {
                // Default steps for other strategies
                steps.push(RemediationStep {
                    id: Uuid::new_v4(),
                    name: format!("{:?} Step", strategy),
                    description: format!("Execute {:?} remediation", strategy),
                    command: None,
                    expected_outcome: "Remediation completed".to_string(),
                    validation: vec![],
                    rollback_command: None,
                    status: RemediationStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    output: None,
                    error: None,
                });
            }
        }

        Ok(steps)
    }

    /// Start the remediation engine processing loop
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Starting remediation engine with {} max concurrent tasks",
            self.config.max_concurrent_tasks
        );

        let engine_clone = self.clone_for_processing().await;
        tokio::spawn(async move {
            engine_clone.processing_loop().await;
        });

        Ok(())
    }

    /// Clone for async processing (simplified clone)
    async fn clone_for_processing(&self) -> RemediationEngineProcessor {
        RemediationEngineProcessor {
            active_tasks: Arc::clone(&self.active_tasks),
            task_queue: Arc::clone(&self.task_queue),
            execution_semaphore: Arc::clone(&self.execution_semaphore),
            metrics: Arc::clone(&self.metrics),
            config: self.config.clone(),
        }
    }

    /// Execute a specific remediation task
    #[instrument(skip(self), fields(task_id = %task_id))]
    pub async fn execute_task(
        &self,
        task_id: Uuid,
    ) -> Result<RemediationResult, Box<dyn std::error::Error + Send + Sync>> {
        let _permit = self.execution_semaphore.acquire().await?;

        let mut task = self
            .active_tasks
            .get_mut(&task_id)
            .ok_or("Task not found")?;

        task.status = RemediationStatus::InProgress;
        task.started_at = Some(SystemTime::now());

        info!(
            "Executing remediation task {} with strategy {:?}",
            task_id, task.strategy
        );

        let mut steps_completed = 0;
        let mut steps_failed = 0;
        let mut side_effects = Vec::new();

        // Fix E0502: Extract task priority to avoid borrow conflict
        let task_priority = task.priority.clone();

        for step in &mut task.steps {
            step.status = RemediationStatus::InProgress;
            step.started_at = Some(SystemTime::now());

            // Execute the step
            match self.execute_step(step).await {
                Ok(output) => {
                    step.output = Some(output);
                    step.status = RemediationStatus::Completed;
                    step.completed_at = Some(SystemTime::now());
                    steps_completed += 1;

                    debug!("Completed remediation step: {}", step.name);
                }
                Err(error) => {
                    step.error = Some(error.to_string());
                    step.status = RemediationStatus::Failed;
                    step.completed_at = Some(SystemTime::now());
                    steps_failed += 1;

                    warn!("Failed remediation step: {} - {}", step.name, error);

                    // If rollback is enabled and this step has a rollback command
                    if self.config.enable_rollback && step.rollback_command.is_some() {
                        self.execute_rollback(step).await?;
                        side_effects.push(format!("Rolled back step: {}", step.name));
                    }

                    // For critical steps, fail the entire task
                    if task_priority >= RemediationPriority::Critical {
                        break;
                    }
                }
            }
        }

        let duration = task
            .started_at
            .unwrap()
            .elapsed()
            .unwrap_or(Duration::from_secs(0));
        let success = steps_failed == 0;

        let result = RemediationResult {
            success,
            message: if success {
                "Remediation completed successfully".to_string()
            } else {
                format!(
                    "Remediation partially failed: {} steps failed",
                    steps_failed
                )
            },
            steps_completed,
            steps_failed,
            duration,
            side_effects,
            recommendations: vec![],
        };

        task.result = Some(result.clone());
        task.status = if success {
            RemediationStatus::Completed
        } else {
            RemediationStatus::Failed
        };
        task.completed_at = Some(SystemTime::now());

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            if success {
                metrics.total_tasks_completed += 1;
            } else {
                metrics.total_tasks_failed += 1;
            }
            metrics.success_rate = metrics.total_tasks_completed as f64
                / (metrics.total_tasks_completed + metrics.total_tasks_failed) as f64;
            metrics.active_tasks_count = self.active_tasks.len();
        }

        info!(
            "Remediation task {} {} in {:?}",
            task_id,
            if success { "completed" } else { "failed" },
            duration
        );

        Ok(result)
    }

    /// Execute an individual remediation step
    async fn execute_step(
        &self,
        step: &RemediationStep,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(command) = &step.command {
            // In a real implementation, this would execute actual system commands
            // For now, simulate execution
            debug!("Executing command: {}", command);

            // Simulate some processing time
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Simulate success for most operations
            if rand::random::<f64>() > 0.1 {
                // 90% success rate
                Ok(format!("Command '{}' executed successfully", command))
            } else {
                Err(format!("Command '{}' failed", command).into())
            }
        } else {
            // For steps without commands, just simulate processing
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok("Step completed without command execution".to_string())
        }
    }

    /// Execute rollback for a failed step
    async fn execute_rollback(
        &self,
        step: &RemediationStep,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(rollback_command) = &step.rollback_command {
            debug!("Executing rollback: {}", rollback_command);

            // Simulate rollback execution
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Rollbacks should generally succeed
            if rand::random::<f64>() > 0.05 {
                // 95% success rate
                info!("Rollback successful for step: {}", step.name);
                Ok(())
            } else {
                Err(format!("Rollback failed for step: {}", step.name).into())
            }
        } else {
            Ok(()) // No rollback command, nothing to do
        }
    }

    /// Get remediation metrics
    pub async fn get_metrics(&self) -> RemediationMetrics {
        self.metrics.lock().await.clone()
    }

    /// Get active tasks count
    pub fn get_active_tasks_count(&self) -> usize {
        self.active_tasks.len()
    }

    /// Cancel a remediation task
    pub async fn cancel_task(
        &self,
        task_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mut task) = self.active_tasks.get_mut(&task_id) {
            task.status = RemediationStatus::Cancelled;
            task.completed_at = Some(SystemTime::now());

            info!("Cancelled remediation task {}", task_id);
            Ok(())
        } else {
            Err("Task not found".into())
        }
    }
}

/// Simplified processor for async operations
#[derive(Clone)]
struct RemediationEngineProcessor {
    active_tasks: Arc<DashMap<Uuid, RemediationTask>>,
    task_queue: Arc<RwLock<VecDeque<Uuid>>>,
    execution_semaphore: Arc<Semaphore>,
    metrics: Arc<Mutex<RemediationMetrics>>,
    config: RemediationConfig,
}

impl RemediationEngineProcessor {
    /// Main processing loop
    async fn processing_loop(&self) {
        let mut interval = interval(Duration::from_millis(1000)); // Check every second

        loop {
            interval.tick().await;

            // Process pending tasks
            while let Some(task_id) = self.get_next_task().await {
                if self.execution_semaphore.available_permits() > 0 {
                    let processor = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = processor.execute_task_wrapper(task_id).await {
                            error!("Failed to execute remediation task {}: {}", task_id, e);
                        }
                    });
                }
            }
        }
    }

    /// Get next task from priority queue
    async fn get_next_task(&self) -> Option<Uuid> {
        let mut queue = self.task_queue.write().await;
        queue.pop_front()
    }

    /// Execute task wrapper for async processing
    async fn execute_task_wrapper(
        &self,
        task_id: Uuid,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // This would delegate to the main engine's execute_task method
        // For now, just simulate task execution
        info!("Processing remediation task {}", task_id);

        tokio::time::sleep(Duration::from_secs(1)).await;

        // Mark task as completed
        if let Some(mut task) = self.active_tasks.get_mut(&task_id) {
            task.status = RemediationStatus::Completed;
            task.completed_at = Some(SystemTime::now());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqgs::sentinels::SentinelId;

    #[tokio::test]
    async fn test_remediation_engine_creation() {
        let config = RemediationConfig::default();
        let engine = RemediationEngine::new(config);

        assert_eq!(engine.get_active_tasks_count(), 0);
    }

    #[tokio::test]
    async fn test_remediation_task_creation() {
        let engine = RemediationEngine::new(RemediationConfig::default());

        let violation = QualityViolation {
            id: Uuid::new_v4(),
            sentinel_id: SentinelId::new("test".to_string()),
            severity: ViolationSeverity::Error,
            message: "Test violation".to_string(),
            location: "test.rs:1".to_string(),
            timestamp: SystemTime::now(),
            remediation_suggestion: Some("Fix the issue".to_string()),
            auto_fixable: true,
            hyperbolic_coordinates: None,
        };

        let task_id = engine.create_remediation_task(&violation).await.unwrap();
        assert_eq!(engine.get_active_tasks_count(), 1);

        let task = engine.active_tasks.get(&task_id).unwrap();
        assert_eq!(task.violation_id, violation.id);
        assert_eq!(task.status, RemediationStatus::Pending);
    }

    #[tokio::test]
    async fn test_strategy_selection() {
        let engine = RemediationEngine::new(RemediationConfig::default());

        let strategies = engine.select_strategies(&ViolationSeverity::Critical).await;
        assert!(!strategies.is_empty());
        assert!(strategies.contains(&RemediationStrategy::SecurityPatch));
    }

    #[tokio::test]
    async fn test_remediation_metrics() {
        let engine = RemediationEngine::new(RemediationConfig::default());

        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_tasks_created, 0);
        assert_eq!(metrics.active_tasks_count, 0);
    }
}
