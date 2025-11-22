use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use tracing::{error, info, instrument, warn};
use chrono::{DateTime, Utc};

/// Production health check endpoints for Constitutional Prime Directive compliance
/// Implements comprehensive system health validation for zero-downtime deployment

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: DateTime<Utc>,
    pub version: String,
    pub uptime_seconds: u64,
    pub constitutional_compliance: bool,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub response_time_ms: u64,
    pub message: String,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

#[derive(Debug, Serialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub status: HealthStatus,
    pub timestamp: DateTime<Utc>,
    pub dependencies: ReadinessDependencies,
    pub constitutional_compliance: bool,
}

#[derive(Debug, Serialize)]
pub struct ReadinessDependencies {
    pub binance_connectivity: bool,
    pub e2b_sandboxes: bool,
    pub database: bool,
    pub model_accuracy_threshold: bool,
    pub emergency_systems: bool,
}

#[derive(Debug, Serialize)]
pub struct LivenessResponse {
    pub alive: bool,
    pub status: HealthStatus,
    pub timestamp: DateTime<Utc>,
    pub last_successful_calculation: Option<DateTime<Utc>>,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub constitutional_violations: u64,
}

#[derive(Debug, Serialize)]
pub struct StartupResponse {
    pub started: bool,
    pub status: HealthStatus,
    pub timestamp: DateTime<Utc>,
    pub startup_duration_seconds: f64,
    pub initialization_complete: bool,
    pub constitutional_compliance_verified: bool,
}

pub struct HealthService {
    binance_client: Arc<dyn BinanceHealthChecker + Send + Sync>,
    e2b_client: Arc<dyn E2BHealthChecker + Send + Sync>,
    database_client: Arc<dyn DatabaseHealthChecker + Send + Sync>,
    model_service: Arc<dyn ModelHealthChecker + Send + Sync>,
    metrics_collector: Arc<dyn MetricsCollector + Send + Sync>,
    startup_time: DateTime<Utc>,
}

// Health checker traits for dependency injection
pub trait BinanceHealthChecker {
    async fn check_connectivity(&self) -> Result<BinanceHealth, HealthError>;
}

pub trait E2BHealthChecker {
    async fn check_sandboxes(&self) -> Result<E2BHealth, HealthError>;
}

pub trait DatabaseHealthChecker {
    async fn check_connection(&self) -> Result<DatabaseHealth, HealthError>;
}

pub trait ModelHealthChecker {
    async fn check_accuracy(&self) -> Result<ModelHealth, HealthError>;
}

pub trait MetricsCollector {
    async fn get_system_metrics(&self) -> Result<SystemMetrics, HealthError>;
}

#[derive(Debug)]
pub struct BinanceHealth {
    pub connected: bool,
    pub active_connections: u32,
    pub last_message_time: Option<DateTime<Utc>>,
    pub websocket_status: String,
}

#[derive(Debug)]
pub struct E2BHealth {
    pub sandboxes_healthy: bool,
    pub active_sandboxes: u32,
    pub training_status: String,
    pub last_successful_training: Option<DateTime<Utc>>,
}

#[derive(Debug)]
pub struct DatabaseHealth {
    pub connected: bool,
    pub query_response_time_ms: u64,
    pub connection_pool_size: u32,
    pub active_connections: u32,
}

#[derive(Debug)]
pub struct ModelHealth {
    pub accuracy: f64,
    pub last_training: Option<DateTime<Utc>>,
    pub convergence_status: String,
    pub performance_score: f64,
}

#[derive(Debug)]
pub struct SystemMetrics {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub constitutional_violations: u64,
    pub emergency_stops: u64,
    pub last_successful_calculation: Option<DateTime<Utc>>,
}

#[derive(Debug, thiserror::Error)]
pub enum HealthError {
    #[error("Binance connectivity error: {0}")]
    BinanceError(String),
    
    #[error("E2B sandbox error: {0}")]
    E2BError(String),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Model health error: {0}")]
    ModelError(String),
    
    #[error("Metrics collection error: {0}")]
    MetricsError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Constitutional Prime Directive violation: {0}")]
    ConstitutionalViolation(String),
}

impl HealthService {
    pub fn new(
        binance_client: Arc<dyn BinanceHealthChecker + Send + Sync>,
        e2b_client: Arc<dyn E2BHealthChecker + Send + Sync>,
        database_client: Arc<dyn DatabaseHealthChecker + Send + Sync>,
        model_service: Arc<dyn ModelHealthChecker + Send + Sync>,
        metrics_collector: Arc<dyn MetricsCollector + Send + Sync>,
    ) -> Self {
        Self {
            binance_client,
            e2b_client,
            database_client,
            model_service,
            metrics_collector,
            startup_time: Utc::now(),
        }
    }
    
    /// Comprehensive health check with Constitutional Prime Directive compliance validation
    #[instrument(skip(self))]
    pub async fn comprehensive_health_check(&self) -> HealthResponse {
        let start_time = std::time::Instant::now();
        let mut checks = Vec::new();
        let mut overall_status = HealthStatus::Healthy;
        let mut constitutional_compliance = true;
        
        // Parallel health checks for better performance
        let (binance_result, e2b_result, database_result, model_result, metrics_result) = tokio::join!(
            self.check_binance_health(),
            self.check_e2b_health(),
            self.check_database_health(),
            self.check_model_health(),
            self.check_system_metrics()
        );
        
        // Process Binance health check
        match binance_result {
            Ok(check) => {
                if check.status != HealthStatus::Healthy {
                    overall_status = HealthStatus::Degraded;
                    if check.status == HealthStatus::Critical {
                        constitutional_compliance = false;
                    }
                }
                checks.push(check);
            }
            Err(e) => {
                error!("Binance health check failed: {}", e);
                overall_status = HealthStatus::Critical;
                constitutional_compliance = false;
                checks.push(HealthCheck {
                    name: "binance_connectivity".to_string(),
                    status: HealthStatus::Critical,
                    response_time_ms: start_time.elapsed().as_millis() as u64,
                    message: format!("CRITICAL: Binance health check failed - {}", e),
                    last_check: Utc::now(),
                });
            }
        }
        
        // Process E2B health check
        match e2b_result {
            Ok(check) => {
                if check.status != HealthStatus::Healthy && check.status == HealthStatus::Critical {
                    overall_status = HealthStatus::Critical;
                    constitutional_compliance = false;
                }
                checks.push(check);
            }
            Err(e) => {
                warn!("E2B health check failed: {}", e);
                checks.push(HealthCheck {
                    name: "e2b_sandboxes".to_string(),
                    status: HealthStatus::Degraded,
                    response_time_ms: start_time.elapsed().as_millis() as u64,
                    message: format!("E2B sandbox health check failed - {}", e),
                    last_check: Utc::now(),
                });
            }
        }
        
        // Process database health check
        match database_result {
            Ok(check) => {
                if check.status == HealthStatus::Critical {
                    overall_status = HealthStatus::Critical;
                    constitutional_compliance = false;
                }
                checks.push(check);
            }
            Err(e) => {
                error!("Database health check failed: {}", e);
                overall_status = HealthStatus::Critical;
                constitutional_compliance = false;
                checks.push(HealthCheck {
                    name: "database_connection".to_string(),
                    status: HealthStatus::Critical,
                    response_time_ms: start_time.elapsed().as_millis() as u64,
                    message: format!("CRITICAL: Database health check failed - {}", e),
                    last_check: Utc::now(),
                });
            }
        }
        
        // Process model health check
        match model_result {
            Ok(check) => {
                if check.status != HealthStatus::Healthy {
                    if overall_status == HealthStatus::Healthy {
                        overall_status = HealthStatus::Degraded;
                    }
                    if check.status == HealthStatus::Critical {
                        constitutional_compliance = false;
                    }
                }
                checks.push(check);
            }
            Err(e) => {
                warn!("Model health check failed: {}", e);
                checks.push(HealthCheck {
                    name: "model_accuracy".to_string(),
                    status: HealthStatus::Degraded,
                    response_time_ms: start_time.elapsed().as_millis() as u64,
                    message: format!("Model health check failed - {}", e),
                    last_check: Utc::now(),
                });
            }
        }
        
        // Process system metrics
        match metrics_result {
            Ok(check) => {
                if check.status == HealthStatus::Critical {
                    overall_status = HealthStatus::Critical;
                    constitutional_compliance = false;
                }
                checks.push(check);
            }
            Err(e) => {
                error!("System metrics check failed: {}", e);
                overall_status = HealthStatus::Critical;
                constitutional_compliance = false;
            }
        }
        
        let uptime = Utc::now().signed_duration_since(self.startup_time).num_seconds() as u64;
        
        info!(
            status = ?overall_status,
            constitutional_compliance = constitutional_compliance,
            uptime_seconds = uptime,
            checks_count = checks.len(),
            "Health check completed"
        );
        
        HealthResponse {
            status: overall_status,
            timestamp: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: uptime,
            constitutional_compliance,
            checks,
        }
    }
    
    /// Readiness probe for Kubernetes zero-downtime deployment
    #[instrument(skip(self))]
    pub async fn readiness_check(&self) -> ReadinessResponse {
        let mut ready = true;
        let mut status = HealthStatus::Healthy;
        let mut constitutional_compliance = true;
        
        // Check critical dependencies for readiness
        let binance_ready = match timeout(Duration::from_secs(5), self.binance_client.check_connectivity()).await {
            Ok(Ok(health)) => health.connected && health.active_connections > 0,
            _ => {
                ready = false;
                status = HealthStatus::Critical;
                constitutional_compliance = false;
                false
            }
        };
        
        let e2b_ready = match timeout(Duration::from_secs(10), self.e2b_client.check_sandboxes()).await {
            Ok(Ok(health)) => health.sandboxes_healthy,
            _ => false, // E2B not critical for readiness
        };
        
        let database_ready = match timeout(Duration::from_secs(5), self.database_client.check_connection()).await {
            Ok(Ok(health)) => health.connected,
            _ => {
                ready = false;
                status = HealthStatus::Critical;
                constitutional_compliance = false;
                false
            }
        };
        
        let model_ready = match timeout(Duration::from_secs(5), self.model_service.check_accuracy()).await {
            Ok(Ok(health)) => health.accuracy >= 0.95, // Constitutional Prime Directive threshold
            _ => {
                ready = false;
                status = HealthStatus::Degraded;
                false
            }
        };
        
        let emergency_systems_ready = match timeout(Duration::from_secs(2), self.metrics_collector.get_system_metrics()).await {
            Ok(Ok(metrics)) => metrics.emergency_stops == 0 && metrics.constitutional_violations == 0,
            _ => {
                ready = false;
                status = HealthStatus::Critical;
                constitutional_compliance = false;
                false
            }
        };
        
        if !ready {
            warn!(
                binance_ready = binance_ready,
                database_ready = database_ready,
                model_ready = model_ready,
                emergency_systems_ready = emergency_systems_ready,
                "System not ready for traffic"
            );
        }
        
        ReadinessResponse {
            ready,
            status,
            timestamp: Utc::now(),
            dependencies: ReadinessDependencies {
                binance_connectivity: binance_ready,
                e2b_sandboxes: e2b_ready,
                database: database_ready,
                model_accuracy_threshold: model_ready,
                emergency_systems: emergency_systems_ready,
            },
            constitutional_compliance,
        }
    }
    
    /// Liveness probe for Kubernetes
    #[instrument(skip(self))]
    pub async fn liveness_check(&self) -> LivenessResponse {
        let mut alive = true;
        let mut status = HealthStatus::Healthy;
        
        // Get system metrics for liveness assessment
        let metrics = match timeout(Duration::from_secs(5), self.metrics_collector.get_system_metrics()).await {
            Ok(Ok(metrics)) => metrics,
            Ok(Err(e)) => {
                error!("Failed to get system metrics for liveness: {}", e);
                alive = false;
                status = HealthStatus::Critical;
                SystemMetrics {
                    memory_usage_mb: 0.0,
                    cpu_usage_percent: 0.0,
                    constitutional_violations: 0,
                    emergency_stops: 0,
                    last_successful_calculation: None,
                }
            }
            Err(_) => {
                error!("Timeout getting system metrics for liveness");
                alive = false;
                status = HealthStatus::Critical;
                SystemMetrics {
                    memory_usage_mb: 0.0,
                    cpu_usage_percent: 0.0,
                    constitutional_violations: 0,
                    emergency_stops: 0,
                    last_successful_calculation: None,
                }
            }
        };
        
        // Check for emergency stops or constitutional violations
        if metrics.emergency_stops > 0 || metrics.constitutional_violations > 0 {
            alive = false;
            status = HealthStatus::Critical;
            error!(
                emergency_stops = metrics.emergency_stops,
                constitutional_violations = metrics.constitutional_violations,
                "Constitutional Prime Directive violation detected in liveness check"
            );
        }
        
        // Check if system is completely stalled (no calculations in 10 minutes)
        if let Some(last_calc) = metrics.last_successful_calculation {
            let stale_duration = Utc::now().signed_duration_since(last_calc);
            if stale_duration.num_minutes() > 10 {
                alive = false;
                status = HealthStatus::Critical;
                error!(
                    minutes_since_last_calculation = stale_duration.num_minutes(),
                    "System appears stalled - no calculations in 10+ minutes"
                );
            }
        }
        
        LivenessResponse {
            alive,
            status,
            timestamp: Utc::now(),
            last_successful_calculation: metrics.last_successful_calculation,
            memory_usage_mb: metrics.memory_usage_mb,
            cpu_usage_percent: metrics.cpu_usage_percent,
            constitutional_violations: metrics.constitutional_violations,
        }
    }
    
    /// Startup probe for Kubernetes
    #[instrument(skip(self))]
    pub async fn startup_check(&self) -> StartupResponse {
        let startup_duration = Utc::now().signed_duration_since(self.startup_time);
        let mut started = true;
        let mut status = HealthStatus::Healthy;
        let mut constitutional_compliance = true;
        
        // Basic connectivity checks during startup
        let basic_checks = tokio::join!(
            timeout(Duration::from_secs(10), self.binance_client.check_connectivity()),
            timeout(Duration::from_secs(15), self.e2b_client.check_sandboxes()),
            timeout(Duration::from_secs(10), self.database_client.check_connection())
        );
        
        // Evaluate startup health
        if basic_checks.0.is_err() || basic_checks.2.is_err() {
            started = false;
            status = HealthStatus::Critical;
            constitutional_compliance = false;
            warn!("Critical dependencies not available during startup");
        }
        
        // Check if startup is taking too long (> 5 minutes is concerning)
        if startup_duration.num_minutes() > 5 {
            status = HealthStatus::Degraded;
            warn!(
                startup_duration_minutes = startup_duration.num_minutes(),
                "Startup taking longer than expected"
            );
        }
        
        StartupResponse {
            started,
            status,
            timestamp: Utc::now(),
            startup_duration_seconds: startup_duration.num_milliseconds() as f64 / 1000.0,
            initialization_complete: started,
            constitutional_compliance_verified: constitutional_compliance,
        }
    }
    
    // Helper methods for individual health checks
    async fn check_binance_health(&self) -> Result<HealthCheck, HealthError> {
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(10), self.binance_client.check_connectivity()).await {
            Ok(Ok(health)) => {
                let status = if health.connected && health.active_connections > 0 {
                    HealthStatus::Healthy
                } else if health.connected {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Critical
                };
                
                Ok(HealthCheck {
                    name: "binance_connectivity".to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    message: format!("Connected: {}, Active connections: {}", 
                                     health.connected, health.active_connections),
                    last_check: Utc::now(),
                })
            }
            Ok(Err(e)) => Err(HealthError::BinanceError(e.to_string())),
            Err(_) => Err(HealthError::TimeoutError("Binance health check timeout".to_string())),
        }
    }
    
    async fn check_e2b_health(&self) -> Result<HealthCheck, HealthError> {
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(15), self.e2b_client.check_sandboxes()).await {
            Ok(Ok(health)) => {
                let status = if health.sandboxes_healthy && health.active_sandboxes > 0 {
                    HealthStatus::Healthy
                } else if health.active_sandboxes > 0 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Critical
                };
                
                Ok(HealthCheck {
                    name: "e2b_sandboxes".to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    message: format!("Healthy: {}, Active sandboxes: {}", 
                                     health.sandboxes_healthy, health.active_sandboxes),
                    last_check: Utc::now(),
                })
            }
            Ok(Err(e)) => Err(HealthError::E2BError(e.to_string())),
            Err(_) => Err(HealthError::TimeoutError("E2B health check timeout".to_string())),
        }
    }
    
    async fn check_database_health(&self) -> Result<HealthCheck, HealthError> {
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(10), self.database_client.check_connection()).await {
            Ok(Ok(health)) => {
                let status = if health.connected && health.query_response_time_ms < 1000 {
                    HealthStatus::Healthy
                } else if health.connected {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Critical
                };
                
                Ok(HealthCheck {
                    name: "database_connection".to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    message: format!("Connected: {}, Query time: {}ms, Pool: {}/{}", 
                                     health.connected, health.query_response_time_ms,
                                     health.active_connections, health.connection_pool_size),
                    last_check: Utc::now(),
                })
            }
            Ok(Err(e)) => Err(HealthError::DatabaseError(e.to_string())),
            Err(_) => Err(HealthError::TimeoutError("Database health check timeout".to_string())),
        }
    }
    
    async fn check_model_health(&self) -> Result<HealthCheck, HealthError> {
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(10), self.model_service.check_accuracy()).await {
            Ok(Ok(health)) => {
                let status = if health.accuracy >= 0.95 {
                    HealthStatus::Healthy
                } else if health.accuracy >= 0.90 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Critical
                };
                
                Ok(HealthCheck {
                    name: "model_accuracy".to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    message: format!("Accuracy: {:.2}%, Performance: {:.2}", 
                                     health.accuracy * 100.0, health.performance_score),
                    last_check: Utc::now(),
                })
            }
            Ok(Err(e)) => Err(HealthError::ModelError(e.to_string())),
            Err(_) => Err(HealthError::TimeoutError("Model health check timeout".to_string())),
        }
    }
    
    async fn check_system_metrics(&self) -> Result<HealthCheck, HealthError> {
        let start = std::time::Instant::now();
        
        match timeout(Duration::from_secs(5), self.metrics_collector.get_system_metrics()).await {
            Ok(Ok(metrics)) => {
                let status = if metrics.constitutional_violations > 0 || metrics.emergency_stops > 0 {
                    HealthStatus::Critical
                } else if metrics.memory_usage_mb > 1800.0 || metrics.cpu_usage_percent > 80.0 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };
                
                Ok(HealthCheck {
                    name: "system_metrics".to_string(),
                    status,
                    response_time_ms: start.elapsed().as_millis() as u64,
                    message: format!("Memory: {:.1}MB, CPU: {:.1}%, Violations: {}", 
                                     metrics.memory_usage_mb, metrics.cpu_usage_percent,
                                     metrics.constitutional_violations),
                    last_check: Utc::now(),
                })
            }
            Ok(Err(e)) => Err(HealthError::MetricsError(e.to_string())),
            Err(_) => Err(HealthError::TimeoutError("System metrics timeout".to_string())),
        }
    }
}

// HTTP handler functions for health endpoints

/// Main health check endpoint
pub async fn health_handler(State(health_service): State<Arc<HealthService>>) -> Result<Json<HealthResponse>, StatusCode> {
    let response = health_service.comprehensive_health_check().await;
    
    let status_code = match response.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still serving traffic
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
        HealthStatus::Critical => StatusCode::SERVICE_UNAVAILABLE,
    };
    
    match status_code {
        StatusCode::OK => Ok(Json(response)),
        _ => {
            error!("Health check failed: {:?}", response);
            Err(status_code)
        }
    }
}

/// Kubernetes readiness probe endpoint
pub async fn readiness_handler(State(health_service): State<Arc<HealthService>>) -> Result<Json<ReadinessResponse>, StatusCode> {
    let response = health_service.readiness_check().await;
    
    if response.ready {
        Ok(Json(response))
    } else {
        error!("Readiness check failed: {:?}", response);
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Kubernetes liveness probe endpoint
pub async fn liveness_handler(State(health_service): State<Arc<HealthService>>) -> Result<Json<LivenessResponse>, StatusCode> {
    let response = health_service.liveness_check().await;
    
    if response.alive {
        Ok(Json(response))
    } else {
        error!("Liveness check failed: {:?}", response);
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Kubernetes startup probe endpoint
pub async fn startup_handler(State(health_service): State<Arc<HealthService>>) -> Result<Json<StartupResponse>, StatusCode> {
    let response = health_service.startup_check().await;
    
    if response.started {
        Ok(Json(response))
    } else {
        error!("Startup check failed: {:?}", response);
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}