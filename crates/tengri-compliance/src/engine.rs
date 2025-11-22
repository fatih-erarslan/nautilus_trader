//! TENGRI Compliance Engine - The heart of zero-tolerance compliance

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rust_decimal::{Decimal, prelude::FromStr};

use crate::audit::{AuditTrail, AuditEventType, AuditOutcome};
use crate::circuit_breaker::{CircuitBreakerManager, CircuitBreaker, TriggerCondition};
use crate::error::{ComplianceError, ComplianceResult};
use crate::metrics::{ComplianceMetrics, PerformanceTracker, TraderMetrics, SymbolMetrics, RiskLevel, TraderStatus};
use crate::rules::{RuleEngine, TradingContext, RuleSet, RuleCategories};
use crate::surveillance::{SurveillanceEngine, TradeRecord, OrderRecord, TradeSide};

use futures;

/// Main TENGRI compliance engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Maximum audit records to keep in memory
    pub max_audit_records: usize,
    
    /// Maximum surveillance history
    pub max_surveillance_history: usize,
    
    /// Real-time monitoring frequency
    pub monitoring_frequency: Duration,
    
    /// Enable automatic circuit breakers
    pub auto_circuit_breakers: bool,
    
    /// Kill switch activation threshold (error rate)
    pub kill_switch_threshold: f64,
    
    /// Rule categories to enforce
    pub enabled_rule_categories: RuleCategories,
    
    /// Strictness level
    pub strictness_level: StrictnessLevel,
    
    /// Performance monitoring
    pub enable_performance_tracking: bool,
    
    /// Write-ahead log path for audit persistence
    pub audit_wal_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrictnessLevel {
    /// Conservative - Very strict rules, low risk tolerance
    Conservative,
    /// Balanced - Moderate rules, balanced risk
    Balanced,
    /// Aggressive - Relaxed rules, higher risk tolerance
    Aggressive,
    /// Custom - User-defined rules
    Custom,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            max_audit_records: 100_000,
            max_surveillance_history: 50_000,
            monitoring_frequency: Duration::from_secs(10),
            auto_circuit_breakers: true,
            kill_switch_threshold: 0.1, // 10% error rate
            enabled_rule_categories: RuleCategories::ALL,
            strictness_level: StrictnessLevel::Balanced,
            enable_performance_tracking: true,
            audit_wal_path: None,
        }
    }
}

/// The main TENGRI compliance engine
pub struct ComplianceEngine {
    config: ComplianceConfig,
    
    // Core components
    rule_engine: Arc<RuleEngine>,
    audit_trail: Arc<AuditTrail>,
    circuit_breaker_manager: Arc<CircuitBreakerManager>,
    surveillance_engine: Arc<SurveillanceEngine>,
    metrics: Arc<ComplianceMetrics>,
    
    // State tracking
    engine_state: Arc<RwLock<EngineState>>,
    startup_time: Instant,
    
    // Performance tracking
    performance_tracker: Option<Arc<PerformanceTracker>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    pub status: EngineStatus,
    pub last_health_check: DateTime<Utc>,
    pub error_count: u64,
    pub total_trades_processed: u64,
    pub critical_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineStatus {
    Starting,
    Running,
    Degraded,
    Emergency,
    Shutdown,
}

impl ComplianceEngine {
    /// Create a new TENGRI compliance engine
    pub async fn new(config: ComplianceConfig) -> ComplianceResult<Self> {
        // Initialize audit trail
        let audit_trail = if let Some(ref wal_path) = config.audit_wal_path {
            Arc::new(AuditTrail::with_wal(config.max_audit_records, wal_path.clone()))
        } else {
            Arc::new(AuditTrail::new(config.max_audit_records))
        };
        
        // Initialize rule engine with appropriate rule set
        let rule_engine = Arc::new(RuleEngine::new());
        let rules = match config.strictness_level {
            StrictnessLevel::Conservative => RuleSet::conservative(),
            StrictnessLevel::Balanced => RuleSet::conservative(), // Use conservative as default for safety
            StrictnessLevel::Aggressive => RuleSet::aggressive(),
            StrictnessLevel::Custom => vec![], // User will add custom rules
        };
        
        for rule in rules {
            rule_engine.add_rule(rule);
        }
        
        // Initialize circuit breaker manager
        let circuit_breaker_manager = Arc::new(CircuitBreakerManager::new());
        
        if config.auto_circuit_breakers {
            // Add default circuit breakers
            Self::setup_default_circuit_breakers(&circuit_breaker_manager);
        }
        
        // Initialize surveillance engine
        let surveillance_engine = Arc::new(SurveillanceEngine::new(config.max_surveillance_history));
        
        // Initialize metrics
        let metrics = Arc::new(ComplianceMetrics::new()?);
        
        // Initialize performance tracker
        let performance_tracker = if config.enable_performance_tracking {
            Some(Arc::new(PerformanceTracker::new(metrics.clone())))
        } else {
            None
        };
        
        let engine = Self {
            config,
            rule_engine,
            audit_trail,
            circuit_breaker_manager,
            surveillance_engine,
            metrics,
            engine_state: Arc::new(RwLock::new(EngineState {
                status: EngineStatus::Starting,
                last_health_check: Utc::now(),
                error_count: 0,
                total_trades_processed: 0,
                critical_errors: Vec::new(),
            })),
            startup_time: Instant::now(),
            performance_tracker,
        };
        
        // Record startup
        engine.audit_trail.record(
            AuditEventType::SystemStartup,
            "system".to_string(),
            serde_json::json!({"version": "1.0.0", "config": engine.config}),
        ).await?;
        
        // Update status to running
        engine.engine_state.write().await.status = EngineStatus::Running;
        
        Ok(engine)
    }

    /// Process a trade through the full compliance pipeline
    pub async fn process_trade(&self, context: TradingContext) -> ComplianceResult<ComplianceDecision> {
        let start_time = Instant::now();
        
        // Record metrics
        self.metrics.record_trade_processed();
        
        // Check if engine is operational
        self.check_engine_health().await?;
        
        // Check circuit breakers first
        self.circuit_breaker_manager.check_all()?;
        
        let decision = if let Some(ref tracker) = self.performance_tracker {
            tracker.track_trade_processing(|| {
                self.internal_process_trade(&context)
            }).await
        } else {
            self.internal_process_trade(&context).await
        };
        
        let processing_time = start_time.elapsed();
        self.metrics.record_trade_processing_time(processing_time.as_secs_f64());
        
        // Update state
        let mut state = self.engine_state.write().await;
        state.total_trades_processed += 1;
        
        match &decision {
            Ok(ComplianceDecision::Approved { .. }) => {
                self.metrics.record_trade_approved();
                self.circuit_breaker_manager.record_success("trade_processing");
            }
            Ok(ComplianceDecision::Rejected { .. }) => {
                self.metrics.record_trade_rejected();
                self.circuit_breaker_manager.record_failure("trade_processing");
            }
            Err(_) => {
                self.metrics.record_compliance_error();
                self.circuit_breaker_manager.record_failure("trade_processing");
                state.error_count += 1;
            }
        }
        
        // Create trade record for surveillance
        if let Ok(ref decision) = decision {
            let trade_record = TradeRecord {
                id: context.order_id,
                timestamp: context.timestamp,
                symbol: context.symbol.clone(),
                side: match context.side {
                    crate::rules::OrderSide::Buy => TradeSide::Buy,
                    crate::rules::OrderSide::Sell => TradeSide::Sell,
                },
                quantity: context.quantity,
                price: context.price.unwrap_or(Decimal::ZERO),
                trader_id: context.trader_id.clone(),
                order_id: context.order_id,
                venue: "internal".to_string(),
                execution_time_ms: processing_time.as_millis() as u64,
            };
            
            if matches!(decision, ComplianceDecision::Approved { .. }) {
                self.surveillance_engine.record_trade(trade_record);
            }
        }
        
        decision
    }

    async fn internal_process_trade(&self, context: &TradingContext) -> ComplianceResult<ComplianceDecision> {
        // Step 1: Evaluate all compliance rules
        let rule_start = Instant::now();
        let rule_results = self.rule_engine.evaluate_all(context).await?;
        let rule_time = rule_start.elapsed();
        
        self.metrics.record_rules_evaluated(rule_results.len() as u64);
        self.metrics.record_rule_evaluation_time(rule_time.as_secs_f64());
        
        // Step 2: Check for rule violations
        let violations: Vec<_> = rule_results.iter()
            .filter(|r| !r.passed)
            .collect();
        
        if !violations.is_empty() {
            self.metrics.record_rule_violation();
            
            // Record in audit trail
            self.audit_trail.record_with_outcome(
                AuditEventType::TradeRejected {
                    order_id: context.order_id,
                    reason: format!("{} rule violations", violations.len()),
                },
                context.trader_id.clone(),
                serde_json::json!({
                    "violations": violations,
                    "context": context
                }),
                AuditOutcome::Critical {
                    alert: "Trade rejected due to compliance violations".to_string(),
                },
            ).await?;
            
            return Ok(ComplianceDecision::Rejected {
                order_id: context.order_id,
                reason: format!("Compliance violations: {}", 
                    violations.iter()
                        .map(|v| v.details.as_deref().unwrap_or("Unknown"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                violations: violations.into_iter().cloned().collect(),
                timestamp: Utc::now(),
            });
        }
        
        // Step 3: Run surveillance analysis (async, non-blocking for this trade)
        let surveillance_patterns = self.surveillance_engine.analyze_patterns().await?;
        
        for pattern in &surveillance_patterns {
            self.metrics.record_suspicious_pattern(&format!("{:?}", pattern.pattern_type));
            
            // Record in audit trail
            self.audit_trail.record(
                AuditEventType::SuspiciousActivityDetected {
                    pattern: format!("{:?}", pattern.pattern_type),
                    confidence: pattern.confidence,
                },
                "surveillance_engine".to_string(),
                serde_json::json!(pattern),
            ).await?;
        }
        
        // Step 4: Record successful compliance check
        self.audit_trail.record(
            AuditEventType::TradeApproved { order_id: context.order_id },
            context.trader_id.clone(),
            serde_json::json!({
                "rule_results": rule_results,
                "context": context
            }),
        ).await?;
        
        Ok(ComplianceDecision::Approved {
            order_id: context.order_id,
            timestamp: Utc::now(),
            rule_results,
            risk_score: self.calculate_trade_risk_score(context, &surveillance_patterns),
        })
    }

    fn calculate_trade_risk_score(&self, context: &TradingContext, patterns: &[crate::surveillance::SuspiciousPattern]) -> f64 {
        let mut risk_score = 0.0;
        
        // Base risk from trade size
        let trade_value = context.quantity * context.price.unwrap_or(Decimal::ZERO);
        let size_risk = (trade_value.to_f64().unwrap_or(0.0) / 1_000_000.0).min(0.3);
        risk_score += size_risk;
        
        // Risk from surveillance patterns
        let pattern_risk: f64 = patterns.iter()
            .map(|p| p.risk_score)
            .sum::<f64>()
            .min(0.5);
        risk_score += pattern_risk;
        
        // Risk from portfolio concentration
        if context.portfolio_value > Decimal::ZERO {
            let concentration = trade_value / context.portfolio_value;
            let concentration_risk = concentration.to_f64().unwrap_or(0.0).min(0.2);
            risk_score += concentration_risk;
        }
        
        risk_score.min(1.0)
    }

    async fn check_engine_health(&self) -> ComplianceResult<()> {
        let state = self.engine_state.read().await;
        
        match state.status {
            EngineStatus::Running => Ok(()),
            EngineStatus::Degraded => {
                tracing::warn!("Compliance engine running in degraded mode");
                Ok(())
            }
            EngineStatus::Emergency => {
                Err(ComplianceError::EngineFailure {
                    component: "compliance_engine".to_string(),
                    error: "Engine in emergency mode".to_string(),
                })
            }
            EngineStatus::Shutdown => {
                Err(ComplianceError::EngineFailure {
                    component: "compliance_engine".to_string(),
                    error: "Engine is shutdown".to_string(),
                })
            }
            _ => {
                Err(ComplianceError::EngineFailure {
                    component: "compliance_engine".to_string(),
                    error: "Engine not ready".to_string(),
                })
            }
        }
    }

    fn setup_default_circuit_breakers(manager: &CircuitBreakerManager) {
        // Trade processing circuit breaker
        let trade_breaker = CircuitBreaker::new(
            "trade_processing".to_string(),
            vec![
                TriggerCondition::ConsecutiveFailures { count: 5 },
                TriggerCondition::FailureRate {
                    threshold: 0.2, // 20% failure rate
                    window: Duration::from_secs(60),
                },
            ],
            Duration::from_secs(300), // 5 minute cooldown
        );
        manager.register_breaker("trade_processing".to_string(), trade_breaker);
        
        // Risk management circuit breaker
        let risk_breaker = CircuitBreaker::new(
            "risk_management".to_string(),
            vec![
                TriggerCondition::LossThreshold {
                    max_loss: Decimal::from(100_000), // $100K loss threshold
                    window: Duration::from_secs(3600), // 1 hour window
                },
            ],
            Duration::from_secs(600), // 10 minute cooldown
        );
        manager.register_breaker("risk_management".to_string(), risk_breaker);
    }

    /// Get comprehensive engine status
    pub async fn get_status(&self) -> EngineStatus {
        let state = self.engine_state.read().await;
        
        // Update metrics
        self.metrics.update_uptime(self.startup_time.elapsed().as_secs_f64());
        
        // Update system health
        let health_score = self.metrics.calculate_system_health_score();
        
        // Determine status based on health and circuit breakers
        let cb_status = self.circuit_breaker_manager.get_status();
        
        if cb_status.kill_switch_active {
            EngineStatus::Emergency
        } else if health_score < 50.0 || cb_status.global_failure_rate > 50.0 {
            EngineStatus::Degraded
        } else {
            state.status.clone()
        }
    }

    /// Perform emergency shutdown
    pub async fn emergency_shutdown(&self, reason: String) -> ComplianceResult<()> {
        // Activate kill switch
        self.circuit_breaker_manager.activate_kill_switch(reason.clone())?;
        
        // Update engine state
        let mut state = self.engine_state.write().await;
        state.status = EngineStatus::Emergency;
        state.critical_errors.push(reason.clone());
        
        // Record in audit trail
        self.audit_trail.record(
            AuditEventType::SystemShutdown { reason: reason.clone() },
            "emergency_system".to_string(),
            serde_json::json!({"timestamp": Utc::now(), "reason": reason}),
        ).await?;
        
        tracing::error!("TENGRI EMERGENCY SHUTDOWN: {}", reason);
        
        Err(ComplianceError::EngineFailure {
            component: "compliance_engine".to_string(),
            error: format!("Emergency shutdown: {}", reason),
        })
    }

    /// Get comprehensive metrics
    pub fn get_metrics(&self) -> Arc<ComplianceMetrics> {
        self.metrics.clone()
    }

    /// Get audit trail
    pub fn get_audit_trail(&self) -> Arc<AuditTrail> {
        self.audit_trail.clone()
    }

    /// Get surveillance engine
    pub fn get_surveillance_engine(&self) -> Arc<SurveillanceEngine> {
        self.surveillance_engine.clone()
    }

    /// Add custom compliance rule
    pub fn add_rule(&self, rule: Box<dyn crate::rules::ComplianceRule>) {
        self.rule_engine.add_rule(rule);
    }

    /// Get rule engine for testing/benchmarking
    pub fn get_rule_engine(&self) -> Arc<RuleEngine> {
        self.rule_engine.clone()
    }
}

/// Result of compliance processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceDecision {
    Approved {
        order_id: Uuid,
        timestamp: DateTime<Utc>,
        rule_results: Vec<crate::rules::RuleResult>,
        risk_score: f64,
    },
    Rejected {
        order_id: Uuid,
        reason: String,
        violations: Vec<crate::rules::RuleResult>,
        timestamp: DateTime<Utc>,
    },
}