//! Main PanarchyAdaptiveDecisionSystem implementation
//!
//! This is the core implementation that directly corresponds to the Python PADS class,
//! providing all the same functionality with enhanced performance and type safety.

use crate::types::*;
use crate::error::{PadsError, PadsResult};
use crate::config::PadsConfig;
use crate::agents::AgentManager;
use crate::board::{BoardSystem, BoardVotingResult};
use crate::panarchy::{PanarchySystem, PanarchyState};
use crate::risk::{RiskManager, RiskAssessment};
use crate::strategies::{StrategyExecutor, DecisionStrategy};
use crate::analyzers::AnalyzerManager;
use crate::hardware::HardwareManager;
use crate::metrics::MetricsCollector;
use crate::core::{DecisionHistory, DecisionFusion, DecisionOverrides, FeedbackProcessor, SystemRecovery};

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use tokio::sync::mpsc;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error, debug, instrument};
use uuid::Uuid;

/// Main Panarchy Adaptive Decision System
/// 
/// This is the central orchestrator that manages all PADS components and provides
/// the primary interface for making trading decisions. It directly corresponds to
/// the Python `PanarchyAdaptiveDecisionSystem` class but with enhanced performance.
#[derive(Debug)]
pub struct PanarchyAdaptiveDecisionSystem {
    /// System configuration
    config: PadsConfig,
    
    /// Unique system identifier
    system_id: Uuid,
    
    /// System name
    name: String,
    
    /// Current panarchy state
    panarchy_state: Arc<RwLock<PanarchyState>>,
    
    /// Agent manager for all quantum agents
    agent_manager: Arc<AgentManager>,
    
    /// Board system for voting and consensus
    board_system: Arc<BoardSystem>,
    
    /// Panarchy system for adaptive cycles
    panarchy_system: Arc<PanarchySystem>,
    
    /// Risk management system
    risk_manager: Arc<RiskManager>,
    
    /// Strategy executor
    strategy_executor: Arc<StrategyExecutor>,
    
    /// Analyzer manager
    analyzer_manager: Arc<AnalyzerManager>,
    
    /// Hardware manager for acceleration
    hardware_manager: Arc<HardwareManager>,
    
    /// Decision history tracker
    decision_history: Arc<DecisionHistory>,
    
    /// Decision fusion engine
    decision_fusion: Arc<DecisionFusion>,
    
    /// Decision override handler
    decision_overrides: Arc<DecisionOverrides>,
    
    /// Feedback processor for learning
    feedback_processor: Arc<FeedbackProcessor>,
    
    /// System recovery handler
    system_recovery: Arc<SystemRecovery>,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    
    /// Current board state
    board_state: Arc<RwLock<BoardState>>,
    
    /// Performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Phase parameters for adaptive behavior
    phase_parameters: Arc<RwLock<HashMap<String, HashMap<String, serde_json::Value>>>>,
    
    /// Decision styles available
    decision_styles: Vec<String>,
    
    /// Current decision style
    current_decision_style: Arc<RwLock<String>>,
    
    /// Confidence thresholds by regime
    confidence_thresholds: HashMap<String, f64>,
    
    /// System initialization timestamp
    initialized_at: DateTime<Utc>,
}

impl PanarchyAdaptiveDecisionSystem {
    /// Create a new PADS instance with the given configuration
    #[instrument(skip(config))]
    pub async fn new(config: PadsConfig) -> PadsResult<Self> {
        info!("Initializing PanarchyAdaptiveDecisionSystem");
        
        let system_id = Uuid::new_v4();
        let name = config.system_name.clone();
        
        // Initialize hardware manager first
        let hardware_manager = Arc::new(
            HardwareManager::new(config.hardware_config.clone()).await
                .map_err(|e| PadsError::hardware(format!("Failed to initialize hardware manager: {}", e)))?
        );
        
        // Initialize metrics collector
        let metrics = Arc::new(MetricsCollector::new());
        
        // Initialize core systems
        let panarchy_state = Arc::new(RwLock::new(PanarchyState::default()));
        let board_state = Arc::new(RwLock::new(BoardState::default()));
        let performance_metrics = Arc::new(RwLock::new(PerformanceMetrics::default()));
        
        // Initialize agent manager with all quantum agents
        let agent_manager = Arc::new(
            AgentManager::new(config.agent_config.clone(), Arc::clone(&hardware_manager)).await
                .map_err(|e| PadsError::agent("agent_manager", format!("Failed to initialize: {}", e)))?
        );
        
        // Initialize board system
        let board_system = Arc::new(
            BoardSystem::new(config.board_config.clone()).await
                .map_err(|e| PadsError::board_consensus(format!("Failed to initialize board system: {}", e)))?
        );
        
        // Initialize panarchy system
        let panarchy_system = Arc::new(
            PanarchySystem::new(config.panarchy_config.clone()).await
                .map_err(|e| PadsError::panarchy(format!("Failed to initialize panarchy system: {}", e)))?
        );
        
        // Initialize risk manager
        let risk_manager = Arc::new(
            RiskManager::new(config.risk_config.clone()).await
                .map_err(|e| PadsError::risk_management(format!("Failed to initialize risk manager: {}", e)))?
        );
        
        // Initialize strategy executor
        let strategy_executor = Arc::new(
            StrategyExecutor::new(config.strategy_config.clone()).await
                .map_err(|e| PadsError::decision_strategy("executor", format!("Failed to initialize: {}", e)))?
        );
        
        // Initialize analyzer manager
        let analyzer_manager = Arc::new(
            AnalyzerManager::new(config.analyzer_config.clone()).await
                .map_err(|e| PadsError::analyzer("manager", format!("Failed to initialize: {}", e)))?
        );
        
        // Initialize decision components
        let decision_history = Arc::new(DecisionHistory::new(config.memory_config.history_size));
        let decision_fusion = Arc::new(DecisionFusion::new());
        let decision_overrides = Arc::new(DecisionOverrides::new());
        let feedback_processor = Arc::new(FeedbackProcessor::new());
        let system_recovery = Arc::new(SystemRecovery::new());
        
        // Load phase parameters
        let phase_parameters = Arc::new(RwLock::new(
            Self::load_phase_parameters(&config)
                .map_err(|e| PadsError::configuration(format!("Failed to load phase parameters: {}", e)))?
        ));
        
        // Set up decision styles
        let decision_styles = vec![
            "consensus".to_string(),
            "opportunistic".to_string(),
            "defensive".to_string(),
            "calculated_risk".to_string(),
            "contrarian".to_string(),
            "momentum_following".to_string(),
        ];
        
        let current_decision_style = Arc::new(RwLock::new("consensus".to_string()));
        
        // Set up confidence thresholds by regime
        let confidence_thresholds = [
            ("conservation".to_string(), 0.65),
            ("growth".to_string(), 0.55),
            ("release".to_string(), 0.75),
            ("reorganization".to_string(), 0.6),
        ].into_iter().collect();
        
        let pads = Self {
            config,
            system_id,
            name,
            panarchy_state,
            agent_manager,
            board_system,
            panarchy_system,
            risk_manager,
            strategy_executor,
            analyzer_manager,
            hardware_manager,
            decision_history,
            decision_fusion,
            decision_overrides,
            feedback_processor,
            system_recovery,
            metrics,
            board_state,
            performance_metrics,
            phase_parameters,
            decision_styles,
            current_decision_style,
            confidence_thresholds,
            initialized_at: Utc::now(),
        };
        
        info!("PanarchyAdaptiveDecisionSystem '{}' initialized successfully", pads.name);
        Ok(pads)
    }
    
    /// Main decision-making method (corresponds to Python make_decision)
    #[instrument(skip(self, market_state, factor_values, position_state))]
    pub async fn make_decision(
        &mut self,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
    ) -> PadsResult<TradingDecision> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting decision making process for pair: {}", market_state.pair);
        
        // Update panarchy state based on market conditions
        self.update_panarchy_state(market_state).await?;
        
        // Get current phase for phase-specific configuration
        let current_phase = {
            let state = self.panarchy_state.read();
            state.phase.to_string()
        };
        
        // Configure agents for current phase
        self.configure_agents_for_phase(&current_phase).await?;
        
        // Check for decision overrides first (emergency conditions)
        if let Some(override_decision) = self.check_for_decision_overrides(
            market_state, 
            factor_values, 
            position_state
        ).await? {
            let latency = start_time.elapsed();
            self.update_performance_metrics(&override_decision, true, latency.as_nanos() as u64).await;
            self.decision_history.add(override_decision.clone()).await;
            
            warn!("Decision override applied: {:?}", override_decision.decision_type);
            return Ok(override_decision);
        }
        
        // Run the boardroom decision process
        let decision = self.run_boardroom_decision(
            market_state,
            factor_values,
            position_state,
            &current_phase,
        ).await?;
        
        // Apply risk management filters
        let filtered_decision = self.apply_risk_management_filters(
            decision,
            market_state,
            factor_values,
            position_state,
        ).await?;
        
        // Adjust decision for current regime
        let final_decision = self.adjust_decision_for_regime(
            filtered_decision,
            market_state,
        ).await?;
        
        // Update metrics and history
        let latency = start_time.elapsed();
        self.update_performance_metrics(&final_decision, true, latency.as_nanos() as u64).await;
        self.decision_history.add(final_decision.clone()).await;
        
        // Update market regime
        self.update_market_regime().await?;
        
        debug!(
            "Decision completed: {:?} with confidence {:.3} in {:?}",
            final_decision.decision_type,
            final_decision.confidence,
            latency
        );
        
        Ok(final_decision)
    }
    
    /// Update panarchy state based on market conditions
    #[instrument(skip(self, market_state))]
    async fn update_panarchy_state(&self, market_state: &MarketState) -> PadsResult<()> {
        self.panarchy_system.update_state(market_state).await
            .map_err(|e| PadsError::panarchy(format!("Failed to update state: {}", e)))?;
        
        // Update our local panarchy state
        let new_state = self.panarchy_system.get_current_state().await?;
        {
            let mut state = self.panarchy_state.write();
            *state = new_state;
        }
        
        Ok(())
    }
    
    /// Configure agents for the current phase
    #[instrument(skip(self))]
    async fn configure_agents_for_phase(&self, phase: &str) -> PadsResult<()> {
        let phase_config = {
            let params = self.phase_parameters.read();
            params.get(phase).cloned()
        };
        
        if let Some(config) = phase_config {
            self.agent_manager.configure_for_phase(phase, &config).await
                .map_err(|e| PadsError::agent("configuration", format!("Failed to configure for phase {}: {}", phase, e)))?;
        } else {
            warn!("No phase configuration found for phase: {}", phase);
        }
        
        Ok(())
    }
    
    /// Check for decision overrides (emergency conditions)
    #[instrument(skip(self, market_state, factor_values, position_state))]
    async fn check_for_decision_overrides(
        &self,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
    ) -> PadsResult<Option<TradingDecision>> {
        self.decision_overrides.check_overrides(market_state, factor_values, position_state).await
            .map_err(|e| PadsError::internal(format!("Override check failed: {}", e)))
    }
    
    /// Run the boardroom decision process
    #[instrument(skip(self, market_state, factor_values, position_state))]
    async fn run_boardroom_decision(
        &self,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
        current_phase: &str,
    ) -> PadsResult<TradingDecision> {
        // Select decision style based on market conditions
        self.select_decision_style(market_state, factor_values).await?;
        
        // Collect votes from all components
        let component_votes = self.collect_component_votes(
            market_state,
            factor_values,
            position_state,
        ).await?;
        
        // Run board voting process
        let board_result = self.board_system.process_votes(&component_votes).await
            .map_err(|e| PadsError::board_consensus(format!("Board voting failed: {}", e)))?;
        
        // Execute decision strategy
        let decision = self.execute_decision_strategy(
            &board_result,
            market_state,
            factor_values,
            position_state,
            current_phase,
        ).await?;
        
        Ok(decision)
    }
    
    /// Select decision style based on market conditions
    #[instrument(skip(self, market_state, factor_values))]
    async fn select_decision_style(
        &self,
        market_state: &MarketState,
        factor_values: &FactorValues,
    ) -> PadsResult<()> {
        // Extract key indicators
        let volatility = market_state.volatility;
        let trend_strength = market_state.trend.abs();
        let black_swan_risk = factor_values.get_or_default("black_swan_risk", 0.1);
        let whale_activity = factor_values.get_or_default("whale_activity", 0.0);
        let anomaly_score = factor_values.get_or_default("anomaly_score", 0.0);
        let antifragility = factor_values.get_or_default("antifragility", 0.5);
        
        // Determine appropriate style
        let selected_style = if black_swan_risk > 0.5 {
            "defensive"
        } else if volatility > 0.7 {
            "defensive"
        } else if whale_activity > 0.7 {
            "opportunistic"
        } else if anomaly_score > 0.7 {
            "defensive"
        } else if trend_strength > 0.7 && volatility < 0.6 {
            "momentum_following"
        } else if antifragility > 0.7 {
            "calculated_risk"
        } else {
            match market_state.phase {
                MarketPhase::Growth => "calculated_risk",
                MarketPhase::Conservation => "consensus",
                MarketPhase::Release => "defensive",
                MarketPhase::Reorganization => "contrarian",
                MarketPhase::Unknown => "consensus",
            }
        };
        
        // Update current style and board state
        {
            let mut style = self.current_decision_style.write();
            *style = selected_style.to_string();
        }
        
        {
            let mut board_state = self.board_state.write();
            board_state.current_strategy = selected_style.to_string();
            
            // Update risk appetite based on style
            board_state.risk_appetite = match selected_style {
                "defensive" => 0.3,
                "opportunistic" => 0.8,
                "calculated_risk" => 0.7,
                "contrarian" => 0.6,
                "momentum_following" => 0.7,
                _ => 0.5, // consensus
            };
        }
        
        debug!("Selected decision style: {} (volatility: {:.3}, trend: {:.3})", 
               selected_style, volatility, trend_strength);
        
        Ok(())
    }
    
    /// Collect votes from all components
    #[instrument(skip(self, market_state, factor_values, position_state))]
    async fn collect_component_votes(
        &self,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
    ) -> PadsResult<HashMap<String, ComponentVote>> {
        // Collect votes from agents
        let agent_votes = self.agent_manager.collect_votes(
            market_state,
            factor_values,
            position_state,
        ).await.map_err(|e| PadsError::agent("voting", format!("Failed to collect agent votes: {}", e)))?;
        
        // Collect votes from analyzers
        let analyzer_votes = self.analyzer_manager.collect_votes(
            market_state,
            factor_values,
            position_state,
        ).await.map_err(|e| PadsError::analyzer("voting", format!("Failed to collect analyzer votes: {}", e)))?;
        
        // Merge all votes
        let mut all_votes = agent_votes;
        all_votes.extend(analyzer_votes);
        
        Ok(all_votes)
    }
    
    /// Execute the selected decision strategy
    #[instrument(skip(self, board_result, market_state, factor_values, position_state))]
    async fn execute_decision_strategy(
        &self,
        board_result: &BoardVotingResult,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
        current_phase: &str,
    ) -> PadsResult<TradingDecision> {
        let strategy_name = {
            let style = self.current_decision_style.read();
            style.clone()
        };
        
        let strategy = DecisionStrategy::from_str(&strategy_name)
            .map_err(|e| PadsError::decision_strategy(&strategy_name, format!("Invalid strategy: {}", e)))?;
        
        let confidence_threshold = self.confidence_thresholds
            .get(current_phase)
            .copied()
            .unwrap_or(0.6);
        
        self.strategy_executor.execute(
            strategy,
            board_result,
            market_state,
            factor_values,
            position_state,
            confidence_threshold,
        ).await.map_err(|e| PadsError::decision_strategy(&strategy_name, format!("Execution failed: {}", e)))
    }
    
    /// Apply risk management filters
    #[instrument(skip(self, decision, market_state, factor_values, position_state))]
    async fn apply_risk_management_filters(
        &self,
        decision: TradingDecision,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
    ) -> PadsResult<TradingDecision> {
        self.risk_manager.apply_filters(
            decision,
            market_state,
            factor_values,
            position_state,
        ).await.map_err(|e| PadsError::risk_management(format!("Filter application failed: {}", e)))
    }
    
    /// Adjust decision for current regime
    #[instrument(skip(self, decision, market_state))]
    async fn adjust_decision_for_regime(
        &self,
        decision: TradingDecision,
        market_state: &MarketState,
    ) -> PadsResult<TradingDecision> {
        self.panarchy_system.adjust_decision_for_regime(decision, market_state).await
            .map_err(|e| PadsError::panarchy(format!("Regime adjustment failed: {}", e)))
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(
        &self,
        decision: &TradingDecision,
        success: bool,
        latency_ns: u64,
    ) {
        let mut metrics = self.performance_metrics.write();
        metrics.update_decision(decision, success, latency_ns);
        
        // Update metrics collector
        self.metrics.record_decision(decision, success, latency_ns).await;
    }
    
    /// Update market regime
    async fn update_market_regime(&self) -> PadsResult<()> {
        self.panarchy_system.update_regime().await
            .map_err(|e| PadsError::panarchy(format!("Regime update failed: {}", e)))
    }
    
    /// Load phase parameters from configuration
    fn load_phase_parameters(config: &PadsConfig) -> PadsResult<HashMap<String, HashMap<String, serde_json::Value>>> {
        // Load from config or use defaults
        if let Some(phase_params) = &config.phase_parameters {
            Ok(phase_params.clone())
        } else {
            // Create default phase parameters
            let default_phases = ["growth", "conservation", "release", "reorganization"];
            let mut params = HashMap::new();
            
            for phase in default_phases {
                let mut phase_config = HashMap::new();
                phase_config.insert("threshold".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.6).unwrap()));
                phase_config.insert("weights".to_string(), serde_json::Value::Object(serde_json::Map::new()));
                params.insert(phase.to_string(), phase_config);
            }
            
            Ok(params)
        }
    }
    
    /// Get risk advice (corresponds to Python get_risk_advice)
    pub fn get_risk_advice(
        &self,
        market_state: &MarketState,
        factor_values: &FactorValues,
        position_state: Option<&PositionState>,
    ) -> PadsResult<RiskAdvice> {
        let current_phase = {
            let state = self.panarchy_state.read();
            state.phase.to_string()
        };
        
        let current_profit = position_state
            .map(|p| p.current_profit)
            .unwrap_or(0.0);
        
        let mut advice = RiskAdvice::new();
        
        // Phase-based adjustments
        match current_phase.as_str() {
            "conservation" => {
                advice.stoploss_adjustment = 0.9;
                advice.position_sizing = 0.8;
                advice.add_reason("Conservation phase requires caution".to_string());
            }
            "release" => {
                advice.stoploss_adjustment = 0.7;
                advice.position_sizing = 0.7;
                advice.add_reason("Release phase requires caution".to_string());
            }
            "reorganization" => {
                advice.stoploss_adjustment = 0.8;
                advice.position_sizing = 0.9;
                advice.add_reason("Reorganization phase suggests moderate caution".to_string());
            }
            "growth" => {
                advice.stoploss_adjustment = 1.1;
                advice.position_sizing = 1.2;
                advice.add_reason("Growth phase allows more aggressive positioning".to_string());
            }
            _ => {}
        }
        
        // Risk factor adjustments
        let black_swan_risk = factor_values.get_or_default("black_swan", 0.1);
        if black_swan_risk > 0.5 {
            let adjustment = (1.0 - black_swan_risk).max(0.5);
            advice.stoploss_adjustment *= adjustment;
            advice.position_sizing *= adjustment;
            advice.add_reason(format!("Black swan risk ({:.2}) requires caution", black_swan_risk));
        }
        
        // Volatility adjustment
        if market_state.volatility > 0.7 {
            let adjustment = (1.0 - (market_state.volatility - 0.5)).max(0.6);
            advice.stoploss_adjustment *= adjustment;
            advice.add_reason(format!("High volatility ({:.2}) requires tighter risk control", market_state.volatility));
        }
        
        // Profitability adjustments
        if current_profit > 0.05 && current_profit > 0.1 {
            advice.take_profit = Some(current_profit * 0.7);
            advice.add_reason(format!("Protecting {:.2}% profit", current_profit * 100.0));
        }
        
        // Antifragility insights
        let antifragility = factor_values.get_or_default("antifragility", 0.5);
        if antifragility > 0.7 {
            advice.stoploss_adjustment *= (1.0 + (antifragility - 0.5)).min(1.3);
            advice.add_reason(format!("High antifragility ({:.2}) allows more flexible risk management", antifragility));
        }
        
        // System fragility
        let soc_fragility = factor_values.get_or_default("soc_fragility", 0.5);
        if soc_fragility > 0.7 {
            advice.apply_conservative(0.8);
            advice.add_reason(format!("High system fragility ({:.2}) requires defensive positioning", soc_fragility));
        }
        
        // Ensure adjustments are within reasonable limits
        advice.stoploss_adjustment = advice.stoploss_adjustment.clamp(0.5, 1.5);
        advice.position_sizing = advice.position_sizing.clamp(0.3, 2.0);
        
        Ok(advice)
    }
    
    /// Provide feedback for learning (corresponds to Python provide_feedback)
    pub async fn provide_feedback(
        &mut self,
        decision: &TradingDecision,
        outcome: bool,
        metrics: Option<&HashMap<String, f64>>,
    ) -> PadsResult<()> {
        self.feedback_processor.process_feedback(decision, outcome, metrics).await
            .map_err(|e| PadsError::internal(format!("Feedback processing failed: {}", e)))?;
        
        // Update agent learning
        self.agent_manager.provide_feedback(decision, outcome, metrics).await
            .map_err(|e| PadsError::agent("feedback", format!("Agent feedback failed: {}", e)))?;
        
        Ok(())
    }
    
    /// Get current panarchy state
    pub fn get_panarchy_state(&self) -> PanarchyState {
        self.panarchy_state.read().clone()
    }
    
    /// Get latest decision
    pub async fn get_latest_decision(&self) -> Option<TradingDecision> {
        self.decision_history.get_latest().await
    }
    
    /// Get decision history
    pub async fn get_decision_history(&self) -> Vec<TradingDecision> {
        self.decision_history.get_all().await
    }
    
    /// Create system summary
    pub fn create_system_summary(&self) -> SystemSummary {
        let latest_decision = futures::executor::block_on(self.get_latest_decision());
        let panarchy_state = self.get_panarchy_state();
        let performance_metrics = self.performance_metrics.read().clone();
        
        SystemSummary {
            timestamp: Utc::now(),
            latest_decision,
            regime_info: [
                ("current_regime".to_string(), serde_json::Value::String(panarchy_state.regime.clone())),
                ("current_phase".to_string(), serde_json::Value::String(panarchy_state.phase.to_string())),
                ("soc_index".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(panarchy_state.soc_index).unwrap())),
                ("black_swan_risk".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(panarchy_state.black_swan_risk).unwrap())),
            ].into_iter().collect(),
            performance_metrics,
            board_members: HashMap::new(), // TODO: Implement board member retrieval
        }
    }
    
    /// Recover system from errors
    pub async fn recover(&mut self) -> PadsResult<()> {
        warn!("System recovery initiated for PADS '{}'", self.name);
        
        self.system_recovery.recover_system(
            &self.agent_manager,
            &self.board_system,
            &self.panarchy_system,
            &self.risk_manager,
        ).await.map_err(|e| PadsError::internal(format!("System recovery failed: {}", e)))?;
        
        // Reset internal state
        {
            let mut state = self.panarchy_state.write();
            *state = PanarchyState::default();
        }
        
        {
            let mut board_state = self.board_state.write();
            *board_state = BoardState::default();
        }
        
        info!("System recovery completed for PADS '{}'", self.name);
        Ok(())
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().clone()
    }
    
    /// Get system information
    pub fn get_system_info(&self) -> HashMap<String, serde_json::Value> {
        [
            ("system_id".to_string(), serde_json::Value::String(self.system_id.to_string())),
            ("name".to_string(), serde_json::Value::String(self.name.clone())),
            ("initialized_at".to_string(), serde_json::Value::String(self.initialized_at.to_rfc3339())),
            ("decision_styles".to_string(), serde_json::Value::Array(
                self.decision_styles.iter().map(|s| serde_json::Value::String(s.clone())).collect()
            )),
            ("current_decision_style".to_string(), serde_json::Value::String(self.current_decision_style.read().clone())),
        ].into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PadsConfig;
    
    #[tokio::test]
    async fn test_pads_initialization() {
        let config = PadsConfig::test_config();
        let pads = PanarchyAdaptiveDecisionSystem::new(config).await;
        assert!(pads.is_ok());
    }
    
    #[tokio::test]
    async fn test_basic_decision_making() {
        let config = PadsConfig::test_config();
        let mut pads = PanarchyAdaptiveDecisionSystem::new(config).await.unwrap();
        
        let market_state = MarketState::new(100.0, 1000.0, 0.2);
        let factor_values = FactorValues::new();
        
        let decision = pads.make_decision(&market_state, &factor_values, None).await;
        assert!(decision.is_ok());
        
        let decision = decision.unwrap();
        assert!(decision.confidence > 0.0);
        assert!(decision.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_risk_advice() {
        let config = PadsConfig::test_config();
        let pads = PanarchyAdaptiveDecisionSystem::new(config).await.unwrap();
        
        let market_state = MarketState::new(100.0, 1000.0, 0.5);
        let mut factor_values = FactorValues::new();
        factor_values.set("black_swan".to_string(), 0.8);
        
        let advice = pads.get_risk_advice(&market_state, &factor_values, None);
        assert!(advice.is_ok());
        
        let advice = advice.unwrap();
        assert!(advice.stoploss_adjustment < 1.0); // Should be conservative
        assert!(advice.position_sizing < 1.0);
        assert!(!advice.reasons.is_empty());
    }
    
    #[tokio::test]
    async fn test_system_summary() {
        let config = PadsConfig::test_config();
        let pads = PanarchyAdaptiveDecisionSystem::new(config).await.unwrap();
        
        let summary = pads.create_system_summary();
        assert!(!summary.regime_info.is_empty());
        assert_eq!(summary.performance_metrics.total_decisions, 0);
    }
    
    #[tokio::test]
    async fn test_feedback_processing() {
        let config = PadsConfig::test_config();
        let mut pads = PanarchyAdaptiveDecisionSystem::new(config).await.unwrap();
        
        let decision = TradingDecision::new(
            DecisionType::Buy,
            0.8,
            "Test decision".to_string(),
        );
        
        let result = pads.provide_feedback(&decision, true, None).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_system_recovery() {
        let config = PadsConfig::test_config();
        let mut pads = PanarchyAdaptiveDecisionSystem::new(config).await.unwrap();
        
        let result = pads.recover().await;
        assert!(result.is_ok());
    }
}