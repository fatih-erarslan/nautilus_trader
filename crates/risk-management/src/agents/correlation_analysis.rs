//! # Correlation Analysis Agent
//!
//! Quantum correlation detection and regime change identification.
//! This agent implements quantum-enhanced correlation analysis with
//! real-time regime detection and sub-100μs correlation calculations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::types::*;
use crate::quantum::*;
use crate::correlation::*;
use super::base::*;
use super::coordination::*;

/// Correlation Analysis Agent using quantum correlation detection
#[derive(Debug)]
pub struct CorrelationAnalysisAgent {
    /// Agent metadata
    pub agent_id: Uuid,
    pub agent_type: AgentType,
    pub status: AgentStatus,
    
    /// Configuration
    pub config: CorrelationAgentConfig,
    
    /// Quantum correlation detector
    pub quantum_correlation_detector: Arc<RwLock<QuantumCorrelationDetector>>,
    
    /// Regime change detector
    pub regime_detector: Arc<RwLock<RegimeChangeDetector>>,
    
    /// Copula analyzer
    pub copula_analyzer: Arc<RwLock<CopulaAnalyzer>>,
    
    /// Dynamic correlation tracker
    pub dynamic_tracker: Arc<RwLock<DynamicCorrelationTracker>>,
    
    /// Performance metrics
    pub performance_metrics: Arc<RwLock<AgentPerformanceMetrics>>,
    
    /// Coordination components
    pub coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
    pub message_router: Arc<RwLock<SwarmMessageRouter>>,
    
    /// Message channels
    pub message_tx: mpsc::UnboundedSender<SwarmMessage>,
    pub message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<SwarmMessage>>>>,
    
    /// TENGRI integration
    pub tengri_client: Arc<RwLock<TengriOversightClient>>,
}

impl CorrelationAnalysisAgent {
    /// Create new correlation analysis agent
    pub async fn new(
        config: CorrelationAgentConfig,
        coordination_hub: Arc<RwLock<AgentCoordinationHub>>,
        message_router: Arc<RwLock<SwarmMessageRouter>>,
    ) -> Result<Self> {
        let agent_id = Uuid::new_v4();
        info!("Creating Correlation Analysis Agent {}", agent_id);
        
        // Create quantum correlation detector
        let quantum_correlation_detector = Arc::new(RwLock::new(
            QuantumCorrelationDetector::new(config.quantum_config.clone()).await?
        ));
        
        // Create regime change detector
        let regime_detector = Arc::new(RwLock::new(
            RegimeChangeDetector::new(config.regime_config.clone()).await?
        ));
        
        // Create copula analyzer
        let copula_analyzer = Arc::new(RwLock::new(
            CopulaAnalyzer::new(config.copula_config.clone()).await?
        ));
        
        // Create dynamic correlation tracker
        let dynamic_tracker = Arc::new(RwLock::new(
            DynamicCorrelationTracker::new(config.dynamic_config.clone()).await?
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            AgentPerformanceMetrics::new(agent_id, AgentType::CorrelationAnalysis)
        ));
        
        // Create message channel
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let message_rx = Arc::new(RwLock::new(Some(message_rx)));
        
        // Initialize TENGRI client
        let tengri_client = Arc::new(RwLock::new(
            TengriOversightClient::new(config.tengri_config.clone()).await?
        ));
        
        Ok(Self {
            agent_id,
            agent_type: AgentType::CorrelationAnalysis,
            status: AgentStatus::Initializing,
            config,
            quantum_correlation_detector,
            regime_detector,
            copula_analyzer,
            dynamic_tracker,
            performance_metrics,
            coordination_hub,
            message_router,
            message_tx,
            message_rx,
            tengri_client,
        })
    }
    
    /// Analyze quantum-enhanced correlations
    pub async fn analyze_quantum_correlations(
        &self,
        assets: &[Asset],
        market_data: &MarketData,
        analysis_config: &CorrelationAnalysisConfig,
    ) -> Result<QuantumCorrelationAnalysis> {
        let start_time = Instant::now();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.start_calculation();
        }
        
        // Extract price and return data
        let price_data = self.extract_price_data(assets, market_data).await?;
        let return_data = self.calculate_returns(&price_data).await?;
        
        // Calculate classical correlations
        let quantum_detector = self.quantum_correlation_detector.read().await;
        let classical_correlations = quantum_detector.calculate_classical_correlations(&return_data).await?;
        
        // Calculate quantum-enhanced correlations
        let quantum_correlations = quantum_detector.calculate_quantum_correlations(
            &return_data,
            analysis_config,
        ).await?;
        
        // Detect regime changes
        let regime_detector = self.regime_detector.read().await;
        let regime_analysis = regime_detector.detect_regime_changes(
            &return_data,
            &quantum_correlations,
        ).await?;
        
        // Analyze copula dependencies
        let copula_analyzer = self.copula_analyzer.read().await;
        let copula_analysis = copula_analyzer.analyze_copula_dependencies(&return_data).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.end_calculation(calculation_time);
        }
        
        // Check performance target
        if calculation_time > Duration::from_micros(100) {
            warn!(
                "Correlation analysis took {:?}, exceeding 100μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        let quantum_correlation_analysis = QuantumCorrelationAnalysis {
            assets: assets.to_vec(),
            classical_correlations,
            quantum_correlations,
            quantum_advantage: quantum_correlations.quantum_advantage,
            regime_analysis,
            copula_analysis,
            correlation_stability: self.assess_correlation_stability(&quantum_correlations).await?,
            tail_dependencies: self.calculate_tail_dependencies(&return_data).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        };
        
        // Report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_correlation_metrics(
                self.agent_id,
                "quantum_correlation_analysis",
                calculation_time,
                quantum_correlation_analysis.quantum_advantage,
            ).await?;
        }
        
        Ok(quantum_correlation_analysis)
    }
    
    /// Monitor real-time correlation changes
    pub async fn monitor_real_time_correlations(
        &self,
        assets: &[Asset],
        streaming_data: &StreamingMarketData,
    ) -> Result<RealTimeCorrelationUpdate> {
        let start_time = Instant::now();
        
        // Update dynamic correlation tracker
        let mut dynamic_tracker = self.dynamic_tracker.write().await;
        dynamic_tracker.update_with_new_data(streaming_data).await?;
        
        // Get current correlation estimates
        let current_correlations = dynamic_tracker.get_current_correlations().await?;
        
        // Detect significant changes
        let correlation_changes = dynamic_tracker.detect_correlation_changes().await?;
        
        // Check for regime changes
        let regime_detector = self.regime_detector.read().await;
        let regime_change_signal = regime_detector.check_real_time_regime_change(
            &current_correlations,
            streaming_data,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        // Check real-time performance target (sub-microsecond)
        if calculation_time > Duration::from_micros(1) {
            warn!(
                "Real-time correlation monitoring took {:?}, exceeding 1μs target for agent {}",
                calculation_time, self.agent_id
            );
        }
        
        let correlation_update = RealTimeCorrelationUpdate {
            assets: assets.to_vec(),
            current_correlations,
            correlation_changes,
            regime_change_signal,
            confidence_level: current_correlations.confidence_level,
            update_timestamp: chrono::Utc::now(),
            calculation_time,
        };
        
        // Send alerts for significant changes
        if correlation_changes.has_significant_changes() || regime_change_signal.is_some() {
            self.send_correlation_change_alert(&correlation_update).await?;
        }
        
        Ok(correlation_update)
    }
    
    /// Detect correlation regime changes
    pub async fn detect_regime_changes(
        &self,
        assets: &[Asset],
        historical_data: &HistoricalMarketData,
        detection_config: &RegimeDetectionConfig,
    ) -> Result<RegimeChangeAnalysis> {
        let start_time = Instant::now();
        
        // Prepare data for regime detection
        let price_data = self.extract_historical_price_data(assets, historical_data).await?;
        let return_data = self.calculate_returns(&price_data).await?;
        
        // Run regime detection algorithm
        let regime_detector = self.regime_detector.read().await;
        let regime_changes = regime_detector.detect_regime_changes_comprehensive(
            &return_data,
            detection_config,
        ).await?;
        
        // Analyze quantum correlation patterns in each regime
        let quantum_detector = self.quantum_correlation_detector.read().await;
        let regime_correlations = quantum_detector.analyze_regime_correlations(
            &return_data,
            &regime_changes,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(RegimeChangeAnalysis {
            assets: assets.to_vec(),
            detected_regimes: regime_changes,
            regime_correlations,
            regime_stability: self.assess_regime_stability(&regime_changes).await?,
            transition_dynamics: self.analyze_transition_dynamics(&regime_changes).await?,
            prediction_accuracy: regime_detector.get_prediction_accuracy().await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Analyze copula dependencies
    pub async fn analyze_copula_dependencies(
        &self,
        assets: &[Asset],
        market_data: &MarketData,
        copula_config: &CopulaConfig,
    ) -> Result<CopulaDependencyAnalysis> {
        let start_time = Instant::now();
        
        // Extract return data
        let price_data = self.extract_price_data(assets, market_data).await?;
        let return_data = self.calculate_returns(&price_data).await?;
        
        // Analyze different copula models
        let copula_analyzer = self.copula_analyzer.read().await;
        let gaussian_copula = copula_analyzer.fit_gaussian_copula(&return_data).await?;
        let t_copula = copula_analyzer.fit_t_copula(&return_data).await?;
        let archimedean_copulas = copula_analyzer.fit_archimedean_copulas(&return_data).await?;
        
        // Select best-fitting copula
        let best_copula = copula_analyzer.select_best_copula(
            &gaussian_copula,
            &t_copula,
            &archimedean_copulas,
        ).await?;
        
        // Calculate tail dependencies
        let tail_dependencies = copula_analyzer.calculate_tail_dependencies(&best_copula).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(CopulaDependencyAnalysis {
            assets: assets.to_vec(),
            gaussian_copula,
            t_copula,
            archimedean_copulas,
            best_copula,
            tail_dependencies,
            dependency_strength: self.calculate_dependency_strength(&best_copula).await?,
            extreme_dependence: self.assess_extreme_dependence(&tail_dependencies).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate dynamic conditional correlations
    pub async fn calculate_dynamic_correlations(
        &self,
        assets: &[Asset],
        market_data: &MarketData,
        dcc_config: &DccConfig,
    ) -> Result<DynamicCorrelationResults> {
        let start_time = Instant::now();
        
        // Extract return data
        let price_data = self.extract_price_data(assets, market_data).await?;
        let return_data = self.calculate_returns(&price_data).await?;
        
        // Fit DCC-GARCH model with quantum enhancement
        let quantum_detector = self.quantum_correlation_detector.read().await;
        let dcc_model = quantum_detector.fit_quantum_dcc_model(&return_data, dcc_config).await?;
        
        // Extract time-varying correlations
        let dynamic_correlations = quantum_detector.extract_dynamic_correlations(&dcc_model).await?;
        
        // Forecast future correlations
        let correlation_forecasts = quantum_detector.forecast_correlations(
            &dcc_model,
            dcc_config.forecast_horizon,
        ).await?;
        
        let calculation_time = start_time.elapsed();
        
        Ok(DynamicCorrelationResults {
            assets: assets.to_vec(),
            dcc_model,
            dynamic_correlations,
            correlation_forecasts,
            model_diagnostics: self.calculate_model_diagnostics(&dcc_model).await?,
            forecast_accuracy: self.assess_forecast_accuracy(&correlation_forecasts).await?,
            calculation_time,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Extract price data from market data
    async fn extract_price_data(&self, assets: &[Asset], market_data: &MarketData) -> Result<Array2<f64>> {
        let n_assets = assets.len();
        let n_observations = market_data.prices.len();
        
        let mut price_matrix = Array2::zeros((n_observations, n_assets));
        
        for (i, asset) in assets.iter().enumerate() {
            if let Some(asset_prices) = market_data.asset_prices.get(&asset.symbol) {
                for (j, price) in asset_prices.iter().enumerate() {
                    if j < n_observations {
                        price_matrix[[j, i]] = *price;
                    }
                }
            }
        }
        
        Ok(price_matrix)
    }
    
    /// Extract historical price data
    async fn extract_historical_price_data(
        &self,
        assets: &[Asset],
        historical_data: &HistoricalMarketData,
    ) -> Result<Array2<f64>> {
        let n_assets = assets.len();
        let n_observations = historical_data.time_series.len();
        
        let mut price_matrix = Array2::zeros((n_observations, n_assets));
        
        for (i, asset) in assets.iter().enumerate() {
            if let Some(asset_data) = historical_data.asset_data.get(&asset.symbol) {
                for (j, data_point) in asset_data.iter().enumerate() {
                    if j < n_observations {
                        price_matrix[[j, i]] = data_point.close_price;
                    }
                }
            }
        }
        
        Ok(price_matrix)
    }
    
    /// Calculate returns from price data
    async fn calculate_returns(&self, price_data: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_obs, n_assets) = price_data.dim();
        if n_obs < 2 {
            return Err(anyhow!("Insufficient data for return calculation"));
        }
        
        let mut return_data = Array2::zeros((n_obs - 1, n_assets));
        
        for i in 1..n_obs {
            for j in 0..n_assets {
                if price_data[[i - 1, j]] > 0.0 {
                    return_data[[i - 1, j]] = (price_data[[i, j]] / price_data[[i - 1, j]]).ln();
                }
            }
        }
        
        Ok(return_data)
    }
    
    /// Assess correlation stability
    async fn assess_correlation_stability(
        &self,
        quantum_correlations: &QuantumCorrelationMatrix,
    ) -> Result<CorrelationStability> {
        // Calculate stability metrics
        let stability_score = quantum_correlations.stability_measures.average_stability;
        let volatility_score = quantum_correlations.stability_measures.correlation_volatility;
        
        let stability_level = if stability_score > 0.8 {
            StabilityLevel::High
        } else if stability_score > 0.5 {
            StabilityLevel::Medium
        } else {
            StabilityLevel::Low
        };
        
        Ok(CorrelationStability {
            stability_level,
            stability_score,
            volatility_score,
            persistence: quantum_correlations.stability_measures.persistence,
            mean_reversion_speed: quantum_correlations.stability_measures.mean_reversion_speed,
        })
    }
    
    /// Calculate tail dependencies
    async fn calculate_tail_dependencies(&self, return_data: &Array2<f64>) -> Result<TailDependencies> {
        let copula_analyzer = self.copula_analyzer.read().await;
        copula_analyzer.calculate_comprehensive_tail_dependencies(return_data).await
    }
    
    /// Assess regime stability
    async fn assess_regime_stability(&self, regime_changes: &[RegimeChange]) -> Result<RegimeStability> {
        let average_regime_duration = regime_changes.iter()
            .map(|change| change.duration.as_secs_f64())
            .sum::<f64>() / regime_changes.len() as f64;
        
        let regime_count = regime_changes.len();
        let stability_score = if average_regime_duration > 30.0 * 24.0 * 3600.0 { // 30 days
            1.0 - (regime_count as f64 / 10.0).min(1.0)
        } else {
            0.5 - (regime_count as f64 / 20.0).min(0.5)
        };
        
        Ok(RegimeStability {
            stability_score,
            average_regime_duration: Duration::from_secs_f64(average_regime_duration),
            regime_count,
            transition_frequency: regime_count as f64 / 365.0, // per year
        })
    }
    
    /// Analyze transition dynamics
    async fn analyze_transition_dynamics(&self, regime_changes: &[RegimeChange]) -> Result<TransitionDynamics> {
        let transition_speeds: Vec<f64> = regime_changes.iter()
            .map(|change| 1.0 / change.transition_period.as_secs_f64())
            .collect();
        
        let average_transition_speed = transition_speeds.iter().sum::<f64>() / transition_speeds.len() as f64;
        
        Ok(TransitionDynamics {
            average_transition_speed,
            transition_volatility: self.calculate_volatility(&transition_speeds).await?,
            transition_patterns: self.identify_transition_patterns(regime_changes).await?,
        })
    }
    
    /// Calculate dependency strength
    async fn calculate_dependency_strength(&self, copula: &CopulaModel) -> Result<f64> {
        // Simplified dependency strength calculation
        match copula {
            CopulaModel::Gaussian { correlation_matrix } => {
                let off_diagonal_sum: f64 = correlation_matrix.iter()
                    .enumerate()
                    .flat_map(|(i, row)| {
                        row.iter().enumerate().filter_map(|(j, &val)| {
                            if i != j { Some(val.abs()) } else { None }
                        })
                    })
                    .sum();
                
                let n = correlation_matrix.len();
                Ok(off_diagonal_sum / (n * (n - 1)) as f64)
            }
            _ => Ok(0.5) // Default for other copula types
        }
    }
    
    /// Assess extreme dependence
    async fn assess_extreme_dependence(&self, tail_dependencies: &TailDependencies) -> Result<ExtremeDependence> {
        let upper_tail_strength = tail_dependencies.upper_tail_dependence.iter().sum::<f64>()
            / tail_dependencies.upper_tail_dependence.len() as f64;
        
        let lower_tail_strength = tail_dependencies.lower_tail_dependence.iter().sum::<f64>()
            / tail_dependencies.lower_tail_dependence.len() as f64;
        
        Ok(ExtremeDependence {
            upper_tail_strength,
            lower_tail_strength,
            asymmetry: (upper_tail_strength - lower_tail_strength).abs(),
            extreme_event_probability: tail_dependencies.extreme_event_probability,
        })
    }
    
    /// Calculate model diagnostics
    async fn calculate_model_diagnostics(&self, dcc_model: &DccModel) -> Result<ModelDiagnostics> {
        Ok(ModelDiagnostics {
            log_likelihood: dcc_model.log_likelihood,
            aic: dcc_model.aic,
            bic: dcc_model.bic,
            convergence_status: dcc_model.convergence_status.clone(),
            parameter_significance: dcc_model.parameter_significance.clone(),
        })
    }
    
    /// Assess forecast accuracy
    async fn assess_forecast_accuracy(&self, forecasts: &CorrelationForecasts) -> Result<ForecastAccuracy> {
        Ok(ForecastAccuracy {
            mean_absolute_error: forecasts.forecast_accuracy.mean_absolute_error,
            root_mean_squared_error: forecasts.forecast_accuracy.root_mean_squared_error,
            directional_accuracy: forecasts.forecast_accuracy.directional_accuracy,
            confidence_interval_coverage: forecasts.forecast_accuracy.confidence_interval_coverage,
        })
    }
    
    /// Calculate volatility of a series
    async fn calculate_volatility(&self, data: &[f64]) -> Result<f64> {
        if data.len() < 2 {
            return Ok(0.0);
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Identify transition patterns
    async fn identify_transition_patterns(&self, regime_changes: &[RegimeChange]) -> Result<Vec<TransitionPattern>> {
        // Simplified pattern identification
        let mut patterns = Vec::new();
        
        for window in regime_changes.windows(3) {
            if let [prev, current, next] = window {
                let pattern = TransitionPattern {
                    from_regime: prev.to_regime.clone(),
                    through_regime: current.to_regime.clone(),
                    to_regime: next.to_regime.clone(),
                    frequency: 1,
                    average_duration: current.duration,
                };
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Send correlation change alert
    async fn send_correlation_change_alert(&self, update: &RealTimeCorrelationUpdate) -> Result<()> {
        let alert_message = SwarmMessage {
            id: Uuid::new_v4(),
            sender_id: self.agent_id,
            sender_type: self.agent_type.clone(),
            recipient_id: None, // Broadcast
            message_type: MessageType::CorrelationChangeAlert,
            content: MessageContent::CorrelationChangeAlert(update.clone()),
            timestamp: chrono::Utc::now(),
            priority: MessagePriority::High,
            requires_response: false,
        };
        
        self.message_tx.send(alert_message)?;
        
        // Also report to TENGRI oversight
        {
            let tengri_client = self.tengri_client.read().await;
            tengri_client.report_correlation_change_alert(self.agent_id, update).await?;
        }
        
        Ok(())
    }
    
    /// Handle incoming swarm messages
    async fn handle_message(&self, message: SwarmMessage) -> Result<()> {
        match message.message_type {
            MessageType::CorrelationAnalysisRequest => {
                self.handle_correlation_analysis_request(message).await?;
            }
            MessageType::RegimeDetectionRequest => {
                self.handle_regime_detection_request(message).await?;
            }
            MessageType::CopulaAnalysisRequest => {
                self.handle_copula_analysis_request(message).await?;
            }
            MessageType::DynamicCorrelationRequest => {
                self.handle_dynamic_correlation_request(message).await?;
            }
            MessageType::HealthCheck => {
                self.handle_health_check(message).await?;
            }
            _ => {
                debug!("Received unhandled message type: {:?}", message.message_type);
            }
        }
        Ok(())
    }
    
    async fn handle_correlation_analysis_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::CorrelationAnalysisRequest { assets, market_data, config } = message.content {
            let correlation_analysis = self.analyze_quantum_correlations(&assets, &market_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::CorrelationAnalysisResponse,
                content: MessageContent::CorrelationAnalysisResponse(correlation_analysis),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_regime_detection_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::RegimeDetectionRequest { assets, historical_data, config } = message.content {
            let regime_analysis = self.detect_regime_changes(&assets, &historical_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::RegimeDetectionResponse,
                content: MessageContent::RegimeDetectionResponse(regime_analysis),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::High,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_copula_analysis_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::CopulaAnalysisRequest { assets, market_data, config } = message.content {
            let copula_analysis = self.analyze_copula_dependencies(&assets, &market_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::CopulaAnalysisResponse,
                content: MessageContent::CopulaAnalysisResponse(copula_analysis),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_dynamic_correlation_request(&self, message: SwarmMessage) -> Result<()> {
        if let MessageContent::DynamicCorrelationRequest { assets, market_data, config } = message.content {
            let dynamic_correlations = self.calculate_dynamic_correlations(&assets, &market_data, &config).await?;
            
            let response = SwarmMessage {
                id: Uuid::new_v4(),
                sender_id: self.agent_id,
                sender_type: self.agent_type.clone(),
                recipient_id: Some(message.sender_id),
                message_type: MessageType::DynamicCorrelationResponse,
                content: MessageContent::DynamicCorrelationResponse(dynamic_correlations),
                timestamp: chrono::Utc::now(),
                priority: MessagePriority::Normal,
                requires_response: false,
            };
            
            self.message_tx.send(response)?;
        }
        Ok(())
    }
    
    async fn handle_health_check(&self, message: SwarmMessage) -> Result<()> {
        let health_status = self.get_health_status().await?;
        
        let response = SwarmMessage {
            id: Uuid::new_v4(),
            sender_id: self.agent_id,
            sender_type: self.agent_type.clone(),
            recipient_id: Some(message.sender_id),
            message_type: MessageType::HealthCheckResponse,
            content: MessageContent::HealthCheckResponse(health_status),
            timestamp: chrono::Utc::now(),
            priority: MessagePriority::Low,
            requires_response: false,
        };
        
        self.message_tx.send(response)?;
        Ok(())
    }
    
    pub async fn get_health_status(&self) -> Result<AgentHealthStatus> {
        let performance_metrics = self.performance_metrics.read().await;
        let health_level = if performance_metrics.average_calculation_time < Duration::from_micros(100) {
            HealthLevel::Healthy
        } else if performance_metrics.average_calculation_time < Duration::from_micros(500) {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };
        
        Ok(AgentHealthStatus {
            agent_id: self.agent_id,
            agent_type: self.agent_type.clone(),
            health_level,
            last_calculation_time: performance_metrics.last_calculation_time,
            average_calculation_time: performance_metrics.average_calculation_time,
            total_calculations: performance_metrics.total_calculations,
            error_count: performance_metrics.error_count,
            uptime: performance_metrics.uptime(),
        })
    }
}

#[async_trait]
impl SwarmAgent for CorrelationAnalysisAgent {
    async fn start(&mut self) -> Result<()> {
        info!("Starting Correlation Analysis Agent {}", self.agent_id);
        self.status = AgentStatus::Starting;
        
        // Start message processing loop
        let mut message_rx = self.message_rx.write().await.take()
            .ok_or_else(|| anyhow!("Message receiver already taken"))?;
        
        let agent_clone = Arc::new(RwLock::new(self));
        tokio::spawn(async move {
            while let Some(message) = message_rx.recv().await {
                let agent = agent_clone.read().await;
                if let Err(e) = agent.handle_message(message).await {
                    error!("Error handling message: {}", e);
                }
            }
        });
        
        self.status = AgentStatus::Running;
        info!("Correlation Analysis Agent {} started successfully", self.agent_id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        info!("Stopping Correlation Analysis Agent {}", self.agent_id);
        self.status = AgentStatus::Stopping;
        
        self.status = AgentStatus::Stopped;
        info!("Correlation Analysis Agent {} stopped successfully", self.agent_id);
        Ok(())
    }
    
    async fn get_agent_id(&self) -> Uuid {
        self.agent_id
    }
    
    async fn get_agent_type(&self) -> AgentType {
        self.agent_type.clone()
    }
    
    async fn get_status(&self) -> AgentStatus {
        self.status.clone()
    }
    
    async fn get_performance_metrics(&self) -> Result<AgentPerformanceMetrics> {
        let metrics = self.performance_metrics.read().await;
        Ok(metrics.clone())
    }
    
    async fn handle_coordination_message(&self, message: CoordinationMessage) -> Result<CoordinationResponse> {
        match message.message_type {
            CoordinationMessageType::CorrelationAnalysis => {
                if let (Some(assets), Some(market_data)) = (message.assets, message.market_data) {
                    let config = message.correlation_config.unwrap_or_default();
                    
                    let correlation_analysis = self.analyze_quantum_correlations(&assets, &market_data, &config).await?;
                    
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: true,
                        result: Some(RiskCalculationResult::CorrelationAnalysis(correlation_analysis)),
                        error: None,
                        calculation_time: correlation_analysis.calculation_time,
                    })
                } else {
                    Ok(CoordinationResponse {
                        agent_id: self.agent_id,
                        success: false,
                        result: None,
                        error: Some("Assets and market data required for correlation analysis".to_string()),
                        calculation_time: Duration::from_nanos(0),
                    })
                }
            }
            _ => {
                Ok(CoordinationResponse {
                    agent_id: self.agent_id,
                    success: false,
                    result: None,
                    error: Some(format!("Unsupported coordination message type: {:?}", message.message_type)),
                    calculation_time: Duration::from_nanos(0),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_correlation_analysis_agent_creation() {
        let config = CorrelationAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = CorrelationAnalysisAgent::new(config, coordination_hub, message_router).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_correlation_analysis_performance() {
        let config = CorrelationAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = CorrelationAnalysisAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let assets = vec![Asset::default(); 5];
        let market_data = MarketData::default();
        let analysis_config = CorrelationAnalysisConfig::default();
        
        let start_time = Instant::now();
        let result = agent.analyze_quantum_correlations(&assets, &market_data, &analysis_config).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(100), "Correlation analysis took {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_real_time_correlation_monitoring_performance() {
        let config = CorrelationAgentConfig::default();
        let coordination_hub = Arc::new(RwLock::new(
            AgentCoordinationHub::new(CoordinationConfig::default()).await.unwrap()
        ));
        let message_router = Arc::new(RwLock::new(
            SwarmMessageRouter::new(RoutingConfig::default()).await.unwrap()
        ));
        
        let agent = CorrelationAnalysisAgent::new(config, coordination_hub, message_router).await.unwrap();
        
        let assets = vec![Asset::default(); 3];
        let streaming_data = StreamingMarketData::default();
        
        let start_time = Instant::now();
        let result = agent.monitor_real_time_correlations(&assets, &streaming_data).await;
        let elapsed = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(elapsed < Duration::from_micros(1), "Real-time correlation monitoring took {:?}", elapsed);
    }
}