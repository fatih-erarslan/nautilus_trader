//! Python bindings for the Game Theory Engine
//! 
//! This module provides PyO3 bindings to expose game theory functionality
//! to Python for integration with trading strategies.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use crate::{
    GameType, PlayerType, MarketRegime, ActionType, StrategyType,
    SophisticationLevel, GameState, Player, Strategy, NashEquilibrium,
    MarketContext, RegulatoryEnvironment, TransparencyLevel,
    nash_solver::NashSolver, machiavellian_tactics::MachiavellianTactician,
    coalition_games::CoalitionAnalyzer
};

/// Python wrapper for GameType enum
#[pyclass]
#[derive(Clone)]
pub struct PyGameType {
    pub inner: GameType,
}

#[pymethods]
impl PyGameType {
    #[new]
    fn new(game_type: &str) -> PyResult<Self> {
        let inner = match game_type {
            "prisoners_dilemma" => GameType::PrisonersDilemma,
            "chicken_game" => GameType::ChickenGame,
            "stag_hunt" => GameType::StagHunt,
            "battle_of_sexes" => GameType::BattleOfSexes,
            "english_auction" => GameType::EnglishAuction,
            "sealed_bid_auction" => GameType::SealedBidAuction,
            "market_making" => GameType::MarketMaking,
            "order_book_game" => GameType::OrderBookGame,
            "whale_battle" => GameType::WhaleBattile,
            "coalition_formation" => GameType::CoalitionFormation,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown game type: {}", game_type)
            )),
        };
        Ok(PyGameType { inner })
    }
    
    fn typical_player_count(&self) -> (usize, usize) {
        self.inner.typical_player_count()
    }
    
    fn is_zero_sum(&self) -> bool {
        self.inner.is_zero_sum()
    }
    
    fn supports_coalitions(&self) -> bool {
        self.inner.supports_coalitions()
    }
    
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Python wrapper for Player
#[pyclass]
#[derive(Clone)]
pub struct PyPlayer {
    pub inner: Player,
}

#[pymethods]
impl PyPlayer {
    #[new]
    #[pyo3(signature = (player_id, player_type, capital, risk_tolerance = 0.5))]
    fn new(player_id: String, player_type: &str, capital: f64, risk_tolerance: Option<f64>) -> PyResult<Self> {
        let risk_tolerance = risk_tolerance.unwrap_or(0.5);
        let player_type = match player_type {
            "retail_trader" => PlayerType::RetailTrader,
            "day_trader" => PlayerType::DayTrader,
            "hedge_fund" => PlayerType::HedgeFund,
            "whale" => PlayerType::Whale,
            "hft_arbitrager" => PlayerType::HFTArbitrager,
            "market_maker" => PlayerType::DesignatedMarketMaker,
            "swarm_trader" => PlayerType::SwarmTrader,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown player type: {}", player_type)
            )),
        };
        
        let sophistication = player_type.sophistication_level();
        
        let strategy = Strategy {
            name: "Default".to_string(),
            strategy_type: StrategyType::Pure,
            parameters: HashMap::new(),
            conditions: vec![],
            actions: vec![],
            expected_payoff: 0.0,
            risk_level: risk_tolerance,
        };
        
        let player = Player {
            id: player_id,
            player_type,
            capital,
            risk_tolerance,
            sophistication,
            current_position: 0.0,
            strategy,
            private_information: HashMap::new(),
            reputation: 0.5,
            cooperation_history: vec![],
        };
        
        Ok(PyPlayer { inner: player })
    }
    
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }
    
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital
    }
    
    #[getter]
    fn risk_tolerance(&self) -> f64 {
        self.inner.risk_tolerance
    }
    
    #[getter]
    fn current_position(&self) -> f64 {
        self.inner.current_position
    }
    
    #[getter]
    fn reputation(&self) -> f64 {
        self.inner.reputation
    }
    
    fn update_position(&mut self, new_position: f64) {
        self.inner.current_position = new_position;
    }
    
    fn update_reputation(&mut self, delta: f64) {
        self.inner.reputation = (self.inner.reputation + delta).clamp(0.0, 1.0);
    }
    
    fn __str__(&self) -> String {
        format!("Player({}, {:?}, capital: {:.2})", 
                self.inner.id, self.inner.player_type, self.inner.capital)
    }
}

/// Python wrapper for Nash Equilibrium Solver
#[pyclass]
pub struct PyNashSolver {
    inner: NashSolver,
}

#[pymethods]
impl PyNashSolver {
    #[new]
    #[pyo3(signature = (tolerance = 1e-6, max_iterations = 1000))]
    fn new(tolerance: f64, max_iterations: u32) -> Self {
        PyNashSolver {
            inner: NashSolver::new(tolerance, max_iterations),
        }
    }
    
    fn analyze_prisoners_dilemma(&self, cooperation_payoff: f64, defection_payoff: f64, 
                                sucker_payoff: f64, temptation_payoff: f64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Simple prisoners dilemma analysis
            let always_defect_payoff = defection_payoff;
            let always_cooperate_payoff = sucker_payoff;
            let mixed_strategy_payoff = (cooperation_payoff + defection_payoff) / 2.0;
            
            dict.set_item("nash_equilibrium", "Always Defect")?;
            dict.set_item("equilibrium_payoff", always_defect_payoff)?;
            dict.set_item("cooperation_payoff", cooperation_payoff)?;
            dict.set_item("defection_payoff", defection_payoff)?;
            dict.set_item("social_optimum", "Mutual Cooperation")?;
            dict.set_item("efficiency_loss", cooperation_payoff - always_defect_payoff)?;
            
            Ok(dict.into())
        })
    }
    
    fn analyze_market_game(&self, players: Vec<PyPlayer>, market_volatility: f64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Analyze market game dynamics
            let total_capital: f64 = players.iter().map(|p| p.inner.capital).sum();
            let avg_risk_tolerance: f64 = players.iter().map(|p| p.inner.risk_tolerance).sum::<f64>() / players.len() as f64;
            
            // Calculate market equilibrium
            let market_stability = (1.0 - market_volatility) * avg_risk_tolerance;
            let competition_intensity = players.len() as f64 / 10.0; // Normalize by typical market size
            
            dict.set_item("total_players", players.len())?;
            dict.set_item("total_capital", total_capital)?;
            dict.set_item("market_stability", market_stability)?;
            dict.set_item("competition_intensity", competition_intensity)?;
            dict.set_item("dominant_player_types", self.get_dominant_types(players.clone()))?;
            dict.set_item("equilibrium_strategy", if market_stability > 0.6 { "Cooperative" } else { "Competitive" })?;
            
            Ok(dict.into())
        })
    }
    
    fn get_dominant_types(&self, players: Vec<PyPlayer>) -> Vec<String> {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for player in &players {
            let type_str = format!("{:?}", player.inner.player_type);
            *type_counts.entry(type_str).or_insert(0) += 1;
        }
        
        let mut types: Vec<_> = type_counts.into_iter().collect();
        types.sort_by(|a, b| b.1.cmp(&a.1));
        types.into_iter().take(3).map(|(t, _)| t).collect()
    }
}

/// Python wrapper for Machiavellian Tactician
#[pyclass]
pub struct PyMachiavellianTactician {
    inner: MachiavellianTactician,
}

#[pymethods]
impl PyMachiavellianTactician {
    #[new]
    #[pyo3(signature = (deception_level = 0.5, manipulation_threshold = 0.3))]
    fn new(deception_level: f64, manipulation_threshold: f64) -> Self {
        PyMachiavellianTactician {
            inner: MachiavellianTactician::new(deception_level, manipulation_threshold),
        }
    }
    
    fn analyze_deception_opportunities(&self, market_volatility: f64, player_sophistication: f64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Calculate deception potential
            let deception_potential = market_volatility * (1.0 - player_sophistication);
            let manipulation_risk = if deception_potential > 0.7 { "High" } else if deception_potential > 0.4 { "Medium" } else { "Low" };
            
            dict.set_item("deception_potential", deception_potential)?;
            dict.set_item("manipulation_risk", manipulation_risk)?;
            dict.set_item("recommended_tactics", vec!["Information hiding", "Strategic bluffing", "Timing manipulation"])?;
            dict.set_item("detection_probability", 1.0 - deception_potential)?;
            
            Ok(dict.into())
        })
    }
    
    fn calculate_manipulation_payoff(&self, action_strength: f64, target_vulnerability: f64) -> f64 {
        // Simple manipulation payoff calculation
        action_strength * target_vulnerability * 0.8 // 0.8 factor for risk adjustment
    }
    
    fn generate_strategic_recommendations(&self, player_type: &str, market_conditions: f64) -> PyResult<Vec<String>> {
        let recommendations = match player_type {
            "whale" => vec![
                "Use large position to influence market sentiment".to_string(),
                "Strategic order timing to maximize impact".to_string(),
                "Information asymmetry exploitation".to_string(),
            ],
            "hft_arbitrager" => vec![
                "Speed advantage exploitation".to_string(),
                "Latency arbitrage opportunities".to_string(),
                "Order flow anticipation".to_string(),
            ],
            "hedge_fund" => vec![
                "Multi-strategy coordination".to_string(),
                "Risk-adjusted alpha generation".to_string(),
                "Regulatory arbitrage".to_string(),
            ],
            _ => vec![
                "Information gathering".to_string(),
                "Timing optimization".to_string(),
                "Risk management".to_string(),
            ],
        };
        
        Ok(recommendations)
    }
}

/// Python wrapper for Coalition Analyzer
#[pyclass]
pub struct PyCoalitionAnalyzer {
    inner: CoalitionAnalyzer,
}

#[pymethods]
impl PyCoalitionAnalyzer {
    #[new]
    #[pyo3(signature = (min_coalition_size = 2, stability_threshold = 0.6))]
    fn new(min_coalition_size: usize, stability_threshold: f64) -> Self {
        PyCoalitionAnalyzer {
            inner: CoalitionAnalyzer::new(min_coalition_size, stability_threshold),
        }
    }
    
    fn analyze_coalition_potential(&self, players: Vec<PyPlayer>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Analyze coalition formation potential
            let total_players = players.len();
            let total_capital: f64 = players.iter().map(|p| p.inner.capital).sum();
            let avg_reputation: f64 = players.iter().map(|p| p.inner.reputation).sum::<f64>() / total_players as f64;
            
            // Calculate coalition stability
            let stability_score = avg_reputation * (total_capital.ln() / 20.0).min(1.0);
            let formation_probability = if stability_score > 0.7 { 0.8 } else if stability_score > 0.4 { 0.5 } else { 0.2 };
            
            dict.set_item("coalition_stability", stability_score)?;
            dict.set_item("formation_probability", formation_probability)?;
            dict.set_item("optimal_coalition_size", (total_players as f64 * 0.6) as usize)?;
            dict.set_item("total_coalition_value", total_capital * stability_score)?;
            dict.set_item("recommended_structure", if stability_score > 0.6 { "Hierarchical" } else { "Flat" })?;
            
            Ok(dict.into())
        })
    }
    
    fn calculate_shapley_values(&self, players: Vec<PyPlayer>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Simplified Shapley value calculation
            let total_value: f64 = players.iter().map(|p| p.inner.capital).sum();
            
            for player in &players {
                let individual_contribution = player.inner.capital;
                let cooperation_bonus = player.inner.reputation * 0.1 * total_value;
                let shapley_value = individual_contribution + cooperation_bonus;
                
                dict.set_item(&player.inner.id, shapley_value)?;
            }
            
            Ok(dict.into())
        })
    }
}

/// Main Game Theory Engine wrapper
#[pyclass]
pub struct PyGameTheoryEngine {
    nash_solver: PyNashSolver,
    machiavellian: PyMachiavellianTactician,
    coalition_analyzer: PyCoalitionAnalyzer,
}

#[pymethods]
impl PyGameTheoryEngine {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let tolerance = config.and_then(|c| c.get_item("tolerance").ok())
            .and_then(|v| v.extract().ok()).unwrap_or(1e-6);
        let max_iterations = config.and_then(|c| c.get_item("max_iterations").ok())
            .and_then(|v| v.extract().ok()).unwrap_or(1000);
        let deception_level = config.and_then(|c| c.get_item("deception_level").ok())
            .and_then(|v| v.extract().ok()).unwrap_or(0.5);
            
        Ok(PyGameTheoryEngine {
            nash_solver: PyNashSolver::new(tolerance, max_iterations),
            machiavellian: PyMachiavellianTactician::new(deception_level, 0.3),
            coalition_analyzer: PyCoalitionAnalyzer::new(2, 0.6),
        })
    }
    
    fn analyze_market_game(&self, players: Vec<PyPlayer>, market_data: &PyDict) -> PyResult<PyObject> {
        let volatility: f64 = market_data.get_item("volatility").ok()
            .and_then(|v| v.extract().ok()).unwrap_or(0.2);
        let volume: f64 = market_data.get_item("volume").ok()
            .and_then(|v| v.extract().ok()).unwrap_or(1000000.0);
        let regime: String = market_data.get_item("regime").ok()
            .and_then(|v| v.extract().ok()).unwrap_or("normal".to_string());
            
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            
            // Get Nash analysis
            let nash_result = self.nash_solver.analyze_market_game(players.clone(), volatility)?;
            result.set_item("nash_analysis", nash_result)?;
            
            // Get Machiavellian analysis
            let avg_sophistication: f64 = players.iter()
                .map(|p| match p.inner.sophistication {
                    SophisticationLevel::Basic => 0.25,
                    SophisticationLevel::Intermediate => 0.5,
                    SophisticationLevel::Advanced => 0.75,
                    SophisticationLevel::Expert => 1.0,
                }).sum::<f64>() / players.len() as f64;
                
            let deception_analysis = self.machiavellian.analyze_deception_opportunities(volatility, avg_sophistication)?;
            result.set_item("deception_analysis", deception_analysis)?;
            
            // Get Coalition analysis
            let coalition_analysis = self.coalition_analyzer.analyze_coalition_potential(players)?;
            result.set_item("coalition_analysis", coalition_analysis)?;
            
            // Overall market assessment
            let market_assessment = PyDict::new(py);
            market_assessment.set_item("volatility", volatility)?;
            market_assessment.set_item("volume", volume)?;
            market_assessment.set_item("regime", regime)?;
            market_assessment.set_item("game_complexity", "High")?;
            market_assessment.set_item("recommended_strategy", "Adaptive")?;
            
            result.set_item("market_assessment", market_assessment)?;
            
            Ok(result.into())
        })
    }
    
    fn get_strategic_recommendation(&self, player: &PyPlayer, market_conditions: &PyDict) -> PyResult<PyObject> {
        let volatility: f64 = market_conditions.get_item("volatility").ok()
            .and_then(|v| v.extract().ok()).unwrap_or(0.2);
        let player_type = format!("{:?}", player.inner.player_type).to_lowercase();
        
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            
            // Generate recommendations based on player type and market conditions
            let recommendations = self.machiavellian.generate_strategic_recommendations(&player_type, volatility)?;
            result.set_item("recommendations", recommendations)?;
            
            // Calculate optimal position sizing
            let risk_adjusted_capital = player.inner.capital * player.inner.risk_tolerance;
            let position_size = risk_adjusted_capital * (1.0 - volatility);
            result.set_item("optimal_position_size", position_size)?;
            
            // Strategy type recommendation
            let strategy_type = if volatility > 0.5 {
                "Defensive"
            } else if player.inner.sophistication == SophisticationLevel::Expert {
                "Aggressive"
            } else {
                "Balanced"
            };
            result.set_item("strategy_type", strategy_type)?;
            
            Ok(result.into())
        })
    }
}

/// Python module definition
#[pymodule]
fn game_theory_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameType>()?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyNashSolver>()?;
    m.add_class::<PyMachiavellianTactician>()?;
    m.add_class::<PyCoalitionAnalyzer>()?;
    m.add_class::<PyGameTheoryEngine>()?;
    
    // Add module metadata
    m.add("__version__", "0.1.0")?;
    m.add("__author__", "TENGRI Trading Swarm")?;
    m.add("__description__", "Game Theory Engine with Machiavellian Tactics and Nash Equilibrium Solving")?;
    
    // Add utility functions
    #[pyfn(m)]
    #[pyo3(name = "create_market_players")]
    fn create_market_players(player_configs: Vec<(String, String, f64)>) -> PyResult<Vec<PyPlayer>> {
        let mut players = Vec::new();
        for (i, (player_type, name_prefix, capital)) in player_configs.iter().enumerate() {
            let player_id = format!("{}_{}", name_prefix, i);
            let player = PyPlayer::new(player_id, &player_type, capital, Some(0.5))?;
            players.push(player);
        }
        Ok(players)
    }
    
    #[pyfn(m)]
    #[pyo3(name = "analyze_prisoners_dilemma")]
    fn analyze_prisoners_dilemma(cooperation: f64, defection: f64, sucker: f64, temptation: f64) -> PyResult<PyObject> {
        let solver = PyNashSolver::new(1e-6, 1000);
        solver.analyze_prisoners_dilemma(cooperation, defection, sucker, temptation)
    }
    
    Ok(())
}