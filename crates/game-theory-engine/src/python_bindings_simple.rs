//! Simplified Python bindings for the Game Theory Engine

use pyo3::prelude::*;
use std::collections::HashMap;
use crate::{
    GameType, PlayerType, SophisticationLevel, Player, Strategy, StrategyType,
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
    #[pyo3(signature = (player_id, player_type, capital, risk_tolerance = None))]
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
    
    fn __str__(&self) -> String {
        format!("Player({}, {:?}, capital: {:.2})", 
                self.inner.id, self.inner.player_type, self.inner.capital)
    }
}

/// Simple Game Theory Engine wrapper
#[pyclass]
pub struct PyGameTheoryEngine {
    nash_solver: NashSolver,
    machiavellian: MachiavellianTactician,
    coalition_analyzer: CoalitionAnalyzer,
}

#[pymethods]
impl PyGameTheoryEngine {
    #[new]
    fn new() -> Self {
        PyGameTheoryEngine {
            nash_solver: NashSolver::new(1e-6, 1000),
            machiavellian: MachiavellianTactician::new(0.5, 0.3),
            coalition_analyzer: CoalitionAnalyzer::new(2, 0.6),
        }
    }
    
    fn analyze_prisoners_dilemma(&self, cooperation_payoff: f64, defection_payoff: f64, 
                                sucker_payoff: f64, _temptation_payoff: f64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            
            // Simple prisoners dilemma analysis
            let always_defect_payoff = defection_payoff;
            let _always_cooperate_payoff = sucker_payoff;
            let _mixed_strategy_payoff = (cooperation_payoff + defection_payoff) / 2.0;
            
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
            let dict = pyo3::types::PyDict::new(py);
            
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
            dict.set_item("equilibrium_strategy", if market_stability > 0.6 { "Cooperative" } else { "Competitive" })?;
            
            Ok(dict.into())
        })
    }
    
    fn get_strategic_recommendation(&self, player: &PyPlayer, market_volatility: f64) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = pyo3::types::PyDict::new(py);
            
            // Calculate optimal position sizing
            let risk_adjusted_capital = player.inner.capital * player.inner.risk_tolerance;
            let position_size = risk_adjusted_capital * (1.0 - market_volatility);
            result.set_item("optimal_position_size", position_size)?;
            
            // Strategy type recommendation
            let strategy_type = if market_volatility > 0.5 {
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
    m.add_class::<PyGameTheoryEngine>()?;
    
    // Add module metadata
    m.add("__version__", "0.1.0")?;
    m.add("__author__", "TENGRI Trading Swarm")?;
    m.add("__description__", "Game Theory Engine with Machiavellian Tactics and Nash Equilibrium Solving")?;
    
    Ok(())
}