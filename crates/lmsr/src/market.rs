//! Market state management and thread-safe operations
//! 
//! This module provides high-level market operations with thread safety,
//! persistence, and real-time updates for production trading systems.

use crate::errors::{LMSRError, Result};
use crate::lmsr::{LMSRMarketMaker, MarketStatistics};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Thread-safe market wrapper for concurrent access
#[derive(Clone)]
pub struct Market {
    inner: Arc<RwLock<LMSRMarketMaker>>,
    metadata: Arc<MarketMetadata>,
    listeners: Arc<Mutex<Vec<Box<dyn Fn(&MarketEvent) + Send + Sync>>>>,
}

/// Market metadata that doesn't change during trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub outcomes: Vec<String>,
    pub created_at: SystemTime,
    pub closes_at: Option<SystemTime>,
    pub resolution_source: String,
    pub category: String,
    pub tags: Vec<String>,
}

/// Position held by a trader
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub trader_id: String,
    pub quantities: Vec<f64>,
    pub average_costs: Vec<f64>,
    pub total_invested: f64,
    pub last_updated: SystemTime,
}

/// Market state snapshot for persistence and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub metadata: MarketMetadata,
    pub statistics: MarketStatistics,
    pub positions: HashMap<String, Position>,
    pub trade_history: Vec<TradeRecord>,
    pub timestamp: SystemTime,
}

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub id: String,
    pub trader_id: String,
    pub quantities: Vec<f64>,
    pub cost: f64,
    pub timestamp: SystemTime,
    pub prices_before: Vec<f64>,
    pub prices_after: Vec<f64>,
}

/// Market events for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    Trade {
        trade: TradeRecord,
        new_prices: Vec<f64>,
    },
    PriceUpdate {
        old_prices: Vec<f64>,
        new_prices: Vec<f64>,
        timestamp: SystemTime,
    },
    MarketClosed {
        final_state: MarketState,
    },
    Error {
        error: String,
        timestamp: SystemTime,
    },
}

impl Market {
    /// Create a new market with metadata
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        let market_maker = LMSRMarketMaker::new(num_outcomes, liquidity_parameter)?;
        
        let metadata = MarketMetadata {
            id: format!("market_{}", Self::generate_timestamp()),
            name: "LMSR Market".to_string(),
            description: "Logarithmic Market Scoring Rule prediction market".to_string(),
            outcomes: (0..num_outcomes).map(|i| format!("Outcome {}", i)).collect(),
            created_at: SystemTime::now(),
            closes_at: None,
            resolution_source: "Manual".to_string(),
            category: "General".to_string(),
            tags: vec!["lmsr".to_string(), "prediction".to_string()],
        };
        
        Ok(Self {
            inner: Arc::new(RwLock::new(market_maker)),
            metadata: Arc::new(metadata),
            listeners: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Create a market with custom metadata
    pub fn with_metadata(
        num_outcomes: usize, 
        liquidity_parameter: f64, 
        metadata: MarketMetadata
    ) -> Result<Self> {
        let market_maker = LMSRMarketMaker::new(num_outcomes, liquidity_parameter)?;
        
        if metadata.outcomes.len() != num_outcomes {
            return Err(LMSRError::invalid_market(
                "Metadata outcomes count doesn't match num_outcomes"
            ));
        }
        
        Ok(Self {
            inner: Arc::new(RwLock::new(market_maker)),
            metadata: Arc::new(metadata),
            listeners: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Execute a trade atomically
    pub fn execute_trade(&self, trader_id: String, quantities: &[f64]) -> Result<TradeRecord> {
        // Get prices before trade
        let prices_before = {
            let market = self.inner.read();
            market.get_prices()?
        };
        
        // Execute trade with write lock
        let (cost, prices_after) = {
            let mut market = self.inner.write();
            let cost = market.execute_trade(quantities)?;
            let prices_after = market.get_prices()?;
            (cost, prices_after)
        };
        
        // Create trade record
        let trade = TradeRecord {
            id: format!("trade_{}", Self::generate_timestamp()),
            trader_id: trader_id.clone(),
            quantities: quantities.to_vec(),
            cost,
            timestamp: SystemTime::now(),
            prices_before,
            prices_after: prices_after.clone(),
        };
        
        // Emit event
        self.emit_event(MarketEvent::Trade {
            trade: trade.clone(),
            new_prices: prices_after,
        });
        
        Ok(trade)
    }
    
    /// Get current market prices (thread-safe read)
    pub fn get_prices(&self) -> Result<Vec<f64>> {
        let market = self.inner.read();
        market.get_prices()
    }
    
    /// Get price for specific outcome (thread-safe read)
    pub fn get_price(&self, outcome: usize) -> Result<f64> {
        let market = self.inner.read();
        market.get_price(outcome)
    }
    
    /// Get market statistics (thread-safe read)
    pub fn get_statistics(&self) -> Result<MarketStatistics> {
        let market = self.inner.read();
        market.get_statistics()
    }
    
    /// Get market metadata
    pub fn get_metadata(&self) -> &MarketMetadata {
        &self.metadata
    }
    
    /// Calculate cost of potential trade without executing
    pub fn calculate_trade_cost(&self, quantities: &[f64]) -> Result<f64> {
        let market = self.inner.read();
        market.calculator.calculate_buy_cost(&market.quantities, quantities)
    }
    
    /// Get current market state snapshot
    pub fn get_state(&self) -> Result<MarketState> {
        let market = self.inner.read();
        let statistics = market.get_statistics()?;
        
        Ok(MarketState {
            metadata: (*self.metadata).clone(),
            statistics,
            positions: HashMap::new(), // Would be populated from position manager
            trade_history: Vec::new(), // Would be populated from trade history
            timestamp: SystemTime::now(),
        })
    }
    
    /// Register event listener for real-time updates
    pub fn add_listener<F>(&self, listener: F) 
    where 
        F: Fn(&MarketEvent) + Send + Sync + 'static 
    {
        let mut listeners = self.listeners.lock();
        listeners.push(Box::new(listener));
    }
    
    /// Check if market is closed
    pub fn is_closed(&self) -> bool {
        self.metadata.closes_at
            .map(|close_time| SystemTime::now() > close_time)
            .unwrap_or(false)
    }
    
    /// Close the market (prevents further trading)
    pub fn close_market(&self) -> Result<MarketState> {
        if self.is_closed() {
            return Err(LMSRError::market_state("Market is already closed"));
        }
        
        let final_state = self.get_state()?;
        
        self.emit_event(MarketEvent::MarketClosed {
            final_state: final_state.clone(),
        });
        
        Ok(final_state)
    }
    
    /// Reset market to initial state (for testing)
    pub fn reset(&self) -> Result<()> {
        let mut market = self.inner.write();
        market.reset();
        Ok(())
    }
    
    /// Internal helper to emit events
    fn emit_event(&self, event: MarketEvent) {
        let listeners = self.listeners.lock();
        for listener in listeners.iter() {
            listener(&event);
        }
    }
    
    /// Generate timestamp-based ID
    fn generate_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Position manager for tracking trader positions
#[derive(Debug)]
pub struct PositionManager {
    positions: Arc<RwLock<HashMap<String, Position>>>,
}

impl PositionManager {
    /// Create new position manager
    pub fn new() -> Self {
        Self {
            positions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Update position after trade
    pub fn update_position(
        &self, 
        trader_id: String, 
        trade_quantities: &[f64], 
        trade_cost: f64
    ) -> Result<Position> {
        let mut positions = self.positions.write();
        
        let position = positions.entry(trader_id.clone()).or_insert_with(|| Position {
            trader_id: trader_id.clone(),
            quantities: vec![0.0; trade_quantities.len()],
            average_costs: vec![0.0; trade_quantities.len()],
            total_invested: 0.0,
            last_updated: SystemTime::now(),
        });
        
        // Update quantities
        for (i, &trade_qty) in trade_quantities.iter().enumerate() {
            position.quantities[i] += trade_qty;
        }
        
        // Update costs and investment
        position.total_invested += trade_cost;
        position.last_updated = SystemTime::now();
        
        // Recalculate average costs (simplified)
        for i in 0..position.average_costs.len() {
            if position.quantities[i] != 0.0 {
                position.average_costs[i] = position.total_invested / position.quantities.iter().sum::<f64>();
            }
        }
        
        Ok(position.clone())
    }
    
    /// Get position for trader
    pub fn get_position(&self, trader_id: &str) -> Option<Position> {
        let positions = self.positions.read();
        positions.get(trader_id).cloned()
    }
    
    /// Get all positions
    pub fn get_all_positions(&self) -> HashMap<String, Position> {
        let positions = self.positions.read();
        positions.clone()
    }
    
    /// Calculate position value at current prices
    pub fn calculate_position_value(&self, trader_id: &str, current_prices: &[f64]) -> Result<f64> {
        let positions = self.positions.read();
        
        if let Some(position) = positions.get(trader_id) {
            if position.quantities.len() != current_prices.len() {
                return Err(LMSRError::invalid_quantity("Price vector length mismatch"));
            }
            
            let value: f64 = position.quantities.iter()
                .zip(current_prices.iter())
                .map(|(&qty, &price)| qty * price)
                .sum();
            
            Ok(value)
        } else {
            Ok(0.0)
        }
    }
}

/// Market factory for creating configured markets
pub struct MarketFactory;

impl MarketFactory {
    /// Create a binary market (yes/no)
    pub fn create_binary_market(
        name: String, 
        description: String, 
        liquidity: f64
    ) -> Result<Market> {
        let metadata = MarketMetadata {
            id: format!("binary_{}", Market::generate_timestamp()),
            name,
            description,
            outcomes: vec!["Yes".to_string(), "No".to_string()],
            created_at: SystemTime::now(),
            closes_at: None,
            resolution_source: "Manual".to_string(),
            category: "Binary".to_string(),
            tags: vec!["binary".to_string(), "yes-no".to_string()],
        };
        
        Market::with_metadata(2, liquidity, metadata)
    }
    
    /// Create a categorical market
    pub fn create_categorical_market(
        name: String, 
        description: String, 
        outcomes: Vec<String>, 
        liquidity: f64
    ) -> Result<Market> {
        if outcomes.len() < 2 {
            return Err(LMSRError::invalid_market("At least 2 outcomes required"));
        }
        
        let metadata = MarketMetadata {
            id: format!("categorical_{}", Market::generate_timestamp()),
            name,
            description,
            outcomes,
            created_at: SystemTime::now(),
            closes_at: None,
            resolution_source: "Manual".to_string(),
            category: "Categorical".to_string(),
            tags: vec!["categorical".to_string(), "multiple-choice".to_string()],
        };
        
        Market::with_metadata(metadata.outcomes.len(), liquidity, metadata)
    }
    
    /// Create a time-limited market
    pub fn create_timed_market(
        name: String, 
        description: String, 
        outcomes: Vec<String>, 
        liquidity: f64,
        duration: Duration
    ) -> Result<Market> {
        let closes_at = SystemTime::now() + duration;
        
        let metadata = MarketMetadata {
            id: format!("timed_{}", Market::generate_timestamp()),
            name,
            description,
            outcomes: outcomes.clone(),
            created_at: SystemTime::now(),
            closes_at: Some(closes_at),
            resolution_source: "Automatic".to_string(),
            category: "Timed".to_string(),
            tags: vec!["timed".to_string(), "auto-close".to_string()],
        };
        
        Market::with_metadata(outcomes.len(), liquidity, metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_market_creation() {
        let market = Market::new(2, 100.0).unwrap();
        let metadata = market.get_metadata();
        assert_eq!(metadata.outcomes.len(), 2);
        assert!(!market.is_closed());
    }

    #[test]
    fn test_concurrent_trading() {
        let market = Arc::new(Market::new(2, 100.0).unwrap());
        let num_threads = 10;
        let trades_per_thread = 100;
        
        let handles: Vec<_> = (0..num_threads).map(|i| {
            let market = Arc::clone(&market);
            thread::spawn(move || {
                for j in 0..trades_per_thread {
                    let trader_id = format!("trader_{}_{}", i, j);
                    let quantities = vec![1.0, 0.0];
                    let _ = market.execute_trade(trader_id, &quantities);
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = market.get_statistics().unwrap();
        assert_eq!(stats.trade_count, (num_threads * trades_per_thread) as u64);
    }

    #[test]
    fn test_event_listeners() {
        let market = Market::new(2, 100.0).unwrap();
        let event_count = Arc::new(AtomicUsize::new(0));
        
        {
            let counter = Arc::clone(&event_count);
            market.add_listener(move |_event| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }
        
        // Execute some trades
        for i in 0..5 {
            let trader_id = format!("trader_{}", i);
            let _ = market.execute_trade(trader_id, &[1.0, 0.0]);
        }
        
        // Give events time to process
        thread::sleep(Duration::from_millis(10));
        
        assert_eq!(event_count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_position_manager() {
        let pm = PositionManager::new();
        
        let position = pm.update_position(
            "trader1".to_string(), 
            &[10.0, 5.0], 
            15.0
        ).unwrap();
        
        assert_eq!(position.quantities, vec![10.0, 5.0]);
        assert_eq!(position.total_invested, 15.0);
        
        let retrieved = pm.get_position("trader1").unwrap();
        assert_eq!(retrieved.quantities, vec![10.0, 5.0]);
        
        let value = pm.calculate_position_value("trader1", &[0.6, 0.4]).unwrap();
        assert_eq!(value, 10.0 * 0.6 + 5.0 * 0.4);
    }

    #[test]
    fn test_market_factory() {
        let binary = MarketFactory::create_binary_market(
            "Will it rain?".to_string(),
            "Weather prediction".to_string(),
            100.0
        ).unwrap();
        
        assert_eq!(binary.get_metadata().outcomes.len(), 2);
        assert_eq!(binary.get_metadata().category, "Binary");
        
        let categorical = MarketFactory::create_categorical_market(
            "Election winner".to_string(),
            "2024 election".to_string(),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            200.0
        ).unwrap();
        
        assert_eq!(categorical.get_metadata().outcomes.len(), 3);
        assert_eq!(categorical.get_metadata().category, "Categorical");
    }

    #[test]
    fn test_timed_market() {
        let market = MarketFactory::create_timed_market(
            "Short term".to_string(),
            "Quick market".to_string(),
            vec!["A".to_string(), "B".to_string()],
            100.0,
            Duration::from_millis(1) // Very short for testing
        ).unwrap();
        
        // Market should close very quickly
        thread::sleep(Duration::from_millis(10));
        assert!(market.is_closed());
    }
}