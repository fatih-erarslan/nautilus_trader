//! Market making engine implementation

use crate::prelude::*;

/// Market making trading engine
#[derive(Debug, Clone)]
pub struct MarketMakingEngine {
    pub spread_target: f64,
    pub inventory_limit: f64,
    pub tick_size: f64,
    pub quote_size: f64,
    pub risk_management: RiskManagement,
    pub current_inventory: f64,
    pub active_quotes: Vec<Quote>,
}

#[derive(Debug, Clone)]
pub struct RiskManagement {
    pub max_position_size: f64,
    pub inventory_skew_factor: f64,
    pub volatility_adjustment: f64,
    pub adverse_selection_protection: f64,
}

#[derive(Debug, Clone)]
pub struct Quote {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
}

impl MarketMakingEngine {
    pub fn new(spread_target: f64, inventory_limit: f64, tick_size: f64, quote_size: f64) -> Self {
        Self {
            spread_target,
            inventory_limit,
            tick_size,
            quote_size,
            risk_management: RiskManagement::default(),
            current_inventory: 0.0,
            active_quotes: Vec::new(),
        }
    }
    
    pub fn generate_quotes(&mut self, market_price: f64, volatility: f64, symbol: String) -> Quote {
        let adjusted_spread = self.calculate_adjusted_spread(volatility);
        let (bid_skew, ask_skew) = self.calculate_inventory_skew();
        
        let mid_price = market_price;
        let half_spread = adjusted_spread / 2.0;
        
        let bid_price = self.round_to_tick(mid_price - half_spread - bid_skew);
        let ask_price = self.round_to_tick(mid_price + half_spread + ask_skew);
        
        let (bid_size, ask_size) = self.calculate_quote_sizes();
        
        let quote = Quote {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            timestamp: chrono::Utc::now(),
            symbol,
        };
        
        self.active_quotes.push(quote.clone());
        quote
    }
    
    fn calculate_adjusted_spread(&self, volatility: f64) -> f64 {
        let vol_adjustment = volatility * self.risk_management.volatility_adjustment;
        let adverse_selection = self.risk_management.adverse_selection_protection;
        
        self.spread_target * (1.0 + vol_adjustment + adverse_selection)
    }
    
    fn calculate_inventory_skew(&self) -> (f64, f64) {
        let inventory_ratio = self.current_inventory / self.inventory_limit;
        let skew_magnitude = inventory_ratio.abs() * self.risk_management.inventory_skew_factor;
        
        if inventory_ratio > 0.0 {
            // Long inventory - widen ask, tighten bid
            (-skew_magnitude * 0.5, skew_magnitude)
        } else {
            // Short inventory - widen bid, tighten ask
            (skew_magnitude, -skew_magnitude * 0.5)
        }
    }
    
    fn calculate_quote_sizes(&self) -> (f64, f64) {
        let inventory_factor = 1.0 - (self.current_inventory.abs() / self.inventory_limit).min(0.8);
        
        let base_bid_size = self.quote_size * inventory_factor;
        let base_ask_size = self.quote_size * inventory_factor;
        
        // Adjust sizes based on inventory position
        if self.current_inventory > 0.0 {
            // Long inventory - prefer to sell
            (base_bid_size * 0.7, base_ask_size * 1.3)
        } else if self.current_inventory < 0.0 {
            // Short inventory - prefer to buy
            (base_bid_size * 1.3, base_ask_size * 0.7)
        } else {
            (base_bid_size, base_ask_size)
        }
    }
    
    fn round_to_tick(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }
    
    pub fn handle_fill(&mut self, side: TradeSide, price: f64, quantity: f64) {
        match side {
            TradeSide::Buy => {
                self.current_inventory += quantity;
            },
            TradeSide::Sell => {
                self.current_inventory -= quantity;
            },
        }
        
        // Update inventory constraints
        if self.current_inventory.abs() > self.inventory_limit {
            self.adjust_risk_parameters();
        }
    }
    
    fn adjust_risk_parameters(&mut self) {
        // Increase spread when near inventory limits
        let inventory_stress = self.current_inventory.abs() / self.inventory_limit;
        if inventory_stress > 0.8 {
            self.risk_management.inventory_skew_factor *= 1.2;
            self.risk_management.adverse_selection_protection *= 1.1;
        }
    }
    
    pub fn update_quotes(&mut self, market_price: f64, volatility: f64, symbol: String) {
        // Cancel old quotes
        self.active_quotes.clear();
        
        // Generate new quotes
        let new_quote = self.generate_quotes(market_price, volatility, symbol);
        
        // Risk check
        if self.current_inventory.abs() < self.inventory_limit * 0.9 {
            self.active_quotes.push(new_quote);
        }
    }
    
    pub fn get_pnl(&self, current_price: f64) -> f64 {
        // Mark-to-market PnL calculation
        let unrealized_pnl = self.current_inventory * current_price;
        
        // Add realized PnL from completed trades (simplified)
        let realized_pnl = 0.0; // This would track actual trade profits
        
        unrealized_pnl + realized_pnl
    }
    
    pub fn get_metrics(&self) -> MarketMakingMetrics {
        MarketMakingMetrics {
            current_inventory: self.current_inventory,
            inventory_utilization: self.current_inventory.abs() / self.inventory_limit,
            active_quote_count: self.active_quotes.len(),
            average_spread: self.calculate_average_spread(),
            risk_score: self.calculate_risk_score(),
        }
    }
    
    fn calculate_average_spread(&self) -> f64 {
        if self.active_quotes.is_empty() {
            return 0.0;
        }
        
        let total_spread: f64 = self.active_quotes
            .iter()
            .map(|q| q.ask_price - q.bid_price)
            .sum();
            
        total_spread / self.active_quotes.len() as f64
    }
    
    fn calculate_risk_score(&self) -> f64 {
        let inventory_risk = (self.current_inventory.abs() / self.inventory_limit).min(1.0);
        let concentration_risk = if self.active_quotes.len() > 0 { 0.2 } else { 1.0 };
        
        (inventory_risk + concentration_risk) / 2.0
    }
}

impl Default for RiskManagement {
    fn default() -> Self {
        Self {
            max_position_size: 1000.0,
            inventory_skew_factor: 0.1,
            volatility_adjustment: 2.0,
            adverse_selection_protection: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct MarketMakingMetrics {
    pub current_inventory: f64,
    pub inventory_utilization: f64,
    pub active_quote_count: usize,
    pub average_spread: f64,
    pub risk_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_making_engine() {
        let mut engine = MarketMakingEngine::new(0.01, 1000.0, 0.01, 100.0);
        
        let quote = engine.generate_quotes(100.0, 0.2, "BTCUSD".to_string());
        
        assert!(quote.ask_price > quote.bid_price);
        assert!(quote.bid_size > 0.0);
        assert!(quote.ask_size > 0.0);
    }
    
    #[test]
    fn test_inventory_management() {
        let mut engine = MarketMakingEngine::new(0.01, 1000.0, 0.01, 100.0);
        
        engine.handle_fill(TradeSide::Buy, 100.0, 200.0);
        assert_eq!(engine.current_inventory, 200.0);
        
        engine.handle_fill(TradeSide::Sell, 101.0, 100.0);
        assert_eq!(engine.current_inventory, 100.0);
    }
}