//! Arbitrage engine implementation

use crate::prelude::*;

/// Arbitrage trading engine for cross-market opportunities
#[derive(Debug, Clone)]
pub struct ArbitrageEngine {
    pub markets: Vec<Market>,
    pub min_profit_threshold: f64,
    pub max_position_size: f64,
    pub latency_tolerance: std::time::Duration,
    pub active_opportunities: Vec<ArbitrageOpportunity>,
    pub execution_history: Vec<ArbitrageTrade>,
}

#[derive(Debug, Clone)]
pub struct Market {
    pub name: String,
    pub base_currency: String,
    pub quote_currency: String,
    pub current_bid: f64,
    pub current_ask: f64,
    pub liquidity_bid: f64,
    pub liquidity_ask: f64,
    pub latency_ms: u64,
    pub trading_fee: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub opportunity_type: ArbitrageType,
    pub buy_market: String,
    pub sell_market: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub max_quantity: f64,
    pub expected_profit: f64,
    pub profit_margin: f64,
    pub execution_time_limit: chrono::DateTime<chrono::Utc>,
    pub risk_score: f64,
}

#[derive(Debug, Clone)]
pub enum ArbitrageType {
    Simple,           // Buy low, sell high across markets
    Triangular,       // Currency triangle arbitrage
    Statistical,      // Mean reversion based
    Latency,         // Speed-based arbitrage
}

#[derive(Debug, Clone)]
pub struct ArbitrageTrade {
    pub opportunity_id: String,
    pub buy_market: String,
    pub sell_market: String,
    pub quantity: f64,
    pub buy_price: f64,
    pub sell_price: f64,
    pub actual_profit: f64,
    pub execution_time: std::time::Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub success: bool,
}

impl ArbitrageEngine {
    pub fn new(min_profit_threshold: f64, max_position_size: f64, latency_tolerance_ms: u64) -> Self {
        Self {
            markets: Vec::new(),
            min_profit_threshold,
            max_position_size,
            latency_tolerance: std::time::Duration::from_millis(latency_tolerance_ms),
            active_opportunities: Vec::new(),
            execution_history: Vec::new(),
        }
    }
    
    pub fn add_market(&mut self, market: Market) {
        self.markets.push(market);
    }
    
    pub fn scan_opportunities(&mut self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Simple arbitrage opportunities
        opportunities.extend(self.scan_simple_arbitrage());
        
        // Triangular arbitrage opportunities
        opportunities.extend(self.scan_triangular_arbitrage());
        
        // Statistical arbitrage opportunities
        opportunities.extend(self.scan_statistical_arbitrage());
        
        // Filter by profitability and risk
        opportunities.retain(|opp| {
            opp.expected_profit >= self.min_profit_threshold &&
            opp.risk_score <= 0.7 &&
            opp.execution_time_limit > chrono::Utc::now()
        });
        
        self.active_opportunities = opportunities.clone();
        opportunities
    }
    
    fn scan_simple_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        for i in 0..self.markets.len() {
            for j in 0..self.markets.len() {
                if i == j { continue; }
                
                let market_a = &self.markets[i];
                let market_b = &self.markets[j];
                
                // Check if same currency pair
                if market_a.base_currency != market_b.base_currency ||
                   market_a.quote_currency != market_b.quote_currency {
                    continue;
                }
                
                // Check buy low in market A, sell high in market B
                if market_a.current_ask < market_b.current_bid {
                    let buy_price = market_a.current_ask;
                    let sell_price = market_b.current_bid;
                    let max_quantity = market_a.liquidity_ask.min(market_b.liquidity_bid)
                        .min(self.max_position_size);
                    
                    let gross_profit = (sell_price - buy_price) * max_quantity;
                    let total_fees = (buy_price * max_quantity * market_a.trading_fee) +
                                   (sell_price * max_quantity * market_b.trading_fee);
                    let net_profit = gross_profit - total_fees;
                    let profit_margin = net_profit / (buy_price * max_quantity);
                    
                    if net_profit >= self.min_profit_threshold {
                        let risk_score = self.calculate_risk_score(market_a, market_b);
                        let execution_window = self.calculate_execution_window(market_a, market_b);
                        
                        opportunities.push(ArbitrageOpportunity {
                            opportunity_type: ArbitrageType::Simple,
                            buy_market: market_a.name.clone(),
                            sell_market: market_b.name.clone(),
                            buy_price,
                            sell_price,
                            max_quantity,
                            expected_profit: net_profit,
                            profit_margin,
                            execution_time_limit: chrono::Utc::now() + execution_window,
                            risk_score,
                        });
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn scan_triangular_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Find currency triangles (A->B->C->A)
        for market_ab in &self.markets {
            for market_bc in &self.markets {
                for market_ca in &self.markets {
                    if self.is_valid_triangle(&market_ab, &market_bc, &market_ca) {
                        if let Some(opportunity) = self.calculate_triangular_opportunity(
                            &market_ab, &market_bc, &market_ca) {
                            opportunities.push(opportunity);
                        }
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn is_valid_triangle(&self, ab: &Market, bc: &Market, ca: &Market) -> bool {
        // Check if markets form a valid triangle: A/B -> B/C -> C/A
        ab.quote_currency == bc.base_currency &&
        bc.quote_currency == ca.base_currency &&
        ca.quote_currency == ab.base_currency
    }
    
    fn calculate_triangular_opportunity(&self, ab: &Market, bc: &Market, ca: &Market) 
        -> Option<ArbitrageOpportunity> {
        
        // Calculate if we can profit from A -> B -> C -> A
        let start_amount = 1000.0; // Starting with 1000 units of currency A
        
        // A -> B (sell A for B)
        let amount_b = start_amount / ab.current_ask;
        let after_fee_b = amount_b * (1.0 - ab.trading_fee);
        
        // B -> C (sell B for C)
        let amount_c = after_fee_b / bc.current_ask;
        let after_fee_c = amount_c * (1.0 - bc.trading_fee);
        
        // C -> A (sell C for A)
        let final_amount_a = after_fee_c * ca.current_bid;
        let after_final_fee = final_amount_a * (1.0 - ca.trading_fee);
        
        let profit = after_final_fee - start_amount;
        let profit_margin = profit / start_amount;
        
        if profit >= self.min_profit_threshold {
            let risk_score = (ab.latency_ms + bc.latency_ms + ca.latency_ms) as f64 / 1000.0;
            let execution_window = std::time::Duration::from_millis(
                (ab.latency_ms + bc.latency_ms + ca.latency_ms) * 2
            );
            
            Some(ArbitrageOpportunity {
                opportunity_type: ArbitrageType::Triangular,
                buy_market: format!("{}->{}->{}", ab.name, bc.name, ca.name),
                sell_market: "triangular".to_string(),
                buy_price: ab.current_ask,
                sell_price: ca.current_bid,
                max_quantity: start_amount,
                expected_profit: profit,
                profit_margin,
                execution_time_limit: chrono::Utc::now() + execution_window,
                risk_score: risk_score.min(1.0),
            })
        } else {
            None
        }
    }
    
    fn scan_statistical_arbitrage(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        // Look for price divergences that historically revert
        for i in 0..self.markets.len() {
            for j in (i+1)..self.markets.len() {
                let market_a = &self.markets[i];
                let market_b = &self.markets[j];
                
                if market_a.base_currency == market_b.base_currency &&
                   market_a.quote_currency == market_b.quote_currency {
                    
                    let price_a = (market_a.current_bid + market_a.current_ask) / 2.0;
                    let price_b = (market_b.current_bid + market_b.current_ask) / 2.0;
                    let price_diff = (price_a - price_b).abs();
                    let avg_price = (price_a + price_b) / 2.0;
                    let divergence = price_diff / avg_price;
                    
                    // If divergence is significant (>1%), create statistical arbitrage opportunity
                    if divergence > 0.01 {
                        let (buy_market, sell_market, buy_price, sell_price) = 
                            if price_a < price_b {
                                (market_a.name.clone(), market_b.name.clone(), price_a, price_b)
                            } else {
                                (market_b.name.clone(), market_a.name.clone(), price_b, price_a)
                            };
                        
                        let max_quantity = market_a.liquidity_ask.min(market_b.liquidity_bid)
                            .min(self.max_position_size);
                        let expected_profit = price_diff * max_quantity * 0.5; // Assume 50% reversion
                        let profit_margin = expected_profit / (buy_price * max_quantity);
                        
                        opportunities.push(ArbitrageOpportunity {
                            opportunity_type: ArbitrageType::Statistical,
                            buy_market,
                            sell_market,
                            buy_price,
                            sell_price,
                            max_quantity,
                            expected_profit,
                            profit_margin,
                            execution_time_limit: chrono::Utc::now() + 
                                std::time::Duration::from_secs(300), // 5 minute window
                            risk_score: divergence, // Higher divergence = higher risk
                        });
                    }
                }
            }
        }
        
        opportunities
    }
    
    fn calculate_risk_score(&self, market_a: &Market, market_b: &Market) -> f64 {
        let latency_risk = ((market_a.latency_ms + market_b.latency_ms) as f64 / 1000.0).min(1.0);
        let liquidity_risk = if market_a.liquidity_ask < 100.0 || market_b.liquidity_bid < 100.0 {
            0.8
        } else {
            0.2
        };
        let staleness_risk = {
            let now = chrono::Utc::now();
            let age_a = (now - market_a.last_update).num_seconds() as f64;
            let age_b = (now - market_b.last_update).num_seconds() as f64;
            ((age_a + age_b) / 60.0).min(1.0) // Risk increases with data age
        };
        
        (latency_risk + liquidity_risk + staleness_risk) / 3.0
    }
    
    fn calculate_execution_window(&self, market_a: &Market, market_b: &Market) -> std::time::Duration {
        let base_window = std::time::Duration::from_secs(30);
        let latency_penalty = std::time::Duration::from_millis(market_a.latency_ms + market_b.latency_ms);
        
        base_window - latency_penalty.min(base_window / 2)
    }
    
    pub fn execute_opportunity(&mut self, opportunity: &ArbitrageOpportunity) -> Result<ArbitrageTrade> {
        let start_time = std::time::Instant::now();
        
        // Simulate trade execution
        let success = self.simulate_execution_success(opportunity);
        let actual_profit = if success {
            opportunity.expected_profit * (0.8 + rand::random::<f64>() * 0.4) // 80-120% of expected
        } else {
            -opportunity.expected_profit * 0.1 // Small loss on failure
        };
        
        let trade = ArbitrageTrade {
            opportunity_id: uuid::Uuid::new_v4().to_string(),
            buy_market: opportunity.buy_market.clone(),
            sell_market: opportunity.sell_market.clone(),
            quantity: opportunity.max_quantity,
            buy_price: opportunity.buy_price,
            sell_price: opportunity.sell_price,
            actual_profit,
            execution_time: start_time.elapsed(),
            timestamp: chrono::Utc::now(),
            success,
        };
        
        self.execution_history.push(trade.clone());
        
        if success {
            Ok(trade)
        } else {
            Err(crate::Error::Execution("Trade execution failed".to_string()))
        }
    }
    
    fn simulate_execution_success(&self, opportunity: &ArbitrageOpportunity) -> bool {
        // Factors affecting execution success
        let time_factor = if chrono::Utc::now() < opportunity.execution_time_limit { 0.9 } else { 0.3 };
        let risk_factor = 1.0 - opportunity.risk_score;
        let profit_factor = (opportunity.profit_margin * 100.0).min(1.0);
        
        let success_probability = time_factor * risk_factor * profit_factor;
        rand::random::<f64>() < success_probability
    }
    
    pub fn get_performance_metrics(&self) -> ArbitrageMetrics {
        let total_trades = self.execution_history.len();
        let successful_trades = self.execution_history.iter().filter(|t| t.success).count();
        let total_profit: f64 = self.execution_history.iter().map(|t| t.actual_profit).sum();
        let win_rate = if total_trades > 0 {
            successful_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        
        let avg_execution_time = if total_trades > 0 {
            let total_time: std::time::Duration = self.execution_history
                .iter()
                .map(|t| t.execution_time)
                .sum();
            total_time / total_trades as u32
        } else {
            std::time::Duration::from_secs(0)
        };
        
        ArbitrageMetrics {
            total_opportunities_found: self.active_opportunities.len(),
            total_trades_executed: total_trades,
            successful_trades,
            win_rate,
            total_profit,
            average_profit_per_trade: if total_trades > 0 { total_profit / total_trades as f64 } else { 0.0 },
            average_execution_time_ms: avg_execution_time.as_millis() as u64,
            active_markets: self.markets.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArbitrageMetrics {
    pub total_opportunities_found: usize,
    pub total_trades_executed: usize,
    pub successful_trades: usize,
    pub win_rate: f64,
    pub total_profit: f64,
    pub average_profit_per_trade: f64,
    pub average_execution_time_ms: u64,
    pub active_markets: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arbitrage_engine_creation() {
        let engine = ArbitrageEngine::new(10.0, 1000.0, 100);
        assert_eq!(engine.min_profit_threshold, 10.0);
        assert_eq!(engine.max_position_size, 1000.0);
    }
    
    #[test]
    fn test_market_addition() {
        let mut engine = ArbitrageEngine::new(10.0, 1000.0, 100);
        
        let market = Market {
            name: "Exchange A".to_string(),
            base_currency: "BTC".to_string(),
            quote_currency: "USD".to_string(),
            current_bid: 50000.0,
            current_ask: 50010.0,
            liquidity_bid: 1000.0,
            liquidity_ask: 1000.0,
            latency_ms: 50,
            trading_fee: 0.001,
            last_update: chrono::Utc::now(),
        };
        
        engine.add_market(market);
        assert_eq!(engine.markets.len(), 1);
    }
    
    #[test]
    fn test_opportunity_scanning() {
        let mut engine = ArbitrageEngine::new(5.0, 1000.0, 100);
        
        // Add two markets with price difference
        let market_a = Market {
            name: "Exchange A".to_string(),
            base_currency: "BTC".to_string(),
            quote_currency: "USD".to_string(),
            current_bid: 49990.0,
            current_ask: 50000.0,
            liquidity_bid: 1000.0,
            liquidity_ask: 1000.0,
            latency_ms: 50,
            trading_fee: 0.001,
            last_update: chrono::Utc::now(),
        };
        
        let market_b = Market {
            name: "Exchange B".to_string(),
            base_currency: "BTC".to_string(),
            quote_currency: "USD".to_string(),
            current_bid: 50020.0,
            current_ask: 50030.0,
            liquidity_bid: 1000.0,
            liquidity_ask: 1000.0,
            latency_ms: 60,
            trading_fee: 0.001,
            last_update: chrono::Utc::now(),
        };
        
        engine.add_market(market_a);
        engine.add_market(market_b);
        
        let opportunities = engine.scan_opportunities();
        assert!(!opportunities.is_empty());
    }
}