//! Arbitrage detection for Polymarket

use crate::error::Result;
use crate::models::*;
use crate::polymarket::client::PolymarketClient;
use rust_decimal::Decimal;
use std::collections::HashMap;
use tracing::{debug, info};

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub market_id: String,
    pub outcomes: Vec<ArbitrageOutcome>,
    pub total_cost: Decimal,
    pub profit: Decimal,
    pub profit_percentage: Decimal,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone)]
pub struct ArbitrageOutcome {
    pub outcome_id: String,
    pub side: OrderSide,
    pub price: Decimal,
    pub size: Decimal,
    pub exchange: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Arbitrage detector configuration
#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    /// Minimum profit threshold (e.g., 0.02 = 2%)
    pub min_profit: Decimal,
    /// Maximum size to trade
    pub max_size: Decimal,
    /// Transaction fee rate
    pub fee_rate: Decimal,
    /// Check interval in seconds
    pub check_interval: u64,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_profit: Decimal::new(2, 2), // 2%
            max_size: Decimal::new(1000, 0),
            fee_rate: Decimal::new(2, 2), // 2%
            check_interval: 5,
        }
    }
}

/// Arbitrage detector
pub struct PolymarketArbitrage {
    client: PolymarketClient,
    config: ArbitrageConfig,
}

impl PolymarketArbitrage {
    pub fn new(client: PolymarketClient, config: ArbitrageConfig) -> Self {
        Self { client, config }
    }

    /// Check for arbitrage opportunities in a market
    pub async fn check_market_arbitrage(&self, market_id: &str) -> Result<Vec<ArbitrageOpportunity>> {
        info!("Checking arbitrage for market: {}", market_id);

        let market = self.client.get_market(market_id).await?;
        let mut opportunities = Vec::new();

        // Check probability sum arbitrage
        if let Some(opp) = self.check_probability_sum_arbitrage(&market).await? {
            opportunities.push(opp);
        }

        // Check cross-exchange arbitrage (if multiple exchanges available)
        // This would require data from other exchanges

        Ok(opportunities)
    }

    /// Check if outcome probabilities sum to more than 100%
    async fn check_probability_sum_arbitrage(
        &self,
        market: &Market,
    ) -> Result<Option<ArbitrageOpportunity>> {
        let mut total_cost = Decimal::ZERO;
        let mut outcomes = Vec::new();

        for outcome in &market.outcomes {
            // Get orderbook to find best prices
            let orderbook = self
                .client
                .get_orderbook(&market.id, &outcome.id)
                .await?;

            // To guarantee all outcomes, we sell at bid prices
            if let Some(bid_price) = orderbook.best_bid() {
                let size = self.config.max_size;
                let cost = bid_price * size;
                total_cost += cost;

                outcomes.push(ArbitrageOutcome {
                    outcome_id: outcome.id.clone(),
                    side: OrderSide::Sell,
                    price: bid_price,
                    size,
                    exchange: "Polymarket".to_string(),
                });
            } else {
                return Ok(None); // Can't execute if no bids
            }
        }

        // Account for fees
        let fees = total_cost * self.config.fee_rate;
        let total_cost_with_fees = total_cost + fees;

        // Calculate profit
        let payout = self.config.max_size; // One outcome will pay out
        let profit = payout - total_cost_with_fees;
        let profit_percentage = (profit / total_cost_with_fees) * Decimal::from(100);

        if profit > Decimal::ZERO && profit_percentage >= self.config.min_profit {
            debug!(
                "Found arbitrage opportunity: profit={}, profit%={}",
                profit, profit_percentage
            );

            Ok(Some(ArbitrageOpportunity {
                market_id: market.id.clone(),
                outcomes,
                total_cost: total_cost_with_fees,
                profit,
                profit_percentage,
                risk_level: self.assess_risk(profit_percentage),
            }))
        } else {
            Ok(None)
        }
    }

    /// Assess risk level of arbitrage opportunity
    pub fn assess_risk(&self, profit_percentage: Decimal) -> RiskLevel {
        if profit_percentage > Decimal::from(10) {
            RiskLevel::Low
        } else if profit_percentage > Decimal::from(5) {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        }
    }

    /// Execute arbitrage opportunity
    pub async fn execute_arbitrage(&self, opportunity: &ArbitrageOpportunity) -> Result<Vec<OrderResponse>> {
        info!("Executing arbitrage for market: {}", opportunity.market_id);

        let mut responses = Vec::new();

        for outcome in &opportunity.outcomes {
            let order_request = OrderRequest {
                market_id: opportunity.market_id.clone(),
                outcome_id: outcome.outcome_id.clone(),
                side: outcome.side,
                order_type: OrderType::Limit,
                size: outcome.size,
                price: Some(outcome.price),
                time_in_force: Some(TimeInForce::IOC), // Immediate or cancel
                client_order_id: None,
            };

            let response = self.client.place_order(order_request).await?;
            responses.push(response);
        }

        Ok(responses)
    }

    /// Calculate expected value of arbitrage
    pub fn calculate_expected_value(&self, opportunity: &ArbitrageOpportunity) -> Decimal {
        // Simple EV calculation
        // In practice, would account for execution risk, partial fills, etc.
        opportunity.profit
    }

    /// Check if arbitrage is still valid
    pub async fn validate_opportunity(&self, opportunity: &ArbitrageOpportunity) -> Result<bool> {
        for outcome in &opportunity.outcomes {
            let orderbook = self
                .client
                .get_orderbook(&opportunity.market_id, &outcome.outcome_id)
                .await?;

            let available_liquidity = match outcome.side {
                OrderSide::Buy => orderbook.total_ask_size(),
                OrderSide::Sell => orderbook.total_bid_size(),
            };

            if available_liquidity < outcome.size {
                debug!("Insufficient liquidity for outcome: {}", outcome.outcome_id);
                return Ok(false);
            }

            let best_price = match outcome.side {
                OrderSide::Buy => orderbook.best_ask(),
                OrderSide::Sell => orderbook.best_bid(),
            };

            if let Some(price) = best_price {
                // Check if price hasn't moved too much
                let price_diff = (price - outcome.price).abs();
                let tolerance = outcome.price * Decimal::new(1, 2); // 1% tolerance

                if price_diff > tolerance {
                    debug!("Price moved too much for outcome: {}", outcome.outcome_id);
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Monitor market for arbitrage opportunities
    pub async fn monitor_market(&self, market_id: &str) -> Result<()> {
        info!("Starting arbitrage monitoring for market: {}", market_id);

        loop {
            match self.check_market_arbitrage(market_id).await {
                Ok(opportunities) => {
                    for opp in opportunities {
                        info!(
                            "Found opportunity: profit={}, profit%={}",
                            opp.profit, opp.profit_percentage
                        );

                        // Validate before executing
                        if self.validate_opportunity(&opp).await? {
                            match self.execute_arbitrage(&opp).await {
                                Ok(responses) => {
                                    info!("Executed arbitrage: {} orders", responses.len());
                                }
                                Err(e) => {
                                    debug!("Failed to execute arbitrage: {}", e);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!("Error checking arbitrage: {}", e);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_secs(self.config.check_interval)).await;
        }
    }
}

/// Cross-market arbitrage detector (for multiple exchanges)
pub struct CrossMarketArbitrage {
    markets: HashMap<String, PolymarketClient>,
    #[allow(dead_code)]
    config: ArbitrageConfig,
}

impl CrossMarketArbitrage {
    pub fn new(config: ArbitrageConfig) -> Self {
        Self {
            markets: HashMap::new(),
            config,
        }
    }

    pub fn add_market(&mut self, name: String, client: PolymarketClient) {
        self.markets.insert(name, client);
    }

    /// Find arbitrage across different markets/exchanges
    pub async fn find_cross_market_opportunities(
        &self,
        _market_id: &str,
    ) -> Result<Vec<ArbitrageOpportunity>> {
        // Implementation would compare prices across different markets
        // For now, simplified version
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_risk_assessment() {
        let config = ArbitrageConfig::default();
        let client = PolymarketClient::new(crate::polymarket::client::ClientConfig::new("test")).unwrap();
        let arb = PolymarketArbitrage::new(client, config);

        assert_eq!(arb.assess_risk(dec!(15)), RiskLevel::Low);
        assert_eq!(arb.assess_risk(dec!(7)), RiskLevel::Medium);
        assert_eq!(arb.assess_risk(dec!(3)), RiskLevel::High);
    }

    #[test]
    fn test_expected_value_calculation() {
        let config = ArbitrageConfig::default();
        let client = PolymarketClient::new(crate::polymarket::client::ClientConfig::new("test")).unwrap();
        let arb = PolymarketArbitrage::new(client, config);

        let opportunity = ArbitrageOpportunity {
            market_id: "test".to_string(),
            outcomes: Vec::new(),
            total_cost: dec!(100),
            profit: dec!(5),
            profit_percentage: dec!(5),
            risk_level: RiskLevel::Medium,
        };

        assert_eq!(arb.calculate_expected_value(&opportunity), dec!(5));
    }
}
