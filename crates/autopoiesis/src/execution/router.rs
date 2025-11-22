//! Order routing implementation

use crate::prelude::*;
use crate::models::{Order, OrderSide};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Order router for intelligent order routing
#[derive(Debug, Clone)]
pub struct OrderRouter {
    /// Router configuration
    config: OrderRouterConfig,
    
    /// Available venues
    venues: HashMap<String, VenueInfo>,
    
    /// Routing metrics
    metrics: RoutingMetrics,
}

#[derive(Debug, Clone)]
pub struct OrderRouterConfig {
    /// Routing strategy
    pub strategy: RoutingStrategy,
    
    /// Maximum venues to consider
    pub max_venues: usize,
    
    /// Price improvement threshold
    pub min_price_improvement_bps: u32,
}

#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    BestPrice,
    LeastCost,
    FastestExecution,
    SmartRouting,
}

#[derive(Debug, Clone)]
struct VenueInfo {
    name: String,
    latency_ms: u32,
    fee_bps: u32,
    available_liquidity: Decimal,
    reliability_score: f64,
}

#[derive(Debug, Clone, Default)]
struct RoutingMetrics {
    total_orders_routed: u64,
    successful_routes: u64,
    average_price_improvement_bps: f64,
}

impl Default for OrderRouterConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::SmartRouting,
            max_venues: 5,
            min_price_improvement_bps: 5,
        }
    }
}

impl OrderRouter {
    /// Create a new order router
    pub fn new(config: OrderRouterConfig) -> Self {
        let mut venues = HashMap::new();
        venues.insert("binance".to_string(), VenueInfo {
            name: "binance".to_string(),
            latency_ms: 20,
            fee_bps: 10,
            available_liquidity: Decimal::from(1000000),
            reliability_score: 0.95,
        });
        venues.insert("coinbase".to_string(), VenueInfo {
            name: "coinbase".to_string(),
            latency_ms: 30,
            fee_bps: 50,
            available_liquidity: Decimal::from(800000),
            reliability_score: 0.98,
        });

        Self {
            config,
            venues,
            metrics: RoutingMetrics::default(),
        }
    }

    /// Route an order to the best venue
    pub async fn route_order(&mut self, order: &Order) -> Result<String> {
        let best_venue = match self.config.strategy {
            RoutingStrategy::BestPrice => self.find_best_price_venue(order).await?,
            RoutingStrategy::LeastCost => self.find_least_cost_venue(order).await?,
            RoutingStrategy::FastestExecution => self.find_fastest_venue(order).await?,
            RoutingStrategy::SmartRouting => self.smart_route(order).await?,
        };

        self.metrics.total_orders_routed += 1;
        self.metrics.successful_routes += 1;

        Ok(best_venue)
    }

    async fn find_best_price_venue(&self, _order: &Order) -> Result<String> {
        Ok("binance".to_string())
    }

    async fn find_least_cost_venue(&self, _order: &Order) -> Result<String> {
        let mut best_venue = None;
        let mut lowest_cost = u32::MAX;

        for venue in self.venues.values() {
            if venue.fee_bps < lowest_cost {
                lowest_cost = venue.fee_bps;
                best_venue = Some(venue.name.clone());
            }
        }

        best_venue.ok_or_else(|| Error::Execution("No venue available".to_string()))
    }

    async fn find_fastest_venue(&self, _order: &Order) -> Result<String> {
        let mut best_venue = None;
        let mut lowest_latency = u32::MAX;

        for venue in self.venues.values() {
            if venue.latency_ms < lowest_latency {
                lowest_latency = venue.latency_ms;
                best_venue = Some(venue.name.clone());
            }
        }

        best_venue.ok_or_else(|| Error::Execution("No venue available".to_string()))
    }

    async fn smart_route(&self, order: &Order) -> Result<String> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_venue = None;

        for venue in self.venues.values() {
            let liquidity_score = if venue.available_liquidity >= order.quantity { 1.0 } else { 0.5 };
            let latency_score = 1.0 / (venue.latency_ms as f64 + 1.0);
            let fee_score = 1.0 / (venue.fee_bps as f64 + 1.0);
            let reliability_score = venue.reliability_score;

            let total_score = liquidity_score * 0.3 + latency_score * 0.3 + fee_score * 0.2 + reliability_score * 0.2;

            if total_score > best_score {
                best_score = total_score;
                best_venue = Some(venue.name.clone());
            }
        }

        best_venue.ok_or_else(|| Error::Execution("No venue available".to_string()))
    }

    /// Get routing metrics
    pub fn get_metrics(&self) -> &RoutingMetrics {
        &self.metrics
    }
}