// Agentic-payments integration for credit-based resource management

mod credits;
mod billing;
mod gateway;

pub use credits::{CreditSystem, CreditAccount, Transaction};
pub use billing::{BillingGateway, Invoice, BillingPeriod};
pub use gateway::{PaymentGateway, PaymentMethod, PaymentResult};

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// User ID type
pub type UserId = String;

/// Credit amount type
pub type CreditAmount = u64;

/// Pricing for different resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePricing {
    /// Credits per MCP tool invocation
    pub mcp_tool_invocation: CreditAmount,

    /// Credits per E2B sandbox hour
    pub e2b_sandbox_hour: CreditAmount,

    /// Credits per neural model inference
    pub neural_inference: CreditAmount,

    /// Credits per GB of data transfer
    pub data_transfer_gb: CreditAmount,

    /// Credits per 1K API calls
    pub api_calls_1k: CreditAmount,

    /// Credits per agent hour
    pub agent_hour: CreditAmount,
}

impl Default for ResourcePricing {
    fn default() -> Self {
        Self {
            mcp_tool_invocation: 1,
            e2b_sandbox_hour: 100,
            neural_inference: 10,
            data_transfer_gb: 50,
            api_calls_1k: 20,
            agent_hour: 80,
        }
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// MCP tool invocations
    pub mcp_tool_invocations: u64,

    /// E2B sandbox hours
    pub e2b_sandbox_hours: f64,

    /// Neural inferences
    pub neural_inferences: u64,

    /// Data transfer (GB)
    pub data_transfer_gb: f64,

    /// API calls
    pub api_calls: u64,

    /// Agent hours
    pub agent_hours: f64,
}

impl ResourceUsage {
    /// Calculate total cost in credits
    pub fn calculate_cost(&self, pricing: &ResourcePricing) -> CreditAmount {
        let mut total = 0;

        total += self.mcp_tool_invocations * pricing.mcp_tool_invocation;
        total += (self.e2b_sandbox_hours * pricing.e2b_sandbox_hour as f64) as u64;
        total += self.neural_inferences * pricing.neural_inference;
        total += (self.data_transfer_gb * pricing.data_transfer_gb as f64) as u64;
        total += (self.api_calls / 1000) * pricing.api_calls_1k;
        total += (self.agent_hours * pricing.agent_hour as f64) as u64;

        total
    }

    /// Merge with another usage
    pub fn merge(&mut self, other: &ResourceUsage) {
        self.mcp_tool_invocations += other.mcp_tool_invocations;
        self.e2b_sandbox_hours += other.e2b_sandbox_hours;
        self.neural_inferences += other.neural_inferences;
        self.data_transfer_gb += other.data_transfer_gb;
        self.api_calls += other.api_calls;
        self.agent_hours += other.agent_hours;
    }
}

/// Usage tracker for recording resource consumption
#[derive(Debug)]
pub struct UsageTracker {
    /// User usage map
    usage: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<UserId, ResourceUsage>>>,

    /// Pricing configuration
    pricing: ResourcePricing,
}

impl UsageTracker {
    /// Create new usage tracker
    pub fn new(pricing: ResourcePricing) -> Self {
        Self {
            usage: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            pricing,
        }
    }

    /// Record MCP tool invocation
    pub async fn record_mcp_invocation(&self, user_id: UserId) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).mcp_tool_invocations += 1;
    }

    /// Record E2B sandbox usage
    pub async fn record_sandbox_usage(&self, user_id: UserId, hours: f64) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).e2b_sandbox_hours += hours;
    }

    /// Record neural inference
    pub async fn record_neural_inference(&self, user_id: UserId) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).neural_inferences += 1;
    }

    /// Record data transfer
    pub async fn record_data_transfer(&self, user_id: UserId, gb: f64) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).data_transfer_gb += gb;
    }

    /// Record API call
    pub async fn record_api_call(&self, user_id: UserId) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).api_calls += 1;
    }

    /// Record agent usage
    pub async fn record_agent_usage(&self, user_id: UserId, hours: f64) {
        let mut usage = self.usage.write().await;
        usage.entry(user_id).or_insert_with(ResourceUsage::default).agent_hours += hours;
    }

    /// Get usage for user
    pub async fn get_usage(&self, user_id: &UserId) -> Option<ResourceUsage> {
        self.usage.read().await.get(user_id).cloned()
    }

    /// Get cost for user
    pub async fn get_cost(&self, user_id: &UserId) -> CreditAmount {
        self.usage
            .read()
            .await
            .get(user_id)
            .map(|u| u.calculate_cost(&self.pricing))
            .unwrap_or(0)
    }

    /// Reset usage for user
    pub async fn reset_usage(&self, user_id: &UserId) {
        self.usage.write().await.remove(user_id);
    }

    /// Get all usage
    pub async fn get_all_usage(&self) -> std::collections::HashMap<UserId, ResourceUsage> {
        self.usage.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_pricing_default() {
        let pricing = ResourcePricing::default();
        assert_eq!(pricing.mcp_tool_invocation, 1);
        assert_eq!(pricing.e2b_sandbox_hour, 100);
    }

    #[test]
    fn test_resource_usage_cost_calculation() {
        let mut usage = ResourceUsage::default();
        usage.mcp_tool_invocations = 100;
        usage.e2b_sandbox_hours = 2.0;
        usage.neural_inferences = 50;

        let pricing = ResourcePricing::default();
        let cost = usage.calculate_cost(&pricing);

        // 100*1 + 2*100 + 50*10 = 100 + 200 + 500 = 800
        assert_eq!(cost, 800);
    }

    #[test]
    fn test_resource_usage_merge() {
        let mut usage1 = ResourceUsage::default();
        usage1.mcp_tool_invocations = 50;
        usage1.api_calls = 1000;

        let mut usage2 = ResourceUsage::default();
        usage2.mcp_tool_invocations = 30;
        usage2.api_calls = 500;

        usage1.merge(&usage2);

        assert_eq!(usage1.mcp_tool_invocations, 80);
        assert_eq!(usage1.api_calls, 1500);
    }

    #[tokio::test]
    async fn test_usage_tracker() {
        let tracker = UsageTracker::new(ResourcePricing::default());

        tracker.record_mcp_invocation("user-1".to_string()).await;
        tracker.record_mcp_invocation("user-1".to_string()).await;
        tracker.record_api_call("user-1".to_string()).await;

        let usage = tracker.get_usage(&"user-1".to_string()).await.unwrap();
        assert_eq!(usage.mcp_tool_invocations, 2);
        assert_eq!(usage.api_calls, 1);
    }
}
