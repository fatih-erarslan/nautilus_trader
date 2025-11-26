// Billing gateway for invoicing and payment processing

use super::{UserId, CreditAmount, ResourceUsage, ResourcePricing};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Billing period
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingPeriod {
    /// Daily billing
    Daily,

    /// Weekly billing
    Weekly,

    /// Monthly billing
    Monthly,

    /// Yearly billing
    Yearly,

    /// Pay-as-you-go (immediate)
    PayAsYouGo,
}

/// Invoice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    /// Invoice ID
    pub id: Uuid,

    /// User ID
    pub user_id: UserId,

    /// Billing period
    pub period: BillingPeriod,

    /// Period start
    pub period_start: chrono::DateTime<chrono::Utc>,

    /// Period end
    pub period_end: chrono::DateTime<chrono::Utc>,

    /// Resource usage
    pub usage: ResourceUsage,

    /// Total amount in credits
    pub total_amount: CreditAmount,

    /// Line items
    pub line_items: Vec<LineItem>,

    /// Invoice status
    pub status: InvoiceStatus,

    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Paid timestamp
    pub paid_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Line item on invoice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    /// Description
    pub description: String,

    /// Quantity
    pub quantity: f64,

    /// Unit price in credits
    pub unit_price: CreditAmount,

    /// Total price
    pub total: CreditAmount,
}

/// Invoice status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvoiceStatus {
    /// Invoice is draft
    Draft,

    /// Invoice is pending payment
    Pending,

    /// Invoice is paid
    Paid,

    /// Invoice is cancelled
    Cancelled,

    /// Invoice is overdue
    Overdue,
}

/// Billing gateway
pub struct BillingGateway {
    /// Invoices
    invoices: Arc<RwLock<HashMap<Uuid, Invoice>>>,

    /// User invoices index
    user_invoices: Arc<RwLock<HashMap<UserId, Vec<Uuid>>>>,

    /// Pricing configuration
    pricing: ResourcePricing,

    /// Default billing period
    default_period: BillingPeriod,
}

impl BillingGateway {
    /// Create new billing gateway
    pub fn new(pricing: ResourcePricing, default_period: BillingPeriod) -> Self {
        Self {
            invoices: Arc::new(RwLock::new(HashMap::new())),
            user_invoices: Arc::new(RwLock::new(HashMap::new())),
            pricing,
            default_period,
        }
    }

    /// Generate invoice from usage
    pub async fn generate_invoice(
        &self,
        user_id: UserId,
        usage: ResourceUsage,
        period: Option<BillingPeriod>,
    ) -> Result<Invoice> {
        let period = period.unwrap_or(self.default_period);
        let now = chrono::Utc::now();

        // Calculate period boundaries
        let (period_start, period_end) = self.calculate_period_boundaries(period, now);

        // Generate line items
        let mut line_items = Vec::new();

        if usage.mcp_tool_invocations > 0 {
            line_items.push(LineItem {
                description: "MCP Tool Invocations".to_string(),
                quantity: usage.mcp_tool_invocations as f64,
                unit_price: self.pricing.mcp_tool_invocation,
                total: usage.mcp_tool_invocations * self.pricing.mcp_tool_invocation,
            });
        }

        if usage.e2b_sandbox_hours > 0.0 {
            let total = (usage.e2b_sandbox_hours * self.pricing.e2b_sandbox_hour as f64) as u64;
            line_items.push(LineItem {
                description: "E2B Sandbox Hours".to_string(),
                quantity: usage.e2b_sandbox_hours,
                unit_price: self.pricing.e2b_sandbox_hour,
                total,
            });
        }

        if usage.neural_inferences > 0 {
            line_items.push(LineItem {
                description: "Neural Inferences".to_string(),
                quantity: usage.neural_inferences as f64,
                unit_price: self.pricing.neural_inference,
                total: usage.neural_inferences * self.pricing.neural_inference,
            });
        }

        if usage.data_transfer_gb > 0.0 {
            let total = (usage.data_transfer_gb * self.pricing.data_transfer_gb as f64) as u64;
            line_items.push(LineItem {
                description: "Data Transfer (GB)".to_string(),
                quantity: usage.data_transfer_gb,
                unit_price: self.pricing.data_transfer_gb,
                total,
            });
        }

        if usage.api_calls > 0 {
            let quantity = (usage.api_calls as f64) / 1000.0;
            let total = (quantity * self.pricing.api_calls_1k as f64) as u64;
            line_items.push(LineItem {
                description: "API Calls (per 1K)".to_string(),
                quantity,
                unit_price: self.pricing.api_calls_1k,
                total,
            });
        }

        if usage.agent_hours > 0.0 {
            let total = (usage.agent_hours * self.pricing.agent_hour as f64) as u64;
            line_items.push(LineItem {
                description: "Agent Hours".to_string(),
                quantity: usage.agent_hours,
                unit_price: self.pricing.agent_hour,
                total,
            });
        }

        // Calculate total
        let total_amount = usage.calculate_cost(&self.pricing);

        // Create invoice
        let invoice = Invoice {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            period,
            period_start,
            period_end,
            usage,
            total_amount,
            line_items,
            status: InvoiceStatus::Pending,
            created_at: now,
            paid_at: None,
        };

        // Store invoice
        let invoice_id = invoice.id;
        self.invoices.write().await.insert(invoice_id, invoice.clone());

        // Index by user
        self.user_invoices
            .write()
            .await
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(invoice_id);

        tracing::info!("Generated invoice {} for user {}", invoice_id, invoice.user_id);
        Ok(invoice)
    }

    /// Mark invoice as paid
    pub async fn mark_paid(&self, invoice_id: &Uuid) -> Result<()> {
        let mut invoices = self.invoices.write().await;
        let invoice = invoices
            .get_mut(invoice_id)
            .ok_or_else(|| DistributedError::PaymentError("Invoice not found".to_string()))?;

        invoice.status = InvoiceStatus::Paid;
        invoice.paid_at = Some(chrono::Utc::now());

        tracing::info!("Invoice {} marked as paid", invoice_id);
        Ok(())
    }

    /// Get invoice
    pub async fn get_invoice(&self, invoice_id: &Uuid) -> Result<Invoice> {
        self.invoices
            .read()
            .await
            .get(invoice_id)
            .cloned()
            .ok_or_else(|| DistributedError::PaymentError("Invoice not found".to_string()))
    }

    /// Get invoices for user
    pub async fn get_user_invoices(&self, user_id: &UserId) -> Vec<Invoice> {
        let user_invoices = self.user_invoices.read().await;
        let invoice_ids = user_invoices.get(user_id);

        if let Some(ids) = invoice_ids {
            let invoices = self.invoices.read().await;
            ids.iter()
                .filter_map(|id| invoices.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate period boundaries
    fn calculate_period_boundaries(
        &self,
        period: BillingPeriod,
        reference: chrono::DateTime<chrono::Utc>,
    ) -> (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>) {
        use chrono::Datelike;

        match period {
            BillingPeriod::Daily => {
                let start = reference.date_naive().and_hms_opt(0, 0, 0).unwrap();
                let end = start + chrono::Duration::days(1);
                (
                    start.and_local_timezone(chrono::Utc).unwrap(),
                    end.and_local_timezone(chrono::Utc).unwrap(),
                )
            }
            BillingPeriod::Weekly => {
                let days_since_monday = reference.weekday().num_days_from_monday();
                let start = (reference - chrono::Duration::days(days_since_monday as i64))
                    .date_naive()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                let end = start + chrono::Duration::weeks(1);
                (
                    start.and_local_timezone(chrono::Utc).unwrap(),
                    end.and_local_timezone(chrono::Utc).unwrap(),
                )
            }
            BillingPeriod::Monthly => {
                let start = reference
                    .date_naive()
                    .with_day(1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                let end = if reference.month() == 12 {
                    chrono::NaiveDate::from_ymd_opt(reference.year() + 1, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                } else {
                    chrono::NaiveDate::from_ymd_opt(reference.year(), reference.month() + 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                };
                (
                    start.and_local_timezone(chrono::Utc).unwrap(),
                    end.and_local_timezone(chrono::Utc).unwrap(),
                )
            }
            BillingPeriod::Yearly => {
                let start = chrono::NaiveDate::from_ymd_opt(reference.year(), 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                let end = chrono::NaiveDate::from_ymd_opt(reference.year() + 1, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                (
                    start.and_local_timezone(chrono::Utc).unwrap(),
                    end.and_local_timezone(chrono::Utc).unwrap(),
                )
            }
            BillingPeriod::PayAsYouGo => (reference, reference),
        }
    }

    /// Get statistics
    pub async fn stats(&self) -> BillingStats {
        let invoices = self.invoices.read().await;

        let total_invoices = invoices.len();
        let pending_invoices = invoices
            .values()
            .filter(|i| i.status == InvoiceStatus::Pending)
            .count();
        let paid_invoices = invoices
            .values()
            .filter(|i| i.status == InvoiceStatus::Paid)
            .count();
        let total_revenue: CreditAmount = invoices
            .values()
            .filter(|i| i.status == InvoiceStatus::Paid)
            .map(|i| i.total_amount)
            .sum();

        BillingStats {
            total_invoices,
            pending_invoices,
            paid_invoices,
            total_revenue,
        }
    }
}

/// Billing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingStats {
    /// Total invoices
    pub total_invoices: usize,

    /// Pending invoices
    pub pending_invoices: usize,

    /// Paid invoices
    pub paid_invoices: usize,

    /// Total revenue in credits
    pub total_revenue: CreditAmount,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_invoice_generation() {
        let gateway = BillingGateway::new(
            ResourcePricing::default(),
            BillingPeriod::Monthly,
        );

        let mut usage = ResourceUsage::default();
        usage.mcp_tool_invocations = 100;
        usage.neural_inferences = 50;

        let invoice = gateway
            .generate_invoice("user-1".to_string(), usage, None)
            .await
            .unwrap();

        assert_eq!(invoice.status, InvoiceStatus::Pending);
        assert!(invoice.total_amount > 0);
        assert_eq!(invoice.line_items.len(), 2);
    }

    #[tokio::test]
    async fn test_mark_paid() {
        let gateway = BillingGateway::new(
            ResourcePricing::default(),
            BillingPeriod::Monthly,
        );

        let invoice = gateway
            .generate_invoice("user-1".to_string(), ResourceUsage::default(), None)
            .await
            .unwrap();

        gateway.mark_paid(&invoice.id).await.unwrap();

        let updated = gateway.get_invoice(&invoice.id).await.unwrap();
        assert_eq!(updated.status, InvoiceStatus::Paid);
        assert!(updated.paid_at.is_some());
    }
}
