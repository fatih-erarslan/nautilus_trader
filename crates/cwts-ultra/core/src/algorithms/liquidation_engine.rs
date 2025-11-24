use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum LiquidationError {
    #[error("Insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: f64, available: f64 },
    #[error("Invalid margin mode: {0}")]
    InvalidMarginMode(String),
    #[error("Liquidation price calculation failed: {0}")]
    CalculationFailed(String),
    #[error("Position not found: {0}")]
    PositionNotFound(String),
    #[error("Atomic operation failed: {0}")]
    AtomicOperationFailed(String),
    #[error("Margin call triggered for position: {0}")]
    MarginCall(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarginMode {
    Cross,    // Cross margin - shares margin across all positions
    Isolated, // Isolated margin - dedicated margin per position
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginPosition {
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub leverage: f64,
    pub margin_mode: MarginMode,
    pub initial_margin: f64,
    pub maintenance_margin: f64,
    pub unrealized_pnl: f64,
    pub liquidation_price: f64,
    pub margin_ratio: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginAccount {
    pub total_balance: f64,
    pub available_balance: f64,
    pub used_margin: f64,
    pub maintenance_margin: f64,
    pub unrealized_pnl: f64,
    pub margin_level: f64,
    pub positions: HashMap<String, MarginPosition>,
    pub margin_call_level: f64,
    pub liquidation_level: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct LiquidationParameters {
    pub initial_margin_rate: f64, // Initial margin requirement (e.g., 0.1 for 10%)
    pub maintenance_margin_rate: f64, // Maintenance margin rate (e.g., 0.05 for 5%)
    pub liquidation_buffer: f64,  // Buffer before liquidation (e.g., 0.01 for 1%)
    pub margin_call_threshold: f64, // Margin call trigger level (e.g., 1.2 for 120%)
    pub liquidation_threshold: f64, // Liquidation trigger level (e.g., 1.05 for 105%)
    pub max_leverage: f64,        // Maximum allowed leverage
    pub funding_rate: f64,        // Funding rate for perpetual contracts
    pub mark_price_premium: f64,  // Premium for mark price calculation
}

impl Default for LiquidationParameters {
    fn default() -> Self {
        Self {
            initial_margin_rate: 0.10,     // 10%
            maintenance_margin_rate: 0.05, // 5%
            liquidation_buffer: 0.01,      // 1%
            margin_call_threshold: 1.20,   // 120%
            liquidation_threshold: 1.05,   // 105%
            max_leverage: 100.0,
            funding_rate: 0.0001,       // 0.01% per hour
            mark_price_premium: 0.0005, // 0.05% premium
        }
    }
}

pub struct LiquidationEngine {
    pub(crate) parameters: LiquidationParameters,
    pub(crate) accounts: Arc<RwLock<HashMap<String, MarginAccount>>>,
    pub(crate) price_feed: Arc<RwLock<HashMap<String, f64>>>,
    pub(crate) liquidation_queue: Arc<Mutex<Vec<(String, String)>>>, // (account_id, symbol)
}

impl LiquidationEngine {
    pub fn new(parameters: LiquidationParameters) -> Self {
        Self {
            parameters,
            accounts: Arc::new(RwLock::new(HashMap::new())),
            price_feed: Arc::new(RwLock::new(HashMap::new())),
            liquidation_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Calculate initial margin requirement
    pub fn calculate_initial_margin(
        &self,
        position_size: f64,
        entry_price: f64,
        leverage: f64,
    ) -> Result<f64, LiquidationError> {
        if leverage <= 0.0 || leverage > self.parameters.max_leverage {
            return Err(LiquidationError::CalculationFailed(format!(
                "Invalid leverage: {}",
                leverage
            )));
        }

        let notional_value = position_size.abs() * entry_price;
        let initial_margin = notional_value / leverage;

        // Additional margin based on initial margin rate
        let required_margin = initial_margin * (1.0 + self.parameters.initial_margin_rate);

        Ok(required_margin)
    }

    /// Calculate maintenance margin requirement
    pub fn calculate_maintenance_margin(&self, position_size: f64, current_price: f64) -> f64 {
        let notional_value = position_size.abs() * current_price;
        notional_value * self.parameters.maintenance_margin_rate
    }

    /// Calculate liquidation price for isolated margin
    pub fn calculate_liquidation_price_isolated(
        &self,
        position: &MarginPosition,
    ) -> Result<f64, LiquidationError> {
        let is_long = position.size > 0.0;
        let abs_size = position.size.abs();

        if abs_size == 0.0 {
            return Err(LiquidationError::CalculationFailed(
                "Position size cannot be zero".to_string(),
            ));
        }

        // For isolated margin, liquidation occurs when:
        // Margin Balance + PnL = Maintenance Margin

        let maintenance_margin =
            self.calculate_maintenance_margin(position.size, position.current_price);

        let liquidation_price = if is_long {
            // Long position: liquidation when price falls
            // entry_price - (initial_margin - maintenance_margin) / size
            let price_diff = (position.initial_margin - maintenance_margin) / abs_size;
            position.entry_price - price_diff
        } else {
            // Short position: liquidation when price rises
            // entry_price + (initial_margin - maintenance_margin) / size
            let price_diff = (position.initial_margin - maintenance_margin) / abs_size;
            position.entry_price + price_diff
        };

        Ok(liquidation_price.max(0.0)) // Price cannot be negative
    }

    /// Calculate liquidation price for cross margin
    pub async fn calculate_liquidation_price_cross(
        &self,
        account_id: &str,
        symbol: &str,
    ) -> Result<f64, LiquidationError> {
        let accounts = self.accounts.read().await;
        let account = accounts
            .get(account_id)
            .ok_or_else(|| LiquidationError::PositionNotFound(account_id.to_string()))?;

        let position = account
            .positions
            .get(symbol)
            .ok_or_else(|| LiquidationError::PositionNotFound(symbol.to_string()))?;

        if position.margin_mode != MarginMode::Cross {
            return Err(LiquidationError::InvalidMarginMode(
                "Expected cross margin mode".to_string(),
            ));
        }

        // Cross margin calculation is more complex as it involves all positions
        let total_maintenance_margin = account.maintenance_margin;
        let available_balance = account.available_balance + account.unrealized_pnl;

        // Simplified cross margin liquidation price calculation
        // In reality, this would require solving a system of equations
        let is_long = position.size > 0.0;
        let abs_size = position.size.abs();

        let liquidation_price = if is_long {
            let required_balance_change = total_maintenance_margin - available_balance;
            position.current_price - (required_balance_change / abs_size)
        } else {
            let required_balance_change = total_maintenance_margin - available_balance;
            position.current_price + (required_balance_change / abs_size)
        };

        Ok(liquidation_price.max(0.0))
    }

    /// Calculate liquidation price for cross margin while holding account lock
    /// CRITICAL: This function must be called while holding the write lock to prevent TOCTOU race conditions
    ///
    /// # Race Condition Fix (CVSS 9.8)
    /// Previously, we would:
    /// 1. Drop write lock
    /// 2. Call calculate_liquidation_price_cross (acquires read lock)
    /// 3. Re-acquire write lock
    ///
    /// This created a window where two honest Byzantine nodes could commit different values
    /// because another thread could modify account state between steps 1-3.
    ///
    /// This locked version ensures atomic read-calculate-update operations.
    fn calculate_liquidation_price_cross_locked(
        &self,
        account: &MarginAccount,
        symbol: &str,
    ) -> Result<f64, LiquidationError> {
        let position = account
            .positions
            .get(symbol)
            .ok_or_else(|| LiquidationError::PositionNotFound(symbol.to_string()))?;

        if position.margin_mode != MarginMode::Cross {
            return Err(LiquidationError::InvalidMarginMode(
                "Expected cross margin mode".to_string(),
            ));
        }

        // Cross margin calculation is more complex as it involves all positions
        let total_maintenance_margin = account.maintenance_margin;
        let available_balance = account.available_balance + account.unrealized_pnl;

        // Simplified cross margin liquidation price calculation
        // In reality, this would require solving a system of equations
        let is_long = position.size > 0.0;
        let abs_size = position.size.abs();

        let liquidation_price = if is_long {
            let required_balance_change = total_maintenance_margin - available_balance;
            position.current_price - (required_balance_change / abs_size)
        } else {
            let required_balance_change = total_maintenance_margin - available_balance;
            position.current_price + (required_balance_change / abs_size)
        };

        Ok(liquidation_price.max(0.0))
    }

    /// Update position with new price and check margin requirements
    pub async fn update_position_margin(
        &self,
        account_id: &str,
        symbol: &str,
        current_price: f64,
    ) -> Result<(), LiquidationError> {
        let mut accounts = self.accounts.write().await;
        let account = accounts
            .get_mut(account_id)
            .ok_or_else(|| LiquidationError::PositionNotFound(account_id.to_string()))?;

        // First, determine margin mode and calculate cross margin price if needed
        // This avoids borrow conflicts when updating position
        let margin_mode = account
            .positions
            .get(symbol)
            .ok_or_else(|| LiquidationError::PositionNotFound(symbol.to_string()))?
            .margin_mode
            .clone();

        let cross_liq_price = if matches!(margin_mode, MarginMode::Cross) {
            Some(self.calculate_liquidation_price_cross_locked(&*account, symbol)?)
        } else {
            None
        };

        // Now get mutable reference to position
        let position = account
            .positions
            .get_mut(symbol)
            .ok_or_else(|| LiquidationError::PositionNotFound(symbol.to_string()))?;

        // Update position with new price
        position.current_price = current_price;
        position.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Calculate unrealized PnL
        let is_long = position.size > 0.0;
        position.unrealized_pnl = if is_long {
            position.size * (current_price - position.entry_price)
        } else {
            position.size * (position.entry_price - current_price)
        };

        // Update maintenance margin
        position.maintenance_margin =
            self.calculate_maintenance_margin(position.size, current_price);

        // Calculate margin ratio
        let equity = position.initial_margin + position.unrealized_pnl;
        position.margin_ratio = if position.maintenance_margin > 0.0 {
            equity / position.maintenance_margin
        } else {
            f64::INFINITY
        };

        // Update liquidation price
        // CRITICAL FIX: Use locked version to prevent TOCTOU race condition (CVSS 9.8)
        // We calculated cross margin price before getting mutable position borrow
        match margin_mode {
            MarginMode::Isolated => {
                position.liquidation_price = self.calculate_liquidation_price_isolated(position)?;
            }
            MarginMode::Cross => {
                // Use pre-calculated price to avoid borrow conflict
                position.liquidation_price = cross_liq_price.unwrap();
            }
        }

        // Check for margin call or liquidation
        self.check_margin_requirements(account_id, symbol, &*position)
            .await?;

        Ok(())
    }

    /// Check margin requirements and trigger margin call or liquidation
    async fn check_margin_requirements(
        &self,
        account_id: &str,
        symbol: &str,
        position: &MarginPosition,
    ) -> Result<(), LiquidationError> {
        // Check for liquidation threshold
        if position.margin_ratio <= self.parameters.liquidation_threshold {
            self.trigger_liquidation(account_id, symbol).await?;
            return Err(LiquidationError::InsufficientMargin {
                required: position.maintenance_margin,
                available: position.initial_margin + position.unrealized_pnl,
            });
        }

        // Check for margin call threshold
        if position.margin_ratio <= self.parameters.margin_call_threshold {
            return Err(LiquidationError::MarginCall(symbol.to_string()));
        }

        Ok(())
    }

    /// Execute forced liquidation
    pub async fn trigger_liquidation(
        &self,
        account_id: &str,
        symbol: &str,
    ) -> Result<(), LiquidationError> {
        // Add to liquidation queue for atomic processing
        {
            let mut queue = self
                .liquidation_queue
                .lock()
                .map_err(|e| LiquidationError::AtomicOperationFailed(e.to_string()))?;
            queue.push((account_id.to_string(), symbol.to_string()));
        }

        // Execute liquidation atomically
        self.execute_liquidation(account_id, symbol).await
    }

    /// Execute liquidation atomically
    async fn execute_liquidation(
        &self,
        account_id: &str,
        symbol: &str,
    ) -> Result<(), LiquidationError> {
        let mut accounts = self.accounts.write().await;
        let account = accounts
            .get_mut(account_id)
            .ok_or_else(|| LiquidationError::PositionNotFound(account_id.to_string()))?;

        let position = account
            .positions
            .get_mut(symbol)
            .ok_or_else(|| LiquidationError::PositionNotFound(symbol.to_string()))?;

        // Calculate liquidation price and realized loss
        let liquidation_price = position.liquidation_price;
        let position_size = position.size;

        // Realize the loss
        let realized_loss = if position_size > 0.0 {
            // Long position
            position_size * (liquidation_price - position.entry_price)
        } else {
            // Short position
            position_size * (position.entry_price - liquidation_price)
        };

        // Update account balance
        account.total_balance += realized_loss; // This will be negative for a loss
        account.available_balance =
            account.total_balance - account.used_margin + position.initial_margin;
        account.used_margin -= position.initial_margin;

        // Remove liquidated position
        account.positions.remove(symbol);

        // Update account timestamp
        account.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(())
    }

    /// Create new margin position
    pub async fn create_position(
        &self,
        account_id: &str,
        symbol: &str,
        size: f64,
        entry_price: f64,
        leverage: f64,
        margin_mode: MarginMode,
    ) -> Result<(), LiquidationError> {
        let initial_margin = self.calculate_initial_margin(size, entry_price, leverage)?;
        let maintenance_margin = self.calculate_maintenance_margin(size, entry_price);

        let mut position = MarginPosition {
            symbol: symbol.to_string(),
            size,
            entry_price,
            current_price: entry_price,
            leverage,
            margin_mode: margin_mode.clone(),
            initial_margin,
            maintenance_margin,
            unrealized_pnl: 0.0,
            liquidation_price: 0.0,
            margin_ratio: f64::INFINITY,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        // Calculate liquidation price
        position.liquidation_price = match margin_mode {
            MarginMode::Isolated => self.calculate_liquidation_price_isolated(&position)?,
            MarginMode::Cross => {
                // For new cross margin positions, use a simplified calculation
                let is_long = size > 0.0;
                if is_long {
                    entry_price * (1.0 - 1.0 / leverage + self.parameters.maintenance_margin_rate)
                } else {
                    entry_price * (1.0 + 1.0 / leverage - self.parameters.maintenance_margin_rate)
                }
            }
        };

        // Update account
        let mut accounts = self.accounts.write().await;
        let account = accounts
            .entry(account_id.to_string())
            .or_insert_with(|| MarginAccount {
                total_balance: 0.0,
                available_balance: 0.0,
                used_margin: 0.0,
                maintenance_margin: 0.0,
                unrealized_pnl: 0.0,
                margin_level: f64::INFINITY,
                positions: HashMap::new(),
                margin_call_level: self.parameters.margin_call_threshold,
                liquidation_level: self.parameters.liquidation_threshold,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            });

        // Check if sufficient margin available
        if account.available_balance < initial_margin {
            return Err(LiquidationError::InsufficientMargin {
                required: initial_margin,
                available: account.available_balance,
            });
        }

        // Update account balances
        account.used_margin += initial_margin;
        account.available_balance -= initial_margin;
        account.maintenance_margin += maintenance_margin;

        // Add position
        account.positions.insert(symbol.to_string(), position);

        // Update account timestamp
        account.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(())
    }

    /// Get margin account information
    pub async fn get_account_info(&self, account_id: &str) -> Option<MarginAccount> {
        let accounts = self.accounts.read().await;
        accounts.get(account_id).cloned()
    }

    /// Calculate funding payment for perpetual contracts
    pub fn calculate_funding_payment(
        &self,
        position_size: f64,
        mark_price: f64,
        funding_rate: f64,
    ) -> f64 {
        // Funding payment = position_size * mark_price * funding_rate
        position_size * mark_price * funding_rate
    }

    /// Update price feed
    pub async fn update_price(&self, symbol: &str, price: f64) {
        let mut prices = self.price_feed.write().await;
        prices.insert(symbol.to_string(), price);
    }

    /// Get mark price with premium
    pub async fn get_mark_price(&self, symbol: &str) -> Option<f64> {
        let prices = self.price_feed.read().await;
        prices
            .get(symbol)
            .map(|price| price * (1.0 + self.parameters.mark_price_premium))
    }

    /// Process liquidation queue atomically
    pub async fn process_liquidation_queue(
        &self,
    ) -> Result<Vec<(String, String)>, LiquidationError> {
        let mut processed = Vec::new();

        // Process all pending liquidations atomically
        loop {
            let next_liquidation = {
                let mut queue = self
                    .liquidation_queue
                    .lock()
                    .map_err(|e| LiquidationError::AtomicOperationFailed(e.to_string()))?;
                queue.pop()
            };

            match next_liquidation {
                Some((account_id, symbol)) => {
                    if let Err(e) = self.execute_liquidation(&account_id, &symbol).await {
                        // Log error but continue processing
                        eprintln!("Liquidation failed for {}:{} - {}", account_id, symbol, e);
                    } else {
                        processed.push((account_id, symbol));
                    }
                }
                None => break,
            }
        }

        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_initial_margin_calculation() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let margin = engine.calculate_initial_margin(1.0, 50000.0, 10.0).unwrap();

        // Position value: 1.0 * 50000 = 50000
        // Base margin: 50000 / 10 = 5000
        // With 10% initial margin rate: 5000 * 1.1 = 5500
        assert_eq!(margin, 5500.0);
    }

    #[tokio::test]
    async fn test_liquidation_price_isolated() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());

        let position = MarginPosition {
            symbol: "BTC".to_string(),
            size: 1.0, // Long position
            entry_price: 50000.0,
            current_price: 50000.0,
            leverage: 10.0,
            margin_mode: MarginMode::Isolated,
            initial_margin: 5500.0,
            maintenance_margin: 2500.0,
            unrealized_pnl: 0.0,
            liquidation_price: 0.0,
            margin_ratio: f64::INFINITY,
            timestamp: 0,
        };

        let liq_price = engine
            .calculate_liquidation_price_isolated(&position)
            .unwrap();

        // Long position liquidation price should be below entry price
        assert!(liq_price < position.entry_price);
    }

    #[tokio::test]
    async fn test_position_creation() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());

        // First, add some balance to the account
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                "test_user".to_string(),
                MarginAccount {
                    total_balance: 10000.0,
                    available_balance: 10000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: 0,
                },
            );
        }

        let result = engine
            .create_position("test_user", "BTC", 1.0, 50000.0, 10.0, MarginMode::Isolated)
            .await;

        assert!(result.is_ok());

        let account = engine.get_account_info("test_user").await.unwrap();
        assert!(account.positions.contains_key("BTC"));
        assert!(account.used_margin > 0.0);
    }
}
