// Credit system for resource allocation

use super::{UserId, CreditAmount};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Credit account for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditAccount {
    /// User ID
    pub user_id: UserId,

    /// Current balance
    pub balance: CreditAmount,

    /// Total credits purchased
    pub total_purchased: CreditAmount,

    /// Total credits spent
    pub total_spent: CreditAmount,

    /// Account created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last transaction timestamp
    pub last_transaction: chrono::DateTime<chrono::Utc>,

    /// Account status
    pub status: AccountStatus,
}

/// Account status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountStatus {
    /// Account is active
    Active,

    /// Account is suspended
    Suspended,

    /// Account is closed
    Closed,
}

impl CreditAccount {
    /// Create new account
    pub fn new(user_id: UserId, initial_balance: CreditAmount) -> Self {
        let now = chrono::Utc::now();
        Self {
            user_id,
            balance: initial_balance,
            total_purchased: initial_balance,
            total_spent: 0,
            created_at: now,
            last_transaction: now,
            status: AccountStatus::Active,
        }
    }

    /// Add credits
    pub fn add_credits(&mut self, amount: CreditAmount) {
        self.balance += amount;
        self.total_purchased += amount;
        self.last_transaction = chrono::Utc::now();
    }

    /// Deduct credits
    pub fn deduct_credits(&mut self, amount: CreditAmount) -> Result<()> {
        if self.balance < amount {
            return Err(DistributedError::InsufficientCredits {
                needed: amount,
                available: self.balance,
            });
        }

        self.balance -= amount;
        self.total_spent += amount;
        self.last_transaction = chrono::Utc::now();
        Ok(())
    }

    /// Check if account can afford amount
    pub fn can_afford(&self, amount: CreditAmount) -> bool {
        self.balance >= amount && self.status == AccountStatus::Active
    }
}

/// Transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction ID
    pub id: Uuid,

    /// User ID
    pub user_id: UserId,

    /// Transaction type
    pub transaction_type: TransactionType,

    /// Amount
    pub amount: CreditAmount,

    /// Balance after transaction
    pub balance_after: CreditAmount,

    /// Description
    pub description: String,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Transaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionType {
    /// Purchase credits
    Purchase,

    /// Spend credits
    Spend,

    /// Refund credits
    Refund,

    /// Transfer credits
    Transfer,

    /// Bonus credits
    Bonus,
}

/// Credit system
pub struct CreditSystem {
    /// User accounts
    accounts: Arc<RwLock<HashMap<UserId, CreditAccount>>>,

    /// Transaction history
    transactions: Arc<RwLock<Vec<Transaction>>>,

    /// Default credit allocation for new users
    default_credits: CreditAmount,
}

impl CreditSystem {
    /// Create new credit system
    pub fn new(default_credits: CreditAmount) -> Self {
        Self {
            accounts: Arc::new(RwLock::new(HashMap::new())),
            transactions: Arc::new(RwLock::new(Vec::new())),
            default_credits,
        }
    }

    /// Create account for user
    pub async fn create_account(&self, user_id: UserId) -> Result<CreditAccount> {
        let mut accounts = self.accounts.write().await;

        if accounts.contains_key(&user_id) {
            return Err(DistributedError::PaymentError(
                "Account already exists".to_string(),
            ));
        }

        let account = CreditAccount::new(user_id.clone(), self.default_credits);
        accounts.insert(user_id.clone(), account.clone());

        // Record transaction
        self.record_transaction(Transaction {
            id: Uuid::new_v4(),
            user_id,
            transaction_type: TransactionType::Bonus,
            amount: self.default_credits,
            balance_after: self.default_credits,
            description: "Initial credit allocation".to_string(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
        .await;

        Ok(account)
    }

    /// Get account
    pub async fn get_account(&self, user_id: &UserId) -> Result<CreditAccount> {
        self.accounts
            .read()
            .await
            .get(user_id)
            .cloned()
            .ok_or_else(|| {
                DistributedError::PaymentError(format!("Account not found: {}", user_id))
            })
    }

    /// Add credits to account
    pub async fn add_credits(
        &self,
        user_id: &UserId,
        amount: CreditAmount,
        transaction_type: TransactionType,
        description: String,
    ) -> Result<CreditAmount> {
        let mut accounts = self.accounts.write().await;
        let account = accounts.get_mut(user_id).ok_or_else(|| {
            DistributedError::PaymentError(format!("Account not found: {}", user_id))
        })?;

        account.add_credits(amount);
        let new_balance = account.balance;

        drop(accounts);

        // Record transaction
        self.record_transaction(Transaction {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            transaction_type,
            amount,
            balance_after: new_balance,
            description,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
        .await;

        Ok(new_balance)
    }

    /// Deduct credits from account
    pub async fn deduct_credits(
        &self,
        user_id: &UserId,
        amount: CreditAmount,
        description: String,
    ) -> Result<CreditAmount> {
        let mut accounts = self.accounts.write().await;
        let account = accounts.get_mut(user_id).ok_or_else(|| {
            DistributedError::PaymentError(format!("Account not found: {}", user_id))
        })?;

        account.deduct_credits(amount)?;
        let new_balance = account.balance;

        drop(accounts);

        // Record transaction
        self.record_transaction(Transaction {
            id: Uuid::new_v4(),
            user_id: user_id.clone(),
            transaction_type: TransactionType::Spend,
            amount,
            balance_after: new_balance,
            description,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
        .await;

        Ok(new_balance)
    }

    /// Check if user can afford amount
    pub async fn can_afford(&self, user_id: &UserId, amount: CreditAmount) -> bool {
        self.accounts
            .read()
            .await
            .get(user_id)
            .map(|a| a.can_afford(amount))
            .unwrap_or(false)
    }

    /// Get balance
    pub async fn get_balance(&self, user_id: &UserId) -> Result<CreditAmount> {
        Ok(self.get_account(user_id).await?.balance)
    }

    /// Get transaction history
    pub async fn get_transactions(
        &self,
        user_id: &UserId,
        limit: Option<usize>,
    ) -> Vec<Transaction> {
        let transactions = self.transactions.read().await;
        let user_transactions: Vec<_> = transactions
            .iter()
            .filter(|t| &t.user_id == user_id)
            .cloned()
            .collect();

        if let Some(limit) = limit {
            user_transactions.into_iter().rev().take(limit).collect()
        } else {
            user_transactions.into_iter().rev().collect()
        }
    }

    /// Record transaction
    async fn record_transaction(&self, transaction: Transaction) {
        self.transactions.write().await.push(transaction);
    }

    /// Get statistics
    pub async fn stats(&self) -> CreditSystemStats {
        let accounts = self.accounts.read().await;
        let transactions = self.transactions.read().await;

        let total_accounts = accounts.len();
        let total_balance: CreditAmount = accounts.values().map(|a| a.balance).sum();
        let total_spent: CreditAmount = accounts.values().map(|a| a.total_spent).sum();
        let total_transactions = transactions.len();

        CreditSystemStats {
            total_accounts,
            total_balance,
            total_spent,
            total_transactions,
        }
    }
}

/// Credit system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditSystemStats {
    /// Total number of accounts
    pub total_accounts: usize,

    /// Total balance across all accounts
    pub total_balance: CreditAmount,

    /// Total credits spent
    pub total_spent: CreditAmount,

    /// Total transactions
    pub total_transactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_account_creation() {
        let account = CreditAccount::new("user-1".to_string(), 1000);
        assert_eq!(account.balance, 1000);
        assert_eq!(account.total_purchased, 1000);
        assert_eq!(account.status, AccountStatus::Active);
    }

    #[test]
    fn test_credit_operations() {
        let mut account = CreditAccount::new("user-1".to_string(), 1000);

        account.add_credits(500);
        assert_eq!(account.balance, 1500);

        assert!(account.deduct_credits(200).is_ok());
        assert_eq!(account.balance, 1300);

        assert!(account.deduct_credits(2000).is_err());
    }

    #[tokio::test]
    async fn test_credit_system() {
        let system = CreditSystem::new(1000);

        let account = system.create_account("user-1".to_string()).await.unwrap();
        assert_eq!(account.balance, 1000);

        let new_balance = system
            .add_credits(
                &"user-1".to_string(),
                500,
                TransactionType::Purchase,
                "Test purchase".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(new_balance, 1500);

        assert!(system.can_afford(&"user-1".to_string(), 1000).await);
    }
}
