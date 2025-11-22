//! Mental Accounting for Prospect Theory
use crate::{Position, ProspectTheoryError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalAccount {
    pub name: String,
    pub balance: f64,
    pub perceived_value: f64,
    pub risk_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct MentalAccounting {
    accounts: HashMap<String, MentalAccount>,
    default_risk_tolerance: f64,
}

impl MentalAccounting {
    pub fn new() -> Self { 
        Self {
            accounts: HashMap::new(),
            default_risk_tolerance: 0.5,
        }
    }
    
    pub fn add_account(&mut self, account: MentalAccount) {
        self.accounts.insert(account.name.clone(), account);
    }
    
    pub fn get_account(&self, name: &str) -> Option<&MentalAccount> {
        self.accounts.get(name)
    }
    
    pub fn get_account_weights(&self, symbol: &str, _position: Option<&Position>) -> Result<AccountWeights> {
        let base_weights = AccountWeights { 
            trading: 1.0, 
            investment: 0.5, 
            speculation: 0.3 
        };
        
        if let Some(account) = self.accounts.get(symbol) {
            // Adjust weights based on account risk tolerance
            let risk_factor = account.risk_tolerance;
            Ok(AccountWeights {
                trading: base_weights.trading * risk_factor,
                investment: base_weights.investment * (1.0 + risk_factor),
                speculation: base_weights.speculation * risk_factor,
            })
        } else {
            Ok(base_weights)
        }
    }
    
    pub fn calculate_bias(&self, symbol: &str, amount: f64) -> f64 {
        if let Some(account) = self.accounts.get(symbol) {
            let ratio = amount / account.balance.max(1.0);
            let risk_adjustment = (account.risk_tolerance - self.default_risk_tolerance) * 0.2;
            (ratio * 0.1 + risk_adjustment).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    }
    
    pub fn update_balance(&mut self, name: &str, new_balance: f64) {
        if let Some(account) = self.accounts.get_mut(name) {
            account.balance = new_balance;
            account.perceived_value = new_balance * (1.0 + account.risk_tolerance * 0.1);
        }
    }
}

impl Default for MentalAccounting {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountWeights {
    pub trading: f64,
    pub investment: f64,
    pub speculation: f64,
}