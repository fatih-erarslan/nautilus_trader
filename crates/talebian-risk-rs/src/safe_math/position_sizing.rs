//! Safe position sizing calculations for financial trading

use super::{SafeMathResult, safe_arithmetic::*, validation::*, is_valid_price};
use crate::error::TalebianError;

/// Position sizing calculation result
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    /// Calculated position size
    pub position_size: f64,
    /// Risk-adjusted position size
    pub risk_adjusted_size: f64,
    /// Maximum allowed position size
    pub max_allowed_size: f64,
    /// Actual risk percentage
    pub actual_risk: f64,
    /// Kelly fraction used
    pub kelly_fraction: f64,
    /// Confidence level
    pub confidence: f64,
    /// Warnings generated during calculation
    pub warnings: Vec<String>,
}

impl PositionSizeResult {
    pub fn new() -> Self {
        Self {
            position_size: 0.0,
            risk_adjusted_size: 0.0,
            max_allowed_size: 0.0,
            actual_risk: 0.0,
            kelly_fraction: 0.0,
            confidence: 0.0,
            warnings: Vec::new(),
        }
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Kelly criterion position sizing calculator
pub struct SafeKellyCalculator {
    max_kelly_fraction: f64,
    confidence_threshold: f64,
    min_sample_size: usize,
}

impl SafeKellyCalculator {
    pub fn new(max_kelly_fraction: f64) -> Self {
        Self {
            max_kelly_fraction: max_kelly_fraction.min(0.25), // Cap at 25%
            confidence_threshold: 0.7,
            min_sample_size: 30,
        }
    }
    
    /// Calculate Kelly fraction with comprehensive safety checks
    pub fn calculate_kelly_fraction(
        &self,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        sample_size: usize,
    ) -> SafeMathResult<f64> {
        // Validate inputs
        validate_kelly_params(win_rate, avg_win, avg_loss)?;
        
        if sample_size < self.min_sample_size {
            return Err(TalebianError::data(format!(
                "Insufficient sample size: {} (minimum: {})",
                sample_size, self.min_sample_size
            )));
        }
        
        // Calculate raw Kelly fraction: f = (bp - q) / b
        // where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        let b = safe_divide(avg_win, avg_loss)?;
        let q = safe_subtract(1.0, win_rate)?;
        let bp = safe_multiply(b, win_rate)?;
        let numerator = safe_subtract(bp, q)?;
        
        let raw_kelly = safe_divide(numerator, b)?;
        
        // Apply confidence adjustment based on sample size
        let confidence = self.calculate_confidence(sample_size);
        let confidence_adjusted = safe_multiply(raw_kelly, confidence)?;
        
        // Cap at maximum allowed fraction
        let final_kelly = confidence_adjusted.min(self.max_kelly_fraction);
        
        // Ensure non-negative (don't trade if Kelly is negative)
        Ok(final_kelly.max(0.0))
    }
    
    /// Calculate confidence based on sample size
    fn calculate_confidence(&self, sample_size: usize) -> f64 {
        if sample_size < self.min_sample_size {
            return 0.0;
        }
        
        // Confidence increases with sample size but plateaus
        let base_confidence = 1.0 - (self.min_sample_size as f64 / sample_size as f64);
        base_confidence.min(0.95) // Max 95% confidence
    }
}

/// Safe position size calculator
pub struct SafePositionSizer {
    max_risk_per_trade: f64,
    max_position_size: f64,
    kelly_calculator: SafeKellyCalculator,
}

impl SafePositionSizer {
    pub fn new(max_risk_per_trade: f64, max_position_size: f64) -> SafeMathResult<Self> {
        if max_risk_per_trade <= 0.0 || max_risk_per_trade > 0.1 {
            return Err(TalebianError::data(
                "Max risk per trade must be between 0 and 10%"
            ));
        }
        
        if max_position_size <= 0.0 || max_position_size > 1.0 {
            return Err(TalebianError::data(
                "Max position size must be between 0 and 100%"
            ));
        }
        
        Ok(Self {
            max_risk_per_trade,
            max_position_size,
            kelly_calculator: SafeKellyCalculator::new(0.25),
        })
    }
    
    /// Calculate position size using multiple methods and return the most conservative
    pub fn calculate_position_size(
        &self,
        capital: f64,
        entry_price: f64,
        stop_loss_price: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        sample_size: usize,
        confidence_multiplier: f64,
    ) -> SafeMathResult<PositionSizeResult> {
        let mut result = PositionSizeResult::new();
        
        // Validate inputs
        validate_position_size_params(capital, self.max_risk_per_trade, self.max_position_size)?;
        
        if !is_valid_price(entry_price) {
            return Err(TalebianError::data(format!("Invalid entry price: {}", entry_price)));
        }
        
        if !is_valid_price(stop_loss_price) {
            return Err(TalebianError::data(format!("Invalid stop loss price: {}", stop_loss_price)));
        }
        
        if entry_price <= stop_loss_price {
            return Err(TalebianError::data(
                "Entry price must be greater than stop loss price for long position"
            ));
        }
        
        if confidence_multiplier <= 0.0 || confidence_multiplier > 1.0 {
            return Err(TalebianError::data(
                "Confidence multiplier must be between 0 and 1"
            ));
        }
        
        // Calculate risk per share
        let risk_per_share = safe_subtract(entry_price, stop_loss_price)?;
        let risk_percentage = safe_divide(risk_per_share, entry_price)?;
        
        // Method 1: Fixed risk position sizing
        let max_risk_amount = safe_multiply(capital, self.max_risk_per_trade)?;
        let fixed_risk_shares = safe_divide(max_risk_amount, risk_per_share)?;
        let fixed_risk_position_value = safe_multiply(fixed_risk_shares, entry_price)?;
        let fixed_risk_position_size = safe_divide(fixed_risk_position_value, capital)?;
        
        // Method 2: Kelly criterion position sizing
        let kelly_fraction = match self.kelly_calculator.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss, sample_size
        ) {
            Ok(fraction) => fraction,
            Err(_) => {
                result.add_warning("Kelly calculation failed, using conservative sizing".to_string());
                0.01 // 1% fallback
            }
        };
        
        let confidence_adjusted_kelly = safe_multiply(kelly_fraction, confidence_multiplier)?;
        result.kelly_fraction = confidence_adjusted_kelly;
        result.confidence = confidence_multiplier;
        
        // Method 3: Volatility-adjusted sizing
        let volatility_adjustment = if risk_percentage > 0.05 {
            result.add_warning("High volatility detected, reducing position size".to_string());
            0.5
        } else if risk_percentage > 0.02 {
            0.8
        } else {
            1.0
        };
        
        // Choose the most conservative size
        let conservative_size = fixed_risk_position_size
            .min(confidence_adjusted_kelly)
            .min(self.max_position_size);
        
        let volatility_adjusted_size = safe_multiply(conservative_size, volatility_adjustment)?;
        
        result.position_size = conservative_size;
        result.risk_adjusted_size = volatility_adjusted_size;
        result.max_allowed_size = self.max_position_size;
        
        // Calculate actual risk
        let actual_position_value = safe_multiply(volatility_adjusted_size, capital)?;
        let actual_shares = safe_divide(actual_position_value, entry_price)?;
        let actual_risk_amount = safe_multiply(actual_shares, risk_per_share)?;
        result.actual_risk = safe_divide(actual_risk_amount, capital)?;
        
        // Add warnings for edge cases
        if result.actual_risk > self.max_risk_per_trade {
            result.add_warning(format!(
                "Actual risk ({:.2}%) exceeds maximum ({:.2}%)",
                result.actual_risk * 100.0,
                self.max_risk_per_trade * 100.0
            ));
        }
        
        if sample_size < 50 {
            result.add_warning("Limited trade history, using conservative sizing".to_string());
        }
        
        if win_rate < 0.4 {
            result.add_warning("Low win rate detected, consider strategy review".to_string());
        }
        
        Ok(result)
    }
    
    /// Calculate position size for short positions
    pub fn calculate_short_position_size(
        &self,
        capital: f64,
        entry_price: f64,
        stop_loss_price: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        sample_size: usize,
        confidence_multiplier: f64,
    ) -> SafeMathResult<PositionSizeResult> {
        if entry_price >= stop_loss_price {
            return Err(TalebianError::data(
                "Entry price must be less than stop loss price for short position"
            ));
        }
        
        // For shorts, risk is (stop_loss - entry) per share
        let risk_per_share = safe_subtract(stop_loss_price, entry_price)?;
        
        // Similar calculation but with inverted logic
        self.calculate_position_size(
            capital,
            entry_price,
            entry_price - risk_per_share, // Convert to equivalent long format
            win_rate,
            avg_win,
            avg_loss,
            sample_size,
            confidence_multiplier,
        )
    }
}

/// Portfolio-level position sizing with correlation adjustments
pub struct SafePortfolioSizer {
    position_sizer: SafePositionSizer,
    max_total_exposure: f64,
    max_correlated_exposure: f64,
}

impl SafePortfolioSizer {
    pub fn new(
        max_risk_per_trade: f64,
        max_position_size: f64,
        max_total_exposure: f64,
        max_correlated_exposure: f64,
    ) -> SafeMathResult<Self> {
        if max_total_exposure <= 0.0 || max_total_exposure > 1.0 {
            return Err(TalebianError::data(
                "Max total exposure must be between 0 and 100%"
            ));
        }
        
        if max_correlated_exposure <= 0.0 || max_correlated_exposure > max_total_exposure {
            return Err(TalebianError::data(
                "Max correlated exposure must be between 0 and max total exposure"
            ));
        }
        
        Ok(Self {
            position_sizer: SafePositionSizer::new(max_risk_per_trade, max_position_size)?,
            max_total_exposure,
            max_correlated_exposure,
        })
    }
    
    /// Calculate position size considering existing portfolio exposure
    pub fn calculate_portfolio_adjusted_size(
        &self,
        capital: f64,
        entry_price: f64,
        stop_loss_price: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        sample_size: usize,
        confidence_multiplier: f64,
        current_total_exposure: f64,
        current_correlated_exposure: f64,
        correlation_with_portfolio: f64,
    ) -> SafeMathResult<PositionSizeResult> {
        // Calculate base position size
        let mut result = self.position_sizer.calculate_position_size(
            capital,
            entry_price,
            stop_loss_price,
            win_rate,
            avg_win,
            avg_loss,
            sample_size,
            confidence_multiplier,
        )?;
        
        // Adjust for portfolio constraints
        let remaining_total_exposure = safe_subtract(self.max_total_exposure, current_total_exposure)?;
        if remaining_total_exposure <= 0.0 {
            result.risk_adjusted_size = 0.0;
            result.add_warning("Maximum total portfolio exposure reached".to_string());
            return Ok(result);
        }
        
        // Check correlation constraints
        if correlation_with_portfolio.abs() > 0.7 {
            let remaining_correlated_exposure = safe_subtract(
                self.max_correlated_exposure,
                current_correlated_exposure
            )?;
            
            if remaining_correlated_exposure <= 0.0 {
                result.risk_adjusted_size = 0.0;
                result.add_warning("Maximum correlated exposure reached".to_string());
                return Ok(result);
            }
            
            // Reduce size based on correlation
            let correlation_adjustment = 1.0 - correlation_with_portfolio.abs() * 0.5;
            result.risk_adjusted_size = safe_multiply(
                result.risk_adjusted_size,
                correlation_adjustment
            )?;
            
            result.add_warning(format!(
                "Position size reduced due to correlation: {:.1}%",
                correlation_adjustment * 100.0
            ));
        }
        
        // Final constraint check
        result.risk_adjusted_size = result.risk_adjusted_size
            .min(remaining_total_exposure);
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_calculator() {
        let calculator = SafeKellyCalculator::new(0.25);
        
        // Test valid Kelly calculation
        let kelly = calculator.calculate_kelly_fraction(0.6, 150.0, 100.0, 50);
        assert!(kelly.is_ok());
        let kelly_value = kelly.unwrap();
        assert!(kelly_value > 0.0);
        assert!(kelly_value <= 0.25);
    }

    #[test]
    fn test_kelly_calculator_insufficient_sample() {
        let calculator = SafeKellyCalculator::new(0.25);
        
        let kelly = calculator.calculate_kelly_fraction(0.6, 150.0, 100.0, 10);
        assert!(kelly.is_err());
    }

    #[test]
    fn test_position_sizer_creation() {
        let sizer = SafePositionSizer::new(0.02, 0.1);
        assert!(sizer.is_ok());
        
        let invalid_sizer = SafePositionSizer::new(0.15, 0.1); // Too high risk
        assert!(invalid_sizer.is_err());
    }

    #[test]
    fn test_position_size_calculation() {
        let sizer = SafePositionSizer::new(0.02, 0.1).unwrap();
        
        let result = sizer.calculate_position_size(
            100000.0, // capital
            100.0,    // entry price
            95.0,     // stop loss
            0.6,      // win rate
            150.0,    // avg win
            100.0,    // avg loss
            50,       // sample size
            0.8,      // confidence
        );
        
        assert!(result.is_ok());
        let pos_result = result.unwrap();
        assert!(pos_result.position_size > 0.0);
        assert!(pos_result.position_size <= 0.1);
        assert!(pos_result.actual_risk <= 0.02);
    }

    #[test]
    fn test_invalid_position_params() {
        let sizer = SafePositionSizer::new(0.02, 0.1).unwrap();
        
        // Test invalid entry/stop loss combination
        let result = sizer.calculate_position_size(
            100000.0, // capital
            95.0,     // entry price
            100.0,    // stop loss (higher than entry)
            0.6,      // win rate
            150.0,    // avg win
            100.0,    // avg loss
            50,       // sample size
            0.8,      // confidence
        );
        
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_sizer() {
        let portfolio_sizer = SafePortfolioSizer::new(0.02, 0.1, 0.8, 0.3);
        assert!(portfolio_sizer.is_ok());
        
        let sizer = portfolio_sizer.unwrap();
        let result = sizer.calculate_portfolio_adjusted_size(
            100000.0, // capital
            100.0,    // entry price
            95.0,     // stop loss
            0.6,      // win rate
            150.0,    // avg win
            100.0,    // avg loss
            50,       // sample size
            0.8,      // confidence
            0.5,      // current total exposure
            0.1,      // current correlated exposure
            0.8,      // correlation with portfolio
        );
        
        assert!(result.is_ok());
    }
}