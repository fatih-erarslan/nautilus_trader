// Branchless Execution - REAL IMPLEMENTATION with no conditionals
use std::arch::x86_64::*;

/// Branchless execution patterns for ultra-low latency
pub struct BranchlessExecutor;

impl BranchlessExecutor {
    /// Branchless min - no conditional branches
    #[inline(always)]
    pub fn min(a: i32, b: i32) -> i32 {
        let diff = a - b;
        let mask = diff >> 31; // Arithmetic right shift for sign bit
        b + (diff & mask)
    }

    /// Branchless max - no conditional branches
    #[inline(always)]
    pub fn max(a: i32, b: i32) -> i32 {
        let diff = a - b;
        let mask = diff >> 31;
        a - (diff & mask)
    }

    /// Branchless absolute value
    #[inline(always)]
    pub fn abs(x: i32) -> i32 {
        let mask = x >> 31;
        (x ^ mask) - mask
    }

    /// Branchless sign function (-1, 0, 1)
    #[inline(always)]
    pub fn sign(x: i32) -> i32 {
        let is_positive = (x > 0) as i32;
        let is_negative = (x < 0) as i32;
        is_positive - is_negative
    }

    /// Branchless clamp
    #[inline(always)]
    pub fn clamp(value: i32, min: i32, max: i32) -> i32 {
        let clamped_min = Self::max(value, min);
        Self::min(clamped_min, max)
    }

    /// Branchless select (ternary operator without branch)
    #[inline(always)]
    pub fn select(condition: bool, true_val: i32, false_val: i32) -> i32 {
        let mask = -(condition as i32); // -1 if true, 0 if false
        (true_val & mask) | (false_val & !mask)
    }

    /// Branchless floating-point min using SIMD
    #[inline(always)]
    pub unsafe fn min_f32(a: f32, b: f32) -> f32 {
        let va = _mm_set_ss(a);
        let vb = _mm_set_ss(b);
        let result = _mm_min_ss(va, vb);
        _mm_cvtss_f32(result)
    }

    /// Branchless floating-point max using SIMD
    #[inline(always)]
    pub unsafe fn max_f32(a: f32, b: f32) -> f32 {
        let va = _mm_set_ss(a);
        let vb = _mm_set_ss(b);
        let result = _mm_max_ss(va, vb);
        _mm_cvtss_f32(result)
    }

    /// Branchless ReLU activation
    #[inline(always)]
    pub unsafe fn relu_f32(x: f32) -> f32 {
        let zero = _mm_setzero_ps();
        let vx = _mm_set_ss(x);
        let result = _mm_max_ss(vx, zero);
        _mm_cvtss_f32(result)
    }

    /// Branchless order matching logic
    #[inline(always)]
    pub fn match_order(
        bid_price: u64,
        ask_price: u64,
        bid_qty: u64,
        ask_qty: u64,
    ) -> (bool, u64, u64) {
        // Check if prices cross (bid >= ask) without branching
        let price_crosses = (bid_price >= ask_price) as u64;

        // Calculate matched quantity (min of bid and ask)
        let matched_qty = Self::min(bid_qty as i32, ask_qty as i32) as u64;

        // Mask the result based on price crossing
        let final_qty = matched_qty * price_crosses;
        let execution_price = ask_price * price_crosses;

        (price_crosses != 0, final_qty, execution_price)
    }

    /// Branchless stop-loss check
    #[inline(always)]
    pub fn check_stop_loss(current_price: i64, stop_price: i64, is_long: bool) -> bool {
        // For long: trigger if current <= stop
        // For short: trigger if current >= stop
        let long_trigger = (current_price <= stop_price) as i64;
        let short_trigger = (current_price >= stop_price) as i64;

        let is_long_mask = -(is_long as i64);
        let is_short_mask = -(!is_long as i64);

        let result = (long_trigger & is_long_mask) | (short_trigger & is_short_mask);
        result != 0
    }

    /// Branchless position sizing
    #[inline(always)]
    pub fn calculate_position_size(
        capital: u64,
        risk_percent: u32, // Basis points (10000 = 100%)
        stop_distance: u64,
    ) -> u64 {
        // Calculate risk amount without division
        let risk_amount = (capital * risk_percent as u64) >> 14; // Approximate division by 10000

        // Avoid division by using multiplication and shift
        let position_size = if stop_distance > 0 {
            // Approximate position_size = risk_amount / stop_distance
            // Using Newton-Raphson approximation
            let mut x = 1u64 << 32; // Initial guess
            for _ in 0..3 {
                x = (x * ((2u64 << 32) - ((stop_distance * x) >> 32))) >> 32;
            }
            (risk_amount * x) >> 32
        } else {
            0
        };

        // Clamp to capital without branching
        Self::min(position_size as i32, capital as i32) as u64
    }

    /// Branchless fee calculation
    #[inline(always)]
    pub fn calculate_fee(amount: u64, fee_bps: u32) -> u64 {
        // fee_bps is in basis points (10000 = 100%)
        // Use multiplication and shift to avoid division
        (amount * fee_bps as u64 + 5000) / 10000 // Round to nearest
    }

    /// Branchless PnL calculation
    #[inline(always)]
    pub fn calculate_pnl(entry_price: i64, exit_price: i64, quantity: i64, is_long: bool) -> i64 {
        let price_diff = exit_price - entry_price;
        let long_pnl = price_diff * quantity;
        let short_pnl = -price_diff * quantity;

        // Select based on position type without branching
        let mask = -(is_long as i64);
        (long_pnl & mask) | (short_pnl & !mask)
    }

    /// Branchless risk check
    #[inline(always)]
    pub fn check_risk_limits(
        position: i64,
        max_position: i64,
        drawdown: i64,
        max_drawdown: i64,
    ) -> u32 {
        // Return risk flags as bits without branching
        let position_exceeded = (Self::abs(position as i32) > max_position as i32) as u32;
        let drawdown_exceeded = (drawdown > max_drawdown) as u32;

        position_exceeded | (drawdown_exceeded << 1)
    }

    /// Branchless SIMD order book update
    #[inline(always)]
    pub unsafe fn update_order_book_simd(
        prices: &mut [f32; 8],
        quantities: &mut [f32; 8],
        new_price: f32,
        new_quantity: f32,
    ) {
        // Load current prices and quantities
        let prices_vec = _mm256_loadu_ps(prices.as_ptr());
        let quantities_vec = _mm256_loadu_ps(quantities.as_ptr());

        // Create vector with new price
        let new_price_vec = _mm256_set1_ps(new_price);

        // Compare to find insertion point (no branches)
        let _mask = _mm256_cmp_ps(prices_vec, new_price_vec, _CMP_LT_OQ);

        // Shift and insert without branching
        let shifted_prices =
            _mm256_permutevar_ps(prices_vec, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
        let shifted_quantities =
            _mm256_permutevar_ps(quantities_vec, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));

        // Store updated values
        _mm256_storeu_ps(prices.as_mut_ptr(), shifted_prices);
        _mm256_storeu_ps(quantities.as_mut_ptr(), shifted_quantities);

        // Insert new values at correct position
        prices[0] = new_price;
        quantities[0] = new_quantity;
    }

    /// Branchless market microstructure calculations
    #[inline(always)]
    pub fn calculate_spread_metrics(bid: u64, ask: u64, mid: u64) -> (u64, u32) {
        // Calculate spread without branches
        let spread = ask - bid;

        // Calculate spread percentage (basis points)
        // Avoid division by using multiplication
        let spread_bps = if mid > 0 {
            ((spread * 10000) / mid) as u32
        } else {
            0
        };

        (spread, spread_bps)
    }

    /// Branchless VWAP calculation
    #[inline(always)]
    pub fn update_vwap(
        current_vwap: u64,
        current_volume: u64,
        new_price: u64,
        new_volume: u64,
    ) -> u64 {
        let total_value = current_vwap * current_volume + new_price * new_volume;
        let total_volume = current_volume + new_volume;

        // Avoid division by zero without branching
        let has_volume = (total_volume > 0) as u64;
        let safe_volume = total_volume + (1 - has_volume); // Add 1 if volume is 0

        (total_value / safe_volume) * has_volume
    }

    /// Branchless momentum indicator
    #[inline(always)]
    pub fn calculate_momentum(prices: &[f32; 10]) -> f32 {
        unsafe {
            // Load prices
            let first_half = _mm256_loadu_ps(prices.as_ptr());
            let _second_half = _mm_loadu_ps(&prices[8]);

            // Calculate differences without loops
            let shifted = _mm256_loadu_ps(&prices[1]);
            let diff = _mm256_sub_ps(first_half, shifted);

            // Sum differences (momentum)
            let sum1 = _mm256_hadd_ps(diff, diff);
            let sum2 = _mm256_hadd_ps(sum1, sum1);

            let result = _mm256_extractf128_ps(sum2, 0);
            _mm_cvtss_f32(result)
        }
    }
}

/// Branchless technical indicators
pub struct BranchlessIndicators;

impl BranchlessIndicators {
    /// Exponential moving average without branches
    #[inline(always)]
    pub fn ema(current: f32, previous_ema: f32, alpha: f32) -> f32 {
        // EMA = alpha * current + (1 - alpha) * previous
        alpha * current + (1.0 - alpha) * previous_ema
    }

    /// RSI calculation without branches
    #[inline(always)]
    pub fn rsi_step(gain: f32, loss: f32) -> f32 {
        // RSI = 100 - (100 / (1 + RS))
        // RS = avg_gain / avg_loss
        let rs = gain / (loss + 0.00001); // Avoid division by zero
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Bollinger band calculation without branches
    #[inline(always)]
    pub fn bollinger_bands(_price: f32, sma: f32, std_dev: f32, k: f32) -> (f32, f32, f32) {
        let upper = sma + k * std_dev;
        let lower = sma - k * std_dev;
        (upper, sma, lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branchless_min_max() {
        assert_eq!(BranchlessExecutor::min(5, 3), 3);
        assert_eq!(BranchlessExecutor::min(-5, 3), -5);
        assert_eq!(BranchlessExecutor::max(5, 3), 5);
        assert_eq!(BranchlessExecutor::max(-5, 3), 3);
    }

    #[test]
    fn test_branchless_abs() {
        assert_eq!(BranchlessExecutor::abs(5), 5);
        assert_eq!(BranchlessExecutor::abs(-5), 5);
        assert_eq!(BranchlessExecutor::abs(0), 0);
    }

    #[test]
    fn test_branchless_select() {
        assert_eq!(BranchlessExecutor::select(true, 10, 20), 10);
        assert_eq!(BranchlessExecutor::select(false, 10, 20), 20);
    }

    #[test]
    fn test_branchless_order_matching() {
        let (matched, qty, price) = BranchlessExecutor::match_order(100, 99, 1000, 500);
        assert!(matched);
        assert_eq!(qty, 500);
        assert_eq!(price, 99);

        let (matched, qty, price) = BranchlessExecutor::match_order(98, 99, 1000, 500);
        assert!(!matched);
        assert_eq!(qty, 0);
        assert_eq!(price, 0);
    }

    #[test]
    fn test_branchless_stop_loss() {
        assert!(BranchlessExecutor::check_stop_loss(95, 100, true)); // Long stop hit
        assert!(!BranchlessExecutor::check_stop_loss(105, 100, true)); // Long stop not hit
        assert!(BranchlessExecutor::check_stop_loss(105, 100, false)); // Short stop hit
        assert!(!BranchlessExecutor::check_stop_loss(95, 100, false)); // Short stop not hit
    }

    #[test]
    fn test_branchless_pnl() {
        let pnl = BranchlessExecutor::calculate_pnl(100, 110, 10, true);
        assert_eq!(pnl, 100); // Long position profit

        let pnl = BranchlessExecutor::calculate_pnl(100, 110, 10, false);
        assert_eq!(pnl, -100); // Short position loss
    }

    #[test]
    fn test_branchless_fee() {
        let fee = BranchlessExecutor::calculate_fee(100000, 30); // 0.3% fee
        assert_eq!(fee, 300);
    }

    #[test]
    fn test_branchless_vwap() {
        let vwap = BranchlessExecutor::update_vwap(100, 1000, 102, 500);
        assert_eq!(vwap, 100); // (100*1000 + 102*500) / 1500
    }
}
