// WebGPU Compute Shader: Risk Calculation System
// Real-time risk computations for parasitic momentum trading
// Optimized for sub-1ms execution with comprehensive risk metrics

struct PositionData {
    size: f32,
    entry_price: f32,
    current_price: f32,
    stop_loss: f32,
    take_profit: f32,
    leverage: f32,
    timestamp: f32,
    direction: i32, // 1 for long, -1 for short
}

struct MarketData {
    price: f32,
    volume: f32,
    bid: f32,
    ask: f32,
    volatility: f32,
    liquidity_score: f32,
    timestamp: f32,
    spread: f32,
}

struct RiskMetrics {
    unrealized_pnl: f32,
    var_95: f32,  // Value at Risk 95%
    var_99: f32,  // Value at Risk 99%
    expected_shortfall: f32,
    maximum_drawdown: f32,
    sharpe_ratio: f32,
    sortino_ratio: f32,
    beta: f32,
    alpha: f32,
    correlation: f32,
}

struct RiskLimits {
    max_position_size: f32,
    max_leverage: f32,
    max_daily_loss: f32,
    max_portfolio_var: f32,
    min_liquidity_score: f32,
    max_concentration: f32,
    stress_test_threshold: f32,
}

struct VolatilityMetrics {
    realized_vol: f32,
    implied_vol: f32,
    vol_of_vol: f32,
    skewness: f32,
    kurtosis: f32,
    garch_forecast: f32,
}

struct LiquidityMetrics {
    bid_ask_spread: f32,
    market_depth: f32,
    volume_weighted_spread: f32,
    liquidity_ratio: f32,
    order_book_imbalance: f32,
    trade_impact_cost: f32,
}

// Input buffers
@group(0) @binding(0) var<storage, read> positions: array<PositionData>;
@group(0) @binding(1) var<storage, read> market_data: array<MarketData>;
@group(0) @binding(2) var<storage, read> historical_returns: array<f32>;
@group(0) @binding(3) var<storage, read> risk_limits: RiskLimits;
@group(0) @binding(4) var<storage, read> benchmark_returns: array<f32>;

// Output buffers
@group(0) @binding(5) var<storage, read_write> risk_metrics: array<RiskMetrics>;
@group(0) @binding(6) var<storage, read_write> volatility_metrics: array<VolatilityMetrics>;
@group(0) @binding(7) var<storage, read_write> liquidity_metrics: array<LiquidityMetrics>;
@group(0) @binding(8) var<storage, read_write> risk_alerts: array<u32>;
@group(0) @binding(9) var<storage, read_write> portfolio_risk: array<f32>; // Aggregated portfolio metrics

// Shared memory for workgroup operations
var<workgroup> shared_returns: array<f32, 256>;
var<workgroup> shared_prices: array<f32, 256>;
var<workgroup> shared_volatilities: array<f32, 64>;

// Constants
const TRADING_DAYS_PER_YEAR: f32 = 252.0;
const CONFIDENCE_95: f32 = 1.645;
const CONFIDENCE_99: f32 = 2.326;
const MIN_OBSERVATIONS: u32 = 30u;
const MAX_LOOKBACK_PERIODS: u32 = 252u;
const GARCH_ALPHA: f32 = 0.1;
const GARCH_BETA: f32 = 0.85;
const RISK_FREE_RATE: f32 = 0.02; // 2% annual

// Mathematical utility functions
fn norm_inv_approx(p: f32) -> f32 {
    // Beasley-Springer-Moro algorithm for inverse normal
    let a0 = 2.50662823884;
    let a1 = -18.61500062529;
    let a2 = 41.39119773534;
    let a3 = -25.44106049637;
    let b1 = -8.47351093090;
    let b2 = 23.08336743743;
    let b3 = -21.06224101826;
    let b4 = 3.13082909833;
    let c0 = 0.3374754822726147;
    let c1 = 0.9761690190917186;
    let c2 = 0.1607979714918209;
    let c3 = 0.0276438810333863;
    let c4 = 0.0038405729373609;
    let c5 = 0.0003951896511919;
    let c6 = 0.0000321767881768;
    let c7 = 0.0000002888167364;
    let c8 = 0.0000003960315187;
    
    let y = p - 0.5;
    
    if (abs(y) < 0.42) {
        let r = y * y;
        return y * (((a3 * r + a2) * r + a1) * r + a0) / ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0);
    }
    
    let r = if y > 0.0 { sqrt(-log(1.0 - p)) } else { sqrt(-log(p)) };
    let x = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
    
    return if y > 0.0 { x } else { -x };
}

fn calculate_mean(values: ptr<function, array<f32, 256>>, count: u32) -> f32 {
    if (count == 0u) {
        return 0.0;
    }
    
    var sum = 0.0;
    for (var i = 0u; i < count; i++) {
        sum += (*values)[i];
    }
    return sum / f32(count);
}

fn calculate_variance(values: ptr<function, array<f32, 256>>, mean: f32, count: u32) -> f32 {
    if (count <= 1u) {
        return 0.0;
    }
    
    var sum_sq_diff = 0.0;
    for (var i = 0u; i < count; i++) {
        let diff = (*values)[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / f32(count - 1u);
}

fn calculate_covariance(x: ptr<function, array<f32, 256>>, y: ptr<function, array<f32, 256>>, 
                       x_mean: f32, y_mean: f32, count: u32) -> f32 {
    if (count <= 1u) {
        return 0.0;
    }
    
    var sum_cov = 0.0;
    for (var i = 0u; i < count; i++) {
        sum_cov += ((*x)[i] - x_mean) * ((*y)[i] - y_mean);
    }
    return sum_cov / f32(count - 1u);
}

fn calculate_skewness(values: ptr<function, array<f32, 256>>, mean: f32, std_dev: f32, count: u32) -> f32 {
    if (count < 3u || std_dev == 0.0) {
        return 0.0;
    }
    
    var sum_cubes = 0.0;
    for (var i = 0u; i < count; i++) {
        let normalized = ((*values)[i] - mean) / std_dev;
        sum_cubes += normalized * normalized * normalized;
    }
    
    let n = f32(count);
    return (n / ((n - 1.0) * (n - 2.0))) * sum_cubes;
}

fn calculate_kurtosis(values: ptr<function, array<f32, 256>>, mean: f32, std_dev: f32, count: u32) -> f32 {
    if (count < 4u || std_dev == 0.0) {
        return 3.0; // Normal distribution kurtosis
    }
    
    var sum_fourth = 0.0;
    for (var i = 0u; i < count; i++) {
        let normalized = ((*values)[i] - mean) / std_dev;
        let sq = normalized * normalized;
        sum_fourth += sq * sq;
    }
    
    let n = f32(count);
    let excess_kurtosis = (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth 
                         - (3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0)));
    return excess_kurtosis + 3.0;
}

fn calculate_var(returns: ptr<function, array<f32, 256>>, count: u32, confidence: f32) -> f32 {
    if (count < MIN_OBSERVATIONS) {
        return 0.0;
    }
    
    // Sort returns (simplified bubble sort for GPU)
    var sorted_returns: array<f32, 256>;
    for (var i = 0u; i < count; i++) {
        sorted_returns[i] = (*returns)[i];
    }
    
    // Bubble sort (not optimal but simple for GPU)
    for (var i = 0u; i < count - 1u; i++) {
        for (var j = 0u; j < count - i - 1u; j++) {
            if (sorted_returns[j] > sorted_returns[j + 1u]) {
                let temp = sorted_returns[j];
                sorted_returns[j] = sorted_returns[j + 1u];
                sorted_returns[j + 1u] = temp;
            }
        }
    }
    
    // Calculate percentile
    let percentile_rank = (1.0 - confidence / 100.0) * f32(count);
    let index = u32(floor(percentile_rank));
    let fraction = percentile_rank - f32(index);
    
    if (index >= count - 1u) {
        return -sorted_returns[count - 1u];
    }
    
    let interpolated = sorted_returns[index] + fraction * (sorted_returns[index + 1u] - sorted_returns[index]);
    return -interpolated; // VaR is positive by convention
}

fn calculate_expected_shortfall(returns: ptr<function, array<f32, 256>>, count: u32, var_threshold: f32) -> f32 {
    if (count == 0u) {
        return 0.0;
    }
    
    var sum_tail_losses = 0.0;
    var tail_count = 0u;
    
    for (var i = 0u; i < count; i++) {
        if ((*returns)[i] <= -var_threshold) {
            sum_tail_losses += (*returns)[i];
            tail_count++;
        }
    }
    
    if (tail_count > 0u) {
        return -sum_tail_losses / f32(tail_count);
    }
    return var_threshold;
}

fn calculate_garch_forecast(returns: ptr<function, array<f32, 256>>, count: u32) -> f32 {
    if (count < 10u) {
        return 0.0;
    }
    
    // Simple GARCH(1,1) implementation
    let long_term_var = calculate_variance(returns, calculate_mean(returns, count), count);
    let omega = long_term_var * (1.0 - GARCH_ALPHA - GARCH_BETA);
    
    var forecast_var = long_term_var;
    let recent_return = (*returns)[count - 1u];
    let recent_var = recent_return * recent_return;
    
    forecast_var = omega + GARCH_ALPHA * recent_var + GARCH_BETA * forecast_var;
    
    return sqrt(forecast_var * TRADING_DAYS_PER_YEAR);
}

fn calculate_maximum_drawdown(prices: ptr<function, array<f32, 256>>, count: u32) -> f32 {
    if (count < 2u) {
        return 0.0;
    }
    
    var max_drawdown = 0.0;
    var peak = (*prices)[0];
    
    for (var i = 1u; i < count; i++) {
        if ((*prices)[i] > peak) {
            peak = (*prices)[i];
        }
        
        let drawdown = (peak - (*prices)[i]) / peak;
        max_drawdown = max(max_drawdown, drawdown);
    }
    
    return max_drawdown;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    if (index >= arrayLength(&positions)) {
        return;
    }
    
    let position = positions[index];
    let current_market = market_data[index];
    
    // Initialize output structures
    var risk = RiskMetrics();
    var vol_metrics = VolatilityMetrics();
    var liq_metrics = LiquidityMetrics();
    var alert_flags = 0u;
    
    // Calculate unrealized PnL
    let price_diff = current_market.price - position.entry_price;
    risk.unrealized_pnl = position.size * price_diff * f32(position.direction) * position.leverage;
    
    // Prepare return series for risk calculations
    let lookback_count = min(MAX_LOOKBACK_PERIODS, arrayLength(&historical_returns));
    var local_returns: array<f32, 256>;
    var local_benchmark: array<f32, 256>;
    var price_series: array<f32, 256>;
    
    let actual_count = min(lookback_count, 256u);
    let start_index = if arrayLength(&historical_returns) > actual_count { 
        arrayLength(&historical_returns) - actual_count 
    } else { 
        0u 
    };
    
    // Load historical data
    for (var i = 0u; i < actual_count; i++) {
        local_returns[i] = historical_returns[start_index + i];
        if (i < arrayLength(&benchmark_returns)) {
            local_benchmark[i] = benchmark_returns[start_index + i];
        }
        
        // Reconstruct price series from returns
        if (i == 0u) {
            price_series[i] = 100.0; // Base price
        } else {
            price_series[i] = price_series[i - 1u] * (1.0 + local_returns[i]);
        }
    }
    
    if (actual_count < MIN_OBSERVATIONS) {
        // Insufficient data for risk calculations
        risk_metrics[index] = risk;
        volatility_metrics[index] = vol_metrics;
        liquidity_metrics[index] = liq_metrics;
        risk_alerts[index] = alert_flags;
        return;
    }
    
    // Calculate basic statistics
    let mean_return = calculate_mean(&local_returns, actual_count);
    let return_variance = calculate_variance(&local_returns, mean_return, actual_count);
    let return_std = sqrt(return_variance);
    let annualized_vol = return_std * sqrt(TRADING_DAYS_PER_YEAR);
    
    // Calculate VaR
    risk.var_95 = calculate_var(&local_returns, actual_count, 95.0) * position.size * position.leverage;
    risk.var_99 = calculate_var(&local_returns, actual_count, 99.0) * position.size * position.leverage;
    
    // Calculate Expected Shortfall
    risk.expected_shortfall = calculate_expected_shortfall(&local_returns, actual_count, risk.var_95 / (position.size * position.leverage)) * position.size * position.leverage;
    
    // Calculate Maximum Drawdown
    risk.maximum_drawdown = calculate_maximum_drawdown(&price_series, actual_count);
    
    // Calculate Sharpe Ratio
    let excess_return = mean_return - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR;
    risk.sharpe_ratio = if return_std > 0.0 { excess_return / return_std * sqrt(TRADING_DAYS_PER_YEAR) } else { 0.0 };
    
    // Calculate Sortino Ratio (downside deviation)
    var downside_variance = 0.0;
    var downside_count = 0u;
    for (var i = 0u; i < actual_count; i++) {
        if (local_returns[i] < mean_return) {
            let diff = local_returns[i] - mean_return;
            downside_variance += diff * diff;
            downside_count++;
        }
    }
    
    if (downside_count > 0u) {
        let downside_std = sqrt(downside_variance / f32(downside_count));
        risk.sortino_ratio = if downside_std > 0.0 { excess_return / downside_std * sqrt(TRADING_DAYS_PER_YEAR) } else { 0.0 };
    }
    
    // Calculate Beta and Alpha (if benchmark data available)
    if (arrayLength(&benchmark_returns) >= actual_count) {
        let benchmark_mean = calculate_mean(&local_benchmark, actual_count);
        let benchmark_variance = calculate_variance(&local_benchmark, benchmark_mean, actual_count);
        let covariance = calculate_covariance(&local_returns, &local_benchmark, mean_return, benchmark_mean, actual_count);
        
        if (benchmark_variance > 0.0) {
            risk.beta = covariance / benchmark_variance;
            risk.alpha = (mean_return - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) - risk.beta * (benchmark_mean - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR);
            risk.alpha *= TRADING_DAYS_PER_YEAR; // Annualize
        }
        
        // Calculate correlation
        let benchmark_std = sqrt(benchmark_variance);
        if (return_std > 0.0 && benchmark_std > 0.0) {
            risk.correlation = covariance / (return_std * benchmark_std);
        }
    }
    
    // Volatility Metrics
    vol_metrics.realized_vol = annualized_vol;
    vol_metrics.implied_vol = current_market.volatility; // From market data
    vol_metrics.skewness = calculate_skewness(&local_returns, mean_return, return_std, actual_count);
    vol_metrics.kurtosis = calculate_kurtosis(&local_returns, mean_return, return_std, actual_count);
    vol_metrics.garch_forecast = calculate_garch_forecast(&local_returns, actual_count);
    
    // Volatility of volatility (rolling window of volatilities)
    if (actual_count >= 60u) {
        var vol_series: array<f32, 64>;
        let vol_window = 20u;
        let vol_count = min(60u, (actual_count - vol_window + 1u));
        
        for (var i = 0u; i < vol_count; i++) {
            var window_returns: array<f32, 256>;
            for (var j = 0u; j < vol_window; j++) {
                window_returns[j] = local_returns[i + j];
            }
            let window_mean = calculate_mean(&window_returns, vol_window);
            let window_var = calculate_variance(&window_returns, window_mean, vol_window);
            vol_series[i] = sqrt(window_var);
        }
        
        let vol_mean = calculate_mean(&vol_series, vol_count);
        vol_metrics.vol_of_vol = sqrt(calculate_variance(&vol_series, vol_mean, vol_count));
    }
    
    // Liquidity Metrics
    liq_metrics.bid_ask_spread = current_market.spread;
    liq_metrics.volume_weighted_spread = current_market.spread * (current_market.volume / 1000000.0);
    liq_metrics.liquidity_ratio = current_market.liquidity_score;
    liq_metrics.market_depth = current_market.volume; // Simplified
    
    // Order book imbalance (simplified calculation)
    let mid_price = (current_market.bid + current_market.ask) / 2.0;
    liq_metrics.order_book_imbalance = (current_market.price - mid_price) / mid_price;
    
    // Trade impact cost estimation
    let position_value = abs(position.size) * current_market.price;
    let avg_daily_volume = current_market.volume * 100.0; // Estimate
    let participation_rate = position_value / avg_daily_volume;
    liq_metrics.trade_impact_cost = sqrt(participation_rate) * current_market.spread;
    
    // Risk Alert Checks
    if (abs(risk.unrealized_pnl) > risk_limits.max_daily_loss) {
        alert_flags |= 1u; // Daily loss limit exceeded
    }
    
    if (abs(position.size) > risk_limits.max_position_size) {
        alert_flags |= 2u; // Position size limit exceeded
    }
    
    if (position.leverage > risk_limits.max_leverage) {
        alert_flags |= 4u; // Leverage limit exceeded
    }
    
    if (risk.var_95 > risk_limits.max_portfolio_var) {
        alert_flags |= 8u; // VaR limit exceeded
    }
    
    if (current_market.liquidity_score < risk_limits.min_liquidity_score) {
        alert_flags |= 16u; // Liquidity threshold breached
    }
    
    if (vol_metrics.realized_vol > risk_limits.stress_test_threshold) {
        alert_flags |= 32u; // High volatility alert
    }
    
    // Store results
    risk_metrics[index] = risk;
    volatility_metrics[index] = vol_metrics;
    liquidity_metrics[index] = liq_metrics;
    risk_alerts[index] = alert_flags;
    
    // Update shared memory for portfolio aggregation
    if (local_index < 64u) {
        shared_returns[local_index] = mean_return;
        shared_prices[local_index] = current_market.price;
        shared_volatilities[local_index] = vol_metrics.realized_vol;
    }
    
    workgroupBarrier();
    
    // Portfolio-level risk aggregation (performed by first thread in workgroup)
    if (local_index == 0u) {
        var portfolio_var = 0.0;
        var portfolio_return = 0.0;
        var portfolio_vol = 0.0;
        
        for (var i = 0u; i < 64u; i++) {
            portfolio_return += shared_returns[i];
            portfolio_vol += shared_volatilities[i];
        }
        
        portfolio_return /= 64.0;
        portfolio_vol /= 64.0;
        
        // Simple portfolio VaR (assumes equal weights and no correlation)
        portfolio_var = portfolio_vol * CONFIDENCE_95 * sqrt(64.0 / 64.0); // Diversification factor
        
        let portfolio_index = global_id.x / 64u;
        if (portfolio_index < arrayLength(&portfolio_risk)) {
            portfolio_risk[portfolio_index] = portfolio_var;
        }
    }
}

// Real-time risk monitoring kernel for streaming updates
@compute @workgroup_size(32, 1, 1)
fn update_streaming_risk(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&positions)) {
        return;
    }
    
    let position = positions[index];
    let current_market = market_data[index];
    
    // Quick risk updates for streaming
    var risk = risk_metrics[index];
    
    // Update unrealized PnL
    let price_diff = current_market.price - position.entry_price;
    risk.unrealized_pnl = position.size * price_diff * f32(position.direction) * position.leverage;
    
    // Quick VaR update using previous calculation and current price movement
    let price_change_pct = (current_market.price - position.entry_price) / position.entry_price;
    let vol_adjustment = 1.0 + abs(price_change_pct);
    risk.var_95 *= vol_adjustment;
    risk.var_99 *= vol_adjustment;
    
    // Update liquidity metrics
    var liq_metrics = liquidity_metrics[index];
    liq_metrics.bid_ask_spread = current_market.spread;
    liq_metrics.liquidity_ratio = current_market.liquidity_score;
    
    // Check for immediate risk alerts
    var alert_flags = 0u;
    if (abs(risk.unrealized_pnl) > risk_limits.max_daily_loss) {
        alert_flags |= 1u;
    }
    if (current_market.liquidity_score < risk_limits.min_liquidity_score) {
        alert_flags |= 16u;
    }
    
    // Store updated results
    risk_metrics[index] = risk;
    liquidity_metrics[index] = liq_metrics;
    risk_alerts[index] = alert_flags;
}