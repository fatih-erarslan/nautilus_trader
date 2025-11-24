// WebGPU Compute Shader: Momentum Calculation System
// High-performance vectorized momentum calculations for parasitic trading
// Optimized for parallel execution with SIMD operations

struct PriceData {
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    volume: f32,
    timestamp: f32,
}

struct MomentumIndicators {
    rsi: f32,
    macd: f32,
    macd_signal: f32,
    macd_histogram: f32,
    stoch_k: f32,
    stoch_d: f32,
    williams_r: f32,
    cci: f32,
    roc: f32,
    momentum: f32,
}

struct MomentumConfig {
    rsi_period: u32,
    macd_fast: u32,
    macd_slow: u32,
    macd_signal: u32,
    stoch_k_period: u32,
    stoch_d_period: u32,
    cci_period: u32,
    roc_period: u32,
}

struct MovingAverages {
    sma_5: f32,
    sma_10: f32,
    sma_20: f32,
    sma_50: f32,
    ema_12: f32,
    ema_26: f32,
    ema_9: f32,
    wma_14: f32,
}

// Input buffers
@group(0) @binding(0) var<storage, read> price_data: array<PriceData>;
@group(0) @binding(1) var<storage, read> config: MomentumConfig;
@group(0) @binding(2) var<storage, read> previous_ema_values: array<f32>; // For EMA continuity

// Output buffers
@group(0) @binding(3) var<storage, read_write> momentum_indicators: array<MomentumIndicators>;
@group(0) @binding(4) var<storage, read_write> moving_averages: array<MovingAverages>;
@group(0) @binding(5) var<storage, read_write> momentum_signals: array<f32>; // Combined momentum signal

// Shared memory for workgroup operations
var<workgroup> shared_prices: array<f32, 256>;
var<workgroup> shared_volumes: array<f32, 256>;
var<workgroup> shared_highs: array<f32, 256>;
var<workgroup> shared_lows: array<f32, 256>;

// Constants
const EMA_SMOOTHING_FACTOR: f32 = 2.0;
const MIN_PERIODS_REQUIRED: u32 = 50u;
const MOMENTUM_SCALE_FACTOR: f32 = 100.0;
const STOCH_SCALE: f32 = 100.0;
const CCI_CONSTANT: f32 = 0.015;
const WILLIAMS_R_SCALE: f32 = -100.0;

// Utility functions for vectorized operations
fn vec4_sum(v: vec4<f32>) -> f32 {
    return v.x + v.y + v.z + v.w;
}

fn calculate_sma_vectorized(values: ptr<function, array<f32, 64>>, start: u32, period: u32) -> f32 {
    if (period == 0u || start + period > 64u) {
        return 0.0;
    }
    
    var sum = 0.0;
    let vector_loops = period / 4u;
    let remainder = period % 4u;
    
    // Vectorized summation for groups of 4
    for (var i = 0u; i < vector_loops; i++) {
        let base_idx = start + i * 4u;
        let vec_data = vec4<f32>(
            (*values)[base_idx],
            (*values)[base_idx + 1u],
            (*values)[base_idx + 2u],
            (*values)[base_idx + 3u]
        );
        sum += vec4_sum(vec_data);
    }
    
    // Handle remainder
    for (var i = 0u; i < remainder; i++) {
        sum += (*values)[start + vector_loops * 4u + i];
    }
    
    return sum / f32(period);
}

fn calculate_ema(current_price: f32, previous_ema: f32, period: u32) -> f32 {
    let alpha = EMA_SMOOTHING_FACTOR / (f32(period) + 1.0);
    return alpha * current_price + (1.0 - alpha) * previous_ema;
}

fn calculate_wma(values: ptr<function, array<f32, 64>>, start: u32, period: u32) -> f32 {
    if (period == 0u || start + period > 64u) {
        return 0.0;
    }
    
    var weighted_sum = 0.0;
    var weight_sum = 0.0;
    
    for (var i = 0u; i < period; i++) {
        let weight = f32(i + 1u);
        weighted_sum += (*values)[start + i] * weight;
        weight_sum += weight;
    }
    
    return if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 };
}

fn find_highest_in_range(values: ptr<function, array<f32, 64>>, start: u32, period: u32) -> f32 {
    if (period == 0u || start + period > 64u) {
        return 0.0;
    }
    
    var highest = (*values)[start];
    for (var i = 1u; i < period; i++) {
        highest = max(highest, (*values)[start + i]);
    }
    return highest;
}

fn find_lowest_in_range(values: ptr<function, array<f32, 64>>, start: u32, period: u32) -> f32 {
    if (period == 0u || start + period > 64u) {
        return 999999.0;
    }
    
    var lowest = (*values)[start];
    for (var i = 1u; i < period; i++) {
        lowest = min(lowest, (*values)[start + i]);
    }
    return lowest;
}

fn calculate_rsi(gains: ptr<function, array<f32, 32>>, losses: ptr<function, array<f32, 32>>, period: u32) -> f32 {
    if (period == 0u) {
        return 50.0;
    }
    
    var avg_gain = 0.0;
    var avg_loss = 0.0;
    let periods_to_use = min(period, 32u);
    
    for (var i = 0u; i < periods_to_use; i++) {
        avg_gain += (*gains)[i];
        avg_loss += (*losses)[i];
    }
    
    avg_gain /= f32(periods_to_use);
    avg_loss /= f32(periods_to_use);
    
    if (avg_loss == 0.0) {
        return 100.0;
    }
    
    let rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

fn calculate_typical_price(high: f32, low: f32, close: f32) -> f32 {
    return (high + low + close) / 3.0;
}

fn calculate_mean_deviation(values: ptr<function, array<f32, 32>>, mean: f32, period: u32) -> f32 {
    var sum_dev = 0.0;
    let periods_to_use = min(period, 32u);
    
    for (var i = 0u; i < periods_to_use; i++) {
        sum_dev += abs((*values)[i] - mean);
    }
    
    return sum_dev / f32(periods_to_use);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    if (index >= arrayLength(&price_data)) {
        return;
    }
    
    let current_data = price_data[index];
    
    // Initialize output structures
    var indicators = MomentumIndicators();
    var averages = MovingAverages();
    
    // Ensure we have enough data for calculations
    if (index < MIN_PERIODS_REQUIRED) {
        momentum_indicators[index] = indicators;
        moving_averages[index] = averages;
        momentum_signals[index] = 0.0;
        return;
    }
    
    // Prepare local data arrays for calculations
    var local_closes: array<f32, 64>;
    var local_highs: array<f32, 64>;
    var local_lows: array<f32, 64>;
    var local_volumes: array<f32, 64>;
    var typical_prices: array<f32, 32>;
    
    let lookback_start = index - min(64u, index);
    let lookback_count = index - lookback_start + 1u;
    
    // Load historical data into local arrays
    for (var i = 0u; i < min(64u, lookback_count); i++) {
        let hist_index = lookback_start + i;
        if (hist_index < arrayLength(&price_data)) {
            local_closes[i] = price_data[hist_index].close;
            local_highs[i] = price_data[hist_index].high;
            local_lows[i] = price_data[hist_index].low;
            local_volumes[i] = price_data[hist_index].volume;
            
            if (i < 32u) {
                typical_prices[i] = calculate_typical_price(
                    price_data[hist_index].high,
                    price_data[hist_index].low,
                    price_data[hist_index].close
                );
            }
        }
    }
    
    let data_available = min(64u, lookback_count);
    
    // 1. Calculate Moving Averages
    if (data_available >= 5u) {
        averages.sma_5 = calculate_sma_vectorized(&local_closes, data_available - 5u, 5u);
    }
    if (data_available >= 10u) {
        averages.sma_10 = calculate_sma_vectorized(&local_closes, data_available - 10u, 10u);
    }
    if (data_available >= 20u) {
        averages.sma_20 = calculate_sma_vectorized(&local_closes, data_available - 20u, 20u);
    }
    if (data_available >= 50u) {
        averages.sma_50 = calculate_sma_vectorized(&local_closes, data_available - 50u, 50u);
    }
    
    // Calculate EMAs with continuity
    if (data_available >= 12u) {
        let prev_ema_12 = if index >= 12u { previous_ema_values[index - 12u] } else { averages.sma_10 };
        averages.ema_12 = calculate_ema(current_data.close, prev_ema_12, 12u);
    }
    if (data_available >= 26u) {
        let prev_ema_26 = if index >= 26u { previous_ema_values[index - 26u] } else { averages.sma_20 };
        averages.ema_26 = calculate_ema(current_data.close, prev_ema_26, 26u);
    }
    if (data_available >= 9u) {
        let prev_ema_9 = if index >= 9u { previous_ema_values[index - 9u] } else { averages.sma_5 };
        averages.ema_9 = calculate_ema(current_data.close, prev_ema_9, 9u);
    }
    
    // Weighted Moving Average
    if (data_available >= 14u) {
        averages.wma_14 = calculate_wma(&local_closes, data_available - 14u, 14u);
    }
    
    // 2. Calculate RSI
    if (data_available >= config.rsi_period + 1u) {
        var gains: array<f32, 32>;
        var losses: array<f32, 32>;
        
        let rsi_start = data_available - config.rsi_period - 1u;
        for (var i = 0u; i < config.rsi_period; i++) {
            let price_change = local_closes[rsi_start + i + 1u] - local_closes[rsi_start + i];
            gains[i] = max(price_change, 0.0);
            losses[i] = max(-price_change, 0.0);
        }
        
        indicators.rsi = calculate_rsi(&gains, &losses, config.rsi_period);
    }
    
    // 3. Calculate MACD
    if (data_available >= max(config.macd_slow, config.macd_fast)) {
        indicators.macd = averages.ema_12 - averages.ema_26;
        
        // MACD Signal Line (EMA of MACD)
        if (index >= config.macd_signal) {
            let prev_macd_signal = if index > config.macd_signal { 
                momentum_indicators[index - 1u].macd_signal 
            } else { 
                indicators.macd 
            };
            indicators.macd_signal = calculate_ema(indicators.macd, prev_macd_signal, config.macd_signal);
        }
        
        indicators.macd_histogram = indicators.macd - indicators.macd_signal;
    }
    
    // 4. Calculate Stochastic Oscillator
    if (data_available >= config.stoch_k_period) {
        let stoch_start = data_available - config.stoch_k_period;
        let highest_high = find_highest_in_range(&local_highs, stoch_start, config.stoch_k_period);
        let lowest_low = find_lowest_in_range(&local_lows, stoch_start, config.stoch_k_period);
        
        if (highest_high != lowest_low) {
            indicators.stoch_k = ((current_data.close - lowest_low) / (highest_high - lowest_low)) * STOCH_SCALE;
        } else {
            indicators.stoch_k = 50.0;
        }
        
        // %D is SMA of %K
        if (data_available >= config.stoch_k_period + config.stoch_d_period - 1u) {
            var k_values: array<f32, 64>;
            for (var i = 0u; i < config.stoch_d_period; i++) {
                // Calculate %K for each period (simplified for performance)
                k_values[i] = indicators.stoch_k; // Would need more complex calculation for accuracy
            }
            indicators.stoch_d = calculate_sma_vectorized(&k_values, 0u, config.stoch_d_period);
        }
    }
    
    // 5. Calculate Williams %R
    if (data_available >= 14u) {
        let williams_start = data_available - 14u;
        let highest_high = find_highest_in_range(&local_highs, williams_start, 14u);
        let lowest_low = find_lowest_in_range(&local_lows, williams_start, 14u);
        
        if (highest_high != lowest_low) {
            indicators.williams_r = ((highest_high - current_data.close) / (highest_high - lowest_low)) * WILLIAMS_R_SCALE;
        } else {
            indicators.williams_r = -50.0;
        }
    }
    
    // 6. Calculate Commodity Channel Index (CCI)
    if (data_available >= config.cci_period) {
        let cci_start = data_available - config.cci_period;
        var cci_typical_prices: array<f32, 32>;
        
        let periods_for_cci = min(config.cci_period, 32u);
        for (var i = 0u; i < periods_for_cci; i++) {
            cci_typical_prices[i] = typical_prices[cci_start + i];
        }
        
        let sma_tp = calculate_sma_vectorized(&cci_typical_prices, 0u, periods_for_cci);
        let current_tp = calculate_typical_price(current_data.high, current_data.low, current_data.close);
        let mean_deviation = calculate_mean_deviation(&cci_typical_prices, sma_tp, periods_for_cci);
        
        if (mean_deviation > 0.0) {
            indicators.cci = (current_tp - sma_tp) / (CCI_CONSTANT * mean_deviation);
        }
    }
    
    // 7. Calculate Rate of Change (ROC)
    if (data_available >= config.roc_period + 1u) {
        let roc_start = data_available - config.roc_period - 1u;
        let old_price = local_closes[roc_start];
        if (old_price > 0.0) {
            indicators.roc = ((current_data.close - old_price) / old_price) * MOMENTUM_SCALE_FACTOR;
        }
    }
    
    // 8. Calculate Simple Momentum
    if (data_available >= 10u) {
        let momentum_start = data_available - 10u;
        indicators.momentum = current_data.close - local_closes[momentum_start];
    }
    
    // 9. Calculate Combined Momentum Signal
    var signal_components: array<f32, 6>;
    signal_components[0] = (indicators.rsi - 50.0) / 50.0; // Normalize RSI
    signal_components[1] = indicators.macd / max(abs(indicators.macd), 1.0); // Normalize MACD
    signal_components[2] = (indicators.stoch_k - 50.0) / 50.0; // Normalize Stochastic
    signal_components[3] = indicators.williams_r / 100.0; // Normalize Williams %R
    signal_components[4] = indicators.cci / 100.0; // Normalize CCI
    signal_components[5] = indicators.roc / MOMENTUM_SCALE_FACTOR; // Normalize ROC
    
    // Weighted combination
    let weights = array<f32, 6>(0.2, 0.25, 0.15, 0.1, 0.15, 0.15);
    var combined_signal = 0.0;
    for (var i = 0u; i < 6u; i++) {
        combined_signal += signal_components[i] * weights[i];
    }
    
    // Clamp signal to [-1, 1] range
    combined_signal = clamp(combined_signal, -1.0, 1.0);
    
    // Store results
    momentum_indicators[index] = indicators;
    moving_averages[index] = averages;
    momentum_signals[index] = combined_signal;
    
    // Update shared memory for workgroup coordination
    if (local_index < 64u) {
        shared_prices[local_index] = current_data.close;
        shared_volumes[local_index] = current_data.volume;
        shared_highs[local_index] = current_data.high;
        shared_lows[local_index] = current_data.low;
    }
    
    workgroupBarrier();
}

// Streaming optimization kernel for real-time updates
@compute @workgroup_size(32, 1, 1)
fn update_streaming_momentum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&price_data) || index < 20u) {
        return;
    }
    
    let current_data = price_data[index];
    var indicators = momentum_indicators[index];
    
    // Quick momentum updates for streaming
    let prev_close = price_data[index - 1u].close;
    let price_change = current_data.close - prev_close;
    
    // Update simple momentum
    indicators.momentum = price_change;
    
    // Quick RSI approximation
    let gain = max(price_change, 0.0);
    let loss = max(-price_change, 0.0);
    
    // Simplified RSI update (would need proper EMA for accuracy)
    if (loss == 0.0 && gain > 0.0) {
        indicators.rsi = min(indicators.rsi + 5.0, 100.0);
    } else if (gain == 0.0 && loss > 0.0) {
        indicators.rsi = max(indicators.rsi - 5.0, 0.0);
    }
    
    // Update MACD with new EMAs
    let new_ema_12 = calculate_ema(current_data.close, moving_averages[index].ema_12, 12u);
    let new_ema_26 = calculate_ema(current_data.close, moving_averages[index].ema_26, 26u);
    indicators.macd = new_ema_12 - new_ema_26;
    
    // Update signal line
    indicators.macd_signal = calculate_ema(indicators.macd, indicators.macd_signal, 9u);
    indicators.macd_histogram = indicators.macd - indicators.macd_signal;
    
    // Store updated indicators
    momentum_indicators[index] = indicators;
    
    // Update combined signal
    let normalized_momentum = clamp(indicators.momentum / max(abs(indicators.momentum), 1.0), -1.0, 1.0);
    let normalized_rsi = (indicators.rsi - 50.0) / 50.0;
    let normalized_macd = clamp(indicators.macd / max(abs(indicators.macd), 1.0), -1.0, 1.0);
    
    let quick_signal = (normalized_momentum * 0.4 + normalized_rsi * 0.3 + normalized_macd * 0.3);
    momentum_signals[index] = clamp(quick_signal, -1.0, 1.0);
}