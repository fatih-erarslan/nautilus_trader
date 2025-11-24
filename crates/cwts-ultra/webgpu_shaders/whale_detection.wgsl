// WebGPU Compute Shader: Whale Detection System
// GPU-accelerated detection of large trades (whales) in market data
// Optimized for parallel processing with sub-1ms execution targets

struct TradeData {
    price: f32,
    volume: f32,
    timestamp: f32,
    market_cap: f32,
}

struct WhaleMetrics {
    volume_threshold: f32,
    price_impact_threshold: f32,
    time_window: f32,
    confidence_score: f32,
}

struct WhaleDetectionResult {
    is_whale: u32,
    confidence: f32,
    volume_ratio: f32,
    price_impact: f32,
    momentum_strength: f32,
    risk_factor: f32,
}

// Input buffers
@group(0) @binding(0) var<storage, read> trade_data: array<TradeData>;
@group(0) @binding(1) var<storage, read> whale_metrics: WhaleMetrics;
@group(0) @binding(2) var<storage, read> historical_avg_volume: array<f32>;
@group(0) @binding(3) var<storage, read> price_history: array<f32>;

// Output buffer
@group(0) @binding(4) var<storage, read_write> detection_results: array<WhaleDetectionResult>;

// Shared memory for workgroup operations
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> price_impact_cache: array<f32, 64>;

// Constants for whale detection
const WHALE_VOLUME_MULTIPLIER: f32 = 10.0;
const PRICE_IMPACT_DECAY: f32 = 0.95;
const MOMENTUM_SMOOTHING: f32 = 0.8;
const MIN_CONFIDENCE_THRESHOLD: f32 = 0.7;
const MAX_LOOKBACK_PERIODS: u32 = 100u;

// Utility functions
fn calculate_volume_z_score(current_volume: f32, avg_volume: f32, std_dev: f32) -> f32 {
    if (std_dev > 0.0) {
        return (current_volume - avg_volume) / std_dev;
    }
    return 0.0;
}

fn calculate_price_impact(price_before: f32, price_after: f32, volume: f32) -> f32 {
    let price_change = abs(price_after - price_before) / price_before;
    let volume_weight = min(volume / 1000000.0, 1.0); // Normalize volume
    return price_change * volume_weight;
}

fn exponential_moving_average(values: ptr<function, array<f32, 32>>, length: u32, alpha: f32) -> f32 {
    if (length == 0u) {
        return 0.0;
    }
    
    var ema = (*values)[0];
    for (var i = 1u; i < length; i++) {
        ema = alpha * (*values)[i] + (1.0 - alpha) * ema;
    }
    return ema;
}

fn calculate_momentum_strength(prices: ptr<function, array<f32, 32>>, volumes: ptr<function, array<f32, 32>>, length: u32) -> f32 {
    if (length < 2u) {
        return 0.0;
    }
    
    var momentum = 0.0;
    var total_weight = 0.0;
    
    for (var i = 1u; i < length; i++) {
        let price_change = ((*prices)[i] - (*prices)[i - 1u]) / (*prices)[i - 1u];
        let volume_weight = (*volumes)[i] / 1000000.0; // Normalize
        momentum += price_change * volume_weight;
        total_weight += volume_weight;
    }
    
    if (total_weight > 0.0) {
        return momentum / total_weight;
    }
    return 0.0;
}

fn detect_volume_spike(current_volume: f32, historical_volumes: ptr<function, array<f32, 32>>, length: u32) -> f32 {
    if (length == 0u) {
        return 0.0;
    }
    
    // Calculate rolling average and standard deviation
    var sum = 0.0;
    var sum_sq = 0.0;
    
    for (var i = 0u; i < length; i++) {
        let vol = (*historical_volumes)[i];
        sum += vol;
        sum_sq += vol * vol;
    }
    
    let avg = sum / f32(length);
    let variance = (sum_sq / f32(length)) - (avg * avg);
    let std_dev = sqrt(max(variance, 0.0));
    
    return calculate_volume_z_score(current_volume, avg, std_dev);
}

fn analyze_order_flow_pattern(trade: TradeData, lookback_trades: ptr<function, array<TradeData, 16>>, count: u32) -> f32 {
    if (count < 3u) {
        return 0.5; // Neutral pattern
    }
    
    var buy_pressure = 0.0;
    var sell_pressure = 0.0;
    var total_volume = 0.0;
    
    // Analyze recent trade pattern
    for (var i = 0u; i < count; i++) {
        let trade_vol = (*lookback_trades)[i].volume;
        let price_change = (*lookback_trades)[i].price - (if i > 0u { (*lookback_trades)[i - 1u].price } else { trade.price });
        
        if (price_change > 0.0) {
            buy_pressure += trade_vol;
        } else {
            sell_pressure += trade_vol;
        }
        total_volume += trade_vol;
    }
    
    if (total_volume > 0.0) {
        return buy_pressure / total_volume; // 0.0 = sell pressure, 1.0 = buy pressure
    }
    return 0.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    if (index >= arrayLength(&trade_data)) {
        return;
    }
    
    let current_trade = trade_data[index];
    
    // Initialize result
    var result = WhaleDetectionResult();
    result.is_whale = 0u;
    result.confidence = 0.0;
    result.volume_ratio = 0.0;
    result.price_impact = 0.0;
    result.momentum_strength = 0.0;
    result.risk_factor = 0.0;
    
    // Gather historical data for analysis
    let lookback_start = max(0u, index - MAX_LOOKBACK_PERIODS);
    let lookback_count = index - lookback_start;
    
    if (lookback_count < 10u) {
        detection_results[index] = result;
        return;
    }
    
    // Prepare local arrays for calculations
    var local_prices: array<f32, 32>;
    var local_volumes: array<f32, 32>;
    var local_trades: array<TradeData, 16>;
    
    let max_local = min(32u, lookback_count);
    for (var i = 0u; i < max_local; i++) {
        let hist_index = lookback_start + i;
        if (hist_index < arrayLength(&trade_data)) {
            local_prices[i] = trade_data[hist_index].price;
            local_volumes[i] = trade_data[hist_index].volume;
            
            if (i < 16u) {
                local_trades[i] = trade_data[hist_index];
            }
        }
    }
    
    // 1. Volume Analysis
    let volume_z_score = detect_volume_spike(current_trade.volume, &local_volumes, max_local);
    result.volume_ratio = volume_z_score;
    
    // Check if volume exceeds whale threshold
    let is_volume_whale = volume_z_score > whale_metrics.volume_threshold;
    
    // 2. Price Impact Analysis
    if (index > 0u && index < arrayLength(&trade_data) - 1u) {
        let price_before = trade_data[index - 1u].price;
        let price_after = if index + 1u < arrayLength(&trade_data) { trade_data[index + 1u].price } else { current_trade.price };
        result.price_impact = calculate_price_impact(price_before, price_after, current_trade.volume);
    }
    
    let is_price_impact_whale = result.price_impact > whale_metrics.price_impact_threshold;
    
    // 3. Momentum Strength Analysis
    result.momentum_strength = calculate_momentum_strength(&local_prices, &local_volumes, max_local);
    
    // 4. Order Flow Pattern Analysis
    let order_flow_pattern = analyze_order_flow_pattern(current_trade, &local_trades, min(16u, max_local));
    
    // 5. Risk Factor Calculation
    let volatility = if max_local > 1u {
        var price_changes: array<f32, 31>;
        for (var i = 1u; i < max_local; i++) {
            price_changes[i - 1u] = abs(local_prices[i] - local_prices[i - 1u]) / local_prices[i - 1u];
        }
        
        var sum_changes = 0.0;
        for (var i = 0u; i < max_local - 1u; i++) {
            sum_changes += price_changes[i];
        }
        sum_changes / f32(max_local - 1u)
    } else {
        0.0
    };
    
    result.risk_factor = min(volatility * 100.0, 1.0); // Normalize to 0-1
    
    // 6. Confidence Score Calculation
    var confidence_factors: array<f32, 5>;
    confidence_factors[0] = if is_volume_whale { 1.0 } else { 0.0 };
    confidence_factors[1] = if is_price_impact_whale { 1.0 } else { 0.0 };
    confidence_factors[2] = min(abs(result.momentum_strength) * 10.0, 1.0);
    confidence_factors[3] = abs(order_flow_pattern - 0.5) * 2.0; // Deviation from neutral
    confidence_factors[4] = min(result.risk_factor * 2.0, 1.0);
    
    // Weighted confidence calculation
    let weights = array<f32, 5>(0.3, 0.25, 0.2, 0.15, 0.1);
    var weighted_confidence = 0.0;
    for (var i = 0u; i < 5u; i++) {
        weighted_confidence += confidence_factors[i] * weights[i];
    }
    
    result.confidence = weighted_confidence;
    
    // 7. Final Whale Detection Decision
    let is_whale_detected = (is_volume_whale || is_price_impact_whale) && 
                           result.confidence >= MIN_CONFIDENCE_THRESHOLD &&
                           abs(result.momentum_strength) > 0.001; // Minimum momentum threshold
    
    result.is_whale = if is_whale_detected { 1u } else { 0u };
    
    // Store result
    detection_results[index] = result;
    
    // Workgroup synchronization for shared computations
    workgroupBarrier();
    
    // Optional: Update shared statistics for next iterations
    if (local_index == 0u) {
        shared_data[0] = result.confidence;
        shared_data[1] = result.volume_ratio;
        shared_data[2] = result.price_impact;
    }
}

// Additional compute kernel for real-time streaming updates
@compute @workgroup_size(32, 1, 1)
fn update_streaming_detection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&trade_data)) {
        return;
    }
    
    // Optimized path for real-time updates with minimal lookback
    let current_trade = trade_data[index];
    let lookback_limit = min(10u, index);
    
    if (lookback_limit < 3u) {
        return;
    }
    
    // Quick volume spike detection
    var recent_avg_volume = 0.0;
    for (var i = index - lookback_limit; i < index; i++) {
        recent_avg_volume += trade_data[i].volume;
    }
    recent_avg_volume /= f32(lookback_limit);
    
    let volume_ratio = current_trade.volume / recent_avg_volume;
    let is_volume_spike = volume_ratio > WHALE_VOLUME_MULTIPLIER;
    
    // Quick momentum check
    let price_momentum = if index > 0u {
        (current_trade.price - trade_data[index - 1u].price) / trade_data[index - 1u].price
    } else {
        0.0
    };
    
    // Update result with streaming optimizations
    if (is_volume_spike && abs(price_momentum) > 0.001) {
        var result = detection_results[index];
        result.is_whale = 1u;
        result.confidence = min(volume_ratio / WHALE_VOLUME_MULTIPLIER, 1.0);
        result.volume_ratio = volume_ratio;
        result.momentum_strength = price_momentum;
        detection_results[index] = result;
    }
}