/**
 * Nautilus MCP Tools - Complete Tool Definitions
 * 120+ Trading Analytics Tools
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

// ============================================================================
// MOVING AVERAGES (10 tools)
// ============================================================================

export const movingAverageTools: Tool[] = [
    {
        name: "indicator_sma",
        description: "Simple Moving Average - arithmetic mean over period",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_ema",
        description: "Exponential Moving Average - weighted toward recent prices",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_wma",
        description: "Weighted Moving Average - linearly weighted",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_hma",
        description: "Hull Moving Average - fast, reduced-lag",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_dema",
        description: "Double Exponential Moving Average",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_tema",
        description: "Triple Exponential Moving Average",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_rma",
        description: "Running Moving Average (Wilder's smoothing)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_vidya",
        description: "Variable Index Dynamic Average",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_ama",
        description: "Adaptive Moving Average (Kaufman's)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
                fast_period: { type: "number" },
                slow_period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_vwap",
        description: "Volume Weighted Average Price",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
            },
            required: ["bars"],
        },
    },
];

// ============================================================================
// MOMENTUM INDICATORS (18 tools)
// ============================================================================

export const momentumTools: Tool[] = [
    {
        name: "indicator_rsi",
        description: "Relative Strength Index (0-100)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_macd",
        description: "Moving Average Convergence Divergence",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                fast: { type: "number", default: 12 },
                slow: { type: "number", default: 26 },
                signal: { type: "number", default: 9 },
            },
            required: ["prices"],
        },
    },
    {
        name: "indicator_cci",
        description: "Commodity Channel Index",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_stochastic",
        description: "Stochastic Oscillator %K/%D",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                k_period: { type: "number" },
                d_period: { type: "number" },
            },
            required: ["bars", "k_period", "d_period"],
        },
    },
    {
        name: "indicator_roc",
        description: "Rate of Change",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_obv",
        description: "On-Balance Volume",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
            },
            required: ["bars"],
        },
    },
    {
        name: "indicator_aroon",
        description: "Aroon Up/Down oscillator",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_cmo",
        description: "Chande Momentum Oscillator",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_williams_r",
        description: "Williams %R",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_mfi",
        description: "Money Flow Index",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_kvo",
        description: "Klinger Volume Oscillator",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
            },
            required: ["bars"],
        },
    },
    {
        name: "indicator_adx",
        description: "Average Directional Index",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_di",
        description: "Directional Indicator (+DI/-DI)",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_ppo",
        description: "Percentage Price Oscillator",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                fast: { type: "number" },
                slow: { type: "number" },
            },
            required: ["prices", "fast", "slow"],
        },
    },
    {
        name: "indicator_trix",
        description: "Triple Smoothed EMA Rate of Change",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_psl",
        description: "Psychological Line",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_vhf",
        description: "Vertical Horizontal Filter (trend strength)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_bias",
        description: "Price deviation from moving average",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
];

// ============================================================================
// VOLATILITY INDICATORS (10 tools)
// ============================================================================

export const volatilityTools: Tool[] = [
    {
        name: "indicator_atr",
        description: "Average True Range",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_bollinger",
        description: "Bollinger Bands",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
                std_dev: { type: "number", default: 2 },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_keltner",
        description: "Keltner Channel",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                ema_period: { type: "number" },
                atr_period: { type: "number" },
                multiplier: { type: "number" },
            },
            required: ["bars", "ema_period", "atr_period", "multiplier"],
        },
    },
    {
        name: "indicator_donchian",
        description: "Donchian Channel",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_natr",
        description: "Normalized Average True Range",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_rvi",
        description: "Relative Volatility Index",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_vr",
        description: "Volatility Ratio",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_bbwidth",
        description: "Bollinger Band Width",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
                std_dev: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "indicator_chaikin_vol",
        description: "Chaikin Volatility",
        inputSchema: {
            type: "object",
            properties: {
                bars: { type: "array", items: { type: "object" } },
                period: { type: "number" },
            },
            required: ["bars", "period"],
        },
    },
    {
        name: "indicator_fuzzy_regime",
        description: "Fuzzy volatility regime classification",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
];

// ============================================================================
// RISK ANALYTICS (20 tools)
// ============================================================================

export const riskTools: Tool[] = [
    {
        name: "risk_var_parametric",
        description: "Value at Risk (Gaussian)",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                confidence: { type: "number" },
            },
            required: ["returns", "confidence"],
        },
    },
    {
        name: "risk_var_historical",
        description: "Value at Risk (Historical)",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                confidence: { type: "number" },
            },
            required: ["returns", "confidence"],
        },
    },
    {
        name: "risk_var_monte_carlo",
        description: "Value at Risk (Monte Carlo simulation)",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                confidence: { type: "number" },
                simulations: { type: "number", default: 10000 },
            },
            required: ["returns", "confidence"],
        },
    },
    {
        name: "risk_cvar",
        description: "Conditional VaR / Expected Shortfall",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                confidence: { type: "number" },
            },
            required: ["returns", "confidence"],
        },
    },
    {
        name: "risk_kelly_criterion",
        description: "Kelly Criterion: f* = (p*b - q) / b",
        inputSchema: {
            type: "object",
            properties: {
                win_rate: { type: "number" },
                win_loss_ratio: { type: "number" },
            },
            required: ["win_rate", "win_loss_ratio"],
        },
    },
    {
        name: "risk_position_size",
        description: "Fixed-risk position sizing",
        inputSchema: {
            type: "object",
            properties: {
                equity: { type: "number" },
                risk_per_trade: { type: "number" },
                entry_price: { type: "number" },
                stop_loss: { type: "number" },
            },
            required: ["equity", "risk_per_trade", "entry_price", "stop_loss"],
        },
    },
    {
        name: "risk_max_drawdown",
        description: "Maximum Drawdown",
        inputSchema: {
            type: "object",
            properties: {
                equity_curve: { type: "array", items: { type: "number" } },
            },
            required: ["equity_curve"],
        },
    },
    {
        name: "risk_drawdown_duration",
        description: "Drawdown duration analysis",
        inputSchema: {
            type: "object",
            properties: {
                equity_curve: { type: "array", items: { type: "number" } },
            },
            required: ["equity_curve"],
        },
    },
    {
        name: "risk_hurst_exponent",
        description: "Hurst Exponent (trend/mean-reversion)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
            },
            required: ["prices"],
        },
    },
    {
        name: "risk_tail_ratio",
        description: "Tail Risk Ratio",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "risk_omega_ratio",
        description: "Omega Ratio",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                threshold: { type: "number", default: 0 },
            },
            required: ["returns"],
        },
    },
    {
        name: "risk_ulcer_index",
        description: "Ulcer Index (downside volatility)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "risk_gain_to_pain",
        description: "Gain-to-Pain Ratio",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "risk_kurtosis",
        description: "Excess Kurtosis (fat tails)",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "risk_skewness",
        description: "Return Distribution Skewness",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "risk_volatility_cone",
        description: "Volatility Cone Analysis",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                windows: { type: "array", items: { type: "number" } },
            },
            required: ["prices", "windows"],
        },
    },
    {
        name: "risk_correlation_breakdown",
        description: "Correlation breakdown detection",
        inputSchema: {
            type: "object",
            properties: {
                asset1_returns: { type: "array", items: { type: "number" } },
                asset2_returns: { type: "array", items: { type: "number" } },
                window: { type: "number" },
            },
            required: ["asset1_returns", "asset2_returns", "window"],
        },
    },
    {
        name: "risk_beta",
        description: "Market Beta",
        inputSchema: {
            type: "object",
            properties: {
                asset_returns: { type: "array", items: { type: "number" } },
                market_returns: { type: "array", items: { type: "number" } },
            },
            required: ["asset_returns", "market_returns"],
        },
    },
    {
        name: "risk_tracking_error",
        description: "Tracking Error vs benchmark",
        inputSchema: {
            type: "object",
            properties: {
                portfolio_returns: { type: "array", items: { type: "number" } },
                benchmark_returns: { type: "array", items: { type: "number" } },
            },
            required: ["portfolio_returns", "benchmark_returns"],
        },
    },
    {
        name: "risk_active_share",
        description: "Active Share vs index",
        inputSchema: {
            type: "object",
            properties: {
                portfolio_weights: { type: "array", items: { type: "number" } },
                benchmark_weights: { type: "array", items: { type: "number" } },
            },
            required: ["portfolio_weights", "benchmark_weights"],
        },
    },
];

// ============================================================================
// PORTFOLIO ANALYTICS (21 tools)
// ============================================================================

export const portfolioTools: Tool[] = [
    {
        name: "portfolio_sharpe",
        description: "Sharpe Ratio: (R - Rf) / σ",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                risk_free_rate: { type: "number", default: 0 },
            },
            required: ["returns"],
        },
    },
    {
        name: "portfolio_sortino",
        description: "Sortino Ratio (downside deviation)",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                risk_free_rate: { type: "number" },
                target: { type: "number", default: 0 },
            },
            required: ["returns"],
        },
    },
    {
        name: "portfolio_calmar",
        description: "Calmar Ratio: CAGR / MaxDD",
        inputSchema: {
            type: "object",
            properties: {
                equity_curve: { type: "array", items: { type: "number" } },
                periods_per_year: { type: "number" },
            },
            required: ["equity_curve", "periods_per_year"],
        },
    },
    {
        name: "portfolio_cagr",
        description: "Compound Annual Growth Rate",
        inputSchema: {
            type: "object",
            properties: {
                equity_curve: { type: "array", items: { type: "number" } },
                periods_per_year: { type: "number" },
            },
            required: ["equity_curve", "periods_per_year"],
        },
    },
    {
        name: "portfolio_win_rate",
        description: "Win Rate percentage",
        inputSchema: {
            type: "object",
            properties: {
                trade_pnls: { type: "array", items: { type: "number" } },
            },
            required: ["trade_pnls"],
        },
    },
    {
        name: "portfolio_profit_factor",
        description: "Profit Factor: gross profit / gross loss",
        inputSchema: {
            type: "object",
            properties: {
                trade_pnls: { type: "array", items: { type: "number" } },
            },
            required: ["trade_pnls"],
        },
    },
    {
        name: "portfolio_expectancy",
        description: "Trade Expectancy",
        inputSchema: {
            type: "object",
            properties: {
                trade_pnls: { type: "array", items: { type: "number" } },
            },
            required: ["trade_pnls"],
        },
    },
    {
        name: "portfolio_risk_return",
        description: "Risk/Return Ratio",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "portfolio_avg_win",
        description: "Average Winning Trade",
        inputSchema: {
            type: "object",
            properties: {
                trade_pnls: { type: "array", items: { type: "number" } },
            },
            required: ["trade_pnls"],
        },
    },
    {
        name: "portfolio_avg_loss",
        description: "Average Losing Trade",
        inputSchema: {
            type: "object",
            properties: {
                trade_pnls: { type: "array", items: { type: "number" } },
            },
            required: ["trade_pnls"],
        },
    },
    {
        name: "portfolio_long_ratio",
        description: "Long Exposure Ratio",
        inputSchema: {
            type: "object",
            properties: {
                positions: { type: "array", items: { type: "object" } },
            },
            required: ["positions"],
        },
    },
    {
        name: "portfolio_returns_volatility",
        description: "Returns Standard Deviation",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
            },
            required: ["returns"],
        },
    },
    {
        name: "portfolio_information_ratio",
        description: "Information Ratio vs benchmark",
        inputSchema: {
            type: "object",
            properties: {
                portfolio_returns: { type: "array", items: { type: "number" } },
                benchmark_returns: { type: "array", items: { type: "number" } },
            },
            required: ["portfolio_returns", "benchmark_returns"],
        },
    },
    {
        name: "portfolio_treynor",
        description: "Treynor Ratio: (R - Rf) / β",
        inputSchema: {
            type: "object",
            properties: {
                portfolio_returns: { type: "array", items: { type: "number" } },
                market_returns: { type: "array", items: { type: "number" } },
                risk_free_rate: { type: "number" },
            },
            required: ["portfolio_returns", "market_returns"],
        },
    },
    {
        name: "portfolio_alpha",
        description: "Jensen's Alpha",
        inputSchema: {
            type: "object",
            properties: {
                portfolio_returns: { type: "array", items: { type: "number" } },
                market_returns: { type: "array", items: { type: "number" } },
                risk_free_rate: { type: "number" },
            },
            required: ["portfolio_returns", "market_returns"],
        },
    },
    {
        name: "portfolio_correlation_matrix",
        description: "Asset Correlation Matrix",
        inputSchema: {
            type: "object",
            properties: {
                returns_matrix: { type: "array", items: { type: "array" } },
            },
            required: ["returns_matrix"],
        },
    },
    {
        name: "portfolio_covariance_matrix",
        description: "Asset Covariance Matrix",
        inputSchema: {
            type: "object",
            properties: {
                returns_matrix: { type: "array", items: { type: "array" } },
            },
            required: ["returns_matrix"],
        },
    },
    {
        name: "portfolio_efficient_frontier",
        description: "Markowitz Efficient Frontier",
        inputSchema: {
            type: "object",
            properties: {
                expected_returns: { type: "array", items: { type: "number" } },
                covariance_matrix: { type: "array", items: { type: "array" } },
                num_portfolios: { type: "number", default: 100 },
            },
            required: ["expected_returns", "covariance_matrix"],
        },
    },
    {
        name: "portfolio_max_diversification",
        description: "Maximum Diversification Portfolio",
        inputSchema: {
            type: "object",
            properties: {
                expected_returns: { type: "array", items: { type: "number" } },
                covariance_matrix: { type: "array", items: { type: "array" } },
            },
            required: ["expected_returns", "covariance_matrix"],
        },
    },
    {
        name: "portfolio_risk_parity",
        description: "Risk Parity Portfolio",
        inputSchema: {
            type: "object",
            properties: {
                covariance_matrix: { type: "array", items: { type: "array" } },
            },
            required: ["covariance_matrix"],
        },
    },
    {
        name: "portfolio_min_variance",
        description: "Minimum Variance Portfolio",
        inputSchema: {
            type: "object",
            properties: {
                covariance_matrix: { type: "array", items: { type: "array" } },
            },
            required: ["covariance_matrix"],
        },
    },
];

// ============================================================================
// EXECUTION ANALYSIS (12 tools)
// ============================================================================

export const executionTools: Tool[] = [
    {
        name: "execution_vwap_slippage",
        description: "VWAP Slippage Analysis",
        inputSchema: {
            type: "object",
            properties: {
                trades: { type: "array", items: { type: "object" } },
                vwap_benchmark: { type: "number" },
            },
            required: ["trades", "vwap_benchmark"],
        },
    },
    {
        name: "execution_twap",
        description: "Time-Weighted Average Price",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
            },
            required: ["prices"],
        },
    },
    {
        name: "execution_arrival_price",
        description: "Arrival Price Benchmark",
        inputSchema: {
            type: "object",
            properties: {
                trades: { type: "array", items: { type: "object" } },
                arrival_price: { type: "number" },
            },
            required: ["trades", "arrival_price"],
        },
    },
    {
        name: "execution_implementation_shortfall",
        description: "Implementation Shortfall",
        inputSchema: {
            type: "object",
            properties: {
                trades: { type: "array", items: { type: "object" } },
                decision_price: { type: "number" },
            },
            required: ["trades", "decision_price"],
        },
    },
    {
        name: "execution_market_impact",
        description: "Estimated Market Impact",
        inputSchema: {
            type: "object",
            properties: {
                order_size: { type: "number" },
                avg_daily_volume: { type: "number" },
                volatility: { type: "number" },
            },
            required: ["order_size", "avg_daily_volume", "volatility"],
        },
    },
    {
        name: "execution_spread_cost",
        description: "Bid-Ask Spread Cost",
        inputSchema: {
            type: "object",
            properties: {
                bid: { type: "number" },
                ask: { type: "number" },
                quantity: { type: "number" },
            },
            required: ["bid", "ask", "quantity"],
        },
    },
    {
        name: "orderflow_imbalance",
        description: "Order Flow Imbalance",
        inputSchema: {
            type: "object",
            properties: {
                trades: { type: "array", items: { type: "object" } },
            },
            required: ["trades"],
        },
    },
    {
        name: "orderflow_toxicity",
        description: "VPIN (Volume-synchronized PIN)",
        inputSchema: {
            type: "object",
            properties: {
                trades: { type: "array", items: { type: "object" } },
                bucket_size: { type: "number" },
            },
            required: ["trades", "bucket_size"],
        },
    },
    {
        name: "orderflow_kyle_lambda",
        description: "Kyle's Lambda (price impact)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                volumes: { type: "array", items: { type: "number" } },
            },
            required: ["prices", "volumes"],
        },
    },
    {
        name: "orderflow_amihud",
        description: "Amihud Illiquidity Ratio",
        inputSchema: {
            type: "object",
            properties: {
                returns: { type: "array", items: { type: "number" } },
                volumes: { type: "array", items: { type: "number" } },
            },
            required: ["returns", "volumes"],
        },
    },
    {
        name: "orderflow_tick_test",
        description: "Tick Test for trade classification",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
            },
            required: ["prices"],
        },
    },
    {
        name: "orderflow_quote_rule",
        description: "Quote Rule for trade classification",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                bids: { type: "array", items: { type: "number" } },
                asks: { type: "array", items: { type: "number" } },
            },
            required: ["prices", "bids", "asks"],
        },
    },
];

// ============================================================================
// REGIME DETECTION (12 tools - HyperPhysics)
// ============================================================================

export const regimeTools: Tool[] = [
    {
        name: "regime_pbit_state",
        description: "pBit Market State (Boltzmann)",
        inputSchema: {
            type: "object",
            properties: {
                market_signal: { type: "number" },
                volatility: { type: "number" },
                temperature: { type: "number" },
            },
            required: ["market_signal", "volatility", "temperature"],
        },
    },
    {
        name: "regime_ising_energy",
        description: "Ising Model Market Coherence",
        inputSchema: {
            type: "object",
            properties: {
                asset_returns: { type: "array", items: { type: "number" } },
            },
            required: ["asset_returns"],
        },
    },
    {
        name: "regime_hyperbolic_embed",
        description: "Hyperbolic Market Embedding (Lorentz)",
        inputSchema: {
            type: "object",
            properties: {
                features: { type: "array", items: { type: "number" } },
            },
            required: ["features"],
        },
    },
    {
        name: "regime_lorentz_distance",
        description: "Hyperbolic distance between states",
        inputSchema: {
            type: "object",
            properties: {
                state1: { type: "array", items: { type: "number" } },
                state2: { type: "array", items: { type: "number" } },
            },
            required: ["state1", "state2"],
        },
    },
    {
        name: "regime_boltzmann_dist",
        description: "Boltzmann Distribution of States",
        inputSchema: {
            type: "object",
            properties: {
                energies: { type: "array", items: { type: "number" } },
                temperature: { type: "number" },
            },
            required: ["energies", "temperature"],
        },
    },
    {
        name: "regime_critical_temp",
        description: "Proximity to Critical Temperature",
        inputSchema: {
            type: "object",
            properties: {
                market_data: { type: "array", items: { type: "number" } },
            },
            required: ["market_data"],
        },
    },
    {
        name: "regime_hmm_filter",
        description: "Hidden Markov Model Filtering",
        inputSchema: {
            type: "object",
            properties: {
                observations: { type: "array", items: { type: "number" } },
                n_states: { type: "number" },
            },
            required: ["observations", "n_states"],
        },
    },
    {
        name: "regime_change_detection",
        description: "Online Regime Change Detection",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                window: { type: "number" },
            },
            required: ["prices", "window"],
        },
    },
    {
        name: "regime_volatility_state",
        description: "Bull/Bear/Sideways Classification",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                period: { type: "number" },
            },
            required: ["prices", "period"],
        },
    },
    {
        name: "regime_phi_coherence",
        description: "Integrated Information (Φ) in Markets",
        inputSchema: {
            type: "object",
            properties: {
                asset_returns: { type: "array", items: { type: "array" } },
            },
            required: ["asset_returns"],
        },
    },
    {
        name: "regime_entropy",
        description: "Market Entropy (disorder measure)",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
                window: { type: "number" },
            },
            required: ["prices", "window"],
        },
    },
    {
        name: "regime_fractal_dimension",
        description: "Fractal Dimension of Price Series",
        inputSchema: {
            type: "object",
            properties: {
                prices: { type: "array", items: { type: "number" } },
            },
            required: ["prices"],
        },
    },
];

// ============================================================================
// CONFORMAL PREDICTION (10 tools)
// ============================================================================

export const conformalTools: Tool[] = [
    {
        name: "conformal_prediction_interval",
        description: "Conformal Prediction Interval",
        inputSchema: {
            type: "object",
            properties: {
                residuals: { type: "array", items: { type: "number" } },
                prediction: { type: "number" },
                confidence: { type: "number" },
            },
            required: ["residuals", "prediction", "confidence"],
        },
    },
    {
        name: "conformal_quantile_regression",
        description: "Conformalized Quantile Regression",
        inputSchema: {
            type: "object",
            properties: {
                residuals: { type: "array", items: { type: "number" } },
                quantiles: { type: "array", items: { type: "number" } },
                prediction: { type: "number" },
            },
            required: ["residuals", "quantiles", "prediction"],
        },
    },
    {
        name: "conformal_calibration",
        description: "Calibration Score Computation",
        inputSchema: {
            type: "object",
            properties: {
                predictions: { type: "array", items: { type: "number" } },
                actuals: { type: "array", items: { type: "number" } },
            },
            required: ["predictions", "actuals"],
        },
    },
    {
        name: "conformal_coverage_test",
        description: "Coverage Validation Test",
        inputSchema: {
            type: "object",
            properties: {
                intervals: { type: "array", items: { type: "object" } },
                actuals: { type: "array", items: { type: "number" } },
                target_coverage: { type: "number" },
            },
            required: ["intervals", "actuals", "target_coverage"],
        },
    },
    {
        name: "conformal_width_analysis",
        description: "Prediction Interval Width Analysis",
        inputSchema: {
            type: "object",
            properties: {
                intervals: { type: "array", items: { type: "object" } },
            },
            required: ["intervals"],
        },
    },
    {
        name: "conformal_adaptive_interval",
        description: "Adaptive Conformal Intervals",
        inputSchema: {
            type: "object",
            properties: {
                residuals: { type: "array", items: { type: "number" } },
                prediction: { type: "number" },
                confidence: { type: "number" },
                decay: { type: "number" },
            },
            required: ["residuals", "prediction", "confidence"],
        },
    },
    {
        name: "conformal_multi_horizon",
        description: "Multi-Horizon Prediction Intervals",
        inputSchema: {
            type: "object",
            properties: {
                residuals_by_horizon: { type: "array", items: { type: "array" } },
                predictions: { type: "array", items: { type: "number" } },
                confidence: { type: "number" },
            },
            required: ["residuals_by_horizon", "predictions", "confidence"],
        },
    },
    {
        name: "conformal_probabilistic",
        description: "Probabilistic Predictions",
        inputSchema: {
            type: "object",
            properties: {
                residuals: { type: "array", items: { type: "number" } },
                prediction: { type: "number" },
                n_quantiles: { type: "number", default: 10 },
            },
            required: ["residuals", "prediction"],
        },
    },
    {
        name: "conformal_classification",
        description: "Conformal Classification Sets",
        inputSchema: {
            type: "object",
            properties: {
                scores: { type: "array", items: { type: "array" } },
                confidence: { type: "number" },
            },
            required: ["scores", "confidence"],
        },
    },
    {
        name: "conformal_split_validate",
        description: "Split Conformal Validation",
        inputSchema: {
            type: "object",
            properties: {
                predictions: { type: "array", items: { type: "number" } },
                actuals: { type: "array", items: { type: "number" } },
                split_ratio: { type: "number", default: 0.5 },
            },
            required: ["predictions", "actuals"],
        },
    },
];

// ============================================================================
// OPTIONS & GREEKS (20 tools)
// ============================================================================

export const greeksTools: Tool[] = [
    {
        name: "greeks_delta",
        description: "Delta: ∂V/∂S",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "greeks_gamma",
        description: "Gamma: ∂²V/∂S²",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry"],
        },
    },
    {
        name: "greeks_theta",
        description: "Theta: ∂V/∂t (daily)",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "greeks_vega",
        description: "Vega: ∂V/∂σ (per 1%)",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry"],
        },
    },
    {
        name: "greeks_rho",
        description: "Rho: ∂V/∂r",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "greeks_vanna",
        description: "Vanna: ∂²V/∂S∂σ",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry"],
        },
    },
    {
        name: "greeks_volga",
        description: "Volga: ∂²V/∂σ²",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry"],
        },
    },
    {
        name: "greeks_charm",
        description: "Charm: ∂²V/∂S∂t",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "options_black_scholes",
        description: "Black-Scholes Pricing",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "options_implied_vol",
        description: "Implied Volatility Solver",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
                market_price: { type: "number" },
            },
            required: ["spot", "strike", "rate", "time_to_expiry", "is_call", "market_price"],
        },
    },
    {
        name: "options_put_call_parity",
        description: "Put-Call Parity Check",
        inputSchema: {
            type: "object",
            properties: {
                call_price: { type: "number" },
                put_price: { type: "number" },
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                time_to_expiry: { type: "number" },
            },
            required: ["call_price", "put_price", "spot", "strike", "rate", "time_to_expiry"],
        },
    },
    {
        name: "options_binomial",
        description: "Binomial Tree Pricing",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                steps: { type: "number" },
                is_call: { type: "boolean" },
                is_american: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "steps", "is_call"],
        },
    },
    {
        name: "options_monte_carlo",
        description: "Monte Carlo Option Pricing",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                simulations: { type: "number", default: 10000 },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "options_american",
        description: "American Option Pricing (LSM)",
        inputSchema: {
            type: "object",
            properties: {
                spot: { type: "number" },
                strike: { type: "number" },
                rate: { type: "number" },
                volatility: { type: "number" },
                time_to_expiry: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"],
        },
    },
    {
        name: "options_surface_fit",
        description: "Volatility Surface Fitting",
        inputSchema: {
            type: "object",
            properties: {
                strikes: { type: "array", items: { type: "number" } },
                expiries: { type: "array", items: { type: "number" } },
                implied_vols: { type: "array", items: { type: "array" } },
            },
            required: ["strikes", "expiries", "implied_vols"],
        },
    },
    {
        name: "options_skew",
        description: "Volatility Skew Analysis",
        inputSchema: {
            type: "object",
            properties: {
                strikes: { type: "array", items: { type: "number" } },
                implied_vols: { type: "array", items: { type: "number" } },
                spot: { type: "number" },
            },
            required: ["strikes", "implied_vols", "spot"],
        },
    },
    {
        name: "options_term_structure",
        description: "Volatility Term Structure",
        inputSchema: {
            type: "object",
            properties: {
                expiries: { type: "array", items: { type: "number" } },
                implied_vols: { type: "array", items: { type: "number" } },
            },
            required: ["expiries", "implied_vols"],
        },
    },
    {
        name: "options_greeks_portfolio",
        description: "Portfolio Greeks Aggregation",
        inputSchema: {
            type: "object",
            properties: {
                positions: { type: "array", items: { type: "object" } },
                spot: { type: "number" },
                rate: { type: "number" },
            },
            required: ["positions", "spot", "rate"],
        },
    },
    {
        name: "options_hedge_ratio",
        description: "Delta Hedge Ratio",
        inputSchema: {
            type: "object",
            properties: {
                option_delta: { type: "number" },
                option_quantity: { type: "number" },
            },
            required: ["option_delta", "option_quantity"],
        },
    },
    {
        name: "options_breakeven",
        description: "Option Breakeven Points",
        inputSchema: {
            type: "object",
            properties: {
                strike: { type: "number" },
                premium: { type: "number" },
                is_call: { type: "boolean" },
            },
            required: ["strike", "premium", "is_call"],
        },
    },
];

// ============================================================================
// ALL TOOLS AGGREGATED
// ============================================================================

export const allTools: Tool[] = [
    ...movingAverageTools,   // 10
    ...momentumTools,        // 18
    ...volatilityTools,      // 10
    ...riskTools,            // 20
    ...portfolioTools,       // 21
    ...executionTools,       // 12
    ...regimeTools,          // 12
    ...conformalTools,       // 10
    ...greeksTools,          // 20
];

export const toolCategories = {
    moving_averages: movingAverageTools.length,
    momentum: momentumTools.length,
    volatility: volatilityTools.length,
    risk: riskTools.length,
    portfolio: portfolioTools.length,
    execution: executionTools.length,
    regime: regimeTools.length,
    conformal: conformalTools.length,
    greeks: greeksTools.length,
};

export const totalToolCount = allTools.length;
