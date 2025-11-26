// Risk Calculations Benchmark
//
// Performance targets:
// - VaR calculation: <2ms
// - CVaR calculation: <3ms
// - Position limit check: <500μs
// - Risk check per signal: <500μs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::{Direction, Symbol};
use nt_portfolio::{Portfolio, Position};
use nt_risk::{RiskManager, RiskMetrics, VarMethod};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn generate_returns(count: usize, volatility: f64) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut returns = Vec::with_capacity(count);

    for _ in 0..count {
        let ret = rng.gen::<f64>() * volatility * 2.0 - volatility;
        returns.push(ret);
    }

    returns
}

fn generate_portfolio_with_positions(position_count: usize) -> Portfolio {
    let mut portfolio = Portfolio::new(dec!(100000));

    for i in 0..position_count {
        let symbol = Symbol::new(&format!("SYM{}", i)).unwrap();
        let position = Position {
            symbol: symbol.clone(),
            quantity: dec!(100),
            avg_entry_price: dec!(100) + Decimal::from(i),
            current_price: dec!(100) + Decimal::from(i) + dec!(5),
            direction: if i % 2 == 0 {
                Direction::Long
            } else {
                Direction::Short
            },
            unrealized_pnl: dec!(500),
            realized_pnl: Decimal::ZERO,
        };

        portfolio.add_position(position);
    }

    portfolio
}

// ============================================================================
// Benchmarks - VaR (Value at Risk) Calculation
// ============================================================================

fn bench_var_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("var_calculation");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [100, 252, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        // Historical VaR
        group.bench_with_input(
            BenchmarkId::new("historical", data_size),
            data_size,
            |b, &data_size| {
                let returns = generate_returns(data_size, 0.02);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_var(
                            black_box(&returns),
                            dec!(0.95),
                            VarMethod::Historical,
                        )
                        .await
                });
            },
        );

        // Parametric VaR
        group.bench_with_input(
            BenchmarkId::new("parametric", data_size),
            data_size,
            |b, &data_size| {
                let returns = generate_returns(data_size, 0.02);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_var(
                            black_box(&returns),
                            dec!(0.95),
                            VarMethod::Parametric,
                        )
                        .await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - CVaR (Conditional Value at Risk)
// ============================================================================

fn bench_cvar_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cvar_calculation");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [100, 252, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let returns = generate_returns(data_size, 0.02);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_cvar(black_box(&returns), dec!(0.95))
                        .await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Position Limit Checks
// ============================================================================

fn bench_position_limit_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_limit_checks");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for position_count in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(position_count),
            position_count,
            |b, &position_count| {
                let portfolio = generate_portfolio_with_positions(position_count);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager.check_position_limits(black_box(&portfolio)).await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Drawdown Calculation
// ============================================================================

fn bench_drawdown_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("drawdown_calculation");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [100, 252, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let mut equity_curve = Vec::with_capacity(data_size);
                let mut equity = 100000.0;

                for i in 0..data_size {
                    equity *= 1.0 + (i as f64 * 0.0001 - 0.05);
                    equity_curve.push(equity);
                }

                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_max_drawdown(black_box(&equity_curve))
                        .await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Sharpe Ratio Calculation
// ============================================================================

fn bench_sharpe_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpe_ratio");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [100, 252, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let returns = generate_returns(data_size, 0.02);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_sharpe_ratio(black_box(&returns), 0.02)
                        .await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Portfolio Risk Metrics (Combined)
// ============================================================================

fn bench_portfolio_risk_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("portfolio_risk_metrics");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for position_count in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(position_count),
            position_count,
            |b, &position_count| {
                let portfolio = generate_portfolio_with_positions(position_count);
                let returns = generate_returns(252, 0.02);
                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    // Calculate all risk metrics
                    let _var = risk_manager
                        .calculate_var(&returns, dec!(0.95), VarMethod::Historical)
                        .await;

                    let _cvar = risk_manager
                        .calculate_cvar(&returns, dec!(0.95))
                        .await;

                    let _sharpe = risk_manager
                        .calculate_sharpe_ratio(&returns, 0.02)
                        .await;

                    let _limits = risk_manager
                        .check_position_limits(&portfolio)
                        .await;
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Stop Loss Calculation
// ============================================================================

fn bench_stop_loss_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stop_loss_calculation");

    group.bench_function("calculate_stop_price", |b| {
        let entry_price = dec!(100);
        let risk_percent = dec!(0.02); // 2% stop loss

        b.iter(|| {
            let stop_price = entry_price * (dec!(1) - risk_percent);
            black_box(stop_price)
        });
    });

    group.bench_function("calculate_position_risk", |b| {
        let entry_price = dec!(100);
        let stop_price = dec!(98);
        let position_size = dec!(100);

        b.iter(|| {
            let risk_per_share = entry_price - stop_price;
            let total_risk = risk_per_share * position_size;
            black_box(total_risk)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Correlation Matrix Calculation
// ============================================================================

fn bench_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");
    group.sample_size(30);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for asset_count in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(asset_count),
            asset_count,
            |b, &asset_count| {
                // Generate returns for multiple assets
                let returns_matrix: Vec<Vec<f64>> = (0..asset_count)
                    .map(|_| generate_returns(252, 0.02))
                    .collect();

                let risk_manager = RiskManager::new();

                b.to_async(&rt).iter(|| async {
                    risk_manager
                        .calculate_correlation_matrix(black_box(&returns_matrix))
                        .await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Risk-Adjusted Position Sizing
// ============================================================================

fn bench_risk_adjusted_sizing(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_adjusted_sizing");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("kelly_criterion", |b| {
        let win_rate = 0.55;
        let avg_win = 1.5;
        let avg_loss = 1.0;
        let risk_manager = RiskManager::new();

        b.to_async(&rt).iter(|| async {
            risk_manager
                .calculate_kelly_fraction(win_rate, avg_win, avg_loss)
                .await
        });
    });

    group.bench_function("fixed_fractional", |b| {
        let account_size = dec!(100000);
        let risk_per_trade = dec!(0.02); // 2%
        let risk_manager = RiskManager::new();

        b.to_async(&rt).iter(|| async {
            let position_size = account_size * risk_per_trade;
            black_box(position_size)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_var_calculation,
        bench_cvar_calculation,
        bench_position_limit_checks,
        bench_drawdown_calculation,
        bench_sharpe_ratio,
        bench_portfolio_risk_metrics,
        bench_stop_loss_calculation,
        bench_correlation_matrix,
        bench_risk_adjusted_sizing
}

criterion_main!(benches);
