use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hedge_algorithms::*;
use std::collections::HashMap;

fn benchmark_hedge_algorithm(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let hedge = HedgeAlgorithms::new(config).unwrap();
    
    let market_data = MarketData::new(
        "BTCUSD".to_string(),
        chrono::Utc::now(),
        [100.0, 105.0, 95.0, 102.0, 1000.0]
    );
    
    c.bench_function("hedge_algorithm_update", |b| {
        b.iter(|| {
            hedge.update_market_data(black_box(&market_data)).unwrap();
        })
    });
    
    c.bench_function("hedge_algorithm_recommendation", |b| {
        b.iter(|| {
            hedge.get_hedge_recommendation().unwrap();
        })
    });
}

fn benchmark_quantum_hedge(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let expert_names = vec!["expert1".to_string(), "expert2".to_string(), "expert3".to_string()];
    let mut quantum_hedge = QuantumHedgeAlgorithm::new(expert_names, config).unwrap();
    
    let market_data = MarketData::new(
        "BTCUSD".to_string(),
        chrono::Utc::now(),
        [100.0, 105.0, 95.0, 102.0, 1000.0]
    );
    
    let mut predictions = HashMap::new();
    predictions.insert("expert1".to_string(), 0.05);
    predictions.insert("expert2".to_string(), -0.02);
    predictions.insert("expert3".to_string(), 0.03);
    
    c.bench_function("quantum_hedge_update", |b| {
        b.iter(|| {
            quantum_hedge.update(black_box(&market_data), black_box(&predictions)).unwrap();
        })
    });
    
    c.bench_function("quantum_hedge_recommendation", |b| {
        b.iter(|| {
            quantum_hedge.get_recommendation().unwrap();
        })
    });
}

fn benchmark_factor_model(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let mut factor_model = StandardFactorModel::new(config).unwrap();
    
    let market_data = MarketData::new(
        "BTCUSD".to_string(),
        chrono::Utc::now(),
        [100.0, 105.0, 95.0, 102.0, 1000.0]
    );
    
    c.bench_function("factor_model_update", |b| {
        b.iter(|| {
            factor_model.update(black_box(&market_data)).unwrap();
        })
    });
    
    c.bench_function("factor_model_exposures", |b| {
        b.iter(|| {
            factor_model.get_exposures().unwrap();
        })
    });
}

fn benchmark_options_hedger(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let hedger = OptionsHedger::new(config);
    
    c.bench_function("options_black_scholes", |b| {
        b.iter(|| {
            hedger.black_scholes_price(
                black_box(100.0),
                black_box(100.0),
                black_box(1.0),
                black_box(0.2),
                black_box(OptionType::Call)
            ).unwrap();
        })
    });
    
    c.bench_function("options_greeks", |b| {
        b.iter(|| {
            hedger.calculate_greeks(
                black_box(100.0),
                black_box(100.0),
                black_box(1.0),
                black_box(0.2),
                black_box(OptionType::Call)
            ).unwrap();
        })
    });
}

fn benchmark_pairs_trader(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let mut pairs_trader = PairsTrader::new(config);
    
    c.bench_function("pairs_trader_update", |b| {
        b.iter(|| {
            pairs_trader.update(black_box(100.0), black_box(98.0)).unwrap();
        })
    });
    
    c.bench_function("pairs_trader_signal", |b| {
        b.iter(|| {
            pairs_trader.generate_signal().unwrap();
        })
    });
}

fn benchmark_whale_detector(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let mut whale_detector = WhaleDetector::new(config);
    
    let market_data = MarketData::new(
        "BTCUSD".to_string(),
        chrono::Utc::now(),
        [100.0, 105.0, 95.0, 102.0, 1000.0]
    );
    
    c.bench_function("whale_detector_update", |b| {
        b.iter(|| {
            whale_detector.update(black_box(&market_data)).unwrap();
        })
    });
    
    c.bench_function("whale_detector_signal", |b| {
        b.iter(|| {
            whale_detector.get_trading_signal().unwrap();
        })
    });
}

fn benchmark_regret_minimizer(c: &mut Criterion) {
    let config = HedgeConfig::default();
    let mut regret_minimizer = RegretMinimizer::new(config);
    
    regret_minimizer.initialize_expert("expert1").unwrap();
    regret_minimizer.initialize_expert("expert2").unwrap();
    
    let mut expert_predictions = HashMap::new();
    expert_predictions.insert("expert1".to_string(), 0.05);
    expert_predictions.insert("expert2".to_string(), -0.02);
    
    c.bench_function("regret_minimizer_update", |b| {
        b.iter(|| {
            regret_minimizer.update_external_regret(
                black_box(&expert_predictions),
                black_box(0.015),
                black_box(0.03)
            ).unwrap();
        })
    });
    
    c.bench_function("regret_minimizer_statistics", |b| {
        b.iter(|| {
            regret_minimizer.get_regret_statistics();
        })
    });
}

fn benchmark_math_utils(c: &mut Criterion) {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let data2 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    
    c.bench_function("math_correlation", |b| {
        b.iter(|| {
            utils::math::correlation(black_box(&data), black_box(&data2)).unwrap();
        })
    });
    
    c.bench_function("math_variance", |b| {
        b.iter(|| {
            utils::math::variance(black_box(&data)).unwrap();
        })
    });
    
    c.bench_function("math_returns", |b| {
        b.iter(|| {
            utils::math::returns(black_box(&data)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    benchmark_hedge_algorithm,
    benchmark_quantum_hedge,
    benchmark_factor_model,
    benchmark_options_hedger,
    benchmark_pairs_trader,
    benchmark_whale_detector,
    benchmark_regret_minimizer,
    benchmark_math_utils
);
criterion_main!(benches);