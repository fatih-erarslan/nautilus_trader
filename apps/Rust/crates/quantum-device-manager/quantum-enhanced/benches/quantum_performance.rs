use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_enhanced::{QuantumPatternEngine, QuantumConfig};
use quantum_enhanced::types::*;
use std::collections::HashMap;
use chrono::Utc;
use ndarray::Array1;

fn benchmark_quantum_pattern_detection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let engine = rt.block_on(QuantumPatternEngine::new(config)).unwrap();
    
    let mut group = c.benchmark_group("quantum_pattern_detection");
    
    for num_instruments in [2, 5, 10, 20].iter() {
        let market_data = create_test_market_data(*num_instruments, 100);
        
        group.bench_with_input(
            BenchmarkId::new("detect_patterns", num_instruments),
            num_instruments,
            |b, _| {
                b.to_async(&rt).iter(|| {
                    engine.detect_quantum_patterns(&market_data)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_ensemble_detection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let engine = rt.block_on(QuantumPatternEngine::new(config)).unwrap();
    
    let market_data = create_test_market_data(5, 100);
    
    c.bench_function("ensemble_detection", |b| {
        b.to_async(&rt).iter(|| {
            engine.detect_ensemble_patterns(&market_data)
        });
    });
}

fn benchmark_superposition_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let detector = rt.block_on(
        quantum_enhanced::quantum_superposition::QuantumSuperposition::new(&config)
    ).unwrap();
    
    let mut group = c.benchmark_group("superposition_creation");
    
    for data_points in [50, 100, 200, 500].iter() {
        let market_data = create_test_market_data(5, *data_points);
        
        group.bench_with_input(
            BenchmarkId::new("create_superposition", data_points),
            data_points,
            |b, _| {
                b.to_async(&rt).iter(|| {
                    detector.create_superposition(&market_data)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_entanglement_detection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let detector = rt.block_on(
        quantum_enhanced::quantum_entanglement::QuantumEntanglement::new(&config)
    ).unwrap();
    
    // Create test quantum data
    let quantum_data = create_test_quantum_data(8, 256);
    
    c.bench_function("entanglement_detection", |b| {
        b.to_async(&rt).iter(|| {
            detector.find_entangled_correlations(&quantum_data)
        });
    });
}

fn benchmark_quantum_fourier_transform(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let qft_engine = rt.block_on(
        quantum_enhanced::quantum_fourier::QuantumFourierTransform::new(&config)
    ).unwrap();
    
    let entanglement = create_test_entanglement_correlation();
    
    c.bench_function("quantum_fourier_transform", |b| {
        b.to_async(&rt).iter(|| {
            qft_engine.transform(&entanglement)
        });
    });
}

fn benchmark_classical_conversion(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let config = QuantumConfig::default();
    let interface = rt.block_on(
        quantum_enhanced::ClassicalInterface::new(config)
    ).unwrap();
    
    let quantum_signal = create_test_quantum_signal();
    let market_data = create_test_market_data(2, 50);
    
    c.bench_function("classical_conversion", |b| {
        b.to_async(&rt).iter(|| {
            interface.convert_to_trading_signal(&quantum_signal, &market_data)
        });
    });
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    for superposition_states in [128, 256, 512, 1024].iter() {
        let mut config = QuantumConfig::default();
        config.max_superposition_states = *superposition_states;
        
        group.bench_with_input(
            BenchmarkId::new("pattern_engine_creation", superposition_states),
            superposition_states,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let _engine = QuantumPatternEngine::new(config.clone()).await.unwrap();
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions to create test data

fn create_test_market_data(num_instruments: usize, data_points: usize) -> MarketData {
    let mut price_history = HashMap::new();
    let mut volume_data = HashMap::new();
    
    for i in 0..num_instruments {
        let instrument = format!("INST{}", i);
        
        let mut prices = Vec::with_capacity(data_points);
        let mut volumes = Vec::with_capacity(data_points);
        
        let base_price = 1000.0 + i as f64 * 100.0;
        
        for j in 0..data_points {
            let noise = (j as f64 * 0.1).sin() * 0.02;
            prices.push(base_price * (1.0 + noise));
            volumes.push(1000.0 + j as f64 * 10.0);
        }
        
        price_history.insert(instrument.clone(), prices);
        volume_data.insert(instrument, volumes);
    }
    
    MarketData {
        price_history,
        volume_data,
        timestamps: (0..data_points).map(|_| Utc::now()).collect(),
        features: ndarray::Array2::zeros((data_points, num_instruments)),
        regime_indicators: Array1::zeros(data_points),
    }
}

fn create_test_quantum_data(num_instruments: usize, num_states: usize) -> QuantumMarketData {
    use num_complex::Complex64;
    use ndarray::Array2;
    
    let superposition_states = Array2::from_shape_fn((num_states, num_instruments), |(i, j)| {
        let phase = (i + j) as f64 * 0.1;
        Complex64::new(phase.cos(), phase.sin()) / (num_states as f64).sqrt()
    });
    
    let amplitudes = Array1::from_shape_fn(num_states, |i| {
        Complex64::new(1.0 / (num_states as f64).sqrt(), 0.0)
    });
    
    let entanglement_matrix = Array2::from_shape_fn((num_instruments, num_instruments), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.5, 0.1)
        }
    });
    
    let phase_matrix = ndarray::Array2::zeros((num_states, num_instruments));
    
    QuantumMarketData {
        superposition_states,
        amplitudes,
        entanglement_matrix,
        phase_matrix,
        classical_data: create_test_market_data(num_instruments, 50),
        coherence_time_ms: 1000.0,
    }
}

fn create_test_entanglement_correlation() -> quantum_enhanced::quantum_entanglement::EntanglementCorrelation {
    use num_complex::Complex64;
    use ndarray::{Array1, Array2};
    
    quantum_enhanced::quantum_entanglement::EntanglementCorrelation {
        strength: 0.8,
        entangled_pairs: vec![("BTC".to_string(), "ETH".to_string())],
        correlation_matrix: Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(0.8, 0.1),
            Complex64::new(0.8, -0.1), Complex64::new(1.0, 0.0),
        ]).unwrap(),
        bell_coefficients: Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]),
        fidelity: 0.9,
        decoherence_rate: 0.1,
    }
}

fn create_test_quantum_signal() -> QuantumSignal {
    let mut signal = QuantumSignal::new(
        0.8, 
        0.9, 
        QuantumPatternType::SuperpositionMomentum, 
        0.85
    );
    signal.affected_instruments = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    signal
}

criterion_group!(
    benches,
    benchmark_quantum_pattern_detection,
    benchmark_ensemble_detection,
    benchmark_superposition_creation,
    benchmark_entanglement_detection,
    benchmark_quantum_fourier_transform,
    benchmark_classical_conversion,
    benchmark_memory_usage
);

criterion_main!(benches);