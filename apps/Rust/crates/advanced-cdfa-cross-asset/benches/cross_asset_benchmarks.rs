//! Cross-Asset Analysis Benchmarks
//!
//! This benchmark suite measures the performance of cross-asset analysis algorithms
//! including correlation, contagion, and systemic risk analysis.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Mock cross-asset analysis framework
#[derive(Debug, Clone)]
pub struct CrossAssetAnalyzer {
    pub correlation_engine: CorrelationEngine,
    pub contagion_detector: ContagionDetector,
    pub systemic_risk_monitor: SystemicRiskMonitor,
    pub network_analyzer: NetworkAnalyzer,
}

#[derive(Debug, Clone)]
pub struct CorrelationEngine {
    pub correlation_matrix: Vec<Vec<f64>>,
    pub rolling_window: usize,
}

#[derive(Debug, Clone)]
pub struct ContagionDetector {
    pub contagion_threshold: f64,
    pub detection_window: usize,
}

#[derive(Debug, Clone)]
pub struct SystemicRiskMonitor {
    pub risk_metrics: HashMap<String, f64>,
    pub stress_scenarios: Vec<StressScenario>,
}

#[derive(Debug, Clone)]
pub struct NetworkAnalyzer {
    pub adjacency_matrix: Vec<Vec<f64>>,
    pub centrality_measures: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub shock_magnitude: f64,
    pub affected_assets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AssetData {
    pub symbol: String,
    pub prices: Vec<f64>,
    pub returns: Vec<f64>,
    pub volatility: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct CorrelationResult {
    pub pearson_correlation: f64,
    pub spearman_correlation: f64,
    pub kendall_correlation: f64,
    pub rolling_correlation: Vec<f64>,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct ContagionResult {
    pub contagion_detected: bool,
    pub contagion_strength: f64,
    pub propagation_path: Vec<String>,
    pub affected_assets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SystemicRiskResult {
    pub overall_risk_score: f64,
    pub individual_risks: HashMap<String, f64>,
    pub interconnectedness: f64,
    pub stability_index: f64,
}

impl CrossAssetAnalyzer {
    pub fn new() -> Self {
        Self {
            correlation_engine: CorrelationEngine::new(),
            contagion_detector: ContagionDetector::new(),
            systemic_risk_monitor: SystemicRiskMonitor::new(),
            network_analyzer: NetworkAnalyzer::new(),
        }
    }
    
    pub async fn analyze_correlation(&self, assets: &[AssetData]) -> CorrelationResult {
        // Mock implementation
        tokio::time::sleep(Duration::from_micros(10)).await;
        
        CorrelationResult {
            pearson_correlation: 0.75,
            spearman_correlation: 0.72,
            kendall_correlation: 0.68,
            rolling_correlation: vec![0.7, 0.75, 0.8, 0.72],
            confidence_interval: (0.65, 0.85),
        }
    }
    
    pub async fn detect_contagion(&self, assets: &[AssetData]) -> ContagionResult {
        // Mock implementation
        tokio::time::sleep(Duration::from_micros(15)).await;
        
        ContagionResult {
            contagion_detected: true,
            contagion_strength: 0.85,
            propagation_path: vec!["SPY".to_string(), "QQQ".to_string(), "IWM".to_string()],
            affected_assets: vec!["AAPL".to_string(), "MSFT".to_string(), "GOOGL".to_string()],
        }
    }
    
    pub async fn analyze_systemic_risk(&self, assets: &[AssetData]) -> SystemicRiskResult {
        // Mock implementation
        tokio::time::sleep(Duration::from_micros(20)).await;
        
        let mut individual_risks = HashMap::new();
        for asset in assets {
            individual_risks.insert(asset.symbol.clone(), 0.65);
        }
        
        SystemicRiskResult {
            overall_risk_score: 0.7,
            individual_risks,
            interconnectedness: 0.8,
            stability_index: 0.3,
        }
    }
    
    pub async fn network_analysis(&self, assets: &[AssetData]) -> NetworkResult {
        // Mock implementation
        tokio::time::sleep(Duration::from_micros(25)).await;
        
        NetworkResult {
            clustering_coefficient: 0.6,
            path_length: 2.5,
            modularity: 0.4,
            communities: vec![
                vec!["AAPL".to_string(), "MSFT".to_string()],
                vec!["JPM".to_string(), "BAC".to_string()],
            ],
        }
    }
    
    pub fn clone(&self) -> Self {
        Self {
            correlation_engine: self.correlation_engine.clone(),
            contagion_detector: self.contagion_detector.clone(),
            systemic_risk_monitor: self.systemic_risk_monitor.clone(),
            network_analyzer: self.network_analyzer.clone(),
        }
    }
}

impl CorrelationEngine {
    pub fn new() -> Self {
        Self {
            correlation_matrix: vec![vec![1.0; 100]; 100],
            rolling_window: 252,
        }
    }
}

impl ContagionDetector {
    pub fn new() -> Self {
        Self {
            contagion_threshold: 0.8,
            detection_window: 20,
        }
    }
}

impl SystemicRiskMonitor {
    pub fn new() -> Self {
        Self {
            risk_metrics: HashMap::new(),
            stress_scenarios: vec![
                StressScenario {
                    name: "Market Crash".to_string(),
                    shock_magnitude: -0.2,
                    affected_assets: vec!["SPY".to_string(), "QQQ".to_string()],
                },
                StressScenario {
                    name: "Interest Rate Shock".to_string(),
                    shock_magnitude: 0.05,
                    affected_assets: vec!["TLT".to_string(), "IEF".to_string()],
                },
            ],
        }
    }
}

impl NetworkAnalyzer {
    pub fn new() -> Self {
        Self {
            adjacency_matrix: vec![vec![0.0; 100]; 100],
            centrality_measures: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkResult {
    pub clustering_coefficient: f64,
    pub path_length: f64,
    pub modularity: f64,
    pub communities: Vec<Vec<String>>,
}

/// Create sample asset data for benchmarking
fn create_sample_asset_data(symbol: &str, size: usize) -> AssetData {
    let prices: Vec<f64> = (0..size)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
        .collect();
    
    let returns: Vec<f64> = prices.windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
        .collect();
    
    let volatility = returns.iter().map(|r| r * r).sum::<f64>().sqrt() / returns.len() as f64;
    
    AssetData {
        symbol: symbol.to_string(),
        prices,
        returns,
        volatility,
        timestamp: 1640995200000,
    }
}

/// Create multiple asset data samples
fn create_multiple_asset_data(count: usize, size: usize) -> Vec<AssetData> {
    let symbols = vec![
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO", "TLT",
        "IEF", "SHY", "GLD", "SLV", "OIL", "GAS", "BTC", "ETH",
    ];
    
    (0..count)
        .map(|i| create_sample_asset_data(symbols[i % symbols.len()], size))
        .collect()
}

/// Benchmark cross-asset analyzer initialization
fn bench_analyzer_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyzer_initialization");
    
    group.bench_function("create_analyzer", |b| {
        b.iter(|| {
            let analyzer = CrossAssetAnalyzer::new();
            black_box(analyzer)
        });
    });
    
    group.finish();
}

/// Benchmark correlation analysis
fn bench_correlation_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("correlation_analysis");
    
    let asset_pairs = vec![2, 5, 10, 25, 50, 100];
    
    for pair_count in asset_pairs {
        group.throughput(Throughput::Elements((pair_count * (pair_count - 1) / 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("asset_pairs", pair_count),
            &pair_count,
            |b, &pair_count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(pair_count, 252);
                        (analyzer, assets)
                    },
                    |(analyzer, assets)| async move {
                        let result = analyzer.analyze_correlation(&assets).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark contagion detection
fn bench_contagion_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("contagion_detection");
    
    let asset_counts = vec![10, 25, 50, 100, 200];
    
    for count in asset_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("asset_count", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(count, 252);
                        (analyzer, assets)
                    },
                    |(analyzer, assets)| async move {
                        let result = analyzer.detect_contagion(&assets).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark systemic risk analysis
fn bench_systemic_risk_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("systemic_risk_analysis");
    
    let portfolio_sizes = vec![20, 50, 100, 250, 500];
    
    for size in portfolio_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("portfolio_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(size, 252);
                        (analyzer, assets)
                    },
                    |(analyzer, assets)| async move {
                        let result = analyzer.analyze_systemic_risk(&assets).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark network analysis
fn bench_network_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("network_analysis");
    
    let network_sizes = vec![10, 25, 50, 100, 200];
    
    for size in network_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("network_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(size, 252);
                        (analyzer, assets)
                    },
                    |(analyzer, assets)| async move {
                        let result = analyzer.network_analysis(&assets).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark different correlation methods
fn bench_correlation_methods(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("correlation_methods");
    
    let methods = vec![
        ("pearson", "pearson"),
        ("spearman", "spearman"),
        ("kendall", "kendall"),
        ("rolling", "rolling"),
    ];
    
    for (method_name, method) in methods {
        group.bench_with_input(
            BenchmarkId::new("method", method_name),
            &method,
            |b, &method| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(50, 252);
                        (analyzer, assets, method)
                    },
                    |(analyzer, assets, method)| async move {
                        let result = analyzer.correlation_by_method(&assets, method).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark stress testing scenarios
fn bench_stress_testing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("stress_testing");
    
    let scenario_counts = vec![1, 5, 10, 25, 50];
    
    for count in scenario_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("scenario_count", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(100, 252);
                        let scenarios: Vec<StressScenario> = (0..count)
                            .map(|i| StressScenario {
                                name: format!("Scenario {}", i),
                                shock_magnitude: -0.1 * (i as f64 + 1.0),
                                affected_assets: vec!["SPY".to_string(), "QQQ".to_string()],
                            })
                            .collect();
                        (analyzer, assets, scenarios)
                    },
                    |(analyzer, assets, scenarios)| async move {
                        let result = analyzer.stress_test(&assets, &scenarios).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark time series data sizes
fn bench_time_series_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("time_series_sizes");
    
    let data_sizes = vec![100, 252, 500, 1000, 2500, 5000];
    
    for size in data_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("data_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(20, size);
                        (analyzer, assets)
                    },
                    |(analyzer, assets)| async move {
                        let result = analyzer.analyze_correlation(&assets).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent analysis
fn bench_concurrent_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_analysis");
    
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrency", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzers: Vec<CrossAssetAnalyzer> = (0..concurrency)
                            .map(|_| CrossAssetAnalyzer::new())
                            .collect();
                        let assets = create_multiple_asset_data(50, 252);
                        (analyzers, assets)
                    },
                    |(analyzers, assets)| async move {
                        let mut handles = Vec::new();
                        
                        for analyzer in analyzers {
                            let assets_clone = assets.clone();
                            let handle = tokio::spawn(async move {
                                analyzer.analyze_correlation(&assets_clone).await
                            });
                            handles.push(handle);
                        }
                        
                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark rolling window analysis
fn bench_rolling_window_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("rolling_window_analysis");
    
    let window_sizes = vec![20, 50, 100, 252, 500];
    
    for window_size in window_sizes {
        group.bench_with_input(
            BenchmarkId::new("window_size", window_size),
            &window_size,
            |b, &window_size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(20, 1000);
                        (analyzer, assets, window_size)
                    },
                    |(analyzer, assets, window_size)| async move {
                        let result = analyzer.rolling_correlation(&assets, window_size).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark lead-lag analysis
fn bench_lead_lag_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lead_lag_analysis");
    
    let lag_depths = vec![1, 5, 10, 20, 50];
    
    for depth in lag_depths {
        group.bench_with_input(
            BenchmarkId::new("lag_depth", depth),
            &depth,
            |b, &depth| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(20, 500);
                        (analyzer, assets, depth)
                    },
                    |(analyzer, assets, depth)| async move {
                        let result = analyzer.lead_lag_analysis(&assets, depth).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark regime-dependent analysis
fn bench_regime_dependent_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("regime_dependent_analysis");
    
    let regime_counts = vec![2, 3, 4, 5];
    
    for count in regime_counts {
        group.bench_with_input(
            BenchmarkId::new("regime_count", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let analyzer = CrossAssetAnalyzer::new();
                        let assets = create_multiple_asset_data(30, 500);
                        (analyzer, assets, count)
                    },
                    |(analyzer, assets, count)| async move {
                        let result = analyzer.regime_dependent_correlation(&assets, count).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

// Additional method implementations for the analyzer
impl CrossAssetAnalyzer {
    async fn correlation_by_method(&self, _assets: &[AssetData], _method: &str) -> CorrelationResult {
        tokio::time::sleep(Duration::from_micros(8)).await;
        CorrelationResult {
            pearson_correlation: 0.75,
            spearman_correlation: 0.72,
            kendall_correlation: 0.68,
            rolling_correlation: vec![0.7, 0.75, 0.8, 0.72],
            confidence_interval: (0.65, 0.85),
        }
    }
    
    async fn stress_test(&self, _assets: &[AssetData], _scenarios: &[StressScenario]) -> Vec<SystemicRiskResult> {
        tokio::time::sleep(Duration::from_micros(30)).await;
        vec![SystemicRiskResult {
            overall_risk_score: 0.8,
            individual_risks: HashMap::new(),
            interconnectedness: 0.9,
            stability_index: 0.2,
        }]
    }
    
    async fn rolling_correlation(&self, _assets: &[AssetData], _window_size: usize) -> Vec<CorrelationResult> {
        tokio::time::sleep(Duration::from_micros(20)).await;
        vec![CorrelationResult {
            pearson_correlation: 0.75,
            spearman_correlation: 0.72,
            kendall_correlation: 0.68,
            rolling_correlation: vec![0.7, 0.75, 0.8, 0.72],
            confidence_interval: (0.65, 0.85),
        }]
    }
    
    async fn lead_lag_analysis(&self, _assets: &[AssetData], _depth: usize) -> LeadLagResult {
        tokio::time::sleep(Duration::from_micros(25)).await;
        LeadLagResult {
            lead_relationships: HashMap::new(),
            lag_coefficients: vec![0.1, 0.2, 0.15, 0.05],
            optimal_lag: 2,
        }
    }
    
    async fn regime_dependent_correlation(&self, _assets: &[AssetData], _regime_count: usize) -> Vec<CorrelationResult> {
        tokio::time::sleep(Duration::from_micros(40)).await;
        vec![CorrelationResult {
            pearson_correlation: 0.75,
            spearman_correlation: 0.72,
            kendall_correlation: 0.68,
            rolling_correlation: vec![0.7, 0.75, 0.8, 0.72],
            confidence_interval: (0.65, 0.85),
        }]
    }
}

#[derive(Debug, Clone)]
pub struct LeadLagResult {
    pub lead_relationships: HashMap<String, String>,
    pub lag_coefficients: Vec<f64>,
    pub optimal_lag: usize,
}

// Helper trait implementations
impl AssetData {
    fn clone(&self) -> Self {
        Self {
            symbol: self.symbol.clone(),
            prices: self.prices.clone(),
            returns: self.returns.clone(),
            volatility: self.volatility,
            timestamp: self.timestamp,
        }
    }
}

criterion_group!(
    benches,
    bench_analyzer_initialization,
    bench_correlation_analysis,
    bench_contagion_detection,
    bench_systemic_risk_analysis,
    bench_network_analysis,
    bench_correlation_methods,
    bench_stress_testing,
    bench_time_series_sizes,
    bench_concurrent_analysis,
    bench_rolling_window_analysis,
    bench_lead_lag_analysis,
    bench_regime_dependent_analysis
);

criterion_main!(benches);