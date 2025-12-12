//! Q* Algorithm Core Benchmarks
//!
//! This benchmark suite measures the performance of the Q* algorithm
//! across different scenarios and configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_star_core::{
    QStarEngine, QStarConfig, QStarAgent, QStarAction, QStarError,
    MarketState, Experience, Policy, ValueFunction, SearchTree,
    ExperienceMemory, ExplorerAgent, ExploiterAgent, CoordinatorAgent,
    CoordinationStrategy, CoordinationResult, QStarSearchResult,
    AgentStats, MarketRegime,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Create a test market state for benchmarking
fn create_test_market_state() -> MarketState {
    MarketState::new(
        50000.0,     // price
        1000000.0,   // volume
        0.02,        // volatility
        0.5,         // momentum
        0.001,       // spread
        MarketRegime::Trending,
        vec![0.1, 0.2, 0.3, 0.4, 0.5], // features
    )
}

/// Create a test experience for training
fn create_test_experience() -> Experience {
    Experience {
        state: create_test_market_state(),
        action: QStarAction::Buy { amount: 1000.0 },
        reward: 0.1,
        next_state: create_test_market_state(),
        done: false,
    }
}

/// Create test Q* configuration
fn create_test_config() -> QStarConfig {
    QStarConfig {
        learning_rate: 0.001,
        discount_factor: 0.99,
        exploration_rate: 0.1,
        exploration_decay: 0.999,
        min_exploration: 0.01,
        max_latency_us: 10,
        min_accuracy: 0.95,
        batch_size: 32,
        replay_buffer_size: 10000,
        target_update_frequency: 100,
        gradient_clip_norm: 1.0,
        network_architecture: vec![256, 128, 64],
        use_double_q: true,
        use_dueling: true,
        use_prioritized_replay: true,
        use_noisy_networks: false,
        use_rainbow: false,
        quantum_enhancement: true,
        neural_integration: true,
        daa_coordination: true,
    }
}

/// Create mock neural network components
fn create_mock_policy() -> Arc<MockPolicy> {
    Arc::new(MockPolicy::new())
}

fn create_mock_value_function() -> Arc<MockValueFunction> {
    Arc::new(MockValueFunction::new())
}

fn create_mock_experience_memory() -> Arc<MockExperienceMemory> {
    Arc::new(MockExperienceMemory::new())
}

fn create_mock_search_tree() -> Arc<MockSearchTree> {
    Arc::new(MockSearchTree::new())
}

/// Benchmark Q* engine creation
fn bench_q_star_engine_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("q_star_engine_creation");
    
    group.bench_function("create_engine", |b| {
        b.to_async(&rt).iter(|| async {
            let config = create_test_config();
            let policy = create_mock_policy();
            let value = create_mock_value_function();
            let memory = create_mock_experience_memory();
            let search = create_mock_search_tree();
            
            let engine = QStarEngine::new(config, policy, value, memory, search);
            black_box(engine)
        });
    });
    
    group.finish();
}

/// Benchmark Q* decision making
fn bench_q_star_decision_making(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("q_star_decision_making");
    
    let batch_sizes = vec![1, 8, 32, 64, 128];
    
    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let policy = create_mock_policy();
                        let value = create_mock_value_function();
                        let memory = create_mock_experience_memory();
                        let search = create_mock_search_tree();
                        
                        let engine = QStarEngine::new(config, policy, value, memory, search);
                        let states: Vec<MarketState> = (0..batch_size)
                            .map(|_| create_test_market_state())
                            .collect();
                        (engine, states)
                    },
                    |(engine, states)| async move {
                        let mut results = Vec::new();
                        for state in states {
                            let result = engine.decide(&state).await;
                            results.push(result);
                        }
                        black_box(results)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark Q* training
fn bench_q_star_training(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("q_star_training");
    
    let experience_counts = vec![1, 10, 50, 100, 500];
    
    for count in experience_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("experiences", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let config = create_test_config();
                        let policy = create_mock_policy();
                        let value = create_mock_value_function();
                        let memory = create_mock_experience_memory();
                        let search = create_mock_search_tree();
                        
                        let engine = QStarEngine::new(config, policy, value, memory, search);
                        let experiences: Vec<Experience> = (0..count)
                            .map(|_| create_test_experience())
                            .collect();
                        (engine, experiences)
                    },
                    |(engine, experiences)| async move {
                        let result = engine.train(&experiences).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark different agent types
fn bench_agent_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("agent_types");
    
    let agent_types = vec![
        ("explorer", "explorer"),
        ("exploiter", "exploiter"),
        ("coordinator", "coordinator"),
    ];
    
    for (agent_name, agent_type) in agent_types {
        group.bench_with_input(
            BenchmarkId::new("agent_type", agent_name),
            &agent_type,
            |b, &agent_type| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let state = create_test_market_state();
                        let agent: Arc<dyn QStarAgent + Send + Sync> = match agent_type {
                            "explorer" => Arc::new(ExplorerAgent::new("test_explorer".to_string(), 0.3)),
                            "exploiter" => Arc::new(ExploiterAgent::new("test_exploiter".to_string(), 0.8)),
                            "coordinator" => Arc::new(CoordinatorAgent::new(
                                "test_coordinator".to_string(),
                                CoordinationStrategy::WeightedAverage,
                            )),
                            _ => Arc::new(ExplorerAgent::new("default".to_string(), 0.1)),
                        };
                        (agent, state)
                    },
                    |(agent, state)| async move {
                        let result = agent.q_star_search(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark coordination strategies
fn bench_coordination_strategies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("coordination_strategies");
    
    let strategies = vec![
        ("weighted_average", CoordinationStrategy::WeightedAverage),
        ("consensus", CoordinationStrategy::Consensus),
        ("hierarchical", CoordinationStrategy::Hierarchical),
        ("dynamic", CoordinationStrategy::Dynamic),
    ];
    
    for (strategy_name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::new("strategy", strategy_name),
            &strategy,
            |b, &strategy| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let coordinator = CoordinatorAgent::new("test".to_string(), strategy.clone());
                        let state = create_test_market_state();
                        
                        // Create multiple search results for coordination
                        let search_results = vec![
                            QStarSearchResult {
                                action: QStarAction::Buy { amount: 1000.0 },
                                q_value: 0.8,
                                confidence: 0.9,
                                search_depth: 5,
                            },
                            QStarSearchResult {
                                action: QStarAction::Sell { amount: 500.0 },
                                q_value: 0.6,
                                confidence: 0.7,
                                search_depth: 3,
                            },
                            QStarSearchResult {
                                action: QStarAction::Hold,
                                q_value: 0.4,
                                confidence: 0.5,
                                search_depth: 2,
                            },
                        ];
                        
                        (coordinator, state, search_results)
                    },
                    |(coordinator, state, search_results)| async move {
                        let result = coordinator.coordinate(&state, &search_results).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_operations");
    
    let memory_sizes = vec![100, 1000, 5000, 10000, 50000];
    
    for size in memory_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let memory = create_mock_experience_memory();
                        let experiences: Vec<Experience> = (0..size)
                            .map(|_| create_test_experience())
                            .collect();
                        (memory, experiences)
                    },
                    |(memory, experiences)| async move {
                        // Store experiences
                        for exp in experiences {
                            let _ = memory.store(exp).await;
                        }
                        
                        // Sample batch
                        let result = memory.sample(32).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark search tree operations
fn bench_search_tree_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("search_tree_operations");
    
    let search_depths = vec![1, 3, 5, 10, 20];
    
    for depth in search_depths {
        group.bench_with_input(
            BenchmarkId::new("search_depth", depth),
            &depth,
            |b, &depth| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let search_tree = create_mock_search_tree();
                        let state = create_test_market_state();
                        (search_tree, state)
                    },
                    |(search_tree, state)| async move {
                        let _ = search_tree.initialize(&state).await;
                        
                        // Perform multiple expansions
                        for _ in 0..depth {
                            let action = QStarAction::Buy { amount: 1000.0 };
                            let _ = search_tree.expand(&state, &action).await;
                        }
                        
                        let result = search_tree.get_best_path().await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark policy evaluation
fn bench_policy_evaluation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("policy_evaluation");
    
    let state_sizes = vec![10, 50, 100, 500, 1000];
    
    for size in state_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("state_size", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let policy = create_mock_policy();
                        let states: Vec<MarketState> = (0..size)
                            .map(|_| create_test_market_state())
                            .collect();
                        (policy, states)
                    },
                    |(policy, states)| async move {
                        let mut results = Vec::new();
                        for state in states {
                            let result = policy.evaluate_state(&state).await;
                            results.push(result);
                        }
                        black_box(results)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark value function estimation
fn bench_value_function_estimation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("value_function_estimation");
    
    let batch_sizes = vec![1, 8, 32, 64, 128];
    
    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let value_function = create_mock_value_function();
                        let states: Vec<MarketState> = (0..batch_size)
                            .map(|_| create_test_market_state())
                            .collect();
                        (value_function, states)
                    },
                    |(value_function, states)| async move {
                        let result = value_function.estimate_batch(&states).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent agents
fn bench_concurrent_agents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_agents");
    
    let agent_counts = vec![1, 5, 10, 20, 50];
    
    for count in agent_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("agent_count", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let agents: Vec<Arc<dyn QStarAgent + Send + Sync>> = (0..count)
                            .map(|i| {
                                Arc::new(ExplorerAgent::new(format!("agent_{}", i), 0.1))
                                    as Arc<dyn QStarAgent + Send + Sync>
                            })
                            .collect();
                        let state = create_test_market_state();
                        (agents, state)
                    },
                    |(agents, state)| async move {
                        let mut handles = Vec::new();
                        
                        for agent in agents {
                            let state_clone = state.clone();
                            let handle = tokio::spawn(async move {
                                agent.q_star_search(&state_clone).await
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

/// Benchmark latency under different conditions
fn bench_latency_conditions(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("latency_conditions");
    
    let latency_targets = vec![1, 5, 10, 50, 100]; // microseconds
    
    for target in latency_targets {
        group.bench_with_input(
            BenchmarkId::new("target_latency_us", target),
            &target,
            |b, &target| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let mut config = create_test_config();
                        config.max_latency_us = target;
                        
                        let policy = create_mock_policy();
                        let value = create_mock_value_function();
                        let memory = create_mock_experience_memory();
                        let search = create_mock_search_tree();
                        
                        let engine = QStarEngine::new(config, policy, value, memory, search);
                        let state = create_test_market_state();
                        (engine, state)
                    },
                    |(engine, state)| async move {
                        let result = engine.decide(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark quantum enhancement
fn bench_quantum_enhancement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("quantum_enhancement");
    
    let quantum_settings = vec![
        ("quantum_enabled", true),
        ("quantum_disabled", false),
    ];
    
    for (setting_name, quantum_enabled) in quantum_settings {
        group.bench_with_input(
            BenchmarkId::new("quantum", setting_name),
            &quantum_enabled,
            |b, &quantum_enabled| {
                b.to_async(&rt).iter_with_setup(
                    || {
                        let mut config = create_test_config();
                        config.quantum_enhancement = quantum_enabled;
                        
                        let policy = create_mock_policy();
                        let value = create_mock_value_function();
                        let memory = create_mock_experience_memory();
                        let search = create_mock_search_tree();
                        
                        let engine = QStarEngine::new(config, policy, value, memory, search);
                        let state = create_test_market_state();
                        (engine, state)
                    },
                    |(engine, state)| async move {
                        let result = engine.decide(&state).await;
                        black_box(result)
                    },
                );
            },
        );
    }
    
    group.finish();
}

// Mock implementations for testing

#[derive(Clone)]
struct MockPolicy;

impl MockPolicy {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Policy for MockPolicy {
    async fn evaluate_state(&self, _state: &MarketState) -> Result<QStarAction, QStarError> {
        Ok(QStarAction::Hold)
    }
    
    async fn update(&self, _experiences: &[Experience]) -> Result<(), QStarError> {
        Ok(())
    }
    
    async fn get_action_probabilities(&self, _state: &MarketState) -> Result<Vec<f64>, QStarError> {
        Ok(vec![0.3, 0.3, 0.4])
    }
}

#[derive(Clone)]
struct MockValueFunction;

impl MockValueFunction {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ValueFunction for MockValueFunction {
    async fn estimate(&self, _state: &MarketState) -> Result<f64, QStarError> {
        Ok(0.5)
    }
    
    async fn estimate_batch(&self, _states: &[MarketState]) -> Result<Vec<f64>, QStarError> {
        Ok(vec![0.5; _states.len()])
    }
    
    async fn update(&self, _experiences: &[Experience]) -> Result<(), QStarError> {
        Ok(())
    }
}

#[derive(Clone)]
struct MockExperienceMemory;

impl MockExperienceMemory {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ExperienceMemory for MockExperienceMemory {
    async fn store(&self, _experience: Experience) -> Result<(), QStarError> {
        Ok(())
    }
    
    async fn sample(&self, _batch_size: usize) -> Result<Vec<Experience>, QStarError> {
        Ok(vec![create_test_experience()])
    }
    
    async fn size(&self) -> usize {
        1000
    }
}

#[derive(Clone)]
struct MockSearchTree;

impl MockSearchTree {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl SearchTree for MockSearchTree {
    async fn initialize(&self, _state: &MarketState) -> Result<(), QStarError> {
        Ok(())
    }
    
    async fn expand(&self, _state: &MarketState, _action: &QStarAction) -> Result<MarketState, QStarError> {
        Ok(create_test_market_state())
    }
    
    async fn get_best_path(&self) -> Result<Vec<QStarAction>, QStarError> {
        Ok(vec![QStarAction::Hold])
    }
}

// Additional trait implementations for coordination
impl CoordinatorAgent {
    async fn coordinate(&self, _state: &MarketState, _results: &[QStarSearchResult]) -> Result<CoordinationResult, QStarError> {
        Ok(CoordinationResult {
            final_action: QStarAction::Hold,
            confidence: 0.8,
            consensus_score: 0.9,
            agent_weights: HashMap::new(),
        })
    }
}

// Additional required structures and implementations
#[derive(Debug, Clone)]
pub struct MarketState {
    pub price: f64,
    pub volume: f64,
    pub volatility: f64,
    pub momentum: f64,
    pub spread: f64,
    pub regime: MarketRegime,
    pub features: Vec<f64>,
}

impl MarketState {
    pub fn new(price: f64, volume: f64, volatility: f64, momentum: f64, spread: f64, regime: MarketRegime, features: Vec<f64>) -> Self {
        Self { price, volume, volatility, momentum, spread, regime, features }
    }
    
    pub fn clone(&self) -> Self {
        Self {
            price: self.price,
            volume: self.volume,
            volatility: self.volatility,
            momentum: self.momentum,
            spread: self.spread,
            regime: self.regime.clone(),
            features: self.features.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct Experience {
    pub state: MarketState,
    pub action: QStarAction,
    pub reward: f64,
    pub next_state: MarketState,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct QStarSearchResult {
    pub action: QStarAction,
    pub q_value: f64,
    pub confidence: f64,
    pub search_depth: usize,
}

#[derive(Debug, Clone)]
pub struct CoordinationResult {
    pub final_action: QStarAction,
    pub confidence: f64,
    pub consensus_score: f64,
    pub agent_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    WeightedAverage,
    Consensus,
    Hierarchical,
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct QStarConfig {
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub exploration_rate: f64,
    pub exploration_decay: f64,
    pub min_exploration: f64,
    pub max_latency_us: u64,
    pub min_accuracy: f64,
    pub batch_size: usize,
    pub replay_buffer_size: usize,
    pub target_update_frequency: usize,
    pub gradient_clip_norm: f64,
    pub network_architecture: Vec<usize>,
    pub use_double_q: bool,
    pub use_dueling: bool,
    pub use_prioritized_replay: bool,
    pub use_noisy_networks: bool,
    pub use_rainbow: bool,
    pub quantum_enhancement: bool,
    pub neural_integration: bool,
    pub daa_coordination: bool,
}

impl Default for QStarConfig {
    fn default() -> Self {
        create_test_config()
    }
}

criterion_group!(
    benches,
    bench_q_star_engine_creation,
    bench_q_star_decision_making,
    bench_q_star_training,
    bench_agent_types,
    bench_coordination_strategies,
    bench_memory_operations,
    bench_search_tree_operations,
    bench_policy_evaluation,
    bench_value_function_estimation,
    bench_concurrent_agents,
    bench_latency_conditions,
    bench_quantum_enhancement
);

criterion_main!(benches);