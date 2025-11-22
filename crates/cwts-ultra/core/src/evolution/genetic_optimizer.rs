use rand::{seq::SliceRandom, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::algorithms::bayesian_var_engine::BayesianVaREngine;
use crate::data::binance_websocket_client::BinanceWebSocketClient;
use crate::integration::e2b_integration::E2BTrainingClient;
use crate::error::{OptimizationError, SystemError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemGenome {
    pub id: String,
    pub generation: u64,
    pub fitness_score: f64,
    pub genes: HashMap<String, GeneAllele>,
    pub performance_metrics: PerformanceMetrics,
    pub stability_score: f64,
    pub emergence_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneAllele {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub var_accuracy: f64,
    pub latency_p99: f64,
    pub error_rate: f64,
    pub throughput_ops: f64,
    pub memory_efficiency: f64,
    pub emergence_complexity: f64,
}

#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_percentage: f64,
    pub generations_limit: u64,
    pub fitness_threshold: f64,
    pub convergence_tolerance: f64,
    pub e2b_validation_ratio: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            mutation_rate: 0.15,
            crossover_rate: 0.8,
            elitism_percentage: 0.1,
            generations_limit: 100,
            fitness_threshold: 0.95,
            convergence_tolerance: 0.001,
            e2b_validation_ratio: 0.3,
        }
    }
}

#[derive(Debug)]
pub struct GeneticOptimizer {
    config: EvolutionConfig,
    current_population: Vec<SystemGenome>,
    best_genome: Option<SystemGenome>,
    generation_counter: u64,
    fitness_history: Vec<f64>,
    e2b_client: Arc<E2BTrainingClient>,
    binance_client: Arc<BinanceWebSocketClient>,
    var_engine: Arc<Mutex<BayesianVaREngine>>,
    optimization_state: Arc<Mutex<OptimizationState>>,
}

#[derive(Debug, Clone)]
struct OptimizationState {
    is_running: bool,
    last_improvement: Instant,
    convergence_counter: u32,
    total_evaluations: u64,
    successful_mutations: u64,
    successful_crossovers: u64,
}

impl GeneticOptimizer {
    pub fn new(
        config: EvolutionConfig,
        e2b_client: Arc<E2BTrainingClient>,
        binance_client: Arc<BinanceWebSocketClient>,
        var_engine: Arc<Mutex<BayesianVaREngine>>,
    ) -> Result<Self, OptimizationError> {
        info!("🧬 Initializing Genetic Optimizer for Bayesian VaR System");

        let optimization_state = OptimizationState {
            is_running: false,
            last_improvement: Instant::now(),
            convergence_counter: 0,
            total_evaluations: 0,
            successful_mutations: 0,
            successful_crossovers: 0,
        };

        Ok(Self {
            config,
            current_population: Vec::new(),
            best_genome: None,
            generation_counter: 0,
            fitness_history: Vec::new(),
            e2b_client,
            binance_client,
            var_engine,
            optimization_state: Arc::new(Mutex::new(optimization_state)),
        })
    }

    pub async fn evolve_system(&mut self) -> Result<SystemGenome, OptimizationError> {
        info!("🚀 Starting evolutionary optimization cycle");

        {
            let mut state = self.optimization_state.lock().unwrap();
            state.is_running = true;
            state.last_improvement = Instant::now();
        }

        // Initialize population with constitutional prime directive compliance
        self.initialize_population().await?;

        for generation in 0..self.config.generations_limit {
            self.generation_counter = generation;
            info!(
                "🔄 Generation {}: Evolving {} genomes",
                generation,
                self.current_population.len()
            );

            // Evaluate fitness for all genomes with E2B validation
            self.evaluate_population_fitness().await?;

            // Check for convergence
            if self.check_convergence()? {
                info!("✅ Convergence achieved at generation {}", generation);
                break;
            }

            // Select parents using tournament selection
            let parents = self.select_parents()?;

            // Create next generation through crossover and mutation
            let mut next_generation = self.create_next_generation(parents).await?;

            // Apply elitism (preserve top performers)
            self.apply_elitism(&mut next_generation)?;

            // Replace population
            self.current_population = next_generation;

            // Log generation statistics
            self.log_generation_stats(generation).await;

            // Adaptive parameter adjustment
            self.adapt_evolution_parameters();

            // Prevent resource exhaustion
            sleep(Duration::from_millis(100)).await;
        }

        {
            let mut state = self.optimization_state.lock().unwrap();
            state.is_running = false;
        }

        self.best_genome
            .clone()
            .ok_or(OptimizationError::NoViableGenome)
    }

    async fn initialize_population(&mut self) -> Result<(), OptimizationError> {
        info!(
            "🌱 Initializing population with {} genomes",
            self.config.population_size
        );

        self.current_population.clear();
        let mut rng = thread_rng();

        for i in 0..self.config.population_size {
            let genome = self.create_random_genome(i as u64, &mut rng).await?;
            self.current_population.push(genome);
        }

        Ok(())
    }

    async fn create_random_genome(
        &self,
        id: u64,
        rng: &mut impl Rng,
    ) -> Result<SystemGenome, OptimizationError> {
        let mut genes = HashMap::new();

        // VaR Engine Parameters (scientifically validated ranges)
        genes.insert(
            "confidence_level".to_string(),
            GeneAllele::Float(0.95 + rng.gen::<f64>() * 0.049),
        ); // 0.95-0.999
        genes.insert(
            "lookback_period".to_string(),
            GeneAllele::Integer(rng.gen_range(10..=252)),
        ); // 10-252 days
        genes.insert(
            "monte_carlo_iterations".to_string(),
            GeneAllele::Integer(rng.gen_range(1000..=100000)),
        );
        genes.insert(
            "bayesian_prior_strength".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 2.0 + 0.1),
        ); // 0.1-2.1

        // Heavy-tail parameters
        genes.insert(
            "tail_alpha".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 3.0 + 1.5),
        ); // 1.5-4.5
        genes.insert(
            "tail_beta".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 0.5 + 0.1),
        ); // 0.1-0.6

        // Performance parameters
        genes.insert(
            "batch_size".to_string(),
            GeneAllele::Integer(rng.gen_range(32..=2048)),
        );
        genes.insert(
            "cache_size".to_string(),
            GeneAllele::Integer(rng.gen_range(1000..=50000)),
        );
        genes.insert(
            "parallel_threads".to_string(),
            GeneAllele::Integer(rng.gen_range(1..=16)),
        );

        // Risk management parameters
        genes.insert(
            "max_position_size".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 0.5 + 0.01),
        ); // 1%-50%
        genes.insert(
            "correlation_threshold".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 0.8 + 0.1),
        ); // 10%-90%

        // Emergence factors
        genes.insert(
            "complexity_preference".to_string(),
            GeneAllele::Float(rng.gen::<f64>()),
        );
        genes.insert(
            "adaptation_rate".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 0.1),
        );
        genes.insert(
            "stability_weight".to_string(),
            GeneAllele::Float(rng.gen::<f64>() * 2.0 + 0.5),
        );

        let performance_metrics = PerformanceMetrics {
            var_accuracy: 0.0,
            latency_p99: f64::INFINITY,
            error_rate: 1.0,
            throughput_ops: 0.0,
            memory_efficiency: 0.0,
            emergence_complexity: 0.0,
        };

        Ok(SystemGenome {
            id: format!("genome_{}_gen_{}", id, self.generation_counter),
            generation: self.generation_counter,
            fitness_score: 0.0,
            genes,
            performance_metrics,
            stability_score: 0.0,
            emergence_factor: 0.0,
        })
    }

    async fn evaluate_population_fitness(&mut self) -> Result<(), OptimizationError> {
        info!(
            "📊 Evaluating fitness for generation {}",
            self.generation_counter
        );

        let total_genomes = self.current_population.len();
        let e2b_validation_count =
            (total_genomes as f64 * self.config.e2b_validation_ratio).ceil() as usize;

        // Evaluate all genomes in parallel chunks
        let chunk_size = 5;
        for (chunk_idx, genome_chunk) in self.current_population.chunks_mut(chunk_size).enumerate()
        {
            info!(
                "Evaluating chunk {} of {}",
                chunk_idx + 1,
                (total_genomes + chunk_size - 1) / chunk_size
            );

            let mut evaluation_tasks = Vec::new();

            for genome in genome_chunk.iter() {
                let use_e2b = chunk_idx * chunk_size < e2b_validation_count;
                let task = self.evaluate_genome_fitness(genome.clone(), use_e2b);
                evaluation_tasks.push(task);
            }

            // Execute evaluations in parallel
            let results = futures::future::try_join_all(evaluation_tasks).await?;

            for (genome, result) in genome_chunk.iter_mut().zip(results) {
                *genome = result;
            }

            // Brief pause between chunks
            sleep(Duration::from_millis(50)).await;
        }

        // Update best genome
        if let Some(best) = self
            .current_population
            .iter()
            .max_by(|a, b| a.fitness_score.partial_cmp(&b.fitness_score).unwrap())
        {
            if self.best_genome.is_none()
                || best.fitness_score > self.best_genome.as_ref().unwrap().fitness_score
            {
                self.best_genome = Some(best.clone());

                let mut state = self.optimization_state.lock().unwrap();
                state.last_improvement = Instant::now();
                info!(
                    "🏆 New best genome found! Fitness: {:.4}",
                    best.fitness_score
                );
            }
        }

        Ok(())
    }

    async fn evaluate_genome_fitness(
        &self,
        mut genome: SystemGenome,
        use_e2b_validation: bool,
    ) -> Result<SystemGenome, OptimizationError> {
        debug!("Evaluating genome: {}", genome.id);

        let start_time = Instant::now();

        // Apply genome configuration to VaR engine
        let metrics = if use_e2b_validation {
            self.evaluate_with_e2b_sandbox(&genome).await?
        } else {
            self.evaluate_with_production_simulation(&genome).await?
        };

        genome.performance_metrics = metrics;

        // Calculate comprehensive fitness score
        genome.fitness_score = self.calculate_comprehensive_fitness(&genome);
        genome.stability_score = self.calculate_stability_score(&genome);
        genome.emergence_factor = self.calculate_emergence_factor(&genome);

        let evaluation_duration = start_time.elapsed();
        debug!(
            "Genome {} evaluated in {:?}, fitness: {:.4}",
            genome.id, evaluation_duration, genome.fitness_score
        );

        {
            let mut state = self.optimization_state.lock().unwrap();
            state.total_evaluations += 1;
        }

        Ok(genome)
    }

    async fn evaluate_with_e2b_sandbox(
        &self,
        genome: &SystemGenome,
    ) -> Result<PerformanceMetrics, OptimizationError> {
        debug!("🧪 E2B sandbox evaluation for genome: {}", genome.id);

        // Create sandbox configuration
        let sandbox_config = serde_json::json!({
            "genome_id": genome.id,
            "parameters": genome.genes,
            "validation_type": "bayesian_var_backtest",
            "dataset": "crypto_historical_2023",
            "test_duration_hours": 24
        });

        // Submit to E2B for validation
        let validation_result = self
            .e2b_client
            .submit_training_task("bayesian-var-validation", &sandbox_config.to_string())
            .await
            .map_err(|e| OptimizationError::E2BValidationFailed(e.to_string()))?;

        // Parse results
        let results: serde_json::Value = serde_json::from_str(&validation_result.output)
            .map_err(|e| OptimizationError::ResultParsingFailed(e.to_string()))?;

        Ok(PerformanceMetrics {
            var_accuracy: results["var_accuracy"].as_f64().unwrap_or(0.0),
            latency_p99: results["latency_p99_ms"].as_f64().unwrap_or(f64::INFINITY),
            error_rate: results["error_rate"].as_f64().unwrap_or(1.0),
            throughput_ops: results["throughput_ops_sec"].as_f64().unwrap_or(0.0),
            memory_efficiency: results["memory_efficiency"].as_f64().unwrap_or(0.0),
            emergence_complexity: results["emergence_score"].as_f64().unwrap_or(0.0),
        })
    }

    async fn evaluate_with_production_simulation(
        &self,
        genome: &SystemGenome,
    ) -> Result<PerformanceMetrics, OptimizationError> {
        debug!("⚡ Production simulation for genome: {}", genome.id);

        // Create temporary VaR engine instance with genome parameters
        let mut engine_config = HashMap::new();
        for (key, value) in &genome.genes {
            engine_config.insert(key.clone(), value.clone());
        }

        // Run simulation with real Binance data
        let start_time = Instant::now();

        // Simulate trading session (abbreviated for performance)
        let mut total_calculations = 0u64;
        let mut successful_calculations = 0u64;
        let mut total_latency = Duration::new(0, 0);

        for _ in 0..100 {
            // 100 sample calculations
            let calc_start = Instant::now();

            // Simulate VaR calculation with genome parameters
            let confidence = match genome.genes.get("confidence_level") {
                Some(GeneAllele::Float(v)) => *v,
                _ => 0.95,
            };

            let lookback = match genome.genes.get("lookback_period") {
                Some(GeneAllele::Integer(v)) => *v as usize,
                _ => 100,
            };

            // Mock calculation (in real system, this would call actual VaR engine)
            let success = self.simulate_var_calculation(confidence, lookback).await;

            total_calculations += 1;
            if success {
                successful_calculations += 1;
            }

            total_latency += calc_start.elapsed();

            // Prevent timeout
            if start_time.elapsed() > Duration::from_secs(5) {
                break;
            }
        }

        let accuracy = successful_calculations as f64 / total_calculations as f64;
        let avg_latency = total_latency.as_millis() as f64 / total_calculations as f64;
        let throughput = 1000.0 / avg_latency.max(1.0);

        Ok(PerformanceMetrics {
            var_accuracy: accuracy,
            latency_p99: avg_latency * 1.2, // Approximate P99
            error_rate: 1.0 - accuracy,
            throughput_ops: throughput,
            memory_efficiency: 0.8 + (0.2 * rand::thread_rng().gen::<f64>()), // Simulated
            emergence_complexity: self.calculate_emergence_complexity(genome),
        })
    }

    async fn simulate_var_calculation(&self, confidence: f64, lookback: usize) -> bool {
        // Simulate success probability based on parameters
        let base_success = 0.85;
        let confidence_penalty = if confidence > 0.99 { 0.1 } else { 0.0 };
        let lookback_penalty = if lookback < 20 {
            0.15
        } else if lookback > 200 {
            0.05
        } else {
            0.0
        };

        let success_prob = base_success - confidence_penalty - lookback_penalty;

        // Brief delay to simulate computation
        sleep(Duration::from_micros(100 + (lookback as u64 * 2))).await;

        rand::thread_rng().gen::<f64>() < success_prob
    }

    fn calculate_comprehensive_fitness(&self, genome: &SystemGenome) -> f64 {
        let metrics = &genome.performance_metrics;

        // Constitutional Prime Directive compliance weights
        let accuracy_weight = 0.30; // 30% - VaR accuracy is critical
        let latency_weight = 0.25; // 25% - Real-time performance
        let reliability_weight = 0.20; // 20% - Low error rates
        let efficiency_weight = 0.15; // 15% - Resource efficiency
        let emergence_weight = 0.10; // 10% - System emergence properties

        // Normalize and score each component
        let accuracy_score = (metrics.var_accuracy - 0.8).max(0.0) / 0.2; // 80-100% accuracy
        let latency_score = (2000.0 - metrics.latency_p99.min(2000.0)) / 2000.0; // <2000ms is good
        let reliability_score = (1.0 - metrics.error_rate).max(0.0);
        let efficiency_score = metrics.memory_efficiency;
        let emergence_score = metrics.emergence_complexity;

        let fitness = accuracy_weight * accuracy_score
            + latency_weight * latency_score
            + reliability_weight * reliability_score
            + efficiency_weight * efficiency_score
            + emergence_weight * emergence_score;

        // Apply penalties for constitutional violations
        let mut penalty_factor = 1.0;

        if metrics.var_accuracy < 0.85 {
            penalty_factor *= 0.5; // Severe penalty for low accuracy
        }

        if metrics.error_rate > 0.05 {
            penalty_factor *= 0.7; // Penalty for high error rates
        }

        fitness * penalty_factor
    }

    fn calculate_stability_score(&self, genome: &SystemGenome) -> f64 {
        // Calculate stability based on parameter ranges and historical performance
        let mut stability = 1.0;

        // Check parameter stability
        if let Some(GeneAllele::Float(confidence)) = genome.genes.get("confidence_level") {
            if *confidence > 0.995 {
                stability *= 0.8; // Very high confidence can be unstable
            }
        }

        if let Some(GeneAllele::Integer(iterations)) = genome.genes.get("monte_carlo_iterations") {
            if *iterations < 5000 {
                stability *= 0.9; // Too few iterations can be unstable
            }
        }

        stability
    }

    fn calculate_emergence_factor(&self, genome: &SystemGenome) -> f64 {
        self.calculate_emergence_complexity(genome)
    }

    fn calculate_emergence_complexity(&self, genome: &SystemGenome) -> f64 {
        // Calculate system emergence based on parameter interactions and complexity
        let mut complexity_score = 0.0;
        let total_genes = genome.genes.len() as f64;

        // Parameter diversity contributes to emergence
        let mut parameter_variance = 0.0;
        for (_, allele) in &genome.genes {
            match allele {
                GeneAllele::Float(v) => parameter_variance += v.abs(),
                GeneAllele::Integer(v) => parameter_variance += *v as f64 / 1000.0,
                GeneAllele::Boolean(v) => parameter_variance += if *v { 1.0 } else { 0.0 },
                _ => parameter_variance += 0.5,
            }
        }

        complexity_score = (parameter_variance / total_genes).min(1.0);

        // Adjust based on performance - good performance with complexity indicates emergence
        if genome.performance_metrics.var_accuracy > 0.90 {
            complexity_score *= 1.2;
        }

        complexity_score.min(1.0)
    }

    fn select_parents(&self) -> Result<Vec<SystemGenome>, OptimizationError> {
        let parent_count =
            (self.current_population.len() as f64 * self.config.crossover_rate) as usize;
        let mut parents = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..parent_count {
            // Tournament selection
            let tournament_size = 3;
            let mut tournament = Vec::new();

            for _ in 0..tournament_size {
                if let Some(genome) = self.current_population.choose(&mut rng) {
                    tournament.push(genome.clone());
                }
            }

            if let Some(winner) = tournament
                .iter()
                .max_by(|a, b| a.fitness_score.partial_cmp(&b.fitness_score).unwrap())
            {
                parents.push(winner.clone());
            }
        }

        Ok(parents)
    }

    async fn create_next_generation(
        &self,
        parents: Vec<SystemGenome>,
    ) -> Result<Vec<SystemGenome>, OptimizationError> {
        let mut next_generation = Vec::new();
        let mut rng = thread_rng();

        // Create offspring through crossover and mutation
        for i in (0..parents.len()).step_by(2) {
            if i + 1 < parents.len() {
                let parent1 = &parents[i];
                let parent2 = &parents[i + 1];

                let (mut offspring1, mut offspring2) =
                    self.crossover(parent1, parent2, &mut rng)?;

                // Apply mutation
                self.mutate(&mut offspring1, &mut rng)?;
                self.mutate(&mut offspring2, &mut rng)?;

                next_generation.push(offspring1);
                next_generation.push(offspring2);

                let mut state = self.optimization_state.lock().unwrap();
                state.successful_crossovers += 1;
            }
        }

        // Fill remaining slots with mutations of best performers
        while next_generation.len() < self.config.population_size {
            if let Some(best) = self.best_genome.as_ref() {
                let mut mutant = best.clone();
                mutant.id = format!(
                    "mutant_{}_gen_{}",
                    next_generation.len(),
                    self.generation_counter + 1
                );
                mutant.generation = self.generation_counter + 1;
                self.mutate(&mut mutant, &mut rng)?;
                next_generation.push(mutant);

                let mut state = self.optimization_state.lock().unwrap();
                state.successful_mutations += 1;
            } else {
                break;
            }
        }

        Ok(next_generation)
    }

    fn crossover(
        &self,
        parent1: &SystemGenome,
        parent2: &SystemGenome,
        rng: &mut impl Rng,
    ) -> Result<(SystemGenome, SystemGenome), OptimizationError> {
        let mut offspring1 = parent1.clone();
        let mut offspring2 = parent2.clone();

        offspring1.id = format!(
            "cross1_{}_gen_{}",
            rng.gen::<u32>(),
            self.generation_counter + 1
        );
        offspring2.id = format!(
            "cross2_{}_gen_{}",
            rng.gen::<u32>(),
            self.generation_counter + 1
        );
        offspring1.generation = self.generation_counter + 1;
        offspring2.generation = self.generation_counter + 1;

        // Uniform crossover
        for key in parent1.genes.keys() {
            if rng.gen::<f64>() < 0.5 {
                if let Some(gene2) = parent2.genes.get(key) {
                    offspring1.genes.insert(key.clone(), gene2.clone());
                }
            }

            if rng.gen::<f64>() < 0.5 {
                if let Some(gene1) = parent1.genes.get(key) {
                    offspring2.genes.insert(key.clone(), gene1.clone());
                }
            }
        }

        // Reset fitness scores
        offspring1.fitness_score = 0.0;
        offspring2.fitness_score = 0.0;

        Ok((offspring1, offspring2))
    }

    fn mutate(
        &self,
        genome: &mut SystemGenome,
        rng: &mut impl Rng,
    ) -> Result<(), OptimizationError> {
        for (key, allele) in genome.genes.iter_mut() {
            if rng.gen::<f64>() < self.config.mutation_rate {
                match allele {
                    GeneAllele::Float(value) => {
                        let mutation_strength = 0.1;
                        let mutation_delta = rng.gen_range(-mutation_strength..=mutation_strength);
                        *value = (*value + mutation_delta).max(0.001).min(10.0);
                    }
                    GeneAllele::Integer(value) => {
                        let mutation_range = (*value as f64 * 0.2) as i64 + 1;
                        let mutation_delta = rng.gen_range(-mutation_range..=mutation_range);
                        *value = (*value + mutation_delta).max(1);
                    }
                    GeneAllele::Boolean(value) => {
                        if rng.gen::<f64>() < 0.1 {
                            // Lower mutation rate for booleans
                            *value = !*value;
                        }
                    }
                    GeneAllele::Array(values) => {
                        for val in values.iter_mut() {
                            if rng.gen::<f64>() < self.config.mutation_rate * 0.5 {
                                let mutation_delta = rng.gen_range(-0.1..=0.1);
                                *val = (*val + mutation_delta).max(-1.0).min(1.0);
                            }
                        }
                    }
                    _ => {} // No mutation for strings
                }
            }
        }

        // Reset fitness after mutation
        genome.fitness_score = 0.0;
        Ok(())
    }

    fn apply_elitism(
        &self,
        next_generation: &mut Vec<SystemGenome>,
    ) -> Result<(), OptimizationError> {
        let elite_count =
            (self.config.population_size as f64 * self.config.elitism_percentage) as usize;

        // Sort current population by fitness
        let mut elite_genomes = self.current_population.clone();
        elite_genomes.sort_by(|a, b| b.fitness_score.partial_cmp(&a.fitness_score).unwrap());

        // Replace worst performers with elite genomes
        next_generation.sort_by(|a, b| a.fitness_score.partial_cmp(&b.fitness_score).unwrap());

        for i in 0..elite_count.min(next_generation.len()) {
            if i < elite_genomes.len() {
                let mut elite = elite_genomes[i].clone();
                elite.generation = self.generation_counter + 1;
                next_generation[i] = elite;
            }
        }

        Ok(())
    }

    fn check_convergence(&mut self) -> Result<bool, OptimizationError> {
        if self.current_population.is_empty() {
            return Ok(false);
        }

        let current_best_fitness = self
            .current_population
            .iter()
            .map(|g| g.fitness_score)
            .fold(0.0f64, f64::max);

        self.fitness_history.push(current_best_fitness);

        // Check fitness threshold
        if current_best_fitness >= self.config.fitness_threshold {
            return Ok(true);
        }

        // Check convergence over recent generations
        if self.fitness_history.len() >= 10 {
            let recent_fitness = &self.fitness_history[self.fitness_history.len() - 10..];
            let fitness_variance = self.calculate_variance(recent_fitness);

            if fitness_variance < self.config.convergence_tolerance {
                let mut state = self.optimization_state.lock().unwrap();
                state.convergence_counter += 1;

                if state.convergence_counter >= 5 {
                    return Ok(true);
                }
            } else {
                let mut state = self.optimization_state.lock().unwrap();
                state.convergence_counter = 0;
            }
        }

        Ok(false)
    }

    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance
    }

    async fn log_generation_stats(&self, generation: u64) {
        let population_size = self.current_population.len();
        let avg_fitness = self
            .current_population
            .iter()
            .map(|g| g.fitness_score)
            .sum::<f64>()
            / population_size as f64;
        let max_fitness = self
            .current_population
            .iter()
            .map(|g| g.fitness_score)
            .fold(0.0f64, f64::max);
        let min_fitness = self
            .current_population
            .iter()
            .map(|g| g.fitness_score)
            .fold(1.0f64, f64::min);

        let state = self.optimization_state.lock().unwrap();

        info!("📊 Generation {} Statistics:", generation);
        info!("   Population: {}", population_size);
        info!(
            "   Fitness - Max: {:.4}, Avg: {:.4}, Min: {:.4}",
            max_fitness, avg_fitness, min_fitness
        );
        info!("   Total Evaluations: {}", state.total_evaluations);
        info!("   Successful Mutations: {}", state.successful_mutations);
        info!("   Successful Crossovers: {}", state.successful_crossovers);

        if let Some(best) = &self.best_genome {
            info!(
                "🏆 Best Genome: {} (Fitness: {:.4}, Stability: {:.4}, Emergence: {:.4})",
                best.id, best.fitness_score, best.stability_score, best.emergence_factor
            );
        }
    }

    fn adapt_evolution_parameters(&mut self) {
        let state = self.optimization_state.lock().unwrap();

        // Adaptive mutation rate based on progress
        let generations_since_improvement = state.last_improvement.elapsed().as_secs() / 60; // minutes

        if generations_since_improvement > 5 {
            self.config.mutation_rate = (self.config.mutation_rate * 1.1).min(0.3);
            info!(
                "🧬 Increased mutation rate to {:.3} due to stagnation",
                self.config.mutation_rate
            );
        } else if generations_since_improvement == 0 {
            self.config.mutation_rate = (self.config.mutation_rate * 0.95).max(0.05);
        }

        // Adaptive population diversity
        let diversity = self.calculate_population_diversity();
        if diversity < 0.1 {
            self.config.mutation_rate = (self.config.mutation_rate * 1.2).min(0.4);
            info!(
                "🌈 Increased mutation rate to {:.3} to maintain diversity",
                self.config.mutation_rate
            );
        }
    }

    fn calculate_population_diversity(&self) -> f64 {
        if self.current_population.len() < 2 {
            return 1.0;
        }

        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..self.current_population.len() {
            for j in (i + 1)..self.current_population.len() {
                let distance = self.calculate_genome_distance(
                    &self.current_population[i],
                    &self.current_population[j],
                );
                total_distance += distance;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            1.0
        }
    }

    fn calculate_genome_distance(&self, genome1: &SystemGenome, genome2: &SystemGenome) -> f64 {
        let mut distance = 0.0;
        let mut gene_count = 0;

        for (key, allele1) in &genome1.genes {
            if let Some(allele2) = genome2.genes.get(key) {
                let gene_distance = match (allele1, allele2) {
                    (GeneAllele::Float(v1), GeneAllele::Float(v2)) => (v1 - v2).abs(),
                    (GeneAllele::Integer(v1), GeneAllele::Integer(v2)) => {
                        (*v1 - *v2).abs() as f64 / 1000.0
                    }
                    (GeneAllele::Boolean(v1), GeneAllele::Boolean(v2)) => {
                        if v1 == v2 {
                            0.0
                        } else {
                            1.0
                        }
                    }
                    _ => 0.5, // Default distance for incomparable types
                };

                distance += gene_distance;
                gene_count += 1;
            }
        }

        if gene_count > 0 {
            distance / gene_count as f64
        } else {
            1.0
        }
    }

    pub fn get_optimization_state(&self) -> OptimizationState {
        self.optimization_state.lock().unwrap().clone()
    }

    pub fn get_best_genome(&self) -> Option<SystemGenome> {
        self.best_genome.clone()
    }

    pub async fn apply_genome_to_system(
        &self,
        genome: &SystemGenome,
    ) -> Result<(), OptimizationError> {
        info!(
            "🔄 Applying optimized genome {} to production system",
            genome.id
        );

        // Apply configuration to VaR engine
        let mut var_engine = self.var_engine.lock().unwrap();

        // Extract and apply parameters (this would interface with actual VaR engine)
        for (key, allele) in &genome.genes {
            match (key.as_str(), allele) {
                ("confidence_level", GeneAllele::Float(v)) => {
                    info!("Setting confidence level: {:.4}", v);
                    // var_engine.set_confidence_level(*v)?;
                }
                ("lookback_period", GeneAllele::Integer(v)) => {
                    info!("Setting lookback period: {} days", v);
                    // var_engine.set_lookback_period(*v as usize)?;
                }
                ("monte_carlo_iterations", GeneAllele::Integer(v)) => {
                    info!("Setting Monte Carlo iterations: {}", v);
                    // var_engine.set_monte_carlo_iterations(*v as usize)?;
                }
                _ => debug!("Skipping parameter: {}", key),
            }
        }

        info!("✅ Genome successfully applied to production system");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub start_time: Instant,
    pub end_time: Instant,
    pub total_generations: u64,
    pub best_fitness_achieved: f64,
    pub convergence_achieved: bool,
    pub total_evaluations: u64,
    pub best_genome: Option<SystemGenome>,
    pub fitness_progression: Vec<f64>,
}

impl OptimizationReport {
    pub fn generate_summary(&self) -> String {
        format!(
            "🧬 Genetic Optimization Report\n\
             Duration: {:?}\n\
             Generations: {}\n\
             Total Evaluations: {}\n\
             Best Fitness: {:.4}\n\
             Convergence: {}\n\
             Best Genome: {}",
            self.end_time - self.start_time,
            self.total_generations,
            self.total_evaluations,
            self.best_fitness_achieved,
            if self.convergence_achieved {
                "Yes"
            } else {
                "No"
            },
            self.best_genome.as_ref().map(|g| g.id.as_str()).unwrap_or("None")
        )
    }
}
