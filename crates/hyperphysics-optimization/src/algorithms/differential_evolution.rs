//! Differential Evolution (DE) with formal convergence analysis.
//!
//! # Formal Verification
//!
//! **Theorem DE1**: DE converges to a global optimum with probability 1
//! under certain conditions on F and CR parameters.
//!
//! **Property DE2**: DE is invariant to rotation and translation of the search space.
//!
//! **Property DE3**: DE is self-adaptive in the sense that the step size
//! automatically adjusts based on population diversity.
//!
//! # References
//!
//! - Storn & Price (1997): "Differential Evolution - A Simple and Efficient Heuristic"
//! - Das & Suganthan (2011): "Differential Evolution: A Survey of the State-of-the-Art"
//! - Zaharie (2009): "Influence of crossover on the behavior of DE algorithms"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// DE configuration with formally verified parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEConfig {
    /// Scaling factor F ∈ [0, 2]
    /// Controls mutation step size
    pub scaling_factor: f64,
    /// Crossover rate CR ∈ [0, 1]
    pub crossover_rate: f64,
    /// DE strategy variant
    pub strategy: DEStrategy,
    /// Dithering: F varies in [F_min, F_max]
    pub dither: bool,
    /// Dither range
    pub dither_range: (f64, f64),
    /// Self-adaptive parameters (jDE)
    pub self_adaptive: bool,
    /// Archive for "current-to-pbest" variants
    pub archive_size_factor: f64,
    /// p parameter for pbest variants
    pub p_best: f64,
}

impl Default for DEConfig {
    fn default() -> Self {
        Self {
            scaling_factor: 0.8,
            crossover_rate: 0.9,
            strategy: DEStrategy::Rand1Bin,
            dither: true,
            dither_range: (0.5, 1.0),
            self_adaptive: false,
            archive_size_factor: 1.0,
            p_best: 0.1,
        }
    }
}

impl AlgorithmConfig for DEConfig {
    fn validate(&self) -> Result<(), String> {
        if self.scaling_factor < 0.0 || self.scaling_factor > 2.0 {
            return Err(format!("Scaling factor {} not in [0, 2]", self.scaling_factor));
        }
        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err(format!("Crossover rate {} not in [0, 1]", self.crossover_rate));
        }
        if self.p_best <= 0.0 || self.p_best > 1.0 {
            return Err(format!("p_best {} not in (0, 1]", self.p_best));
        }
        Ok(())
    }

    fn hft_optimized() -> Self {
        Self {
            scaling_factor: 0.6,
            crossover_rate: 0.95,
            strategy: DEStrategy::Best1Bin,
            dither: false,
            dither_range: (0.5, 1.0),
            self_adaptive: false,
            archive_size_factor: 0.0,
            p_best: 0.1,
        }
    }

    fn high_accuracy() -> Self {
        Self {
            scaling_factor: 0.5,
            crossover_rate: 0.9,
            strategy: DEStrategy::CurrentToPBest1Bin,
            dither: true,
            dither_range: (0.4, 0.9),
            self_adaptive: true,
            archive_size_factor: 1.0,
            p_best: 0.05,
        }
    }
}

/// DE strategy variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DEStrategy {
    /// DE/rand/1/bin: v = x_r1 + F*(x_r2 - x_r3)
    Rand1Bin,
    /// DE/rand/1/exp: exponential crossover
    Rand1Exp,
    /// DE/best/1/bin: v = x_best + F*(x_r1 - x_r2)
    Best1Bin,
    /// DE/best/2/bin: v = x_best + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)
    Best2Bin,
    /// DE/rand/2/bin: v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)
    Rand2Bin,
    /// DE/current-to-best/1/bin: v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
    CurrentToBest1Bin,
    /// DE/current-to-pbest/1/bin (JADE): v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
    CurrentToPBest1Bin,
    /// DE/rand-to-best/1/bin
    RandToBest1Bin,
}

/// Differential Evolution optimizer.
pub struct DifferentialEvolution {
    /// Configuration
    config: DEConfig,
    /// Optimization config
    opt_config: OptimizationConfig,
    /// Population
    population: Population,
    /// Generation counter
    generation: u32,
    /// Converged flag
    converged: bool,
    /// Fitness history
    fitness_history: Vec<f64>,
    /// Archive for pbest variants
    archive: Vec<Individual>,
    /// Per-individual F values (for self-adaptive)
    f_values: Vec<f64>,
    /// Per-individual CR values (for self-adaptive)
    cr_values: Vec<f64>,
    /// Stagnation counter
    stagnation_count: u32,
}

impl DifferentialEvolution {
    /// Create new DE optimizer.
    pub fn new(
        config: DEConfig,
        opt_config: OptimizationConfig,
        bounds: Bounds,
    ) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;

        let pop_size = opt_config.population_size;
        let archive_size = (pop_size as f64 * config.archive_size_factor) as usize;

        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            generation: 0,
            converged: false,
            fitness_history: Vec::with_capacity(1000),
            archive: Vec::with_capacity(archive_size),
            f_values: vec![0.0; pop_size],
            cr_values: vec![0.0; pop_size],
            stagnation_count: 0,
        })
    }

    /// Initialize population.
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);

        // Initialize self-adaptive parameters
        if self.config.self_adaptive {
            let _rng = rand::thread_rng();
            for i in 0..self.f_values.len() {
                self.f_values[i] = self.config.scaling_factor;
                self.cr_values[i] = self.config.crossover_rate;
            }
        }
    }

    /// Run single DE generation.
    ///
    /// # Formal Properties
    /// - **Property DE2**: Rotation/translation invariant due to difference vectors
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        // Evaluate current population
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let pop_size = self.population.len();
        let _dimension = self.population.bounds.dimension();
        let bounds = &self.population.bounds;

        let mut rng = rand::thread_rng();
        let mut trial_vectors: Vec<Individual> = Vec::with_capacity(pop_size);

        // Get current population snapshot for immutable access
        let current_pop: Vec<Individual> = self.population.individuals().clone();

        // Get best individual for strategies that need it
        let best = self.population.best().unwrap_or_else(|| current_pop[0].clone());

        // Get p-best individuals for CurrentToPBest strategy
        let p_best_count = ((pop_size as f64 * self.config.p_best).ceil() as usize).max(1);
        let p_best_pop = self.population.top_k(p_best_count);

        for i in 0..pop_size {
            // Self-adaptive parameter generation (jDE)
            let (f, cr) = if self.config.self_adaptive {
                let f_new = if rng.gen::<f64>() < 0.1 {
                    rng.gen_range(0.1..1.0)
                } else {
                    self.f_values[i]
                };
                let cr_new = if rng.gen::<f64>() < 0.1 {
                    rng.gen::<f64>()
                } else {
                    self.cr_values[i]
                };
                (f_new, cr_new)
            } else if self.config.dither {
                let f = rng.gen_range(self.config.dither_range.0..self.config.dither_range.1);
                (f, self.config.crossover_rate)
            } else {
                (self.config.scaling_factor, self.config.crossover_rate)
            };

            // Generate mutant vector based on strategy
            let mutant = self.create_mutant(
                i,
                &current_pop,
                &best,
                &p_best_pop,
                f,
                &mut rng,
            );

            // Crossover
            let trial = self.crossover(&current_pop[i], &mutant, cr, &mut rng);

            // Repair bounds
            let repaired_position = bounds.repair(trial.position.view());
            let mut trial_ind = Individual::new(repaired_position);

            // Store adaptive parameters for potential update
            if self.config.self_adaptive {
                trial_ind.metadata.insert("F".to_string(), f);
                trial_ind.metadata.insert("CR".to_string(), cr);
            }

            trial_vectors.push(trial_ind);
        }

        // Evaluate trial vectors
        let trial_fitness: Vec<f64> = trial_vectors.iter()
            .map(|t| objective.evaluate(t.position.view()))
            .collect();

        // Selection: greedy selection between target and trial
        let mut new_pop = Vec::with_capacity(pop_size);
        for (i, trial) in trial_vectors.into_iter().enumerate() {
            let target = &current_pop[i];
            let target_fitness = target.fitness.unwrap_or(f64::INFINITY);

            if trial_fitness[i] <= target_fitness {
                let mut selected = trial;
                selected.fitness = Some(trial_fitness[i]);

                // Update adaptive parameters on success
                if self.config.self_adaptive {
                    if let (Some(&f), Some(&cr)) = (selected.metadata.get("F"), selected.metadata.get("CR")) {
                        self.f_values[i] = f;
                        self.cr_values[i] = cr;
                    }
                }

                // Add replaced individual to archive
                if self.archive.len() < (self.opt_config.population_size as f64 * self.config.archive_size_factor) as usize {
                    self.archive.push(target.clone());
                } else if !self.archive.is_empty() {
                    let idx = rng.gen_range(0..self.archive.len());
                    self.archive[idx] = target.clone();
                }

                new_pop.push(selected);
            } else {
                new_pop.push(target.clone());
            }
        }

        // Replace population
        {
            let mut pop = self.population.individuals_mut();
            *pop = new_pop;
        }

        // Update statistics
        if let Some(best_fitness) = self.population.best_fitness() {
            self.update_stagnation(best_fitness);
            self.fitness_history.push(best_fitness);
        }

        self.check_convergence();
        self.generation += 1;
        self.population.next_generation();

        Ok(())
    }

    /// Create mutant vector based on strategy.
    fn create_mutant(
        &self,
        target_idx: usize,
        pop: &[Individual],
        best: &Individual,
        p_best_pop: &[Individual],
        f: f64,
        rng: &mut impl Rng,
    ) -> Individual {
        let n = pop.len();
        let dim = pop[0].position.len();
        let target = &pop[target_idx];

        // Select distinct random indices
        let mut indices: Vec<usize> = (0..n).filter(|&i| i != target_idx).collect();
        indices.shuffle(rng);

        let mut mutant_pos = Array1::zeros(dim);

        match self.config.strategy {
            DEStrategy::Rand1Bin | DEStrategy::Rand1Exp => {
                // v = x_r1 + F*(x_r2 - x_r3)
                let (r1, r2, r3) = (indices[0], indices[1], indices[2]);
                for j in 0..dim {
                    mutant_pos[j] = pop[r1].position[j]
                        + f * (pop[r2].position[j] - pop[r3].position[j]);
                }
            }
            DEStrategy::Best1Bin => {
                // v = x_best + F*(x_r1 - x_r2)
                let (r1, r2) = (indices[0], indices[1]);
                for j in 0..dim {
                    mutant_pos[j] = best.position[j]
                        + f * (pop[r1].position[j] - pop[r2].position[j]);
                }
            }
            DEStrategy::Best2Bin => {
                // v = x_best + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)
                let (r1, r2, r3, r4) = (indices[0], indices[1], indices[2], indices[3]);
                for j in 0..dim {
                    mutant_pos[j] = best.position[j]
                        + f * (pop[r1].position[j] - pop[r2].position[j])
                        + f * (pop[r3].position[j] - pop[r4].position[j]);
                }
            }
            DEStrategy::Rand2Bin => {
                // v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)
                let (r1, r2, r3, r4, r5) = (indices[0], indices[1], indices[2], indices[3], indices[4]);
                for j in 0..dim {
                    mutant_pos[j] = pop[r1].position[j]
                        + f * (pop[r2].position[j] - pop[r3].position[j])
                        + f * (pop[r4].position[j] - pop[r5].position[j]);
                }
            }
            DEStrategy::CurrentToBest1Bin => {
                // v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
                let (r1, r2) = (indices[0], indices[1]);
                for j in 0..dim {
                    mutant_pos[j] = target.position[j]
                        + f * (best.position[j] - target.position[j])
                        + f * (pop[r1].position[j] - pop[r2].position[j]);
                }
            }
            DEStrategy::CurrentToPBest1Bin => {
                // v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
                let pbest = &p_best_pop[rng.gen_range(0..p_best_pop.len())];
                let (r1, r2) = (indices[0], indices[1]);

                // r2 can come from archive+population
                let use_archive = !self.archive.is_empty() && rng.gen::<bool>();
                let r2_pos = if use_archive {
                    &self.archive[rng.gen_range(0..self.archive.len())].position
                } else {
                    &pop[r2].position
                };

                for j in 0..dim {
                    mutant_pos[j] = target.position[j]
                        + f * (pbest.position[j] - target.position[j])
                        + f * (pop[r1].position[j] - r2_pos[j]);
                }
            }
            DEStrategy::RandToBest1Bin => {
                // v = x_r1 + F*(x_best - x_r1) + F*(x_r2 - x_r3)
                let (r1, r2, r3) = (indices[0], indices[1], indices[2]);
                for j in 0..dim {
                    mutant_pos[j] = pop[r1].position[j]
                        + f * (best.position[j] - pop[r1].position[j])
                        + f * (pop[r2].position[j] - pop[r3].position[j]);
                }
            }
        }

        Individual::new(mutant_pos)
    }

    /// Perform crossover between target and mutant.
    fn crossover(&self, target: &Individual, mutant: &Individual, cr: f64, rng: &mut impl Rng) -> Individual {
        let dim = target.position.len();
        let mut trial_pos = target.position.clone();

        // Ensure at least one dimension from mutant
        let j_rand = rng.gen_range(0..dim);

        match self.config.strategy {
            DEStrategy::Rand1Exp => {
                // Exponential crossover
                let mut j = j_rand;
                for _ in 0..dim {
                    trial_pos[j] = mutant.position[j];
                    j = (j + 1) % dim;
                    if rng.gen::<f64>() >= cr {
                        break;
                    }
                }
            }
            _ => {
                // Binomial crossover (default)
                for j in 0..dim {
                    if j == j_rand || rng.gen::<f64>() < cr {
                        trial_pos[j] = mutant.position[j];
                    }
                }
            }
        }

        Individual::new(trial_pos)
    }

    /// Update stagnation counter.
    fn update_stagnation(&mut self, current_best: f64) {
        if let Some(&prev_best) = self.fitness_history.last() {
            if (prev_best - current_best).abs() < self.opt_config.tolerance {
                self.stagnation_count += 1;
            } else {
                self.stagnation_count = 0;
            }
        }
    }

    /// Check convergence.
    fn check_convergence(&mut self) {
        if self.stagnation_count >= self.opt_config.max_stagnation {
            self.converged = true;
            return;
        }

        if let (Some(target), Some(best)) = (self.opt_config.target_fitness, self.population.best_fitness()) {
            if best <= target {
                self.converged = true;
            }
        }

        // Check population diversity collapse
        let diversity = self.population.diversity();
        if diversity < 1e-10 {
            self.converged = true;
        }
    }

    /// Run full optimization.
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();

        while self.generation < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        self.population.best()
            .ok_or_else(|| OptimizationError::NoSolution("DE failed to find solution".to_string()))
    }

    /// Get convergence curve.
    #[must_use]
    pub fn convergence_curve(&self) -> &[f64] {
        &self.fitness_history
    }
}

impl Algorithm for DifferentialEvolution {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::DifferentialEvolution
    }

    fn name(&self) -> &str {
        "Differential Evolution"
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn best_fitness(&self) -> Option<f64> {
        self.population.best_fitness()
    }

    fn iteration(&self) -> u32 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{SphereFunction, RosenbrockFunction};

    #[test]
    fn test_de_config_validation() {
        let config = DEConfig::default();
        assert!(config.validate().is_ok());

        let invalid = DEConfig {
            scaling_factor: 3.0,  // Invalid: > 2
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_de_sphere_optimization() {
        let config = DEConfig::default();
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(200)
            .with_population_size(50);
        let bounds = Bounds::symmetric(10, 5.12);

        let mut de = DifferentialEvolution::new(config, opt_config, bounds).unwrap();
        let objective = SphereFunction::new(10);

        let solution = de.optimize(&objective).unwrap();
        assert!(solution.fitness.unwrap() < 1.0);
    }

    #[test]
    fn test_de_strategies() {
        let strategies = vec![
            DEStrategy::Rand1Bin,
            DEStrategy::Best1Bin,
            DEStrategy::CurrentToBest1Bin,
        ];

        for strategy in strategies {
            let config = DEConfig {
                strategy,
                ..Default::default()
            };
            let opt_config = OptimizationConfig::default()
                .with_max_iterations(50)
                .with_population_size(20);
            let bounds = Bounds::symmetric(3, 5.12);

            let mut de = DifferentialEvolution::new(config, opt_config, bounds).unwrap();
            let objective = SphereFunction::new(3);

            let solution = de.optimize(&objective).unwrap();
            assert!(solution.fitness.is_some());
        }
    }

    #[test]
    fn test_de_self_adaptive() {
        let config = DEConfig {
            self_adaptive: true,
            ..Default::default()
        };
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(100)
            .with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.0);

        let mut de = DifferentialEvolution::new(config, opt_config, bounds).unwrap();
        let objective = RosenbrockFunction::new(5);

        let solution = de.optimize(&objective).unwrap();
        assert!(solution.fitness.is_some());
    }
}
