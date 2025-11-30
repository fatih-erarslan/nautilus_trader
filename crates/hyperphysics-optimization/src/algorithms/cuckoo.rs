//! Cuckoo Search (CS) with Lévy flights.
//!
//! # References
//! - Yang & Deb (2009): "Cuckoo Search via Lévy Flights"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Approximation of Gamma function using Stirling's formula.
fn gamma_approx(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_approx(1.0 - x))
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * lanczos_coefficients(x)
    }
}

fn lanczos_coefficients(x: f64) -> f64 {
    let coeffs = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    let mut sum = coeffs[0];
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    sum
}

/// Cuckoo Search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuckooConfig {
    /// Discovery probability (fraction of worst nests abandoned)
    pub pa: f64,
    /// Lévy flight exponent (typically 1.5)
    pub beta: f64,
    /// Step size scaling factor
    pub alpha: f64,
}

impl Default for CuckooConfig {
    fn default() -> Self {
        Self { pa: 0.25, beta: 1.5, alpha: 0.01 }
    }
}

impl AlgorithmConfig for CuckooConfig {
    fn validate(&self) -> Result<(), String> {
        if self.pa < 0.0 || self.pa > 1.0 { return Err("pa must be in [0, 1]".to_string()); }
        if self.beta < 1.0 || self.beta > 2.0 { return Err("beta must be in [1, 2]".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { pa: 0.3, beta: 1.5, alpha: 0.05 } }
    fn high_accuracy() -> Self { Self { pa: 0.2, beta: 1.5, alpha: 0.01 } }
}

/// Cuckoo Search optimizer.
pub struct CuckooSearch {
    config: CuckooConfig,
    opt_config: OptimizationConfig,
    population: Population,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl CuckooSearch {
    /// Create a new Cuckoo Search optimizer
    pub fn new(config: CuckooConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize nest population using Latin Hypercube Sampling
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    /// Generate Lévy flight step using Mantegna algorithm.
    fn levy_flight(&self, rng: &mut impl Rng, dim: usize) -> Array1<f64> {
        let beta = self.config.beta;

        // Mantegna's algorithm for sigma_u
        // sigma_u = (Gamma(1+beta) * sin(pi*beta/2) / (Gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta)
        // Using approximation for stability
        let numerator = ((std::f64::consts::PI * beta / 2.0).sin() * gamma_approx(1.0 + beta)).abs();
        let denominator = gamma_approx((1.0 + beta) / 2.0) * beta * 2f64.powf((beta - 1.0) / 2.0);
        let sigma_u = if denominator.abs() > 1e-10 {
            (numerator / denominator).powf(1.0 / beta)
        } else {
            1.0
        };
        let sigma_u = sigma_u.max(0.3).min(2.0); // Stabilize

        Array1::from_iter((0..dim).map(|_| {
            let u: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let v: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let v_abs = v.abs().max(1e-10);
            u * sigma_u / v_abs.powf(1.0 / beta)
        }))
    }

    /// Execute one iteration with Lévy flights and nest abandonment
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let best = self.population.best().unwrap();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let ranges = bounds.ranges();
        let mut rng = rand::thread_rng();

        let pop_snapshot: Vec<Individual> = self.population.individuals().clone();
        let n = pop_snapshot.len();

        // Generate new solutions via Lévy flights
        let mut new_nests: Vec<Individual> = Vec::new();
        for nest in &pop_snapshot {
            let levy = self.levy_flight(&mut rng, dim);
            let mut new_pos = Array1::zeros(dim);
            for j in 0..dim {
                new_pos[j] = nest.position[j] + self.config.alpha * ranges[j] * levy[j] * (nest.position[j] - best.position[j]);
            }
            let repaired = bounds.repair(new_pos.view());
            let fitness = objective.evaluate(repaired.view());

            // Keep new if better
            if fitness < nest.fitness.unwrap_or(f64::INFINITY) {
                let mut ind = Individual::new(repaired);
                ind.fitness = Some(fitness);
                new_nests.push(ind);
            } else {
                new_nests.push(nest.clone());
            }
        }

        // Abandon worst nests (pa fraction)
        new_nests.sort();
        let abandon_count = (n as f64 * self.config.pa) as usize;
        for i in (n - abandon_count)..n {
            let mut new_pos = Array1::zeros(dim);
            let r1 = rng.gen_range(0..n);
            let r2 = rng.gen_range(0..n);
            for j in 0..dim {
                let k: f64 = rng.gen();
                new_pos[j] = new_nests[i].position[j] + k * (pop_snapshot[r1].position[j] - pop_snapshot[r2].position[j]);
            }
            let repaired = bounds.repair(new_pos.view());
            let fitness = objective.evaluate(repaired.view());
            new_nests[i] = Individual::new(repaired);
            new_nests[i].fitness = Some(fitness);
        }

        {
            let mut pop = self.population.individuals_mut();
            *pop = new_nests;
        }

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Run the full cuckoo search optimization until convergence or max iterations
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("CS failed".to_string()))
    }
}

impl Algorithm for CuckooSearch {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::CuckooSearch }
    fn name(&self) -> &str { "Cuckoo Search" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_cuckoo_optimization() {
        let config = CuckooConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(25);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut cs = CuckooSearch::new(config, opt_config, bounds).unwrap();
        let solution = cs.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
