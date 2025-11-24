//! Social Spider Optimization (SSO) - Spider colony social behavior.
//!
//! # References
//! - Cuevas et al. (2013): "A Swarm Optimization Algorithm Inspired in the Behavior of the Social-Spider"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// SSO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSOConfig {
    /// Female ratio in population
    pub pf: f64,
    /// Attraction probability
    pub pa: f64,
    /// Vibration attenuation rate
    pub ra: f64,
    /// Random walk coefficient
    pub pm: f64,
}

impl Default for SSOConfig {
    fn default() -> Self {
        Self { pf: 0.65, pa: 0.7, ra: 0.1, pm: 0.1 }
    }
}

impl AlgorithmConfig for SSOConfig {
    fn validate(&self) -> Result<(), String> {
        if self.pf < 0.0 || self.pf > 1.0 { return Err("pf must be in [0, 1]".to_string()); }
        if self.pa < 0.0 || self.pa > 1.0 { return Err("pa must be in [0, 1]".to_string()); }
        if self.pm < 0.0 || self.pm > 1.0 { return Err("pm must be in [0, 1]".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { pf: 0.6, pa: 0.6, ra: 0.2, pm: 0.15 } }
    fn high_accuracy() -> Self { Self { pf: 0.7, pa: 0.8, ra: 0.05, pm: 0.05 } }
}

/// Spider with gender information.
#[derive(Clone)]
struct Spider {
    individual: Individual,
    is_female: bool,
    weight: f64,
}

/// Social Spider Optimizer.
pub struct SocialSpider {
    config: SSOConfig,
    opt_config: OptimizationConfig,
    population: Population,
    spiders: Vec<Spider>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl SocialSpider {
    pub fn new(config: SSOConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let pop_size = opt_config.population_size;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            spiders: Vec::new(),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
        let mut rng = rand::thread_rng();

        // Assign genders based on pf ratio
        self.spiders = self.population.individuals().iter().map(|ind| {
            Spider {
                individual: ind.clone(),
                is_female: rng.gen::<f64>() < self.config.pf,
                weight: 0.0,
            }
        }).collect();
    }

    fn calculate_weight(&self, fitness: f64, worst: f64, best: f64) -> f64 {
        Self::compute_weight(fitness, worst, best)
    }

    fn compute_weight(fitness: f64, worst: f64, best: f64) -> f64 {
        if (worst - best).abs() < 1e-10 {
            0.5
        } else {
            (fitness - worst) / (best - worst)
        }
    }

    fn calculate_vibration(&self, weight: f64, distance: f64) -> f64 {
        weight * (-distance * self.config.ra).exp()
    }

    fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Sync spider individuals with population
        for (i, ind) in self.population.individuals().iter().enumerate() {
            self.spiders[i].individual = ind.clone();
        }

        // Calculate weights
        let fitnesses: Vec<f64> = self.spiders.iter()
            .map(|s| s.individual.fitness.unwrap_or(f64::INFINITY))
            .collect();
        let best_fit = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_fit = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate weights using static helper to avoid borrow issues
        for (i, spider) in self.spiders.iter_mut().enumerate() {
            let fit = fitnesses[i];
            spider.weight = Self::compute_weight(fit, worst_fit, best_fit);
        }

        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let mut rng = rand::thread_rng();
        let n = self.spiders.len();

        // Find best spider
        let best_idx = self.spiders.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.individual.fitness.unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.individual.fitness.unwrap_or(f64::INFINITY))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let spider_snapshot: Vec<Spider> = self.spiders.clone();
        let mut new_positions: Vec<Array1<f64>> = Vec::new();

        for i in 0..n {
            let spider = &spider_snapshot[i];
            let mut new_pos = spider.individual.position.clone();

            if spider.is_female {
                // Female cooperative operator
                let r = rng.gen::<f64>();

                // Find nearest spider with higher weight
                let mut nearest_better = None;
                let mut min_dist = f64::INFINITY;
                for (j, other) in spider_snapshot.iter().enumerate() {
                    if j != i && other.weight > spider.weight {
                        let dist = Self::euclidean_distance(&spider.individual.position, &other.individual.position);
                        if dist < min_dist {
                            min_dist = dist;
                            nearest_better = Some(j);
                        }
                    }
                }

                if r < self.config.pa {
                    // Move towards best and nearest better
                    let vib_best = self.calculate_vibration(
                        spider_snapshot[best_idx].weight,
                        Self::euclidean_distance(&spider.individual.position, &spider_snapshot[best_idx].individual.position)
                    );

                    for d in 0..dim {
                        let alpha: f64 = rng.gen();
                        let beta: f64 = rng.gen();
                        let delta: f64 = rng.gen();

                        new_pos[d] = spider.individual.position[d]
                            + alpha * vib_best * (spider_snapshot[best_idx].individual.position[d] - spider.individual.position[d])
                            + beta * (rng.gen::<f64>() - 0.5);

                        if let Some(nb_idx) = nearest_better {
                            let vib_nb = self.calculate_vibration(
                                spider_snapshot[nb_idx].weight,
                                min_dist
                            );
                            new_pos[d] += delta * vib_nb * (spider_snapshot[nb_idx].individual.position[d] - spider.individual.position[d]);
                        }
                    }
                } else {
                    // Repulsion
                    for d in 0..dim {
                        let alpha: f64 = rng.gen();
                        new_pos[d] = spider.individual.position[d] - alpha * (spider_snapshot[best_idx].individual.position[d] - spider.individual.position[d]);
                    }
                }
            } else {
                // Male operator
                let male_median_weight = {
                    let mut male_weights: Vec<f64> = spider_snapshot.iter()
                        .filter(|s| !s.is_female)
                        .map(|s| s.weight)
                        .collect();
                    male_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if male_weights.is_empty() {
                        0.5
                    } else {
                        male_weights[male_weights.len() / 2]
                    }
                };

                if spider.weight > male_median_weight {
                    // Dominant male - move towards nearest female
                    let nearest_female = spider_snapshot.iter()
                        .enumerate()
                        .filter(|(_, s)| s.is_female)
                        .min_by(|(_, a), (_, b)| {
                            let dist_a = Self::euclidean_distance(&spider.individual.position, &a.individual.position);
                            let dist_b = Self::euclidean_distance(&spider.individual.position, &b.individual.position);
                            dist_a.partial_cmp(&dist_b).unwrap()
                        })
                        .map(|(idx, _)| idx);

                    if let Some(f_idx) = nearest_female {
                        let vib_f = self.calculate_vibration(
                            spider_snapshot[f_idx].weight,
                            Self::euclidean_distance(&spider.individual.position, &spider_snapshot[f_idx].individual.position)
                        );
                        for d in 0..dim {
                            let alpha: f64 = rng.gen();
                            new_pos[d] = spider.individual.position[d]
                                + alpha * vib_f * (spider_snapshot[f_idx].individual.position[d] - spider.individual.position[d]);
                        }
                    }
                } else {
                    // Non-dominant male - move towards center of mass of dominant males
                    let dominant_males: Vec<&Spider> = spider_snapshot.iter()
                        .filter(|s| !s.is_female && s.weight > male_median_weight)
                        .collect();

                    if !dominant_males.is_empty() {
                        let total_weight: f64 = dominant_males.iter().map(|s| s.weight).sum();
                        let mut center: Array1<f64> = Array1::zeros(dim);
                        for dm in &dominant_males {
                            for d in 0..dim {
                                center[d] += dm.weight * dm.individual.position[d] / total_weight;
                            }
                        }
                        for d in 0..dim {
                            let alpha: f64 = rng.gen();
                            let center_val: f64 = center[d];
                            new_pos[d] = spider.individual.position[d]
                                + alpha * (center_val - spider.individual.position[d]);
                        }
                    }
                }
            }

            // Random mating (mutation)
            if rng.gen::<f64>() < self.config.pm {
                for d in 0..dim {
                    let (min, max) = bounds.box_bounds[d];
                    new_pos[d] = rng.gen_range(min..max);
                }
            }

            new_positions.push(bounds.repair(new_pos.view()));
        }

        // Update population
        {
            let mut pop = self.population.individuals_mut();
            for (i, ind) in pop.iter_mut().enumerate() {
                ind.position = new_positions[i].clone();
                ind.fitness = None;
            }
        }

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("SSO failed".to_string()))
    }
}

impl Algorithm for SocialSpider {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::SocialSpider }
    fn name(&self) -> &str { "Social Spider Optimization" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_sso_optimization() {
        let config = SSOConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(50).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut sso = SocialSpider::new(config, opt_config, bounds).unwrap();
        let solution = sso.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
