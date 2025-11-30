//! Thread-safe population management with parallel evaluation.
//!
//! # Formal Properties
//!
//! **Invariant P1**: `population.len() ≤ capacity`
//! **Invariant P2**: All individuals have same dimensionality
//! **Invariant P3**: Global best is always the minimum fitness individual

use crate::core::{Bounds, Individual, ObjectiveFunction};
use crate::OptimizationError;
use ndarray::Array1;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_distr::{Distribution, Uniform};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Thread-safe population for metaheuristic algorithms.
#[derive(Debug)]
pub struct Population {
    /// The individuals in the population
    individuals: RwLock<Vec<Individual>>,
    /// Maximum population size
    pub capacity: usize,
    /// Search space bounds
    pub bounds: Bounds,
    /// Global best individual
    global_best: RwLock<Option<Individual>>,
    /// Generation counter
    generation: RwLock<u32>,
    /// Total fitness evaluations
    evaluations: RwLock<u64>,
}

impl Population {
    /// Create a new empty population.
    #[must_use]
    pub fn new(capacity: usize, bounds: Bounds) -> Self {
        Self {
            individuals: RwLock::new(Vec::with_capacity(capacity)),
            capacity,
            bounds,
            global_best: RwLock::new(None),
            generation: RwLock::new(0),
            evaluations: RwLock::new(0),
        }
    }

    /// Initialize population with random individuals.
    ///
    /// Uses uniform random initialization within bounds.
    pub fn initialize_random(&self, size: usize) {
        let mut rng = rand::thread_rng();
        let _dimension = self.bounds.dimension();

        let individuals: Vec<Individual> = (0..size.min(self.capacity))
            .map(|_| {
                let position = Array1::from_iter(
                    self.bounds.box_bounds.iter().map(|(min, max)| {
                        Uniform::new(*min, *max).sample(&mut rng)
                    })
                );
                Individual::new(position)
            })
            .collect();

        *self.individuals.write() = individuals;
    }

    /// Initialize population with Latin Hypercube Sampling (LHS).
    ///
    /// Provides better coverage of the search space than random.
    pub fn initialize_lhs(&self, size: usize) {
        let mut rng = rand::thread_rng();
        let dimension = self.bounds.dimension();
        let n = size.min(self.capacity);

        // Generate LHS samples
        let samples: Vec<Vec<f64>> = (0..dimension)
            .map(|_| {
                let mut indices: Vec<usize> = (0..n).collect();
                indices.shuffle(&mut rng);
                indices
                    .into_iter()
                    .map(|i| {
                        let lower = i as f64 / n as f64;
                        let upper = (i + 1) as f64 / n as f64;
                        Uniform::new(lower, upper).sample(&mut rng)
                    })
                    .collect()
            })
            .collect();

        // Convert to individuals with proper bounds scaling
        let individuals: Vec<Individual> = (0..n)
            .map(|i| {
                let position = Array1::from_iter(
                    (0..dimension).map(|d| {
                        let (min, max) = self.bounds.box_bounds[d];
                        min + samples[d][i] * (max - min)
                    })
                );
                Individual::new(position)
            })
            .collect();

        *self.individuals.write() = individuals;
    }

    /// Get current population size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.individuals.read().len()
    }

    /// Check if population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.individuals.read().is_empty()
    }

    /// Get current generation.
    #[must_use]
    pub fn generation(&self) -> u32 {
        *self.generation.read()
    }

    /// Increment generation counter.
    pub fn next_generation(&self) {
        *self.generation.write() += 1;
    }

    /// Get total fitness evaluations.
    #[must_use]
    pub fn evaluations(&self) -> u64 {
        *self.evaluations.read()
    }

    /// Evaluate all individuals using the objective function (parallel).
    ///
    /// # Formal Postcondition
    /// `∀ ind ∈ population: ind.is_evaluated()`
    #[cfg(feature = "parallel")]
    pub fn evaluate_parallel<F: ObjectiveFunction + Sync>(&self, objective: &F) -> Result<(), OptimizationError> {
        let fitness_values: Vec<f64> = {
            let individuals = self.individuals.read();
            individuals
                .par_iter()
                .map(|ind| objective.evaluate(ind.position.view()))
                .collect()
        };

        {
            let mut individuals = self.individuals.write();
            for (ind, fitness) in individuals.iter_mut().zip(fitness_values.iter()) {
                ind.update_fitness(*fitness);
            }
        }

        *self.evaluations.write() += self.len() as u64;
        self.update_global_best();
        Ok(())
    }

    /// Evaluate all individuals sequentially.
    #[cfg(not(feature = "parallel"))]
    pub fn evaluate_parallel<F: ObjectiveFunction>(&self, objective: &F) -> Result<(), OptimizationError> {
        self.evaluate_sequential(objective)
    }

    /// Evaluate all individuals sequentially.
    pub fn evaluate_sequential<F: ObjectiveFunction>(&self, objective: &F) -> Result<(), OptimizationError> {
        let mut individuals = self.individuals.write();
        for ind in individuals.iter_mut() {
            let fitness = objective.evaluate(ind.position.view());
            ind.update_fitness(fitness);
        }
        drop(individuals);

        *self.evaluations.write() += self.len() as u64;
        self.update_global_best();
        Ok(())
    }

    /// Update global best from current population.
    fn update_global_best(&self) {
        let individuals = self.individuals.read();
        if let Some(best) = individuals.iter().filter(|i| i.is_evaluated()).min() {
            let mut global = self.global_best.write();
            let should_update = global
                .as_ref()
                .map_or(true, |g| best.fitness < g.fitness);
            if should_update {
                *global = Some(best.clone());
            }
        }
    }

    /// Get the global best individual.
    #[must_use]
    pub fn best(&self) -> Option<Individual> {
        self.global_best.read().clone()
    }

    /// Get the best fitness value.
    #[must_use]
    pub fn best_fitness(&self) -> Option<f64> {
        self.global_best.read().as_ref().and_then(|b| b.fitness)
    }

    /// Calculate population diversity (average pairwise distance).
    ///
    /// # Formal Specification
    /// ```text
    /// diversity = (2 / (n * (n-1))) * Σ_{i<j} distance(ind_i, ind_j)
    /// ```
    #[must_use]
    pub fn diversity(&self) -> f64 {
        let individuals = self.individuals.read();
        let n = individuals.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                total_distance += individuals[i].distance_to(&individuals[j]);
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Get mean fitness of the population.
    #[must_use]
    pub fn mean_fitness(&self) -> f64 {
        let individuals = self.individuals.read();
        let evaluated: Vec<f64> = individuals
            .iter()
            .filter_map(|i| i.fitness)
            .collect();

        if evaluated.is_empty() {
            f64::INFINITY
        } else {
            evaluated.iter().sum::<f64>() / evaluated.len() as f64
        }
    }

    /// Get fitness standard deviation.
    #[must_use]
    pub fn fitness_std(&self) -> f64 {
        let individuals = self.individuals.read();
        let evaluated: Vec<f64> = individuals
            .iter()
            .filter_map(|i| i.fitness)
            .collect();

        if evaluated.len() < 2 {
            return 0.0;
        }

        let mean = evaluated.iter().sum::<f64>() / evaluated.len() as f64;
        let variance = evaluated.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / (evaluated.len() - 1) as f64;
        variance.sqrt()
    }

    /// Sort population by fitness (best first).
    pub fn sort_by_fitness(&self) {
        self.individuals.write().sort();
    }

    /// Get top k individuals.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<Individual> {
        let mut individuals = self.individuals.read().clone();
        individuals.sort();
        individuals.truncate(k);
        individuals
    }

    /// Replace worst individuals with new ones.
    pub fn replace_worst(&self, new_individuals: Vec<Individual>) {
        let mut population = self.individuals.write();
        population.sort();

        let replace_count = new_individuals.len().min(population.len());
        let start = population.len() - replace_count;

        for (i, new_ind) in new_individuals.into_iter().enumerate() {
            if start + i < population.len() {
                population[start + i] = new_ind;
            }
        }
    }

    /// Apply a mutation operator to all individuals.
    pub fn mutate<M>(&self, mutator: M)
    where
        M: Fn(&mut Individual, &Bounds),
    {
        let mut individuals = self.individuals.write();
        for ind in individuals.iter_mut() {
            mutator(ind, &self.bounds);
        }
    }

    /// Get read access to individuals.
    pub fn individuals(&self) -> impl std::ops::Deref<Target = Vec<Individual>> + '_ {
        self.individuals.read()
    }

    /// Get write access to individuals.
    pub fn individuals_mut(&self) -> impl std::ops::DerefMut<Target = Vec<Individual>> + '_ {
        self.individuals.write()
    }

    /// Get individual by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Individual> {
        self.individuals.read().get(index).cloned()
    }

    /// Set individual at index.
    pub fn set(&self, index: usize, individual: Individual) {
        let mut individuals = self.individuals.write();
        if index < individuals.len() {
            individuals[index] = individual;
        }
    }

    /// Add an individual to the population.
    pub fn push(&self, individual: Individual) {
        let mut individuals = self.individuals.write();
        if individuals.len() < self.capacity {
            individuals.push(individual);
        }
    }

    /// Clear the population.
    pub fn clear(&self) {
        self.individuals.write().clear();
        *self.global_best.write() = None;
        *self.generation.write() = 0;
    }
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            individuals: RwLock::new(self.individuals.read().clone()),
            capacity: self.capacity,
            bounds: self.bounds.clone(),
            global_best: RwLock::new(self.global_best.read().clone()),
            generation: RwLock::new(*self.generation.read()),
            evaluations: RwLock::new(*self.evaluations.read()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ObjectiveFunction;
    use ndarray::ArrayView1;

    struct SphereFunction;

    impl ObjectiveFunction for SphereFunction {
        fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
            x.iter().map(|v| v.powi(2)).sum()
        }

        fn bounds(&self) -> &Bounds {
            panic!("Not used in test")
        }

        fn dimension(&self) -> usize {
            0
        }

        fn is_minimization(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_population_creation() {
        let bounds = Bounds::symmetric(3, 10.0);
        let pop = Population::new(50, bounds);

        assert_eq!(pop.capacity, 50);
        assert!(pop.is_empty());
    }

    #[test]
    fn test_random_initialization() {
        let bounds = Bounds::symmetric(3, 10.0);
        let pop = Population::new(50, bounds.clone());
        pop.initialize_random(50);

        assert_eq!(pop.len(), 50);

        // Check all individuals are within bounds
        for ind in pop.individuals().iter() {
            assert!(bounds.is_within_box(ind.position.view()));
        }
    }

    #[test]
    fn test_lhs_initialization() {
        let bounds = Bounds::symmetric(2, 10.0);
        let pop = Population::new(20, bounds.clone());
        pop.initialize_lhs(20);

        assert_eq!(pop.len(), 20);

        // Check all individuals are within bounds
        for ind in pop.individuals().iter() {
            assert!(bounds.is_within_box(ind.position.view()));
        }
    }

    #[test]
    fn test_evaluation() {
        let bounds = Bounds::symmetric(3, 10.0);
        let pop = Population::new(10, bounds);
        pop.initialize_random(10);

        let objective = SphereFunction;
        pop.evaluate_sequential(&objective).unwrap();

        // All should be evaluated
        for ind in pop.individuals().iter() {
            assert!(ind.is_evaluated());
        }

        // Global best should be set
        assert!(pop.best().is_some());
    }

    #[test]
    fn test_diversity() {
        let bounds = Bounds::symmetric(2, 10.0);
        let pop = Population::new(10, bounds);
        pop.initialize_random(10);

        let diversity = pop.diversity();
        assert!(diversity > 0.0);
    }

    #[test]
    fn test_top_k() {
        let bounds = Bounds::symmetric(2, 10.0);
        let pop = Population::new(10, bounds);
        pop.initialize_random(10);

        let objective = SphereFunction;
        pop.evaluate_sequential(&objective).unwrap();

        let top3 = pop.top_k(3);
        assert_eq!(top3.len(), 3);

        // Check they are sorted
        for i in 0..2 {
            assert!(top3[i].fitness <= top3[i + 1].fitness);
        }
    }
}
