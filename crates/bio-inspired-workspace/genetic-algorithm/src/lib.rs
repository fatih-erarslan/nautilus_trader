//! Genetic Algorithm (GA) Implementation
//! 
//! This crate provides a basic implementation of the Genetic Algorithm,
//! inspired by the process of natural selection and genetic inheritance.

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Genetic Algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticAlgorithmParameters {
    pub population_size: usize,
    pub max_generations: u32,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub elitism_rate: f64,
}

impl Default for GeneticAlgorithmParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 1000,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            selection_pressure: 2.0,
            elitism_rate: 0.1,
        }
    }
}

/// Individual chromosome in the population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chromosome {
    pub genes: Vec<f64>,
    pub fitness: f64,
}

impl Chromosome {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let genes = (0..dimension)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        Self {
            genes,
            fitness: 0.0,
        }
    }
    
    pub fn crossover(&self, other: &Self) -> (Self, Self) {
        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0..self.genes.len());
        
        let mut child1 = self.clone();
        let mut child2 = other.clone();
        
        for i in crossover_point..self.genes.len() {
            child1.genes[i] = other.genes[i];
            child2.genes[i] = self.genes[i];
        }
        
        (child1, child2)
    }
    
    pub fn mutate(&mut self, mutation_rate: f64, bounds: &[(f64, f64)]) {
        let mut rng = rand::thread_rng();
        
        for i in 0..self.genes.len() {
            if rng.gen::<f64>() < mutation_rate {
                if i < bounds.len() {
                    self.genes[i] = rng.gen_range(bounds[i].0..bounds[i].1);
                } else {
                    self.genes[i] = rng.gen_range(0.0..1.0);
                }
            }
        }
    }
}

/// Genetic Algorithm optimizer
#[derive(Debug)]
pub struct GeneticAlgorithmOptimizer {
    pub parameters: GeneticAlgorithmParameters,
    pub population: Vec<Chromosome>,
    pub best_solution: Option<Chromosome>,
    pub generation: u32,
}

impl GeneticAlgorithmOptimizer {
    pub fn new(parameters: GeneticAlgorithmParameters, dimension: usize) -> Self {
        let population = (0..parameters.population_size)
            .map(|_| Chromosome::new(dimension))
            .collect();
        
        Self {
            parameters,
            population,
            best_solution: None,
            generation: 0,
        }
    }
    
    pub fn initialize_population(&mut self, bounds: &[(f64, f64)]) {
        let mut rng = rand::thread_rng();
        
        for chromosome in &mut self.population {
            chromosome.genes = bounds.iter()
                .map(|(min, max)| rng.gen_range(*min..*max))
                .collect();
            chromosome.fitness = 0.0;
        }
    }
    
    pub fn selection(&self) -> Vec<Chromosome> {
        let mut selected = Vec::new();
        let mut rng = rand::thread_rng();
        
        for _ in 0..self.parameters.population_size {
            let tournament_size = (self.parameters.selection_pressure as usize).max(2);
            let mut tournament: Vec<&Chromosome> = (0..tournament_size)
                .map(|_| &self.population[rng.gen_range(0..self.population.len())])
                .collect();
            
            tournament.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            selected.push(tournament[0].clone());
        }
        
        selected
    }
    
    pub fn evolve(&mut self, bounds: &[(f64, f64)]) {
        let mut rng = rand::thread_rng();
        
        // Selection
        let selected = self.selection();
        
        // Crossover and mutation
        let mut new_population = Vec::new();
        
        // Elitism
        let mut sorted_population = self.population.clone();
        sorted_population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        let elite_count = (self.parameters.elitism_rate * self.parameters.population_size as f64) as usize;
        
        for i in 0..elite_count {
            new_population.push(sorted_population[i].clone());
        }
        
        // Generate rest of population
        while new_population.len() < self.parameters.population_size {
            let parent1 = &selected[rng.gen_range(0..selected.len())];
            let parent2 = &selected[rng.gen_range(0..selected.len())];
            
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.parameters.crossover_rate {
                parent1.crossover(parent2)
            } else {
                (parent1.clone(), parent2.clone())
            };
            
            child1.mutate(self.parameters.mutation_rate, bounds);
            child2.mutate(self.parameters.mutation_rate, bounds);
            
            new_population.push(child1);
            if new_population.len() < self.parameters.population_size {
                new_population.push(child2);
            }
        }
        
        self.population = new_population;
        self.generation += 1;
    }
    
    pub fn get_best_solution(&self) -> Option<&Chromosome> {
        self.best_solution.as_ref()
    }
    
    pub fn get_population_diversity(&self) -> f64 {
        if self.population.is_empty() {
            return 0.0;
        }
        
        let mut diversity = 0.0;
        let n = self.population.len();
        
        for i in 0..n {
            for j in i + 1..n {
                let distance: f64 = self.population[i].genes.iter()
                    .zip(&self.population[j].genes)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                diversity += distance;
            }
        }
        
        diversity / (n * (n - 1) / 2) as f64
    }
    
    pub fn is_converged(&self) -> bool {
        self.get_population_diversity() < 1e-6
    }
}

/// Genetic Algorithm errors
#[derive(Error, Debug)]
pub enum GeneticAlgorithmError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

/// Genetic Algorithm result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticAlgorithmResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub generations: u32,
    pub convergence_history: Vec<f64>,
    pub success: bool,
}

/// Objective function trait
#[async_trait]
pub trait GeneticAlgorithmObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, GeneticAlgorithmError>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
}

/// Simple objective function for testing
pub struct SimpleObjective {
    pub dimension: usize,
    pub bounds: Vec<(f64, f64)>,
}

impl SimpleObjective {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            bounds: vec![(-10.0, 10.0); dimension],
        }
    }
}

#[async_trait]
impl GeneticAlgorithmObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, GeneticAlgorithmError> {
        // Simple sphere function: minimize sum of squares
        let fitness = solution.iter().map(|x| x.powi(2)).sum::<f64>();
        Ok(fitness)
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }
    
    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chromosome_creation() {
        let chromosome = Chromosome::new(3);
        assert_eq!(chromosome.genes.len(), 3);
        assert_eq!(chromosome.fitness, 0.0);
    }
    
    #[test]
    fn test_crossover() {
        let parent1 = Chromosome { genes: vec![1.0, 2.0, 3.0], fitness: 0.0 };
        let parent2 = Chromosome { genes: vec![4.0, 5.0, 6.0], fitness: 0.0 };
        
        let (child1, child2) = parent1.crossover(&parent2);
        assert_eq!(child1.genes.len(), 3);
        assert_eq!(child2.genes.len(), 3);
    }
    
    #[test]
    fn test_parameters_default() {
        let params = GeneticAlgorithmParameters::default();
        assert_eq!(params.population_size, 100);
        assert_eq!(params.max_generations, 1000);
    }
    
    #[tokio::test]
    async fn test_simple_objective() {
        let objective = SimpleObjective::new(2);
        let result = objective.evaluate(&[1.0, 2.0]).await.unwrap();
        assert_eq!(result, 5.0); // 1^2 + 2^2 = 5
    }
}