//! Core traits for swarm intelligence algorithms

use async_trait::async_trait;
use std::fmt::Debug;
use crate::core::{SwarmError, SwarmResult, OptimizationProblem, Population, Individual};

/// Core trait for all swarm intelligence algorithms
///
/// This trait defines the essential interface that all swarm algorithms must implement.
/// It supports both synchronous and asynchronous execution patterns, with built-in
/// support for parallel processing and performance monitoring.
#[async_trait]
pub trait SwarmAlgorithm: Send + Sync + Debug {
    /// The type representing an individual in the population
    type Individual: Individual + Clone + Send + Sync;
    
    /// The type representing fitness values
    type Fitness: PartialOrd + Clone + Send + Sync + Debug;
    
    /// The type for algorithm-specific parameters
    type Parameters: Clone + Send + Sync + Debug;
    
    /// Initialize the algorithm with the given optimization problem
    ///
    /// This method sets up the initial population, initializes internal state,
    /// and prepares the algorithm for optimization.
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError>;
    
    /// Execute a single optimization step
    ///
    /// This performs one iteration of the algorithm, updating the population
    /// and internal state according to the algorithm's rules.
    async fn step(&mut self) -> Result<(), SwarmError>;
    
    /// Run the complete optimization process
    ///
    /// This is a convenience method that performs initialization and runs
    /// the specified number of iterations.
    async fn optimize(&mut self, max_iterations: usize) -> Result<SwarmResult<Self::Fitness>, SwarmError> {
        let mut best_solution = None;
        let mut convergence_history = Vec::with_capacity(max_iterations);
        
        for iteration in 0..max_iterations {
            self.step().await?;
            
            let current_best = self.get_best_individual();
            if let Some(individual) = current_best {
                let fitness = individual.fitness().clone();
                convergence_history.push(fitness.clone());
                
                if best_solution.is_none() || 
                   fitness < best_solution.as_ref().unwrap().best_fitness {
                    best_solution = Some(SwarmResult {
                        best_position: individual.position().clone(),
                        best_fitness: fitness,
                        iterations: iteration + 1,
                        convergence_history: convergence_history.clone(),
                        algorithm_name: self.name().to_string(),
                    });
                }
            }
            
            // Check for early convergence
            if self.has_converged() {
                break;
            }
        }
        
        best_solution.ok_or(SwarmError::OptimizationError(
            "No solution found during optimization".to_string()
        ))
    }
    
    /// Get the current best individual from the population
    fn get_best_individual(&self) -> Option<&Self::Individual>;
    
    /// Get the current population
    fn get_population(&self) -> &Population<Self::Individual>;
    
    /// Get a mutable reference to the current population  
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual>;
    
    /// Check if the algorithm has converged
    fn has_converged(&self) -> bool {
        false // Default implementation - algorithms can override
    }
    
    /// Get the algorithm name
    fn name(&self) -> &'static str;
    
    /// Get current algorithm parameters
    fn parameters(&self) -> &Self::Parameters;
    
    /// Update algorithm parameters (for adaptive algorithms)
    fn update_parameters(&mut self, params: Self::Parameters);
    
    /// Get performance metrics for the current state
    fn metrics(&self) -> AlgorithmMetrics {
        AlgorithmMetrics::default()
    }
    
    /// Reset the algorithm to initial state
    async fn reset(&mut self) -> Result<(), SwarmError>;
    
    /// Clone the algorithm with current parameters
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
        Individual = Self::Individual,
        Fitness = Self::Fitness,
        Parameters = Self::Parameters
    >>;
}

/// Trait for adaptive algorithms that can tune their parameters
pub trait AdaptiveAlgorithm: SwarmAlgorithm {
    /// Adapt parameters based on current performance
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics);
    
    /// Get parameter adaptation strategy
    fn adaptation_strategy(&self) -> AdaptationStrategy;
}

/// Trait for algorithms that support parallel execution
pub trait ParallelAlgorithm: SwarmAlgorithm {
    /// Execute algorithm step in parallel across multiple threads
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError>;
    
    /// Get optimal thread count for this algorithm
    fn optimal_thread_count(&self) -> usize {
        num_cpus::get()
    }
}

/// Trait for algorithms that can be hybridized with others
pub trait HybridizableAlgorithm: SwarmAlgorithm {
    /// Exchange information with another algorithm
    fn exchange_information<T>(&mut self, other: &T) -> Result<(), SwarmError>
    where
        T: SwarmAlgorithm;
    
    /// Merge populations with another algorithm
    fn merge_populations<T>(&mut self, other: &T) -> Result<(), SwarmError>
    where
        T: SwarmAlgorithm<Individual = Self::Individual>;
}

/// Performance metrics for algorithm evaluation
#[derive(Debug, Clone, Default)]
pub struct AlgorithmMetrics {
    /// Current iteration number
    pub iteration: usize,
    
    /// Best fitness achieved so far
    pub best_fitness: Option<f64>,
    
    /// Average fitness of current population
    pub average_fitness: Option<f64>,
    
    /// Population diversity measure
    pub diversity: Option<f64>,
    
    /// Convergence rate
    pub convergence_rate: Option<f64>,
    
    /// Evaluation count
    pub evaluations: usize,
    
    /// Time per iteration (microseconds)
    pub time_per_iteration: Option<u64>,
    
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
}

/// Parameter adaptation strategies
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Linear adaptation over time
    Linear { start: f64, end: f64 },
    
    /// Exponential decay
    Exponential { initial: f64, decay_rate: f64 },
    
    /// Adaptive based on performance feedback
    Feedback { sensitivity: f64 },
    
    /// Success-based adaptation
    SuccessBased { increase_factor: f64, decrease_factor: f64 },
    
    /// Custom adaptation function
    Custom(Box<dyn Fn(usize, &AlgorithmMetrics) -> f64 + Send + Sync>),
}

/// Trait for evaluating optimization problems
pub trait ObjectiveFunction: Send + Sync {
    type Input;
    type Output: PartialOrd + Clone + Send + Sync;
    
    /// Evaluate the objective function
    fn evaluate(&self, input: &Self::Input) -> Self::Output;
    
    /// Batch evaluation for multiple inputs
    fn evaluate_batch(&self, inputs: &[Self::Input]) -> Vec<Self::Output> {
        inputs.iter().map(|x| self.evaluate(x)).collect()
    }
    
    /// Parallel batch evaluation
    fn evaluate_parallel(&self, inputs: &[Self::Input]) -> Vec<Self::Output>
    where
        Self::Input: Sync,
    {
        use rayon::prelude::*;
        inputs.par_iter().map(|x| self.evaluate(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    
    // Mock algorithm for testing
    #[derive(Debug, Clone)]
    struct MockAlgorithm {
        population: Population<BasicIndividual>,
        parameters: MockParameters,
    }
    
    #[derive(Debug, Clone)]
    struct MockParameters {
        population_size: usize,
    }
    
    #[async_trait]
    impl SwarmAlgorithm for MockAlgorithm {
        type Individual = BasicIndividual;
        type Fitness = f64;
        type Parameters = MockParameters;
        
        async fn initialize(&mut self, _problem: OptimizationProblem) -> Result<(), SwarmError> {
            Ok(())
        }
        
        async fn step(&mut self) -> Result<(), SwarmError> {
            Ok(())
        }
        
        fn get_best_individual(&self) -> Option<&Self::Individual> {
            self.population.individuals.first()
        }
        
        fn get_population(&self) -> &Population<Self::Individual> {
            &self.population
        }
        
        fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
            &mut self.population
        }
        
        fn name(&self) -> &'static str {
            "MockAlgorithm"
        }
        
        fn parameters(&self) -> &Self::Parameters {
            &self.parameters
        }
        
        fn update_parameters(&mut self, params: Self::Parameters) {
            self.parameters = params;
        }
        
        async fn reset(&mut self) -> Result<(), SwarmError> {
            self.population.individuals.clear();
            Ok(())
        }
        
        fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
            Individual = Self::Individual,
            Fitness = Self::Fitness,
            Parameters = Self::Parameters
        >> {
            Box::new(self.clone())
        }
    }
    
    #[tokio::test]
    async fn test_swarm_algorithm_trait() {
        let mut algorithm = MockAlgorithm {
            population: Population::new(),
            parameters: MockParameters { population_size: 10 },
        };
        
        assert_eq!(algorithm.name(), "MockAlgorithm");
        assert!(algorithm.initialize(OptimizationProblem::default()).await.is_ok());
        assert!(algorithm.step().await.is_ok());
    }
}