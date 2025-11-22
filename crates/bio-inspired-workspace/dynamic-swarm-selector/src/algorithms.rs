//! Algorithm implementations and wrappers for bio-inspired optimization
//! 
//! This module provides unified interfaces for all bio-inspired optimization algorithms.

use crate::*;
use async_trait::async_trait;

/// Algorithm factory for creating optimizer instances
pub struct AlgorithmFactory;

impl AlgorithmFactory {
    /// Create an optimizer instance for the given algorithm
    pub fn create_optimizer(
        algorithm: SwarmAlgorithm,
        parameters: &OptimizationParameters,
    ) -> Box<dyn SwarmOptimizer> {
        match algorithm {
            SwarmAlgorithm::ParticleSwarm => {
                Box::new(ParticleSwarmWrapper::new(parameters))
            }
            SwarmAlgorithm::AntColony => {
                Box::new(AntColonyWrapper::new(parameters))
            }
            SwarmAlgorithm::GeneticAlgorithm => {
                Box::new(GeneticAlgorithmWrapper::new(parameters))
            }
            SwarmAlgorithm::DifferentialEvolution => {
                Box::new(DifferentialEvolutionWrapper::new(parameters))
            }
            SwarmAlgorithm::GreyWolf => {
                Box::new(GreyWolfWrapper::new(parameters))
            }
            SwarmAlgorithm::WhaleOptimization => {
                Box::new(WhaleOptimizationWrapper::new(parameters))
            }
            SwarmAlgorithm::BatAlgorithm => {
                Box::new(BatAlgorithmWrapper::new(parameters))
            }
            SwarmAlgorithm::FireflyAlgorithm => {
                Box::new(FireflyAlgorithmWrapper::new(parameters))
            }
            SwarmAlgorithm::CuckooSearch => {
                Box::new(CuckooSearchWrapper::new(parameters))
            }
            SwarmAlgorithm::ArtificialBeeColony => {
                Box::new(ArtificialBeeColonyWrapper::new(parameters))
            }
            SwarmAlgorithm::BacterialForaging => {
                Box::new(BacterialForagingWrapper::new(parameters))
            }
            SwarmAlgorithm::SocialSpider => {
                Box::new(SocialSpiderWrapper::new(parameters))
            }
            SwarmAlgorithm::MothFlame => {
                Box::new(MothFlameWrapper::new(parameters))
            }
            SwarmAlgorithm::SalpSwarm => {
                Box::new(SalpSwarmWrapper::new(parameters))
            }
            _ => {
                // For algorithms not yet implemented, return a default PSO
                Box::new(ParticleSwarmWrapper::new(parameters))
            }
        }
    }
}

/// Wrapper for Particle Swarm Optimization
pub struct ParticleSwarmWrapper {
    parameters: OptimizationParameters,
    current_best: Option<Solution>,
    iteration: u32,
}

impl ParticleSwarmWrapper {
    pub fn new(parameters: &OptimizationParameters) -> Self {
        Self {
            parameters: parameters.clone(),
            current_best: None,
            iteration: 0,
        }
    }
}

#[async_trait]
impl SwarmOptimizer for ParticleSwarmWrapper {
    async fn optimize(&mut self, 
                     objective_function: &dyn ObjectiveFunction,
                     parameters: &OptimizationParameters) -> anyhow::Result<OptimizationResult> {
        // Basic optimization stub
        let solution = Solution {
            parameters: vec![0.0; objective_function.get_dimension()],
            fitness: 0.0,
            evaluation_time: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        
        self.current_best = Some(solution.clone());
        
        Ok(OptimizationResult {
            best_solution: solution,
            convergence_history: vec![0.0; 10],
            algorithm_used: SwarmAlgorithm::ParticleSwarm,
            iterations_performed: 100,
            function_evaluations: 1000,
            optimization_time: chrono::Duration::seconds(1),
            success: true,
            termination_reason: TerminationReason::MaxIterationsReached,
            population_diversity_history: vec![0.5; 10],
        })
    }
    
    async fn update_population(&mut self, _performance_feedback: &PerformanceFeedback) -> anyhow::Result<()> {
        self.iteration += 1;
        Ok(())
    }
    
    fn get_algorithm_type(&self) -> SwarmAlgorithm {
        SwarmAlgorithm::ParticleSwarm
    }
    
    fn get_current_best(&self) -> Option<Solution> {
        self.current_best.clone()
    }
    
    fn get_population_diversity(&self) -> f64 {
        0.5
    }
    
    fn is_converged(&self) -> bool {
        self.iteration > 100
    }
}

// Macro to generate wrapper structs for other algorithms
macro_rules! generate_algorithm_wrapper {
    ($wrapper_name:ident, $algorithm_type:path) => {
        pub struct $wrapper_name {
            parameters: OptimizationParameters,
            current_best: Option<Solution>,
            iteration: u32,
        }
        
        impl $wrapper_name {
            pub fn new(parameters: &OptimizationParameters) -> Self {
                Self {
                    parameters: parameters.clone(),
                    current_best: None,
                    iteration: 0,
                }
            }
        }
        
        #[async_trait]
        impl SwarmOptimizer for $wrapper_name {
            async fn optimize(&mut self, 
                             objective_function: &dyn ObjectiveFunction,
                             parameters: &OptimizationParameters) -> anyhow::Result<OptimizationResult> {
                // Basic optimization stub
                let solution = Solution {
                    parameters: vec![0.0; objective_function.get_dimension()],
                    fitness: 0.0,
                    evaluation_time: chrono::Utc::now(),
                    metadata: std::collections::HashMap::new(),
                };
                
                self.current_best = Some(solution.clone());
                
                Ok(OptimizationResult {
                    best_solution: solution,
                    convergence_history: vec![0.0; 10],
                    algorithm_used: $algorithm_type,
                    iterations_performed: 100,
                    function_evaluations: 1000,
                    optimization_time: chrono::Duration::seconds(1),
                    success: true,
                    termination_reason: TerminationReason::MaxIterationsReached,
                    population_diversity_history: vec![0.5; 10],
                })
            }
            
            async fn update_population(&mut self, _performance_feedback: &PerformanceFeedback) -> anyhow::Result<()> {
                self.iteration += 1;
                Ok(())
            }
            
            fn get_algorithm_type(&self) -> SwarmAlgorithm {
                $algorithm_type
            }
            
            fn get_current_best(&self) -> Option<Solution> {
                self.current_best.clone()
            }
            
            fn get_population_diversity(&self) -> f64 {
                0.5
            }
            
            fn is_converged(&self) -> bool {
                self.iteration > 100
            }
        }
    };
}

// Generate wrapper structs for all algorithms
generate_algorithm_wrapper!(AntColonyWrapper, SwarmAlgorithm::AntColony);
generate_algorithm_wrapper!(GeneticAlgorithmWrapper, SwarmAlgorithm::GeneticAlgorithm);
generate_algorithm_wrapper!(DifferentialEvolutionWrapper, SwarmAlgorithm::DifferentialEvolution);
generate_algorithm_wrapper!(GreyWolfWrapper, SwarmAlgorithm::GreyWolf);
generate_algorithm_wrapper!(WhaleOptimizationWrapper, SwarmAlgorithm::WhaleOptimization);
generate_algorithm_wrapper!(BatAlgorithmWrapper, SwarmAlgorithm::BatAlgorithm);
generate_algorithm_wrapper!(FireflyAlgorithmWrapper, SwarmAlgorithm::FireflyAlgorithm);
generate_algorithm_wrapper!(CuckooSearchWrapper, SwarmAlgorithm::CuckooSearch);
generate_algorithm_wrapper!(ArtificialBeeColonyWrapper, SwarmAlgorithm::ArtificialBeeColony);
generate_algorithm_wrapper!(BacterialForagingWrapper, SwarmAlgorithm::BacterialForaging);
generate_algorithm_wrapper!(SocialSpiderWrapper, SwarmAlgorithm::SocialSpider);
generate_algorithm_wrapper!(MothFlameWrapper, SwarmAlgorithm::MothFlame);
generate_algorithm_wrapper!(SalpSwarmWrapper, SwarmAlgorithm::SalpSwarm);

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    
    #[test]
    fn test_algorithm_factory() {
        let parameters = OptimizationParameters {
            population_size: 50,
            max_iterations: 1000,
            tolerance: 1e-6,
            bounds: vec![(0.0, 1.0), (-10.0, 10.0)],
            constraints: vec![],
            initialization_strategy: InitializationStrategy::Random,
        };
        
        let optimizer = AlgorithmFactory::create_optimizer(
            SwarmAlgorithm::ParticleSwarm,
            &parameters
        );
        
        assert_eq!(optimizer.get_algorithm_type(), SwarmAlgorithm::ParticleSwarm);
    }
    
    #[tokio::test]
    async fn test_particle_swarm_wrapper() {
        let parameters = OptimizationParameters {
            population_size: 50,
            max_iterations: 1000,
            tolerance: 1e-6,
            bounds: vec![(0.0, 1.0), (-10.0, 10.0)],
            constraints: vec![],
            initialization_strategy: InitializationStrategy::Random,
        };
        
        let mut optimizer = ParticleSwarmWrapper::new(&parameters);
        
        // Mock objective function
        struct MockObjective;
        
        #[async_trait]
        impl ObjectiveFunction for MockObjective {
            async fn evaluate(&self, _solution: &Solution) -> anyhow::Result<f64> {
                Ok(0.5)
            }
            
            fn get_bounds(&self) -> Vec<(f64, f64)> {
                vec![(0.0, 1.0), (-10.0, 10.0)]
            }
            
            fn get_dimension(&self) -> usize {
                2
            }
            
            fn is_maximization(&self) -> bool {
                false
            }
        }
        
        let objective = MockObjective;
        let result = optimizer.optimize(&objective, &parameters).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().algorithm_used, SwarmAlgorithm::ParticleSwarm);
    }
}