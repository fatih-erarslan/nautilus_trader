"""
Genetic algorithm optimizer for trading system parameters.

Implements evolutionary optimization with selection, crossover, and mutation
for finding optimal trading strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import time
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .parameter_optimizer import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents an individual in the genetic population."""
    params: Dict[str, Any]
    fitness: Optional[float] = None
    generation: int = 0
    
    def __lt__(self, other):
        """Compare individuals by fitness (lower is better)."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer."""
    
    def __init__(self, search_space: Dict[str, Dict[str, Any]],
                 population_size: int = 50,
                 elite_size: int = None,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 constraints: Optional[List[Callable]] = None,
                 selection_method: str = 'tournament',
                 tournament_size: int = 3,
                 track_diversity: bool = False,
                 enable_restart: bool = False,
                 stagnation_threshold: int = 50):
        """
        Initialize genetic optimizer.
        
        Args:
            search_space: Parameter definitions
            population_size: Size of population
            elite_size: Number of elite individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            constraints: List of constraint functions
            selection_method: Selection method ('tournament', 'roulette', 'rank')
            tournament_size: Size of tournament selection
            track_diversity: Track population diversity
            enable_restart: Enable restart on stagnation
            stagnation_threshold: Generations without improvement before restart
        """
        super().__init__(search_space, constraints)
        self.population_size = population_size
        self.elite_size = elite_size or max(1, population_size // 10)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.track_diversity = track_diversity
        self.enable_restart = enable_restart
        self.stagnation_threshold = stagnation_threshold
        self.rng = np.random.RandomState()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            # Try to generate valid individual
            for attempt in range(1000):
                params = self.sample_params()
                if self.validate_params(params):
                    population.append(Individual(params=params))
                    break
            else:
                logger.warning("Failed to generate valid individual")
        
        return population
    
    def _evaluate_population(self, population: List[Individual],
                           objective_function: Callable,
                           parallel: bool = False,
                           num_workers: int = 4) -> None:
        """Evaluate fitness for all individuals without fitness."""
        to_evaluate = [ind for ind in population if ind.fitness is None]
        
        if not to_evaluate:
            return
        
        if parallel and len(to_evaluate) > 10:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(objective_function, ind.params): ind
                    for ind in to_evaluate
                }
                
                for future in as_completed(futures):
                    individual = futures[future]
                    try:
                        individual.fitness = future.result()
                    except Exception as e:
                        logger.error(f"Error evaluating individual: {e}")
                        individual.fitness = float('inf')
        else:
            # Sequential evaluation
            for individual in to_evaluate:
                try:
                    individual.fitness = objective_function(individual.params)
                except Exception as e:
                    logger.error(f"Error evaluating individual: {e}")
                    individual.fitness = float('inf')
    
    def _selection(self, population: List[Individual], n_select: int) -> List[Individual]:
        """Select individuals for breeding."""
        if self.selection_method == 'tournament':
            return self._tournament_selection(population, n_select)
        elif self.selection_method == 'roulette':
            return self._roulette_selection(population, n_select)
        elif self.selection_method == 'rank':
            return self._rank_selection(population, n_select)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self, population: List[Individual], 
                            n_select: int) -> List[Individual]:
        """Tournament selection."""
        selected = []
        
        for _ in range(n_select):
            tournament = self.rng.choice(population, size=self.tournament_size, 
                                       replace=False)
            winner = min(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self, population: List[Individual],
                          n_select: int) -> List[Individual]:
        """Roulette wheel selection."""
        # Convert fitness to selection probabilities (minimize)
        fitness_values = np.array([ind.fitness for ind in population])
        
        # Handle negative fitness
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1
        
        # Invert for minimization
        if np.max(fitness_values) > 0:
            fitness_values = np.max(fitness_values) - fitness_values + 1
        else:
            fitness_values = np.ones_like(fitness_values)
        
        # Calculate probabilities
        total_fitness = np.sum(fitness_values)
        if total_fitness > 0:
            probabilities = fitness_values / total_fitness
        else:
            probabilities = np.ones(len(population)) / len(population)
        
        # Select individuals
        indices = self.rng.choice(len(population), size=n_select, p=probabilities)
        return [population[i] for i in indices]
    
    def _rank_selection(self, population: List[Individual],
                       n_select: int) -> List[Individual]:
        """Rank-based selection."""
        # Sort by fitness (ascending)
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        
        # Assign rank-based probabilities
        ranks = np.arange(1, len(sorted_pop) + 1)
        probabilities = ranks / np.sum(ranks)
        
        # Select individuals
        indices = self.rng.choice(len(sorted_pop), size=n_select, p=probabilities)
        return [sorted_pop[i] for i in indices]
    
    def _crossover(self, parent1: Individual, parent2: Individual,
                  generation: int) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if self.rng.rand() > self.crossover_rate:
            # No crossover, return copies
            return (Individual(params=parent1.params.copy(), generation=generation),
                   Individual(params=parent2.params.copy(), generation=generation))
        
        child1_params = {}
        child2_params = {}
        
        for param, config in self.search_space.items():
            param_type = config['type']
            
            if param_type in ['float', 'int']:
                # Blend crossover for numeric parameters
                alpha = self.rng.rand()
                value1 = parent1.params[param]
                value2 = parent2.params[param]
                
                child1_value = alpha * value1 + (1 - alpha) * value2
                child2_value = (1 - alpha) * value1 + alpha * value2
                
                if param_type == 'int':
                    child1_value = int(round(child1_value))
                    child2_value = int(round(child2_value))
                
                # Ensure within bounds
                child1_value = np.clip(child1_value, config['min'], config['max'])
                child2_value = np.clip(child2_value, config['min'], config['max'])
                
                child1_params[param] = child1_value
                child2_params[param] = child2_value
                
            elif param_type == 'categorical':
                # Random swap for categorical
                if self.rng.rand() < 0.5:
                    child1_params[param] = parent1.params[param]
                    child2_params[param] = parent2.params[param]
                else:
                    child1_params[param] = parent2.params[param]
                    child2_params[param] = parent1.params[param]
        
        return (Individual(params=child1_params, generation=generation),
               Individual(params=child2_params, generation=generation))
    
    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        mutated_params = individual.params.copy()
        
        for param, config in self.search_space.items():
            if self.rng.rand() < self.mutation_rate:
                param_type = config['type']
                
                if param_type == 'float':
                    # Gaussian mutation
                    param_range = config['max'] - config['min']
                    std = 0.1 * param_range  # 10% of range
                    value = mutated_params[param] + self.rng.normal(0, std)
                    value = np.clip(value, config['min'], config['max'])
                    mutated_params[param] = value
                    
                elif param_type == 'int':
                    # Integer mutation
                    param_range = config['max'] - config['min']
                    std = max(1, int(0.1 * param_range))
                    value = mutated_params[param] + int(self.rng.normal(0, std))
                    value = np.clip(value, config['min'], config['max'])
                    mutated_params[param] = value
                    
                elif param_type == 'categorical':
                    # Random selection
                    mutated_params[param] = self.rng.choice(config['values'])
        
        return Individual(params=mutated_params, generation=individual.generation)
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = 0.0
                n_params = 0
                
                for param, config in self.search_space.items():
                    param_type = config['type']
                    
                    if param_type in ['float', 'int']:
                        # Normalized distance
                        param_range = config['max'] - config['min']
                        if param_range > 0:
                            d = abs(population[i].params[param] - 
                                   population[j].params[param]) / param_range
                            dist += d
                            n_params += 1
                    elif param_type == 'categorical':
                        # Binary distance
                        if population[i].params[param] != population[j].params[param]:
                            dist += 1
                        n_params += 1
                
                if n_params > 0:
                    distances.append(dist / n_params)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self, objective_function: Callable,
                max_evaluations: int = 1000,
                max_generations: Optional[int] = None,
                target_score: Optional[float] = None,
                convergence_tolerance: Optional[float] = None,
                convergence_window: int = 10,
                diversity_threshold: Optional[float] = None,
                convergence_criteria: Optional[Dict[str, Any]] = None,
                early_stop_callback: Optional[Callable] = None,
                parallel: bool = False,
                num_workers: int = 4) -> OptimizationResult:
        """
        Run genetic algorithm optimization.
        
        Args:
            objective_function: Function to minimize
            max_evaluations: Maximum function evaluations
            max_generations: Maximum generations
            target_score: Stop if score below target
            convergence_tolerance: Convergence tolerance
            convergence_window: Window for convergence check
            diversity_threshold: Stop if diversity below threshold
            convergence_criteria: Dictionary of convergence criteria
            early_stop_callback: Custom early stopping function
            parallel: Use parallel evaluation
            num_workers: Number of parallel workers
            
        Returns:
            OptimizationResult
        """
        start_time = time.time()
        
        # Initialize
        population = self._initialize_population()
        history = []
        best_overall = None
        evaluations = 0
        generation = 0
        converged = False
        convergence_iteration = None
        stagnation_count = 0
        restart_count = 0
        
        # Metrics tracking
        metrics = {
            'generation_best': [],
            'generation_mean': [],
            'restart_count': 0
        }
        
        if self.track_diversity:
            metrics['population_diversity'] = []
        
        # Check convergence criteria
        if convergence_criteria:
            min_iterations = convergence_criteria.get('min_iterations', 0)
            patience = convergence_criteria.get('patience')
            abs_tol = convergence_criteria.get('absolute_tolerance', convergence_tolerance)
            rel_tol = convergence_criteria.get('relative_tolerance')
        else:
            min_iterations = 0
            patience = None
            abs_tol = convergence_tolerance
            rel_tol = None
        
        # Main evolution loop
        while evaluations < max_evaluations:
            if max_generations and generation >= max_generations:
                break
            
            # Evaluate population
            self._evaluate_population(population, objective_function, 
                                    parallel, num_workers)
            evaluations += sum(1 for ind in population if ind.fitness is not None)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Update best
            if best_overall is None or population[0].fitness < best_overall.fitness:
                best_overall = Individual(
                    params=population[0].params.copy(),
                    fitness=population[0].fitness,
                    generation=generation
                )
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Record history
            for ind in population:
                if ind.fitness is not None:
                    history.append({
                        'params': ind.params.copy(),
                        'score': ind.fitness,
                        'iteration': len(history),
                        'generation': generation
                    })
            
            # Track metrics
            fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
            metrics['generation_best'].append(min(fitness_values))
            metrics['generation_mean'].append(np.mean(fitness_values))
            
            if self.track_diversity:
                diversity = self._calculate_diversity(population)
                metrics['population_diversity'].append(diversity)
                
                # Check diversity threshold
                if diversity_threshold and diversity < diversity_threshold:
                    converged = True
                    convergence_iteration = evaluations
                    break
            
            # Check convergence
            if generation >= min_iterations:
                # Target score
                if target_score and best_overall.fitness <= target_score:
                    converged = True
                    convergence_iteration = evaluations
                    break
                
                # Tolerance-based convergence
                if len(metrics['generation_best']) >= convergence_window:
                    recent_best = metrics['generation_best'][-convergence_window:]
                    
                    if abs_tol and max(recent_best) - min(recent_best) < abs_tol:
                        converged = True
                        convergence_iteration = evaluations
                        break
                    
                    if rel_tol:
                        if abs(recent_best[0]) > 1e-10:
                            rel_change = abs(recent_best[-1] - recent_best[0]) / abs(recent_best[0])
                            if rel_change < rel_tol:
                                converged = True
                                convergence_iteration = evaluations
                                break
                
                # Patience-based stopping
                if patience and stagnation_count >= patience:
                    converged = True
                    convergence_iteration = evaluations
                    break
            
            # Custom early stopping
            if early_stop_callback:
                if early_stop_callback(generation, best_overall.fitness, history):
                    converged = True
                    convergence_iteration = evaluations
                    metrics['callback_stopped'] = True
                    break
            
            # Restart on stagnation
            if (self.enable_restart and 
                stagnation_count >= self.stagnation_threshold and
                evaluations < max_evaluations * 0.8):  # Don't restart near end
                
                logger.info(f"Restarting population at generation {generation}")
                
                # Keep elite individuals
                elite = population[:self.elite_size]
                
                # Generate new population
                new_population = self._initialize_population()
                population = elite + new_population[:-self.elite_size]
                
                stagnation_count = 0
                restart_count += 1
                metrics['restart_count'] = restart_count
            
            # Generate next generation
            else:
                # Select elite
                elite = population[:self.elite_size]
                
                # Generate offspring
                offspring = []
                while len(offspring) < self.population_size - self.elite_size:
                    # Select parents
                    parents = self._selection(population, 2)
                    
                    # Crossover
                    child1, child2 = self._crossover(parents[0], parents[1], 
                                                   generation + 1)
                    
                    # Mutation
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    
                    # Add valid offspring
                    if self.validate_params(child1.params):
                        offspring.append(child1)
                    if len(offspring) < self.population_size - self.elite_size:
                        if self.validate_params(child2.params):
                            offspring.append(child2)
                
                # New population
                population = elite + offspring
            
            generation += 1
        
        # Final evaluation
        self._evaluate_population(population, objective_function, parallel, num_workers)
        
        # Compile final result
        return OptimizationResult(
            best_params=best_overall.params,
            best_score=best_overall.fitness,
            history=history,
            converged=converged,
            iterations=evaluations,
            convergence_iteration=convergence_iteration,
            metrics=metrics,
            early_stopped=target_score and best_overall.fitness <= target_score,
            callback_stopped=metrics.get('callback_stopped', False)
        )