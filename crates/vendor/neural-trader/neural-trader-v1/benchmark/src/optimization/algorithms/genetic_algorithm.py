"""Genetic Algorithm implementation for parameter optimization.

This module provides a comprehensive genetic algorithm framework
for evolving trading strategy parameters with advanced operators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
import random
from concurrent.futures import ProcessPoolExecutor
import json

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITIST = "elitist"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"


class CrossoverMethod(Enum):
    """Crossover methods."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLX_ALPHA = "blx_alpha"
    SBX = "sbx"  # Simulated Binary Crossover


class MutationMethod(Enum):
    """Mutation methods."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


@dataclass
class Individual:
    """Individual in genetic algorithm population."""
    genes: Dict[str, Any]
    fitness: float
    age: int = 0
    diversity: float = 0.0
    
    def __hash__(self):
        return hash(json.dumps(self.genes, sort_keys=True))


@dataclass
class GeneticAlgorithmResult:
    """Result from genetic algorithm optimization."""
    best_individual: Individual
    population: List[Individual]
    fitness_history: List[float]
    diversity_history: List[float]
    generation_stats: List[Dict[str, float]]
    n_generations: int
    total_evaluations: int


class GeneticAlgorithm:
    """Genetic Algorithm for parameter optimization."""
    
    def __init__(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        population_size: int = 100,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.SBX,
        mutation_method: MutationMethod = MutationMethod.POLYNOMIAL,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        alpha: float = 0.5,  # BLX-alpha parameter
        eta_c: float = 20.0,  # SBX distribution index
        eta_m: float = 20.0,  # Polynomial mutation distribution index
        adaptive_mutation: bool = True,
        diversity_maintenance: bool = True,
        niching: bool = False,
        random_state: Optional[int] = None
    ):
        """Initialize genetic algorithm.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Dictionary defining parameter bounds
            population_size: Size of population
            selection_method: Selection method
            crossover_method: Crossover method
            mutation_method: Mutation method
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_rate: Fraction of elite individuals to preserve
            tournament_size: Size for tournament selection
            alpha: BLX-alpha parameter
            eta_c: SBX distribution index
            eta_m: Polynomial mutation distribution index
            adaptive_mutation: Whether to use adaptive mutation
            diversity_maintenance: Whether to maintain diversity
            niching: Whether to use fitness sharing
            random_state: Random seed
        """
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.adaptive_mutation = adaptive_mutation
        self.diversity_maintenance = diversity_maintenance
        self.niching = niching
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
            
        # Parameter processing
        self.param_names = list(parameter_space.keys())
        self.param_bounds = self._process_parameter_space()
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        
        # Statistics
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.generation_stats: List[Dict[str, float]] = []
        self.n_evaluations = 0
        
        # Adaptive parameters
        self.initial_mutation_rate = mutation_rate
        self.stagnation_counter = 0
        self.last_best_fitness = float('-inf')
        
        logger.info(f"Initialized GA with population size {population_size}")
        
    def _process_parameter_space(self) -> Dict[str, Tuple]:
        """Process parameter space into bounds."""
        bounds = {}
        
        for param_name, param_def in self.parameter_space.items():
            if isinstance(param_def, tuple) and len(param_def) == 2:
                # Continuous parameter
                bounds[param_name] = ('continuous', param_def[0], param_def[1])
            elif isinstance(param_def, list):
                # Categorical parameter
                bounds[param_name] = ('categorical', param_def)
            elif isinstance(param_def, dict):
                # Detailed specification
                if param_def.get('type') == 'integer':
                    bounds[param_name] = ('integer', param_def['low'], param_def['high'])
                elif param_def.get('type') == 'categorical':
                    bounds[param_name] = ('categorical', param_def['choices'])
                else:
                    bounds[param_name] = ('continuous', param_def['low'], param_def['high'])
            else:
                raise ValueError(f"Invalid parameter definition for {param_name}")
                
        return bounds
        
    def _create_random_individual(self) -> Individual:
        """Create random individual."""
        genes = {}
        
        for param_name, (param_type, *bounds) in self.param_bounds.items():
            if param_type == 'continuous':
                low, high = bounds
                genes[param_name] = np.random.uniform(low, high)
            elif param_type == 'integer':
                low, high = bounds
                genes[param_name] = np.random.randint(low, high + 1)
            elif param_type == 'categorical':
                choices = bounds[0]
                genes[param_name] = np.random.choice(choices)
                
        return Individual(genes=genes, fitness=float('-inf'))
        
    def _evaluate_individual(self, individual: Individual) -> float:
        """Evaluate individual fitness."""
        try:
            fitness = self.objective_function(individual.genes)
            self.n_evaluations += 1
            return float(fitness)
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return float('-inf')
            
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            individual.fitness = self._evaluate_individual(individual)
            population.append(individual)
            
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        return population
        
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
        
    def _roulette_selection(self, population: List[Individual]) -> Individual:
        """Roulette wheel selection."""
        # Shift fitness to avoid negative values
        min_fitness = min(ind.fitness for ind in population)
        shifted_fitness = [ind.fitness - min_fitness + 1e-10 for ind in population]
        
        total_fitness = sum(shifted_fitness)
        if total_fitness == 0:
            return random.choice(population)
            
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(shifted_fitness):
            current += fitness
            if current > pick:
                return population[i]
                
        return population[-1]
        
    def _rank_selection(self, population: List[Individual]) -> Individual:
        """Rank-based selection."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        
        # Assign ranks (1 to n)
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        
        for i, rank in enumerate(ranks):
            current += rank
            if current > pick:
                return sorted_pop[i]
                
        return sorted_pop[-1]
        
    def _select_parent(self, population: List[Individual]) -> Individual:
        """Select parent using configured method."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population)
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population)
        else:
            return random.choice(population)
            
    def _single_point_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Select crossover point
        crossover_point = random.randint(1, len(self.param_names) - 1)
        param_list = list(self.param_names)
        
        # Swap genes after crossover point
        for i in range(crossover_point, len(param_list)):
            param = param_list[i]
            genes1[param], genes2[param] = genes2[param], genes1[param]
            
        return (
            Individual(genes=genes1, fitness=float('-inf')),
            Individual(genes=genes2, fitness=float('-inf'))
        )
        
    def _uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Randomly swap each gene
        for param in self.param_names:
            if random.random() < 0.5:
                genes1[param], genes2[param] = genes2[param], genes1[param]
                
        return (
            Individual(genes=genes1, fitness=float('-inf')),
            Individual(genes=genes2, fitness=float('-inf'))
        )
        
    def _arithmetic_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Arithmetic crossover for continuous parameters."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        alpha = random.uniform(0, 1)
        
        for param in self.param_names:
            param_type = self.param_bounds[param][0]
            
            if param_type == 'continuous':
                val1 = parent1.genes[param]
                val2 = parent2.genes[param]
                
                genes1[param] = alpha * val1 + (1 - alpha) * val2
                genes2[param] = (1 - alpha) * val1 + alpha * val2
                
        return (
            Individual(genes=genes1, fitness=float('-inf')),
            Individual(genes=genes2, fitness=float('-inf'))
        )
        
    def _sbx_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Simulated Binary Crossover."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        for param in self.param_names:
            param_type, *bounds = self.param_bounds[param]
            
            if param_type == 'continuous':
                x1 = parent1.genes[param]
                x2 = parent2.genes[param]
                xl, xu = bounds
                
                if abs(x1 - x2) > 1e-14:
                    # Calculate beta
                    if random.random() <= 0.5:
                        beta = (2 * random.random()) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - random.random()))) ** (1.0 / (self.eta_c + 1))
                        
                    # Create offspring
                    c1 = 0.5 * ((x1 + x2) - beta * abs(x1 - x2))
                    c2 = 0.5 * ((x1 + x2) + beta * abs(x1 - x2))
                    
                    # Bound checking
                    genes1[param] = max(xl, min(xu, c1))
                    genes2[param] = max(xl, min(xu, c2))
                    
        return (
            Individual(genes=genes1, fitness=float('-inf')),
            Individual(genes=genes2, fitness=float('-inf'))
        )
        
    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform crossover using configured method."""
        if self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.SBX:
            return self._sbx_crossover(parent1, parent2)
        else:
            return parent1, parent2
            
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        mutated = Individual(genes=individual.genes.copy(), fitness=float('-inf'))
        
        for param in self.param_names:
            if random.random() < self.mutation_rate:
                param_type, *bounds = self.param_bounds[param]
                
                if param_type == 'continuous':
                    low, high = bounds
                    range_size = high - low
                    sigma = range_size * 0.1  # 10% of range
                    
                    new_value = mutated.genes[param] + np.random.normal(0, sigma)
                    mutated.genes[param] = max(low, min(high, new_value))
                    
                elif param_type == 'integer':
                    low, high = bounds
                    if random.random() < 0.5:
                        mutated.genes[param] = min(high, mutated.genes[param] + 1)
                    else:
                        mutated.genes[param] = max(low, mutated.genes[param] - 1)
                        
                elif param_type == 'categorical':
                    choices = bounds[0]
                    mutated.genes[param] = random.choice(choices)
                    
        return mutated
        
    def _polynomial_mutation(self, individual: Individual) -> Individual:
        """Polynomial mutation."""
        mutated = Individual(genes=individual.genes.copy(), fitness=float('-inf'))
        
        for param in self.param_names:
            if random.random() < self.mutation_rate:
                param_type, *bounds = self.param_bounds[param]
                
                if param_type == 'continuous':
                    x = mutated.genes[param]
                    xl, xu = bounds
                    
                    # Polynomial mutation
                    delta1 = (x - xl) / (xu - xl)
                    delta2 = (xu - x) / (xu - xl)
                    
                    rnd = random.random()
                    mut_pow = 1.0 / (self.eta_m + 1.0)
                    
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.eta_m + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.eta_m + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                        
                    x = x + deltaq * (xu - xl)
                    mutated.genes[param] = max(xl, min(xu, x))
                    
        return mutated
        
    def _mutate(self, individual: Individual) -> Individual:
        """Mutate individual using configured method."""
        if self.mutation_method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif self.mutation_method == MutationMethod.POLYNOMIAL:
            return self._polynomial_mutation(individual)
        else:
            return self._gaussian_mutation(individual)
            
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
            
        total_distance = 0.0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._individual_distance(population[i], population[j])
                total_distance += distance
                count += 1
                
        return total_distance / count if count > 0 else 0.0
        
    def _individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals."""
        distance = 0.0
        
        for param in self.param_names:
            param_type, *bounds = self.param_bounds[param]
            
            if param_type in ['continuous', 'integer']:
                low, high = bounds
                range_size = high - low
                
                diff = abs(ind1.genes[param] - ind2.genes[param])
                normalized_diff = diff / range_size if range_size > 0 else 0
                distance += normalized_diff ** 2
            else:
                # Categorical
                distance += 0 if ind1.genes[param] == ind2.genes[param] else 1
                
        return np.sqrt(distance)
        
    def _adaptive_parameters(self, generation: int):
        """Adapt parameters based on evolution progress."""
        if not self.adaptive_mutation:
            return
            
        # Check for stagnation
        current_best = max(ind.fitness for ind in self.population)
        
        if current_best <= self.last_best_fitness + 1e-6:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        self.last_best_fitness = current_best
        
        # Increase mutation rate if stagnating
        if self.stagnation_counter > 5:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
        else:
            self.mutation_rate = max(
                self.initial_mutation_rate,
                self.mutation_rate * 0.95
            )
            
    def _maintain_diversity(self, population: List[Individual]) -> List[Individual]:
        """Maintain population diversity."""
        if not self.diversity_maintenance:
            return population
            
        # Calculate diversity scores
        for i, individual in enumerate(population):
            total_distance = 0.0
            for j, other in enumerate(population):
                if i != j:
                    total_distance += self._individual_distance(individual, other)
            individual.diversity = total_distance / (len(population) - 1)
            
        return population
        
    def optimize(
        self,
        n_generations: int = 100,
        callback: Optional[Callable] = None,
        early_stopping: bool = True,
        patience: int = 20,
        min_delta: float = 1e-6
    ) -> GeneticAlgorithmResult:
        """Run genetic algorithm optimization.
        
        Args:
            n_generations: Number of generations
            callback: Callback function called after each generation
            early_stopping: Whether to use early stopping
            patience: Generations to wait for improvement
            min_delta: Minimum improvement threshold
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting GA optimization for {n_generations} generations")
        
        # Initialize population
        self.population = self._initialize_population()
        self.best_individual = self.population[0]
        
        # Track stopping criteria
        no_improvement_count = 0
        last_best_fitness = float('-inf')
        
        for generation in range(n_generations):
            # Create new generation
            new_population = []
            
            # Elitism - keep best individuals
            n_elites = int(self.elitism_rate * self.population_size)
            elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n_elites]
            new_population.extend(deepcopy(elites))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._select_parent(self.population)
                parent2 = self._select_parent(self.population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                    
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Evaluate children
                child1.fitness = self._evaluate_individual(child1)
                child2.fitness = self._evaluate_individual(child2)
                
                new_population.extend([child1, child2])
                
            # Trim to population size
            new_population = new_population[:self.population_size]
            
            # Update population
            self.population = new_population
            
            # Maintain diversity
            self.population = self._maintain_diversity(self.population)
            
            # Update best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = deepcopy(current_best)
                
            # Calculate statistics
            fitness_values = [ind.fitness for ind in self.population]
            diversity = self._calculate_diversity(self.population)
            
            stats = {
                'generation': generation,
                'best_fitness': max(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'diversity': diversity
            }
            
            self.fitness_history.append(stats['best_fitness'])
            self.diversity_history.append(diversity)
            self.generation_stats.append(stats)
            
            # Adaptive parameters
            self._adaptive_parameters(generation)
            
            # Early stopping check
            if early_stopping:
                if stats['best_fitness'] > last_best_fitness + min_delta:
                    no_improvement_count = 0
                    last_best_fitness = stats['best_fitness']
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping at generation {generation}")
                    break
                    
            # Callback
            if callback:
                callback(stats)
                
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: best={stats['best_fitness']:.6f}, "
                          f"avg={stats['avg_fitness']:.6f}, diversity={diversity:.4f}")
                          
        logger.info(f"GA optimization completed. Best fitness: {self.best_individual.fitness:.6f}")
        
        return GeneticAlgorithmResult(
            best_individual=self.best_individual,
            population=self.population,
            fitness_history=self.fitness_history,
            diversity_history=self.diversity_history,
            generation_stats=self.generation_stats,
            n_generations=len(self.generation_stats),
            total_evaluations=self.n_evaluations
        )
        
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance based on population variance."""
        if not self.population:
            return {}
            
        importance = {}
        
        for param in self.param_names:
            param_type, *bounds = self.param_bounds[param]
            
            if param_type in ['continuous', 'integer']:
                values = [ind.genes[param] for ind in self.population]
                variance = np.var(values)
                
                # Normalize by parameter range
                if param_type == 'continuous':
                    low, high = bounds
                    range_size = high - low
                    normalized_variance = variance / (range_size ** 2)
                else:
                    low, high = bounds
                    range_size = high - low + 1
                    normalized_variance = variance / (range_size ** 2)
                    
                importance[param] = normalized_variance
            else:
                # For categorical, use entropy
                values = [ind.genes[param] for ind in self.population]
                unique_values = list(set(values))
                
                entropy = 0.0
                for value in unique_values:
                    p = values.count(value) / len(values)
                    if p > 0:
                        entropy -= p * np.log2(p)
                        
                importance[param] = entropy
                
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
            
        return importance