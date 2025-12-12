import time
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import h2o
from h2o.automl import H2OAutoML
import gym
from gym import spaces
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import pickle
import uuid
from gymnasium.spaces import Box, Discrete
import random
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime
from collections import deque
import threading
from catboost import CatBoostClassifier, Pool
from h2o.estimators import H2OGradientBoostingEstimator

from hardware_manager import HardwareManager



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantum.rl")


class OptimizationGoal(Enum):
    """Possible optimization goals for the RL agent"""

    SIGNAL_TO_NOISE = auto()
    LATENCY = auto()
    ACCURACY = auto()
    MULTI_OBJECTIVE = auto()


@dataclass
class RewardConfig:
    """Configuration for the reward function"""

    snr_weight: float = 0.5
    latency_weight: float = 0.2
    accuracy_weight: float = 0.3
    min_reward: float = -10.0
    max_reward: float = 10.0
    use_shaping: bool = True
    reward_scaling: float = 1.0


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quantum.signal")


class ProcessingMode(Enum):
    """Signal processing execution modes"""

    CLASSICAL = auto()
    QUANTUM = auto()
    HYBRID = auto()
    AUTO = auto()


class QuantumBackend(Enum):
    """Supported quantum backends"""

    PENNYLANE = auto()
    QISKIT = auto()
    CIRQ = auto()
    CUSTOM = auto()


@dataclass
class SignalMetadata:
    """Metadata for signal processing"""

    sample_rate: float
    dimension: int
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    labels: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[str] = field(default_factory=list)


class ANFISGAFilter:
    """Adaptive Neuro-Fuzzy Inference System with Genetic Algorithm optimization for noise filtering."""

    def __init__(self, parameters=None):
        self.parameters = parameters or {}
        self.membership_functions = self.parameters.get("membership_functions", 3)
        self.genetic_population = self.parameters.get("genetic_population", 20)
        self.genetic_generations = self.parameters.get("genetic_generations", 10)
        self.fuzzy_rules = []
        self.trained = False
        self.performance_history = []
        self.adaptation_rate = 0.1
        self.window_size = self.parameters.get("window", 5)

        # Create a cache for previously computed filters to avoid redundant calculations
        self.filter_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"ANFIS-GA filter initialized with {self.membership_functions} membership functions"
        )

    def _generate_fuzzy_rules(self, training_data):
        """Generate fuzzy rules based on training data using genetic algorithm."""
        # Implementation would use a GA to find optimal fuzzy rules
        # This is a simplified version
        return [
            {"weight": 0.8, "condition": "increasing", "action": "smooth"},
            {"weight": 0.6, "condition": "volatile", "action": "dampen"},
            {"weight": 0.7, "condition": "trend_reversal", "action": "amplify"},
        ]

    def train(self, data):
        """Train the ANFIS system on historical data."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning("Invalid training data provided to ANFIS-GA")
            return False

        try:
            # Extract features for training
            features = self._extract_features(data)

            # Generate fuzzy rules using genetic algorithm
            self.fuzzy_rules = self._generate_fuzzy_rules(features)

            # Calculate initial performance metrics
            perf_metrics = self._evaluate_performance(data)
            self.performance_history.append(perf_metrics)

            self.trained = True
            logger.info(
                f"ANFIS-GA trained successfully: {len(self.fuzzy_rules)} rules generated"
            )
            return True

        except Exception as e:
            logger.error(f"Error training ANFIS-GA: {e}")
            return False

    def _extract_features(self, data):
        """Extract relevant features for fuzzy rule generation."""
        features = {}

        # Price movement features
        if "close" in data.columns:
            features["price_changes"] = data["close"].pct_change().dropna().values
            features["volatility"] = features["price_changes"].std()
            features["trend"] = np.sign(data["close"].iloc[-1] - data["close"].iloc[0])

        # Detect patterns (simplified)
        features["has_spike"] = (
            any(abs(data["close"].pct_change()) > 0.03)
            if "close" in data.columns
            else False
        )

        return features

    def _apply_fuzzy_rules(self, value, context):
        """Apply fuzzy rules to a value based on context."""
        # Real implementation would do fuzzy inference
        adjusted_value = value

        # Simple logic based on rules (simplified)
        for rule in self.fuzzy_rules:
            if rule["condition"] == "volatile" and context.get("volatility", 0) > 0.02:
                adjusted_value = value * 0.8  # Dampen volatility
            elif rule["condition"] == "trend_reversal" and context.get(
                "potential_reversal", False
            ):
                adjusted_value = value * 1.1  # Amplify trend reversals

        return adjusted_value

    def filter_noise(self, data):
        """Filter noise from data using trained ANFIS-GA system."""
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning("Invalid or empty DataFrame input for ANFIS-GA filtering.")
            return pd.DataFrame()

        try:
            filtered = data.copy()

            # Check cache first
            cache_key = hash(
                tuple(data["close"].values) if "close" in data.columns else None
            )
            if cache_key in self.filter_cache:
                self.cache_hits += 1
                logger.debug(
                    f"Cache hit for ANFIS-GA filtering (hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses):.2f})"
                )
                return self.filter_cache[cache_key]

            self.cache_misses += 1

            # Calculate context for fuzzy rule application
            context = self._extract_features(data)

            # Apply fuzzy filtering
            if "close" in data.columns:
                filtered_values = []
                raw_values = data["close"].values

                for i in range(len(raw_values)):
                    window_start = max(0, i - self.window_size)
                    window_values = raw_values[window_start : i + 1]

                    # Basic smoothing (would be replaced by fuzzy inference)
                    if len(window_values) > 0:
                        base_value = window_values[-1]
                        smoothed = np.mean(window_values)

                        # Apply fuzzy rules with context
                        local_context = context.copy()
                        local_context["position"] = i / len(raw_values)
                        local_context["potential_reversal"] = self._detect_reversal(
                            window_values
                        )

                        adjusted = self._apply_fuzzy_rules(smoothed, local_context)
                        filtered_values.append(adjusted)
                    else:
                        filtered_values.append(raw_values[i])

                filtered["close"] = filtered_values

                # Also adjust high/low for consistency if available
                if "high" in filtered.columns and "low" in filtered.columns:
                    offset_high = filtered["high"] - data["close"]
                    offset_low = data["close"] - filtered["low"]
                    filtered["high"] = filtered["close"] + offset_high
                    filtered["low"] = filtered["close"] - offset_low

            # Store in cache (limit cache size)
            if len(self.filter_cache) > 100:  # Max cache size
                self.filter_cache.pop(next(iter(self.filter_cache)))
            self.filter_cache[cache_key] = filtered

            logger.info("ANFIS-GA filtering applied with fuzzy rule system")
            return filtered

        except Exception as e:
            logger.error(f"Error in ANFIS-GA filtering: {e}")
            return data  # Return original data on error

    def _detect_reversal(self, values):
        """Detect potential trend reversals in a window of values."""
        if len(values) < 3:
            return False

        # Simple reversal detection (would be more sophisticated in real implementation)
        trend_before = np.sign(values[-2] - values[-3])
        trend_after = np.sign(values[-1] - values[-2])

        return trend_before != trend_after and abs(trend_after) > 0

    def _evaluate_performance(self, data):
        """Evaluate the performance of the filtering."""
        filtered = self.filter_noise(data)

        # Calculate noise reduction
        if "close" in data.columns and "close" in filtered.columns:
            original_volatility = data["close"].pct_change().std()
            filtered_volatility = filtered["close"].pct_change().std()
            noise_reduction = 1 - (filtered_volatility / original_volatility)

            return {
                "noise_reduction": noise_reduction,
                "timestamp": pd.Timestamp.now(),
                "data_points": len(data),
            }

        return {
            "noise_reduction": 0,
            "timestamp": pd.Timestamp.now(),
            "data_points": len(data),
        }

    def update_with_feedback(self, original_data, filtered_data, trading_outcome):
        """Update the filter based on trading outcome feedback."""
        # Use trading outcomes to adjust fuzzy rule weights
        try:
            profit = trading_outcome.get("profit", 0)

            # Adjust rule weights based on profit
            for rule in self.fuzzy_rules:
                # Reinforce rules that led to profit, reduce weight of rules that led to loss
                adjustment = (
                    self.adaptation_rate * np.sign(profit) * min(abs(profit), 0.1)
                )
                rule["weight"] = max(0.1, min(1.0, rule["weight"] + adjustment))

            logger.info(
                f"Updated ANFIS-GA fuzzy rules based on trading outcome: profit={profit:.4f}"
            )

        except Exception as e:
            logger.error(f"Error updating ANFIS-GA with feedback: {e}")

class GeneticFuzzyOptimizer:
    """
    Genetic Algorithm-based optimizer for fuzzy system parameters.
    """

    def __init__(
        self,
        population_size: int = 50,
        chromosome_length: int = None,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_count: int = 5,
        fitness_function: Callable = None,
    ):
        """
        Initialize the genetic optimizer.

        Args:
            population_size: Size of the population
            chromosome_length: Length of each chromosome (solution)
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_count: Number of top individuals to preserve
            fitness_function: Function to evaluate chromosome fitness
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.fitness_function = fitness_function

        # Initialize population
        self.population = None
        self.fitness_scores = None
        self.generation = 0
        self.best_solution = None
        self.best_fitness = -float("inf")

        self.fitness_history = []

        logger.info(
            f"Initialized Genetic Optimizer with population size {population_size}"
        )

    def initialize_population(self, chromosome_length: int = None):
        """
        Initialize the population with random chromosomes.

        Args:
            chromosome_length: Length of each chromosome (solution)
        """
        if chromosome_length is not None:
            self.chromosome_length = chromosome_length

        if self.chromosome_length is None:
            raise ValueError("Chromosome length must be specified")

        # Generate random population
        self.population = np.random.uniform(
            0, 1, (self.population_size, self.chromosome_length)
        )
        self.fitness_scores = np.zeros(self.population_size)
        self.generation = 0

        logger.info(
            f"Initialized population with chromosome length {self.chromosome_length}"
        )

    def random_rule_set(self):
        """Generates a random fuzzy rule set with market conditions in mind."""
        return {
            "low_volatility_signal": random.uniform(-0.8, 0.8),
            "medium_volatility_signal": random.uniform(-0.8, 0.8),
            "high_volatility_signal": random.uniform(-1.0, 1.0),
        }

    def evaluate_fitness(self, fitness_function: Callable = None):
        """
        Evaluate fitness of each chromosome in the population.

        Args:
            fitness_function: Function to evaluate chromosome fitness
        """
        if fitness_function is not None:
            self.fitness_function = fitness_function

        if self.fitness_function is None:
            raise ValueError("Fitness function must be provided")

        # Evaluate fitness for each chromosome
        for i in range(self.population_size):
            self.fitness_scores[i] = self.fitness_function(self.population[i])

        # Update best solution
        max_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[max_idx]
            self.best_solution = self.population[max_idx].copy()

        # Record fitness history
        self.fitness_history.append(
            {
                "mean": np.mean(self.fitness_scores),
                "max": np.max(self.fitness_scores),
                "min": np.min(self.fitness_scores),
                "std": np.std(self.fitness_scores),
            }
        )

        logger.debug(
            f"Generation {self.generation}: Best fitness = {self.best_fitness:.4f}"
        )

    def select_parents(self) -> np.ndarray:
        """
        Select parents for reproduction using tournament selection.

        Returns:
            Selected parents indices
        """
        # Tournament selection
        tournament_size = 3
        selected_indices = np.zeros(self.population_size, dtype=int)

        for i in range(self.population_size):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(
                self.population_size, tournament_size, replace=False
            )

            # Select the best individual from tournament
            tournament_fitness = self.fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]

            selected_indices[i] = winner_idx

        return selected_indices

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two offspring chromosomes
        """
        # Check if crossover should occur
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Two-point crossover
        points = np.sort(
            np.random.choice(self.chromosome_length - 1, 2, replace=False) + 1
        )

        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        # Swap segments between parents
        if points[0] < points[1]:
            offspring1[points[0] : points[1]] = parent2[points[0] : points[1]]
            offspring2[points[0] : points[1]] = parent1[points[0] : points[1]]

        return offspring1, offspring2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Apply mutation to a chromosome.

        Args:
            chromosome: Chromosome to mutate

        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()

        # Apply mutation with given probability
        for i in range(self.chromosome_length):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, 0.1)
                mutated[i] += mutation

        # Ensure values stay in valid range [0, 1]
        mutated = np.clip(mutated, 0, 1)

        return mutated

    def evolve(self):
        """
        Evolve the population to the next generation.
        """
        # Select parents
        parent_indices = self.select_parents()

        # Create new population
        new_population = np.zeros_like(self.population)

        # Elitism: carry over best individuals unchanged
        elite_indices = np.argsort(self.fitness_scores)[-self.elitism_count :]
        for i in range(self.elitism_count):
            new_population[i] = self.population[elite_indices[i]]

        # Create offspring for rest of population
        for i in range(self.elitism_count, self.population_size, 2):
            # Select two parents
            parent1_idx = parent_indices[i - self.elitism_count]
            parent2_idx = parent_indices[
                i - self.elitism_count + 1 if i < self.population_size - 1 else 0
            ]

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Create offspring through crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Apply mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Add to new population
            new_population[i] = offspring1
            if i + 1 < self.population_size:
                new_population[i + 1] = offspring2

        # Update population
        self.population = new_population
        self.generation += 1

    def get_best_rule_set(self) -> Dict[str, float]:
        """Returns the best optimized fuzzy rule set."""
        return max(self.population, key=self.fitness_function)

    def optimize(self, model=None, data=None):
        """
        Optimizes the given model or produces an optimized rule set.
        Can be used to optimize SANFIS models if provided.

        Args:
            model: Optional, the SANFIS model to optimize
            data: Optional, training data if model is provided

        Returns:
            The best rule set or optimized model
        """
        self.evolve()
        best_rules = self.get_best_rule_set()

    def run(self, num_generations: int, fitness_function: Callable = None):
        """
        Run the genetic algorithm for the specified number of generations.

        Args:
            num_generations: Number of generations to run
            fitness_function: Function to evaluate chromosome fitness
        """
        if fitness_function is not None:
            self.fitness_function = fitness_function

        if self.population is None:
            self.initialize_population()

        # Run for specified generations
        for _ in range(num_generations):
            self.evaluate_fitness()
            self.evolve()

        # Final evaluation
        self.evaluate_fitness()

        logger.info(
            f"Completed {num_generations} generations. Best fitness: {self.best_fitness:.4f}"
        )

        return self.best_solution, self.best_fitness



class HybridANFISFilter(ANFISGAFilter):
    """
    Hybrid ANFIS Filter combining SANFIS model with GA optimization.
    """

    def __init__(self, num_rules: int = 5, learning_rate: float = 0.05):
        self.sanfis_model = SANFISModel(
            num_rules=num_rules, learning_rate=learning_rate
        )
        self.ga_optimizer = GeneticFuzzyOptimizer(adaptive_mutation=True)
        self.optimized = False
        logger.info("Hybrid ANFIS Filter initialized")
        # Preload with simple training data to avoid "not trained" warnings
        self._preload_simple_data()

    def _preload_simple_data(self):
        """Preload with simple synthetic data to ensure system is ready"""
        try:
            # Create simple synthetic data
            import numpy as np

            x = np.linspace(-1, 1, 20)
            y = np.linspace(0, 1, 20)

            # Create a simple sine pattern
            mock_data = pd.DataFrame(
                {
                    "momentum": np.sin(x * 3),
                    "volatility": 0.2 + 0.1 * np.abs(np.sin(y * 2)),
                    "close": 100 + 10 * np.sin(x),
                }
            )

            # Add required columns for optimization
            mock_data["momentum_norm"] = mock_data["momentum"]
            mock_data["volatility_norm"] = mock_data["volatility"]

            # Train on this data
            self._optimize_filter(mock_data)
            logger.info("ANFIS filter preloaded with synthetic data")
        except Exception as e:
            logger.warning(f"Failed to preload ANFIS filter: {e}")

    def filter_noise(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filter noise from market data using SANFIS and GA.

        Args:
            dataframe: Input market data

        Returns:
            pd.DataFrame: Filtered data with adaptive signals
        """
        if dataframe.empty:
            logger.warning("Empty dataframe provided to ANFIS filter")
            return dataframe

        try:
            # Create a copy to avoid modifying the original
            result = dataframe.copy()

            # Make sure we have volatility and momentum features
            if "volatility" not in result.columns:
                result["volatility"] = (
                    result["close"].pct_change().rolling(window=14).std()
                )

            if "momentum" not in result.columns:
                result["momentum"] = result["close"].pct_change(periods=10)

            # Fill NaN values
            result.ffill(inplace=True)
            result.fillna(0, inplace=True)

            # Optimize filter if not already optimized
            if not self.optimized and len(result) > 50:
                self._optimize_filter(result)

            # Apply SANFIS evaluation to generate adaptive signal
            result["adaptive_signal"] = result.apply(
                lambda row: self.sanfis_model.evaluate(
                    row["momentum"], row["volatility"]
                ),
                axis=1,
            )

            return result

        except Exception as e:
            logger.error(f"Error in ANFIS filter: {e}")
            # Return original data if filtering fails
            return dataframe

    def _optimize_filter(self, data: pd.DataFrame) -> bool:
        """
        Optimize the ANFIS filter using GA.

        Args:
            data: Market data for optimization

        Returns:
            bool: True if optimization was successful
        """
        try:
            # Use GA to optimize rules
            best_rules = self.ga_optimizer.optimize()

            # Apply optimized parameters to SANFIS model
            self.sanfis_model.adapt_parameters(data["momentum"].mean())

            self.optimized = True
            logger.info("ANFIS filter optimized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize ANFIS filter: {e}")
            return False


class SANFISModel:
    """
    Self-Adaptive Neuro-Fuzzy Inference System model for RL state representation
    and action prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_rules: int = 5,
        learning_rate: float = 0.01,
        adaptive_rules: bool = True,
        *args, 
        **kwargs
    ):
        """
        Initialize the SANFIS model.

        Args:
            input_dim: Input dimensionality (state dimension)
            output_dim: Output dimensionality (action dimension)
            num_rules: Number of fuzzy rules
            learning_rate: Learning rate for parameter updates
            adaptive_rules: Whether to adapt rule count and membership functions
        """
        print(">>> SANFISModel __init__ ENTERED <<<") 
        print(f"RECEIVED input_dim: {input_dim} (Type: {type(input_dim).__name__})")
        print(f"RECEIVED output_dim: {output_dim} (Type: {type(output_dim).__name__})")
        # Print values safely using .get() on kwargs if they MIGHT be keywords instead
        print(f"RECEIVED num_rules (via keyword or default): {kwargs.get('num_rules', 5)}") 
        print(f"RECEIVED learning_rate (via keyword or default): {kwargs.get('learning_rate', 0.01)}")
        print(f"RECEIVED adaptive_rules (via keyword or default): {kwargs.get('adaptive_rules', True)}")
        if args: logger.critical(f"RECEIVED *args: {args}") 
        if kwargs: logger.critical(f"RECEIVED **kwargs: {kwargs}") 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rules = num_rules
        self.learning_rate = learning_rate
        self.adaptive_rules = adaptive_rules

        # Initialize membership function parameters
        # Each input dimension has num_rules membership functions
        # Each MF has center and width parameters
        self.centers = np.random.uniform(-1, 1, (input_dim, num_rules))
        self.widths = np.random.uniform(0.1, 0.5, (input_dim, num_rules))

        # Initialize rule consequent parameters
        # Each rule has a linear function for each output dimension
        self.consequents = np.random.uniform(
            -0.1, 0.1, (num_rules, input_dim + 1, output_dim)
        )

        # Initialize rule weights
        self.rule_weights = np.ones(num_rules) / num_rules

        # Training history
        self.loss_history = []

        logger.info(
            f"Initialized SANFIS model with {num_rules} rules, "
            f"{input_dim} inputs, and {output_dim} outputs"
        )

    def membership_degree(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate membership degrees for each rule and input dimension.

        Args:
            x: Input vector [input_dim]

        Returns:
            Membership degrees [input_dim, num_rules]
        """
        # Ensure x has the right shape
        x = np.atleast_1d(x)

        # Calculate Gaussian membership degrees
        # exp(-(x - center)^2 / (2 * width^2))
        memberships = np.zeros((self.input_dim, self.num_rules))

        for i in range(self.input_dim):
            for j in range(self.num_rules):
                memberships[i, j] = np.exp(
                    -((x[i] - self.centers[i, j]) ** 2) / (2 * self.widths[i, j] ** 2)
                )

        return memberships

    def rule_firing_strengths(self, memberships: np.ndarray) -> np.ndarray:
        """
        Calculate rule firing strengths using product t-norm.

        Args:
            memberships: Membership degrees [input_dim, num_rules]

        Returns:
            Rule firing strengths [num_rules]
        """
        # Product t-norm (multiply memberships across input dimensions)
        firing_strengths = np.prod(memberships, axis=0)

        # Apply rule weights
        weighted_strengths = firing_strengths * self.rule_weights

        # Normalize firing strengths
        sum_strengths = np.sum(weighted_strengths)
        if sum_strengths > 0:
            normalized_strengths = weighted_strengths / sum_strengths
        else:
            normalized_strengths = np.ones(self.num_rules) / self.num_rules

        return normalized_strengths

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the SANFIS model.

        Args:
            x: Input vector [input_dim]

        Returns:
            Predicted output vector [output_dim]
        """
        # Ensure x has the right shape
        x = np.atleast_1d(x)

        # Calculate membership degrees
        memberships = self.membership_degree(x)

        # Calculate rule firing strengths
        firing_strengths = self.rule_firing_strengths(memberships)

        # Calculate rule outputs
        rule_outputs = np.zeros((self.num_rules, self.output_dim))

        for i in range(self.num_rules):
            # Concatenate input with bias term
            x_bias = np.concatenate([x, [1.0]])

            # Calculate linear output for each rule and output dimension
            for j in range(self.output_dim):
                rule_outputs[i, j] = np.dot(x_bias, self.consequents[i, :, j])

        # Weighted sum of rule outputs
        output = np.zeros(self.output_dim)
        for i in range(self.num_rules):
            output += firing_strengths[i] * rule_outputs[i]

        return output

    def update(self, x: np.ndarray, y: np.ndarray, sample_weight: float = 1.0) -> float:
        """
        Update model parameters using backpropagation.

        Args:
            x: Input vector [input_dim]
            y: Target output vector [output_dim]
            sample_weight: Weight for this sample

        Returns:
            Loss value
        """
        # Make prediction
        y_pred = self.predict(x)

        # Calculate loss
        loss = np.mean((y - y_pred) ** 2)

        # Backpropagate error
        # This is a simplified implementation - in practice, you'd use automatic differentiation

        # Calculate error gradient for output
        output_grad = -2 * (y - y_pred) * sample_weight

        # Calculate membership degrees
        memberships = self.membership_degree(x)

        # Calculate rule firing strengths
        firing_strengths = self.rule_firing_strengths(memberships)

        # Update consequent parameters
        for i in range(self.num_rules):
            for j in range(self.output_dim):
                # Gradient for consequent parameters
                x_bias = np.concatenate([x, [1.0]])
                consequent_grad = firing_strengths[i] * output_grad[j] * x_bias

                # Update consequent parameters
                self.consequents[i, :, j] -= self.learning_rate * consequent_grad

        # Update membership function parameters (centers and widths)
        # This is a simplified implementation - full gradient calculation is complex

        # Record loss
        self.loss_history.append(loss)

        # Adapt rule count if needed
        if self.adaptive_rules and len(self.loss_history) % 100 == 0:
            self._adapt_rules()

        return loss

    def _adapt_rules(self):
        """Adapt the number of rules and membership functions"""
        # Check if we need more rules (high loss)
        if len(self.loss_history) >= 100:
            recent_loss = np.mean(self.loss_history[-20:])
            previous_loss = np.mean(self.loss_history[-100:-80])

            # If loss is not decreasing significantly, add a rule
            if recent_loss > previous_loss * 0.95 and self.num_rules < 20:
                self._add_rule()

            # If loss is decreasing well, consider removing a rule
            elif recent_loss < previous_loss * 0.5 and self.num_rules > 2:
                self._remove_weakest_rule()

    def _add_rule(self):
        """Add a new fuzzy rule"""
        # Add new center, width, consequent and rule weight
        new_centers = np.zeros((self.input_dim, self.num_rules + 1))
        new_widths = np.zeros((self.input_dim, self.num_rules + 1))
        new_consequents = np.zeros(
            (self.num_rules + 1, self.input_dim + 1, self.output_dim)
        )
        new_weights = np.zeros(self.num_rules + 1)

        # Copy existing values
        new_centers[:, : self.num_rules] = self.centers
        new_widths[:, : self.num_rules] = self.widths
        new_consequents[: self.num_rules] = self.consequents
        new_weights[: self.num_rules] = self.rule_weights

        # Initialize new rule
        # Set center of new rule to region of high error
        new_centers[:, -1] = np.random.uniform(-1, 1, self.input_dim)
        new_widths[:, -1] = np.random.uniform(0.1, 0.5, self.input_dim)
        new_consequents[-1] = np.random.uniform(
            -0.1, 0.1, (self.input_dim + 1, self.output_dim)
        )
        new_weights[-1] = 1.0 / (self.num_rules + 1)

        # Normalize weights
        new_weights = new_weights / np.sum(new_weights)

        # Update model parameters
        self.centers = new_centers
        self.widths = new_widths
        self.consequents = new_consequents
        self.rule_weights = new_weights
        self.num_rules += 1

        logger.info(f"Added new rule, now using {self.num_rules} rules")

    def _remove_weakest_rule(self):
        """Remove the weakest rule"""
        # Find the weakest rule (lowest average firing strength)
        if len(self.rule_weights) <= 2:
            return  # Keep at least 2 rules

        weakest_idx = np.argmin(self.rule_weights)

        # Remove rule parameters
        self.centers = np.delete(self.centers, weakest_idx, axis=1)
        self.widths = np.delete(self.widths, weakest_idx, axis=1)
        self.consequents = np.delete(self.consequents, weakest_idx, axis=0)
        self.rule_weights = np.delete(self.rule_weights, weakest_idx)

        # Normalize remaining weights
        self.rule_weights = self.rule_weights / np.sum(self.rule_weights)
        self.num_rules -= 1

        logger.info(f"Removed weakest rule, now using {self.num_rules} rules")

    def save(self, filepath: str):
        """Save model to file"""
        model_data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_rules": self.num_rules,
            "learning_rate": self.learning_rate,
            "adaptive_rules": self.adaptive_rules,
            "centers": self.centers,
            "widths": self.widths,
            "consequents": self.consequents,
            "rule_weights": self.rule_weights,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load model from file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        model = cls(
            input_dim=model_data["input_dim"],
            output_dim=model_data["output_dim"],
            num_rules=model_data["num_rules"],
            learning_rate=model_data["learning_rate"],
            adaptive_rules=model_data["adaptive_rules"],
        )

        model.centers = model_data["centers"]
        model.widths = model_data["widths"]
        model.consequents = model_data["consequents"]
        model.rule_weights = model_data["rule_weights"]

        return model


class H2OManager:
    """
    Manager for H2O AutoML integration with reinforcement learning.
    """

    def __init__(
        self,
        max_models: int = 20,
        max_runtime_secs: int = 300,
        seed: int = 42,
        model_path: str = "./h2o_models",
    ):
        """
        Initialize H2O manager.

        Args:
            max_models: Maximum number of models to train
            max_runtime_secs: Maximum runtime in seconds
            seed: Random seed for reproducibility
            model_path: Path to save models
        """
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.model_path = model_path

        # Initialize H2O
        self._init_h2o()

        # Store models
        self.models = {}
        self.automl = None
        self.current_leaderboard = None

        os.makedirs(model_path, exist_ok=True)

        logger.info(
            f"Initialized H2O Manager with max_models={max_models}, "
            f"max_runtime_secs={max_runtime_secs}"
        )

    def _init_h2o(self):
        """Initialize H2O cluster"""
        try:
            h2o.init()
            logger.info(f"H2O initialized: {h2o.cluster().info}")
        except Exception as e:
            logger.error(f"Error initializing H2O: {str(e)}")
            raise

    def train_model(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        model_name: str,
        include_algos: List[str] = None,
        exclude_algos: List[str] = None,
    ) -> h2o.automl.H2OAutoML:
        """
        Train an AutoML model to predict actions from states.

        Args:
            states: List of state vectors
            actions: List of action vectors
            rewards: List of rewards for weighting
            model_name: Name for the model
            include_algos: Algorithms to include
            exclude_algos: Algorithms to exclude

        Returns:
            Trained AutoML model
        """
        # Convert data to H2O format
        states_array = np.array(states)
        actions_array = np.array(actions)

        # Create feature names
        state_cols = [f"state_{i}" for i in range(states_array.shape[1])]
        action_cols = [f"action_{i}" for i in range(actions_array.shape[1])]

        # Create pandas DataFrame
        import pandas as pd

        df = pd.DataFrame(states_array, columns=state_cols)

        # Add actions as targets
        for i, col in enumerate(action_cols):
            df[col] = actions_array[:, i]

        # Add rewards as weights
        df["weight"] = np.array(rewards)

        # Convert to H2O frame
        h2o_frame = h2o.H2OFrame(df)

        # Split into train/test
        train, test = h2o_frame.split_frame(ratios=[0.8], seed=self.seed)

        # Define AutoML settings
        automl = H2OAutoML(
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            sort_metric="RMSE",
            include_algos=include_algos,
            exclude_algos=exclude_algos,
        )

        # Train model
        automl.train(
            x=state_cols,
            y=action_cols,
            training_frame=train,
            validation_frame=test,
            weights_column="weight",
        )

        # Store model
        self.models[model_name] = automl
        self.automl = automl
        self.current_leaderboard = automl.leaderboard

        # Save model
        model_path = os.path.join(self.model_path, model_name)
        h2o.save_model(automl.leader, path=model_path)

        logger.info(
            f"Trained H2O AutoML model '{model_name}'. "
            f"Best model: {automl.leader.model_id}"
        )

        return automl

    def predict_actions(self, model_name: str, states: List[np.ndarray]) -> np.ndarray:
        """
        Predict actions using the specified model.

        Args:
            model_name: Name of the model to use
            states: List of state vectors

        Returns:
            Predicted actions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        # Convert states to H2O frame
        states_array = np.array(states)
        state_cols = [f"state_{i}" for i in range(states_array.shape[1])]

        import pandas as pd

        df = pd.DataFrame(states_array, columns=state_cols)
        h2o_frame = h2o.H2OFrame(df)

        # Get predictions
        preds = self.models[model_name].leader.predict(h2o_frame)

        # Convert to numpy array
        action_cols = [col for col in preds.columns if col.startswith("action_")]
        actions = preds[action_cols].as_data_frame().values

        return actions

    def get_leaderboard(self, model_name: str = None) -> h2o.H2OFrame:
        """
        Get the leaderboard for the specified model.

        Args:
            model_name: Name of the model

        Returns:
            Leaderboard as H2O frame
        """
        if model_name is not None and model_name in self.models:
            return self.models[model_name].leaderboard
        elif self.current_leaderboard is not None:
            return self.current_leaderboard
        else:
            raise ValueError("No models trained yet")

    def load_model(
        self, model_name: str, model_path: str = None
    ) -> h2o.automl.H2OAutoML:
        """
        Load a saved model.

        Args:
            model_name: Name for the model
            model_path: Path to the model file

        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = os.path.join(self.model_path, model_name)

        try:
            model = h2o.load_model(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded model '{model_name}' from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            raise

    def shutdown(self):
        """Shutdown H2O cluster"""
        try:
            h2o.cluster().shutdown()
            logger.info("H2O cluster shut down")
        except Exception as e:
            logger.warning(f"Error shutting down H2O cluster: {str(e)}")


class CatBoostRLModel:
    """
    CatBoost-based model for reinforcement learning tasks.
    Handles both continuous and categorical state/action spaces with
    efficient handling of mixed data types.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        categorical_features: List[int] = None,
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        task_type: str = "CPU",
        seed: int = 42,
        model_path: str = "./catboost_models",
    ):
        """
        Initialize CatBoost RL model.

        Args:
            state_dim: State dimensionality
            action_dim: Action dimensionality
            categorical_features: Indices of categorical features
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            task_type: 'CPU' or 'GPU'
            seed: Random seed
            model_path: Path to save models
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.categorical_features = categorical_features if categorical_features else []
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        self.seed = seed
        self.model_path = model_path

        # Initialize CatBoost models (one for each action dimension)
        try:
            from catboost import CatBoostRegressor

            self.models = [
                CatBoostRegressor(
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    task_type=task_type,
                    random_seed=seed,
                    verbose=100,
                    cat_features=categorical_features,
                )
                for _ in range(action_dim)
            ]
            self.model_class = CatBoostRegressor
            logger.info(
                f"Initialized CatBoost models for {action_dim} action dimensions"
            )
        except ImportError:
            logger.error(
                "CatBoost not installed. Please install with: pip install catboost"
            )
            raise

        os.makedirs(model_path, exist_ok=True)

        # Store training history
        self.train_losses = [[] for _ in range(action_dim)]
        self.val_losses = [[] for _ in range(action_dim)]

    def train(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        weights: List[float] = None,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 50,
        model_name: str = "model",
    ):
        """
        Train CatBoost models on state-action pairs.

        Args:
            states: List of state vectors
            actions: List of action vectors
            weights: Optional sample weights
            validation_fraction: Fraction of data to use for validation
            early_stopping_rounds: Early stopping parameter
            model_name: Name for the model
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split

        states_array = np.array(states)
        actions_array = np.array(actions)

        # Create feature names
        feature_cols = [f"feature_{i}" for i in range(states_array.shape[1])]

        # Create DataFrame for states
        df = pd.DataFrame(states_array, columns=feature_cols)

        # Split into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            states_array,
            actions_array,
            test_size=validation_fraction,
            random_state=self.seed,
        )

        # Create DataFrames
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)

        # Train a model for each action dimension
        for i in range(self.action_dim):
            logger.info(
                f"Training model for action dimension {i + 1}/{self.action_dim}"
            )

            # Get target values for this dimension
            y_train_i = y_train[:, i]
            y_val_i = y_val[:, i]

            # Train the model
            self.models[i].fit(
                X_train_df,
                y_train_i,
                eval_set=(X_val_df, y_val_i),
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True,
                verbose=100,
            )

            # Store training history
            self.train_losses[i] = self.models[i].evals_result_["learn"]["RMSE"]
            self.val_losses[i] = self.models[i].evals_result_["validation"]["RMSE"]

        # Save models
        self.save(model_name)

        logger.info(f"Trained CatBoost models and saved as '{model_name}'")

    def predict(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Predict actions for given states.

        Args:
            states: List of state vectors

        Returns:
            Predicted actions
        """
        import pandas as pd

        states_array = np.array(states)

        # Create feature names
        feature_cols = [f"feature_{i}" for i in range(states_array.shape[1])]

        # Create DataFrame
        df = pd.DataFrame(states_array, columns=feature_cols)

        # Predict each action dimension
        actions = np.zeros((len(states), self.action_dim))

        for i in range(self.action_dim):
            actions[:, i] = self.models[i].predict(df)

        return actions

    def save(self, model_name: str):
        """
        Save models to disk.

        Args:
            model_name: Name for the model
        """
        for i, model in enumerate(self.models):
            model_path = os.path.join(self.model_path, f"{model_name}_dim{i}.cbm")
            model.save_model(model_path)

    def load(self, model_name: str):
        """
        Load models from disk.

        Args:
            model_name: Name of the model to load
        """
        try:
            self.models = []

            for i in range(self.action_dim):
                model_path = os.path.join(self.model_path, f"{model_name}_dim{i}.cbm")
                model = self.model_class()
                model.load_model(model_path)
                self.models.append(model)

            logger.info(f"Loaded CatBoost models '{model_name}'")
        except Exception as e:
            logger.error(f"Error loading CatBoost models: {str(e)}")
            raise

    def get_feature_importance(self, model_index: int = 0):
        """
        Get feature importance for a specific model.

        Args:
            model_index: Index of the model

        Returns:
            Feature importance
        """
        if model_index >= len(self.models):
            raise ValueError(f"Invalid model index: {model_index}")

        return self.models[model_index].get_feature_importance()


class H2OCatBoostHybrid:
    """
    A hybrid implementation combining H2O and CatBoost with hardware-specific GPU acceleration
    for AMD RX6800XT and Nvidia GPUs with intelligent fallback mechanisms.
    """

    def __init__(
        self,
        max_depth: int = 7,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        h2o_port: int = 54321,
        h2o_nthreads: int = -1,
        catboost_gpu_id: int = 0,
        prefer: str = "auto",  # Options: "auto", "h2o", "catboost"
        gpu_memory_allocation: float = 0.9,  # Fraction of GPU memory to use
        enable_performance_monitoring: bool = True,
    ):
        """
        Initialize the hybrid model with specified parameters.

        Args:
            max_depth: Maximum tree depth for both models
            n_estimators: Number of estimators/trees
            learning_rate: Learning rate for gradient boosting
            h2o_port: Port for H2O server
            h2o_nthreads: Number of threads for H2O (-1 for all)
            catboost_gpu_id: GPU ID for CatBoost
            prefer: Which model to prefer when both are available
            gpu_memory_allocation: Fraction of GPU memory to allocate (0.0-1.0)
            enable_performance_monitoring: Whether to monitor and report performance metrics
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.h2o_port = h2o_port
        self.h2o_nthreads = h2o_nthreads
        self.catboost_gpu_id = catboost_gpu_id
        self.prefer = prefer
        self.gpu_memory_allocation = gpu_memory_allocation
        self.enable_performance_monitoring = enable_performance_monitoring

        # Model instances
        self.h2o_model = None
        self.catboost_model = None

        # Status flags
        self.h2o_available = False
        self.catboost_gpu_available = False
        self.active_model = (
            None  # Will be set to "h2o", "catboost_gpu", or "catboost_cpu"
        )

        # GPU detection
        #self.gpu_info = GPUInfo()

        # Performance monitoring
        self.performance_metrics = {
            "train_time": 0,
            "predict_time": 0,
            "train_samples_per_second": 0,
            "predict_samples_per_second": 0,
        }

        # Initialize both platforms and detect availability
        self._initialize_platforms()

    def _initialize_platforms(self):
        """
        Initialize H2O and CatBoost platforms and detect GPU availability.
        """
        # Try to initialize H2O
        try:
            h2o.init(port=self.h2o_port, nthreads=self.h2o_nthreads)
            self.h2o_available = True
            logger.info("H2O initialized successfully")
        except Exception as e:
            logger.warning(f"H2O initialization failed: {str(e)}")
            self.h2o_available = False

        # Check if GPU is available for CatBoost
        self.catboost_gpu_available = self._check_catboost_gpu()

        # Determine which model to use as primary
        self._configure_active_model()

    def _check_catboost_gpu(self) -> bool:
        """
        Check if GPU is available for CatBoost.

        Returns:
            bool: True if GPU is available for CatBoost
        """
        # First check if CUDA is available via PyTorch as a quick test
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                # Create a tiny CatBoost model with task_type=GPU
                tiny_model = CatBoostClassifier(
                    iterations=1,
                    depth=1,
                    task_type="GPU",
                    devices=f"{self.catboost_gpu_id}",
                    verbose=False,
                )

                # Create minimal test data
                X = np.random.random((5, 3))
                y = np.random.randint(0, 2, 5)

                # Try to fit the model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tiny_model.fit(X, y)

                logger.info("CatBoost GPU acceleration available")
                return True
            except Exception as e:
                logger.warning(f"CatBoost GPU check failed: {str(e)}")

        logger.info("CatBoost will run in CPU mode")
        return False

    def _configure_active_model(self):
        """
        Configure which model to use as primary based on availability, hardware, and preference.
        """
        # Hardware-aware selection logic
        if self.prefer == "auto":
            # Auto-select with hardware-specific logic
            if self.catboost_gpu_available:
                # For RX6800XT, use a faster hybrid approach
                if self.gpu_info.is_rx6800xt:
                    self.active_model = "catboost_gpu"
                    logger.info("Using RX6800XT-optimized CatBoost configuration")
                # For other GPUs, standard selection logic
                else:
                    self.active_model = "catboost_gpu"
            elif self.h2o_available:
                self.active_model = "h2o"
            else:
                self.active_model = "catboost_cpu"
        elif self.prefer == "h2o" and self.h2o_available:
            self.active_model = "h2o"
        elif self.prefer == "catboost":
            if self.catboost_gpu_available:
                self.active_model = "catboost_gpu"
            else:
                self.active_model = "catboost_cpu"
        else:
            # Default fallback with hardware awareness
            if self.catboost_gpu_available:
                self.active_model = "catboost_gpu"
            elif self.h2o_available:
                self.active_model = "h2o"
            else:
                self.active_model = "catboost_cpu"

        logger.info(f"Active model set to: {self.active_model}")

        # Initialize the active model
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize the selected models based on active_model setting with hardware-specific optimizations.
        """
        # Initialize H2O model if needed
        if self.active_model == "h2o" or (
            self.h2o_available and self.active_model == "catboost_gpu"
        ):
            # Create base H2O model parameters
            h2o_params = {
                "ntrees": self.n_estimators,
                "max_depth": self.max_depth,
                "learn_rate": self.learning_rate,
                "distribution": "bernoulli",
            }

            # Add hardware-specific optimizations for H2O
            if self.gpu_info.has_gpu:
                # H2O can benefit from GPUs for some operations
                if self.gpu_info.gpu_type == "nvidia" and self.gpu_info.cuda_available:
                    # Nvidia-specific optimizations
                    h2o_params.update(
                        {
                            "histogram_type": "UniformAdaptive",
                            "build_tree_one_node": False,
                            "sample_rate": 0.8 if self.gpu_info.vram_mb > 8000 else 0.7,
                        }
                    )
                elif self.gpu_info.gpu_type == "amd" and self.gpu_info.is_rx6800xt:
                    # RX6800XT specific optimizations
                    h2o_params.update(
                        {
                            "histogram_type": "QuantilesGlobal",
                            "build_tree_one_node": False,
                            "sample_rate": 0.8,
                            "col_sample_rate_per_tree": 0.8,
                        }
                    )

            # Create the model with optimized parameters
            self.h2o_model = H2OGradientBoostingEstimator(**h2o_params)

        # Initialize CatBoost model if needed
        if "catboost" in self.active_model:
            # Base CatBoost parameters
            catboost_params = {
                "iterations": self.n_estimators,
                "depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "loss_function": "Logloss",
                "verbose": False,
                "thread_count": -1,  # Use all CPU cores
            }

            # Add GPU parameters if available
            if self.active_model == "catboost_gpu":
                # Get GPU-optimized parameters based on detected hardware
                gpu_params = self.gpu_info.get_optimal_catboost_params()

                # Update with GPU memory allocation from user setting
                if "gpu_ram_part" in gpu_params:
                    gpu_params["gpu_ram_part"] = min(
                        self.gpu_memory_allocation, gpu_params["gpu_ram_part"]
                    )

                # Add hardware-specific parameter adjustments
                if self.gpu_info.is_rx6800xt:
                    # Special parameters for RX6800XT
                    catboost_params.update(
                        {
                            "bootstrap_type": "MVS",  # Works well on AMD
                            "grow_policy": "Depthwise",  # Better for RDNA2 architecture
                            "min_data_in_leaf": 5,
                            "l2_leaf_reg": 2.0,
                            "random_strength": 1.0,
                            "rsm": 0.95,  # Use most features to leverage RDNA2 compute units
                        }
                    )
                elif self.gpu_info.gpu_type == "nvidia":
                    # Different optimizations for NVIDIA
                    if self.gpu_info.vram_mb >= 16000:  # High-end with lots of VRAM
                        catboost_params.update(
                            {
                                "bootstrap_type": "Bayesian",
                                "grow_policy": "SymmetricTree",
                                "min_data_in_leaf": 3,
                                "l2_leaf_reg": 1.0,
                            }
                        )
                    else:  # More conservative for mid/low-end
                        catboost_params.update(
                            {
                                "grow_policy": "Lossguide",
                                "min_data_in_leaf": 5,
                                "l2_leaf_reg": 3.0,
                            }
                        )

                # Merge in GPU parameters
                catboost_params.update(gpu_params)

                # Ensure device ID is set correctly
                catboost_params["devices"] = f"{self.catboost_gpu_id}"

            # Create model with appropriate parameters
            self.catboost_model = CatBoostClassifier(**catboost_params)

            # Log the parameters for debugging/optimization
            logger.debug(
                f"CatBoost initialized with parameters: {json.dumps(catboost_params, default=str)}"
            )

    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> bool:
        """
        Fit the hybrid model with hardware-aware fallback mechanisms.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            bool: True if fitting was successful
        """
        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Check dataset size for optimizations
        n_samples, n_features = X.shape
        logger.info(
            f"Training on dataset with {n_samples} samples and {n_features} features"
        )

        # Start performance monitoring if enabled
        start_time = None
        if self.enable_performance_monitoring:
            start_time = time.time()

        # Try to fit the active model first
        success = False

        # AMD RX6800XT specific training logic
        if self.gpu_info.is_rx6800xt and "catboost" in self.active_model:
            logger.info("Using RX6800XT-optimized training pipeline")

            # For large datasets on RX6800XT, use batched training for better memory management
            if n_samples > 100000:
                success = self._fit_catboost_batched(X, y, batch_size=50000)
            else:
                success = self._fit_catboost(X, y)

            # For RX6800XT, we still want an H2O backup model if available
            if success and self.h2o_available:
                try:
                    logger.info("Training H2O backup model for RX6800XT")
                    self._fit_h2o(X, y)
                except Exception as e:
                    logger.warning(f"H2O backup model training failed: {str(e)}")

            # If RX6800XT training failed, try standard fallbacks
            if not success:
                if self.h2o_available:
                    logger.info("RX6800XT CatBoost fitting failed, falling back to H2O")
                    self.active_model = "h2o"
                    self._initialize_models()
                    success = self._fit_h2o(X, y)
                else:
                    logger.info("RX6800XT CatBoost fitting failed, falling back to CPU")
                    self.active_model = "catboost_cpu"
                    self._initialize_models()
                    success = self._fit_catboost(X, y)

        # Standard training logic for other hardware
        elif self.active_model == "h2o":
            success = self._fit_h2o(X, y)
            if not success and self.catboost_gpu_available:
                logger.info("H2O fitting failed, falling back to CatBoost GPU")
                self.active_model = "catboost_gpu"
                self._initialize_models()
                success = self._fit_catboost(X, y)
            elif not success:
                logger.info("H2O fitting failed, falling back to CatBoost CPU")
                self.active_model = "catboost_cpu"
                self._initialize_models()
                success = self._fit_catboost(X, y)

        elif "catboost" in self.active_model:
            # Standard CatBoost training
            success = self._fit_catboost(X, y)

            # Fallback handling for other hardware
            if not success and self.h2o_available:
                logger.info("CatBoost fitting failed, falling back to H2O")
                self.active_model = "h2o"
                self._initialize_models()
                success = self._fit_h2o(X, y)
            elif not success and self.active_model == "catboost_gpu":
                logger.info("CatBoost GPU fitting failed, falling back to CPU")
                self.active_model = "catboost_cpu"
                self._initialize_models()
                success = self._fit_catboost(X, y)

        # Train backup model if appropriate
        # If we have H2O as backup and primary didn't fail, also train H2O model
        if (
            success
            and self.active_model != "h2o"
            and self.h2o_available
            and not self.gpu_info.is_rx6800xt
        ):
            # For RX6800XT, we've already handled backup model logic above
            try:
                # Train H2O model in background as backup
                logger.info("Training H2O backup model")
                self._fit_h2o(X, y)
            except Exception as e:
                logger.warning(f"H2O backup model training failed: {str(e)}")

        # Record performance metrics if enabled
        if self.enable_performance_monitoring and start_time is not None:
            train_time = time.time() - start_time
            self.performance_metrics["train_time"] = train_time
            self.performance_metrics["train_samples_per_second"] = (
                n_samples / train_time
            )
            logger.info(
                f"Training completed in {train_time:.2f} seconds ({n_samples / train_time:.2f} samples/sec)"
            )

        return success

    def _fit_catboost_batched(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 50000
    ) -> bool:
        """
        Train CatBoost model in batches for better memory management on GPUs.

        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Size of batches for training

        Returns:
            bool: True if fitting was successful
        """
        try:
            n_samples = X.shape[0]
            logger.info(f"Training CatBoost in batches of {batch_size} samples")

            # Initialize model
            if self.catboost_model is None:
                self._initialize_models()

            # Create initial pool with a subset of data
            init_size = min(batch_size, n_samples)
            train_pool = Pool(X[:init_size], y[:init_size])

            # Initial fit
            self.catboost_model.fit(train_pool)

            # Process remaining data in batches if needed
            for i in range(init_size, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                logger.info(f"Processing batch {i // batch_size + 1} ({i}:{end})")

                # Create batch pool
                batch_pool = Pool(X[i:end], y[i:end])

                # Update model with new data
                # Use continue_from_model parameter to resume training
                self.catboost_model.fit(
                    batch_pool,
                    init_model=self.catboost_model,  # Continue from current model
                    eval_set=batch_pool,  # Use current batch as validation
                    use_best_model=True,
                )

            return True
        except Exception as e:
            logger.error(f"Batched CatBoost training failed: {str(e)}")
            return False

    def _fit_h2o(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Fit the H2O model with hardware optimizations.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            bool: True if fitting was successful
        """
        try:
            # Check if model needs initialization
            if self.h2o_model is None:
                self._initialize_models()

            # Convert data to H2O format
            train = h2o.H2OFrame(np.column_stack((X, y)))
            train.columns = [f"feat_{i}" for i in range(X.shape[1])] + ["target"]

            # AMD RX6800XT specific optimization for H2O
            if self.gpu_info.is_rx6800xt:
                # Add additional optimal parameters for RX6800XT
                self.h2o_model.sample_rate = 0.85  # Slightly better for RDNA2
                self.h2o_model.col_sample_rate_per_tree = 0.85

                # Add specialized H2O settings for AMD GPUs if necessary
                # These are applied at model creation time so they're in _initialize_models

            # Fit the model
            self.h2o_model.train(
                x=[f"feat_{i}" for i in range(X.shape[1])],
                y="target",
                training_frame=train,
            )
            return True
        except Exception as e:
            logger.error(f"H2O model training failed: {str(e)}")
            return False

    def _fit_catboost(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Fit the CatBoost model with hardware-aware optimizations.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            bool: True if fitting was successful
        """
        try:
            # Check dataset size
            n_samples, n_features = X.shape

            # Create pool for faster training with appropriate parameters
            if n_samples > 100000 and n_features > 50:
                # For large datasets, use more optimized memory settings
                train_pool = Pool(
                    X,
                    y,
                    thread_count=-1,  # Use all cores for data processing
                )
            else:
                train_pool = Pool(X, y)

            # Special VRAM management for AMD RX6800XT
            if self.gpu_info.is_rx6800xt and self.active_model == "catboost_gpu":
                try:
                    # RX6800XT-specific training approach
                    logger.info("Using RX6800XT-specific training method")

                    # Special parameters for RX6800XT when training
                    # These complement the ones already set in _initialize_models
                    rx6800xt_train_params = {
                        "use_best_model": True,
                        "early_stopping_rounds": 50,  # Helps with VRAM management
                    }

                    # Apply specialized XGBoost training parameters
                    for param, value in rx6800xt_train_params.items():
                        self.catboost_model.set_params(**{param: value})

                    # Utilize RX6800XT specialized training
                    self.catboost_model.fit(train_pool)

                except Exception as e:
                    logger.warning(
                        f"RX6800XT-specific training failed: {str(e)}, falling back to standard approach"
                    )
                    # Fall back to standard training
                    self.catboost_model.fit(train_pool)
            else:
                # Standard or NVIDIA-specific approach
                try:
                    # If using NVIDIA GPU with CUDA, add CUDA specialized parameters
                    if (
                        self.gpu_info.gpu_type == "nvidia"
                        and self.gpu_info.cuda_available
                        and self.active_model == "catboost_gpu"
                    ):
                        # Get CUDA version to optimize parameters
                        cuda_major = int(self.gpu_info.cuda_version.split(".")[0])

                        # CUDA 11+ optimizations
                        if cuda_major >= 11:
                            self.catboost_model.set_params(
                                use_best_model=True,
                                fold_len_multiplier=1.2,  # Better for newer CUDA
                            )

                    # Standard training
                    self.catboost_model.fit(train_pool)

                except Exception as e:
                    # If we're trying GPU but it fails, we'll fall back to CPU
                    if self.active_model == "catboost_gpu" and "GPU" in str(e):
                        logger.warning(
                            "GPU training failed, falling back to CPU within CatBoost"
                        )
                        # Update model to CPU mode
                        self.catboost_model.set_params(task_type="CPU")
                        self.catboost_model.fit(train_pool)
                    else:
                        # Re-raise if it's not a GPU-related error
                        raise

            return True
        except Exception as e:
            logger.error(f"CatBoost model training failed: {str(e)}")
            return False

    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """
        Generate probability predictions with hardware-aware fallback mechanisms.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Probability predictions
        """
        # Convert input to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Start performance monitoring if enabled
        start_time = None
        if self.enable_performance_monitoring:
            start_time = time.time()

        # Check prediction size for optimizations
        n_samples, n_features = X.shape

        # RX6800XT-specific prediction optimization
        if self.gpu_info.is_rx6800xt and "catboost" in self.active_model:
            # For large datasets on RX6800XT, use batched prediction
            if n_samples > 50000:
                try:
                    logger.info(
                        f"Using batched prediction for {n_samples} samples on RX6800XT"
                    )
                    result = self._predict_catboost_batched(X, batch_size=50000)

                    # Record performance metrics if enabled
                    if self.enable_performance_monitoring and start_time is not None:
                        predict_time = time.time() - start_time
                        self.performance_metrics["predict_time"] = predict_time
                        self.performance_metrics["predict_samples_per_second"] = (
                            n_samples / predict_time
                        )
                        logger.info(
                            f"Prediction completed in {predict_time:.2f} seconds ({n_samples / predict_time:.2f} samples/sec)"
                        )

                    return result
                except Exception as e:
                    logger.warning(
                        f"Batched prediction failed: {str(e)}, falling back to standard prediction"
                    )
                    # If batched prediction fails, continue with standard prediction

        # Try primary model first
        try:
            result = None
            if self.active_model == "h2o":
                result = self._predict_h2o(X)
            else:  # catboost models
                result = self._predict_catboost(X)

            # Record performance metrics if enabled
            if self.enable_performance_monitoring and start_time is not None:
                predict_time = time.time() - start_time
                self.performance_metrics["predict_time"] = predict_time
                self.performance_metrics["predict_samples_per_second"] = (
                    n_samples / predict_time
                )
                logger.info(
                    f"Prediction completed in {predict_time:.2f} seconds ({n_samples / predict_time:.2f} samples/sec)"
                )

            return result
        except Exception as e:
            logger.warning(
                f"Prediction with {self.active_model} failed: {str(e)}, trying fallbacks"
            )

            # Try fallbacks in order of preference
            fallbacks = []
            if "catboost" in self.active_model and self.h2o_available:
                fallbacks.append(("h2o", self._predict_h2o))
            elif self.active_model == "h2o":
                if self.catboost_gpu_available:
                    fallbacks.append(("catboost_gpu", self._predict_catboost))
                else:
                    fallbacks.append(("catboost_cpu", self._predict_catboost))

            # Try each fallback
            for name, predictor in fallbacks:
                try:
                    logger.info(f"Attempting fallback prediction with {name}")
                    result = predictor(X)

                    # Record performance metrics if enabled
                    if self.enable_performance_monitoring and start_time is not None:
                        predict_time = time.time() - start_time
                        self.performance_metrics["predict_time"] = predict_time
                        self.performance_metrics["predict_samples_per_second"] = (
                            n_samples / predict_time
                        )
                        logger.info(
                            f"Fallback prediction completed in {predict_time:.2f} seconds"
                        )

                    return result
                except Exception as e2:
                    logger.warning(
                        f"Fallback prediction with {name} also failed: {str(e2)}"
                    )

            # If all fallbacks fail, raise the original error
            raise RuntimeError(
                f"All prediction methods failed. Original error: {str(e)}"
            )

    def _predict_catboost_batched(
        self, X: np.ndarray, batch_size: int = 50000
    ) -> np.ndarray:
        """
        Generate probability predictions using CatBoost model in batches.

        Args:
            X: Feature matrix
            batch_size: Size of prediction batches

        Returns:
            np.ndarray: Probability predictions
        """
        # Check if model exists
        if self.catboost_model is None:
            raise RuntimeError("CatBoost model not trained")

        n_samples = X.shape[0]
        results = []

        # Process data in batches
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            logger.debug(f"Predicting batch {i // batch_size + 1} ({i}:{end})")

            # Create batch test pool
            batch_pool = Pool(X[i:end])

            # Make predictions on batch
            batch_preds = self.catboost_model.predict_proba(batch_pool)

            # Append predictions
            results.append(batch_preds[:, 1])

        # Combine batches and reshape
        return np.concatenate(results).reshape(-1, 1)

    def _predict_h2o(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions using H2O model with hardware optimizations.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Probability predictions
        """
        # Check if model exists
        if self.h2o_model is None:
            raise RuntimeError("H2O model not trained")

        n_samples = X.shape[0]

        # For very large datasets, use batched prediction
        if n_samples > 100000:
            logger.info(
                f"Using batched H2O prediction for large dataset ({n_samples} samples)"
            )
            return self._predict_h2o_batched(X, batch_size=100000)

        # Convert to H2O frame
        h2o_frame = h2o.H2OFrame(X)
        h2o_frame.columns = [f"feat_{i}" for i in range(X.shape[1])]

        # Make predictions
        preds = self.h2o_model.predict(h2o_frame)

        # Extract the probabilities column and convert to numpy
        return preds["p1"].as_data_frame().values

    def _predict_h2o_batched(
        self, X: np.ndarray, batch_size: int = 100000
    ) -> np.ndarray:
        """
        Generate probability predictions using H2O model in batches.

        Args:
            X: Feature matrix
            batch_size: Size of prediction batches

        Returns:
            np.ndarray: Probability predictions
        """
        # Check if model exists
        if self.h2o_model is None:
            raise RuntimeError("H2O model not trained")

        n_samples = X.shape[0]
        results = []

        # Process data in batches
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            logger.debug(f"Predicting H2O batch {i // batch_size + 1} ({i}:{end})")

            # Convert batch to H2O frame
            batch_frame = h2o.H2OFrame(X[i:end])
            batch_frame.columns = [f"feat_{i}" for i in range(X.shape[1])]

            # Make predictions on batch
            batch_preds = self.h2o_model.predict(batch_frame)

            # Extract probabilities and convert to numpy
            batch_np = batch_preds["p1"].as_data_frame().values

            # Append predictions
            results.append(batch_np)

        # Combine batches
        return np.concatenate(results)

    def _predict_catboost(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions using CatBoost model with hardware optimizations.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Probability predictions
        """
        # Check if model exists
        if self.catboost_model is None:
            raise RuntimeError("CatBoost model not trained")

        # Determine optimal thread count for prediction based on hardware
        thread_count = -1  # Default to all cores

        # Adjust thread count for AMD GPUs to avoid conflicts
        if self.gpu_info.gpu_type == "amd" and self.gpu_info.is_rx6800xt:
            # Use slightly fewer threads to avoid contention with GPU
            thread_count = max(os.cpu_count() - 2, 4)

        # Create test pool for faster prediction with optimal thread count
        test_pool = Pool(X, thread_count=thread_count)

        # Make predictions
        preds = self.catboost_model.predict_proba(test_pool)

        # Return only the probability of the positive class
        return preds[:, 1].reshape(-1, 1)

    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        """
        Generate class predictions with hardware optimizations.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Class predictions (0 or 1)
        """
        # For AMD RX6800XT, we can use a more efficient threshold calculation
        # that avoids an extra array allocation
        if self.gpu_info.is_rx6800xt and "catboost" in self.active_model:
            try:
                # Direct prediction with threshold for better memory efficiency
                if not isinstance(X, np.ndarray):
                    X = np.array(X)

                # For large datasets, use batched prediction
                n_samples = X.shape[0]
                if n_samples > 50000:
                    # Use efficient thread count for AMD GPUs
                    thread_count = max(os.cpu_count() - 2, 4)

                    # Process in batches
                    batch_size = 50000
                    results = []

                    for i in range(0, n_samples, batch_size):
                        end = min(i + batch_size, n_samples)
                        batch_pool = Pool(X[i:end], thread_count=thread_count)

                        # Direct class prediction (more efficient)
                        batch_preds = self.catboost_model.predict(
                            batch_pool, prediction_type="Class"
                        )

                        results.append(batch_preds)

                    return np.concatenate(results).reshape(-1, 1)
                else:
                    # Standard prediction for smaller datasets
                    test_pool = Pool(X)
                    return self.catboost_model.predict(
                        test_pool, prediction_type="Class"
                    ).reshape(-1, 1)
            except Exception as e:
                logger.warning(
                    f"Direct class prediction failed: {str(e)}, falling back to standard method"
                )

        # Standard method for all other cases
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from the active model with hardware-aware optimizations.

        Returns:
            Dict: Feature importance values
        """
        importances = {}

        # For AMD RX6800XT, optimize feature importance calculation
        if self.gpu_info.is_rx6800xt and "catboost" in self.active_model:
            try:
                # Get CatBoost feature importance with specialized parameters
                importances["catboost"] = self.catboost_model.get_feature_importance(
                    type="FeatureImportance",  # More efficient for RDNA2 architecture
                    thread_count=max(
                        os.cpu_count() - 2, 4
                    ),  # Avoid contention with GPU
                )

                # If we have a backup H2O model, get that too
                if self.h2o_model is not None:
                    try:
                        imp_frame = self.h2o_model.varimp(use_pandas=True)
                        importances["h2o"] = np.array(imp_frame["percentage"].values)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get H2O feature importance: {str(e)}"
                        )

                return importances
            except Exception as e:
                logger.warning(
                    f"RX6800XT optimized feature importance failed: {str(e)}, using standard method"
                )

        # Standard approach for other hardware
        try:
            if self.active_model == "h2o" and self.h2o_model is not None:
                # Get H2O feature importance
                imp_frame = self.h2o_model.varimp(use_pandas=True)
                importances["h2o"] = np.array(imp_frame["percentage"].values)
        except Exception as e:
            logger.warning(f"Failed to get H2O feature importance: {str(e)}")

        try:
            if "catboost" in self.active_model and self.catboost_model is not None:
                # Get CatBoost feature importance
                if self.gpu_info.gpu_type == "nvidia" and self.gpu_info.vram_mb >= 8000:
                    # For high-end NVIDIA GPUs, use more detailed importance
                    importances["catboost"] = (
                        self.catboost_model.get_feature_importance(
                            type="PredictionValuesChange"  # More detailed but requires more VRAM
                        )
                    )
                else:
                    # Standard feature importance for other hardware
                    importances["catboost"] = (
                        self.catboost_model.get_feature_importance()
                    )
        except Exception as e:
            logger.warning(f"Failed to get CatBoost feature importance: {str(e)}")

        return importances

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the model.

        Returns:
            Dict: Performance metrics
        """
        return self.performance_metrics.copy()

    def optimize_for_inference(self) -> bool:
        """
        Optimize the model specifically for inference performance.

        Returns:
            bool: True if optimization was successful
        """
        success = False

        # Only applicable for CatBoost models
        if "catboost" not in self.active_model or self.catboost_model is None:
            logger.info("Optimization for inference only supported for CatBoost models")
            return False

        try:
            # AMD RX6800XT specific optimization
            if self.gpu_info.is_rx6800xt:
                logger.info("Optimizing CatBoost model for inference on RX6800XT")

                # Save model to a temporary file
                temp_model_path = "temp_model.cbm"
                self.catboost_model.save_model(temp_model_path)

                # Load with inference optimized settings
                inference_params = {
                    "task_type": "GPU",
                    "devices": f"{self.catboost_gpu_id}",
                    "gpu_ram_part": 0.95,  # Use more memory for inference
                    "thread_count": max(os.cpu_count() - 2, 4),  # Optimal for AMD
                    "used_ram_limit": "8gb",  # Memory limit for optimal performance
                }

                # Create a new model with inference parameters
                inference_model = CatBoostClassifier(**inference_params)
                inference_model.load_model(temp_model_path)

                # Replace the existing model
                self.catboost_model = inference_model

                # Clean up the temporary file
                try:
                    os.remove(temp_model_path)
                except:
                    pass

                success = True
            elif self.gpu_info.gpu_type == "nvidia" and self.gpu_info.cuda_available:
                # NVIDIA-specific optimization
                logger.info(
                    f"Optimizing CatBoost model for inference on NVIDIA {self.gpu_info.gpu_model}"
                )

                # Save model to a temporary file
                temp_model_path = "temp_model.cbm"
                self.catboost_model.save_model(temp_model_path)

                # Calculate optimal thread count based on CPU cores and CUDA cores
                optimal_threads = min(os.cpu_count(), 8)  # Cap at 8 threads

                # Load with inference optimized settings
                inference_params = {
                    "task_type": "GPU",
                    "devices": f"{self.catboost_gpu_id}",
                    "thread_count": optimal_threads,
                    "gpu_ram_part": 0.9,
                }

                # Create a new model with inference parameters
                inference_model = CatBoostClassifier(**inference_params)
                inference_model.load_model(temp_model_path)

                # Replace the existing model
                self.catboost_model = inference_model

                # Clean up the temporary file
                try:
                    os.remove(temp_model_path)
                except:
                    pass

                success = True
            else:
                # CPU optimization
                logger.info("Optimizing CatBoost model for CPU inference")

                # For CPU, optimize thread count
                optimal_threads = max(os.cpu_count() - 1, 1)
                self.catboost_model.set_params(thread_count=optimal_threads)

                success = True

        except Exception as e:
            logger.error(f"Failed to optimize model for inference: {str(e)}")
            success = False

        return success

    def save_models(self, directory: str) -> Dict[str, str]:
        """
        Save trained models to disk with hardware-specific optimizations.

        Args:
            directory: Directory to save models

        Returns:
            Dict: Paths to saved models
        """
        os.makedirs(directory, exist_ok=True)
        saved_paths = {}

        # Save hardware configuration info
        try:
            hardware_info = {
                "gpu_type": self.gpu_info.gpu_type,
                "gpu_model": self.gpu_info.gpu_model,
                "is_rx6800xt": self.gpu_info.is_rx6800xt,
                "vram_mb": self.gpu_info.vram_mb,
                "cuda_available": self.gpu_info.cuda_available,
                "cuda_version": self.gpu_info.cuda_version,
                "rocm_available": self.gpu_info.rocm_available,
                "compute_units": self.gpu_info.compute_units,
            }

            # Save hardware info to JSON
            hardware_path = os.path.join(directory, "hardware_info.json")
            with open(hardware_path, "w") as f:
                json.dump(hardware_info, f, indent=2)

            saved_paths["hardware_info"] = hardware_path
            logger.info(f"Hardware info saved to {hardware_path}")
        except Exception as e:
            logger.warning(f"Failed to save hardware info: {str(e)}")

        # Save performance metrics
        if self.enable_performance_monitoring:
            try:
                metrics_path = os.path.join(directory, "performance_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(self.performance_metrics, f, indent=2)

                saved_paths["performance_metrics"] = metrics_path
                logger.info(f"Performance metrics saved to {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to save performance metrics: {str(e)}")

        # Save H2O model if available
        if self.h2o_model is not None:
            try:
                h2o_path = os.path.join(directory, "h2o_model")
                model_path = h2o.save_model(
                    model=self.h2o_model, path=h2o_path, force=True
                )
                saved_paths["h2o"] = model_path
                logger.info(f"H2O model saved to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save H2O model: {str(e)}")

        # Save CatBoost model if available
        if self.catboost_model is not None:
            try:
                # For RX6800XT, use specialized save format
                if self.gpu_info.is_rx6800xt:
                    # Save in both binary and JSON format for maximum compatibility
                    catboost_bin_path = os.path.join(directory, "catboost_model.cbm")
                    self.catboost_model.save_model(catboost_bin_path)
                    saved_paths["catboost"] = catboost_bin_path

                    # Also save in JSON format for detailed inspection
                    catboost_json_path = os.path.join(directory, "catboost_model.json")
                    self.catboost_model.save_model(
                        catboost_json_path, format="json", pool=None
                    )
                    saved_paths["catboost_json"] = catboost_json_path

                    logger.info(f"CatBoost model saved in both binary and JSON formats")
                else:
                    # Standard save for other hardware
                    catboost_path = os.path.join(directory, "catboost_model.cbm")
                    self.catboost_model.save_model(catboost_path)
                    saved_paths["catboost"] = catboost_path
                    logger.info(f"CatBoost model saved to {catboost_path}")
            except Exception as e:
                logger.error(f"Failed to save CatBoost model: {str(e)}")

        return saved_paths

    def load_models(
        self,
        h2o_path: Optional[str] = None,
        catboost_path: Optional[str] = None,
        hardware_info_path: Optional[str] = None,
    ) -> None:
        """
        Load trained models from disk with hardware-specific optimizations.

        Args:
            h2o_path: Path to H2O model
            catboost_path: Path to CatBoost model
            hardware_info_path: Path to hardware info JSON file
        """
        # Check if we have hardware info
        saved_hardware_info = None
        if hardware_info_path is not None and os.path.exists(hardware_info_path):
            try:
                with open(hardware_info_path, "r") as f:
                    saved_hardware_info = json.load(f)
                logger.info(f"Loaded hardware info from {hardware_info_path}")

                # Check if we're loading on the same hardware type
                if (
                    saved_hardware_info.get("is_rx6800xt", False)
                    and not self.gpu_info.is_rx6800xt
                ):
                    logger.warning(
                        "Model was trained on RX6800XT but is being loaded on different hardware"
                    )
                elif saved_hardware_info.get("gpu_type") != self.gpu_info.gpu_type:
                    logger.warning(
                        f"Model was trained on {saved_hardware_info.get('gpu_type')} GPU but is being loaded on {self.gpu_info.gpu_type}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load hardware info: {str(e)}")

        # Load H2O model if path provided
        if h2o_path is not None:
            try:
                self.h2o_model = h2o.load_model(h2o_path)
                logger.info(f"H2O model loaded from {h2o_path}")
                if self.active_model is None:
                    self.active_model = "h2o"
            except Exception as e:
                logger.error(f"Failed to load H2O model: {str(e)}")

        # Load CatBoost model if path provided
        if catboost_path is not None:
            try:
                # If we're on RX6800XT, initialize with optimal parameters
                if self.gpu_info.is_rx6800xt:
                    # Get optimal parameters for the hardware
                    gpu_params = self.gpu_info.get_optimal_catboost_params()

                    # Create model with RX6800XT-optimized parameters
                    self.catboost_model = CatBoostClassifier(**gpu_params)
                    logger.info(
                        "Initializing CatBoost with RX6800XT-optimized parameters"
                    )
                elif (
                    self.gpu_info.gpu_type == "nvidia" and self.gpu_info.cuda_available
                ):
                    # Get optimal parameters for NVIDIA
                    gpu_params = self.gpu_info.get_optimal_catboost_params()

                    # Create model with NVIDIA-optimized parameters
                    self.catboost_model = CatBoostClassifier(**gpu_params)
                    logger.info(
                        "Initializing CatBoost with NVIDIA-optimized parameters"
                    )
                else:
                    # Initialize with default parameters for CPU
                    self.catboost_model = CatBoostClassifier(thread_count=-1)

                # Load the saved model
                self.catboost_model.load_model(catboost_path)
                logger.info(f"CatBoost model loaded from {catboost_path}")

                # Set the active model based on hardware
                if self.gpu_info.has_gpu:
                    self.active_model = "catboost_gpu"
                else:
                    self.active_model = "catboost_cpu"
            except Exception as e:
                logger.error(f"Failed to load CatBoost model: {str(e)}")

                # Try fallback approach
                try:
                    logger.info("Attempting fallback loading method")
                    self.catboost_model = CatBoostClassifier()
                    self.catboost_model.load_model(catboost_path)
                    logger.info("CatBoost model loaded with fallback method")
                    self.active_model = "catboost_cpu"  # Default to CPU for fallback
                except Exception as e2:
                    logger.error(f"Fallback loading also failed: {str(e2)}")

        # Configure for optimal inference
        if self.catboost_model is not None and "catboost" in self.active_model:
            self.optimize_for_inference()

    def shutdown(self):
        """
        Clean up resources, especially H2O cluster and GPU resources.
        """
        # Log performance metrics before shutdown if enabled
        if self.enable_performance_monitoring:
            logger.info("Final performance metrics:")
            for key, value in self.performance_metrics.items():
                logger.info(f"  {key}: {value}")

        # Special cleanup for AMD GPUs
        if self.gpu_info.gpu_type == "amd" and self.gpu_info.is_rx6800xt:
            logger.info("Performing RX6800XT-specific cleanup")
            # Free model memory explicitly
            if self.catboost_model is not None:
                self.catboost_model = None

            # Force garbage collection
            import gc

            gc.collect()

            # Wait a moment for resources to be released
            import time

            time.sleep(0.5)

        # Shutdown H2O cluster if available
        if self.h2o_available:
            try:
                h2o.cluster().shutdown()
                logger.info("H2O cluster shut down")
            except Exception as e:
                logger.warning(f"Error shutting down H2O cluster: {str(e)}")

        logger.info("H2O-CatBoost hybrid model shutdown complete")

class MLPerformanceTracker:
    """
    Memory-efficient ML performance tracking with circular buffer
    and hardware-optimized storage
    """

    def __init__(self, max_entries=1000, hardware_manager=None):
        self.hardware_manager = hardware_manager or HardwareManager()

        # Configure buffer size based on available RAM
        if self.hardware_manager.system_ram_capacity >= 64:  # High RAM
            self.max_entries = 2000
        else:  # Test unit
            self.max_entries = 1000

        # Override with provided value if specified
        if max_entries is not None:
            self.max_entries = max_entries

        # Initialize circular buffer
        from collections import deque

        self.buffer = deque(maxlen=self.max_entries)

        # Thread safety
        import threading

        self._lock = threading.RLock()

        # Track aggregate statistics to avoid recalculating
        self.stats = {
            "total_correct": 0,
            "total_entries": 0,
            "total_profit": 0.0,
            "peak_memory_mb": 0,
        }

        logger.info(
            f"ML performance tracker initialized with {self.max_entries} max entries"
        )

    def add_entry(self, entry):
        """Add performance entry with memory monitoring"""
        with self._lock:
            # Handle buffer rotation
            if len(self.buffer) >= self.max_entries:
                # Update stats for item being removed
                old_item = self.buffer[0]
                if old_item.get("prediction_correct", False):
                    self.stats["total_correct"] -= 1
                self.stats["total_profit"] -= old_item.get("subsequent_return", 0)
                self.stats["total_entries"] -= 1

            # Add new entry
            self.buffer.append(entry)

            # Update stats
            if entry.get("prediction_correct", False):
                self.stats["total_correct"] += 1
            self.stats["total_profit"] += entry.get("subsequent_return", 0)
            self.stats["total_entries"] += 1

            # Check memory usage occasionally
            if self.stats["total_entries"] % 100 == 0:
                self._check_memory_usage()

    def _check_memory_usage(self):
        """Monitor memory usage of the buffer"""
        try:
            import sys

            # Estimate memory usage (rough approximation)
            sample_size = min(10, len(self.buffer))
            if sample_size > 0:
                sample_items = list(self.buffer)[-sample_size:]
                avg_item_size_bytes = (
                    sum(sys.getsizeof(item) for item in sample_items) / sample_size
                )
                total_memory_mb = (avg_item_size_bytes * len(self.buffer)) / (
                    1024 * 1024
                )

                # Update peak memory usage
                self.stats["peak_memory_mb"] = max(
                    self.stats["peak_memory_mb"], total_memory_mb
                )

                # If buffer is taking too much memory, reduce size
                if total_memory_mb > 500:  # 500MB limit
                    logger.warning(
                        f"ML performance buffer using high memory: {total_memory_mb:.1f}MB"
                    )
                    # Reduce to 75% of current size
                    new_max_entries = int(self.max_entries * 0.75)
                    self._resize_buffer(new_max_entries)
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def _resize_buffer(self, new_size):
        """Resize the buffer while maintaining stats"""
        with self._lock:
            if new_size >= self.max_entries:
                return

            # Create new buffer with smaller size
            from collections import deque

            new_buffer = deque(maxlen=new_size)

            # Keep the most recent entries
            items_to_keep = list(self.buffer)[-new_size:]
            new_buffer.extend(items_to_keep)

            # Recalculate stats
            self.stats["total_correct"] = sum(
                1 for item in new_buffer if item.get("prediction_correct", False)
            )
            self.stats["total_profit"] = sum(
                item.get("subsequent_return", 0) for item in new_buffer
            )
            self.stats["total_entries"] = len(new_buffer)

            # Update buffer and max size
            self.buffer = new_buffer
            self.max_entries = new_size

            logger.info(f"Resized ML performance buffer to {new_size} entries")

    def get_accuracy(self):
        """Get current prediction accuracy"""
        with self._lock:
            if self.stats["total_entries"] == 0:
                return 0.0
            return self.stats["total_correct"] / self.stats["total_entries"]

    def get_metrics(self):
        """Get comprehensive performance metrics"""
        with self._lock:
            if not self.buffer:
                return {
                    "accuracy": 0.0,
                    "sample_count": 0,
                    "avg_return": 0.0,
                    "memory_usage_mb": 0.0,
                }

            # Calculate metrics
            accuracy = self.get_accuracy()
            avg_return = self.stats["total_profit"] / self.stats["total_entries"]

            # Calculate additional metrics
            high_conf_returns = []
            corr_values = []

            # Last 500 entries for correlation calculation
            recent_entries = list(self.buffer)[-500:]

            if recent_entries:
                # Extract confidence values and returns
                ml_values = [
                    entry.get("ml_confidence", 0.5) for entry in recent_entries
                ]
                returns = [
                    entry.get("subsequent_return", 0.0) for entry in recent_entries
                ]

                # Filter high confidence predictions
                high_conf_entries = [
                    entry
                    for entry in recent_entries
                    if abs(entry.get("ml_confidence", 0.5) - 0.5) > 0.2
                ]

                if high_conf_entries:
                    high_conf_returns = [
                        entry.get("subsequent_return", 0.0)
                        for entry in high_conf_entries
                    ]

                # Calculate correlation if enough data
                try:
                    import numpy as np

                    if len(ml_values) > 2:
                        # Adjust confidence values to be centered around 0
                        adjusted_conf = [c - 0.5 for c in ml_values]
                        correlation = np.corrcoef(adjusted_conf, returns)[0, 1]
                        corr_values = [correlation]
                except Exception:
                    corr_values = [0]

            return {
                "accuracy": accuracy,
                "sample_count": self.stats["total_entries"],
                "avg_return": avg_return,
                "high_confidence_avg_return": sum(high_conf_returns)
                / len(high_conf_returns)
                if high_conf_returns
                else 0,
                "correlation": corr_values[0] if corr_values else 0,
                "memory_usage_mb": self.stats["peak_memory_mb"],
            }

    def cleanup(self):
        """Release resources"""
        with self._lock:
            self.buffer.clear()
            self.stats = {
                "total_correct": 0,
                "total_entries": 0,
                "total_profit": 0.0,
                "peak_memory_mb": 0,
            }
