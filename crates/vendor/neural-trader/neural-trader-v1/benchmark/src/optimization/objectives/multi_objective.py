"""Multi-objective optimization functions and utilities."""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating multiple objectives."""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_PRODUCT = "weighted_product"
    SCALARIZATION = "scalarization"
    GOAL_PROGRAMMING = "goal_programming"


@dataclass
class Objective:
    """Single objective definition."""
    name: str
    function: Callable
    weight: float = 1.0
    minimize: bool = False
    target: Optional[float] = None
    

@dataclass
class ParetoSolution:
    """Solution on Pareto front."""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    fitness: float
    rank: int = 0
    crowding_distance: float = 0.0


class MultiObjectiveFunction:
    """Multi-objective function combining multiple objectives."""
    
    def __init__(
        self,
        objectives: List[Objective],
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM,
        normalize: bool = True
    ):
        """Initialize multi-objective function.
        
        Args:
            objectives: List of objective definitions
            aggregation_method: Method to combine objectives
            normalize: Whether to normalize objective values
        """
        self.objectives = objectives
        self.aggregation_method = aggregation_method
        self.normalize = normalize
        
        # Normalization statistics
        self.objective_stats = {}
        self.evaluation_history = []
        
    def __call__(self, params: Dict[str, Any]) -> float:
        """Evaluate multi-objective function."""
        try:
            # Evaluate all objectives
            objective_values = {}
            
            for obj in self.objectives:
                value = obj.function(params)
                
                # Convert to maximization
                if obj.minimize:
                    value = -value
                    
                objective_values[obj.name] = value
                
            # Store for normalization
            self.evaluation_history.append(objective_values)
            
            # Normalize if enabled
            if self.normalize and len(self.evaluation_history) > 1:
                objective_values = self._normalize_objectives(objective_values)
                
            # Aggregate objectives
            aggregated_value = self._aggregate_objectives(objective_values)
            
            return aggregated_value
            
        except Exception as e:
            logger.error(f"Error in multi-objective evaluation: {str(e)}")
            return float('-inf')
            
    def _normalize_objectives(self, objective_values: Dict[str, float]) -> Dict[str, float]:
        """Normalize objective values."""
        normalized = {}
        
        # Update statistics
        for obj_name, value in objective_values.items():
            if obj_name not in self.objective_stats:
                self.objective_stats[obj_name] = {'min': value, 'max': value, 'values': []}
                
            stats = self.objective_stats[obj_name]
            stats['values'].append(value)
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            
            # Normalize to [0, 1]
            if stats['max'] > stats['min']:
                normalized[obj_name] = (value - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized[obj_name] = 0.0
                
        return normalized
        
    def _aggregate_objectives(self, objective_values: Dict[str, float]) -> float:
        """Aggregate multiple objectives into single value."""
        if self.aggregation_method == AggregationMethod.WEIGHTED_SUM:
            return self._weighted_sum(objective_values)
        elif self.aggregation_method == AggregationMethod.WEIGHTED_PRODUCT:
            return self._weighted_product(objective_values)
        elif self.aggregation_method == AggregationMethod.SCALARIZATION:
            return self._scalarization(objective_values)
        elif self.aggregation_method == AggregationMethod.GOAL_PROGRAMMING:
            return self._goal_programming(objective_values)
        else:
            return self._weighted_sum(objective_values)
            
    def _weighted_sum(self, objective_values: Dict[str, float]) -> float:
        """Weighted sum aggregation."""
        total = 0.0
        total_weight = 0.0
        
        for obj in self.objectives:
            if obj.name in objective_values:
                total += obj.weight * objective_values[obj.name]
                total_weight += obj.weight
                
        return total / total_weight if total_weight > 0 else 0.0
        
    def _weighted_product(self, objective_values: Dict[str, float]) -> float:
        """Weighted product aggregation."""
        product = 1.0
        
        for obj in self.objectives:
            if obj.name in objective_values:
                value = objective_values[obj.name]
                # Ensure positive values for product
                if value <= 0:
                    value = 1e-6
                product *= (value ** obj.weight)
                
        return product
        
    def _scalarization(self, objective_values: Dict[str, float]) -> float:
        """Tchebycheff scalarization."""
        max_deviation = 0.0
        
        for obj in self.objectives:
            if obj.name in objective_values:
                # Use target if specified, otherwise use maximum observed
                if obj.target is not None:
                    target = obj.target
                else:
                    stats = self.objective_stats.get(obj.name, {'max': 1.0})
                    target = stats['max']
                    
                deviation = obj.weight * abs(objective_values[obj.name] - target)
                max_deviation = max(max_deviation, deviation)
                
        return -max_deviation  # Minimize maximum deviation
        
    def _goal_programming(self, objective_values: Dict[str, float]) -> float:
        """Goal programming aggregation."""
        total_deviation = 0.0
        
        for obj in self.objectives:
            if obj.name in objective_values and obj.target is not None:
                value = objective_values[obj.name]
                deviation = max(0, obj.target - value)  # Only penalize underachievement
                total_deviation += obj.weight * (deviation ** 2)
                
        return -total_deviation  # Minimize total deviation
        
    def evaluate_pareto(self, param_sets: List[Dict[str, Any]]) -> List[ParetoSolution]:
        """Evaluate Pareto front from parameter sets."""
        solutions = []
        
        for params in param_sets:
            # Evaluate all objectives separately
            objective_values = {}
            
            for obj in self.objectives:
                value = obj.function(params)
                objective_values[obj.name] = value
                
            # Calculate fitness (aggregated value)
            fitness = self(params)
            
            solution = ParetoSolution(
                parameters=params,
                objectives=objective_values,
                fitness=fitness
            )
            solutions.append(solution)
            
        return solutions


class ParetoDominance:
    """Utilities for Pareto dominance operations."""
    
    @staticmethod
    def dominates(solution1: ParetoSolution, solution2: ParetoSolution, minimize_objectives: List[str] = None) -> bool:
        """Check if solution1 dominates solution2."""
        if minimize_objectives is None:
            minimize_objectives = []
            
        better_in_any = False
        
        for obj_name in solution1.objectives:
            if obj_name not in solution2.objectives:
                continue
                
            val1 = solution1.objectives[obj_name]
            val2 = solution2.objectives[obj_name]
            
            if obj_name in minimize_objectives:
                # For minimization objectives
                if val1 > val2:  # Worse
                    return False
                elif val1 < val2:  # Better
                    better_in_any = True
            else:
                # For maximization objectives
                if val1 < val2:  # Worse
                    return False
                elif val1 > val2:  # Better
                    better_in_any = True
                    
        return better_in_any
        
    @staticmethod
    def get_pareto_front(solutions: List[ParetoSolution], minimize_objectives: List[str] = None) -> List[ParetoSolution]:
        """Extract Pareto front from solutions."""
        if minimize_objectives is None:
            minimize_objectives = []
            
        pareto_front = []
        
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if ParetoDominance.dominates(other, candidate, minimize_objectives):
                    is_dominated = True
                    break
                    
            if not is_dominated:
                pareto_front.append(candidate)
                
        return pareto_front
        
    @staticmethod
    def non_dominated_sort(solutions: List[ParetoSolution], minimize_objectives: List[str] = None) -> List[List[ParetoSolution]]:
        """Perform non-dominated sorting."""
        if minimize_objectives is None:
            minimize_objectives = []
            
        fronts = []
        n = len(solutions)
        
        # Initialize domination count and dominated solutions
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        
        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if ParetoDominance.dominates(solutions[i], solutions[j], minimize_objectives):
                        dominated_solutions[i].append(j)
                    elif ParetoDominance.dominates(solutions[j], solutions[i], minimize_objectives):
                        domination_count[i] += 1
                        
        # Find first front
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                solutions[i].rank = 0
                current_front.append(i)
                
        fronts.append([solutions[i] for i in current_front])
        
        # Find subsequent fronts
        front_number = 0
        while current_front:
            next_front = []
            
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        solutions[j].rank = front_number + 1
                        next_front.append(j)
                        
            front_number += 1
            current_front = next_front
            
            if current_front:
                fronts.append([solutions[i] for i in current_front])
                
        return fronts
        
    @staticmethod
    def calculate_crowding_distance(front: List[ParetoSolution]) -> None:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
            
        # Initialize distances
        for solution in front:
            solution.crowding_distance = 0.0
            
        # Get all objectives
        objectives = list(front[0].objectives.keys())
        
        for obj_name in objectives:
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_name])
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_range = front[-1].objectives[obj_name] - front[0].objectives[obj_name]
            
            if obj_range == 0:
                continue
                
            # Calculate crowding distance
            for i in range(1, len(front) - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (front[i+1].objectives[obj_name] - front[i-1].objectives[obj_name]) / obj_range
                    front[i].crowding_distance += distance


class HyperVolumeIndicator:
    """Hypervolume indicator for multi-objective optimization quality."""
    
    def __init__(self, reference_point: Dict[str, float]):
        """Initialize hypervolume calculator.
        
        Args:
            reference_point: Reference point for hypervolume calculation
        """
        self.reference_point = reference_point
        
    def calculate(self, pareto_front: List[ParetoSolution]) -> float:
        """Calculate hypervolume of Pareto front."""
        if not pareto_front:
            return 0.0
            
        # Simple hypervolume calculation (for 2D case)
        if len(self.reference_point) == 2:
            return self._calculate_2d_hypervolume(pareto_front)
        else:
            # For higher dimensions, use approximation
            return self._calculate_approximate_hypervolume(pareto_front)
            
    def _calculate_2d_hypervolume(self, pareto_front: List[ParetoSolution]) -> float:
        """Calculate 2D hypervolume."""
        obj_names = list(self.reference_point.keys())
        
        # Extract points
        points = []
        for solution in pareto_front:
            point = [solution.objectives[obj] for obj in obj_names]
            points.append(point)
            
        if not points:
            return 0.0
            
        # Sort by first objective
        points.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate hypervolume
        hypervolume = 0.0
        ref_point = [self.reference_point[obj] for obj in obj_names]
        
        prev_y = ref_point[1]
        
        for point in points:
            if point[1] > prev_y:
                width = point[0] - ref_point[0]
                height = point[1] - prev_y
                hypervolume += width * height
                prev_y = point[1]
                
        return hypervolume
        
    def _calculate_approximate_hypervolume(self, pareto_front: List[ParetoSolution]) -> float:
        """Approximate hypervolume for higher dimensions."""
        # Simple approximation: product of objective ranges
        if not pareto_front:
            return 0.0
            
        hypervolume = 1.0
        
        for obj_name in self.reference_point:
            obj_values = [sol.objectives.get(obj_name, 0) for sol in pareto_front]
            
            if obj_values:
                max_val = max(obj_values)
                ref_val = self.reference_point[obj_name]
                
                if max_val > ref_val:
                    hypervolume *= (max_val - ref_val)
                else:
                    hypervolume = 0.0
                    break
                    
        return hypervolume