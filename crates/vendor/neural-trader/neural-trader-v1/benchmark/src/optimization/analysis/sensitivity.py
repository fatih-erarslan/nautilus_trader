"""Parameter sensitivity analysis for optimization results."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""
    parameter_importance: Dict[str, float]
    sensitivity_scores: Dict[str, float]
    perturbation_results: pd.DataFrame
    stability_metrics: Dict[str, Any]


class SensitivityAnalyzer:
    """Analyze parameter sensitivity and importance."""
    
    def __init__(self, objective_function: Callable):
        """Initialize sensitivity analyzer."""
        self.objective_function = objective_function
        
    def analyze(
        self,
        best_params: Dict[str, Any],
        parameter_space: Dict[str, Any],
        n_perturbations: int = 100,
        perturbation_size: float = 0.1
    ) -> SensitivityResult:
        """Perform sensitivity analysis."""
        logger.info("Starting parameter sensitivity analysis")
        
        # Generate perturbations
        perturbation_results = []
        
        for param_name in best_params:
            param_sensitivities = []
            
            for _ in range(n_perturbations):
                # Create perturbed parameters
                perturbed_params = best_params.copy()
                
                # Perturb single parameter
                original_value = best_params[param_name]
                
                if isinstance(original_value, (int, float)):
                    # Numeric parameter
                    noise = np.random.normal(0, perturbation_size * abs(original_value))
                    perturbed_params[param_name] = original_value + noise
                    
                    # Bound check
                    if param_name in parameter_space:
                        param_def = parameter_space[param_name]
                        if isinstance(param_def, tuple):
                            low, high = param_def
                            perturbed_params[param_name] = np.clip(
                                perturbed_params[param_name], low, high
                            )
                
                # Evaluate perturbed parameters
                try:
                    score = self.objective_function(perturbed_params)
                    param_sensitivities.append(score)
                    
                    perturbation_results.append({
                        'parameter': param_name,
                        'original_value': original_value,
                        'perturbed_value': perturbed_params[param_name],
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error in perturbation: {str(e)}")
                    
        # Calculate sensitivity metrics
        results_df = pd.DataFrame(perturbation_results)
        
        parameter_importance = {}
        sensitivity_scores = {}
        
        for param_name in best_params:
            param_data = results_df[results_df['parameter'] == param_name]
            
            if len(param_data) > 0:
                # Variance in performance
                score_variance = np.var(param_data['score'])
                parameter_importance[param_name] = score_variance
                
                # Correlation between parameter change and score
                if len(param_data) > 1:
                    correlation = np.corrcoef(
                        param_data['perturbed_value'],
                        param_data['score']
                    )[0, 1]
                    sensitivity_scores[param_name] = abs(correlation)
                else:
                    sensitivity_scores[param_name] = 0.0
                    
        # Normalize importance scores
        total_importance = sum(parameter_importance.values())
        if total_importance > 0:
            parameter_importance = {
                k: v / total_importance for k, v in parameter_importance.items()
            }
            
        stability_metrics = {
            'total_perturbations': len(perturbation_results),
            'avg_score_variance': np.mean(list(parameter_importance.values())),
            'max_sensitivity': max(sensitivity_scores.values()) if sensitivity_scores else 0.0
        }
        
        return SensitivityResult(
            parameter_importance=parameter_importance,
            sensitivity_scores=sensitivity_scores,
            perturbation_results=results_df,
            stability_metrics=stability_metrics
        )