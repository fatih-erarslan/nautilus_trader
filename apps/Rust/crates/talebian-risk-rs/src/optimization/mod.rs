//! Optimization algorithms for Talebian risk management

use crate::error::TalebianResult as Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization objective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Maximize antifragility
    MaximizeAntifragility,
    /// Minimize black swan probability
    MinimizeBlackSwanProbability,
    /// Maximize risk-adjusted return
    MaximizeRiskAdjustedReturn,
    /// Maximize convexity
    MaximizeConvexity,
    /// Minimize tail risk
    MinimizeTailRisk,
}

/// Optimization constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound  
    pub upper_bound: f64,
    /// Target value (for equality constraints)
    pub target: Option<f64>,
}

/// Constraint types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Weight constraint
    Weight,
    /// Risk constraint
    Risk,
    /// Return constraint
    Return,
    /// Sector constraint
    Sector,
    /// Turnover constraint
    Turnover,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimal weights
    pub weights: HashMap<String, f64>,
    /// Objective value achieved
    pub objective_value: f64,
    /// Constraints satisfied
    pub constraints_satisfied: bool,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Optimization metadata
    pub metadata: HashMap<String, f64>,
}

/// Portfolio optimizer
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    /// Optimization objective
    objective: OptimizationObjective,
    /// Constraints
    constraints: Vec<OptimizationConstraint>,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl PortfolioOptimizer {
    /// Create a new portfolio optimizer
    pub fn new(objective: OptimizationObjective) -> Self {
        Self {
            objective,
            constraints: Vec::new(),
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
    
    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: OptimizationConstraint) {
        self.constraints.push(constraint);
    }
    
    /// Set maximum iterations
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }
    
    /// Set convergence tolerance
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }
    
    /// Optimize portfolio weights
    pub fn optimize(&self, assets: &[String], initial_weights: Option<HashMap<String, f64>>) -> Result<OptimizationResult> {
        // Simplified optimization implementation
        // In practice, this would use advanced optimization algorithms
        
        let mut weights = initial_weights.unwrap_or_else(|| {
            let equal_weight = 1.0 / assets.len() as f64;
            assets.iter().map(|asset| (asset.clone(), equal_weight)).collect()
        });
        
        let mut iterations = 0;
        let mut converged = false;
        let mut objective_value = self.evaluate_objective(&weights)?;
        
        // Simple gradient descent-like optimization
        for _ in 0..self.max_iterations {
            iterations += 1;
            
            let mut improved = false;
            let step_size = 0.01;
            
            // Try small adjustments to each weight
            for asset in assets {
                let original_weight = weights[asset];
                
                // Try increasing weight
                weights.insert(asset.clone(), original_weight + step_size);
                self.normalize_weights(&mut weights);
                
                if self.check_constraints(&weights)? {
                    let new_objective = self.evaluate_objective(&weights)?;
                    
                    if self.is_better_objective(new_objective, objective_value) {
                        objective_value = new_objective;
                        improved = true;
                        continue;
                    }
                }
                
                // Try decreasing weight
                weights.insert(asset.clone(), original_weight - step_size);
                self.normalize_weights(&mut weights);
                
                if self.check_constraints(&weights)? {
                    let new_objective = self.evaluate_objective(&weights)?;
                    
                    if self.is_better_objective(new_objective, objective_value) {
                        objective_value = new_objective;
                        improved = true;
                        continue;
                    }
                }
                
                // Revert if no improvement
                weights.insert(asset.clone(), original_weight);
            }
            
            if !improved {
                converged = true;
                break;
            }
        }
        
        let constraints_satisfied = self.check_constraints(&weights)?;
        
        let mut metadata = HashMap::new();
        metadata.insert("final_objective".to_string(), objective_value);
        metadata.insert("iterations".to_string(), iterations as f64);
        
        Ok(OptimizationResult {
            weights,
            objective_value,
            constraints_satisfied,
            iterations,
            converged,
            metadata,
        })
    }
    
    /// Evaluate objective function
    fn evaluate_objective(&self, weights: &HashMap<String, f64>) -> Result<f64> {
        match self.objective {
            OptimizationObjective::MaximizeAntifragility => {
                // Simplified antifragility calculation
                let mut score = 0.0;
                for (asset, &weight) in weights {
                    // Assume assets with higher volatility have more antifragile potential
                    let antifragility_proxy = if asset.contains("OPTION") || asset.contains("CRYPTO") {
                        0.8
                    } else if asset.contains("GOLD") || asset.contains("REAL_ESTATE") {
                        0.6
                    } else if asset.contains("STOCK") {
                        0.3
                    } else {
                        0.1
                    };
                    score += weight * antifragility_proxy;
                }
                Ok(score)
            }
            OptimizationObjective::MinimizeBlackSwanProbability => {
                // Simplified black swan probability
                let mut probability = 0.0;
                for (asset, &weight) in weights {
                    let asset_probability = if asset.contains("CRYPTO") {
                        0.1
                    } else if asset.contains("STOCK") {
                        0.05
                    } else if asset.contains("BOND") {
                        0.01
                    } else {
                        0.02
                    };
                    probability += weight * asset_probability;
                }
                Ok(-probability) // Negative because we want to minimize
            }
            OptimizationObjective::MaximizeRiskAdjustedReturn => {
                // Simplified Sharpe ratio proxy
                let mut return_score = 0.0;
                let mut risk_score = 0.0;
                
                for (asset, &weight) in weights {
                    let (expected_return, volatility) = if asset.contains("CRYPTO") {
                        (0.15, 0.6)
                    } else if asset.contains("STOCK") {
                        (0.08, 0.2)
                    } else if asset.contains("BOND") {
                        (0.03, 0.05)
                    } else {
                        (0.05, 0.1)
                    };
                    
                    return_score += weight * expected_return;
                    risk_score += weight * volatility;
                }
                
                if risk_score > 0.0 {
                    Ok(return_score / risk_score)
                } else {
                    Ok(0.0)
                }
            }
            OptimizationObjective::MaximizeConvexity => {
                // Simplified convexity measure
                let mut convexity = 0.0;
                for (asset, &weight) in weights {
                    let asset_convexity = if asset.contains("OPTION") {
                        1.0
                    } else if asset.contains("CRYPTO") {
                        0.6
                    } else if asset.contains("GROWTH") {
                        0.4
                    } else {
                        0.1
                    };
                    convexity += weight * asset_convexity;
                }
                Ok(convexity)
            }
            OptimizationObjective::MinimizeTailRisk => {
                // Simplified tail risk measure
                let mut tail_risk = 0.0;
                for (asset, &weight) in weights {
                    let asset_tail_risk = if asset.contains("CRYPTO") {
                        0.8
                    } else if asset.contains("SMALL_CAP") {
                        0.6
                    } else if asset.contains("EMERGING") {
                        0.5
                    } else if asset.contains("BOND") {
                        0.1
                    } else {
                        0.3
                    };
                    tail_risk += weight * asset_tail_risk;
                }
                Ok(-tail_risk) // Negative because we want to minimize
            }
        }
    }
    
    /// Check if constraints are satisfied
    fn check_constraints(&self, weights: &HashMap<String, f64>) -> Result<bool> {
        for constraint in &self.constraints {
            match constraint.constraint_type {
                ConstraintType::Weight => {
                    for &weight in weights.values() {
                        if weight < constraint.lower_bound || weight > constraint.upper_bound {
                            return Ok(false);
                        }
                    }
                }
                ConstraintType::Risk => {
                    // Simplified risk constraint check
                    let total_risk: f64 = weights.values().sum();
                    if total_risk < constraint.lower_bound || total_risk > constraint.upper_bound {
                        return Ok(false);
                    }
                }
                _ => {
                    // Other constraint types not implemented in this example
                }
            }
        }
        Ok(true)
    }
    
    /// Check if new objective value is better than current
    fn is_better_objective(&self, new_value: f64, current_value: f64) -> bool {
        match self.objective {
            OptimizationObjective::MaximizeAntifragility |
            OptimizationObjective::MaximizeRiskAdjustedReturn |
            OptimizationObjective::MaximizeConvexity => new_value > current_value,
            OptimizationObjective::MinimizeBlackSwanProbability |
            OptimizationObjective::MinimizeTailRisk => new_value < current_value,
        }
    }
    
    /// Normalize weights to sum to 1.0
    fn normalize_weights(&self, weights: &mut HashMap<String, f64>) {
        let total: f64 = weights.values().sum();
        if total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer_creation() {
        let optimizer = PortfolioOptimizer::new(OptimizationObjective::MaximizeAntifragility);
        assert_eq!(optimizer.objective, OptimizationObjective::MaximizeAntifragility);
        assert_eq!(optimizer.max_iterations, 1000);
    }
    
    #[test]
    fn test_add_constraint() {
        let mut optimizer = PortfolioOptimizer::new(OptimizationObjective::MaximizeAntifragility);
        
        let constraint = OptimizationConstraint {
            constraint_type: ConstraintType::Weight,
            lower_bound: 0.0,
            upper_bound: 0.3,
            target: None,
        };
        
        optimizer.add_constraint(constraint);
        assert_eq!(optimizer.constraints.len(), 1);
    }
    
    #[test]
    fn test_optimize() {
        let mut optimizer = PortfolioOptimizer::new(OptimizationObjective::MaximizeAntifragility);
        
        // Add weight constraints
        optimizer.add_constraint(OptimizationConstraint {
            constraint_type: ConstraintType::Weight,
            lower_bound: 0.0,
            upper_bound: 0.5,
            target: None,
        });
        
        let assets = vec![
            "BONDS".to_string(),
            "STOCKS".to_string(),
            "CRYPTO".to_string(),
            "GOLD".to_string(),
        ];
        
        let result = optimizer.optimize(&assets, None).unwrap();
        
        assert_eq!(result.weights.len(), 4);
        assert!(result.objective_value >= 0.0);
        assert!(result.constraints_satisfied);
        
        // Check that weights sum to 1.0
        let total_weight: f64 = result.weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_different_objectives() {
        let objectives = vec![
            OptimizationObjective::MaximizeAntifragility,
            OptimizationObjective::MinimizeBlackSwanProbability,
            OptimizationObjective::MaximizeRiskAdjustedReturn,
            OptimizationObjective::MaximizeConvexity,
            OptimizationObjective::MinimizeTailRisk,
        ];
        
        let assets = vec!["BONDS".to_string(), "STOCKS".to_string()];
        
        for objective in objectives {
            let optimizer = PortfolioOptimizer::new(objective);
            let result = optimizer.optimize(&assets, None).unwrap();
            
            assert_eq!(result.weights.len(), 2);
            assert!(result.objective_value.is_finite());
        }
    }
}