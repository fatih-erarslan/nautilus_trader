//! Integration tests for prospect theory implementation

use prospect_theory_rs::*;
use prospect_theory_rs::probability_weighting::WeightingFunction;
use approx::assert_relative_eq;

#[test]
fn test_complete_prospect_theory_calculation() {
    // Create default prospect theory components
    let value_function = ValueFunction::default_kt();
    let probability_weighting = ProbabilityWeighting::default_tk();
    
    // Define a simple lottery: 50% chance of winning $100, 50% chance of losing $50
    let outcomes = vec![100.0, -50.0];
    let probabilities = vec![0.5, 0.5];
    
    // Calculate values
    let values = value_function.values(&outcomes).unwrap();
    assert_eq!(values.len(), 2);
    
    // Calculate decision weights
    let decision_weights = probability_weighting
        .decision_weights(&probabilities, &outcomes)
        .unwrap();
    assert_eq!(decision_weights.len(), 2);
    
    // Calculate prospect value
    let prospect_value: f64 = values
        .iter()
        .zip(decision_weights.iter())
        .map(|(&value, &weight)| value * weight)
        .sum();
    
    // The prospect value should be negative due to loss aversion
    assert!(prospect_value < 0.0);
}

#[test]
fn test_loss_aversion_demonstration() {
    let vf = ValueFunction::default_kt();
    
    // Equal magnitude gains and losses
    let gain_value = vf.value(100.0).unwrap();
    let loss_value = vf.value(-100.0).unwrap();
    
    // Loss should have greater absolute value due to loss aversion
    assert!(loss_value.abs() > gain_value);
    
    // Loss aversion ratio should be greater than 1
    let ratio = vf.loss_aversion_ratio(100.0, -100.0).unwrap();
    assert!(ratio > 1.0);
    assert!(ratio > 2.0); // Should be close to lambda parameter (2.25)
}

#[test]
fn test_probability_weighting_inverse_s_curve() {
    let pw = ProbabilityWeighting::default_tk();
    
    // Test overweighting of small probabilities
    let w_01 = pw.weight_gains(0.01).unwrap();
    assert!(w_01 > 0.01);
    
    // Test underweighting of medium probabilities
    let w_50 = pw.weight_gains(0.5).unwrap();
    assert!(w_50 < 0.5);
    
    // Test underweighting of large probabilities
    let w_99 = pw.weight_gains(0.99).unwrap();
    assert!(w_99 < 0.99);
}

#[test]
fn test_certainty_effect() {
    let pw = ProbabilityWeighting::default_tk();
    
    // Certainty should be weighted at exactly 1.0
    let w_certain = pw.weight_gains(1.0).unwrap();
    assert_relative_eq!(w_certain, 1.0, epsilon = FINANCIAL_PRECISION);
    
    // Near certainty should be underweighted
    let w_near_certain = pw.weight_gains(0.99).unwrap();
    assert!(w_near_certain < 0.99);
}

#[test]
fn test_reference_point_dependence() {
    // Test with different reference points
    let params1 = ValueFunctionParams::new(0.88, 0.88, 2.25, 0.0).unwrap();
    let params2 = ValueFunctionParams::new(0.88, 0.88, 2.25, 50.0).unwrap();
    
    let vf1 = ValueFunction::new(params1).unwrap();
    let vf2 = ValueFunction::new(params2).unwrap();
    
    // Same absolute outcome, different reference points
    let outcome = 100.0;
    let value1 = vf1.value(outcome).unwrap();
    let value2 = vf2.value(outcome).unwrap();
    
    // Values should be different due to different reference points
    assert!((value1 - value2).abs() > FINANCIAL_PRECISION);
}

#[test]
fn test_risk_premium_calculation() {
    let vf = ValueFunction::default_kt();
    
    // Risky lottery: 50% chance of $200, 50% chance of $0
    let outcomes = vec![200.0, 0.0];
    let probabilities = vec![0.5, 0.5];
    
    let risk_premium = vf.risk_premium(&outcomes, &probabilities).unwrap();
    
    // Risk premium should be positive (risk aversion)
    assert!(risk_premium > 0.0);
}

#[test]
fn test_marginal_utility_diminishing() {
    let vf = ValueFunction::default_kt();
    
    // Test diminishing marginal utility for gains
    let mv_10 = vf.marginal_value(10.0).unwrap();
    let mv_100 = vf.marginal_value(100.0).unwrap();
    
    // Marginal value should decrease as outcome increases
    assert!(mv_10 > mv_100);
}

#[test]
fn test_parameter_sensitivity() {
    // Test different alpha values
    let params_low_alpha = ValueFunctionParams::new(0.5, 0.88, 2.25, 0.0).unwrap();
    let params_high_alpha = ValueFunctionParams::new(0.9, 0.88, 2.25, 0.0).unwrap();
    
    let vf_low = ValueFunction::new(params_low_alpha).unwrap();
    let vf_high = ValueFunction::new(params_high_alpha).unwrap();
    
    let outcome = 100.0;
    let value_low = vf_low.value(outcome).unwrap();
    let value_high = vf_high.value(outcome).unwrap();
    
    // Higher alpha should lead to higher value for gains
    assert!(value_high > value_low);
}

#[test]
fn test_decision_weights_sum_to_one() {
    let pw = ProbabilityWeighting::default_tk();
    
    let probabilities = vec![0.2, 0.3, 0.3, 0.2];
    let outcomes = vec![100.0, 50.0, -25.0, -75.0];
    
    let decision_weights = pw.decision_weights(&probabilities, &outcomes).unwrap();
    let sum: f64 = decision_weights.iter().sum();
    
    // Decision weights should sum to approximately 1
    assert_relative_eq!(sum, 1.0, epsilon = FINANCIAL_PRECISION * 100.0);
}

#[test]
fn test_prelec_vs_tversky_kahneman() {
    let params = WeightingParams::default();
    let pw_tk = ProbabilityWeighting::new(params.clone(), WeightingFunction::TverskyKahneman).unwrap();
    let pw_prelec = ProbabilityWeighting::prelec(params).unwrap();
    
    let prob = 0.3;
    let weight_tk = pw_tk.weight_gains(prob).unwrap();
    let weight_prelec = pw_prelec.weight_gains(prob).unwrap();
    
    // Both should overweight small probabilities, but values will differ
    assert!(weight_tk > prob);
    assert!(weight_prelec > prob);
    assert!((weight_tk - weight_prelec).abs() > FINANCIAL_PRECISION);
}

#[test]
fn test_parallel_computation_consistency() {
    let vf = ValueFunction::default_kt();
    let outcomes: Vec<f64> = (0..1000).map(|i| i as f64 - 500.0).collect();
    
    let values_sequential = vf.values(&outcomes).unwrap();
    let values_parallel = vf.values_parallel(&outcomes).unwrap();
    
    assert_eq!(values_sequential.len(), values_parallel.len());
    
    for (seq, par) in values_sequential.iter().zip(values_parallel.iter()) {
        assert_relative_eq!(seq, par, epsilon = FINANCIAL_PRECISION);
    }
}

#[test]
fn test_financial_precision_bounds() {
    let vf = ValueFunction::default_kt();
    
    // Test very small values
    let small_value = vf.value(FINANCIAL_PRECISION / 2.0).unwrap();
    assert!(small_value >= 0.0);
    
    // Test at bounds
    let max_value = vf.value(MAX_INPUT_VALUE - 1.0).unwrap();
    assert!(max_value.is_finite());
    
    let min_value = vf.value(MIN_INPUT_VALUE + 1.0).unwrap();
    assert!(min_value.is_finite());
}

#[test]
fn test_error_handling_robustness() {
    let vf = ValueFunction::default_kt();
    let pw = ProbabilityWeighting::default_tk();
    
    // Test invalid inputs
    assert!(vf.value(f64::NAN).is_err());
    assert!(vf.value(f64::INFINITY).is_err());
    assert!(vf.value(MAX_INPUT_VALUE + 1.0).is_err());
    
    assert!(pw.weight_gains(-0.1).is_err());
    assert!(pw.weight_gains(1.1).is_err());
    assert!(pw.weight_gains(f64::NAN).is_err());
}

#[test]
fn test_thread_safety() {
    use std::thread;
    use std::sync::Arc;
    
    let vf = Arc::new(ValueFunction::default_kt());
    let pw = Arc::new(ProbabilityWeighting::default_tk());
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let vf_clone = Arc::clone(&vf);
        let pw_clone = Arc::clone(&pw);
        
        let handle = thread::spawn(move || {
            let outcome = (i as f64) * 10.0;
            let prob = 0.1 * (i as f64+ 1.0);
            
            let value = vf_clone.value(outcome).unwrap();
            let weight = pw_clone.weight_gains(prob).unwrap();
            
            (value, weight)
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        let (value, weight) = handle.join().unwrap();
        assert!(value.is_finite());
        assert!(weight.is_finite());
        assert!(weight >= 0.0 && weight <= 1.0);
    }
}

#[test]
fn test_memory_safety_large_datasets() {
    let vf = ValueFunction::default_kt();
    let pw = ProbabilityWeighting::default_tk();
    
    // Test with large datasets to check for memory leaks
    let large_outcomes: Vec<f64> = (0..100_000).map(|i| (i as f64) - 50_000.0).collect();
    let large_probs: Vec<f64> = (0..100_000).map(|i| (i as f64) / 100_000.0).collect();
    
    let values = vf.values_parallel(&large_outcomes).unwrap();
    let weights = pw.weights_gains_parallel(&large_probs[1..99_999].to_vec()).unwrap(); // Exclude 0 and 1
    
    assert_eq!(values.len(), large_outcomes.len());
    assert_eq!(weights.len(), 99_998);
    
    // Verify no memory corruption
    for value in values.iter().take(100) {
        assert!(value.is_finite());
    }
    
    for weight in weights.iter().take(100) {
        assert!(weight.is_finite());
        assert!(*weight >= 0.0 && *weight <= 1.0);
    }
}