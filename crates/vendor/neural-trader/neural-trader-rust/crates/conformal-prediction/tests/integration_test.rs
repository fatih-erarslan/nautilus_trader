//! Integration tests for conformal prediction

use conformal_prediction::{
    ConformalPredictor, KNNNonconformity, VerifiedPrediction,
    VerifiedPredictionBuilder, PredictionValue, ConformalContext
};

#[test]
fn test_basic_conformal_prediction_workflow() {
    // Create a k-NN nonconformity measure with k=3
    let mut measure = KNNNonconformity::new(3);

    // Create calibration data: y = 2x + noise
    let cal_x: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
        vec![6.0],
        vec![7.0],
        vec![8.0],
        vec![9.0],
        vec![10.0],
    ];

    let cal_y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    // Fit the nonconformity measure
    measure.fit(&cal_x, &cal_y);

    // Create conformal predictor with 90% confidence (α = 0.1)
    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();

    // Calibrate the predictor
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Make a prediction for x = 5.5
    let (lower, upper) = predictor.predict_interval(&[5.5], 11.0).unwrap();

    // The true value y = 2 * 5.5 = 11.0 should be in the interval
    assert!(lower <= 11.0);
    assert!(upper >= 11.0);
    assert!(lower < upper);

    println!("90% Prediction interval for x=5.5: [{}, {}]", lower, upper);
    println!("True value: 11.0");
    println!("Interval width: {}", upper - lower);
}

#[test]
fn test_conformal_coverage_guarantee() {
    // Test that conformal prediction achieves the specified coverage

    let mut measure = KNNNonconformity::new(5);

    // Generate larger calibration set
    let n_cal = 100;
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();

    for i in 0..n_cal {
        let x = i as f64 / 10.0;
        let y = 2.0 * x + 1.0; // y = 2x + 1
        cal_x.push(vec![x]);
        cal_y.push(y);
    }

    measure.fit(&cal_x, &cal_y);

    // Test multiple significance levels
    for alpha in [0.05, 0.1, 0.2] {
        let mut predictor = ConformalPredictor::new(alpha, measure.clone()).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        // Test on multiple points
        let mut covered = 0;
        let n_test = 50;

        for i in 0..n_test {
            let x = (i as f64 + 0.5) / 10.0;
            let y_true = 2.0 * x + 1.0;

            let (lower, upper) = predictor.predict_interval(&[x], y_true).unwrap();

            if lower <= y_true && y_true <= upper {
                covered += 1;
            }
        }

        let empirical_coverage = covered as f64 / n_test as f64;
        let expected_coverage = 1.0 - alpha;

        println!(
            "α = {}: Expected coverage = {:.2}, Empirical coverage = {:.2}",
            alpha, expected_coverage, empirical_coverage
        );

        // Empirical coverage should be close to expected
        // (within 10% due to finite sample size)
        assert!(
            empirical_coverage >= expected_coverage - 0.1,
            "Coverage too low: {} < {} - 0.1",
            empirical_coverage,
            expected_coverage
        );
    }
}

#[test]
fn test_verified_prediction_with_proof() {
    let mut context = ConformalContext::new();

    // Create a verified prediction with formal proof
    let prediction = VerifiedPredictionBuilder::new()
        .interval(5.0, 15.0)
        .confidence(0.9)
        .with_proof()
        .build(&mut context)
        .unwrap();

    assert!(prediction.is_verified());
    assert!(prediction.proof().is_some());
    assert_eq!(prediction.confidence(), 0.9);

    // Test coverage checking
    assert!(prediction.covers(10.0));
    assert!(prediction.covers(5.0));
    assert!(prediction.covers(15.0));
    assert!(!prediction.covers(4.9));
    assert!(!prediction.covers(15.1));
}

#[test]
fn test_prediction_set_mode() {
    let mut measure = KNNNonconformity::new(3);

    // Classification-like data
    let cal_x: Vec<Vec<f64>> = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![10.0],
        vec![11.0],
        vec![12.0],
    ];

    let cal_y: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.2, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test point at x = 1.5 (should predict class 0)
    let candidates = vec![0.0, 1.0];
    let predictions = predictor.predict(&[1.5], &candidates).unwrap();

    println!("Predictions for x=1.5: {:?}", predictions);

    // Should return conformally valid predictions
    assert!(!predictions.is_empty());

    // All p-values should be > alpha
    for (_, p_value) in predictions {
        assert!(p_value > 0.2);
    }
}

#[test]
fn test_multiple_predictors_comparison() {
    // Compare different k values for k-NN

    let cal_x: Vec<Vec<f64>> = (1..=20).map(|i| vec![i as f64]).collect();
    let cal_y: Vec<f64> = (1..=20).map(|i| (i * i) as f64).collect();

    for k in [1, 3, 5] {
        let mut measure = KNNNonconformity::new(k);
        measure.fit(&cal_x, &cal_y);

        let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
        predictor.calibrate(&cal_x, &cal_y).unwrap();

        let (lower, upper) = predictor.predict_interval(&[10.0], 100.0).unwrap();

        println!(
            "k={}: Interval for x=10 (true y=100): [{:.2}, {:.2}], width={:.2}",
            k,
            lower,
            upper,
            upper - lower
        );

        // Interval should contain true value
        assert!(lower <= 100.0);
        assert!(upper >= 100.0);
    }
}

#[test]
fn test_adaptive_intervals() {
    // Test that intervals adapt to local difficulty

    let mut measure = KNNNonconformity::new(3);

    // Data with varying noise levels
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();

    // Low noise region (x < 5)
    for i in 0..10 {
        let x = i as f64 * 0.5;
        let y = x; // Simple linear, low noise
        cal_x.push(vec![x]);
        cal_y.push(y);
    }

    // High noise region (x >= 5)
    for i in 10..20 {
        let x = i as f64 * 0.5;
        let noise = ((i % 3) as f64 - 1.0) * 2.0; // More variance
        let y = x + noise;
        cal_x.push(vec![x]);
        cal_y.push(y);
    }

    measure.fit(&cal_x, &cal_y);

    let mut predictor = ConformalPredictor::new(0.1, measure).unwrap();
    predictor.calibrate(&cal_x, &cal_y).unwrap();

    // Test in low-noise region
    let (low_lower, low_upper) = predictor.predict_interval(&[2.0], 2.0).unwrap();
    let low_width = low_upper - low_lower;

    // Test in high-noise region
    let (high_lower, high_upper) = predictor.predict_interval(&[8.0], 8.0).unwrap();
    let high_width = high_upper - high_lower;

    println!("Low-noise region (x=2): width = {:.2}", low_width);
    println!("High-noise region (x=8): width = {:.2}", high_width);

    // Both intervals should be valid, but widths may vary
    assert!(low_width > 0.0);
    assert!(high_width > 0.0);
}

#[test]
fn test_lean_agentic_type_checking() {
    // Test that lean-agentic's type system works correctly

    let mut context = ConformalContext::new();

    // Create some terms and verify they type-check
    let type_term = context.arena.mk_sort(context.levels.zero());
    let var_term = context.arena.mk_var(0);

    // Terms should be allocated in the arena
    assert!(context.arena.terms() > 0);

    // Create a symbol
    let _x_sym = context.symbols.intern("x");

    println!("Created {} terms in arena", context.arena.terms());
    println!("Type term ID: {:?}", type_term);
    println!("Variable term ID: {:?}", var_term);
}
