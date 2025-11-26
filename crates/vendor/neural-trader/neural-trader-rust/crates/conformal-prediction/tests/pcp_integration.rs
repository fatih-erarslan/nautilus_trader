//! Integration tests for Posterior Conformal Prediction

use conformal_prediction::pcp::PosteriorConformalPredictor;
use conformal_prediction::Result;

#[test]
fn test_pcp_coverage_guarantee() -> Result<()> {
    // Test that empirical coverage meets target
    let alpha = 0.1;
    let mut predictor = PosteriorConformalPredictor::new(alpha)?;

    // Generate synthetic data with clear clusters
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();
    let mut cal_pred = Vec::new();

    // Cluster 1: y = 2x + noise
    for i in 0..50 {
        let x = i as f64 * 0.1;
        cal_x.push(vec![x]);
        cal_y.push(2.0 * x + (i as f64 % 3.0 - 1.0) * 0.1);
        cal_pred.push(2.0 * x);
    }

    // Cluster 2: y = 10 + x + noise
    for i in 0..50 {
        let x = i as f64 * 0.1;
        cal_x.push(vec![x + 10.0]);
        cal_y.push(10.0 + x + (i as f64 % 3.0 - 1.0) * 0.5);
        cal_pred.push(10.0 + x);
    }

    predictor.fit(&cal_x, &cal_y, &cal_pred, 2)?;

    // Test coverage on independent data
    let mut covered = 0;
    let n_test = 100;

    for i in 0..n_test {
        let x_val = (i as f64 % 50.0) * 0.1;
        let test_x = if i < n_test / 2 {
            vec![x_val]
        } else {
            vec![x_val + 10.0]
        };

        let true_y = if i < n_test / 2 {
            2.0 * x_val
        } else {
            10.0 + x_val
        };

        let pred_y = true_y;
        let (lower, upper) = predictor.predict_cluster_aware(&test_x, pred_y)?;

        if lower <= true_y && true_y <= upper {
            covered += 1;
        }
    }

    let empirical_coverage = covered as f64 / n_test as f64;
    let target_coverage = 1.0 - alpha;

    println!("Empirical coverage: {:.1}%", empirical_coverage * 100.0);
    println!("Target coverage: {:.1}%", target_coverage * 100.0);

    // Allow some slack due to finite sample size
    assert!(empirical_coverage >= target_coverage - 0.1);

    Ok(())
}

#[test]
fn test_pcp_cluster_adaptation() -> Result<()> {
    // Test that different clusters get different interval widths
    let mut predictor = PosteriorConformalPredictor::new(0.1)?;

    // Create clusters with different error magnitudes
    let cal_x = vec![
        vec![0.0], vec![0.1], vec![0.2], vec![0.3], vec![0.4],  // Cluster 1
        vec![10.0], vec![10.1], vec![10.2], vec![10.3], vec![10.4],  // Cluster 2
    ];

    let cal_y = vec![
        1.0, 1.0, 1.0, 1.0, 1.0,  // Cluster 1: small errors
        10.0, 10.0, 10.0, 10.0, 10.0,  // Cluster 2: large errors
    ];

    let cal_pred = vec![
        1.05, 0.98, 1.02, 0.97, 1.03,  // Errors ~0.05
        10.5, 9.2, 10.8, 9.5, 10.3,  // Errors ~0.5
    ];

    predictor.fit(&cal_x, &cal_y, &cal_pred, 2)?;

    // Get intervals for each cluster
    let (lower1, upper1) = predictor.predict_cluster_aware(&[0.15], 1.0)?;
    let (lower2, upper2) = predictor.predict_cluster_aware(&[10.15], 10.0)?;

    let width1 = upper1 - lower1;
    let width2 = upper2 - lower2;

    println!("Cluster 1 interval width: {:.3}", width1);
    println!("Cluster 2 interval width: {:.3}", width2);

    // Cluster 2 should have wider intervals
    assert!(width2 > width1 * 2.0, "Cluster 2 should have significantly wider intervals");

    Ok(())
}

#[test]
fn test_pcp_soft_vs_hard() -> Result<()> {
    // Compare soft and hard clustering predictions
    let mut predictor = PosteriorConformalPredictor::new(0.1)?;

    let cal_x = vec![
        vec![0.0], vec![0.5], vec![1.0],
        vec![10.0], vec![10.5], vec![11.0],
    ];

    let cal_y = vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0];
    let cal_pred = vec![1.1, 0.9, 1.05, 10.5, 9.5, 10.2];

    predictor.fit(&cal_x, &cal_y, &cal_pred, 2)?;

    // Test point between clusters
    let test_x = vec![5.0];
    let (lower_hard, upper_hard) = predictor.predict_cluster_aware(&test_x, 5.0)?;
    let (lower_soft, upper_soft) = predictor.predict_soft(&test_x, 5.0)?;

    // Both should be valid intervals
    assert!(lower_hard < upper_hard);
    assert!(lower_soft < upper_soft);

    // Intervals can differ but should be in reasonable range
    let width_hard = upper_hard - lower_hard;
    let width_soft = upper_soft - lower_soft;
    assert!(width_hard > 0.0 && width_soft > 0.0);

    println!("Hard clustering width: {:.3}", width_hard);
    println!("Soft clustering width: {:.3}", width_soft);

    Ok(())
}

#[test]
fn test_pcp_single_cluster() -> Result<()> {
    // Test that PCP works with single cluster (degenerates to standard CP)
    let mut predictor = PosteriorConformalPredictor::new(0.1)?;

    let cal_x = vec![
        vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0],
    ];

    let cal_y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cal_pred = vec![1.1, 1.9, 3.2, 3.8, 5.1];

    predictor.fit(&cal_x, &cal_y, &cal_pred, 1)?;

    // All points should get same cluster
    let cluster1 = predictor.predict_cluster(&[0.5])?;
    let cluster2 = predictor.predict_cluster(&[3.5])?;
    assert_eq!(cluster1, cluster2);

    // Predictions should still work
    let (lower, upper) = predictor.predict_cluster_aware(&[2.5], 3.0)?;
    assert!(lower < 3.0 && 3.0 < upper);

    Ok(())
}

#[test]
fn test_pcp_many_clusters() -> Result<()> {
    // Test with more clusters than typical
    let mut predictor = PosteriorConformalPredictor::new(0.1)?;

    // Create 5 distinct clusters
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();
    let mut cal_pred = Vec::new();

    for cluster in 0..5 {
        for i in 0..10 {
            let base = cluster as f64 * 10.0;
            cal_x.push(vec![base + i as f64 * 0.1]);
            cal_y.push(base);
            cal_pred.push(base + 0.1);
        }
    }

    predictor.fit(&cal_x, &cal_y, &cal_pred, 5)?;

    let sizes = predictor.cluster_sizes()?;
    assert_eq!(sizes.len(), 5);
    println!("Cluster sizes with 5 clusters: {:?}", sizes);

    // Should be able to make predictions
    let (lower, upper) = predictor.predict_cluster_aware(&[15.0], 15.0)?;
    assert!(lower <= 15.0 && 15.0 <= upper);

    Ok(())
}
