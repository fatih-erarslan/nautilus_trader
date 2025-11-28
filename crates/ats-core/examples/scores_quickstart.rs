//! Quickstart example for nonconformity scores module
//!
//! Demonstrates basic usage of all implemented scorers.

use ats_core::scores::*;

fn main() {
    println!("🎯 ATS-Core Nonconformity Scores - Quickstart Example\n");

    // Example softmax probabilities for a 3-class problem
    let softmax = vec![0.6, 0.3, 0.1];
    let true_label = 1; // Second class
    let u = 0.5; // Uniform random value for tie-breaking

    println!("Input:");
    println!("  Softmax probabilities: {:?}", softmax);
    println!("  True label: {} (probability: {})", true_label, softmax[true_label]);
    println!("  Random u value: {}\n", u);

    // 1. RAPS (Regularized Adaptive Prediction Sets)
    println!("1. RAPS (Regularized Adaptive Prediction Sets)");
    let raps_config = RapsConfig {
        lambda: 0.01,
        k_reg: 5,
        randomize_ties: true,
    };
    let raps_scorer = RapsScorer::new(raps_config);
    let raps_score = raps_scorer.score(&softmax, true_label, u);
    println!("   Score: {:.6}", raps_score);
    println!("   Config: λ=0.01, k_reg=5\n");

    // 2. APS (Adaptive Prediction Sets)
    println!("2. APS (Adaptive Prediction Sets)");
    let aps_scorer = ApsScorer::default();
    let aps_score = aps_scorer.score(&softmax, true_label, u);
    println!("   Score: {:.6}", aps_score);
    println!("   (RAPS without regularization)\n");

    // 3. SAPS (Sorted Adaptive Prediction Sets)
    println!("3. SAPS (Sorted Adaptive Prediction Sets)");
    let saps_scorer = SapsScorer::default();
    let saps_score = saps_scorer.score(&softmax, true_label, u);
    println!("   Score: {:.6}", saps_score);
    println!("   (APS with size penalty)\n");

    // 4. THR (Threshold-based)
    println!("4. THR (Threshold-based)");
    let thr_scorer = ThresholdScorer::default();
    let thr_score = thr_scorer.score(&softmax, true_label, u);
    println!("   Score: {:.6}", thr_score);
    println!("   (Simple: 1 - π̂_y)\n");

    // 5. LAC (Least Ambiguous Classifiers)
    println!("5. LAC (Least Ambiguous Classifiers)");
    let lac_scorer = LacScorer::default();
    let lac_score = lac_scorer.score(&softmax, true_label, u);
    println!("   Score: {:.6}", lac_score);
    println!("   (Weighted sum with uniform weights)\n");

    // Demonstrate batch processing
    println!("📊 Batch Processing Example\n");

    let batch = vec![
        vec![0.6, 0.3, 0.1],
        vec![0.5, 0.3, 0.2],
        vec![0.4, 0.4, 0.2],
    ];
    let labels = vec![0, 1, 2];
    let u_values = vec![0.5, 0.5, 0.5];

    println!("Processing {} samples...", batch.len());

    let raps_scores = raps_scorer.score_batch(&batch, &labels, &u_values);
    println!("RAPS scores: {:?}", raps_scores);

    let aps_scores = aps_scorer.score_batch(&batch, &labels, &u_values);
    println!("APS scores:  {:?}", aps_scores);

    println!("\n✅ Quickstart example completed successfully!");
    println!("\nNext steps:");
    println!("  - See tests/ for comprehensive examples");
    println!("  - Run benchmarks: cargo bench --bench scores_benchmark");
    println!("  - Read docs: docs/scores_module_implementation_report.md");
}
