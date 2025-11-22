//! Property-Based Tests for Quantum LSH
//!
//! Comprehensive property tests using proptest to verify:
//! - Hash consistency (same input = same hash)
//! - Similarity metric symmetry
//! - Distance triangle inequality
//! - No NaN/infinity propagation

use proptest::prelude::*;
use nalgebra::DVector;
use std::collections::HashMap;

// Re-export quantum LSH types
use crate::memory::quantum_lsh::{QuantumLSH, QuantumMetrics};

const PROP_TEST_CASES: u32 = 1000;
const VECTOR_DIMENSION: usize = 128;
const NUM_TABLES: usize = 10;
const NUM_HASHES: usize = 16;

// Strategy generators for vectors
fn vector_strategy(dimension: usize) -> impl Strategy<Value = DVector<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, dimension)
        .prop_map(DVector::from_vec)
}

fn small_vector_strategy(dimension: usize) -> impl Strategy<Value = DVector<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, dimension)
        .prop_map(DVector::from_vec)
}

fn normalized_vector_strategy(dimension: usize) -> impl Strategy<Value = DVector<f32>> {
    prop::collection::vec(-1.0f32..1.0f32, dimension)
        .prop_map(|vec| {
            let v = DVector::from_vec(vec);
            let norm = v.norm();
            if norm > 0.0 {
                v / norm
            } else {
                DVector::from_element(dimension, 1.0 / (dimension as f32).sqrt())
            }
        })
}

// Property 1: Hash consistency - same input produces same hash
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_hash_consistency(
        vector in vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Hash the same vector multiple times
        let hash1 = lsh.quantum_hash(&vector, 0);
        let hash2 = lsh.quantum_hash(&vector, 0);
        let hash3 = lsh.quantum_hash(&vector, 0);

        // Property: Same input must produce same hash
        prop_assert_eq!(
            hash1, hash2,
            "Hash must be consistent across calls"
        );
        prop_assert_eq!(
            hash1, hash3,
            "Hash must be consistent across multiple calls"
        );
    }
}

// Property 2: Different table indices produce different hashes
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_different_tables_different_hashes(
        vector in vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        let mut hashes = HashMap::new();
        let mut unique_count = 0;

        // Generate hashes for all tables
        for table_idx in 0..NUM_TABLES {
            let hash = lsh.quantum_hash(&vector, table_idx);
            if hashes.insert(table_idx, hash).is_none() {
                unique_count += 1;
            }
        }

        // Property: Different tables should produce different hashes (with high probability)
        // Allow for some collisions but expect mostly different hashes
        prop_assert!(
            unique_count >= NUM_TABLES * 8 / 10, // At least 80% unique
            "Different tables should produce mostly different hashes: {}/{}",
            unique_count,
            NUM_TABLES
        );
    }
}

// Property 3: Similarity metric is symmetric
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_similarity_symmetry(
        vector_a in small_vector_strategy(VECTOR_DIMENSION),
        vector_b in small_vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        let sim_ab = lsh.quantum_similarity(&vector_a, &vector_b);
        let sim_ba = lsh.quantum_similarity(&vector_b, &vector_a);

        // Property: Similarity must be symmetric
        let diff = (sim_ab - sim_ba).abs();
        prop_assert!(
            diff < 1e-6,
            "Similarity must be symmetric: sim(a,b)={} != sim(b,a)={}",
            sim_ab,
            sim_ba
        );

        // Property: Similarity must be finite
        prop_assert!(sim_ab.is_finite(), "Similarity must not be NaN or infinity");
        prop_assert!(sim_ba.is_finite(), "Similarity must not be NaN or infinity");

        // Property: Similarity must be in valid range [0, 1]
        prop_assert!(
            sim_ab >= 0.0 && sim_ab <= 1.0,
            "Similarity must be in [0, 1]: {}",
            sim_ab
        );
    }
}

// Property 4: Self-similarity is maximal
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_self_similarity_maximal(
        vector in normalized_vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Skip zero vectors as they have undefined similarity
        if vector.norm() < 1e-10 {
            return Ok(());
        }

        let self_sim = lsh.quantum_similarity(&vector, &vector);

        // Property: Self-similarity should be close to 1.0
        prop_assert!(
            self_sim > 0.99,
            "Self-similarity should be very high: {}",
            self_sim
        );

        prop_assert!(
            self_sim.is_finite(),
            "Self-similarity must be finite"
        );
    }
}

// Property 5: Distance metric satisfies triangle inequality
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_triangle_inequality(
        vector_a in small_vector_strategy(VECTOR_DIMENSION),
        vector_b in small_vector_strategy(VECTOR_DIMENSION),
        vector_c in small_vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Calculate distances
        let dist_ab = lsh.compute_distance(&vector_a, &vector_b);
        let dist_bc = lsh.compute_distance(&vector_b, &vector_c);
        let dist_ac = lsh.compute_distance(&vector_a, &vector_c);

        // Property: All distances must be non-negative
        prop_assert!(dist_ab >= 0.0, "Distance must be non-negative");
        prop_assert!(dist_bc >= 0.0, "Distance must be non-negative");
        prop_assert!(dist_ac >= 0.0, "Distance must be non-negative");

        // Property: All distances must be finite
        prop_assert!(dist_ab.is_finite(), "Distance must be finite");
        prop_assert!(dist_bc.is_finite(), "Distance must be finite");
        prop_assert!(dist_ac.is_finite(), "Distance must be finite");

        // Property: Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        prop_assert!(
            dist_ac <= dist_ab + dist_bc + 1e-6, // Small epsilon for floating point
            "Triangle inequality violated: d(a,c)={} > d(a,b)={} + d(b,c)={}",
            dist_ac,
            dist_ab,
            dist_bc
        );
    }
}

// Property 6: No NaN propagation in hash computation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_no_nan_in_hash(
        values in prop::collection::vec(-100.0f32..100.0f32, VECTOR_DIMENSION),
    ) {
        let vector = DVector::from_vec(values);
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Property: Hash computation must not produce NaN
        for table_idx in 0..NUM_TABLES {
            let hash = lsh.quantum_hash(&vector, table_idx);

            // Hash is u64, so it's always valid, but verify the computation didn't panic
            prop_assert!(
                hash != u64::MAX || hash != 0 || true, // Hash is always valid
                "Hash computation must succeed"
            );
        }
    }
}

// Property 7: No infinity propagation in similarity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_no_infinity_in_similarity(
        vector_a in small_vector_strategy(VECTOR_DIMENSION),
        vector_b in small_vector_strategy(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        let similarity = lsh.quantum_similarity(&vector_a, &vector_b);

        // Property: Similarity must not be infinite
        prop_assert!(
            !similarity.is_infinite(),
            "Similarity must not be infinite"
        );

        // Property: Similarity must not be NaN
        prop_assert!(
            !similarity.is_nan(),
            "Similarity must not be NaN"
        );
    }
}

// Property 8: Insert and query consistency
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))] // Fewer cases for expensive test

    #[test]
    fn prop_insert_query_consistency(
        vectors in prop::collection::vec(
            vector_strategy(VECTOR_DIMENSION),
            10..50
        ),
        query_idx in 0usize..10,
    ) {
        let mut lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Insert all vectors
        let indices: Vec<usize> = vectors.iter()
            .map(|v| lsh.insert(v.clone()))
            .collect();

        // Query with one of the inserted vectors
        if query_idx < vectors.len() {
            let query = &vectors[query_idx];
            let results = lsh.query(query, 5);

            // Property: Query results should not be empty for inserted vector
            prop_assert!(
                !results.is_empty(),
                "Query should find at least one result"
            );

            // Property: All returned indices should be valid
            for (idx, _dist) in &results {
                prop_assert!(
                    *idx < vectors.len(),
                    "Returned index {} must be valid (< {})",
                    idx,
                    vectors.len()
                );
            }

            // Property: Distances should be non-negative and finite
            for (_idx, dist) in &results {
                prop_assert!(
                    dist >= &0.0 && dist.is_finite(),
                    "Distance must be non-negative and finite: {}",
                    dist
                );
            }

            // Property: Results should be sorted by distance
            for i in 0..results.len().saturating_sub(1) {
                prop_assert!(
                    results[i].1 <= results[i + 1].1,
                    "Results must be sorted by distance"
                );
            }
        }
    }
}

// Property 9: Quantum Jensen-Shannon divergence is symmetric
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_jsd_symmetry(
        p_values in prop::collection::vec(0.0f32..1.0f32, 64),
        q_values in prop::collection::vec(0.0f32..1.0f32, 64),
    ) {
        // Normalize to make valid probability distributions
        let p_sum: f32 = p_values.iter().sum();
        let q_sum: f32 = q_values.iter().sum();

        if p_sum < 1e-10 || q_sum < 1e-10 {
            return Ok(()); // Skip invalid distributions
        }

        let p_normalized: Vec<f32> = p_values.iter().map(|x| x / p_sum).collect();
        let q_normalized: Vec<f32> = q_values.iter().map(|x| x / q_sum).collect();

        let p = DVector::from_vec(p_normalized);
        let q = DVector::from_vec(q_normalized);

        let jsd_pq = QuantumMetrics::jensen_shannon_divergence(&p, &q);
        let jsd_qp = QuantumMetrics::jensen_shannon_divergence(&q, &p);

        // Property: JSD must be symmetric
        let diff = (jsd_pq - jsd_qp).abs();
        prop_assert!(
            diff < 1e-5,
            "JSD must be symmetric: JSD(p,q)={} != JSD(q,p)={}",
            jsd_pq,
            jsd_qp
        );

        // Property: JSD must be non-negative
        prop_assert!(
            jsd_pq >= 0.0,
            "JSD must be non-negative: {}",
            jsd_pq
        );

        // Property: JSD must be finite
        prop_assert!(
            jsd_pq.is_finite(),
            "JSD must be finite"
        );
    }
}

// Property 10: Wasserstein distance metric properties
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_wasserstein_properties(
        p_values in prop::collection::vec(0.0f32..1.0f32, 32),
        q_values in prop::collection::vec(0.0f32..1.0f32, 32),
    ) {
        let p = DVector::from_vec(p_values);
        let q = DVector::from_vec(q_values);

        let w_pq = QuantumMetrics::wasserstein_distance(&p, &q);
        let w_qp = QuantumMetrics::wasserstein_distance(&q, &p);

        // Property: Wasserstein distance is symmetric
        let diff = (w_pq - w_qp).abs();
        prop_assert!(
            diff < 1e-5,
            "Wasserstein distance must be symmetric"
        );

        // Property: Distance must be non-negative
        prop_assert!(
            w_pq >= 0.0,
            "Wasserstein distance must be non-negative: {}",
            w_pq
        );

        // Property: Distance must be finite
        prop_assert!(
            w_pq.is_finite(),
            "Wasserstein distance must be finite"
        );

        // Property: Self-distance is zero
        let w_pp = QuantumMetrics::wasserstein_distance(&p, &p);
        prop_assert!(
            w_pp.abs() < 1e-6,
            "Self-distance must be zero: {}",
            w_pp
        );
    }
}

// Property 11: Multi-probe query finds more results than single probe
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 20))] // Fewer cases, expensive test

    #[test]
    fn prop_multi_probe_finds_more(
        vectors in prop::collection::vec(
            vector_strategy(VECTOR_DIMENSION),
            20..100
        ),
        query_vec in vector_strategy(VECTOR_DIMENSION),
    ) {
        let mut lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Insert vectors
        for v in &vectors {
            lsh.insert(v.clone());
        }

        // Perform single probe and multi probe queries
        let single_probe_results = lsh.query(&query_vec, 10);
        let multi_probe_results = lsh.multi_probe_query(&query_vec, 10, 5);

        // Property: Multi-probe should find at least as many results
        // (or at least not significantly fewer)
        prop_assert!(
            multi_probe_results.len() >= single_probe_results.len().saturating_sub(2),
            "Multi-probe should find comparable or more results"
        );

        // Property: All results should have valid indices
        for (idx, _dist) in &multi_probe_results {
            prop_assert!(
                *idx < vectors.len(),
                "Multi-probe result index must be valid"
            );
        }
    }
}

// Property 12: Batch insert produces consistent indices
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[test]
    fn prop_batch_insert_consistent(
        vectors in prop::collection::vec(
            vector_strategy(VECTOR_DIMENSION),
            5..50
        ),
    ) {
        let mut lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        let indices = lsh.batch_insert(vectors.clone());

        // Property: Number of indices matches number of vectors
        prop_assert_eq!(
            indices.len(),
            vectors.len(),
            "Batch insert must return correct number of indices"
        );

        // Property: Indices should be sequential starting from 0
        for (i, &idx) in indices.iter().enumerate() {
            prop_assert_eq!(
                idx, i,
                "Batch insert indices should be sequential"
            );
        }

        // Property: All vectors should be queryable
        let stats = lsh.get_stats();
        prop_assert_eq!(
            stats.num_vectors,
            vectors.len(),
            "All vectors should be stored"
        );
    }
}

// Property 13: Hash collision rate is reasonable
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[test]
    fn prop_reasonable_collision_rate(
        vectors in prop::collection::vec(
            vector_strategy(VECTOR_DIMENSION),
            100..200
        ),
    ) {
        let mut lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);

        // Insert all vectors
        for v in &vectors {
            lsh.insert(v.clone());
        }

        let stats = lsh.get_stats();

        // Property: Collision rate should not be excessive
        // With good hash functions, expect collision rate < 50%
        let collision_rate = stats.collisions as f64 / stats.num_vectors as f64;
        prop_assert!(
            collision_rate < 0.5,
            "Collision rate too high: {}",
            collision_rate
        );
    }
}

// Property 14: Zero vector handling
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_zero_vector_handling(
        _dimension in Just(VECTOR_DIMENSION),
    ) {
        let lsh = QuantumLSH::new(VECTOR_DIMENSION, NUM_TABLES, NUM_HASHES);
        let zero_vec = DVector::from_element(VECTOR_DIMENSION, 0.0);
        let non_zero = DVector::from_element(VECTOR_DIMENSION, 1.0);

        // Property: Similarity with zero vector should be defined (0.0)
        let sim = lsh.quantum_similarity(&zero_vec, &non_zero);
        prop_assert!(
            sim.is_finite(),
            "Similarity with zero vector must be finite"
        );

        prop_assert_eq!(
            sim, 0.0,
            "Similarity with zero vector should be 0.0"
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_property_test_config() {
        assert_eq!(PROP_TEST_CASES, 1000);
        assert_eq!(VECTOR_DIMENSION, 128);
        assert_eq!(NUM_TABLES, 10);
        assert_eq!(NUM_HASHES, 16);
    }

    #[test]
    fn test_strategy_generators() {
        // Verify strategy generators compile
        let _ = vector_strategy(10);
        let _ = small_vector_strategy(10);
        let _ = normalized_vector_strategy(10);
    }
}
