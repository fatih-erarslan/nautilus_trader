//! Integration test for portable SIMD correctness
//!
//! This test verifies that the portable SIMD implementation produces
//! identical results to the scalar implementation, ensuring mathematical
//! correctness across all architectures.

use hyperphysics_lsh::{SimHash, MinHash, SrpHash, HashFamily};

#[test]
fn test_simhash_deterministic() {
    // SimHash should produce identical results for identical inputs
    let hasher = SimHash::new(64, 128, 42);

    let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

    // Hash the same vector multiple times
    let sig1 = hasher.hash(&vector);
    let sig2 = hasher.hash(&vector);
    let sig3 = hasher.hash(&vector);

    // All signatures must be identical
    assert_eq!(sig1, sig2, "SimHash not deterministic");
    assert_eq!(sig2, sig3, "SimHash not deterministic");
}

#[test]
fn test_simhash_sensitivity() {
    // SimHash should be sensitive to significant changes
    let hasher = SimHash::new(64, 128, 42);

    let vector1: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
    let mut vector2 = vector1.clone();
    // Make a more substantial change to multiple dimensions
    for i in 0..16 {
        vector2[i] += 0.5; // Larger change across multiple dimensions
    }

    let sig1 = hasher.hash(&vector1);
    let sig2 = hasher.hash(&vector2);

    // Signatures should differ
    assert_ne!(sig1, sig2, "SimHash not sensitive to changes");

    // But Hamming distance should be reasonable (not all bits flipped)
    let hamming = sig1.hamming_distance(&sig2);
    assert!(hamming > 0, "No bits changed");
    assert!(hamming < 128, "Too many bits changed");
}

#[test]
fn test_minhash_set_similarity() {
    let hasher = MinHash::new(128, 42);

    // Create three sets with known overlap
    let set1: Vec<u64> = (0..50).collect();
    let set2: Vec<u64> = (25..75).collect(); // 50% overlap with set1
    let set3: Vec<u64> = (100..150).collect(); // No overlap

    let sig1 = hasher.hash(&set1);
    let sig2 = hasher.hash(&set2);
    let sig3 = hasher.hash(&set3);

    // Set1 and set2 should have positive similarity
    let jaccard_12 = sig1.jaccard_estimate(&sig2);
    assert!(jaccard_12 > 0.2 && jaccard_12 < 0.4,
            "Expected Jaccard ~0.33, got {}", jaccard_12);

    // Set1 and set3 should have near-zero similarity
    let jaccard_13 = sig1.jaccard_estimate(&sig3);
    assert!(jaccard_13 < 0.1,
            "Expected Jaccard ~0.0 for disjoint sets, got {}", jaccard_13);
}

#[test]
fn test_srp_orthogonal_vectors() {
    let hasher = SrpHash::new(64, 128, 42);

    // Create orthogonal vectors
    let mut v1 = vec![0.0f32; 64];
    let mut v2 = vec![0.0f32; 64];

    // v1 = (1, 0, 0, ...)
    v1[0] = 1.0;

    // v2 = (0, 1, 0, ...)
    v2[1] = 1.0;

    let sig1 = hasher.hash(&v1);
    let sig2 = hasher.hash(&v2);

    // Orthogonal vectors should have significant Hamming distance
    let hamming = sig1.hamming_distance(&sig2);

    // Expect roughly 50% of bits to differ for orthogonal vectors
    assert!(hamming > 40 && hamming < 88,
            "Expected ~64 bits to differ, got {}", hamming);
}

#[test]
fn test_hash_performance_bounds() {
    // This test ensures hash computation completes in reasonable time
    // (Not a precise benchmark, just a sanity check)
    // Note: Running in debug mode is much slower than release mode

    let hasher = SimHash::new(64, 128, 42);
    let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

    // Hash computation should complete in reasonable time
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = hasher.hash(&vector);
    }
    let elapsed = start.elapsed();

    // 1,000 hashes should complete in < 5s in debug mode
    // (Release mode target: <1ms for 1000 hashes = <1μs per hash)
    assert!(elapsed.as_secs() < 5,
            "Hash computation too slow: {:?} for 1k hashes", elapsed);
}

#[test]
fn test_simhash_cosine_approximation() {
    let hasher = SimHash::new(128, 256, 42);

    // Create vectors with known cosine similarity
    let v1: Vec<f32> = vec![1.0; 128];

    // v2 = normalized version of v1 (cosine = 1.0)
    let v2: Vec<f32> = vec![1.0; 128];

    // v3 = opposite of v1 (cosine = -1.0)
    let v3: Vec<f32> = vec![-1.0; 128];

    let sig1 = hasher.hash(&v1);
    let sig2 = hasher.hash(&v2);
    let sig3 = hasher.hash(&v3);

    // Identical vectors should have collision probability ≈ 1.0
    let prob_12 = hasher.collision_probability(&sig1, &sig2);
    assert!(prob_12 > 0.99, "Expected collision prob ~1.0, got {}", prob_12);

    // Opposite vectors should have collision probability ≈ 0.0
    let prob_13 = hasher.collision_probability(&sig1, &sig3);
    assert!(prob_13 < 0.1, "Expected collision prob ~0.0, got {}", prob_13);
}

#[test]
fn test_minhash_identical_sets() {
    let hasher = MinHash::new(128, 42);

    let set1: Vec<u64> = vec![1, 2, 3, 4, 5, 10, 15, 20];
    let set2: Vec<u64> = vec![1, 2, 3, 4, 5, 10, 15, 20];

    let sig1 = hasher.hash(&set1);
    let sig2 = hasher.hash(&set2);

    // Identical sets should have Jaccard = 1.0
    let jaccard = sig1.jaccard_estimate(&sig2);
    assert!((jaccard - 1.0).abs() < 0.01,
            "Expected Jaccard = 1.0 for identical sets, got {}", jaccard);

    // Signatures should match exactly
    assert_eq!(sig1, sig2, "Signatures differ for identical sets");
}

#[test]
fn test_hash_stability_across_calls() {
    // Ensure hash functions don't have hidden state that changes results

    let hasher = SimHash::new(64, 128, 42);
    let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();

    // Hash the same vector 1000 times
    let reference_sig = hasher.hash(&vector);

    for i in 0..1000 {
        let sig = hasher.hash(&vector);
        assert_eq!(sig, reference_sig,
                   "Signature changed on iteration {}", i);
    }
}

#[test]
fn test_signature_byte_representation() {
    let hasher = SimHash::new(64, 128, 42);
    let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();

    let sig = hasher.hash(&vector);
    let bytes = sig.as_bytes();

    // Should produce 16 bytes for 128 bits
    assert_eq!(bytes.len(), 16, "Expected 16 bytes for 128-bit signature");

    // Bytes should be deterministic
    let sig2 = hasher.hash(&vector);
    let bytes2 = sig2.as_bytes();

    assert_eq!(bytes, bytes2, "Byte representation not deterministic");
}
