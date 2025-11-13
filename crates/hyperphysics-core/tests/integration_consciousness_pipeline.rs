//! End-to-End Integration Tests for Consciousness Measurement Pipeline
//!
//! Tests the full workflow:
//! 1. Create pBit lattice in hyperbolic space
//! 2. Calculate hierarchical integrated information (Φ)
//! 3. Calculate resonance complexity (CI)
//! 4. Compute thermodynamic measures (negentropy, syntergy)
//! 5. Cryptographically sign the consciousness state
//! 6. Verify signature and tamper detection
//! 7. Generate audit trail

use hyperphysics_core::crypto::{SignedConsciousnessState, ConsciousnessMetrics, StateMetadata};
use hyperphysics_pbit::PBitLattice;
use hyperphysics_consciousness::{
    HierarchicalPhiCalculator, ClusteringMethod, CICalculator, CausalDensityEstimator,
};
use hyperphysics_thermo::EntropyCalculator;

#[test]
fn test_full_consciousness_pipeline_basic() {
    // Step 1: Create pBit lattice
    let temperature = 1.0;
    let lattice = PBitLattice::roi_48(temperature)
        .expect("Failed to create pBit lattice");

    println!("✓ Created pBit lattice with {} nodes", lattice.size());

    // Step 2: Calculate hierarchical Φ
    let phi_calculator = HierarchicalPhiCalculator::new(
        3,                              // 3 hierarchical levels
        ClusteringMethod::Tessellation, // Fast tessellation-based clustering
        2.0,                            // 2x scale factor between levels
        3,                              // Minimum 3 nodes per cluster
    );

    let phi_result = phi_calculator
        .calculate(&lattice)
        .expect("Failed to calculate hierarchical Φ");

    println!("✓ Hierarchical Φ = {:.4}", phi_result.phi_total);
    assert!(phi_result.phi_total >= 0.0);
    assert!(phi_result.phi_total.is_finite());

    // Step 3: Calculate resonance complexity (CI)
    let ci_calculator = CICalculator::new();
    let ci_result = ci_calculator
        .calculate(&lattice)
        .expect("Failed to calculate CI");

    println!("✓ Resonance Complexity (CI) = {:.4}", ci_result.ci);
    assert!(ci_result.ci >= 0.0);
    assert!(ci_result.ci.is_finite());

    // Step 4: Calculate thermodynamic measures
    let entropy_calculator = EntropyCalculator::new();
    let entropy = entropy_calculator.entropy_from_pbits(&lattice);
    let negentropy = entropy_calculator.negentropy(entropy, lattice.size());

    // Calculate causal density as a proxy for syntergy
    let syntergy = CausalDensityEstimator::causal_density(&lattice);

    println!("✓ Negentropy = {:.4e}", negentropy);
    println!("✓ Syntergy (causal density) = {:.4}", syntergy);

    assert!(negentropy >= 0.0);
    assert!(negentropy.is_finite());
    assert!(syntergy >= 0.0);
    assert!(syntergy.is_finite());

    // Step 5: Create signed consciousness state
    let metrics = ConsciousnessMetrics {
        phi: phi_result.phi_total,
        ci: ci_result.ci,
        syntergy,
        negentropy,
    };

    let signing_key = vec![42u8; 32]; // Placeholder key
    let signed_state = SignedConsciousnessState::create_and_sign(
        metrics,
        &signing_key,
        None,
    )
    .expect("Failed to create signed state");

    println!("✓ Created cryptographically signed state");

    // Step 6: Verify signature
    let verification_result = signed_state
        .verify()
        .expect("Failed to verify signed state");

    assert!(verification_result, "Signature verification should pass");
    println!("✓ Signature verification passed");

    // Step 7: Generate audit trail
    let audit_record = signed_state.audit_trail();
    assert!(audit_record.verified);
    assert_eq!(audit_record.phi, phi_result.phi_total);
    assert_eq!(audit_record.ci, ci_result.ci);

    println!("✓ Generated audit trail record");
    println!("\n=== Pipeline Completed Successfully ===");
}

#[test]
fn test_pipeline_with_metadata() {
    // Create lattice
    let lattice = PBitLattice::roi_48(1.5).expect("Failed to create lattice");

    // Calculate metrics
    let phi_calculator = HierarchicalPhiCalculator::default();
    let phi_result = phi_calculator.calculate(&lattice).unwrap();

    let ci_calculator = CICalculator::new();
    let ci_result = ci_calculator.calculate(&lattice).unwrap();

    let entropy_calculator = EntropyCalculator::new();
    let entropy = entropy_calculator.entropy_from_pbits(&lattice);
    let negentropy = entropy_calculator.negentropy(entropy, lattice.size());
    let syntergy = CausalDensityEstimator::causal_density(&lattice);

    // Create metadata
    let metadata = StateMetadata {
        subject_id: Some("experiment_001".to_string()),
        location: Some("lab_alpha".to_string()),
        operator: Some("researcher_bob".to_string()),
        protocol_version: "v1.0.0".to_string(),
    };

    // Sign with metadata
    let metrics = ConsciousnessMetrics {
        phi: phi_result.phi_total,
        ci: ci_result.ci,
        syntergy,
        negentropy,
    };

    let signing_key = vec![99u8; 32];
    let signed_state = SignedConsciousnessState::create_and_sign(
        metrics,
        &signing_key,
        Some(metadata.clone()),
    )
    .unwrap();

    // Verify metadata preservation
    let state_metadata = signed_state.metadata.unwrap();
    assert_eq!(state_metadata.subject_id, metadata.subject_id);
    assert_eq!(state_metadata.location, metadata.location);
    assert_eq!(state_metadata.operator, metadata.operator);
    assert_eq!(state_metadata.protocol_version, metadata.protocol_version);

    println!("✓ Metadata preserved through pipeline");
}

#[test]
fn test_pipeline_tamper_detection() {
    // Create and process through pipeline
    let lattice = PBitLattice::roi_48(1.0).unwrap();

    let phi_calculator = HierarchicalPhiCalculator::default();
    let phi_result = phi_calculator.calculate(&lattice).unwrap();

    let ci_calculator = CICalculator::new();
    let ci_result = ci_calculator.calculate(&lattice).unwrap();

    let entropy_calculator = EntropyCalculator::new();
    let entropy = entropy_calculator.entropy_from_pbits(&lattice);
    let negentropy = entropy_calculator.negentropy(entropy, lattice.size());
    let syntergy = CausalDensityEstimator::causal_density(&lattice);

    let metrics = ConsciousnessMetrics {
        phi: phi_result.phi_total,
        ci: ci_result.ci,
        syntergy,
        negentropy,
    };

    let signing_key = vec![7u8; 32];
    let mut signed_state = SignedConsciousnessState::create_and_sign(
        metrics,
        &signing_key,
        None,
    )
    .unwrap();

    // Verify original state
    assert!(signed_state.verify().unwrap());
    assert!(!signed_state.is_tampered());

    // Tamper with phi value
    signed_state.phi = 999.999;

    // Verification should fail
    assert!(signed_state.is_tampered());
    println!("✓ Tamper detection working correctly");
}

#[test]
fn test_pipeline_json_serialization() {
    // Full pipeline
    let lattice = PBitLattice::roi_48(2.0).unwrap();

    let phi_calculator = HierarchicalPhiCalculator::new(
        2,
        ClusteringMethod::HyperbolicKMeans,
        2.5,
        3,
    );
    let phi_result = phi_calculator.calculate(&lattice).unwrap();

    let ci_calculator = CICalculator::new();
    let ci_result = ci_calculator.calculate(&lattice).unwrap();

    let entropy_calculator = EntropyCalculator::new();
    let entropy = entropy_calculator.entropy_from_pbits(&lattice);
    let negentropy = entropy_calculator.negentropy(entropy, lattice.size());
    let syntergy = CausalDensityEstimator::causal_density(&lattice);

    let metrics = ConsciousnessMetrics {
        phi: phi_result.phi_total,
        ci: ci_result.ci,
        syntergy,
        negentropy,
    };

    let signing_key = vec![13u8; 32];
    let signed_state = SignedConsciousnessState::create_and_sign(
        metrics,
        &signing_key,
        None,
    )
    .unwrap();

    // Serialize to JSON
    let json = signed_state.to_json().expect("Failed to serialize to JSON");
    assert!(json.contains("phi"));
    assert!(json.contains("state_hash"));
    assert!(json.contains("signature"));

    // Deserialize from JSON
    let deserialized = SignedConsciousnessState::from_json(&json)
        .expect("Failed to deserialize from JSON");

    // Verify all metrics preserved
    assert!((deserialized.phi - signed_state.phi).abs() < 1e-10);
    assert!((deserialized.ci - signed_state.ci).abs() < 1e-10);
    assert!((deserialized.syntergy - signed_state.syntergy).abs() < 1e-10);
    assert!((deserialized.negentropy - signed_state.negentropy).abs() < 1e-10);

    // Verify signature still valid
    assert!(deserialized.verify().unwrap());

    println!("✓ JSON serialization/deserialization working");
}

#[test]
fn test_pipeline_multiple_temperatures() {
    // Test pipeline across different temperature regimes
    let temperatures = vec![0.5, 1.0, 2.0, 5.0];

    for temp in temperatures {
        println!("\n--- Testing temperature = {} ---", temp);

        let lattice = PBitLattice::roi_48(temp)
            .expect(&format!("Failed to create lattice at T={}", temp));

        let phi_calculator = HierarchicalPhiCalculator::default();
        let phi_result = phi_calculator
            .calculate(&lattice)
            .expect(&format!("Failed to calculate Φ at T={}", temp));

        let ci_calculator = CICalculator::new();
        let ci_result = ci_calculator
            .calculate(&lattice)
            .expect(&format!("Failed to calculate CI at T={}", temp));

        let entropy_calculator = EntropyCalculator::new();
        let entropy = entropy_calculator.entropy_from_pbits(&lattice);
        let negentropy = entropy_calculator.negentropy(entropy, lattice.size());
        let syntergy = CausalDensityEstimator::causal_density(&lattice);

        // All values should be valid
        assert!(phi_result.phi_total >= 0.0);
        assert!(phi_result.phi_total.is_finite());
        assert!(ci_result.ci >= 0.0);
        assert!(ci_result.ci.is_finite());
        assert!(negentropy >= 0.0);
        assert!(negentropy.is_finite());
        assert!(syntergy >= 0.0);
        assert!(syntergy.is_finite());

        println!("  Φ = {:.4}", phi_result.phi_total);
        println!("  CI = {:.4}", ci_result.ci);
        println!("  Negentropy = {:.4e}", negentropy);
        println!("  Syntergy = {:.4}", syntergy);

        // Sign the state
        let metrics = ConsciousnessMetrics {
            phi: phi_result.phi_total,
            ci: ci_result.ci,
            syntergy,
            negentropy,
        };

        let signing_key = vec![temp.to_bits() as u8; 32];
        let signed_state = SignedConsciousnessState::create_and_sign(
            metrics,
            &signing_key,
            None,
        )
        .expect(&format!("Failed to sign state at T={}", temp));

        assert!(signed_state.verify().unwrap());
        println!("  ✓ Signature valid");
    }

    println!("\n✓ Pipeline works across all temperature regimes");
}

#[test]
fn test_pipeline_clustering_methods() {
    // Test pipeline with different clustering methods
    let lattice = PBitLattice::roi_48(1.0).unwrap();

    let methods = vec![
        ClusteringMethod::Tessellation,
        ClusteringMethod::HyperbolicKMeans,
        ClusteringMethod::Agglomerative,
    ];

    let entropy_calculator = EntropyCalculator::new();
    let ci_calculator = CICalculator::new();

    for method in methods {
        println!("\n--- Testing clustering method: {:?} ---", method);

        let phi_calculator = HierarchicalPhiCalculator::new(3, method, 2.0, 3);
        let phi_result = phi_calculator
            .calculate(&lattice)
            .expect(&format!("Failed with method {:?}", method));

        assert!(phi_result.phi_total >= 0.0);
        assert!(phi_result.phi_total.is_finite());
        assert_eq!(phi_result.scales.len(), 3);
        assert_eq!(phi_result.phi_per_scale.len(), 3);

        println!("  Φ_total = {:.4}", phi_result.phi_total);
        println!("  Scales: {:?}", phi_result.scales);

        // Verify can be signed
        let ci_result = ci_calculator.calculate(&lattice).unwrap();
        let entropy = entropy_calculator.entropy_from_pbits(&lattice);
        let negentropy = entropy_calculator.negentropy(entropy, lattice.size());
        let syntergy = CausalDensityEstimator::causal_density(&lattice);

        let metrics = ConsciousnessMetrics {
            phi: phi_result.phi_total,
            ci: ci_result.ci,
            syntergy,
            negentropy,
        };

        let signing_key = vec![42u8; 32];
        let signed_state = SignedConsciousnessState::create_and_sign(
            metrics,
            &signing_key,
            None,
        )
        .unwrap();

        assert!(signed_state.verify().unwrap());
        println!("  ✓ State signed and verified");
    }

    println!("\n✓ All clustering methods work through pipeline");
}
