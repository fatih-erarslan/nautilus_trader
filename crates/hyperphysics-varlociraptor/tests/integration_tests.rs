//! Integration tests for HyperPhysics-Varlociraptor bridge

use hyperphysics_varlociraptor::*;

#[test]
fn test_module_imports() {
    // Verify all modules are accessible
    let _config = VariantCallConfig::default();
}

#[tokio::test]
#[ignore = "Requires varlociraptor installation"]
async fn test_varlociraptor_version() {
    let version = get_varlociraptor_version().await;
    assert!(version.is_ok(), "Should get varlociraptor version");
    println!("Varlociraptor version: {}", version.unwrap());
}

#[test]
#[ignore = "Requires varlociraptor installation"]
fn test_bridge_creation() {
    let bridge = VarlociraptorBridge::new();
    assert!(bridge.is_ok(), "Should create bridge");
}

#[test]
fn test_config_validation() {
    let mut config = VariantCallConfig::default();

    // Invalid VAF
    config.min_vaf = 1.5;
    assert!(config.validate().is_err());

    config.min_vaf = 0.05;
    assert!(config.validate().is_ok());
}

#[test]
fn test_hyperbolic_variant_space() {
    use hyperphysics_varlociraptor::HyperbolicVariantSpace;
    use hyperphysics_varlociraptor::vcf::VcfVariant;

    let space = HyperbolicVariantSpace::new(5);

    // Create test variant
    let line = "chr1\t12345\t.\tA\tG\t30.0\tPASS\tDP=50;AF=0.3\tGT:DP:AF\t0/1:50:0.3";
    let variant = VcfVariant::from_vcf_line(line).unwrap();

    // Test mapping
    let point = space.map_variant(&variant);
    assert!(point.is_ok(), "Should map variant to hyperbolic space");
}

#[test]
fn test_vcf_parsing() {
    use hyperphysics_varlociraptor::vcf::VcfVariant;

    let line = "chr1\t12345\trs123\tA\tG,T\t30.0\tPASS\tDP=50;AF=0.3,0.2\tGT:DP:AF\t0/1:50:0.3";
    let variant = VcfVariant::from_vcf_line(line).unwrap();

    assert_eq!(variant.chrom, "chr1");
    assert_eq!(variant.pos, 12345);
    assert_eq!(variant.id, Some("rs123".to_string()));
    assert_eq!(variant.ref_allele, "A");
    assert_eq!(variant.alt_alleles.len(), 2);
    assert_eq!(variant.qual, Some(30.0));

    // Test VAF extraction
    let vaf = variant.get_vaf(0);
    assert_eq!(vaf, Some(0.3));

    // Test depth extraction
    let depth = variant.get_depth(0);
    assert_eq!(depth, Some(50));
}

#[tokio::test]
async fn test_bayesian_optimization() {
    use hyperphysics_varlociraptor::optimization::BayesianParameterOptimizer;

    let mut optimizer = BayesianParameterOptimizer::new();

    // Simple objective function
    let objective = |params: &hyperphysics_varlociraptor::optimization::VariantCallingParameters| {
        params.prior_somatic * 100.0
    };

    let result = optimizer.optimize(objective, 5).await;
    assert!(result.is_ok());
}

#[test]
fn test_variant_clustering() {
    use hyperphysics_varlociraptor::HyperbolicVariantSpace;
    use hyperphysics_varlociraptor::vcf::VcfVariant;

    let space = HyperbolicVariantSpace::new(5);

    // Create test variants
    let variants: Vec<VcfVariant> = vec![
        "chr1\t100\t.\tA\tG\t30.0\tPASS\tDP=50\tGT:DP:AF\t0/1:50:0.3",
        "chr1\t200\t.\tC\tT\t40.0\tPASS\tDP=60\tGT:DP:AF\t0/1:60:0.4",
        "chr1\t300\t.\tG\tA\t35.0\tPASS\tDP=55\tGT:DP:AF\t0/1:55:0.35",
    ]
    .iter()
    .map(|line| VcfVariant::from_vcf_line(line).unwrap())
    .collect();

    // Test clustering
    let clusters = space.cluster_variants(&variants, 2);
    assert!(clusters.is_ok());
    assert_eq!(clusters.unwrap().len(), variants.len());
}

#[test]
fn test_variant_distance() {
    use hyperphysics_varlociraptor::HyperbolicVariantSpace;
    use hyperphysics_varlociraptor::vcf::VcfVariant;

    let space = HyperbolicVariantSpace::new(5);

    let v1 = VcfVariant::from_vcf_line(
        "chr1\t100\t.\tA\tG\t30.0\tPASS\tDP=50\tGT:DP:AF\t0/1:50:0.3"
    ).unwrap();

    let v2 = VcfVariant::from_vcf_line(
        "chr1\t200\t.\tC\tT\t40.0\tPASS\tDP=60\tGT:DP:AF\t0/1:60:0.4"
    ).unwrap();

    let distance = space.variant_distance(&v1, &v2);
    assert!(distance.is_ok());
    assert!(distance.unwrap() >= 0.0);
}
