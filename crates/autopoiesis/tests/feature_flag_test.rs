//! Test file to verify feature flag configuration works correctly

#[test]
fn test_default_features() {
    // Default features should not include benchmarks or property-tests
    #[cfg(feature = "benchmarks")]
    panic!("benchmarks feature should not be enabled by default");
    
    #[cfg(feature = "property-tests")]
    panic!("property-tests feature should not be enabled by default");
    
    #[cfg(feature = "test-utils")]
    panic!("test-utils feature should not be enabled by default");
}

#[test]
#[cfg(feature = "full-tests")]
fn test_full_tests_feature() {
    // When full-tests is enabled, benchmarks and property-tests should be enabled
    #[cfg(not(feature = "benchmarks"))]
    panic!("benchmarks should be enabled with full-tests");
    
    #[cfg(not(feature = "property-tests"))]
    panic!("property-tests should be enabled with full-tests");
    
    #[cfg(not(feature = "test-utils"))]
    panic!("test-utils should be enabled with full-tests");
}

#[test]
#[cfg(feature = "benchmarks")]
fn test_benchmarks_feature() {
    // When benchmarks is enabled, test-utils should also be enabled
    #[cfg(not(feature = "test-utils"))]
    panic!("test-utils should be enabled with benchmarks");
}

#[test]
#[cfg(feature = "property-tests")]
fn test_property_tests_feature() {
    // When property-tests is enabled, test-utils should also be enabled
    #[cfg(not(feature = "test-utils"))]
    panic!("test-utils should be enabled with property-tests");
}