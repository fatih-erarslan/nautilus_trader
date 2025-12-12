//! Simple test to validate GWO compilation

use swarm_intelligence::algorithms::gwo::{GreyWolfOptimizer, GwoParameters, GwoVariant};
use nalgebra::DVector;

fn main() {
    println!("Testing GWO compilation...");
    
    // Test basic parameter creation
    let params = GwoParameters::default();
    println!("Default parameters created: {:?}", params.variant);
    
    // Test optimizer creation
    let bounds = vec![(-5.0, 5.0); 3];
    match GreyWolfOptimizer::new(params, bounds) {
        Ok(_) => println!("✅ GWO optimizer created successfully"),
        Err(e) => println!("❌ Failed to create GWO optimizer: {:?}", e),
    }
    
    println!("GWO compilation test completed!");
}