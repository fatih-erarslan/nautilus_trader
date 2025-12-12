//! Basic compilation test for GWO

use swarm_intelligence::algorithms::gwo::{GwoParameters, GwoVariant};

fn main() {
    println!("Testing basic GWO types...");
    
    // Test parameter creation
    let mut params = GwoParameters::default();
    params.variant = GwoVariant::Enhanced;
    println!("✅ GwoParameters created: {:?}", params.variant);
    
    // Test bounds
    let bounds = vec![(-5.0, 5.0); 3];
    println!("✅ Bounds created: {} dimensions", bounds.len());
    
    println!("Basic GWO test completed!");
}