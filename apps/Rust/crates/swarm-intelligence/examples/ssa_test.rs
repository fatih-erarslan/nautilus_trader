//! Basic SSA test example

use swarm_intelligence::{SalpSwarmAlgorithm, SsaParameters, SsaVariant, ChainTopology};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Salp Swarm Algorithm implementation...");
    
    // Create SSA parameters with different variants
    let params = SsaParameters {
        population_size: 20,
        max_iterations: 100,
        variant: SsaVariant::Marine,
        chain_topology: ChainTopology::Adaptive,
        ..Default::default()
    };
    
    // Create SSA instance
    let ssa = SalpSwarmAlgorithm::new(params)?;
    
    println!("SSA Algorithm: {}", ssa.name());
    println!("Population size: {}", ssa.parameters().population_size);
    println!("Variant: {:?}", ssa.parameters().variant);
    println!("Chain topology: {:?}", ssa.parameters().chain_topology);
    
    println!("SSA implementation test completed successfully!");
    
    Ok(())
}