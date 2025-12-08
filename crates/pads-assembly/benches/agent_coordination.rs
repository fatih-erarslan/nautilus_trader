use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pads_assembly::coordination::AgentCoordinator;
use pads_assembly::assembly::PadsAssembly;
use pads_assembly::config::PadsConfig;

fn agent_coordination_benchmark(c: &mut Criterion) {
    let config = PadsConfig::default();
    let coordinator = AgentCoordinator::new(config);
    let agents = vec![1, 2, 3, 4, 5, 6, 7, 8];
    
    c.bench_function("agent_coordination", |b| {
        b.iter(|| {
            coordinator.coordinate_agents(black_box(&agents))
        })
    });
}

fn pads_assembly_benchmark(c: &mut Criterion) {
    let assembly = PadsAssembly::new();
    let components = vec![
        "component1".to_string(),
        "component2".to_string(),
        "component3".to_string(),
    ];
    
    c.bench_function("pads_assembly", |b| {
        b.iter(|| {
            assembly.assemble_components(black_box(&components))
        })
    });
}

criterion_group!(benches, agent_coordination_benchmark, pads_assembly_benchmark);
criterion_main!(benches);