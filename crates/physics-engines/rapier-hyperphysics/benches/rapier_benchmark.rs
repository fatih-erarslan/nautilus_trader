use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Vector3;
use rapier3d::prelude::*;
use rapier_hyperphysics::{
    MarketMapper, MarketParticipant, MarketState, ParticipantType, PhysicsSimulator,
    RapierHyperPhysicsAdapter, SimulatorConfig,
};

fn benchmark_physics_step(c: &mut Criterion) {
    let mut adapter = RapierHyperPhysicsAdapter::new();

    // Setup a complex scene with multiple bodies to simulate a busy market
    for i in 0..100 {
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector3::new(i as f32 * 0.1, i as f32 * 0.5, 0.0))
            .build();
        let rb_handle = adapter.rigid_bodies_mut().insert(rb);

        let collider = ColliderBuilder::ball(0.5).build();
        adapter.add_collider_with_parent(collider, rb_handle);
    }

    c.bench_function("rapier_physics_step_100_bodies", |b| {
        b.iter(|| {
            adapter.step();
        })
    });
}

fn benchmark_full_simulation_cycle(c: &mut Criterion) {
    let mut adapter = RapierHyperPhysicsAdapter::new();
    let mapper = MarketMapper::new();
    let simulator = PhysicsSimulator::with_config(SimulatorConfig {
        steps: 10, // Simulate 10 steps per cycle
        dt: 0.016,
        convergence_threshold: 0.001,
        external_forces: false,
    });

    // Create a realistic market state
    let market_state = MarketState {
        bids: (0..50).map(|i| (100.0 - i as f64 * 0.1, 10.0)).collect(),
        asks: (0..50).map(|i| (100.0 + i as f64 * 0.1, 10.0)).collect(),
        trades: vec![],
        participants: vec![
            MarketParticipant {
                participant_type: ParticipantType::Whale,
                capital: 1000000.0,
                position_size: 1000.0,
                aggressiveness: 0.5,
            },
            MarketParticipant {
                participant_type: ParticipantType::HFT,
                capital: 10000.0,
                position_size: 100.0,
                aggressiveness: 0.9,
            },
        ],
        mid_price: 100.0,
        volatility: 0.01,
    };

    c.bench_function("full_simulation_cycle", |b| {
        b.iter(|| {
            // Reset adapter for each iteration to keep state consistent-ish
            // In reality we might not reset, but for benchmarking the mapping cost + sim cost:
            adapter.reset();

            // Map market to physics
            let (rigid_bodies, colliders) = adapter.split_sets_mut();
            let _mapping = mapper
                .map_to_physics(black_box(&market_state), rigid_bodies, colliders)
                .unwrap();

            // Run simulation
            let _result = simulator.simulate(&mut adapter).unwrap();
        })
    });
}

criterion_group!(
    benches,
    benchmark_physics_step,
    benchmark_full_simulation_cycle
);
criterion_main!(benches);
