// Quantum Agent Unification Benchmarks
// Copyright (c) 2025 TENGRI Trading Swarm

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use quantum_agent_unification::{
    QuantumAgentSystem, QuantumAgent, AgentNetwork, QuantumCommunication,
    UnificationEngine, QuantumCoherence, AgentSynchronization, QuantumEntanglement,
    CollectiveIntelligence, QuantumConsensus, AgentCoordinator
};
use std::time::Instant;

fn create_agent_network(size: usize) -> AgentNetwork {
    let mut network = AgentNetwork::new();
    for i in 0..size {
        let agent = QuantumAgent::new(format!("agent_{}", i), vec!["trading", "analysis", "risk"]);
        network.add_agent(agent);
    }
    network
}

fn benchmark_agent_unification(c: &mut Criterion) {
    let mut group = c.benchmark_group("agent_unification");
    
    for agent_count in [5, 10, 25, 50, 100].iter() {
        let network = create_agent_network(*agent_count);
        let engine = UnificationEngine::new();
        
        group.bench_with_input(BenchmarkId::new("unify_agents", agent_count), agent_count, |b, _| {
            b.iter(|| {
                engine.unify_agent_network(&network)
            })
        });
    }
    group.finish();
}

fn benchmark_quantum_communication(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_communication");
    
    let network = create_agent_network(20);
    let comm_system = QuantumCommunication::new();
    
    for message_size in [100, 500, 1000, 5000].iter() {
        let message = vec![0u8; *message_size];
        
        group.bench_with_input(BenchmarkId::new("quantum_broadcast", message_size), message_size, |b, _| {
            b.iter(|| {
                comm_system.quantum_broadcast(&network, &message)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("entangled_communication", message_size), message_size, |b, _| {
            b.iter(|| {
                comm_system.entangled_communication(&network, &message)
            })
        });
    }
    group.finish();
}

fn benchmark_quantum_coherence(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_coherence");
    
    for agent_count in [10, 25, 50].iter() {
        let network = create_agent_network(*agent_count);
        let coherence = QuantumCoherence::new();
        
        group.bench_with_input(BenchmarkId::new("maintain_coherence", agent_count), agent_count, |b, _| {
            b.iter(|| {
                coherence.maintain_quantum_coherence(&network)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("measure_coherence", agent_count), agent_count, |b, _| {
            b.iter(|| {
                coherence.measure_coherence_level(&network)
            })
        });
    }
    group.finish();
}

fn benchmark_agent_synchronization(c: &mut Criterion) {
    let mut group = c.benchmark_group("agent_synchronization");
    
    let network = create_agent_network(30);
    let sync_system = AgentSynchronization::new();
    
    group.bench_function("synchronize_states", |b| {
        b.iter(|| {
            sync_system.synchronize_agent_states(&network)
        })
    });
    
    group.bench_function("phase_alignment", |b| {
        b.iter(|| {
            sync_system.align_quantum_phases(&network)
        })
    });
    
    group.bench_function("temporal_sync", |b| {
        b.iter(|| {
            sync_system.temporal_synchronization(&network)
        })
    });
    
    group.finish();
}

fn benchmark_quantum_entanglement(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_entanglement");
    
    let network = create_agent_network(20);
    let entanglement = QuantumEntanglement::new();
    
    for pair_count in [5, 10, 15, 20].iter() {
        group.bench_with_input(BenchmarkId::new("create_entanglement", pair_count), pair_count, |b, &pairs| {
            b.iter(|| {
                entanglement.create_entangled_pairs(&network, pairs)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("measure_entanglement", pair_count), pair_count, |b, &pairs| {
            b.iter(|| {
                entanglement.measure_entanglement_strength(&network, pairs)
            })
        });
    }
    group.finish();
}

fn benchmark_collective_intelligence(c: &mut Criterion) {
    let mut group = c.benchmark_group("collective_intelligence");
    
    let network = create_agent_network(40);
    let intelligence = CollectiveIntelligence::new();
    
    group.bench_function("aggregate_knowledge", |b| {
        b.iter(|| {
            intelligence.aggregate_agent_knowledge(&network)
        })
    });
    
    group.bench_function("distributed_learning", |b| {
        b.iter(|| {
            intelligence.distributed_learning(&network)
        })
    });
    
    group.bench_function("emergent_behavior", |b| {
        b.iter(|| {
            intelligence.detect_emergent_behavior(&network)
        })
    });
    
    group.finish();
}

fn benchmark_quantum_consensus(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_consensus");
    
    for agent_count in [10, 20, 30, 50].iter() {
        let network = create_agent_network(*agent_count);
        let consensus = QuantumConsensus::new();
        
        group.bench_with_input(BenchmarkId::new("reach_consensus", agent_count), agent_count, |b, _| {
            b.iter(|| {
                consensus.reach_quantum_consensus(&network)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("voting_protocol", agent_count), agent_count, |b, _| {
            b.iter(|| {
                consensus.quantum_voting_protocol(&network)
            })
        });
    }
    group.finish();
}

fn benchmark_agent_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("agent_coordination");
    
    let network = create_agent_network(35);
    let coordinator = AgentCoordinator::new();
    
    group.bench_function("coordinate_tasks", |b| {
        b.iter(|| {
            coordinator.coordinate_agent_tasks(&network)
        })
    });
    
    group.bench_function("load_balancing", |b| {
        b.iter(|| {
            coordinator.balance_agent_workload(&network)
        })
    });
    
    group.bench_function("resource_allocation", |b| {
        b.iter(|| {
            coordinator.allocate_resources(&network)
        })
    });
    
    group.finish();
}

fn benchmark_quantum_system_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_scaling");
    
    let system = QuantumAgentSystem::new();
    
    for scale_factor in [1.0, 2.0, 5.0, 10.0].iter() {
        group.bench_with_input(BenchmarkId::new("scale_system", format!("{:.1}x", scale_factor)), scale_factor, |b, &factor| {
            b.iter(|| {
                system.scale_quantum_system(factor)
            })
        });
    }
    
    group.bench_function("dynamic_scaling", |b| {
        b.iter(|| {
            system.dynamic_scaling_adaptation()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_agent_unification,
    benchmark_quantum_communication,
    benchmark_quantum_coherence,
    benchmark_agent_synchronization,
    benchmark_quantum_entanglement,
    benchmark_collective_intelligence,
    benchmark_quantum_consensus,
    benchmark_agent_coordination,
    benchmark_quantum_system_scaling
);
criterion_main!(benches);