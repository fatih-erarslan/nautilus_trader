//! Benchmarks for hyperbolic lattice operations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantum_hive::*;

fn benchmark_lattice_creation(c: &mut Criterion) {
    c.bench_function("lattice_100_nodes", |b| {
        b.iter(|| {
            let _nodes = black_box(AutopoieticHive::create_hyperbolic_lattice(100));
        });
    });
}

criterion_group!(benches, benchmark_lattice_creation);
criterion_main!(benches);