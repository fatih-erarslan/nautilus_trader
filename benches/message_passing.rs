use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::mpsc;
use std::thread;

/// Benchmark message passing latency between threads
fn bench_message_passing(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_passing");

    // Benchmark standard mpsc channel
    group.bench_function("mpsc_channel", |b| {
        b.iter(|| {
            let (tx, rx) = mpsc::channel();
            let handle = thread::spawn(move || {
                rx.recv().unwrap()
            });
            tx.send(black_box(42)).unwrap();
            black_box(handle.join().unwrap());
        });
    });

    // Benchmark with different message sizes
    for size in [1, 64, 256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("mpsc_vec", size), size, |b, &size| {
            b.iter(|| {
                let (tx, rx) = mpsc::channel();
                let data: Vec<u8> = vec![0u8; size];
                let handle = thread::spawn(move || {
                    rx.recv().unwrap()
                });
                tx.send(black_box(data)).unwrap();
                black_box(handle.join().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark lock-free data structures
fn bench_lockfree(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut group = c.benchmark_group("lockfree");

    group.bench_function("atomic_increment", |b| {
        let counter = Arc::new(AtomicU64::new(0));
        b.iter(|| {
            counter.fetch_add(black_box(1), Ordering::SeqCst);
        });
    });

    group.bench_function("atomic_cas", |b| {
        let counter = Arc::new(AtomicU64::new(0));
        b.iter(|| {
            let mut current = counter.load(Ordering::SeqCst);
            loop {
                match counter.compare_exchange(
                    current,
                    current + 1,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => current = x,
                }
            }
        });
    });

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    group.bench_function("vec_allocation", |b| {
        b.iter(|| {
            let v: Vec<f64> = Vec::with_capacity(black_box(1000));
            black_box(v);
        });
    });

    group.bench_function("vec_push", |b| {
        b.iter(|| {
            let mut v = Vec::new();
            for i in 0..black_box(1000) {
                v.push(black_box(i as f64));
            }
            black_box(v);
        });
    });

    group.bench_function("vec_with_capacity", |b| {
        b.iter(|| {
            let mut v = Vec::with_capacity(black_box(1000));
            for i in 0..1000 {
                v.push(black_box(i as f64));
            }
            black_box(v);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_message_passing, bench_lockfree, bench_memory);
criterion_main!(benches);
