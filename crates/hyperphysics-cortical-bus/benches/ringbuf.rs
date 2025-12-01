//! Ring buffer latency benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hyperphysics_cortical_bus::ringbuf::{SpscRingBuffer, MpscRingBuffer};

fn bench_spsc_push_pop(c: &mut Criterion) {
    let buf: SpscRingBuffer<u64, 4096> = SpscRingBuffer::new();

    c.bench_function("spsc_push", |b| {
        b.iter(|| {
            buf.push(black_box(42u64));
            buf.pop();
        })
    });
}

fn bench_spsc_batch(c: &mut Criterion) {
    let buf: SpscRingBuffer<u64, 4096> = SpscRingBuffer::new();
    let items: Vec<u64> = (0..1000).collect();
    let mut output = [0u64; 1000];

    c.bench_function("spsc_batch_1000", |b| {
        b.iter(|| {
            buf.push_batch(black_box(&items));
            buf.pop_batch(black_box(&mut output));
        })
    });
}

fn bench_mpsc_push(c: &mut Criterion) {
    let buf: MpscRingBuffer<u64, 4096> = MpscRingBuffer::new();

    c.bench_function("mpsc_push", |b| {
        b.iter(|| {
            buf.push(black_box(42u64));
            buf.pop();
        })
    });
}

fn bench_mpsc_try_push(c: &mut Criterion) {
    let buf: MpscRingBuffer<u64, 4096> = MpscRingBuffer::new();

    c.bench_function("mpsc_try_push", |b| {
        b.iter(|| {
            let _ = buf.try_push(black_box(42u64));
            buf.pop();
        })
    });
}

criterion_group!(benches, bench_spsc_push_pop, bench_spsc_batch, bench_mpsc_push, bench_mpsc_try_push);
criterion_main!(benches);
