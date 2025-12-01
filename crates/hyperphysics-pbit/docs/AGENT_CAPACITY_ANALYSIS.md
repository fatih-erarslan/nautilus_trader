# pBit Agent Capacity Analysis

## Overview

This document analyzes the theoretical and measured capacity for concurrent agents
running on the scalable pBit substrate, based on benchmarks from the dual GPU
implementation (AMD RX 6800 XT + RX 5500 XT).

## Measured Performance

| Scale | Hardware | Throughput | ns/spin |
|-------|----------|------------|---------|
| 1K pBits | CPU | 27.7M/s | 36ns |
| 64K pBits | CPU | 23.5M/s | 42ns |
| 100K pBits | Dual GPU | 15.5M/s | 64ns |
| 1M pBits | Dual GPU | 87.1M/s | 11.5ns |

Key insight: GPU throughput **improves** with scale due to better occupancy.

## Agent Configurations

| Config | pBits/Agent | Sweeps/Decision | Use Case |
|--------|-------------|-----------------|----------|
| Minimal | 64 | 5 | Simple pattern detector |
| Standard | 256 | 10 | Trading signal generator |
| Complex | 1,024 | 20 | Full strategy with memory |

## Capacity Calculations

### Formula

```
Agents_realtime = Throughput / (pBits_per_agent × sweeps_per_decision × update_rate_hz)
```

### Dual GPU (Current Hardware)

**Compute-Bound (Real-Time @ 100Hz)**

| Agent Type | pBits | Sweeps | Decisions/sec | Concurrent Agents |
|------------|-------|--------|---------------|-------------------|
| Minimal | 64 | 5 | 272,187 | **2,721** |
| Standard | 256 | 10 | 34,023 | **340** |
| Complex | 1,024 | 20 | 4,252 | **42** |

**Memory-Bound (Batch Mode)**

- Total VRAM: 24GB (16GB + 8GB)
- Memory per pBit: ~90 bytes (state + avg 10 couplings)
- Max pBits: 266 million

| Agent Type | pBits | Max Agents |
|------------|-------|------------|
| Minimal | 64 | 4,160,000 |
| Standard | 256 | 1,040,000 |
| Complex | 1,024 | 260,000 |

## Scaling Projections

### Horizontal Scaling (More GPUs)

| GPUs | VRAM | Est. Throughput | Standard Agents (RT) |
|------|------|-----------------|----------------------|
| 2 (current) | 24GB | 87M/s | 340 |
| 4 | 64GB | 300M/s | 1,170 |
| 8 | 128GB | 600M/s | 2,340 |
| 16 | 256GB | 1.2B/s | 4,680 |

### Optimization Potential

Current GPU utilization is suboptimal. With full optimization:

| Optimization | Speedup | New Throughput | Agents (RT) |
|--------------|---------|----------------|-------------|
| Baseline | 1× | 87M/s | 340 |
| Async dispatch | 2× | 174M/s | 680 |
| Memory coalescing | 1.5× | 261M/s | 1,020 |
| Shared memory | 2× | 522M/s | 2,040 |
| **Combined** | **6×** | **522M/s** | **2,040** |

## Trade-offs

### Real-Time vs Batch

```
┌─────────────────────────────────────────────────────────────────┐
│                     OPERATING MODES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  REAL-TIME (100Hz)                BATCH (async)                 │
│  ─────────────────                ─────────────                  │
│  • 340 agents                     • 1M+ agents                  │
│  • 10ms decision budget           • No latency constraint       │
│  • Market-making, HFT             • Strategy optimization       │
│  • GPU memory resident            • Stream from disk            │
│                                                                  │
│  HYBRID                                                          │
│  ──────                                                          │
│  • 100 RT agents + 10K batch agents                             │
│  • RT gets priority GPU time                                    │
│  • Batch fills idle cycles                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Precision vs Throughput

| Precision | Sweeps | Agents (RT) | Quality |
|-----------|--------|-------------|---------|
| Fast | 3 | 1,130 | Lower accuracy |
| Normal | 10 | 340 | Good balance |
| High | 30 | 113 | Best accuracy |
| Ultra | 100 | 34 | Research-grade |

## Recommendations

### For Trading Systems

1. **Market Making (100+ symbols)**: Use Minimal agents (64 pBits)
   - Capacity: ~2,700 agents @ 100Hz
   - Each agent handles one symbol

2. **Strategy Execution (10-50 strategies)**: Use Standard agents (256 pBits)
   - Capacity: ~340 agents @ 100Hz
   - Rich pattern recognition

3. **Portfolio Optimization (1-5 portfolios)**: Use Complex agents (1,024 pBits)
   - Capacity: ~42 agents @ 100Hz
   - Full market memory

### For Research/Backtesting

- Use batch mode with 100K+ agents
- 34K decisions/second throughput
- 1M agent population in VRAM

## Future Scaling

### Target: 10,000 Real-Time Agents

Required improvements:
1. **10× GPU throughput**: ~870M spins/sec
2. **Achieved via**:
   - 4-GPU cluster (4×)
   - Optimization (2.5×)

### Target: 1M Batch Agents

Already achievable with current hardware in memory.
Throughput: 34,000 decisions/second across population.

## Appendix: Benchmark Commands

```bash
# CPU stress test
cargo run -p hyperphysics-pbit --example stress_test --release

# Dual GPU benchmark
cargo run -p hyperphysics-gpu-unified --example dual_gpu_pbit --release

# MAX-CUT solver
cargo run -p hyperphysics-pbit --example dual_gpu_solver --release
```

## References

- Camsari et al. (2017) "Stochastic p-bits for invertible logic"
- Kaiser & Datta (2021) "Probabilistic computing with p-bits"
- Metropolis et al. (1953) "Equation of state calculations"
