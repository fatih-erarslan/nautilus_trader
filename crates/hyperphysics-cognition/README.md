# hyperphysics-cognition

Bio-Digital Isomorphic Cognition System implementing hyperbolic attention, self-referential loops, dream state consolidation, and hierarchical meta-learning.

## Features

- **Hyperbolic Attention** (H^11 Lorentz model)
  - Curvature-modulated focus (κ ∈ [0.1, 10.0])
  - Locus Coeruleus gain modulation
  - Attention bandwidth: BW = k / κ

- **Self-Referential Loop** (40Hz gamma rhythm)
  - 6-stage cognitive cycle: Perception → Cognition → Neocortex → Agency → Consciousness → Action
  - Message-based inter-phase communication
  - Strict 25ms period enforcement

- **Dream State Consolidation**
  - Episodic replay during low arousal (<0.3)
  - Memory consolidation (short-term → long-term)
  - Homeostatic plasticity maintenance

- **Bateson's Learning Levels**
  - Level 0: Proto-learning (reflexive)
  - Level I: Learning (conditioning)
  - Level II: Deutero-learning (meta-learning)
  - Level III: Paradigm shift (identity transformation)

- **Cortical Bus Integration**
  - Ultra-low-latency spike routing (<50ns)
  - Pattern memory (HNSW + LSH)
  - pBit fabric coordination

## Quick Start

```rust
use hyperphysics_cognition::prelude::*;

// Create cognition system
let config = CognitionConfig::default();
let mut cognition = CognitionSystem::new(config)?;

// Modulate attention via hyperbolic curvature
cognition.set_attention_curvature(5.0)?; // Narrow focus

// Process through self-referential loop
let perception = PerceptionInput::new(sensory_data);
let action = cognition.process_loop(perception).await?;

// Dream state consolidation
cognition.enter_dream_state()?;
cognition.consolidate_memories(replay_buffer).await?;
cognition.exit_dream_state()?;

// Meta-learning with Bateson Level II
cognition.update_learning_strategy(context)?;
```

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYPERPHYSICS COGNITION SYSTEM                        │
│                                                                         │
│  ┌────────────────────┐      ┌────────────────────┐                    │
│  │ Hyperbolic         │      │ Self-Referential   │                    │
│  │ Attention          │◄────►│ Loop Coordinator   │                    │
│  │ (H^11 curvature)   │      │ (40Hz gamma)       │                    │
│  └──────┬─────────────┘      └──────┬─────────────┘                    │
│         │                           │                                  │
│         ▼                           ▼                                  │
│  ┌────────────────────┐      ┌────────────────────┐                    │
│  │ Dream State        │      │ Bateson Learning   │                    │
│  │ Consolidation      │◄────►│ Levels (0-III)     │                    │
│  │ (Offline learning) │      │ (Meta-learning)    │                    │
│  └──────┬─────────────┘      └──────┬─────────────┘                    │
│         │                           │                                  │
│         └───────────┬───────────────┘                                  │
│                     ▼                                                  │
│              ┌─────────────┐                                           │
│              │ Cortical    │                                           │
│              │ Bus         │                                           │
│              │ Integration │                                           │
│              └─────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Scientific Grounding

Based on peer-reviewed research:

- **Hyperbolic Geometry**: Chami et al. (2020) "Hyperbolic Graph Neural Networks", NeurIPS
- **Attention Mechanism**: Vaswani et al. (2017) "Attention Is All You Need"
- **Gamma Synchronization**: Fries (2009) "Neuronal gamma-band synchronization"
- **Dream Consolidation**: Wilson & McNaughton (1994) "Reactivation of hippocampal ensemble memories"
- **Bateson's Learning**: Bateson (1972) "Steps to an Ecology of Mind"
- **Active Inference**: Friston (2010) "The free-energy principle: a unified brain theory?"
- **IIT**: Tononi (2004) "An information integration theory of consciousness"

## Performance Specifications

- **Loop Frequency**: 40Hz gamma rhythm (25ms period)
- **Attention Update**: <1ms hyperbolic curvature modulation
- **Message Routing**: <50ns via cortical bus
- **Dream Replay**: Configurable rate (default: 100 episodes/s)
- **Memory Overhead**: O(n) where n = replay buffer size

## Usage Examples

### Hyperbolic Attention

```rust
use hyperphysics_cognition::prelude::*;

let attention = HyperbolicAttention::new(1.0)?;

// Narrow focus (high curvature)
attention.set_curvature(5.0)?;
assert!(attention.bandwidth() < 5.0);

// Broad awareness (low curvature)
attention.set_curvature(0.5)?;
assert!(attention.bandwidth() > 15.0);

// Modulate with arousal
attention.modulate_arousal(ArousalLevel::new(0.9));
```

### Self-Referential Loop

```rust
use hyperphysics_cognition::prelude::*;

let loop_coord = SelfReferentialLoop::new(40.0)?;

// Run one complete cycle (25ms)
loop_coord.run_cycle().await?;

// Transition through phases
loop_coord.transition()?; // Perceiving → Cognizing
loop_coord.transition()?; // Cognizing → Deliberating
// ... continues through all 6 phases
```

### Dream State Consolidation

```rust
use hyperphysics_cognition::prelude::*;

let dream = DreamConsolidator::new(0.3)?;

// Add episodic memories
for experience in experiences {
    dream.add_episode(experience);
}

// Enter dream state when arousal drops
dream.update_arousal(ArousalLevel::new(0.2))?;

// Consolidate memories (replay and learn)
let metrics = dream.consolidate(batch_size).await?;
println!("Replayed {} episodes", metrics.episodes_replayed);
```

### Bateson's Learning Levels

```rust
use hyperphysics_cognition::prelude::*;

let learner = BatesonLearner::new()?;

// Start at Level 0 (Proto-learning)
assert_eq!(learner.level(), LearningLevel::ProtoLearning);

// Ascend to Level I (Learning)
learner.ascend()?;
learner.learn(stimulus, response, reward, context);

// Ascend to Level II (Deutero-learning)
learner.ascend()?;
// Now adapts learning rate based on context

// Ascend to Level III (Paradigm shift)
learner.ascend()?;
// Accumulates restructuring pressure for paradigm shifts
```

## Cargo Features

- `full` (default): All features enabled
- `attention`: Hyperbolic attention mechanism
- `loops`: Self-referential loop coordinator
- `dream`: Dream state consolidation
- `learning`: Bateson's learning levels
- `integration`: Cortical bus integration
- `cortical-bus`: Enable cortical bus dependency

## License

MIT OR Apache-2.0

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Citation

If you use this crate in academic work, please cite:

```bibtex
@software{hyperphysics_cognition,
  title = {HyperPhysics Cognition: Bio-Digital Isomorphic Cognition System},
  author = {HyperPhysics Team},
  year = {2025},
  url = {https://github.com/hyperphysics/hyperphysics}
}
```
