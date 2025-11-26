# AgentDB Integration Guide

Comprehensive guide to integrating AgentDB for self-learning capabilities.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
- [Vector Storage](#vector-storage)
- [Reinforcement Learning](#reinforcement-learning)
- [Memory Patterns](#memory-patterns)
- [Performance Optimization](#performance-optimization)
- [Advanced Features](#advanced-features)

---

## Introduction

AgentDB is a high-performance vector database with built-in reinforcement learning capabilities. It provides:

- **150x faster** similarity search with HNSW indexing
- **4-32x memory reduction** with quantization
- **9 RL algorithms** including Decision Transformer
- **Persistent memory** for self-learning systems
- **Experience replay** with pattern recognition

---

## Installation

```bash
npm install agentdb@latest
```

**Requirements**:
- Node.js >=18.0.0
- Write permissions for database directory

---

## Core Concepts

### Collections

Collections store related data (like tables in SQL):

```typescript
// Create collections for different data types
await db.store({
  collection: 'patterns',
  data: { features, outcome, label }
});

await db.store({
  collection: 'trajectories',
  data: { state, action, reward, nextState }
});

await db.store({
  collection: 'insights',
  data: { key, value, timestamp }
});
```

### Embeddings

Convert data to vectors for similarity search:

```typescript
// AgentDB auto-generates embeddings
await db.store({
  collection: 'patterns',
  data: {
    features: [0.1, 0.2, 0.3, 0.4],  // Your feature vector
    outcome: { success: true, value: 0.8 },
    label: 'profitable_pattern'
  }
});

// Or provide custom embeddings
await db.store({
  collection: 'patterns',
  data: { ... },
  embedding: customEmbedding // Pre-computed vector
});
```

### Trajectories

Store sequential experiences for RL:

```typescript
// Store experience for reinforcement learning
await db.storeTrajectory({
  state: currentFeatures,      // Current state vector
  action: decision,            // Action taken
  reward: outcome.value,       // Reward received
  nextState: nextFeatures,     // Resulting state
  done: false                  // Episode complete?
});
```

---

## Basic Usage

### Initialization

```typescript
import { AgentDB } from 'agentdb';

// Create database instance
const db = new AgentDB('./my-memory.db');

// Initialize with options
await db.initialize({
  indexType: 'hnsw',        // HNSW for fast search
  quantization: '8bit',     // Reduce memory usage
  dimension: 128            // Vector dimension
});
```

### Storing Data

```typescript
// Store single item
await db.store({
  collection: 'patterns',
  data: {
    id: 'pattern-1',
    features: [0.1, 0.2, 0.3],
    outcome: { success: true, value: 0.9 },
    label: 'uptrend',
    timestamp: Date.now()
  }
});

// Store batch
await db.storeBatch({
  collection: 'patterns',
  data: [
    { id: 'pattern-1', features: [...], outcome: {...} },
    { id: 'pattern-2', features: [...], outcome: {...} },
    { id: 'pattern-3', features: [...], outcome: {...} }
  ]
});
```

### Querying Data

```typescript
// Query by filter
const results = await db.query({
  collection: 'patterns',
  filter: { label: 'uptrend' },
  limit: 10
});

// Get by ID
const pattern = await db.get({
  collection: 'patterns',
  id: 'pattern-1'
});

// Count documents
const count = await db.count({
  collection: 'patterns',
  filter: { 'outcome.success': true }
});
```

### Similarity Search

```typescript
// Find similar patterns
const similar = await db.similaritySearch({
  collection: 'patterns',
  query: currentFeatures,  // Query vector
  k: 10,                   // Top 10 results
  filter: { label: 'uptrend' }  // Optional filter
});

// Results include similarity scores
similar.forEach(result => {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
  console.log('Data:', result.data);
});
```

### Cleanup

```typescript
// Close database connection
await db.close();

// Delete collection
await db.deleteCollection('old_patterns');

// Clear all data
await db.clear();
```

---

## Vector Storage

### HNSW Indexing

Hierarchical Navigable Small World graphs for fast similarity search:

```typescript
// Initialize with HNSW
await db.initialize({
  indexType: 'hnsw',
  M: 16,              // Connections per layer (higher = better recall)
  efConstruction: 200 // Build quality (higher = better quality)
});

// Build index after batch inserts
await db.storeBatch({
  collection: 'patterns',
  data: largeDataset
});

await db.buildIndex({
  collection: 'patterns',
  indexType: 'hnsw'
});

// Search with HNSW (150x faster)
const results = await db.similaritySearch({
  collection: 'patterns',
  query: currentFeatures,
  k: 10,
  ef: 100  // Search quality (higher = more accurate)
});
```

### Quantization

Reduce memory usage with minimal accuracy loss:

```typescript
// 8-bit quantization (8x memory reduction)
await db.initialize({
  quantization: '8bit'
});

// 4-bit quantization (16x memory reduction)
await db.initialize({
  quantization: '4bit'
});

// Compare memory usage
const stats = await db.getStats();
console.log('Memory usage:', stats.memoryUsage);
console.log('Quantization:', stats.quantization);
```

### Custom Embeddings

Use your own embedding function:

```typescript
import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function generateEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  });

  return response.data[0].embedding;
}

// Store with custom embedding
const embedding = await generateEmbedding('Bullish market pattern');

await db.store({
  collection: 'patterns',
  data: { text: 'Bullish market pattern', label: 'uptrend' },
  embedding
});
```

---

## Reinforcement Learning

### Supported Algorithms

AgentDB includes 9 RL algorithms:

1. **Decision Transformer** - Sequence modeling for RL
2. **Q-Learning** - Value-based learning
3. **SARSA** - On-policy TD learning
4. **Actor-Critic** - Policy gradient with baseline
5. **Deep Q-Network (DQN)** - Neural network Q-learning
6. **PPO** - Proximal Policy Optimization
7. **A3C** - Asynchronous Actor-Critic
8. **TD3** - Twin Delayed DDPG
9. **SAC** - Soft Actor-Critic

### Training RL Models

```typescript
// Store trajectories
for (const experience of experiences) {
  await db.storeTrajectory({
    state: experience.state,
    action: experience.action,
    reward: experience.reward,
    nextState: experience.nextState,
    done: experience.done
  });
}

// Train Decision Transformer
const model = await db.trainRL({
  algorithm: 'decision_transformer',
  collection: 'trajectories',
  config: {
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    contextLength: 20  // History window
  }
});

// Save model
await db.saveModel({
  name: 'trading_policy',
  model,
  metadata: {
    algorithm: 'decision_transformer',
    trainedAt: Date.now(),
    performance: metrics
  }
});
```

### Using Trained Models

```typescript
// Load model
const model = await db.loadModel('trading_policy');

// Make prediction
const action = await db.predict({
  model,
  state: currentFeatures,
  returnTarget: 1.0  // Desired cumulative return
});

console.log('Recommended action:', action);
```

### Q-Learning Example

```typescript
// Train Q-Learning agent
const model = await db.trainRL({
  algorithm: 'q_learning',
  collection: 'trajectories',
  config: {
    learningRate: 0.1,
    discountFactor: 0.99,
    epsilon: 0.1,        // Exploration rate
    epsilonDecay: 0.995
  }
});

// Get Q-values for state
const qValues = await db.predict({
  model,
  state: currentState
});

// Select best action
const bestAction = qValues.indexOf(Math.max(...qValues));
```

---

## Memory Patterns

### Experience Replay

Store and retrieve past experiences:

```typescript
export class ExperienceReplay {
  constructor(private db: AgentDB) {}

  async store(experience: Experience): Promise<void> {
    await this.db.storeTrajectory(experience);
  }

  async sample(batchSize: number): Promise<Experience[]> {
    // Random sampling
    const total = await this.db.count({ collection: 'trajectories' });
    const indices = Array.from({ length: batchSize }, () =>
      Math.floor(Math.random() * total)
    );

    const experiences = [];
    for (const index of indices) {
      const exp = await this.db.query({
        collection: 'trajectories',
        skip: index,
        limit: 1
      });
      experiences.push(exp[0]);
    }

    return experiences;
  }

  async sampleSimilar(
    state: number[],
    batchSize: number
  ): Promise<Experience[]> {
    // Sample similar experiences
    return await this.db.similaritySearch({
      collection: 'trajectories',
      query: state,
      k: batchSize
    });
  }
}
```

### Memory Distillation

Extract insights from trajectories:

```typescript
export class MemoryDistiller {
  async distill(trajectories: Trajectory[]): Promise<Insight[]> {
    // Group by outcome
    const successful = trajectories.filter(t => t.reward > 0);
    const failed = trajectories.filter(t => t.reward <= 0);

    // Extract common patterns
    const successPatterns = await this.findCommonPatterns(successful);
    const failurePatterns = await this.findCommonPatterns(failed);

    // Store insights
    const insights = [
      {
        type: 'success_patterns',
        patterns: successPatterns,
        confidence: successPatterns.length / successful.length
      },
      {
        type: 'failure_patterns',
        patterns: failurePatterns,
        confidence: failurePatterns.length / failed.length
      }
    ];

    for (const insight of insights) {
      await this.db.store({
        collection: 'insights',
        data: insight
      });
    }

    return insights;
  }

  private async findCommonPatterns(
    trajectories: Trajectory[]
  ): Promise<Pattern[]> {
    // Cluster similar states
    const clusters = await this.clusterStates(
      trajectories.map(t => t.state)
    );

    return clusters.map(cluster => ({
      centroid: cluster.centroid,
      frequency: cluster.members.length / trajectories.length,
      avgReward: cluster.avgReward
    }));
  }
}
```

### Pattern Recognition

Learn recurring patterns:

```typescript
export class PatternRecognizer {
  async learn(
    features: number[],
    outcome: Outcome,
    label?: string
  ): Promise<Pattern> {
    // Check if similar pattern exists
    const similar = await this.db.similaritySearch({
      collection: 'patterns',
      query: features,
      k: 1,
      threshold: 0.9  // High similarity threshold
    });

    if (similar.length > 0) {
      // Update existing pattern
      const pattern = similar[0];
      pattern.data.occurrences++;
      pattern.data.outcomes.push(outcome);

      await this.db.update({
        collection: 'patterns',
        id: pattern.id,
        data: pattern.data
      });

      return pattern.data;
    } else {
      // Create new pattern
      const pattern = {
        features,
        outcomes: [outcome],
        label: label || `pattern-${Date.now()}`,
        occurrences: 1,
        confidence: 1.0,
        createdAt: Date.now()
      };

      await this.db.store({
        collection: 'patterns',
        data: pattern
      });

      return pattern;
    }
  }

  async recognize(features: number[]): Promise<Pattern | null> {
    const similar = await this.db.similaritySearch({
      collection: 'patterns',
      query: features,
      k: 1,
      threshold: 0.8
    });

    if (similar.length > 0) {
      return similar[0].data;
    }

    return null;
  }
}
```

---

## Performance Optimization

### Batching

Batch operations for better performance:

```typescript
// ❌ BAD: Individual inserts
for (const pattern of patterns) {
  await db.store({
    collection: 'patterns',
    data: pattern
  });
}

// ✅ GOOD: Batch insert
await db.storeBatch({
  collection: 'patterns',
  data: patterns
});
```

### Indexing

Build indexes for faster queries:

```typescript
// Build HNSW index
await db.buildIndex({
  collection: 'patterns',
  indexType: 'hnsw',
  M: 16,
  efConstruction: 200
});

// Check index status
const stats = await db.getStats();
console.log('Index status:', stats.indexStatus);
```

### Caching

Cache frequently accessed data:

```typescript
export class CachedAgentDB {
  private cache = new Map<string, any>();
  private cacheSize = 1000;

  constructor(private db: AgentDB) {}

  async get(id: string): Promise<any> {
    const cacheKey = `get:${id}`;

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const result = await this.db.get({
      collection: 'patterns',
      id
    });

    this.cache.set(cacheKey, result);

    if (this.cache.size > this.cacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    return result;
  }
}
```

---

## Advanced Features

### Multi-Collection Queries

Query across multiple collections:

```typescript
// Query patterns and insights
const results = await Promise.all([
  db.query({ collection: 'patterns', filter: { label: 'uptrend' } }),
  db.query({ collection: 'insights', filter: { type: 'success_patterns' } })
]);

const [patterns, insights] = results;
```

### Custom Distance Metrics

Use custom similarity metrics:

```typescript
// Cosine similarity (default)
await db.initialize({
  distanceMetric: 'cosine'
});

// Euclidean distance
await db.initialize({
  distanceMetric: 'euclidean'
});

// Custom metric
await db.initialize({
  distanceMetric: (a, b) => {
    // Your custom distance function
    return customDistance(a, b);
  }
});
```

### Versioning

Track changes over time:

```typescript
// Store with version
await db.store({
  collection: 'patterns',
  data: {
    id: 'pattern-1',
    features: [...],
    version: 1,
    timestamp: Date.now()
  }
});

// Update creates new version
await db.store({
  collection: 'patterns',
  data: {
    id: 'pattern-1',
    features: [...],
    version: 2,
    timestamp: Date.now()
  }
});

// Query specific version
const v1 = await db.query({
  collection: 'patterns',
  filter: { id: 'pattern-1', version: 1 }
});
```

---

## Complete Example

```typescript
import { AgentDB } from 'agentdb';

export class SelfLearningTrader {
  private db: AgentDB;

  async initialize(): Promise<void> {
    // Initialize AgentDB
    this.db = new AgentDB('./trading-memory.db');
    await this.db.initialize({
      indexType: 'hnsw',
      quantization: '8bit',
      dimension: 128
    });

    // Load existing model if available
    try {
      this.model = await this.db.loadModel('trading_policy');
    } catch {
      console.log('No existing model found');
    }
  }

  async analyzeAndTrade(marketData: MarketData): Promise<Decision> {
    // Extract features
    const features = this.extractFeatures(marketData);

    // Recall similar past experiences
    const similar = await this.db.similaritySearch({
      collection: 'experiences',
      query: features,
      k: 5
    });

    // Get model prediction if available
    let action;
    if (this.model) {
      action = await this.db.predict({
        model: this.model,
        state: features,
        returnTarget: 1.0
      });
    } else {
      action = this.randomAction();
    }

    return { action, similar };
  }

  async learn(outcome: Outcome): Promise<void> {
    // Store experience
    await this.db.storeTrajectory({
      state: outcome.state,
      action: outcome.action,
      reward: outcome.reward,
      nextState: outcome.nextState,
      done: outcome.done
    });

    // Retrain every 100 experiences
    const count = await this.db.count({ collection: 'experiences' });

    if (count % 100 === 0) {
      console.log(`Retraining model with ${count} experiences...`);

      this.model = await this.db.trainRL({
        algorithm: 'decision_transformer',
        collection: 'experiences',
        config: {
          epochs: 50,
          batchSize: 32,
          learningRate: 0.001
        }
      });

      await this.db.saveModel({
        name: 'trading_policy',
        model: this.model
      });
    }
  }

  async close(): Promise<void> {
    await this.db.close();
  }
}
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Performance Optimization](./BEST_PRACTICES.md#performance-optimization)

---

Built with ❤️ by the Neural Trader team
