/**
 * AgentDB Vector Operations Performance Benchmarks
 *
 * Tests vector similarity search, fraud pattern detection, and embedding generation.
 * Target: <100µs per query on 10K vectors
 */

import Benchmark from 'benchmark';
import { performance } from 'perf_hooks';

// Mock AgentDB functionality for benchmarking
class MockAgentDB {
  private vectors: Map<string, Float32Array> = new Map();
  private metadata: Map<string, Record<string, any>> = new Map();

  insert(id: string, vector: Float32Array, meta: Record<string, any>): void {
    this.vectors.set(id, vector);
    this.metadata.set(id, meta);
  }

  search(query: Float32Array, topK: number = 10): Array<{ id: string; score: number }> {
    const results: Array<{ id: string; score: number }> = [];

    for (const [id, vector] of this.vectors.entries()) {
      const score = this.cosineSimilarity(query, vector);
      results.push({ id, score });
    }

    // Sort by score descending and return top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  count(): number {
    return this.vectors.size;
  }

  clear(): void {
    this.vectors.clear();
    this.metadata.clear();
  }
}

// Helper functions
function generateRandomVector(dimensions: number): Float32Array {
  const vector = new Float32Array(dimensions);
  for (let i = 0; i < dimensions; i++) {
    vector[i] = Math.random() * 2 - 1; // Range [-1, 1]
  }
  return vector;
}

function generateTransactionEmbedding(tx: any): Float32Array {
  // Mock embedding generation from transaction data
  const dimensions = 128;
  const vector = new Float32Array(dimensions);

  // Simple hash-based embedding for benchmarking
  const hash = tx.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  for (let i = 0; i < dimensions; i++) {
    vector[i] = Math.sin(hash * i) * Math.cos(tx.amount * i);
  }

  return vector;
}

function populateDatabase(db: MockAgentDB, count: number, dimensions: number): void {
  console.log(`Populating database with ${count} vectors (${dimensions}D)...`);
  const start = performance.now();

  for (let i = 0; i < count; i++) {
    const vector = generateRandomVector(dimensions);
    db.insert(`vec_${i}`, vector, {
      type: i % 3 === 0 ? 'fraud' : 'normal',
      timestamp: Date.now() - i * 1000,
      amount: Math.random() * 10000,
    });
  }

  const elapsed = performance.now() - start;
  console.log(`✅ Populated ${count} vectors in ${elapsed.toFixed(2)}ms`);
}

// Benchmark Suites
async function runBenchmarks(): Promise<void> {
  console.log('\n🚀 Starting AgentDB Vector Operations Benchmarks\n');

  // Benchmark 1: Vector Similarity Search (varying dataset sizes)
  console.log('📊 Benchmark 1: Vector Similarity Search');
  for (const vectorCount of [1000, 10000, 100000]) {
    const db = new MockAgentDB();
    const dimensions = 128;
    populateDatabase(db, vectorCount, dimensions);

    const query = generateRandomVector(dimensions);
    const iterations = 100;

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      db.search(query, 10);
    }
    const elapsed = performance.now() - start;
    const avgTime = elapsed / iterations;

    console.log(`  ${vectorCount} vectors: ${avgTime.toFixed(2)}ms per query (${(avgTime * 1000).toFixed(2)}µs)`);
    console.log(`    ${iterations} iterations in ${elapsed.toFixed(2)}ms total`);

    const targetMicros = 100;
    const status = avgTime * 1000 < targetMicros ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMicros}µs per query\n`);
  }

  // Benchmark 2: Fraud Pattern Detection Queries
  console.log('📊 Benchmark 2: Fraud Pattern Detection');
  const fraudDB = new MockAgentDB();
  const dimensions = 128;
  populateDatabase(fraudDB, 10000, dimensions);

  // Create fraud pattern vector
  const fraudPattern = generateRandomVector(dimensions);

  const iterations = 100;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fraudDB.search(fraudPattern, 50); // Top 50 potential fraud cases
  }
  const elapsed = performance.now() - start;
  const avgTime = elapsed / iterations;

  console.log(`  Average fraud detection query: ${avgTime.toFixed(2)}ms (${(avgTime * 1000).toFixed(2)}µs)`);
  console.log(`  ${iterations} iterations in ${elapsed.toFixed(2)}ms total`);

  const targetMicros = 100;
  const status = avgTime * 1000 < targetMicros ? '✅ PASS' : '❌ FAIL';
  console.log(`  ${status} Target: <${targetMicros}µs per query\n`);

  // Benchmark 3: Transaction Embedding Generation
  console.log('📊 Benchmark 3: Transaction Embedding Generation');
  const transactions = Array.from({ length: 1000 }, (_, i) => ({
    id: `tx_${i}`,
    amount: Math.random() * 10000,
    timestamp: Date.now() - i * 1000,
    type: i % 2 === 0 ? 'buy' : 'sell',
  }));

  const embedStart = performance.now();
  for (const tx of transactions) {
    generateTransactionEmbedding(tx);
  }
  const embedElapsed = performance.now() - embedStart;
  const embedAvg = embedElapsed / transactions.length;

  console.log(`  Average embedding generation: ${embedAvg.toFixed(3)}ms per transaction`);
  console.log(`  ${transactions.length} embeddings in ${embedElapsed.toFixed(2)}ms total`);
  console.log(`  Throughput: ${(transactions.length / (embedElapsed / 1000)).toFixed(0)} embeddings/second\n`);

  // Benchmark 4: Batch Insert Performance
  console.log('📊 Benchmark 4: Batch Insert Performance');
  for (const batchSize of [100, 1000, 10000]) {
    const batchDB = new MockAgentDB();
    const vectors = Array.from({ length: batchSize }, (_, i) => ({
      id: `vec_${i}`,
      vector: generateRandomVector(dimensions),
      metadata: { index: i },
    }));

    const insertStart = performance.now();
    for (const { id, vector, metadata } of vectors) {
      batchDB.insert(id, vector, metadata);
    }
    const insertElapsed = performance.now() - insertStart;

    console.log(`  ${batchSize} vectors: ${insertElapsed.toFixed(2)}ms total (${(insertElapsed / batchSize).toFixed(3)}ms per insert)`);
    console.log(`    Throughput: ${(batchSize / (insertElapsed / 1000)).toFixed(0)} inserts/second\n`);
  }

  // Benchmark 5: Query Throughput (Concurrent-like)
  console.log('📊 Benchmark 5: Query Throughput');
  const throughputDB = new MockAgentDB();
  populateDatabase(throughputDB, 10000, dimensions);

  const queries = Array.from({ length: 1000 }, () => generateRandomVector(dimensions));

  const throughputStart = performance.now();
  for (const query of queries) {
    throughputDB.search(query, 10);
  }
  const throughputElapsed = performance.now() - throughputStart;
  const qps = queries.length / (throughputElapsed / 1000);

  console.log(`  ${queries.length} queries in ${throughputElapsed.toFixed(2)}ms`);
  console.log(`  Throughput: ${qps.toFixed(0)} queries/second`);
  console.log(`  Average: ${(throughputElapsed / queries.length).toFixed(2)}ms per query\n`);

  // Benchmark 6: Memory Usage Estimation
  console.log('📊 Benchmark 6: Memory Usage Estimation');
  const memoryDB = new MockAgentDB();

  const vectorCounts = [1000, 10000, 100000];
  for (const count of vectorCounts) {
    memoryDB.clear();
    populateDatabase(memoryDB, count, dimensions);

    // Estimate memory: each float32 is 4 bytes
    const vectorMemory = count * dimensions * 4;
    const vectorMemoryMB = vectorMemory / (1024 * 1024);

    console.log(`  ${count} vectors: ~${vectorMemoryMB.toFixed(2)} MB (vector data only)`);
  }

  console.log('\n✅ All benchmarks completed!\n');
}

// Run benchmarks
runBenchmarks().catch(console.error);
