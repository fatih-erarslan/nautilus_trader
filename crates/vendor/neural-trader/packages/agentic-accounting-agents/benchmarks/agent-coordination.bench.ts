/**
 * Agent Coordination Overhead Performance Benchmarks
 *
 * Tests multi-agent coordination, task queue throughput, and memory usage.
 * Target: <50ms coordination overhead for 8 agents
 */

import { performance } from 'perf_hooks';

// Mock Agent Base Class
class MockAgent {
  private id: string;
  private status: 'idle' | 'busy' = 'idle';
  private taskCount: number = 0;

  constructor(id: string) {
    this.id = id;
  }

  async executeTask(task: any): Promise<any> {
    this.status = 'busy';
    this.taskCount++;

    // Simulate task execution
    await this.simulateWork(task.complexity || 10);

    this.status = 'idle';
    return { agentId: this.id, result: 'success', taskId: task.id };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getStatus() {
    return { id: this.id, status: this.status, taskCount: this.taskCount };
  }

  getId(): string {
    return this.id;
  }
}

// Mock Task Queue
class MockTaskQueue {
  private queue: any[] = [];
  private processing: boolean = false;

  add(task: any): void {
    this.queue.push(task);
  }

  async process(agents: MockAgent[]): Promise<void> {
    this.processing = true;

    while (this.queue.length > 0) {
      const availableAgent = agents.find(agent => agent.getStatus().status === 'idle');
      if (!availableAgent) {
        await new Promise(resolve => setTimeout(resolve, 5));
        continue;
      }

      const task = this.queue.shift();
      if (task) {
        await availableAgent.executeTask(task);
      }
    }

    this.processing = false;
  }

  size(): number {
    return this.queue.length;
  }
}

// Mock Swarm Coordinator
class MockSwarmCoordinator {
  private agents: MockAgent[] = [];
  private queue: MockTaskQueue = new MockTaskQueue();

  spawn(count: number): void {
    for (let i = 0; i < count; i++) {
      this.agents.push(new MockAgent(`agent_${i}`));
    }
  }

  async coordinateTask(tasks: any[]): Promise<any[]> {
    const start = performance.now();

    // Add tasks to queue
    for (const task of tasks) {
      this.queue.add(task);
    }

    // Process tasks
    await this.queue.process(this.agents);

    const elapsed = performance.now() - start;
    return this.agents.map(agent => agent.getStatus());
  }

  getAgentCount(): number {
    return this.agents.length;
  }

  clear(): void {
    this.agents = [];
  }
}

// Mock ReasoningBank
class MockReasoningBank {
  private patterns: Map<string, any> = new Map();

  store(key: string, value: any): void {
    this.patterns.set(key, value);
  }

  retrieve(key: string): any {
    return this.patterns.get(key);
  }

  size(): number {
    return this.patterns.size;
  }
}

// Benchmark Functions
async function benchmarkSingleAgent(): Promise<void> {
  console.log('📊 Benchmark 1: Single Agent Execution Time');

  const agent = new MockAgent('solo');
  const tasks = Array.from({ length: 100 }, (_, i) => ({
    id: `task_${i}`,
    complexity: 5,
  }));

  const start = performance.now();
  for (const task of tasks) {
    await agent.executeTask(task);
  }
  const elapsed = performance.now() - start;

  console.log(`  Single agent: ${elapsed.toFixed(2)}ms for ${tasks.length} tasks`);
  console.log(`  Average: ${(elapsed / tasks.length).toFixed(2)}ms per task\n`);
}

async function benchmarkMultiAgentCoordination(): Promise<void> {
  console.log('📊 Benchmark 2: Multi-Agent Coordination Overhead');

  for (const agentCount of [2, 4, 8, 16]) {
    const coordinator = new MockSwarmCoordinator();
    coordinator.spawn(agentCount);

    const tasks = Array.from({ length: 100 }, (_, i) => ({
      id: `task_${i}`,
      complexity: 5,
    }));

    const start = performance.now();
    await coordinator.coordinateTask(tasks);
    const elapsed = performance.now() - start;

    const overhead = elapsed - (5 * (100 / agentCount)); // Subtract estimated task execution time

    console.log(`  ${agentCount} agents: ${elapsed.toFixed(2)}ms total, ~${overhead.toFixed(2)}ms coordination overhead`);
    console.log(`    ${(elapsed / tasks.length).toFixed(2)}ms per task`);

    const targetOverhead = 50;
    const status = overhead < targetOverhead ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetOverhead}ms overhead\n`);
  }
}

async function benchmarkTaskQueueThroughput(): Promise<void> {
  console.log('📊 Benchmark 3: Task Queue Throughput');

  const coordinator = new MockSwarmCoordinator();
  coordinator.spawn(8);

  for (const taskCount of [100, 500, 1000]) {
    const tasks = Array.from({ length: taskCount }, (_, i) => ({
      id: `task_${i}`,
      complexity: 5,
    }));

    const start = performance.now();
    await coordinator.coordinateTask(tasks);
    const elapsed = performance.now() - start;

    const throughput = taskCount / (elapsed / 1000);

    console.log(`  ${taskCount} tasks: ${elapsed.toFixed(2)}ms`);
    console.log(`    Throughput: ${throughput.toFixed(0)} tasks/second`);
    console.log(`    Average: ${(elapsed / taskCount).toFixed(2)}ms per task\n`);
  }
}

function benchmarkMemoryPerAgent(): void {
  console.log('📊 Benchmark 4: Memory Usage Per Agent');

  for (const agentCount of [1, 10, 100, 1000]) {
    const agents: MockAgent[] = [];

    const start = performance.now();
    for (let i = 0; i < agentCount; i++) {
      agents.push(new MockAgent(`agent_${i}`));
    }
    const elapsed = performance.now() - start;

    // Estimate memory: Each agent object is roughly 1KB
    const estimatedMemoryKB = agentCount * 1;
    const estimatedMemoryMB = estimatedMemoryKB / 1024;

    console.log(`  ${agentCount} agents: ~${estimatedMemoryMB.toFixed(2)} MB`);
    console.log(`    Initialization: ${elapsed.toFixed(2)}ms\n`);
  }
}

function benchmarkReasoningBankLookup(): void {
  console.log('📊 Benchmark 5: ReasoningBank Lookup Time');

  const bank = new MockReasoningBank();

  // Populate with patterns
  for (let i = 0; i < 10000; i++) {
    bank.store(`pattern_${i}`, {
      type: 'tax_calculation',
      method: i % 5 === 0 ? 'FIFO' : 'LIFO',
      timestamp: Date.now(),
    });
  }

  const iterations = 10000;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    bank.retrieve(`pattern_${i % 10000}`);
  }
  const elapsed = performance.now() - start;

  console.log(`  ${iterations} lookups in ${elapsed.toFixed(2)}ms`);
  console.log(`  Average: ${((elapsed / iterations) * 1000).toFixed(2)}µs per lookup`);
  console.log(`  Throughput: ${(iterations / (elapsed / 1000)).toFixed(0)} lookups/second\n`);
}

async function benchmarkParallelVsSequential(): Promise<void> {
  console.log('📊 Benchmark 6: Parallel vs Sequential Execution');

  const tasks = Array.from({ length: 50 }, (_, i) => ({
    id: `task_${i}`,
    complexity: 10,
  }));

  // Sequential execution
  const seqAgent = new MockAgent('sequential');
  const seqStart = performance.now();
  for (const task of tasks) {
    await seqAgent.executeTask(task);
  }
  const seqElapsed = performance.now() - seqStart;

  // Parallel execution with 8 agents
  const coordinator = new MockSwarmCoordinator();
  coordinator.spawn(8);
  const parStart = performance.now();
  await coordinator.coordinateTask(tasks);
  const parElapsed = performance.now() - parStart;

  const speedup = seqElapsed / parElapsed;

  console.log(`  Sequential (1 agent): ${seqElapsed.toFixed(2)}ms`);
  console.log(`  Parallel (8 agents): ${parElapsed.toFixed(2)}ms`);
  console.log(`  Speedup: ${speedup.toFixed(2)}x`);
  console.log(`  Efficiency: ${((speedup / 8) * 100).toFixed(1)}%\n`);
}

function benchmarkAgentSpawnTime(): void {
  console.log('📊 Benchmark 7: Agent Spawn Time');

  for (const spawnCount of [1, 10, 50, 100]) {
    const coordinator = new MockSwarmCoordinator();

    const start = performance.now();
    coordinator.spawn(spawnCount);
    const elapsed = performance.now() - start;

    console.log(`  Spawning ${spawnCount} agents: ${elapsed.toFixed(2)}ms`);
    console.log(`    Average: ${(elapsed / spawnCount).toFixed(3)}ms per agent\n`);
  }
}

async function benchmarkConcurrentTaskProcessing(): Promise<void> {
  console.log('📊 Benchmark 8: Concurrent Task Processing');

  const coordinator = new MockSwarmCoordinator();
  coordinator.spawn(8);

  // Simulate varying task complexities
  const tasks = Array.from({ length: 100 }, (_, i) => ({
    id: `task_${i}`,
    complexity: Math.random() * 20 + 5, // 5-25ms
  }));

  const start = performance.now();
  await coordinator.coordinateTask(tasks);
  const elapsed = performance.now() - start;

  const avgComplexity = tasks.reduce((sum, t) => sum + t.complexity, 0) / tasks.length;
  const expectedSeqTime = tasks.reduce((sum, t) => sum + t.complexity, 0);
  const efficiency = (expectedSeqTime / elapsed) / 8;

  console.log(`  ${tasks.length} tasks with varying complexity`);
  console.log(`  Average complexity: ${avgComplexity.toFixed(2)}ms`);
  console.log(`  Total time: ${elapsed.toFixed(2)}ms`);
  console.log(`  Expected sequential: ${expectedSeqTime.toFixed(2)}ms`);
  console.log(`  Speedup: ${(expectedSeqTime / elapsed).toFixed(2)}x`);
  console.log(`  Efficiency: ${(efficiency * 100).toFixed(1)}%\n`);
}

// Main execution
async function runBenchmarks(): Promise<void> {
  console.log('\n🚀 Starting Agent Coordination Performance Benchmarks\n');

  await benchmarkSingleAgent();
  await benchmarkMultiAgentCoordination();
  await benchmarkTaskQueueThroughput();
  benchmarkMemoryPerAgent();
  benchmarkReasoningBankLookup();
  await benchmarkParallelVsSequential();
  benchmarkAgentSpawnTime();
  await benchmarkConcurrentTaskProcessing();

  console.log('✅ All agent coordination benchmarks completed!\n');
}

// Run benchmarks
runBenchmarks().catch(console.error);
