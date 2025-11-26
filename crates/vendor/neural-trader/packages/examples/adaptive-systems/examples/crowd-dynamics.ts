/**
 * Crowd Dynamics using Ant Colony Optimization
 *
 * Demonstrates:
 * - Emergency evacuation pathfinding
 * - Crowd flow optimization
 * - Multiple exit strategies
 * - Bottleneck detection
 */

import { AntColonyOptimization, EmergenceDetector, type SystemState } from '../src';

async function runCrowdDynamicsSimulation() {
  console.log('👥 Crowd Dynamics Simulation with ACO\n');

  // Create building layout (grid-based)
  const width = 500;
  const height = 500;
  const spacing = 25;

  const aco = new AntColonyOptimization({
    numAnts: 50,
    alpha: 1.0,
    beta: 2.5,
    evaporationRate: 0.3,
    pheromoneDeposit: 100,
    maxIterations: 100
  });

  console.log('Creating building layout...');

  // Create grid of nodes
  const nodes: Array<{ id: string; x: number; y: number }> = [];

  for (let y = 0; y < height; y += spacing) {
    for (let x = 0; x < width; x += spacing) {
      const id = `node-${x}-${y}`;
      nodes.push({ id, x, y, type: 'normal' });
      aco.addNode({ id, x, y, type: 'normal' });
    }
  }

  console.log(`Created ${nodes.length} nodes`);

  // Add edges (4-connected grid with some walls)
  let edgeCount = 0;

  for (let y = 0; y < height; y += spacing) {
    for (let x = 0; x < width; x += spacing) {
      const currentId = `node-${x}-${y}`;

      // Create walls/obstacles
      const isWall = (
        // Horizontal walls with gaps (exits)
        (y === 250 && x > 100 && x < 200) ||
        (y === 250 && x > 300 && x < 400) ||
        // Vertical wall
        (x === 250 && y > 100 && y < 200)
      );

      if (isWall) continue;

      // Connect to right neighbor
      if (x + spacing < width) {
        const rightId = `node-${x + spacing}-${y}`;
        const isRightWall = (
          (y === 250 && x + spacing > 100 && x + spacing < 200) ||
          (y === 250 && x + spacing > 300 && x + spacing < 400) ||
          (x + spacing === 250 && y > 100 && y < 200)
        );

        if (!isRightWall) {
          aco.addEdge(currentId, rightId, true);
          edgeCount++;
        }
      }

      // Connect to bottom neighbor
      if (y + spacing < height) {
        const bottomId = `node-${x}-${y + spacing}`;
        const isBottomWall = (
          (y + spacing === 250 && x > 100 && x < 200) ||
          (y + spacing === 250 && x > 300 && x < 400) ||
          (x === 250 && y + spacing > 100 && y + spacing < 200)
        );

        if (!isBottomWall) {
          aco.addEdge(currentId, bottomId, true);
          edgeCount++;
        }
      }
    }
  }

  console.log(`Created ${edgeCount} edges`);

  // Define starting positions (crowd locations)
  const startPoints = [
    'node-100-100',
    'node-100-400',
    'node-400-100',
    'node-400-400'
  ];

  // Define exits
  const exits = [
    'node-0-250', // Left exit
    'node-500-250', // Right exit (won't exist, use closest)
    'node-250-0', // Top exit
    'node-250-500' // Bottom exit (won't exist, use closest)
  ].filter(id => nodes.some(n => n.id === id));

  console.log(`\nStarting points: ${startPoints.length}`);
  console.log(`Exit points: ${exits.length}`);

  // Initialize emergence detector
  const emergence = new EmergenceDetector(process.env.OPENAI_API_KEY);

  // Run pathfinding for each start-exit pair
  console.log('\n' + '='.repeat(60));
  console.log('Finding Optimal Evacuation Routes');
  console.log('='.repeat(60));

  const results: Array<{
    start: string;
    exit: string;
    path: string[];
    length: number;
    iterations: number;
  }> = [];

  for (const start of startPoints) {
    for (const exit of exits) {
      console.log(`\nFinding path: ${start} → ${exit}`);

      try {
        const result = await aco.optimize(start, exit);
        results.push({
          start,
          exit,
          path: result.path,
          length: result.length,
          iterations: result.iterations
        });

        console.log(`  ✅ Path found!`);
        console.log(`     Length: ${result.length.toFixed(2)}`);
        console.log(`     Steps: ${result.path.length}`);
        console.log(`     Iterations: ${result.iterations}`);

        // Add to emergence detector
        const edges = aco.getEdges();
        const pathEdges = result.path.slice(0, -1).map((nodeId, i) => {
          const nextNodeId = result.path[i + 1];
          const edgeKey = `${nodeId}->${nextNodeId}`;
          return edges.find(e => `${e.from}->${e.to}` === edgeKey);
        }).filter(e => e !== undefined);

        const avgPheromone = pathEdges.reduce((sum, e) => sum + (e?.pheromone || 0), 0) / pathEdges.length;

        const state: SystemState = {
          timestamp: Date.now(),
          agents: result.path.map((nodeId, i) => {
            const node = nodes.find(n => n.id === nodeId);
            return {
              id: nodeId,
              position: { x: node?.x || 0, y: node?.y || 0 },
              state: { step: i },
              neighbors: []
            };
          }),
          globalMetrics: {
            entropy: 1 - (result.iterations / 100),
            order: avgPheromone / 10,
            complexity: result.path.length / 50,
            connectivity: pathEdges.length / result.path.length
          }
        };

        await emergence.addState(state);
      } catch (error) {
        console.log(`  ❌ No path found`);
      }
    }

    // Clear for next start point to get fresh pheromone trails
    // (Comment out to see emergent preferred paths)
    // aco.clear();
  }

  // Analysis
  console.log('\n' + '='.repeat(60));
  console.log('Crowd Flow Analysis');
  console.log('='.repeat(60));

  // Find most efficient exits
  const exitStats = new Map<string, { totalLength: number; count: number }>();

  for (const result of results) {
    const stats = exitStats.get(result.exit) || { totalLength: 0, count: 0 };
    stats.totalLength += result.length;
    stats.count++;
    exitStats.set(result.exit, stats);
  }

  console.log('\nExit Efficiency Rankings:');
  const sortedExits = Array.from(exitStats.entries())
    .map(([exit, stats]) => ({
      exit,
      avgLength: stats.totalLength / stats.count,
      usage: stats.count
    }))
    .sort((a, b) => a.avgLength - b.avgLength);

  sortedExits.forEach((stat, i) => {
    console.log(`  ${i + 1}. ${stat.exit}`);
    console.log(`     Average path length: ${stat.avgLength.toFixed(2)}`);
    console.log(`     Usage count: ${stat.usage}`);
  });

  // Emergence metrics
  const metrics = emergence.getLatestMetrics();
  console.log('\nEmergence Metrics:');
  console.log(`  Self-Organization: ${(metrics.selfOrganization * 100).toFixed(1)}%`);
  console.log(`  Complexity: ${(metrics.complexity * 100).toFixed(1)}%`);
  console.log(`  Adaptability: ${(metrics.adaptability * 100).toFixed(1)}%`);

  // Identify bottlenecks
  console.log('\nBottleneck Analysis:');
  const edges = aco.getEdges();
  const sortedEdges = edges
    .sort((a, b) => b.pheromone - a.pheromone)
    .slice(0, 10);

  console.log('  Most traveled edges (potential bottlenecks):');
  sortedEdges.forEach((edge, i) => {
    console.log(`    ${i + 1}. ${edge.from} → ${edge.to}`);
    console.log(`       Pheromone level: ${edge.pheromone.toFixed(2)}`);
    console.log(`       Distance: ${edge.distance.toFixed(2)}`);
  });

  console.log('\n✅ Simulation complete!');
  console.log('\n💡 Insights:');
  console.log('  - Ants collectively found optimal evacuation routes');
  console.log('  - Pheromone trails indicate popular paths');
  console.log('  - High pheromone edges are potential bottlenecks');
  console.log('  - Multiple exits distribute crowd flow');
}

// Run if called directly
if (require.main === module) {
  runCrowdDynamicsSimulation().catch(console.error);
}

export { runCrowdDynamicsSimulation };
