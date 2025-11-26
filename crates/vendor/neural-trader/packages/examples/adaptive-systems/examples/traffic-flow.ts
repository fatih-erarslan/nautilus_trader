/**
 * Traffic Flow Simulation using Boids
 *
 * Demonstrates:
 * - Lane formation in bidirectional traffic
 * - Congestion emergence
 * - Self-organizing traffic patterns
 * - Obstacle avoidance
 */

import { BoidsSimulation, EmergenceDetector, type SystemState, type Vector2D } from '../src';

async function runTrafficSimulation() {
  console.log('🚗 Traffic Flow Simulation with Self-Organizing Behavior\n');

  // Simulation parameters
  const width = 1000;
  const height = 200;
  const numVehicles = 100;
  const simulationSteps = 500;

  // Create simulation with traffic-like parameters
  const traffic = new BoidsSimulation(
    { width, height },
    {
      separationWeight: 2.0, // Strong separation (collision avoidance)
      alignmentWeight: 1.5, // Moderate alignment (follow traffic flow)
      cohesionWeight: 0.5, // Weak cohesion (maintain spacing)
      separationRadius: 30,
      alignmentRadius: 50,
      cohesionRadius: 40,
      maxSpeed: 5,
      maxForce: 0.2,
      boundaryBehavior: 'wrap'
    }
  );

  // Initialize emergence detector
  const emergence = new EmergenceDetector(process.env.OPENAI_API_KEY);

  // Create vehicles (half going left, half going right)
  console.log(`Initializing ${numVehicles} vehicles...`);
  for (let i = 0; i < numVehicles; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;

    // Create two opposing flows
    const direction = i < numVehicles / 2 ? 1 : -1;
    const velocity: Vector2D = {
      x: direction * (3 + Math.random() * 2),
      y: (Math.random() - 0.5) * 0.5
    };

    traffic.addBoid(`vehicle-${i}`, { x, y }, velocity);
  }

  // Run simulation
  console.log('\nRunning simulation...');
  const snapshotInterval = 50;
  let laneFormationDetected = false;
  let congestionDetected = false;

  for (let step = 0; step < simulationSteps; step++) {
    await traffic.update();

    // Take periodic snapshots for emergence detection
    if (step % snapshotInterval === 0) {
      const boids = traffic.getBoids();

      // Analyze lane formation
      const lanes = analyzeLaneFormation(boids, height);

      // Calculate metrics
      const avgSpeed = boids.reduce((sum, b) => {
        const speed = Math.sqrt(b.velocity.x ** 2 + b.velocity.y ** 2);
        return sum + speed;
      }, 0) / boids.length;

      const densityVariance = calculateDensityVariance(boids, width, height);

      // Create system state
      const state: SystemState = {
        timestamp: Date.now(),
        agents: boids.map(b => ({
          id: b.id,
          position: b.position,
          velocity: b.velocity,
          state: { speed: Math.sqrt(b.velocity.x ** 2 + b.velocity.y ** 2) },
          neighbors: [] // Simplified for this example
        })),
        globalMetrics: {
          entropy: densityVariance / 100,
          order: lanes.separationScore,
          complexity: Math.min(1, lanes.numLanes / 4),
          connectivity: avgSpeed / 5
        }
      };

      await emergence.addState(state);

      // Report progress
      console.log(`\nStep ${step}:`);
      console.log(`  Lanes detected: ${lanes.numLanes}`);
      console.log(`  Lane separation: ${(lanes.separationScore * 100).toFixed(1)}%`);
      console.log(`  Average speed: ${avgSpeed.toFixed(2)}`);
      console.log(`  Density variance: ${densityVariance.toFixed(2)}`);

      // Detect lane formation
      if (!laneFormationDetected && lanes.numLanes >= 2 && lanes.separationScore > 0.7) {
        console.log('\n✅ EMERGENCE: Clear lane formation detected!');
        laneFormationDetected = true;
      }

      // Detect congestion
      if (!congestionDetected && avgSpeed < 2 && densityVariance > 50) {
        console.log('\n⚠️  EMERGENCE: Traffic congestion pattern detected!');
        congestionDetected = true;
      }
    }
  }

  // Final analysis
  console.log('\n' + '='.repeat(60));
  console.log('Final Analysis');
  console.log('='.repeat(60));

  const metrics = emergence.getLatestMetrics();
  console.log('\nEmergence Metrics:');
  console.log(`  Self-Organization: ${(metrics.selfOrganization * 100).toFixed(1)}%`);
  console.log(`  Complexity: ${(metrics.complexity * 100).toFixed(1)}%`);
  console.log(`  Coherence: ${(metrics.coherence * 100).toFixed(1)}%`);
  console.log(`  Adaptability: ${(metrics.adaptability * 100).toFixed(1)}%`);
  console.log(`  Robustness: ${(metrics.robustness * 100).toFixed(1)}%`);
  console.log(`  Novelty: ${(metrics.novelty * 100).toFixed(1)}%`);

  const events = emergence.getEmergenceEvents();
  if (events.length > 0) {
    console.log('\nEmergence Events Detected:');
    events.forEach((event, i) => {
      console.log(`\n${i + 1}. ${event.type} (confidence: ${(event.confidence * 100).toFixed(1)}%)`);
      console.log(`   ${event.description}`);
    });
  }

  console.log('\n✅ Simulation complete!');
}

/**
 * Analyze lane formation in traffic
 */
function analyzeLaneFormation(
  boids: Array<{ position: Vector2D; velocity: Vector2D }>,
  height: number
): { numLanes: number; separationScore: number } {
  // Divide space into horizontal bands
  const numBands = 10;
  const bandHeight = height / numBands;
  const bandCounts = new Array(numBands).fill(0);
  const bandDirections = new Array(numBands).fill(0);

  // Count vehicles in each band and track direction
  for (const boid of boids) {
    const band = Math.floor(boid.position.y / bandHeight);
    if (band >= 0 && band < numBands) {
      bandCounts[band]++;
      bandDirections[band] += Math.sign(boid.velocity.x);
    }
  }

  // Find lanes (bands with significant traffic moving in same direction)
  const threshold = 5; // Minimum vehicles per lane
  let numLanes = 0;
  let totalSeparation = 0;
  let laneCount = 0;

  for (let i = 0; i < numBands; i++) {
    if (bandCounts[i] >= threshold) {
      const directionConsistency = Math.abs(bandDirections[i]) / bandCounts[i];

      if (directionConsistency > 0.6) {
        numLanes++;
        totalSeparation += directionConsistency;
        laneCount++;
      }
    }
  }

  const separationScore = laneCount > 0 ? totalSeparation / laneCount : 0;

  return { numLanes, separationScore };
}

/**
 * Calculate spatial density variance
 */
function calculateDensityVariance(
  boids: Array<{ position: Vector2D }>,
  width: number,
  height: number
): number {
  const gridSize = 50;
  const cols = Math.ceil(width / gridSize);
  const rows = Math.ceil(height / gridSize);
  const grid = Array(rows).fill(0).map(() => Array(cols).fill(0));

  // Count vehicles in each cell
  for (const boid of boids) {
    const col = Math.floor(boid.position.x / gridSize);
    const row = Math.floor(boid.position.y / gridSize);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      grid[row][col]++;
    }
  }

  // Calculate variance
  const counts = grid.flat();
  const mean = counts.reduce((sum, c) => sum + c, 0) / counts.length;
  const variance = counts.reduce((sum, c) => sum + (c - mean) ** 2, 0) / counts.length;

  return Math.sqrt(variance);
}

// Run if called directly
if (require.main === module) {
  runTrafficSimulation().catch(console.error);
}

export { runTrafficSimulation };
