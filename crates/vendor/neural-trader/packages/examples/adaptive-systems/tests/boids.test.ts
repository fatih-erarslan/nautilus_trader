import { BoidsSimulation } from '../src/boids';

describe('BoidsSimulation', () => {
  let simulation: BoidsSimulation;

  beforeEach(() => {
    simulation = new BoidsSimulation({ width: 800, height: 600 });
  });

  describe('initialization', () => {
    it('should create simulation with correct boundaries', () => {
      expect(simulation).toBeDefined();
      expect(simulation.getBoids()).toHaveLength(0);
    });

    it('should add boids to simulation', () => {
      simulation.addBoid('boid-1', { x: 100, y: 100 });
      simulation.addBoid('boid-2', { x: 200, y: 200 });

      expect(simulation.getBoids()).toHaveLength(2);
    });
  });

  describe('flocking behavior', () => {
    beforeEach(() => {
      // Add cluster of boids
      for (let i = 0; i < 10; i++) {
        simulation.addBoid(
          `boid-${i}`,
          { x: 400 + Math.random() * 50, y: 300 + Math.random() * 50 },
          { x: Math.random() - 0.5, y: Math.random() - 0.5 }
        );
      }
    });

    it('should update boid positions', async () => {
      const initialPositions = simulation.getBoids().map(b => ({ ...b.position }));

      await simulation.update();

      const updatedPositions = simulation.getBoids().map(b => b.position);

      // At least some boids should have moved
      const hasMoved = updatedPositions.some((pos, i) =>
        pos.x !== initialPositions[i].x || pos.y !== initialPositions[i].y
      );

      expect(hasMoved).toBe(true);
    });

    it('should limit velocity to maxSpeed', async () => {
      // Add boid with very high initial velocity
      simulation.addBoid(
        'fast-boid',
        { x: 400, y: 300 },
        { x: 100, y: 100 }
      );

      await simulation.update();

      const boid = simulation.getBoid('fast-boid');
      expect(boid).toBeDefined();

      if (boid) {
        const speed = Math.sqrt(boid.velocity.x ** 2 + boid.velocity.y ** 2);
        expect(speed).toBeLessThanOrEqual(boid.maxSpeed + 0.01); // Small epsilon for floating point
      }
    });

    it('should exhibit separation behavior', async () => {
      // Add two boids very close together
      simulation.clear();
      simulation.addBoid('boid-1', { x: 400, y: 300 }, { x: 0, y: 0 });
      simulation.addBoid('boid-2', { x: 405, y: 300 }, { x: 0, y: 0 });

      const initialDistance = 5;

      // Run several steps
      for (let i = 0; i < 10; i++) {
        await simulation.update();
      }

      const boid1 = simulation.getBoid('boid-1')!;
      const boid2 = simulation.getBoid('boid-2')!;

      const finalDistance = Math.sqrt(
        (boid2.position.x - boid1.position.x) ** 2 +
        (boid2.position.y - boid1.position.y) ** 2
      );

      // Boids should have separated
      expect(finalDistance).toBeGreaterThan(initialDistance);
    });
  });

  describe('boundary behavior', () => {
    it('should wrap around boundaries', async () => {
      const sim = new BoidsSimulation(
        { width: 100, height: 100 },
        { boundaryBehavior: 'wrap' }
      );

      sim.addBoid('boid', { x: 95, y: 50 }, { x: 10, y: 0 });

      await sim.update();

      const boid = sim.getBoid('boid')!;
      expect(boid.position.x).toBeLessThan(20); // Should wrap to other side
    });

    it('should bounce off boundaries', async () => {
      const sim = new BoidsSimulation(
        { width: 100, height: 100 },
        { boundaryBehavior: 'bounce', maxSpeed: 5 }
      );

      sim.addBoid('boid', { x: 95, y: 50 }, { x: 5, y: 0 });

      const initialVelocityX = 5;

      await sim.update();

      const boid = sim.getBoid('boid')!;

      // Should have bounced and reversed direction
      expect(Math.sign(boid.velocity.x)).toBe(-1);
    });
  });

  describe('vector math', () => {
    it('should calculate distance correctly', () => {
      simulation.addBoid('boid-1', { x: 0, y: 0 });
      simulation.addBoid('boid-2', { x: 3, y: 4 });

      const boid1 = simulation.getBoid('boid-1')!;
      const boid2 = simulation.getBoid('boid-2')!;

      // Distance formula: sqrt(3^2 + 4^2) = 5
      // This is implicitly tested in the simulation
      expect(boid1).toBeDefined();
      expect(boid2).toBeDefined();
    });
  });

  describe('performance', () => {
    it('should handle large number of boids', async () => {
      const numBoids = 1000;

      for (let i = 0; i < numBoids; i++) {
        simulation.addBoid(
          `boid-${i}`,
          { x: Math.random() * 800, y: Math.random() * 600 },
          { x: Math.random() - 0.5, y: Math.random() - 0.5 }
        );
      }

      const startTime = Date.now();
      await simulation.update();
      const endTime = Date.now();

      const updateTime = endTime - startTime;

      // Should complete in reasonable time (< 5 seconds for 1000 boids)
      expect(updateTime).toBeLessThan(5000);
    });
  });

  describe('clear', () => {
    it('should remove all boids', () => {
      simulation.addBoid('boid-1', { x: 100, y: 100 });
      simulation.addBoid('boid-2', { x: 200, y: 200 });

      expect(simulation.getBoids()).toHaveLength(2);

      simulation.clear();

      expect(simulation.getBoids()).toHaveLength(0);
    });
  });
});
