import { AntColonyOptimization } from '../src/ant-colony';

describe('AntColonyOptimization', () => {
  let aco: AntColonyOptimization;

  beforeEach(() => {
    aco = new AntColonyOptimization({
      numAnts: 10,
      maxIterations: 50
    });
  });

  describe('graph construction', () => {
    it('should add nodes to graph', () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 100, y: 0 });

      const nodes = aco.getNodes();
      expect(nodes).toHaveLength(2);
      expect(nodes.find(n => n.id === 'A')).toBeDefined();
      expect(nodes.find(n => n.id === 'B')).toBeDefined();
    });

    it('should add edges between nodes', () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 100, y: 0 });
      aco.addEdge('A', 'B');

      const edges = aco.getEdges();
      expect(edges.length).toBeGreaterThanOrEqual(2); // Bidirectional
    });

    it('should calculate edge distances', () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 3, y: 4 });
      aco.addEdge('A', 'B');

      const edges = aco.getEdges();
      const edge = edges.find(e => e.from === 'A' && e.to === 'B');

      expect(edge).toBeDefined();
      expect(edge?.distance).toBeCloseTo(5, 1); // sqrt(3^2 + 4^2) = 5
    });

    it('should throw error for non-existent nodes', () => {
      expect(() => aco.addEdge('X', 'Y')).toThrow();
    });
  });

  describe('pathfinding', () => {
    beforeEach(() => {
      // Create simple grid graph
      //   A --- B --- C
      //   |     |     |
      //   D --- E --- F
      const nodes = [
        { id: 'A', x: 0, y: 0 },
        { id: 'B', x: 100, y: 0 },
        { id: 'C', x: 200, y: 0 },
        { id: 'D', x: 0, y: 100 },
        { id: 'E', x: 100, y: 100 },
        { id: 'F', x: 200, y: 100 }
      ];

      nodes.forEach(n => aco.addNode(n));

      aco.addEdge('A', 'B');
      aco.addEdge('B', 'C');
      aco.addEdge('A', 'D');
      aco.addEdge('B', 'E');
      aco.addEdge('C', 'F');
      aco.addEdge('D', 'E');
      aco.addEdge('E', 'F');
    });

    it('should find path between nodes', async () => {
      const result = await aco.optimize('A', 'F');

      expect(result).toBeDefined();
      expect(result.path).toContain('A');
      expect(result.path).toContain('F');
      expect(result.path[0]).toBe('A');
      expect(result.path[result.path.length - 1]).toBe('F');
    });

    it('should find optimal path', async () => {
      const result = await aco.optimize('A', 'F');

      // Optimal path should be A -> B -> E -> F or A -> D -> E -> F
      expect(result.path).toHaveLength(4);
      expect(result.length).toBeCloseTo(300, 10);
    });

    it('should improve path over iterations', async () => {
      const acoShort = new AntColonyOptimization({ maxIterations: 5 });
      const acoLong = new AntColonyOptimization({ maxIterations: 50 });

      // Build same graph for both
      const nodes = [
        { id: 'A', x: 0, y: 0 },
        { id: 'B', x: 100, y: 0 },
        { id: 'C', x: 200, y: 0 }
      ];

      nodes.forEach(n => {
        acoShort.addNode(n);
        acoLong.addNode(n);
      });

      acoShort.addEdge('A', 'B');
      acoShort.addEdge('B', 'C');

      acoLong.addEdge('A', 'B');
      acoLong.addEdge('B', 'C');

      const resultShort = await acoShort.optimize('A', 'C');
      const resultLong = await acoLong.optimize('A', 'C');

      // More iterations should converge faster
      expect(resultLong.iterations).toBeGreaterThan(resultShort.iterations);
    });

    it('should throw error for unreachable goal', async () => {
      aco.addNode({ id: 'X', x: 1000, y: 1000 }); // Isolated node

      await expect(aco.optimize('A', 'X')).rejects.toThrow();
    });
  });

  describe('pheromone management', () => {
    it('should initialize edges with pheromone', () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 100, y: 0 });
      aco.addEdge('A', 'B');

      const edges = aco.getEdges();
      edges.forEach(edge => {
        expect(edge.pheromone).toBeGreaterThan(0);
      });
    });

    it('should update pheromones after optimization', async () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 100, y: 0 });
      aco.addEdge('A', 'B');

      const initialPheromone = aco.getEdges()[0].pheromone;

      await aco.optimize('A', 'B');

      const finalPheromone = aco.getEdges()[0].pheromone;

      // Pheromone should change after optimization
      expect(finalPheromone).not.toBe(initialPheromone);
    });
  });

  describe('configuration', () => {
    it('should respect custom configuration', () => {
      const customAco = new AntColonyOptimization({
        numAnts: 50,
        alpha: 2.0,
        beta: 3.0,
        evaporationRate: 0.7
      });

      expect(customAco).toBeDefined();
    });
  });

  describe('clear', () => {
    it('should remove all data', () => {
      aco.addNode({ id: 'A', x: 0, y: 0 });
      aco.addNode({ id: 'B', x: 100, y: 0 });
      aco.addEdge('A', 'B');

      expect(aco.getNodes()).toHaveLength(2);

      aco.clear();

      expect(aco.getNodes()).toHaveLength(0);
      expect(aco.getEdges()).toHaveLength(0);
      expect(aco.getBestPath()).toBeNull();
    });
  });

  describe('performance', () => {
    it('should handle complex graphs', async () => {
      // Create 10x10 grid
      for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
          aco.addNode({ id: `node-${x}-${y}`, x: x * 50, y: y * 50 });
        }
      }

      // Connect nodes
      for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
          const current = `node-${x}-${y}`;

          if (x < 9) {
            aco.addEdge(current, `node-${x + 1}-${y}`);
          }
          if (y < 9) {
            aco.addEdge(current, `node-${x}-${y + 1}`);
          }
        }
      }

      const startTime = Date.now();
      const result = await aco.optimize('node-0-0', 'node-9-9');
      const endTime = Date.now();

      expect(result.path).toBeDefined();
      expect(endTime - startTime).toBeLessThan(10000); // Should complete in < 10s
    });
  });
});
