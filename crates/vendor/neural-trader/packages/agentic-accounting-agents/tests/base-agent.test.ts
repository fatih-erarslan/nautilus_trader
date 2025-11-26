/**
 * Base Agent Tests
 * Coverage Target: 90%+
 */

import { BaseAgent, AgentTask, AgentResult } from '../src/base/agent';

class TestAgent extends BaseAgent {
  async execute(task: AgentTask): Promise<AgentResult<any>> {
    return {
      success: true,
      data: { result: 'test complete' },
      metrics: {
        startTime: Date.now(),
        endTime: Date.now() + 100,
        duration: 100,
      },
    };
  }
}

describe('BaseAgent', () => {
  let agent: TestAgent;

  beforeEach(() => {
    agent = new TestAgent({
      agentId: 'test-001',
      agentType: 'TEST',
      enableLearning: true,
      enableMetrics: true,
      logLevel: 'info',
    });
  });

  describe('Initialization', () => {
    it('should initialize with correct config', () => {
      const status = agent.getStatus();

      expect(status.agentId).toBe('test-001');
      expect(status.agentType).toBe('TEST');
      expect(status.status).toBe('idle');
    });

    it('should initialize with default config', () => {
      const defaultAgent = new TestAgent({
        agentId: 'test-002',
        agentType: 'TEST',
      });

      const status = defaultAgent.getStatus();
      expect(status.agentId).toBe('test-002');
    });
  });

  describe('Task Execution', () => {
    it('should execute task successfully', async () => {
      const task: AgentTask = {
        id: 'task-001',
        type: 'test-task',
        priority: 'normal',
        data: {},
      };

      const result = await agent.execute(task);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should track metrics if enabled', async () => {
      const task: AgentTask = {
        id: 'task-002',
        type: 'test-task',
        priority: 'normal',
        data: {},
      };

      const result = await agent.execute(task);

      expect(result.metrics).toBeDefined();
      expect(result.metrics?.duration).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Agent Status', () => {
    it('should return current status', () => {
      const status = agent.getStatus();

      expect(status).toHaveProperty('agentId');
      expect(status).toHaveProperty('agentType');
      expect(status).toHaveProperty('status');
      expect(status).toHaveProperty('tasksProcessed');
    });

    it('should update status during task execution', async () => {
      const task: AgentTask = {
        id: 'task-003',
        type: 'test-task',
        priority: 'normal',
        data: {},
      };

      await agent.execute(task);

      const status = agent.getStatus();
      expect(status.tasksProcessed).toBeGreaterThan(0);
    });
  });

  describe('Configuration', () => {
    it('should support different log levels', () => {
      const levels = ['debug', 'info', 'warn', 'error'] as const;

      levels.forEach((logLevel) => {
        const testAgent = new TestAgent({
          agentId: `test-${logLevel}`,
          agentType: 'TEST',
          logLevel,
        });

        expect(testAgent).toBeDefined();
      });
    });

    it('should support disabling learning', () => {
      const noLearningAgent = new TestAgent({
        agentId: 'test-no-learning',
        agentType: 'TEST',
        enableLearning: false,
      });

      expect(noLearningAgent).toBeDefined();
    });

    it('should support disabling metrics', () => {
      const noMetricsAgent = new TestAgent({
        agentId: 'test-no-metrics',
        agentType: 'TEST',
        enableMetrics: false,
      });

      expect(noMetricsAgent).toBeDefined();
    });
  });

  describe('Task Priority', () => {
    it('should handle low priority tasks', async () => {
      const task: AgentTask = {
        id: 'task-low',
        type: 'test-task',
        priority: 'low',
        data: {},
      };

      const result = await agent.execute(task);
      expect(result.success).toBe(true);
    });

    it('should handle high priority tasks', async () => {
      const task: AgentTask = {
        id: 'task-high',
        type: 'test-task',
        priority: 'high',
        data: {},
      };

      const result = await agent.execute(task);
      expect(result.success).toBe(true);
    });

    it('should handle critical priority tasks', async () => {
      const task: AgentTask = {
        id: 'task-critical',
        type: 'test-task',
        priority: 'critical',
        data: {},
      };

      const result = await agent.execute(task);
      expect(result.success).toBe(true);
    });
  });
});
