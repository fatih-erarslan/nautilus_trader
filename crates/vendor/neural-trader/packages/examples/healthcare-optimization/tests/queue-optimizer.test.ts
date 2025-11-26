/**
 * Queue Optimizer Tests
 */

import { QueueOptimizer } from '../src/queue-optimizer';
import type { Patient, ResourcePool, OptimizationObjective } from '../src/types';
import { describe, it, expect, beforeEach } from '@jest/globals';

describe('QueueOptimizer', () => {
  let optimizer: QueueOptimizer;

  beforeEach(() => {
    optimizer = new QueueOptimizer({
      targetUtilization: 0.75,
      maxWaitTime: 30,
      abandonmentThreshold: 120,
      reallocateInterval: 60
    });

    // Add resource pools
    optimizer.addResourcePool({
      id: 'exam-rooms',
      type: 'exam_room',
      capacity: 10,
      available: 10,
      utilizationTarget: 0.75
    });

    optimizer.addResourcePool({
      id: 'imaging',
      type: 'imaging',
      capacity: 3,
      available: 3,
      utilizationTarget: 0.80
    });
  });

  describe('Patient Queue Management', () => {
    it('should add patients to queue', () => {
      const patient = createPatient('p1', 3);

      optimizer.addPatient(patient);

      const state = optimizer.getState();
      expect(state.waitingPatients).toHaveLength(1);
      expect(state.waitingPatients[0].id).toBe('p1');
    });

    it('should prioritize by acuity', () => {
      const lowAcuity = createPatient('p1', 5);
      const highAcuity = createPatient('p2', 1);
      const medAcuity = createPatient('p3', 3);

      optimizer.addPatient(lowAcuity);
      optimizer.addPatient(highAcuity);
      optimizer.addPatient(medAcuity);

      const state = optimizer.getState();
      expect(state.waitingPatients[0].id).toBe('p2'); // highest priority
      expect(state.waitingPatients[1].id).toBe('p3');
      expect(state.waitingPatients[2].id).toBe('p1'); // lowest priority
    });

    it('should process next patient', () => {
      const patient = createPatient('p1', 3);
      optimizer.addPatient(patient);

      const processed = optimizer.processNextPatient('exam-rooms');

      expect(processed).toBeTruthy();
      expect(processed?.id).toBe('p1');

      const state = optimizer.getState();
      expect(state.waitingPatients).toHaveLength(0);
      expect(state.inServicePatients).toHaveLength(1);
    });

    it('should complete patient service', () => {
      const patient = createPatient('p1', 3);
      optimizer.addPatient(patient);
      optimizer.processNextPatient('exam-rooms');

      optimizer.completeService('p1', 'exam-rooms');

      const state = optimizer.getState();
      expect(state.inServicePatients).toHaveLength(0);
      expect(state.resources[0].available).toBe(10); // back to full
    });
  });

  describe('Resource Management', () => {
    it('should track resource availability', () => {
      const patients = Array.from({ length: 5 }, (_, i) => createPatient(`p${i}`, 3));

      patients.forEach(p => optimizer.addPatient(p));

      // Process all
      for (let i = 0; i < 5; i++) {
        optimizer.processNextPatient('exam-rooms');
      }

      const state = optimizer.getState();
      expect(state.resources[0].available).toBe(5); // 10 - 5
      expect(state.resources[0].capacity).toBe(10);
    });

    it('should not process when resources unavailable', () => {
      // Fill up imaging capacity
      for (let i = 0; i < 3; i++) {
        const patient = createPatient(`p${i}`, 2);
        optimizer.addPatient(patient);
        optimizer.processNextPatient('imaging');
      }

      // Try to process one more
      const patient = createPatient('p4', 2);
      optimizer.addPatient(patient);
      const result = optimizer.processNextPatient('imaging');

      expect(result).toBeNull();

      const state = optimizer.getState();
      expect(state.waitingPatients).toHaveLength(1);
    });
  });

  describe('Queue Metrics', () => {
    it('should calculate queue metrics', () => {
      const patients = Array.from({ length: 10 }, (_, i) => createPatient(`p${i}`, 3));

      patients.forEach(p => optimizer.addPatient(p));

      const state = optimizer.getState();

      expect(state.metrics.queueLength).toBe(10);
      expect(state.metrics.timestamp).toBeInstanceOf(Date);
    });

    it('should track wait times', () => {
      const oldPatient = createPatient('old', 3);
      oldPatient.arrivalTime = new Date(Date.now() - 30 * 60 * 1000); // 30 min ago

      optimizer.addPatient(oldPatient);

      const state = optimizer.getState();
      expect(state.metrics.averageWaitTime).toBeGreaterThan(20);
    });

    it('should maintain metrics history', () => {
      // Add and process some patients
      for (let i = 0; i < 5; i++) {
        const patient = createPatient(`p${i}`, 3);
        optimizer.addPatient(patient);
        if (i < 3) {
          optimizer.processNextPatient('exam-rooms');
        }
      }

      const history = optimizer.getMetricsHistory();
      expect(history.length).toBeGreaterThan(0);
    });
  });

  describe('Resource Optimization', () => {
    beforeEach(() => {
      // Create high load scenario
      for (let i = 0; i < 20; i++) {
        const patient = createPatient(`p${i}`, 3);
        optimizer.addPatient(patient);
      }
    });

    it('should recommend resource increases for high load', () => {
      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.6,
        maximizeUtilization: 0.2,
        minimizeCost: 0.1,
        maximizePatientOutcomes: 0.1
      };

      const result = optimizer.optimizeResources(objective);

      expect(result.recommendations.length).toBeGreaterThan(0);

      // Should recommend increases for high queue
      const examRoomRec = result.recommendations.find(r => r.resourceId === 'exam-rooms');
      expect(examRoomRec).toBeTruthy();
    });

    it('should estimate impact of resource changes', () => {
      const objective: OptimizationObjective = {
        minimizeWaitTime: 0.5,
        maximizeUtilization: 0.3,
        minimizeCost: 0.1,
        maximizePatientOutcomes: 0.1
      };

      const result = optimizer.optimizeResources(objective);

      expect(result.expectedImpact).toBeTruthy();
      expect(result.expectedImpact.waitTimeReduction).toBeDefined();
      expect(result.expectedImpact.utilizationChange).toBeDefined();
      expect(result.expectedImpact.costChange).toBeDefined();
    });

    it('should balance wait time and cost objectives', () => {
      // High wait time penalty
      const waitFocused: OptimizationObjective = {
        minimizeWaitTime: 0.9,
        maximizeUtilization: 0.05,
        minimizeCost: 0.05,
        maximizePatientOutcomes: 0.0
      };

      // High cost penalty
      const costFocused: OptimizationObjective = {
        minimizeWaitTime: 0.1,
        maximizeUtilization: 0.1,
        minimizeCost: 0.8,
        maximizePatientOutcomes: 0.0
      };

      const waitResult = optimizer.optimizeResources(waitFocused);
      const costResult = optimizer.optimizeResources(costFocused);

      // Wait-focused should recommend more increases
      const waitIncreases = waitResult.recommendations.filter(r => r.action === 'increase').length;
      const costIncreases = costResult.recommendations.filter(r => r.action === 'increase').length;

      expect(waitIncreases).toBeGreaterThanOrEqual(costIncreases);
    });
  });
});

/**
 * Create test patient
 */
function createPatient(id: string, acuity: 1 | 2 | 3 | 4 | 5): Patient {
  return {
    id,
    arrivalTime: new Date(),
    acuity,
    chiefComplaint: 'test complaint',
    estimatedServiceTime: 30,
    requiredSkills: ['diagnosis']
  };
}
