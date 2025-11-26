/**
 * Scheduler Tests
 */

import { Scheduler } from '../src/scheduler';
import type { ScheduleConstraints, ForecastResult, StaffMember } from '../src/types';
import { describe, it, expect, beforeEach } from '@jest/globals';

describe('Scheduler', () => {
  let scheduler: Scheduler;

  beforeEach(() => {
    scheduler = new Scheduler({
      planningHorizonDays: 7,
      shiftDuration: 8,
      costPerConstraintViolation: 1000
    });
  });

  describe('Staff Management', () => {
    it('should add staff members', () => {
      const staff: StaffMember = {
        id: 'staff-1',
        name: 'Dr. Smith',
        role: 'physician',
        skills: ['emergency_care', 'diagnosis'],
        shiftPreference: 'day',
        maxHoursPerWeek: 48,
        costPerHour: 150
      };

      scheduler.addStaff(staff);

      const staffList = scheduler.getStaff();
      expect(staffList).toHaveLength(1);
      expect(staffList[0].id).toBe('staff-1');
    });

    it('should manage multiple staff members', () => {
      const staff = generateSyntheticStaff(20);

      staff.forEach(s => scheduler.addStaff(s));

      const staffList = scheduler.getStaff();
      expect(staffList).toHaveLength(20);
    });
  });

  describe('Schedule Generation', () => {
    beforeEach(() => {
      const staff = generateSyntheticStaff(30);
      staff.forEach(s => scheduler.addStaff(s));
    });

    it('should generate schedule from forecasts', async () => {
      const forecasts = generateMockForecasts(24 * 7);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          physician: 2,
          nurse: 4,
          technician: 2,
          specialist: 1
        },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: ['emergency_care', 'patient_care']
      };

      const startDate = new Date('2024-01-15T00:00:00');
      const solution = await scheduler.generateSchedule(forecasts, constraints, startDate);

      expect(solution.shifts.length).toBeGreaterThan(0);
      expect(solution.totalCost).toBeGreaterThan(0);
      expect(solution.coverageScore).toBeGreaterThanOrEqual(0);
      expect(solution.coverageScore).toBeLessThanOrEqual(1);
      expect(solution.fairnessScore).toBeGreaterThanOrEqual(0);
      expect(solution.fairnessScore).toBeLessThanOrEqual(1);
    });

    it('should respect max hours per week constraint', async () => {
      const forecasts = generateMockForecasts(24 * 7);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          physician: 1,
          nurse: 2
        },
        maxConsecutiveHours: 12,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const startDate = new Date('2024-01-15T00:00:00');
      const solution = await scheduler.generateSchedule(forecasts, constraints, startDate);

      // Check that no staff member exceeds max weekly hours
      const hoursPerStaff: Record<string, number> = {};

      for (const shift of solution.shifts) {
        const hours = (shift.end.getTime() - shift.start.getTime()) / (1000 * 60 * 60);
        hoursPerStaff[shift.staffId] = (hoursPerStaff[shift.staffId] || 0) + hours;
      }

      const staff = scheduler.getStaff();
      for (const s of staff) {
        const totalHours = hoursPerStaff[s.id] || 0;
        expect(totalHours).toBeLessThanOrEqual(s.maxHoursPerWeek);
      }
    });

    it('should prefer staff with matching shift preferences', async () => {
      // Create staff with clear preferences
      const dayStaff: StaffMember = {
        id: 'day-1',
        name: 'Day Worker',
        role: 'physician',
        skills: ['diagnosis'],
        shiftPreference: 'day',
        maxHoursPerWeek: 40,
        costPerHour: 150
      };

      const nightStaff: StaffMember = {
        id: 'night-1',
        name: 'Night Worker',
        role: 'physician',
        skills: ['diagnosis'],
        shiftPreference: 'night',
        maxHoursPerWeek: 40,
        costPerHour: 150
      };

      const newScheduler = new Scheduler({
        planningHorizonDays: 1,
        shiftDuration: 8,
        costPerConstraintViolation: 1000
      });

      newScheduler.addStaff(dayStaff);
      newScheduler.addStaff(nightStaff);

      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: { physician: 1 },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const startDate = new Date('2024-01-15T00:00:00');
      const solution = await newScheduler.generateSchedule(forecasts, constraints, startDate);

      // Check shift assignments
      const dayShifts = solution.shifts.filter(s => s.staffId === 'day-1');
      const nightShifts = solution.shifts.filter(s => s.staffId === 'night-1');

      // Day staff should work during day
      if (dayShifts.length > 0) {
        const avgHour = dayShifts.reduce((sum, s) => sum + s.start.getHours(), 0) / dayShifts.length;
        expect(avgHour).toBeGreaterThanOrEqual(7);
        expect(avgHour).toBeLessThan(16);
      }

      // Night staff should work at night
      if (nightShifts.length > 0) {
        const hasNightShift = nightShifts.some(s => {
          const hour = s.start.getHours();
          return hour >= 23 || hour < 7;
        });
        expect(hasNightShift).toBe(true);
      }
    });

    it('should distribute workload fairly', async () => {
      const forecasts = generateMockForecasts(24 * 7);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          nurse: 2
        },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const startDate = new Date('2024-01-15T00:00:00');
      const solution = await scheduler.generateSchedule(forecasts, constraints, startDate);

      // Fairness score should be reasonable
      expect(solution.fairnessScore).toBeGreaterThan(0.5);
    });
  });

  describe('Constraint Violations', () => {
    beforeEach(() => {
      const staff = generateSyntheticStaff(5); // Limited staff
      staff.forEach(s => scheduler.addStaff(s));
    });

    it('should report constraint violations when understaffed', async () => {
      const forecasts = generateMockForecasts(24);
      const constraints: ScheduleConstraints = {
        minStaffPerShift: {
          physician: 10, // Impossible to meet
          nurse: 20
        },
        maxConsecutiveHours: 16,
        minRestBetweenShifts: 8,
        requiredSkillCoverage: []
      };

      const startDate = new Date('2024-01-15T00:00:00');
      const solution = await scheduler.generateSchedule(forecasts, constraints, startDate);

      expect(solution.constraintViolations.length).toBeGreaterThan(0);
    });
  });
});

/**
 * Generate synthetic staff
 */
function generateSyntheticStaff(count: number): StaffMember[] {
  const staff: StaffMember[] = [];
  const roles = ['physician', 'nurse', 'technician', 'specialist'] as const;
  const preferences = ['day', 'evening', 'night', 'any'] as const;

  for (let i = 0; i < count; i++) {
    const role = roles[i % roles.length];

    staff.push({
      id: `staff-${i}`,
      name: `Staff Member ${i}`,
      role,
      skills: getSkillsForRole(role),
      shiftPreference: preferences[i % preferences.length],
      maxHoursPerWeek: 40 + Math.floor(Math.random() * 20),
      costPerHour: getCostForRole(role)
    });
  }

  return staff;
}

function getSkillsForRole(role: string): string[] {
  const skillMap: Record<string, string[]> = {
    physician: ['diagnosis', 'emergency_care', 'procedures'],
    nurse: ['patient_care', 'medication', 'monitoring'],
    technician: ['lab', 'imaging', 'equipment'],
    specialist: ['surgery', 'cardiology', 'neurology']
  };

  return skillMap[role] || [];
}

function getCostForRole(role: string): number {
  const costMap: Record<string, number> = {
    physician: 150,
    nurse: 50,
    technician: 35,
    specialist: 200
  };

  return costMap[role] || 40;
}

/**
 * Generate mock forecasts
 */
function generateMockForecasts(hours: number): ForecastResult[] {
  const forecasts: ForecastResult[] = [];
  const startDate = new Date('2024-01-15T00:00:00');

  for (let i = 0; i < hours; i++) {
    const timestamp = new Date(startDate);
    timestamp.setHours(timestamp.getHours() + i);

    const hour = timestamp.getHours();
    let baseArrivals = 8;

    // Peak hours
    if ((hour >= 9 && hour <= 11) || (hour >= 18 && hour <= 20)) {
      baseArrivals = 15;
    } else if (hour >= 0 && hour <= 6) {
      baseArrivals = 4;
    }

    forecasts.push({
      timestamp,
      predictedArrivals: baseArrivals,
      lowerBound: baseArrivals - 3,
      upperBound: baseArrivals + 3,
      confidence: 0.95,
      seasonalComponent: 1.0,
      trendComponent: 0
    });
  }

  return forecasts;
}
