/**
 * Staff Scheduler
 *
 * Optimizes staff scheduling with skill constraints, shift preferences,
 * and fairness considerations using constraint programming.
 */

import type {
  StaffMember,
  Shift,
  ScheduleConstraints,
  ScheduleSolution,
  ForecastResult
} from './types.js';

export interface SchedulerConfig {
  planningHorizonDays: number;
  shiftDuration: number; // hours
  costPerConstraintViolation: number;
}

export class Scheduler {
  private config: SchedulerConfig;
  private staff: StaffMember[];

  constructor(config: SchedulerConfig) {
    this.config = config;
    this.staff = [];
  }

  /**
   * Add staff member to pool
   */
  addStaff(member: StaffMember): void {
    this.staff.push(member);
  }

  /**
   * Generate optimal schedule based on forecasted demand
   */
  async generateSchedule(
    forecasts: ForecastResult[],
    constraints: ScheduleConstraints,
    startDate: Date
  ): Promise<ScheduleSolution> {
    const shifts: Shift[] = [];
    const violations: string[] = [];

    // Calculate required staff per shift based on forecasts
    const shiftRequirements = this.calculateShiftRequirements(forecasts, constraints);

    // Generate shifts for planning horizon
    const daysToSchedule = this.config.planningHorizonDays;
    const shiftsPerDay = 24 / this.config.shiftDuration;

    for (let day = 0; day < daysToSchedule; day++) {
      for (let shiftNum = 0; shiftNum < shiftsPerDay; shiftNum++) {
        const shiftStart = new Date(startDate);
        shiftStart.setDate(shiftStart.getDate() + day);
        shiftStart.setHours(shiftNum * this.config.shiftDuration, 0, 0, 0);

        const shiftEnd = new Date(shiftStart);
        shiftEnd.setHours(shiftEnd.getHours() + this.config.shiftDuration);

        // Assign staff to this shift
        const shiftAssignments = this.assignStaffToShift(
          shiftStart,
          shiftEnd,
          shiftRequirements[day * shiftsPerDay + shiftNum],
          constraints,
          shifts
        );

        shifts.push(...shiftAssignments.shifts);
        violations.push(...shiftAssignments.violations);
      }
    }

    // Calculate solution quality metrics
    const totalCost = this.calculateTotalCost(shifts);
    const coverageScore = this.calculateCoverageScore(shifts, shiftRequirements);
    const fairnessScore = this.calculateFairnessScore(shifts);

    return {
      shifts,
      totalCost,
      coverageScore,
      fairnessScore,
      constraintViolations: violations
    };
  }

  /**
   * Calculate shift requirements from forecasts
   */
  private calculateShiftRequirements(
    forecasts: ForecastResult[],
    constraints: ScheduleConstraints
  ): Array<Record<string, number>> {
    const requirements: Array<Record<string, number>> = [];
    const shiftsPerDay = 24 / this.config.shiftDuration;

    for (let i = 0; i < forecasts.length; i += this.config.shiftDuration) {
      // Average forecast over shift duration
      const shiftForecasts = forecasts.slice(i, i + this.config.shiftDuration);
      const avgArrivals = shiftForecasts.reduce((sum, f) => sum + f.predictedArrivals, 0) / shiftForecasts.length;

      // Calculate required staff based on patient-to-staff ratio
      // Typical ED: 1 physician per 2.5 patients, 1 nurse per 4 patients
      const requirement: Record<string, number> = {
        physician: Math.ceil(avgArrivals / 2.5),
        nurse: Math.ceil(avgArrivals / 4),
        technician: Math.ceil(avgArrivals / 6),
        specialist: Math.max(1, Math.ceil(avgArrivals / 10))
      };

      // Apply minimum constraints
      for (const [role, minCount] of Object.entries(constraints.minStaffPerShift)) {
        requirement[role] = Math.max(requirement[role] || 0, minCount);
      }

      requirements.push(requirement);
    }

    return requirements;
  }

  /**
   * Assign staff to a specific shift
   */
  private assignStaffToShift(
    start: Date,
    end: Date,
    requirements: Record<string, number>,
    constraints: ScheduleConstraints,
    existingShifts: Shift[]
  ): { shifts: Shift[]; violations: string[] } {
    const shifts: Shift[] = [];
    const violations: string[] = [];

    // Group staff by role
    const staffByRole = this.groupStaffByRole();

    for (const [role, requiredCount] of Object.entries(requirements)) {
      const availableStaff = staffByRole[role] || [];
      let assignedCount = 0;

      // Score and sort staff by suitability for this shift
      const scoredStaff = availableStaff.map(staff => ({
        staff,
        score: this.scoreStaffForShift(staff, start, end, existingShifts, constraints)
      }));

      scoredStaff.sort((a, b) => b.score - a.score);

      // Assign top-scoring staff
      for (const { staff, score } of scoredStaff) {
        if (assignedCount >= requiredCount) {
          break;
        }

        if (score > 0) {
          shifts.push({
            id: `${staff.id}-${start.toISOString()}`,
            staffId: staff.id,
            start,
            end,
            role,
            assignedArea: this.assignArea(role)
          });
          assignedCount++;
        }
      }

      // Check if we met requirements
      if (assignedCount < requiredCount) {
        violations.push(
          `Insufficient ${role} coverage: ${assignedCount}/${requiredCount} at ${start.toISOString()}`
        );
      }
    }

    return { shifts, violations };
  }

  /**
   * Score staff member for shift assignment
   */
  private scoreStaffForShift(
    staff: StaffMember,
    start: Date,
    end: Date,
    existingShifts: Shift[],
    constraints: ScheduleConstraints
  ): number {
    let score = 100;

    // Check if already assigned to overlapping shift
    const hasOverlap = existingShifts.some(shift => {
      return shift.staffId === staff.id &&
        shift.start < end &&
        shift.end > start;
    });

    if (hasOverlap) {
      return 0; // Cannot assign
    }

    // Check weekly hours
    const weeklyHours = this.calculateWeeklyHours(staff.id, start, existingShifts);
    const shiftHours = (end.getTime() - start.getTime()) / (1000 * 60 * 60);

    if (weeklyHours + shiftHours > staff.maxHoursPerWeek) {
      return 0; // Would exceed max hours
    }

    // Check rest period
    const lastShift = this.getLastShift(staff.id, start, existingShifts);
    if (lastShift) {
      const restHours = (start.getTime() - lastShift.end.getTime()) / (1000 * 60 * 60);
      if (restHours < constraints.minRestBetweenShifts) {
        return 0; // Insufficient rest
      }
    }

    // Check consecutive hours
    const consecutiveHours = this.calculateConsecutiveHours(staff.id, start, existingShifts);
    if (consecutiveHours + shiftHours > constraints.maxConsecutiveHours) {
      score -= 30;
    }

    // Preference matching
    const shiftPeriod = this.getShiftPeriod(start);
    if (staff.shiftPreference !== 'any' && staff.shiftPreference !== shiftPeriod) {
      score -= 20;
    } else if (staff.shiftPreference === shiftPeriod) {
      score += 10;
    }

    // Balance workload - prefer staff with fewer hours
    const hoursRatio = weeklyHours / staff.maxHoursPerWeek;
    score -= hoursRatio * 20;

    // Cost factor - prefer lower cost staff when appropriate
    score -= (staff.costPerHour / 100) * 5;

    return score;
  }

  /**
   * Calculate total weekly hours for staff member
   */
  private calculateWeeklyHours(
    staffId: string,
    referenceDate: Date,
    shifts: Shift[]
  ): number {
    const weekStart = new Date(referenceDate);
    weekStart.setDate(weekStart.getDate() - weekStart.getDay());
    weekStart.setHours(0, 0, 0, 0);

    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);

    return shifts
      .filter(s => s.staffId === staffId && s.start >= weekStart && s.start < weekEnd)
      .reduce((total, s) => {
        const hours = (s.end.getTime() - s.start.getTime()) / (1000 * 60 * 60);
        return total + hours;
      }, 0);
  }

  /**
   * Get last shift before reference date
   */
  private getLastShift(
    staffId: string,
    referenceDate: Date,
    shifts: Shift[]
  ): Shift | null {
    const staffShifts = shifts
      .filter(s => s.staffId === staffId && s.end <= referenceDate)
      .sort((a, b) => b.end.getTime() - a.end.getTime());

    return staffShifts[0] || null;
  }

  /**
   * Calculate consecutive hours worked
   */
  private calculateConsecutiveHours(
    staffId: string,
    referenceDate: Date,
    shifts: Shift[]
  ): number {
    const staffShifts = shifts
      .filter(s => s.staffId === staffId && s.end <= referenceDate)
      .sort((a, b) => b.end.getTime() - a.end.getTime());

    let consecutiveHours = 0;
    let currentDate = referenceDate;

    for (const shift of staffShifts) {
      const gap = (currentDate.getTime() - shift.end.getTime()) / (1000 * 60 * 60);

      if (gap > 12) {
        break; // No longer consecutive
      }

      const shiftHours = (shift.end.getTime() - shift.start.getTime()) / (1000 * 60 * 60);
      consecutiveHours += shiftHours;
      currentDate = shift.start;
    }

    return consecutiveHours;
  }

  /**
   * Get shift period (day/evening/night)
   */
  private getShiftPeriod(date: Date): 'day' | 'evening' | 'night' {
    const hour = date.getHours();

    if (hour >= 7 && hour < 15) {
      return 'day';
    } else if (hour >= 15 && hour < 23) {
      return 'evening';
    } else {
      return 'night';
    }
  }

  /**
   * Assign area based on role
   */
  private assignArea(role: string): 'ed' | 'or' | 'icu' | 'ward' {
    // Simplified area assignment
    if (role === 'physician' || role === 'nurse') {
      return 'ed';
    } else if (role === 'specialist') {
      return 'or';
    } else {
      return 'ward';
    }
  }

  /**
   * Group staff by role
   */
  private groupStaffByRole(): Record<string, StaffMember[]> {
    const grouped: Record<string, StaffMember[]> = {};

    for (const staff of this.staff) {
      if (!grouped[staff.role]) {
        grouped[staff.role] = [];
      }
      grouped[staff.role].push(staff);
    }

    return grouped;
  }

  /**
   * Calculate total cost of schedule
   */
  private calculateTotalCost(shifts: Shift[]): number {
    return shifts.reduce((total, shift) => {
      const staff = this.staff.find(s => s.id === shift.staffId);
      if (!staff) return total;

      const hours = (shift.end.getTime() - shift.start.getTime()) / (1000 * 60 * 60);
      return total + hours * staff.costPerHour;
    }, 0);
  }

  /**
   * Calculate coverage score (0-1)
   */
  private calculateCoverageScore(
    shifts: Shift[],
    requirements: Array<Record<string, number>>
  ): number {
    let totalRequired = 0;
    let totalMet = 0;

    for (const requirement of requirements) {
      for (const [role, count] of Object.entries(requirement)) {
        totalRequired += count;
        // Count how many shifts of this role we have
        // This is simplified - in production, match by time slot
        const assigned = shifts.filter(s => s.role === role).length;
        totalMet += Math.min(assigned, count);
      }
    }

    return totalRequired > 0 ? totalMet / totalRequired : 1;
  }

  /**
   * Calculate fairness score (0-1)
   */
  private calculateFairnessScore(shifts: Shift[]): number {
    // Calculate standard deviation of hours worked
    const hoursPerStaff: Record<string, number> = {};

    for (const shift of shifts) {
      if (!hoursPerStaff[shift.staffId]) {
        hoursPerStaff[shift.staffId] = 0;
      }
      const hours = (shift.end.getTime() - shift.start.getTime()) / (1000 * 60 * 60);
      hoursPerStaff[shift.staffId] += hours;
    }

    const hours = Object.values(hoursPerStaff);
    if (hours.length === 0) return 1;

    const mean = hours.reduce((a, b) => a + b, 0) / hours.length;
    const variance = hours.reduce((sum, h) => sum + Math.pow(h - mean, 2), 0) / hours.length;
    const stdDev = Math.sqrt(variance);

    // Convert to 0-1 score (lower stdDev = higher fairness)
    const maxStdDev = 20; // assume max acceptable is 20 hours difference
    return Math.max(0, 1 - stdDev / maxStdDev);
  }

  /**
   * Get staff list
   */
  getStaff(): StaffMember[] {
    return [...this.staff];
  }
}
