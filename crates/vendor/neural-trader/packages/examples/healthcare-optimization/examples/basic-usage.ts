/**
 * Healthcare Optimization - Basic Usage Example
 *
 * Demonstrates:
 * 1. Patient arrival forecasting with seasonal patterns
 * 2. Staff scheduling optimization with constraints
 * 3. Queue management and resource allocation
 * 4. Swarm intelligence for exploring solutions
 * 5. AI-powered triage with OpenRouter
 */

import { HealthcareOptimizer } from '../src/index.js';
import type {
  HealthcareOptimizationConfig,
  StaffMember,
  ResourcePool
} from '../src/types.js';

async function main() {
  console.log('🏥 Healthcare Optimization Example\n');
  console.log('=' .repeat(60));

  // Configuration
  const config: HealthcareOptimizationConfig = {
    openRouterApiKey: process.env.OPENROUTER_API_KEY,
    openRouterModel: 'anthropic/claude-3.5-sonnet',
    agentdbPath: './data/healthcare.db',
    enableNapiRS: true, // Use Rust for performance
    privacy: {
      useSyntheticData: true,
      anonymizationLevel: 'full',
      dataRetentionDays: 90,
      complianceMode: 'hipaa'
    },
    optimization: {
      minimizeWaitTime: 0.4,
      maximizeUtilization: 0.3,
      minimizeCost: 0.2,
      maximizePatientOutcomes: 0.1
    },
    swarm: {
      populationSize: 20,
      maxIterations: 50,
      explorationRate: 0.3,
      convergenceThreshold: 0.01,
      elitismRate: 0.2
    },
    constraints: {
      minStaffPerShift: {
        physician: 2,
        nurse: 4,
        technician: 2,
        specialist: 1
      },
      maxConsecutiveHours: 16,
      minRestBetweenShifts: 8,
      requiredSkillCoverage: ['emergency_care', 'patient_care', 'diagnosis']
    }
  };

  // Initialize optimizer
  const optimizer = new HealthcareOptimizer(config);

  // Step 1: Train forecaster with historical data
  console.log('\n📊 Step 1: Training arrival forecaster...\n');

  const historicalData = generateHistoricalData(90);
  await optimizer.trainForecaster(historicalData);

  console.log('✅ Trained on 90 days of historical data');
  console.log(`   Total patient arrivals: ${historicalData.reduce((sum, d) => sum + d.arrivals, 0)}`);

  // Step 2: Add hospital staff
  console.log('\n👥 Step 2: Adding hospital staff...\n');

  const staff = generateStaffPool();
  staff.forEach(s => optimizer.addStaff(s));

  console.log(`✅ Added ${staff.length} staff members`);
  console.log(`   Physicians: ${staff.filter(s => s.role === 'physician').length}`);
  console.log(`   Nurses: ${staff.filter(s => s.role === 'nurse').length}`);
  console.log(`   Technicians: ${staff.filter(s => s.role === 'technician').length}`);
  console.log(`   Specialists: ${staff.filter(s => s.role === 'specialist').length}`);

  // Step 3: Configure resource pools
  console.log('\n🏥 Step 3: Configuring resource pools...\n');

  const resources: ResourcePool[] = [
    {
      id: 'ed-exam-rooms',
      type: 'exam_room',
      capacity: 12,
      available: 12,
      utilizationTarget: 0.75
    },
    {
      id: 'imaging-suite',
      type: 'imaging',
      capacity: 4,
      available: 4,
      utilizationTarget: 0.80
    },
    {
      id: 'icu-beds',
      type: 'icu_bed',
      capacity: 20,
      available: 20,
      utilizationTarget: 0.70
    },
    {
      id: 'operating-rooms',
      type: 'or',
      capacity: 6,
      available: 6,
      utilizationTarget: 0.85
    }
  ];

  resources.forEach(r => optimizer.addResourcePool(r));

  console.log('✅ Configured resource pools:');
  resources.forEach(r => {
    console.log(`   ${r.id}: ${r.capacity} units (target: ${(r.utilizationTarget * 100).toFixed(0)}% utilization)`);
  });

  // Step 4: Run optimization
  console.log('\n🎯 Step 4: Running optimization workflow...\n');
  console.log('This may take a few minutes...\n');

  const startDate = new Date('2024-01-15T00:00:00');
  const result = await optimizer.optimize(startDate);

  // Display results
  console.log('\n' + '='.repeat(60));
  console.log('📈 OPTIMIZATION RESULTS');
  console.log('='.repeat(60));

  console.log('\n💰 Cost Analysis:');
  console.log(`   Total weekly cost: $${result.totalCost.toFixed(2)}`);
  console.log(`   Cost per shift: $${(result.totalCost / result.schedule.shifts.length).toFixed(2)}`);

  console.log('\n📊 Schedule Quality:');
  console.log(`   Coverage score: ${(result.schedule.coverageScore * 100).toFixed(1)}%`);
  console.log(`   Fairness score: ${(result.schedule.fairnessScore * 100).toFixed(1)}%`);
  console.log(`   Constraint violations: ${result.schedule.constraintViolations.length}`);

  console.log('\n⏱️  Queue Performance:');
  console.log(`   Expected wait time: ${result.expectedWaitTime.toFixed(1)} minutes`);
  console.log(`   Expected utilization: ${(result.expectedUtilization * 100).toFixed(1)}%`);

  console.log('\n✨ Overall Quality:');
  console.log(`   Quality score: ${(result.qualityScore * 100).toFixed(1)}%`);
  console.log(`   Simulation runs: ${result.simulationRuns}`);

  console.log('\n📅 Sample Schedule:');
  const sampleShifts = result.schedule.shifts.slice(0, 10);
  sampleShifts.forEach(shift => {
    const start = shift.start.toLocaleString('en-US', { hour: '2-digit', minute: '2-digit' });
    const end = shift.end.toLocaleString('en-US', { hour: '2-digit', minute: '2-digit' });
    console.log(`   ${shift.staffId} (${shift.role}): ${start} - ${end} [${shift.assignedArea}]`);
  });

  if (result.schedule.shifts.length > 10) {
    console.log(`   ... and ${result.schedule.shifts.length - 10} more shifts`);
  }

  // Step 5: Demonstrate AI triage
  if (config.openRouterApiKey) {
    console.log('\n🤖 Step 5: AI-Powered Triage Examples...\n');

    const testCases = [
      { chiefComplaint: 'Severe chest pain and shortness of breath' },
      { chiefComplaint: 'Minor cut on finger, bleeding controlled' },
      { chiefComplaint: 'High fever of 103°F and difficulty breathing' },
      { chiefComplaint: 'Sprained ankle from sports injury' }
    ];

    for (const testCase of testCases) {
      const triage = await optimizer.triagePatient(testCase);

      console.log(`📋 "${testCase.chiefComplaint}"`);
      console.log(`   Acuity: ${triage.assignedAcuity} (${getAcuityLabel(triage.assignedAcuity)})`);
      console.log(`   Path: ${triage.recommendedPath}`);
      console.log(`   Reasoning: ${triage.reasoning}`);
      console.log(`   Confidence: ${(triage.confidence * 100).toFixed(0)}%\n`);
    }
  } else {
    console.log('\n💡 Tip: Set OPENROUTER_API_KEY to enable AI-powered triage\n');
  }

  // Step 6: Continuous learning
  console.log('\n🧠 Step 6: Continuous Learning...\n');

  console.log('Simulating actual outcomes for learning...');

  const learningDate = new Date('2024-01-15T10:00:00');
  const actualArrivals = 15;
  const actualWaitTime = 22;

  await optimizer.updateWithActuals(learningDate, actualArrivals, actualWaitTime);

  console.log('✅ Updated model with actual data:');
  console.log(`   Time: ${learningDate.toLocaleString()}`);
  console.log(`   Actual arrivals: ${actualArrivals}`);
  console.log(`   Actual wait time: ${actualWaitTime} minutes`);

  console.log('\n' + '='.repeat(60));
  console.log('✅ Healthcare Optimization Complete!');
  console.log('='.repeat(60));
  console.log('\n💡 Key Benefits:');
  console.log('   • Reduced patient wait times');
  console.log('   • Optimized staff utilization');
  console.log('   • Fair workload distribution');
  console.log('   • Cost-effective scheduling');
  console.log('   • Self-learning seasonal patterns');
  console.log('   • Privacy-preserving (synthetic data only)');
  console.log('');
}

/**
 * Generate historical arrival data
 */
function generateHistoricalData(days: number) {
  const data: Array<{ timestamp: Date; arrivals: number }> = [];
  const startDate = new Date('2023-10-15T00:00:00');

  for (let day = 0; day < days; day++) {
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = new Date(startDate);
      timestamp.setDate(timestamp.getDate() + day);
      timestamp.setHours(hour);

      const month = timestamp.getMonth();
      const dayOfWeek = timestamp.getDay();

      // Base pattern
      let baseArrivals = 8;

      // Time of day pattern
      if (hour >= 9 && hour <= 11 || hour >= 18 && hour <= 20) {
        baseArrivals = 18; // peak hours
      } else if (hour >= 0 && hour <= 6) {
        baseArrivals = 4; // night
      } else if (hour >= 7 && hour <= 20) {
        baseArrivals = 12; // day
      }

      // Weekend adjustment
      if (dayOfWeek === 0 || dayOfWeek === 6) {
        baseArrivals *= 0.8;
      }

      // Flu season (Oct-Feb)
      if (month >= 9 || month <= 1) {
        baseArrivals *= 1.4;
      }

      // Random variation
      const arrivals = Math.max(0, Math.floor(baseArrivals + (Math.random() - 0.5) * 6));

      data.push({ timestamp, arrivals });
    }
  }

  return data;
}

/**
 * Generate staff pool
 */
function generateStaffPool(): StaffMember[] {
  const staff: StaffMember[] = [];

  // Physicians
  for (let i = 0; i < 15; i++) {
    staff.push({
      id: `physician-${i}`,
      name: `Dr. Physician ${i}`,
      role: 'physician',
      skills: ['emergency_care', 'diagnosis', 'procedures'],
      shiftPreference: ['day', 'evening', 'night', 'any'][i % 4] as any,
      maxHoursPerWeek: 48,
      costPerHour: 150
    });
  }

  // Nurses
  for (let i = 0; i < 30; i++) {
    staff.push({
      id: `nurse-${i}`,
      name: `Nurse ${i}`,
      role: 'nurse',
      skills: ['patient_care', 'medication', 'monitoring'],
      shiftPreference: ['day', 'evening', 'night', 'any'][i % 4] as any,
      maxHoursPerWeek: 40,
      costPerHour: 50
    });
  }

  // Technicians
  for (let i = 0; i < 15; i++) {
    staff.push({
      id: `tech-${i}`,
      name: `Technician ${i}`,
      role: 'technician',
      skills: ['lab', 'imaging', 'equipment'],
      shiftPreference: ['day', 'evening', 'night', 'any'][i % 4] as any,
      maxHoursPerWeek: 40,
      costPerHour: 35
    });
  }

  // Specialists
  for (let i = 0; i < 10; i++) {
    staff.push({
      id: `specialist-${i}`,
      name: `Specialist ${i}`,
      role: 'specialist',
      skills: ['surgery', 'cardiology', 'neurology'],
      shiftPreference: 'day',
      maxHoursPerWeek: 50,
      costPerHour: 200
    });
  }

  return staff;
}

/**
 * Get acuity label
 */
function getAcuityLabel(acuity: number): string {
  const labels: Record<number, string> = {
    1: 'Critical',
    2: 'Urgent',
    3: 'Semi-urgent',
    4: 'Less urgent',
    5: 'Non-urgent'
  };

  return labels[acuity] || 'Unknown';
}

// Run example
main().catch(console.error);
