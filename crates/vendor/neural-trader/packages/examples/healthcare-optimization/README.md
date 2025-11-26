# @neural-trader/example-healthcare-optimization

> Healthcare optimization with self-learning patient forecasting and swarm-based staff scheduling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Comprehensive healthcare optimization system that combines machine learning, queue theory, and swarm intelligence to optimize hospital operations. Features patient arrival forecasting, staff scheduling, resource allocation, and AI-powered triage.

## Key Features

### 🔮 **Patient Arrival Forecasting**
- Self-learning prediction with uncertainty quantification
- Seasonal pattern detection (flu season, holidays, etc.)
- Confidence intervals for capacity planning
- Online learning from actual outcomes
- Uses `@neural-trader/predictor` with NAPI-RS

### 👥 **Staff Scheduling Optimization**
- Skill-based constraint satisfaction
- Shift preference matching
- Fair workload distribution
- Cost optimization
- Maximum hours and rest period enforcement

### 📊 **Queue Theory Optimization**
- M/M/c and M/G/c queuing models
- Erlang C calculations for wait time prediction
- Dynamic resource allocation
- Real-time utilization monitoring
- Abandonment rate tracking

### 🐝 **Swarm Intelligence**
- Parallel exploration of scheduling heuristics
- Multi-objective optimization
- Convergence detection
- Elite solution preservation
- Memory-persistent learning with AgentDB

### 🤖 **AI-Powered Triage**
- OpenRouter integration for intelligent triage
- Rule-based fallback system
- Acuity assignment (1-5 scale)
- Treatment path recommendation
- Confidence scoring

### 🔒 **Privacy-Preserving**
- Synthetic data generation only
- HIPAA/GDPR compliance modes
- Configurable anonymization
- No real patient data required

## Installation

```bash
npm install @neural-trader/example-healthcare-optimization
```

### Dependencies

```bash
npm install @neural-trader/predictor agentdb agentic-flow openai
```

## Quick Start

```typescript
import { HealthcareOptimizer } from '@neural-trader/example-healthcare-optimization';

// Configure optimizer
const optimizer = new HealthcareOptimizer({
  openRouterApiKey: process.env.OPENROUTER_API_KEY,
  agentdbPath: './data/healthcare.db',
  enableNapiRS: true,
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
    requiredSkillCoverage: ['emergency_care', 'patient_care']
  }
});

// Train forecaster
const historicalData = [
  { timestamp: new Date('2024-01-01T08:00:00'), arrivals: 12 },
  { timestamp: new Date('2024-01-01T09:00:00'), arrivals: 18 },
  // ... more data
];
await optimizer.trainForecaster(historicalData);

// Add staff
optimizer.addStaff({
  id: 'dr-smith',
  name: 'Dr. Smith',
  role: 'physician',
  skills: ['emergency_care', 'diagnosis'],
  shiftPreference: 'day',
  maxHoursPerWeek: 48,
  costPerHour: 150
});

// Add resources
optimizer.addResourcePool({
  id: 'exam-rooms',
  type: 'exam_room',
  capacity: 12,
  available: 12,
  utilizationTarget: 0.75
});

// Run optimization
const result = await optimizer.optimize(new Date());

console.log(`Quality Score: ${(result.qualityScore * 100).toFixed(1)}%`);
console.log(`Expected Wait Time: ${result.expectedWaitTime.toFixed(1)} minutes`);
console.log(`Total Cost: $${result.totalCost.toFixed(2)}`);
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   HealthcareOptimizer                       │
│                    (Main Controller)                        │
└───────────┬──────────────┬──────────────┬──────────────────┘
            │              │              │
    ┌───────▼──────┐ ┌────▼─────┐ ┌──────▼───────┐
    │   Arrival    │ │  Queue   │ │   Swarm      │
    │  Forecaster  │ │ Optimizer│ │ Coordinator  │
    │              │ │          │ │              │
    │ • Predictor  │ │ • M/M/c  │ │ • Agentic    │
    │ • AgentDB    │ │ • Erlang │ │   Flow       │
    │ • NAPI-RS    │ │   C      │ │ • AgentDB    │
    └──────────────┘ └──────────┘ └──────┬───────┘
                                          │
                                  ┌───────▼────────┐
                                  │   Scheduler    │
                                  │                │
                                  │ • Constraints  │
                                  │ • Optimization │
                                  └────────────────┘
```

### Workflow

1. **Historical Training**: Train forecaster on past arrival data
2. **Demand Forecasting**: Predict patient arrivals with uncertainty
3. **Swarm Optimization**: Explore scheduling solutions in parallel
4. **Queue Simulation**: Evaluate wait times and resource needs
5. **Resource Allocation**: Optimize capacity based on demand
6. **Continuous Learning**: Update models with actual outcomes

## API Reference

### HealthcareOptimizer

Main class for healthcare optimization.

#### Constructor

```typescript
new HealthcareOptimizer(config: HealthcareOptimizationConfig)
```

#### Methods

##### `trainForecaster(data)`
Train arrival forecaster on historical data.

```typescript
await optimizer.trainForecaster([
  { timestamp: new Date(), arrivals: 15 }
]);
```

##### `addStaff(staff)`
Add staff member to scheduling pool.

```typescript
optimizer.addStaff({
  id: 'staff-1',
  role: 'physician',
  skills: ['emergency_care'],
  shiftPreference: 'day',
  maxHoursPerWeek: 48,
  costPerHour: 150
});
```

##### `addResourcePool(pool)`
Add resource pool for queue management.

```typescript
optimizer.addResourcePool({
  id: 'exam-rooms',
  type: 'exam_room',
  capacity: 12,
  available: 12,
  utilizationTarget: 0.75
});
```

##### `optimize(startDate?)`
Run full optimization workflow.

```typescript
const result = await optimizer.optimize(new Date());
```

Returns:
- `schedule`: Optimized staff schedule
- `expectedWaitTime`: Predicted wait time (minutes)
- `expectedUtilization`: Resource utilization (0-1)
- `totalCost`: Total weekly cost
- `qualityScore`: Overall quality (0-1)

##### `triagePatient(patient)`
AI-powered triage assessment.

```typescript
const triage = await optimizer.triagePatient({
  chiefComplaint: 'Chest pain and shortness of breath'
});

console.log(triage.assignedAcuity); // 1 (critical)
console.log(triage.recommendedPath); // "immediate"
```

##### `updateWithActuals(timestamp, arrivals, waitTime)`
Update with actual data for learning.

```typescript
await optimizer.updateWithActuals(
  new Date(),
  15, // actual arrivals
  22  // actual wait time
);
```

### ArrivalForecaster

Patient arrival forecasting with seasonal patterns.

```typescript
import { ArrivalForecaster } from '@neural-trader/example-healthcare-optimization';

const forecaster = new ArrivalForecaster({
  agentdbPath: './data/forecaster.db',
  enableNapiRS: true,
  lookbackDays: 90,
  forecastHorizon: 24,
  confidenceLevel: 0.95
});

await forecaster.train(historicalData);
const forecast = await forecaster.forecast(new Date());
```

### Scheduler

Staff scheduling with constraints.

```typescript
import { Scheduler } from '@neural-trader/example-healthcare-optimization';

const scheduler = new Scheduler({
  planningHorizonDays: 7,
  shiftDuration: 8,
  costPerConstraintViolation: 1000
});

scheduler.addStaff(staff);
const schedule = await scheduler.generateSchedule(forecasts, constraints, startDate);
```

### QueueOptimizer

Queue theory-based optimization.

```typescript
import { QueueOptimizer } from '@neural-trader/example-healthcare-optimization';

const optimizer = new QueueOptimizer({
  targetUtilization: 0.75,
  maxWaitTime: 30,
  abandonmentThreshold: 120,
  reallocateInterval: 60
});

optimizer.addPatient(patient);
const recommendations = optimizer.optimizeResources(objective);
```

### SwarmCoordinator

Swarm intelligence for scheduling.

```typescript
import { SwarmCoordinator } from '@neural-trader/example-healthcare-optimization';

const swarm = new SwarmCoordinator(
  {
    populationSize: 20,
    maxIterations: 50,
    explorationRate: 0.3,
    convergenceThreshold: 0.01,
    elitismRate: 0.2
  },
  './data/swarm.db'
);

const result = await swarm.optimize(forecasts, constraints, objective, config, startDate);
```

## Configuration

### Optimization Objectives

Balance multiple objectives with weights (must sum to 1.0):

```typescript
optimization: {
  minimizeWaitTime: 0.4,      // Reduce patient wait times
  maximizeUtilization: 0.3,   // Increase resource efficiency
  minimizeCost: 0.2,          // Reduce staffing costs
  maximizePatientOutcomes: 0.1 // Improve care quality
}
```

### Scheduling Constraints

```typescript
constraints: {
  minStaffPerShift: {
    physician: 2,    // Minimum physicians per shift
    nurse: 4,        // Minimum nurses per shift
    technician: 2,   // Minimum technicians per shift
    specialist: 1    // Minimum specialists per shift
  },
  maxConsecutiveHours: 16,    // Maximum consecutive work hours
  minRestBetweenShifts: 8,    // Minimum rest period (hours)
  requiredSkillCoverage: [    // Skills that must be covered
    'emergency_care',
    'patient_care',
    'diagnosis'
  ]
}
```

### Swarm Parameters

```typescript
swarm: {
  populationSize: 20,         // Number of agents in swarm
  maxIterations: 50,          // Maximum optimization iterations
  explorationRate: 0.3,       // Exploration vs exploitation (0-1)
  convergenceThreshold: 0.01, // Stop when improvement < threshold
  elitismRate: 0.2            // % of top solutions to preserve
}
```

## Performance Metrics

### Forecasting Accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Prediction error with outlier penalty
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

### Schedule Quality
- **Coverage Score** (0-1): % of required positions filled
- **Fairness Score** (0-1): Equality of workload distribution
- **Constraint Violations**: Number of unmet constraints

### Queue Performance
- **Average Wait Time**: Mean patient wait time (minutes)
- **Max Wait Time**: Longest patient wait time
- **Utilization**: Resource usage (0-1)
- **Throughput**: Patients served per hour
- **Abandonment Rate**: % of patients leaving before service

## Use Cases

### Emergency Department
- Forecast patient arrivals by hour
- Optimize ED staff scheduling
- Minimize wait times
- Balance resource utilization

### Operating Room
- Schedule surgical staff
- Optimize OR allocation
- Manage turnover times
- Balance surgical volume

### Hospital-Wide
- Cross-departmental staffing
- Seasonal pattern adaptation (flu season)
- Weekend vs weekday optimization
- Holiday coverage planning

## Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm test arrival-forecaster
npm test scheduler
npm test queue-optimizer
npm test swarm

# Watch mode
npm test -- --watch
```

## Examples

```bash
# Run basic example
npm run example

# With OpenRouter AI triage
OPENROUTER_API_KEY=your_key npm run example

# Development mode
npm run dev
```

## Performance

### Benchmarks (on typical workstation)

- **Forecasting**: 100 predictions/second (NAPI-RS)
- **Scheduling**: 7-day schedule in 2-5 seconds
- **Swarm**: 50 iterations with 20 agents in 30-60 seconds
- **Queue Simulation**: 1000 patients in <1 second

### Scalability

- **Staff**: Tested up to 200 staff members
- **Shifts**: 500+ shifts per week
- **Forecasting**: 365 days history, 48 hours ahead
- **Swarm**: 100 agents, 200 iterations

## Privacy & Compliance

### Synthetic Data Only
This example uses **only synthetic data**. No real patient information is required or should be used.

### HIPAA Compliance Mode
```typescript
privacy: {
  useSyntheticData: true,
  anonymizationLevel: 'full',
  dataRetentionDays: 90,
  complianceMode: 'hipaa'
}
```

### GDPR Compliance Mode
```typescript
privacy: {
  useSyntheticData: true,
  anonymizationLevel: 'full',
  dataRetentionDays: 30,
  complianceMode: 'gdpr'
}
```

## Limitations

- Uses synthetic data only (not production-ready for real patients)
- Simplified queue models (real EDs have complex flows)
- No integration with EMR systems
- AI triage requires OpenRouter API key
- Swarm optimization can be computationally intensive

## Future Enhancements

- [ ] Multi-site optimization
- [ ] Integration with EMR systems
- [ ] Real-time dashboard
- [ ] Mobile app for staff
- [ ] Advanced queue networks
- [ ] ICU capacity planning
- [ ] Ambulance dispatch optimization
- [ ] Federated learning across hospitals

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT © Neural Trader

## Related Packages

- [@neural-trader/predictor](../../core/predictor) - ML prediction engine
- [@neural-trader/risk-manager](../risk-manager) - Risk management examples
- [@neural-trader/portfolio-optimizer](../portfolio-optimizer) - Portfolio optimization

## Support

- 📚 [Documentation](https://neural-trader.dev)
- 🐛 [Issues](https://github.com/neural-trader/neural-trader/issues)
- 💬 [Discussions](https://github.com/neural-trader/neural-trader/discussions)

## Citation

If you use this in research, please cite:

```bibtex
@software{neural_trader_healthcare_2024,
  title = {Neural Trader Healthcare Optimization},
  author = {Neural Trader Team},
  year = {2024},
  url = {https://github.com/neural-trader/neural-trader}
}
```

---

**Built with ❤️ using @neural-trader/predictor, AgentDB, and agentic-flow**
