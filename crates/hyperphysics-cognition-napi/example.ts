/**
 * Bun.js Example: HyperPhysics Cognition System
 *
 * Run with: bun run example.ts
 */

import {
  initTracing,
  CognitionSystem,
  defaultConfig,
  CognitionPhase,
  nextPhase,
  phaseName,
  version
} from '.'

// Initialize tracing
initTracing()

console.log('🧠 HyperPhysics Cognition System')
console.log(`Version: ${version()}`)
console.log('================================\n')

// Create cognition system with default config
const config = defaultConfig()
console.log('Configuration:', JSON.stringify(config, null, 2))

const cognition = new CognitionSystem(config)
console.log('\n✅ Cognition system initialized\n')

// Test arousal modulation
console.log('Testing arousal modulation:')
cognition.setArousal(0.2)
console.log(`  Low arousal (sleep): ${cognition.getArousal()}`)
console.log(`  Healthy: ${cognition.isHealthy()}`)

cognition.setArousal(0.8)
console.log(`  High arousal (awake): ${cognition.getArousal()}`)
console.log(`  Healthy: ${cognition.isHealthy()}`)

// Test cognitive load
console.log('\nTesting cognitive load:')
cognition.setLoad(0.3)
console.log(`  Low load: ${cognition.getLoad()}`)

cognition.setLoad(0.9)
console.log(`  High load (overloaded): ${cognition.getLoad()}`)

// Test cognition phases
console.log('\nCognition phase cycle:')
let phase = CognitionPhase.Perceiving
for (let i = 0; i < 7; i++) {
  console.log(`  ${i}: ${phaseName(phase)}`)
  phase = nextPhase(phase)
}

// Simulate cognitive cycle
console.log('\n🔄 Simulating 40Hz cognitive cycle:')
setTimeout(() => {
  console.log('  Cycle 1: Perceiving sensory input')
  cognition.setArousal(0.7)
  cognition.setLoad(0.4)
}, 0)

setTimeout(() => {
  console.log('  Cycle 2: Cognizing representations')
  cognition.setLoad(0.6)
}, 25)

setTimeout(() => {
  console.log('  Cycle 3: Deliberating in neocortex')
  cognition.setLoad(0.8)
}, 50)

setTimeout(() => {
  console.log('  Cycle 4: Forming intentions (agency)')
  cognition.setLoad(0.5)
}, 75)

setTimeout(() => {
  console.log('  Cycle 5: Integrating consciousness')
  cognition.setLoad(0.4)
}, 100)

setTimeout(() => {
  console.log('  Cycle 6: Executing actions')
  cognition.setLoad(0.3)
  console.log(`\n✅ Cycle complete. System healthy: ${cognition.isHealthy()}`)
}, 125)

// Dream state simulation
setTimeout(() => {
  console.log('\n💤 Simulating dream state:')
  cognition.setArousal(0.2)  // Enter dream state (< 0.3)
  console.log(`  Arousal: ${cognition.getArousal()} (dreaming)`)
  console.log(`  Dream consolidation active...`)
}, 200)

setTimeout(() => {
  cognition.setArousal(0.7)  // Wake up
  console.log(`\n☀️ Waking up (arousal: ${cognition.getArousal()})`)
  console.log(`  System healthy: ${cognition.isHealthy()}`)
  console.log('\n✅ Demo complete!')
}, 300)
