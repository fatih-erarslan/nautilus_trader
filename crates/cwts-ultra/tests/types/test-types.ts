/**
 * Comprehensive Type Definitions for TDD Framework
 * Supporting multi-language component validation and mathematical rigor
 */

// Core test result types
export interface TestResult {
  component: string;
  success: boolean;
  details: any;
  timestamp: Date;
  duration: number;
  metadata?: TestMetadata;
}

export interface TestMetadata {
  testSuite: string;
  testRunner: string;
  environment: string;
  version: string;
  tags: string[];
}

// Component validation results
export interface ComponentValidationResult {
  compilationSuccess: boolean;
  testsPassed: boolean;
  memoryLeaks: string[];
  unsafeCodeBlocks: string[];
  performanceMetrics: PerformanceMetrics;
  typeCheckPassed?: boolean;
  lintingPassed?: boolean;
  codeQuality?: CodeQualityMetrics;
  memoryEfficiency?: number;
  performanceBenchmarks?: BenchmarkResult;
}

export interface PerformanceMetrics {
  executionTime: number;
  memoryUsage: number;
  cpuUsage: number;
  throughput?: number;
  latency?: number;
  allocations?: number;
  deallocations?: number;
}

export interface CodeQualityMetrics {
  score: number;
  issues: CodeIssue[];
  complexity: ComplexityMetrics;
  maintainabilityIndex: number;
  technicalDebt: number;
}

export interface CodeIssue {
  severity: 'error' | 'warning' | 'info';
  type: string;
  message: string;
  file: string;
  line: number;
  column: number;
}

export interface ComplexityMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  halsteadMetrics: HalsteadMetrics;
  linesOfCode: number;
  maintainabilityIndex: number;
}

export interface HalsteadMetrics {
  volume: number;
  difficulty: number;
  effort: number;
  timeRequiredToProgram: number;
  numberOfDeliveredBugs: number;
}

export interface BenchmarkResult {
  passed: boolean;
  benchmarks: Benchmark[];
  summary: BenchmarkSummary;
}

export interface Benchmark {
  name: string;
  duration: number;
  iterations: number;
  throughput: number;
  memoryUsage: number;
  passed: boolean;
}

export interface BenchmarkSummary {
  totalBenchmarks: number;
  passedBenchmarks: number;
  averageDuration: number;
  totalThroughput: number;
  memoryEfficiency: number;
}

// Reproducibility testing
export interface ReproducibilityResult {
  deterministicResults: boolean;
  seedConsistency: boolean;
  environmentIsolation: boolean;
  reproducibilityScore: number;
  testRuns: ReproducibilityTestRun[];
}

export interface ReproducibilityTestRun {
  runId: string;
  seed: number;
  hash: string;
  seedValid: boolean;
  environmentClean: boolean;
  success: boolean;
  duration: number;
  memoryUsage: number;
}

// Mathematical validation types
export interface MathematicalValidationResult {
  algorithmName: string;
  mathematicallyValid: boolean;
  rigorScore: number;
  precisionError: number;
  boundaryTests: BoundaryTestResult[];
  performanceMetrics: MathematicalPerformanceMetrics;
  statisticalValidation: StatisticalValidationResult;
  numericalStability: NumericalStabilityResult;
}

export interface BoundaryTestResult {
  condition: string;
  input: any;
  expectedOutput: any;
  actualOutput: any;
  passed: boolean;
  error?: string;
  tolerance: number;
}

export interface MathematicalPerformanceMetrics {
  executionTime: number;
  memoryUsage: number;
  numericalAccuracy: number;
  convergenceRate: number;
  iterationsToConverge: number;
  stabilityIndex: number;
}

export interface StatisticalValidationResult {
  normalityTest: StatisticalTest;
  stationarityTest: StatisticalTest;
  autocorrelationTest: StatisticalTest;
  distributionFit: DistributionFitResult;
  hypothesisTests: HypothesisTestResult[];
}

export interface StatisticalTest {
  testName: string;
  statistic: number;
  pValue: number;
  criticalValue: number;
  degreesOfFreedom?: number;
  passed: boolean;
  confidenceLevel: number;
}

export interface DistributionFitResult {
  distributions: DistributionTest[];
  bestFit: string;
  goodnessOfFit: number;
  parameters: { [key: string]: number };
}

export interface DistributionTest {
  name: string;
  parameters: { [key: string]: number };
  logLikelihood: number;
  aic: number;
  bic: number;
  goodnessOfFit: number;
}

export interface HypothesisTestResult {
  hypothesis: string;
  nullHypothesis: string;
  alternativeHypothesis: string;
  testStatistic: number;
  pValue: number;
  criticalValue: number;
  rejected: boolean;
  power: number;
  effectSize: number;
  confidenceInterval: ConfidenceInterval;
  sampleSize: number;
}

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  level: number;
}

export interface NumericalStabilityResult {
  stable: boolean;
  conditionNumber: number;
  errorPropagation: number;
  catastrophicCancellation: boolean;
  roundoffErrors: number[];
  stabilityIndex: number;
  convergenceAnalysis: ConvergenceAnalysis;
}

export interface ConvergenceAnalysis {
  converged: boolean;
  convergenceRate: number;
  iterations: number;
  residualNorm: number;
  oscillationDetected: boolean;
  monotonicity: boolean;
}

// Visual validation types
export interface VisualValidationResult {
  browser: string;
  passed: boolean;
  screenshots: ScreenshotResult[];
  consoleErrors: ConsoleError[];
  performanceMetrics: BrowserPerformanceMetrics;
  accessibilityResults: AccessibilityResult[];
  responsiveTests: ResponsiveTestResult[];
}

export interface ScreenshotResult {
  scenario: string;
  screenshotPath: string;
  baselinePath: string;
  diffPath: string;
  pixelDifference: number;
  passed: boolean;
  regressionDetected: boolean;
}

export interface ConsoleError {
  type: 'error' | 'warning' | 'info';
  message: string;
  source: string;
  line: number;
  column: number;
  timestamp: Date;
  stackTrace?: string;
}

export interface BrowserPerformanceMetrics {
  firstPaint: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  firstInputDelay: number;
  cumulativeLayoutShift: number;
  timeToInteractive: number;
  totalBlockingTime: number;
  memoryUsage: MemoryUsage;
  networkUsage: NetworkUsage;
}

export interface MemoryUsage {
  used: number;
  total: number;
  limit: number;
  heapUsed: number;
  heapTotal: number;
  external: number;
}

export interface NetworkUsage {
  totalRequests: number;
  totalBytes: number;
  averageLatency: number;
  failedRequests: number;
  cachedRequests: number;
}

export interface AccessibilityResult {
  rule: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  help: string;
  helpUrl: string;
  nodes: AccessibilityNode[];
}

export interface AccessibilityNode {
  target: string[];
  html: string;
  failureSummary: string;
  element: string;
}

export interface ResponsiveTestResult {
  viewport: ViewportConfig;
  passed: boolean;
  layoutTests: LayoutTest[];
  elementTests: ElementTest[];
  screenshotPath: string;
}

export interface ViewportConfig {
  width: number;
  height: number;
  deviceScaleFactor: number;
  isMobile: boolean;
  hasTouch: boolean;
  name: string;
}

export interface LayoutTest {
  testName: string;
  selector: string;
  expectedBehavior: string;
  actualBehavior: string;
  passed: boolean;
  error?: string;
}

export interface ElementTest {
  element: string;
  property: string;
  expected: any;
  actual: any;
  passed: boolean;
  tolerance?: number;
}

// Integration test types
export interface IntegrationTestResult {
  testSuite: string;
  passed: boolean;
  components: ComponentIntegrationResult[];
  endToEndTests: E2ETestResult[];
  dataFlowTests: DataFlowTestResult[];
  synchronizationTests: SynchronizationTestResult[];
}

export interface ComponentIntegrationResult {
  componentA: string;
  componentB: string;
  interfaceType: string;
  communicationProtocol: string;
  latency: number;
  throughput: number;
  errorRate: number;
  passed: boolean;
  errors: string[];
}

export interface E2ETestResult {
  workflow: string;
  steps: WorkflowStep[];
  totalDuration: number;
  passed: boolean;
  criticalPath: string[];
  bottlenecks: Bottleneck[];
}

export interface WorkflowStep {
  stepName: string;
  component: string;
  duration: number;
  inputData: any;
  outputData: any;
  passed: boolean;
  error?: string;
}

export interface Bottleneck {
  component: string;
  operation: string;
  duration: number;
  impact: number;
  recommendation: string;
}

export interface DataFlowTestResult {
  pipeline: string;
  inputVolume: number;
  outputVolume: number;
  throughput: number;
  latency: number;
  dataIntegrity: boolean;
  transformationAccuracy: number;
  passed: boolean;
}

export interface SynchronizationTestResult {
  components: string[];
  clockSkew: number;
  eventOrdering: boolean;
  stateConsistency: boolean;
  conflictResolution: boolean;
  passed: boolean;
}

// Coverage analysis types
export interface CoverageAnalysisResult {
  overall: CoverageMetrics;
  byComponent: Map<string, CoverageMetrics>;
  byLanguage: Map<string, CoverageMetrics>;
  mathematicalValidation: MathematicalCoverageValidation;
  requirements: CoverageRequirements;
  gaps: CoverageGap[];
}

export interface CoverageMetrics {
  lines: CoverageDetail;
  branches: CoverageDetail;
  functions: CoverageDetail;
  statements: CoverageDetail;
  conditions: CoverageDetail;
  paths: CoverageDetail;
}

export interface CoverageDetail {
  total: number;
  covered: number;
  percentage: number;
  uncovered: UncoveredItem[];
}

export interface UncoveredItem {
  file: string;
  line: number;
  column: number;
  type: string;
  reason: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface MathematicalCoverageValidation {
  rigorScore: number;
  confidence: number;
  testSufficiency: TestSufficiencyAnalysis;
  boundaryValueAnalysis: BoundaryValueAnalysis;
  equivalenceClassAnalysis: EquivalenceClassAnalysis;
}

export interface TestSufficiencyAnalysis {
  totalTestCases: number;
  sufficientTestCases: number;
  missingTestCases: MissingTestCase[];
  redundantTestCases: RedundantTestCase[];
  effectivenessScore: number;
}

export interface MissingTestCase {
  category: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedEffort: number;
  component: string;
}

export interface RedundantTestCase {
  testCase: string;
  redundantWith: string[];
  reason: string;
  component: string;
}

export interface BoundaryValueAnalysis {
  totalBoundaries: number;
  testedBoundaries: number;
  coverage: number;
  missingBoundaries: BoundaryValue[];
}

export interface BoundaryValue {
  parameter: string;
  minValue: number;
  maxValue: number;
  tested: boolean;
  testCases: string[];
}

export interface EquivalenceClassAnalysis {
  totalClasses: number;
  testedClasses: number;
  coverage: number;
  missingClasses: EquivalenceClass[];
}

export interface EquivalenceClass {
  parameter: string;
  validClass: boolean;
  range: any;
  tested: boolean;
  testCases: string[];
}

export interface CoverageRequirements {
  lines: number;
  branches: number;
  functions: number;
  statements: number;
  mathematicalRigor: number;
  riskThreshold: number;
}

export interface CoverageGap {
  component: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  impact: number;
  recommendation: string;
  estimatedEffort: number;
}

// Security validation types
export interface SecurityValidationResult {
  passed: boolean;
  vulnerabilities: SecurityVulnerability[];
  securityScore: number;
  threatModel: ThreatModelResult;
  penetrationTests: PenetrationTestResult[];
  codeSecurityAnalysis: CodeSecurityResult;
}

export interface SecurityVulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  description: string;
  cwe: string;
  cvss: number;
  component: string;
  file: string;
  line: number;
  remediation: string;
}

export interface ThreatModelResult {
  threats: ThreatAnalysis[];
  riskScore: number;
  mitigations: MitigationStrategy[];
}

export interface ThreatAnalysis {
  threat: string;
  probability: number;
  impact: number;
  riskLevel: number;
  attackVectors: string[];
  assets: string[];
}

export interface MitigationStrategy {
  threat: string;
  strategy: string;
  effectiveness: number;
  implementation: string;
  cost: number;
}

export interface PenetrationTestResult {
  testName: string;
  target: string;
  success: boolean;
  findings: SecurityFinding[];
  recommendations: string[];
}

export interface SecurityFinding {
  type: string;
  severity: string;
  description: string;
  evidence: string;
  remediation: string;
}

export interface CodeSecurityResult {
  staticAnalysis: StaticSecurityAnalysis;
  dynamicAnalysis: DynamicSecurityAnalysis;
  dependencyAnalysis: DependencySecurityAnalysis;
}

export interface StaticSecurityAnalysis {
  issues: SecurityIssue[];
  rulesApplied: string[];
  coverage: number;
  falsePositives: number;
}

export interface SecurityIssue {
  rule: string;
  severity: string;
  file: string;
  line: number;
  message: string;
  cwe: string;
}

export interface DynamicSecurityAnalysis {
  testCases: SecurityTestCase[];
  vulnerabilities: string[];
  coverage: number;
}

export interface SecurityTestCase {
  name: string;
  type: string;
  passed: boolean;
  vulnerabilityDetected: boolean;
  details: string;
}

export interface DependencySecurityAnalysis {
  dependencies: DependencySecurity[];
  vulnerablePackages: number;
  totalPackages: number;
  riskScore: number;
}

export interface DependencySecurity {
  package: string;
  version: string;
  vulnerabilities: PackageVulnerability[];
  riskLevel: string;
}

export interface PackageVulnerability {
  id: string;
  severity: string;
  description: string;
  fixedIn: string;
  patchAvailable: boolean;
}

// Test execution and reporting types
export interface TestExecutionPlan {
  phases: TestPhase[];
  parallelExecution: boolean;
  timeout: number;
  retryPolicy: RetryPolicy;
  dependencies: TestDependency[];
}

export interface TestPhase {
  name: string;
  tests: TestDescriptor[];
  prerequisites: string[];
  timeout: number;
  parallel: boolean;
}

export interface TestDescriptor {
  id: string;
  name: string;
  type: string;
  component: string;
  priority: number;
  timeout: number;
  tags: string[];
  parameters: { [key: string]: any };
}

export interface RetryPolicy {
  maxRetries: number;
  backoffStrategy: 'linear' | 'exponential' | 'fixed';
  baseDelay: number;
  maxDelay: number;
  retryOnFailure: boolean;
  retryOnTimeout: boolean;
}

export interface TestDependency {
  test: string;
  dependsOn: string[];
  type: 'hard' | 'soft';
  optional: boolean;
}

export interface ComprehensiveTestReport {
  summary: TestSummary;
  componentResults: Map<string, ComponentValidationResult>;
  mathematicalValidation: Map<string, MathematicalValidationResult>;
  visualValidation: Map<string, VisualValidationResult>;
  integrationTests: IntegrationTestResult[];
  coverageAnalysis: CoverageAnalysisResult;
  securityValidation: SecurityValidationResult;
  performanceBenchmarks: BenchmarkResult[];
  qualityMetrics: QualityMetrics;
  recommendations: TestRecommendation[];
  timestamp: Date;
  duration: number;
  environment: TestEnvironment;
}

export interface TestSummary {
  totalTests: number;
  passedTests: number;
  failedTests: number;
  skippedTests: number;
  successRate: number;
  coverage: number;
  qualityScore: number;
  riskLevel: string;
}

export interface QualityMetrics {
  codeQuality: number;
  testQuality: number;
  maintainability: number;
  reliability: number;
  security: number;
  performance: number;
  overall: number;
}

export interface TestRecommendation {
  category: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  component: string;
  effort: number;
  impact: number;
  implementation: string;
}

export interface TestEnvironment {
  platform: string;
  architecture: string;
  runtime: string;
  version: string;
  dependencies: { [key: string]: string };
  configuration: { [key: string]: any };
}

// Utility types for test framework
export type TestStatus = 'pending' | 'running' | 'passed' | 'failed' | 'skipped' | 'timeout';
export type TestSeverity = 'low' | 'medium' | 'high' | 'critical';
export type TestCategory = 'unit' | 'integration' | 'e2e' | 'visual' | 'performance' | 'security' | 'mathematical';
export type ComponentType = 'rust' | 'python' | 'javascript' | 'wasm' | 'integration';
export type ValidationLevel = 'basic' | 'standard' | 'comprehensive' | 'mathematical' | 'scientific';