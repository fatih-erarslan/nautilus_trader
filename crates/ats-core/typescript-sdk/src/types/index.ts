/**
 * TypeScript Type Definitions for ATS-Core API
 * 
 * Comprehensive type definitions matching the Rust backend API
 * for conformal prediction streaming and model management.
 */

// Core Data Types
export interface ConformalPredictionResult {
  point_prediction: number;
  prediction_intervals: PredictionInterval[];
  temperature: number;
  calibration_scores: number[];
}

export interface PredictionInterval {
  lower_bound: number;
  upper_bound: number;
  confidence: number;
}

export interface Confidence {
  0: number; // Tuple-like structure from Rust
}

// WebSocket Message Types
export type WebSocketMessage = 
  | SubscribeMessage
  | UnsubscribeMessage
  | PredictionUpdateMessage
  | BatchPredictionUpdateMessage
  | ModelConfigUpdateMessage
  | PingMessage
  | PongMessage
  | ErrorMessage
  | WelcomeMessage
  | MetricsUpdateMessage;

export interface SubscribeMessage {
  type: 'Subscribe';
  data: {
    model_id: string;
    confidence_levels: number[];
    update_frequency: number; // Duration in milliseconds
  };
}

export interface UnsubscribeMessage {
  type: 'Unsubscribe';
  data: {
    model_id: string;
  };
}

export interface PredictionUpdateMessage {
  type: 'PredictionUpdate';
  data: {
    model_id: string;
    prediction: ConformalPredictionResult;
    timestamp: string;
    latency_us: number;
  };
}

export interface BatchPredictionUpdateMessage {
  type: 'BatchPredictionUpdate';
  data: {
    model_id: string;
    predictions: ConformalPredictionResult[];
    timestamp: string;
    batch_latency_us: number;
  };
}

export interface ModelConfigUpdateMessage {
  type: 'ModelConfigUpdate';
  data: {
    model_id: string;
    config: any;
    version: number;
  };
}

export interface PingMessage {
  type: 'Ping';
  data: {
    timestamp: string;
  };
}

export interface PongMessage {
  type: 'Pong';
  data: {
    timestamp: string;
    server_time: string;
  };
}

export interface ErrorMessage {
  type: 'Error';
  data: {
    code: string;
    message: string;
    request_id?: string;
  };
}

export interface WelcomeMessage {
  type: 'Welcome';
  data: {
    client_id: string;
    server_version: string;
    supported_protocols: string[];
  };
}

export interface MetricsUpdateMessage {
  type: 'MetricsUpdate';
  data: {
    metrics: PerformanceMetrics;
    timestamp: string;
  };
}

// REST API Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  request_id: string;
  timestamp: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
  request_id?: string;
}

export interface ModelConfigRequest {
  model_id: string;
  model_type: string;
  confidence_levels: number[];
  temperature_config?: TemperatureConfig;
  parameters: Record<string, any>;
}

export interface TemperatureConfig {
  initial_temperature: number;
  learning_rate: number;
  max_temperature: number;
  min_temperature: number;
  update_frequency: number; // Duration in milliseconds
}

export interface ModelConfigResponse {
  config_id: string;
  model_id: string;
  status: ConfigurationStatus;
  config: ModelConfigRequest;
  created_at: string;
  updated_at: string;
}

export enum ConfigurationStatus {
  Pending = 'Pending',
  Active = 'Active',
  Inactive = 'Inactive',
  Error = 'Error'
}

export interface BatchPredictionRequest {
  model_id: string;
  features: number[][];
  confidence_levels: number[];
  options: PredictionOptions;
}

export interface PredictionOptions {
  use_simd: boolean;
  parallel_processing: boolean;
  timeout_ms?: number;
  include_metrics: boolean;
}

export interface BatchPredictionResponse {
  request_id: string;
  model_id: string;
  predictions: ConformalPredictionResult[];
  metrics?: ProcessingMetrics;
  timestamp: string;
}

export interface ProcessingMetrics {
  total_processing_time_us: number;
  avg_sample_time_us: number;
  memory_usage_bytes: number;
  cpu_usage_percent: number;
  simd_utilization?: number;
}

export interface ModelStatusRequest {
  model_id: string;
  include_metrics: boolean;
}

export interface ModelStatusResponse {
  model_id: string;
  status: ModelStatus;
  configuration?: ModelConfigRequest;
  metrics?: ModelMetrics;
  last_activity: string;
  uptime_seconds: number;
}

export enum ModelStatus {
  Initializing = 'Initializing',
  Ready = 'Ready',
  Processing = 'Processing',
  Error = 'Error',
  Offline = 'Offline'
}

export interface ModelMetrics {
  total_predictions: number;
  predictions_per_second: number;
  avg_latency_us: number;
  error_rate: number;
  memory_usage_mb: number;
  accuracy_metrics?: AccuracyMetrics;
}

export interface AccuracyMetrics {
  mae: number;
  rmse: number;
  coverage_rate: number;
  avg_interval_width: number;
}

export interface HealthStatus {
  status: ServiceStatus;
  websocket_status: ServiceStatus;
  rest_status: ServiceStatus;
  prediction_engine_status: ServiceStatus;
  memory_usage: MemoryMetrics;
  connection_metrics: ConnectionMetrics;
  timestamp: string;
}

export enum ServiceStatus {
  Healthy = 'Healthy',
  Degraded = 'Degraded',
  Unhealthy = 'Unhealthy',
  Offline = 'Offline'
}

export interface MemoryMetrics {
  total_allocated: number;
  used: number;
  available: number;
  peak_usage: number;
}

export interface ConnectionMetrics {
  active_websocket_connections: number;
  total_connections_served: number;
  average_connection_duration: number; // Duration in milliseconds
  connections_per_second: number;
}

export interface PerformanceMetrics {
  average_latency_us: number;
  p95_latency_us: number;
  p99_latency_us: number;
  max_latency_us: number;
  requests_per_second: number;
  error_rate: number;
  throughput_mbps: number;
  cpu_usage: number;
  memory_usage: number;
}

export interface CalibrationRequest {
  model_id: string;
  calibration_data: CalibrationSample[];
  confidence_levels: number[];
  method: CalibrationMethod;
}

export interface CalibrationSample {
  features: number[];
  true_value: number;
}

export enum CalibrationMethod {
  SplitConformal = 'SplitConformal',
  CrossConformal = 'CrossConformal',
  JackknifeePlus = 'JackknifeePlus',
  Adaptive = 'Adaptive'
}

export interface CalibrationResponse {
  calibration_id: string;
  model_id: string;
  status: CalibrationStatus;
  results?: CalibrationResults;
  timestamp: string;
}

export enum CalibrationStatus {
  Running = 'Running',
  Completed = 'Completed',
  Failed = 'Failed'
}

export interface CalibrationResults {
  confidence_levels: number[];
  coverage_rates: number[];
  interval_widths: number[];
  calibration_scores: number[];
  processing_time_ms: number;
}

export interface BenchmarkRequest {
  model_ids: string[];
  sample_count: number;
  confidence_levels: number[];
  options: BenchmarkOptions;
}

export interface BenchmarkOptions {
  memory_profiling: boolean;
  latency_distribution: boolean;
  warmup_iterations: number;
  test_iterations: number;
}

export interface BenchmarkResponse {
  benchmark_id: string;
  model_benchmarks: ModelBenchmark[];
  system_metrics: SystemMetrics;
  timestamp: string;
}

export interface ModelBenchmark {
  model_id: string;
  latency_stats: LatencyStats;
  throughput: ThroughputMetrics;
  memory_usage: MemoryUsage;
  error_stats: ErrorStats;
}

export interface LatencyStats {
  avg_us: number;
  median_us: number;
  p95_us: number;
  p99_us: number;
  max_us: number;
  std_dev_us: number;
}

export interface ThroughputMetrics {
  predictions_per_second: number;
  data_throughput_mbps: number;
  peak_throughput: number;
}

export interface MemoryUsage {
  peak_mb: number;
  avg_mb: number;
  allocations: number;
  deallocations: number;
}

export interface ErrorStats {
  total_errors: number;
  error_rate: number;
  error_breakdown: Record<string, number>;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_io?: DiskIOStats;
  network_io?: NetworkIOStats;
}

export interface DiskIOStats {
  bytes_read: number;
  bytes_written: number;
  read_ops: number;
  write_ops: number;
}

export interface NetworkIOStats {
  bytes_received: number;
  bytes_sent: number;
  packets_received: number;
  packets_sent: number;
}

// Configuration Types
export interface AtsCoreClientConfig {
  // REST API Configuration
  restApiUrl: string;
  restTimeout: number;
  
  // WebSocket Configuration  
  websocketUrl: string;
  websocketTimeout: number;
  reconnectAttempts: number;
  reconnectDelay: number;
  
  // Authentication
  apiKey?: string;
  bearerToken?: string;
  
  // Performance Options
  enableBinaryProtocol: boolean;
  enableCompression: boolean;
  bufferSize: number;
  
  // Circuit Breaker Configuration
  circuitBreaker: CircuitBreakerConfig;
  
  // Retry Configuration
  retryConfig: RetryConfig;
}

export interface CircuitBreakerConfig {
  enabled: boolean;
  failureThreshold: number;
  resetTimeout: number;
  monitoringPeriod: number;
}

export interface RetryConfig {
  enabled: boolean;
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
}

// Client Events
export interface ClientEventMap {
  'connected': [{ client_id: string }];
  'disconnected': [{ reason: string }];
  'error': [{ error: ApiError }];
  'prediction': [PredictionUpdateMessage['data']];
  'batch_prediction': [BatchPredictionUpdateMessage['data']];
  'model_config_update': [ModelConfigUpdateMessage['data']];
  'metrics_update': [MetricsUpdateMessage['data']];
  'reconnecting': [{ attempt: number }];
  'reconnected': [];
  'latency_warning': [{ latency_us: number; threshold_us: number }];
  'circuit_breaker_open': [{ service: string }];
  'circuit_breaker_closed': [{ service: string }];
}

// Subscription Management
export interface SubscriptionConfig {
  model_id: string;
  confidence_levels: number[];
  update_frequency: number;
  auto_reconnect: boolean;
  buffer_updates: boolean;
  max_buffer_size: number;
}

export interface SubscriptionStatus {
  model_id: string;
  active: boolean;
  last_update?: string;
  update_count: number;
  error_count: number;
  average_latency_us: number;
}

// Performance Monitoring
export interface ClientPerformanceMetrics {
  connection_uptime: number;
  messages_received: number;
  messages_sent: number;
  average_latency_us: number;
  error_rate: number;
  reconnection_count: number;
  bytes_transferred: number;
}

// Binary Protocol Support
export interface BinaryPredictionMessage {
  msg_type: number;
  model_id_hash: bigint;
  timestamp_ns: bigint;
  prediction: number;
  lower_bound: number;
  upper_bound: number;
  confidence: number;
  latency_ns: bigint;
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Error Types
export class AtsCoreError extends Error {
  public readonly code: string;
  public readonly details?: any;
  public readonly timestamp: Date;

  constructor(code: string, message: string, details?: any) {
    super(message);
    this.name = 'AtsCoreError';
    this.code = code;
    this.details = details;
    this.timestamp = new Date();
  }
}

export class ConnectionError extends AtsCoreError {
  constructor(message: string, details?: any) {
    super('CONNECTION_ERROR', message, details);
    this.name = 'ConnectionError';
  }
}

export class ValidationError extends AtsCoreError {
  constructor(message: string, details?: any) {
    super('VALIDATION_ERROR', message, details);
    this.name = 'ValidationError';
  }
}

export class TimeoutError extends AtsCoreError {
  constructor(message: string, details?: any) {
    super('TIMEOUT_ERROR', message, details);
    this.name = 'TimeoutError';
  }
}

export class CircuitBreakerError extends AtsCoreError {
  constructor(message: string, details?: any) {
    super('CIRCUIT_BREAKER_OPEN', message, details);
    this.name = 'CircuitBreakerError';
  }
}