/**
 * ATS-Core TypeScript SDK
 * 
 * High-performance client library for real-time conformal prediction streaming
 * with sub-25μs latency guarantees, comprehensive error handling, and monitoring.
 * 
 * @example
 * ```typescript
 * import { AtsCoreClient } from '@ats-core/typescript-sdk';
 * 
 * const client = new AtsCoreClient({
 *   restApiUrl: 'http://localhost:8081',
 *   websocketUrl: 'ws://localhost:8080',
 *   enableBinaryProtocol: true
 * });
 * 
 * await client.connect();
 * 
 * // Subscribe to real-time predictions
 * await client.subscribeToPredictions({
 *   model_id: 'lstm_model_1',
 *   confidence_levels: [0.95, 0.99],
 *   update_frequency: 100 // 100ms
 * });
 * 
 * client.on('prediction', (data) => {
 *   console.log('New prediction:', data.prediction);
 * });
 * ```
 */

export * from './types';
export * from './websocket/client';
export * from './rest/client';

import { AtsCoreWebSocketClient } from './websocket/client';
import { AtsCoreRestClient } from './rest/client';
import {
  AtsCoreClientConfig,
  ClientEventMap,
  SubscriptionConfig,
  SubscriptionStatus,
  ClientPerformanceMetrics,
  HealthStatus,
  PerformanceMetrics,
  ModelConfigRequest,
  ModelConfigResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  CalibrationRequest,
  CalibrationResponse,
  BenchmarkRequest,
  BenchmarkResponse,
  ApiResponse,
  ConnectionError,
  ValidationError,
  TimeoutError,
  CircuitBreakerError,
  AtsCoreError,
} from './types';
import EventEmitter from 'eventemitter3';

/**
 * Main ATS-Core client combining WebSocket and REST functionality
 */
export class AtsCoreClient extends EventEmitter<ClientEventMap> {
  private wsClient: AtsCoreWebSocketClient;
  private restClient: AtsCoreRestClient;
  private config: AtsCoreClientConfig;
  private isInitialized = false;

  constructor(config: AtsCoreClientConfig) {
    super();
    
    this.config = this.validateAndNormalizeConfig(config);
    this.wsClient = new AtsCoreWebSocketClient(this.config);
    this.restClient = new AtsCoreRestClient(this.config);
    
    this.setupEventForwarding();
  }

  /**
   * Validate and normalize client configuration
   */
  private validateAndNormalizeConfig(config: AtsCoreClientConfig): AtsCoreClientConfig {
    if (!config.websocketUrl) {
      throw new ValidationError('WebSocket URL is required');
    }

    if (!config.restApiUrl) {
      throw new ValidationError('REST API URL is required');
    }

    // Set sensible defaults
    return {
      restTimeout: 30000,
      websocketTimeout: 30000,
      reconnectAttempts: 5,
      reconnectDelay: 1000,
      enableBinaryProtocol: true,
      enableCompression: true,
      bufferSize: 1000,
      circuitBreaker: {
        enabled: true,
        failureThreshold: 5,
        resetTimeout: 60000,
        monitoringPeriod: 10000,
      },
      retryConfig: {
        enabled: true,
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffMultiplier: 2,
      },
      ...config,
    };
  }

  /**
   * Setup event forwarding from WebSocket client
   */
  private setupEventForwarding(): void {
    // Forward all WebSocket events to main client
    const forwardEvents: (keyof ClientEventMap)[] = [
      'connected',
      'disconnected',
      'error',
      'prediction',
      'batch_prediction',
      'model_config_update',
      'metrics_update',
      'reconnecting',
      'reconnected',
      'latency_warning',
      'circuit_breaker_open',
      'circuit_breaker_closed',
    ];

    forwardEvents.forEach(event => {
      this.wsClient.on(event, (...args: any[]) => {
        this.emit(event, ...args);
      });
    });
  }

  // Connection Management

  /**
   * Initialize and connect all services
   */
  public async connect(): Promise<void> {
    console.log('🚀 Initializing ATS-Core client...');
    
    try {
      // Test REST API connectivity first
      const isRestHealthy = await this.restClient.testConnection();
      if (!isRestHealthy) {
        throw new ConnectionError('REST API health check failed');
      }

      // Connect WebSocket for real-time streaming
      await this.wsClient.connect();
      
      this.isInitialized = true;
      console.log('✅ ATS-Core client connected successfully');
      
    } catch (error) {
      console.error('❌ Failed to connect ATS-Core client:', error);
      throw error;
    }
  }

  /**
   * Disconnect from all services
   */
  public async disconnect(): Promise<void> {
    console.log('📴 Disconnecting ATS-Core client...');
    
    await this.wsClient.disconnect();
    this.isInitialized = false;
    
    console.log('✅ ATS-Core client disconnected');
  }

  /**
   * Check if client is connected and ready
   */
  public isConnected(): boolean {
    return this.isInitialized && this.wsClient.isWebSocketConnected();
  }

  // Real-time Streaming

  /**
   * Subscribe to real-time predictions
   */
  public async subscribeToPredictions(config: SubscriptionConfig): Promise<void> {
    this.ensureConnected();
    await this.wsClient.subscribe(config);
  }

  /**
   * Unsubscribe from predictions
   */
  public async unsubscribeFromPredictions(modelId: string): Promise<void> {
    this.ensureConnected();
    await this.wsClient.unsubscribe(modelId);
  }

  /**
   * Get subscription status
   */
  public getSubscriptionStatus(modelId: string): SubscriptionStatus | undefined {
    return this.wsClient.getSubscriptionStatus(modelId);
  }

  /**
   * Get all subscription statuses
   */
  public getAllSubscriptionStatuses(): Map<string, SubscriptionStatus> {
    return this.wsClient.getAllSubscriptionStatuses();
  }

  // Health and Monitoring

  /**
   * Get system health status
   */
  public async getHealth(): Promise<ApiResponse<any>> {
    return this.restClient.getHealth();
  }

  /**
   * Get detailed health status
   */
  public async getDetailedHealth(): Promise<ApiResponse<HealthStatus>> {
    return this.restClient.getDetailedHealth();
  }

  /**
   * Get performance metrics
   */
  public async getPerformanceMetrics(): Promise<ApiResponse<PerformanceMetrics>> {
    return this.restClient.getPerformanceMetrics();
  }

  /**
   * Get client-side performance metrics
   */
  public getClientMetrics(): ClientPerformanceMetrics {
    return this.wsClient.getPerformanceMetrics();
  }

  // Model Management

  /**
   * List all available models
   */
  public async listModels(page = 0, limit = 50): Promise<ApiResponse<any>> {
    return this.restClient.listModels(page, limit);
  }

  /**
   * Create new model configuration
   */
  public async createModel(config: ModelConfigRequest): Promise<ApiResponse<ModelConfigResponse>> {
    return this.restClient.createModel(config);
  }

  /**
   * Get model details
   */
  public async getModel(modelId: string): Promise<ApiResponse<any>> {
    return this.restClient.getModel(modelId);
  }

  /**
   * Update model configuration
   */
  public async updateModel(modelId: string, config: ModelConfigRequest): Promise<ApiResponse<ModelConfigResponse>> {
    return this.restClient.updateModel(modelId, config);
  }

  /**
   * Delete model
   */
  public async deleteModel(modelId: string): Promise<ApiResponse<any>> {
    return this.restClient.deleteModel(modelId);
  }

  /**
   * Get model status
   */
  public async getModelStatus(modelId: string, includeMetrics = false): Promise<ApiResponse<any>> {
    return this.restClient.getModelStatus(modelId, includeMetrics);
  }

  // Predictions

  /**
   * Make single prediction
   */
  public async predict(
    modelId: string,
    features: number[],
    confidenceLevels: number[]
  ): Promise<ApiResponse<any>> {
    return this.restClient.predict(modelId, features, confidenceLevels);
  }

  /**
   * Make batch predictions
   */
  public async batchPredict(request: BatchPredictionRequest): Promise<ApiResponse<BatchPredictionResponse>> {
    return this.restClient.batchPredict(request);
  }

  // Calibration

  /**
   * Start model calibration
   */
  public async startCalibration(request: CalibrationRequest): Promise<ApiResponse<CalibrationResponse>> {
    return this.restClient.startCalibration(request);
  }

  /**
   * Get calibration status
   */
  public async getCalibrationStatus(calibrationId: string): Promise<ApiResponse<CalibrationResponse>> {
    return this.restClient.getCalibrationStatus(calibrationId);
  }

  /**
   * Get calibration results
   */
  public async getCalibrationResults(calibrationId: string): Promise<ApiResponse<any>> {
    return this.restClient.getCalibrationResults(calibrationId);
  }

  // Benchmarking

  /**
   * Start performance benchmark
   */
  public async startBenchmark(request: BenchmarkRequest): Promise<ApiResponse<any>> {
    return this.restClient.startBenchmark(request);
  }

  /**
   * Get benchmark results
   */
  public async getBenchmarkResults(benchmarkId: string): Promise<ApiResponse<any>> {
    return this.restClient.getBenchmarkResults(benchmarkId);
  }

  // Utility Methods

  /**
   * Test overall connectivity
   */
  public async testConnectivity(): Promise<{
    rest: boolean;
    websocket: boolean;
    overall: boolean;
  }> {
    const rest = await this.restClient.testConnection();
    const websocket = this.wsClient.isWebSocketConnected();
    
    return {
      rest,
      websocket,
      overall: rest && websocket,
    };
  }

  /**
   * Get comprehensive client statistics
   */
  public getClientStatistics() {
    return {
      connection: {
        isConnected: this.isConnected(),
        connectionId: this.wsClient.getConnectionId(),
      },
      websocket: this.wsClient.getPerformanceMetrics(),
      rest: this.restClient.getClientMetrics(),
      subscriptions: this.getAllSubscriptionStatuses(),
    };
  }

  /**
   * Reset all client state and reconnect
   */
  public async reset(): Promise<void> {
    console.log('🔄 Resetting ATS-Core client...');
    
    await this.disconnect();
    await this.connect();
    
    console.log('✅ ATS-Core client reset complete');
  }

  /**
   * Update authentication credentials
   */
  public setAuthentication(apiKey?: string, bearerToken?: string): void {
    if (apiKey) {
      this.restClient.setApiKey(apiKey);
      this.config.apiKey = apiKey;
    }
    
    if (bearerToken) {
      this.restClient.setBearerToken(bearerToken);
      this.config.bearerToken = bearerToken;
    }
  }

  /**
   * Update request timeout
   */
  public setTimeout(timeout: number): void {
    this.restClient.setTimeout(timeout);
    this.config.restTimeout = timeout;
  }

  /**
   * Get current configuration
   */
  public getConfiguration(): Readonly<AtsCoreClientConfig> {
    return Object.freeze({ ...this.config });
  }

  /**
   * Ensure client is connected before operations
   */
  private ensureConnected(): void {
    if (!this.isConnected()) {
      throw new ConnectionError('Client is not connected. Call connect() first.');
    }
  }

  // Static factory methods

  /**
   * Create client with minimal configuration
   */
  public static createBasic(
    restApiUrl: string,
    websocketUrl: string,
    options: Partial<AtsCoreClientConfig> = {}
  ): AtsCoreClient {
    return new AtsCoreClient({
      restApiUrl,
      websocketUrl,
      ...options,
    });
  }

  /**
   * Create client optimized for high-frequency trading
   */
  public static createHighFrequency(
    restApiUrl: string,
    websocketUrl: string,
    options: Partial<AtsCoreClientConfig> = {}
  ): AtsCoreClient {
    return new AtsCoreClient({
      restApiUrl,
      websocketUrl,
      enableBinaryProtocol: true,
      enableCompression: true,
      bufferSize: 10000,
      websocketTimeout: 5000,
      restTimeout: 5000,
      reconnectAttempts: 10,
      reconnectDelay: 100,
      circuitBreaker: {
        enabled: true,
        failureThreshold: 3,
        resetTimeout: 30000,
        monitoringPeriod: 5000,
      },
      retryConfig: {
        enabled: true,
        maxRetries: 2,
        baseDelay: 50,
        maxDelay: 1000,
        backoffMultiplier: 1.5,
      },
      ...options,
    });
  }

  /**
   * Create client with robust error handling for production
   */
  public static createProduction(
    restApiUrl: string,
    websocketUrl: string,
    options: Partial<AtsCoreClientConfig> = {}
  ): AtsCoreClient {
    return new AtsCoreClient({
      restApiUrl,
      websocketUrl,
      enableBinaryProtocol: true,
      enableCompression: true,
      bufferSize: 5000,
      websocketTimeout: 30000,
      restTimeout: 30000,
      reconnectAttempts: 10,
      reconnectDelay: 5000,
      circuitBreaker: {
        enabled: true,
        failureThreshold: 5,
        resetTimeout: 120000,
        monitoringPeriod: 30000,
      },
      retryConfig: {
        enabled: true,
        maxRetries: 5,
        baseDelay: 2000,
        maxDelay: 30000,
        backoffMultiplier: 2,
      },
      ...options,
    });
  }
}

// Re-export main client as default
export default AtsCoreClient;

// Version information
export const VERSION = '1.0.0';
export const SDK_NAME = '@ats-core/typescript-sdk';

// Utility function to check SDK compatibility with server
export async function checkCompatibility(client: AtsCoreClient): Promise<{
  compatible: boolean;
  clientVersion: string;
  serverVersion?: string;
  issues: string[];
}> {
  const issues: string[] = [];
  let serverVersion: string | undefined;

  try {
    const health = await client.getHealth();
    
    if (health.success && health.data?.version) {
      serverVersion = health.data.version;
      
      // Simple version compatibility check
      // In practice, this would be more sophisticated
      const clientMajor = parseInt(VERSION.split('.')[0]);
      const serverMajor = parseInt(serverVersion.split('.')[0]);
      
      if (clientMajor !== serverMajor) {
        issues.push('Major version mismatch between client and server');
      }
    } else {
      issues.push('Unable to retrieve server version');
    }
  } catch (error) {
    issues.push(`Health check failed: ${error.message}`);
  }

  return {
    compatible: issues.length === 0,
    clientVersion: VERSION,
    serverVersion,
    issues,
  };
}