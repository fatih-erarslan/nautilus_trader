/**
 * High-Performance REST API Client
 * 
 * Comprehensive REST client for ATS-Core API with advanced error handling,
 * circuit breaker patterns, retry logic, and performance monitoring.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import {
  AtsCoreClientConfig,
  ApiResponse,
  ModelConfigRequest,
  ModelConfigResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  ModelStatusRequest,
  ModelStatusResponse,
  CalibrationRequest,
  CalibrationResponse,
  BenchmarkRequest,
  BenchmarkResponse,
  HealthStatus,
  PerformanceMetrics,
  AtsCoreError,
  TimeoutError,
  ValidationError,
  CircuitBreakerError,
  ConnectionError,
} from '../types';
import { CircuitBreaker } from '../utils/circuit-breaker';
import { RetryHandler } from '../utils/retry-handler';
import { RequestTracker } from '../utils/request-tracker';

export class AtsCoreRestClient {
  private client: AxiosInstance;
  private config: AtsCoreClientConfig;
  private circuitBreaker: CircuitBreaker;
  private retryHandler: RetryHandler;
  private requestTracker: RequestTracker;

  constructor(config: AtsCoreClientConfig) {
    this.config = this.validateConfig(config);
    this.circuitBreaker = new CircuitBreaker(config.circuitBreaker);
    this.retryHandler = new RetryHandler(config.retryConfig);
    this.requestTracker = new RequestTracker();
    this.client = this.createAxiosInstance();
    
    this.setupCircuitBreakerEvents();
  }

  /**
   * Validate and sanitize client configuration
   */
  private validateConfig(config: AtsCoreClientConfig): AtsCoreClientConfig {
    if (!config.restApiUrl) {
      throw new ValidationError('REST API URL is required');
    }

    // Set defaults
    return {
      restTimeout: 30000,
      enableCompression: true,
      ...config,
      circuitBreaker: {
        enabled: true,
        failureThreshold: 5,
        resetTimeout: 60000,
        monitoringPeriod: 10000,
        ...config.circuitBreaker,
      },
      retryConfig: {
        enabled: true,
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffMultiplier: 2,
        ...config.retryConfig,
      },
    };
  }

  /**
   * Create and configure Axios instance
   */
  private createAxiosInstance(): AxiosInstance {
    const client = axios.create({
      baseURL: this.config.restApiUrl,
      timeout: this.config.restTimeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': '@ats-core/typescript-sdk/1.0.0',
      },
    });

    // Add request interceptor for authentication and monitoring
    client.interceptors.request.use(
      (config) => {
        // Add authentication headers
        if (this.config.apiKey) {
          config.headers['X-API-Key'] = this.config.apiKey;
        }
        if (this.config.bearerToken) {
          config.headers['Authorization'] = `Bearer ${this.config.bearerToken}`;
        }

        // Enable compression if configured
        if (this.config.enableCompression) {
          config.headers['Accept-Encoding'] = 'gzip, deflate, br';
        }

        // Add request tracking
        const requestId = this.requestTracker.startRequest(config.url || '', config.method || 'GET');
        config.metadata = { requestId };

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling and monitoring
    client.interceptors.response.use(
      (response) => {
        // Complete request tracking
        if (response.config.metadata?.requestId) {
          this.requestTracker.completeRequest(
            response.config.metadata.requestId,
            response.status,
            response.headers['content-length'] ? parseInt(response.headers['content-length']) : 0
          );
        }

        this.circuitBreaker.recordSuccess();
        return response;
      },
      (error) => {
        // Complete request tracking with error
        if (error.config?.metadata?.requestId) {
          this.requestTracker.completeRequest(
            error.config.metadata.requestId,
            error.response?.status || 0,
            0,
            error.message
          );
        }

        this.circuitBreaker.recordFailure();
        return Promise.reject(this.transformAxiosError(error));
      }
    );

    return client;
  }

  /**
   * Transform Axios error to ATS-Core error
   */
  private transformAxiosError(error: any): AtsCoreError {
    if (error.code === 'ECONNABORTED') {
      return new TimeoutError('Request timeout');
    }

    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return new ConnectionError('Connection failed');
    }

    if (error.response) {
      const { status, data } = error.response;
      
      if (data && typeof data === 'object' && data.error) {
        return new AtsCoreError(data.error.code, data.error.message, data.error.details);
      }

      return new AtsCoreError(
        `HTTP_${status}`,
        `HTTP ${status}: ${error.response.statusText}`,
        { status, url: error.config?.url }
      );
    }

    return new AtsCoreError('UNKNOWN_ERROR', error.message);
  }

  /**
   * Setup circuit breaker event handlers
   */
  private setupCircuitBreakerEvents(): void {
    this.circuitBreaker.on('open', () => {
      console.warn('⚠️ REST API circuit breaker opened');
    });

    this.circuitBreaker.on('closed', () => {
      console.log('✅ REST API circuit breaker closed');
    });
  }

  /**
   * Make HTTP request with circuit breaker and retry logic
   */
  private async makeRequest<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<ApiResponse<T>> {
    // Check circuit breaker
    if (this.circuitBreaker.isOpen()) {
      throw new CircuitBreakerError('REST API circuit breaker is open');
    }

    // Execute with retry logic
    return this.retryHandler.execute(async () => {
      const response: AxiosResponse<ApiResponse<T>> = await this.client.request({
        method,
        url: endpoint,
        data,
        ...config,
      });

      if (!response.data.success && response.data.error) {
        throw new AtsCoreError(
          response.data.error.code,
          response.data.error.message,
          response.data.error.details
        );
      }

      return response.data;
    });
  }

  // Health and Monitoring APIs

  /**
   * Get basic health status
   */
  public async getHealth(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/health');
  }

  /**
   * Get detailed health status
   */
  public async getDetailedHealth(): Promise<ApiResponse<HealthStatus>> {
    return this.makeRequest('GET', '/health/detailed');
  }

  /**
   * Get server performance metrics
   */
  public async getPerformanceMetrics(): Promise<ApiResponse<PerformanceMetrics>> {
    return this.makeRequest('GET', '/metrics/performance');
  }

  /**
   * Get system metrics
   */
  public async getSystemMetrics(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/metrics/system');
  }

  // Model Management APIs

  /**
   * List all models
   */
  public async listModels(page = 0, limit = 50): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/models', undefined, {
      params: { page, limit },
    });
  }

  /**
   * Create model configuration
   */
  public async createModel(config: ModelConfigRequest): Promise<ApiResponse<ModelConfigResponse>> {
    return this.makeRequest('POST', '/models', config);
  }

  /**
   * Get model details
   */
  public async getModel(modelId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', `/models/${modelId}`);
  }

  /**
   * Update model configuration
   */
  public async updateModel(modelId: string, config: ModelConfigRequest): Promise<ApiResponse<ModelConfigResponse>> {
    return this.makeRequest('PUT', `/models/${modelId}`, config);
  }

  /**
   * Delete model
   */
  public async deleteModel(modelId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('DELETE', `/models/${modelId}`);
  }

  /**
   * Get model status
   */
  public async getModelStatus(modelId: string, includeMetrics = false): Promise<ApiResponse<ModelStatusResponse>> {
    return this.makeRequest('GET', `/models/${modelId}/status`, undefined, {
      params: { include_metrics: includeMetrics },
    });
  }

  /**
   * Get model metrics
   */
  public async getModelMetrics(modelId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', `/models/${modelId}/metrics`);
  }

  // Prediction APIs

  /**
   * Make single prediction
   */
  public async predict(
    modelId: string,
    features: number[],
    confidenceLevels: number[]
  ): Promise<ApiResponse<any>> {
    return this.makeRequest('POST', `/predict/${modelId}`, {
      features,
      confidence_levels: confidenceLevels,
    });
  }

  /**
   * Make batch predictions
   */
  public async batchPredict(request: BatchPredictionRequest): Promise<ApiResponse<BatchPredictionResponse>> {
    return this.makeRequest('POST', `/predict/${request.model_id}/batch`, request);
  }

  /**
   * Get prediction results by request ID
   */
  public async getPredictionResults(requestId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', `/predict/results/${requestId}`);
  }

  // Calibration APIs

  /**
   * Start model calibration
   */
  public async startCalibration(request: CalibrationRequest): Promise<ApiResponse<CalibrationResponse>> {
    return this.makeRequest('POST', '/calibration', request);
  }

  /**
   * Get calibration status
   */
  public async getCalibrationStatus(calibrationId: string): Promise<ApiResponse<CalibrationResponse>> {
    return this.makeRequest('GET', `/calibration/${calibrationId}`);
  }

  /**
   * Get calibration results
   */
  public async getCalibrationResults(calibrationId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', `/calibration/${calibrationId}/results`);
  }

  /**
   * Cancel calibration
   */
  public async cancelCalibration(calibrationId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('POST', `/calibration/${calibrationId}/cancel`);
  }

  /**
   * Get calibration history
   */
  public async getCalibrationHistory(page = 0, limit = 50): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/calibration/history', undefined, {
      params: { page, limit },
    });
  }

  // Benchmarking APIs

  /**
   * Start performance benchmark
   */
  public async startBenchmark(request: BenchmarkRequest): Promise<ApiResponse<BenchmarkResponse>> {
    return this.makeRequest('POST', '/benchmark', request);
  }

  /**
   * Get benchmark status
   */
  public async getBenchmarkStatus(benchmarkId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', `/benchmark/${benchmarkId}`);
  }

  /**
   * Get benchmark results
   */
  public async getBenchmarkResults(benchmarkId: string): Promise<ApiResponse<BenchmarkResponse>> {
    return this.makeRequest('GET', `/benchmark/${benchmarkId}/results`);
  }

  /**
   * Cancel benchmark
   */
  public async cancelBenchmark(benchmarkId: string): Promise<ApiResponse<any>> {
    return this.makeRequest('POST', `/benchmark/${benchmarkId}/cancel`);
  }

  /**
   * Get benchmark history
   */
  public async getBenchmarkHistory(page = 0, limit = 50): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/benchmark/history', undefined, {
      params: { page, limit },
    });
  }

  // System Management APIs

  /**
   * Get system status
   */
  public async getSystemStatus(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/system/status');
  }

  /**
   * Get memory usage statistics
   */
  public async getMemoryUsage(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/system/memory');
  }

  /**
   * Get CPU usage statistics
   */
  public async getCpuUsage(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/system/cpu');
  }

  /**
   * Trigger garbage collection
   */
  public async triggerGarbageCollection(): Promise<ApiResponse<any>> {
    return this.makeRequest('POST', '/system/gc');
  }

  /**
   * Clear cache
   */
  public async clearCache(): Promise<ApiResponse<any>> {
    return this.makeRequest('POST', '/system/cache/clear');
  }

  // Administrative APIs

  /**
   * Get server configuration
   */
  public async getServerConfig(): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/admin/config');
  }

  /**
   * Update server configuration
   */
  public async updateServerConfig(config: any): Promise<ApiResponse<any>> {
    return this.makeRequest('PUT', '/admin/config', config);
  }

  /**
   * Get server logs
   */
  public async getLogs(lines = 100): Promise<ApiResponse<any>> {
    return this.makeRequest('GET', '/admin/logs', undefined, {
      params: { lines },
    });
  }

  // Utility Methods

  /**
   * Test API connectivity
   */
  public async testConnection(): Promise<boolean> {
    try {
      await this.getHealth();
      return true;
    } catch (error) {
      console.error('❌ API connection test failed:', error);
      return false;
    }
  }

  /**
   * Get client performance metrics
   */
  public getClientMetrics() {
    return {
      requestTracker: this.requestTracker.getMetrics(),
      circuitBreaker: {
        isOpen: this.circuitBreaker.isOpen(),
        failureCount: this.circuitBreaker.getFailureCount(),
        successCount: this.circuitBreaker.getSuccessCount(),
      },
      retryHandler: this.retryHandler.getMetrics(),
    };
  }

  /**
   * Reset circuit breaker
   */
  public resetCircuitBreaker(): void {
    this.circuitBreaker.reset();
  }

  /**
   * Set API key for authentication
   */
  public setApiKey(apiKey: string): void {
    this.config.apiKey = apiKey;
  }

  /**
   * Set bearer token for authentication
   */
  public setBearerToken(token: string): void {
    this.config.bearerToken = token;
  }

  /**
   * Update request timeout
   */
  public setTimeout(timeout: number): void {
    this.config.restTimeout = timeout;
    this.client.defaults.timeout = timeout;
  }

  /**
   * Get current configuration
   */
  public getConfig(): Readonly<AtsCoreClientConfig> {
    return Object.freeze({ ...this.config });
  }
}