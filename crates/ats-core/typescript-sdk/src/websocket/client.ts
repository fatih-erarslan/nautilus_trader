/**
 * High-Performance WebSocket Client
 * 
 * Real-time conformal prediction streaming client with sub-25μs latency optimization,
 * automatic reconnection, circuit breaker patterns, and comprehensive error handling.
 */

import ReconnectingWebSocket from 'reconnecting-websocket';
import EventEmitter from 'eventemitter3';
import {
  WebSocketMessage,
  SubscribeMessage,
  UnsubscribeMessage,
  PredictionUpdateMessage,
  ClientEventMap,
  AtsCoreClientConfig,
  SubscriptionConfig,
  SubscriptionStatus,
  ClientPerformanceMetrics,
  BinaryPredictionMessage,
  ConnectionError,
  TimeoutError,
  CircuitBreakerError,
  AtsCoreError
} from '../types';
import { CircuitBreaker } from '../utils/circuit-breaker';
import { LatencyTracker } from '../utils/latency-tracker';
import { BinaryProtocolHandler } from '../utils/binary-protocol';

export class AtsCoreWebSocketClient extends EventEmitter<ClientEventMap> {
  private ws: ReconnectingWebSocket | null = null;
  private config: AtsCoreClientConfig;
  private subscriptions = new Map<string, SubscriptionConfig>();
  private subscriptionStatuses = new Map<string, SubscriptionStatus>();
  private circuitBreaker: CircuitBreaker;
  private latencyTracker: LatencyTracker;
  private binaryHandler: BinaryProtocolHandler;
  private connectionId: string | null = null;
  private isConnected = false;
  private reconnectAttempt = 0;
  private performanceMetrics: ClientPerformanceMetrics;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageBuffer: WebSocketMessage[] = [];
  private maxBufferSize: number;

  constructor(config: AtsCoreClientConfig) {
    super();
    this.config = this.validateConfig(config);
    this.circuitBreaker = new CircuitBreaker(config.circuitBreaker);
    this.latencyTracker = new LatencyTracker();
    this.binaryHandler = new BinaryProtocolHandler();
    this.maxBufferSize = config.bufferSize || 1000;
    this.performanceMetrics = this.initializeMetrics();
    
    this.setupCircuitBreakerEvents();
  }

  /**
   * Validate and sanitize client configuration
   */
  private validateConfig(config: AtsCoreClientConfig): AtsCoreClientConfig {
    if (!config.websocketUrl) {
      throw new ValidationError('WebSocket URL is required');
    }

    if (!config.restApiUrl) {
      throw new ValidationError('REST API URL is required');
    }

    // Set defaults
    return {
      websocketTimeout: 30000,
      reconnectAttempts: 5,
      reconnectDelay: 1000,
      enableBinaryProtocol: true,
      enableCompression: true,
      bufferSize: 1000,
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
   * Initialize performance metrics
   */
  private initializeMetrics(): ClientPerformanceMetrics {
    return {
      connection_uptime: 0,
      messages_received: 0,
      messages_sent: 0,
      average_latency_us: 0,
      error_rate: 0,
      reconnection_count: 0,
      bytes_transferred: 0,
    };
  }

  /**
   * Connect to WebSocket server
   */
  public async connect(): Promise<void> {
    if (this.circuitBreaker.isOpen()) {
      throw new CircuitBreakerError('WebSocket circuit breaker is open');
    }

    try {
      console.log(`🔌 Connecting to WebSocket: ${this.config.websocketUrl}`);
      
      const wsOptions = {
        maxReconnectAttempts: this.config.reconnectAttempts,
        reconnectInterval: this.config.reconnectDelay,
        debug: false,
      };

      this.ws = new ReconnectingWebSocket(this.config.websocketUrl, [], wsOptions);
      this.setupWebSocketEventHandlers();
      
      // Wait for connection
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new TimeoutError('WebSocket connection timeout'));
        }, this.config.websocketTimeout);

        this.once('connected', () => {
          clearTimeout(timeout);
          resolve();
        });

        this.once('error', (error) => {
          clearTimeout(timeout);
          reject(error.error);
        });
      });

    } catch (error) {
      this.circuitBreaker.recordFailure();
      throw error;
    }
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupWebSocketEventHandlers(): void {
    if (!this.ws) return;

    this.ws.addEventListener('open', this.handleOpen.bind(this));
    this.ws.addEventListener('message', this.handleMessage.bind(this));
    this.ws.addEventListener('close', this.handleClose.bind(this));
    this.ws.addEventListener('error', this.handleError.bind(this));
  }

  /**
   * Handle WebSocket connection open
   */
  private handleOpen(): void {
    console.log('✅ WebSocket connected');
    this.isConnected = true;
    this.reconnectAttempt = 0;
    this.circuitBreaker.recordSuccess();
    this.startHeartbeat();
    this.flushMessageBuffer();
    
    // Reset performance metrics on new connection
    this.performanceMetrics.connection_uptime = Date.now();
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(event: MessageEvent): void {
    const startTime = performance.now();
    
    try {
      let message: WebSocketMessage;

      // Handle binary protocol for ultra-low latency
      if (event.data instanceof ArrayBuffer && this.config.enableBinaryProtocol) {
        const binaryMsg = this.binaryHandler.decode(event.data);
        message = this.convertBinaryToWebSocketMessage(binaryMsg);
      } else {
        message = JSON.parse(event.data);
      }

      this.processMessage(message);
      
      // Update performance metrics
      const processingTime = (performance.now() - startTime) * 1000; // Convert to microseconds
      this.latencyTracker.record(processingTime);
      this.performanceMetrics.messages_received++;
      this.performanceMetrics.bytes_transferred += event.data.byteLength || event.data.length;
      this.performanceMetrics.average_latency_us = this.latencyTracker.getAverageLatency();

    } catch (error) {
      console.error('❌ Failed to process message:', error);
      this.emit('error', { error: new AtsCoreError('MESSAGE_PROCESSING_ERROR', error.message) });
    }
  }

  /**
   * Process parsed WebSocket message
   */
  private processMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'Welcome':
        this.connectionId = message.data.client_id;
        console.log(`🎉 Connected with client ID: ${this.connectionId}`);
        this.emit('connected', { client_id: this.connectionId });
        break;

      case 'PredictionUpdate':
        this.handlePredictionUpdate(message);
        break;

      case 'BatchPredictionUpdate':
        this.handleBatchPredictionUpdate(message);
        break;

      case 'ModelConfigUpdate':
        this.emit('model_config_update', message.data);
        break;

      case 'MetricsUpdate':
        this.emit('metrics_update', message.data);
        break;

      case 'Pong':
        // Handle heartbeat response
        break;

      case 'Error':
        const error = new AtsCoreError(message.data.code, message.data.message);
        this.emit('error', { error });
        break;

      default:
        console.warn('⚠️ Unknown message type:', message);
    }
  }

  /**
   * Handle prediction update message
   */
  private handlePredictionUpdate(message: PredictionUpdateMessage): void {
    const { model_id, prediction, latency_us } = message.data;
    
    // Update subscription status
    const status = this.subscriptionStatuses.get(model_id);
    if (status) {
      status.update_count++;
      status.last_update = new Date().toISOString();
      status.average_latency_us = (status.average_latency_us + latency_us) / 2;
    }

    // Emit prediction event
    this.emit('prediction', message.data);

    // Check for latency warnings
    if (latency_us > 25) { // Sub-25μs target
      this.emit('latency_warning', { latency_us, threshold_us: 25 });
    }
  }

  /**
   * Handle batch prediction update message  
   */
  private handleBatchPredictionUpdate(message: BatchPredictionUpdateMessage): void {
    const { model_id } = message.data;
    
    // Update subscription status
    const status = this.subscriptionStatuses.get(model_id);
    if (status) {
      status.update_count += message.data.predictions.length;
      status.last_update = new Date().toISOString();
    }

    this.emit('batch_prediction', message.data);
  }

  /**
   * Convert binary message to WebSocket message
   */
  private convertBinaryToWebSocketMessage(binaryMsg: BinaryPredictionMessage): PredictionUpdateMessage {
    return {
      type: 'PredictionUpdate',
      data: {
        model_id: binaryMsg.model_id_hash.toString(), // In practice, would need proper ID mapping
        prediction: {
          point_prediction: binaryMsg.prediction,
          prediction_intervals: [{
            lower_bound: binaryMsg.lower_bound,
            upper_bound: binaryMsg.upper_bound,
            confidence: binaryMsg.confidence,
          }],
          temperature: 1.0,
          calibration_scores: [],
        },
        timestamp: new Date(Number(binaryMsg.timestamp_ns) / 1000000).toISOString(),
        latency_us: Number(binaryMsg.latency_ns) / 1000,
      },
    };
  }

  /**
   * Handle WebSocket close
   */
  private handleClose(event: CloseEvent): void {
    console.log(`📴 WebSocket closed: ${event.code} - ${event.reason}`);
    this.isConnected = false;
    this.stopHeartbeat();
    this.performanceMetrics.reconnection_count++;
    
    this.emit('disconnected', { reason: event.reason });
    
    // Clear subscription statuses on disconnect
    this.subscriptionStatuses.forEach(status => {
      status.active = false;
    });
  }

  /**
   * Handle WebSocket error
   */
  private handleError(event: Event): void {
    console.error('❌ WebSocket error:', event);
    this.circuitBreaker.recordFailure();
    
    const error = new ConnectionError('WebSocket connection error');
    this.emit('error', { error });
  }

  /**
   * Subscribe to model predictions
   */
  public async subscribe(config: SubscriptionConfig): Promise<void> {
    if (!this.isConnected) {
      if (config.buffer_updates) {
        // Buffer the subscription for later
        this.subscriptions.set(config.model_id, config);
        return;
      } else {
        throw new ConnectionError('WebSocket not connected');
      }
    }

    const message: SubscribeMessage = {
      type: 'Subscribe',
      data: {
        model_id: config.model_id,
        confidence_levels: config.confidence_levels,
        update_frequency: config.update_frequency,
      },
    };

    await this.sendMessage(message);
    
    // Store subscription config and initialize status
    this.subscriptions.set(config.model_id, config);
    this.subscriptionStatuses.set(config.model_id, {
      model_id: config.model_id,
      active: true,
      update_count: 0,
      error_count: 0,
      average_latency_us: 0,
    });

    console.log(`📡 Subscribed to model: ${config.model_id}`);
  }

  /**
   * Unsubscribe from model predictions
   */
  public async unsubscribe(modelId: string): Promise<void> {
    if (!this.isConnected) {
      throw new ConnectionError('WebSocket not connected');
    }

    const message: UnsubscribeMessage = {
      type: 'Unsubscribe',
      data: { model_id: modelId },
    };

    await this.sendMessage(message);
    
    // Remove subscription and status
    this.subscriptions.delete(modelId);
    this.subscriptionStatuses.delete(modelId);

    console.log(`📡 Unsubscribed from model: ${modelId}`);
  }

  /**
   * Send message to WebSocket server
   */
  private async sendMessage(message: WebSocketMessage): Promise<void> {
    if (!this.ws || !this.isConnected) {
      if (this.messageBuffer.length < this.maxBufferSize) {
        this.messageBuffer.push(message);
        return;
      } else {
        throw new ConnectionError('WebSocket not connected and buffer full');
      }
    }

    try {
      let data: string | ArrayBuffer;
      
      if (this.config.enableBinaryProtocol && message.type === 'PredictionUpdate') {
        // Use binary protocol for prediction updates
        data = this.binaryHandler.encode(message as any);
      } else {
        data = JSON.stringify(message);
      }

      this.ws.send(data);
      this.performanceMetrics.messages_sent++;
      
      if (typeof data === 'string') {
        this.performanceMetrics.bytes_transferred += data.length;
      } else {
        this.performanceMetrics.bytes_transferred += data.byteLength;
      }

    } catch (error) {
      throw new ConnectionError(`Failed to send message: ${error.message}`);
    }
  }

  /**
   * Flush buffered messages
   */
  private async flushMessageBuffer(): Promise<void> {
    while (this.messageBuffer.length > 0 && this.isConnected) {
      const message = this.messageBuffer.shift()!;
      try {
        await this.sendMessage(message);
      } catch (error) {
        console.error('❌ Failed to flush buffered message:', error);
        // Re-buffer if connection lost during flush
        this.messageBuffer.unshift(message);
        break;
      }
    }
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(async () => {
      if (this.isConnected) {
        try {
          await this.sendMessage({
            type: 'Ping',
            data: { timestamp: new Date().toISOString() },
          });
        } catch (error) {
          console.error('❌ Heartbeat failed:', error);
        }
      }
    }, 30000); // 30 second heartbeat
  }

  /**
   * Stop heartbeat mechanism
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Setup circuit breaker events
   */
  private setupCircuitBreakerEvents(): void {
    this.circuitBreaker.on('open', () => {
      console.warn('⚠️ WebSocket circuit breaker opened');
      this.emit('circuit_breaker_open', { service: 'websocket' });
    });

    this.circuitBreaker.on('halfOpen', () => {
      console.log('🔄 WebSocket circuit breaker half-open, testing...');
    });

    this.circuitBreaker.on('closed', () => {
      console.log('✅ WebSocket circuit breaker closed');
      this.emit('circuit_breaker_closed', { service: 'websocket' });
    });
  }

  /**
   * Get subscription status
   */
  public getSubscriptionStatus(modelId: string): SubscriptionStatus | undefined {
    return this.subscriptionStatuses.get(modelId);
  }

  /**
   * Get all subscription statuses
   */
  public getAllSubscriptionStatuses(): Map<string, SubscriptionStatus> {
    return new Map(this.subscriptionStatuses);
  }

  /**
   * Get current performance metrics
   */
  public getPerformanceMetrics(): ClientPerformanceMetrics {
    return {
      ...this.performanceMetrics,
      connection_uptime: this.performanceMetrics.connection_uptime 
        ? Date.now() - this.performanceMetrics.connection_uptime 
        : 0,
      error_rate: this.performanceMetrics.messages_received > 0 
        ? (this.subscriptionStatuses.size > 0 
            ? Array.from(this.subscriptionStatuses.values()).reduce((sum, status) => sum + status.error_count, 0) / this.performanceMetrics.messages_received
            : 0)
        : 0,
    };
  }

  /**
   * Check if connected
   */
  public isWebSocketConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Get connection ID
   */
  public getConnectionId(): string | null {
    return this.connectionId;
  }

  /**
   * Disconnect from WebSocket server
   */
  public async disconnect(): Promise<void> {
    console.log('📴 Disconnecting WebSocket...');
    
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.isConnected = false;
    this.connectionId = null;
    this.subscriptions.clear();
    this.subscriptionStatuses.clear();
    this.messageBuffer.length = 0;
    
    console.log('✅ WebSocket disconnected');
  }

  /**
   * Reconnect to WebSocket server
   */
  public async reconnect(): Promise<void> {
    await this.disconnect();
    await this.connect();
    
    // Restore subscriptions
    for (const [modelId, config] of this.subscriptions) {
      try {
        await this.subscribe(config);
      } catch (error) {
        console.error(`❌ Failed to restore subscription for ${modelId}:`, error);
      }
    }
  }
}