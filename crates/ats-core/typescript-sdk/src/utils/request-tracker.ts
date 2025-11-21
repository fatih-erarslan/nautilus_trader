/**
 * Request Tracking and Analytics
 * 
 * Tracks HTTP requests for performance monitoring, error analysis,
 * and API usage statistics.
 */

interface RequestRecord {
  id: string;
  url: string;
  method: string;
  startTime: number;
  endTime?: number;
  status?: number;
  bytes?: number;
  error?: string;
}

interface RequestMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  totalBytesTransferred: number;
  requestsPerSecond: number;
  errorRate: number;
  endpoints: Record<string, EndpointMetrics>;
}

interface EndpointMetrics {
  totalRequests: number;
  averageResponseTime: number;
  successRate: number;
  lastAccessed: number;
}

export class RequestTracker {
  private activeRequests = new Map<string, RequestRecord>();
  private completedRequests: RequestRecord[] = [];
  private maxHistorySize: number;

  constructor(maxHistorySize = 1000) {
    this.maxHistorySize = maxHistorySize;
  }

  /**
   * Start tracking a new request
   */
  public startRequest(url: string, method: string): string {
    const id = this.generateRequestId();
    const record: RequestRecord = {
      id,
      url,
      method,
      startTime: Date.now(),
    };

    this.activeRequests.set(id, record);
    return id;
  }

  /**
   * Complete request tracking
   */
  public completeRequest(
    id: string,
    status: number,
    bytes = 0,
    error?: string
  ): void {
    const record = this.activeRequests.get(id);
    if (!record) {
      console.warn(`Request ${id} not found in active requests`);
      return;
    }

    record.endTime = Date.now();
    record.status = status;
    record.bytes = bytes;
    record.error = error;

    // Move to completed requests
    this.activeRequests.delete(id);
    this.completedRequests.push(record);

    // Maintain history size
    if (this.completedRequests.length > this.maxHistorySize) {
      this.completedRequests.shift();
    }
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get comprehensive metrics
   */
  public getMetrics(): RequestMetrics {
    const now = Date.now();
    const timeWindow = 60000; // 1 minute
    const recentRequests = this.completedRequests.filter(
      req => req.endTime && (now - req.endTime) <= timeWindow
    );

    const totalRequests = this.completedRequests.length;
    const successfulRequests = this.completedRequests.filter(
      req => req.status && req.status >= 200 && req.status < 400
    ).length;
    const failedRequests = totalRequests - successfulRequests;

    const totalResponseTime = this.completedRequests.reduce(
      (sum, req) => sum + (req.endTime ? req.endTime - req.startTime : 0),
      0
    );
    const averageResponseTime = totalRequests > 0 ? totalResponseTime / totalRequests : 0;

    const totalBytesTransferred = this.completedRequests.reduce(
      (sum, req) => sum + (req.bytes || 0),
      0
    );

    const requestsPerSecond = recentRequests.length / (timeWindow / 1000);
    const errorRate = totalRequests > 0 ? failedRequests / totalRequests : 0;

    // Calculate per-endpoint metrics
    const endpoints: Record<string, EndpointMetrics> = {};
    
    this.completedRequests.forEach(req => {
      const endpoint = this.normalizeEndpoint(req.url, req.method);
      
      if (!endpoints[endpoint]) {
        endpoints[endpoint] = {
          totalRequests: 0,
          averageResponseTime: 0,
          successRate: 0,
          lastAccessed: 0,
        };
      }

      const endpointMetrics = endpoints[endpoint];
      const responseTime = req.endTime ? req.endTime - req.startTime : 0;
      const isSuccess = req.status && req.status >= 200 && req.status < 400;

      // Update metrics
      endpointMetrics.totalRequests++;
      endpointMetrics.averageResponseTime = 
        (endpointMetrics.averageResponseTime * (endpointMetrics.totalRequests - 1) + responseTime) / 
        endpointMetrics.totalRequests;
      endpointMetrics.successRate = 
        this.completedRequests
          .filter(r => this.normalizeEndpoint(r.url, r.method) === endpoint && r.status && r.status >= 200 && r.status < 400)
          .length / endpointMetrics.totalRequests;
      endpointMetrics.lastAccessed = Math.max(endpointMetrics.lastAccessed, req.startTime);
    });

    return {
      totalRequests,
      successfulRequests,
      failedRequests,
      averageResponseTime,
      totalBytesTransferred,
      requestsPerSecond,
      errorRate,
      endpoints,
    };
  }

  /**
   * Normalize endpoint URL for grouping (remove IDs, query params, etc.)
   */
  private normalizeEndpoint(url: string, method: string): string {
    let normalized = url;

    // Remove query parameters
    normalized = normalized.split('?')[0];

    // Replace common ID patterns with placeholders
    normalized = normalized.replace(/\/\d+/g, '/:id');
    normalized = normalized.replace(/\/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/g, '/:uuid');
    normalized = normalized.replace(/\/[a-zA-Z0-9_-]{20,}/g, '/:token');

    return `${method} ${normalized}`;
  }

  /**
   * Get active request count
   */
  public getActiveRequestCount(): number {
    return this.activeRequests.size;
  }

  /**
   * Get slowest requests
   */
  public getSlowestRequests(count = 10): Array<{
    url: string;
    method: string;
    responseTime: number;
    status?: number;
    timestamp: number;
  }> {
    return this.completedRequests
      .filter(req => req.endTime)
      .map(req => ({
        url: req.url,
        method: req.method,
        responseTime: req.endTime! - req.startTime,
        status: req.status,
        timestamp: req.startTime,
      }))
      .sort((a, b) => b.responseTime - a.responseTime)
      .slice(0, count);
  }

  /**
   * Get failed requests
   */
  public getFailedRequests(count = 10): Array<{
    url: string;
    method: string;
    status?: number;
    error?: string;
    timestamp: number;
  }> {
    return this.completedRequests
      .filter(req => req.error || (req.status && req.status >= 400))
      .map(req => ({
        url: req.url,
        method: req.method,
        status: req.status,
        error: req.error,
        timestamp: req.startTime,
      }))
      .slice(-count)
      .reverse();
  }

  /**
   * Get request timeline for visualization
   */
  public getRequestTimeline(timeWindow = 300000): Array<{
    timestamp: number;
    requestCount: number;
    errorCount: number;
    averageResponseTime: number;
  }> {
    const now = Date.now();
    const bucketSize = timeWindow / 60; // 60 data points
    const timeline: Array<{
      timestamp: number;
      requestCount: number;
      errorCount: number;
      averageResponseTime: number;
    }> = [];

    for (let i = 0; i < 60; i++) {
      const bucketStart = now - timeWindow + (i * bucketSize);
      const bucketEnd = bucketStart + bucketSize;

      const bucketRequests = this.completedRequests.filter(
        req => req.startTime >= bucketStart && req.startTime < bucketEnd
      );

      const requestCount = bucketRequests.length;
      const errorCount = bucketRequests.filter(
        req => req.error || (req.status && req.status >= 400)
      ).length;

      const totalResponseTime = bucketRequests.reduce(
        (sum, req) => sum + (req.endTime ? req.endTime - req.startTime : 0),
        0
      );
      const averageResponseTime = requestCount > 0 ? totalResponseTime / requestCount : 0;

      timeline.push({
        timestamp: bucketStart,
        requestCount,
        errorCount,
        averageResponseTime,
      });
    }

    return timeline;
  }

  /**
   * Clear completed request history
   */
  public clearHistory(): void {
    this.completedRequests = [];
  }

  /**
   * Cancel active request tracking
   */
  public cancelRequest(id: string): void {
    this.activeRequests.delete(id);
  }

  /**
   * Get current status summary
   */
  public getStatusSummary(): {
    active: number;
    completed: number;
    recentSuccessRate: number;
    recentAverageResponseTime: number;
  } {
    const now = Date.now();
    const recentWindow = 300000; // 5 minutes
    
    const recentRequests = this.completedRequests.filter(
      req => req.endTime && (now - req.endTime) <= recentWindow
    );

    const recentSuccessful = recentRequests.filter(
      req => req.status && req.status >= 200 && req.status < 400
    ).length;

    const recentTotalResponseTime = recentRequests.reduce(
      (sum, req) => sum + (req.endTime! - req.startTime),
      0
    );

    return {
      active: this.activeRequests.size,
      completed: this.completedRequests.length,
      recentSuccessRate: recentRequests.length > 0 ? recentSuccessful / recentRequests.length : 1,
      recentAverageResponseTime: recentRequests.length > 0 ? recentTotalResponseTime / recentRequests.length : 0,
    };
  }

  /**
   * Export request data for external analysis
   */
  public exportData(): {
    active: RequestRecord[];
    completed: RequestRecord[];
    exportTimestamp: number;
  } {
    return {
      active: Array.from(this.activeRequests.values()),
      completed: [...this.completedRequests],
      exportTimestamp: Date.now(),
    };
  }
}