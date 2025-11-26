/**
 * Performance Monitoring System
 * Comprehensive monitoring for Supabase operations and system health
 */

import { supabase, supabaseAdmin } from '../supabase.config'
import { RealtimeChannelManager } from '../real-time/channels'

export interface PerformanceMetric {
  timestamp: Date
  metric_type: string
  value: number
  entity_type: string
  entity_id: string
  metadata?: any
}

export interface SystemHealth {
  database_health: {
    connection_status: 'healthy' | 'degraded' | 'unhealthy'
    response_time_ms: number
    active_connections: number
    query_performance: any
  }
  realtime_health: {
    connection_status: 'connected' | 'connecting' | 'disconnected'
    active_subscriptions: number
    message_latency_ms: number
  }
  edge_functions_health: {
    status: 'operational' | 'degraded' | 'down'
    response_times: Record<string, number>
    error_rates: Record<string, number>
  }
  storage_health: {
    status: 'healthy' | 'degraded' | 'unhealthy'
    usage_percentage: number
    available_space_gb: number
  }
}

export class PerformanceMonitor {
  private metrics: PerformanceMetric[] = []
  private realtimeManager: RealtimeChannelManager
  private monitoringInterval?: NodeJS.Timeout
  private isMonitoring = false

  constructor() {
    this.realtimeManager = new RealtimeChannelManager()
  }

  // Start performance monitoring
  async startMonitoring(intervalMs: number = 60000): Promise<void> {
    if (this.isMonitoring) {
      console.warn('Performance monitoring already running')
      return
    }

    this.isMonitoring = true
    console.log('Starting performance monitoring...')

    // Set up real-time monitoring for critical events
    this.setupRealtimeMonitoring()

    // Start periodic health checks
    this.monitoringInterval = setInterval(async () => {
      await this.collectSystemMetrics()
    }, intervalMs)

    // Initial metrics collection
    await this.collectSystemMetrics()
  }

  // Stop performance monitoring
  stopMonitoring(): void {
    if (!this.isMonitoring) {
      return
    }

    this.isMonitoring = false
    console.log('Stopping performance monitoring...')

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval)
      this.monitoringInterval = undefined
    }

    this.realtimeManager.unsubscribeAll()
  }

  // Set up real-time monitoring
  private setupRealtimeMonitoring(): void {
    this.realtimeManager.setHandlers({
      onError: (error) => {
        this.recordMetric({
          timestamp: new Date(),
          metric_type: 'realtime_error',
          value: 1,
          entity_type: 'system',
          entity_id: 'realtime',
          metadata: { error: error.message }
        })
      },
      onSubscribed: (status) => {
        this.recordMetric({
          timestamp: new Date(),
          metric_type: 'subscription_status',
          value: status.includes('SUBSCRIBED') ? 1 : 0,
          entity_type: 'system',
          entity_id: 'realtime',
          metadata: { status }
        })
      }
    })

    // Monitor performance metrics table for real-time insights
    this.realtimeManager.subscribeToPerformanceMetrics()
  }

  // Collect comprehensive system metrics
  private async collectSystemMetrics(): Promise<void> {
    try {
      const health = await this.checkSystemHealth()
      
      // Record database health metrics
      this.recordMetric({
        timestamp: new Date(),
        metric_type: 'database_response_time',
        value: health.database_health.response_time_ms,
        entity_type: 'system',
        entity_id: 'database'
      })

      this.recordMetric({
        timestamp: new Date(),
        metric_type: 'database_connection_status',
        value: health.database_health.connection_status === 'healthy' ? 1 : 0,
        entity_type: 'system',
        entity_id: 'database'
      })

      // Record real-time health metrics
      this.recordMetric({
        timestamp: new Date(),
        metric_type: 'realtime_connection_status',
        value: health.realtime_health.connection_status === 'connected' ? 1 : 0,
        entity_type: 'system',
        entity_id: 'realtime'
      })

      this.recordMetric({
        timestamp: new Date(),
        metric_type: 'realtime_message_latency',
        value: health.realtime_health.message_latency_ms,
        entity_type: 'system',
        entity_id: 'realtime'
      })

      // Record edge functions health
      Object.entries(health.edge_functions_health.response_times).forEach(([func, responseTime]) => {
        this.recordMetric({
          timestamp: new Date(),
          metric_type: 'edge_function_response_time',
          value: responseTime,
          entity_type: 'edge_function',
          entity_id: func
        })
      })

      // Store metrics in database
      await this.persistMetrics()

    } catch (error) {
      console.error('Error collecting system metrics:', error)
      this.recordMetric({
        timestamp: new Date(),
        metric_type: 'monitoring_error',
        value: 1,
        entity_type: 'system',
        entity_id: 'monitor',
        metadata: { error: error.message }
      })
    }
  }

  // Check overall system health
  async checkSystemHealth(): Promise<SystemHealth> {
    const health: SystemHealth = {
      database_health: await this.checkDatabaseHealth(),
      realtime_health: await this.checkRealtimeHealth(),
      edge_functions_health: await this.checkEdgeFunctionsHealth(),
      storage_health: await this.checkStorageHealth()
    }

    return health
  }

  // Check database health
  private async checkDatabaseHealth(): Promise<SystemHealth['database_health']> {
    const startTime = Date.now()
    
    try {
      // Test basic connectivity
      const { data, error } = await supabase
        .from('profiles')
        .select('id')
        .limit(1)

      const responseTime = Date.now() - startTime

      if (error) {
        return {
          connection_status: 'unhealthy',
          response_time_ms: responseTime,
          active_connections: 0,
          query_performance: { error: error.message }
        }
      }

      // Test query performance with more complex query
      const complexQueryStart = Date.now()
      await supabase
        .from('performance_metrics')
        .select('*')
        .gte('timestamp', new Date(Date.now() - 60000).toISOString())
        .limit(100)
      
      const complexQueryTime = Date.now() - complexQueryStart

      return {
        connection_status: responseTime < 1000 ? 'healthy' : 'degraded',
        response_time_ms: responseTime,
        active_connections: 1, // Simplified - would query actual connection pool in production
        query_performance: {
          simple_query_ms: responseTime,
          complex_query_ms: complexQueryTime
        }
      }
    } catch (error) {
      return {
        connection_status: 'unhealthy',
        response_time_ms: Date.now() - startTime,
        active_connections: 0,
        query_performance: { error: error.message }
      }
    }
  }

  // Check real-time health
  private async checkRealtimeHealth(): Promise<SystemHealth['realtime_health']> {
    return {
      connection_status: 'connected', // Would check actual connection status
      active_subscriptions: this.realtimeManager.getActiveChannels().length,
      message_latency_ms: 50 // Would measure actual latency
    }
  }

  // Check edge functions health
  private async checkEdgeFunctionsHealth(): Promise<SystemHealth['edge_functions_health']> {
    const functions = ['market-data-processor', 'signal-generator', 'risk-calculator']
    const responseTimes: Record<string, number> = {}
    const errorRates: Record<string, number> = {}

    for (const func of functions) {
      try {
        const startTime = Date.now()
        
        // Test endpoint with minimal payload
        const response = await fetch(`/functions/v1/${func}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ test: true })
        })

        responseTimes[func] = Date.now() - startTime
        errorRates[func] = response.ok ? 0 : 1

      } catch (error) {
        responseTimes[func] = 5000 // Timeout value
        errorRates[func] = 1
      }
    }

    const avgResponseTime = Object.values(responseTimes).reduce((sum, time) => sum + time, 0) / functions.length
    const avgErrorRate = Object.values(errorRates).reduce((sum, rate) => sum + rate, 0) / functions.length

    return {
      status: avgResponseTime < 2000 && avgErrorRate < 0.1 ? 'operational' : 
              avgResponseTime < 5000 && avgErrorRate < 0.3 ? 'degraded' : 'down',
      response_times: responseTimes,
      error_rates: errorRates
    }
  }

  // Check storage health
  private async checkStorageHealth(): Promise<SystemHealth['storage_health']> {
    // Simplified storage health check
    // In production, would check actual storage usage via Supabase admin API
    return {
      status: 'healthy',
      usage_percentage: 25, // Mock value
      available_space_gb: 15 // Mock value
    }
  }

  // Record a performance metric
  private recordMetric(metric: PerformanceMetric): void {
    this.metrics.push(metric)
    
    // Keep only recent metrics in memory (last 1000)
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000)
    }
  }

  // Persist metrics to database
  private async persistMetrics(): Promise<void> {
    if (this.metrics.length === 0) return

    try {
      const metricsToStore = this.metrics.map(metric => ({
        entity_type: metric.entity_type,
        entity_id: metric.entity_id,
        metric_type: metric.metric_type,
        metric_value: metric.value,
        timestamp: metric.timestamp.toISOString(),
        metadata: metric.metadata || {}
      }))

      const { error } = await supabaseAdmin
        .from('performance_metrics')
        .insert(metricsToStore)

      if (error) {
        console.error('Failed to persist metrics:', error)
      } else {
        // Clear persisted metrics
        this.metrics = []
      }
    } catch (error) {
      console.error('Error persisting metrics:', error)
    }
  }

  // Get recent metrics
  getRecentMetrics(metricType?: string, limit: number = 100): PerformanceMetric[] {
    let filtered = this.metrics

    if (metricType) {
      filtered = filtered.filter(metric => metric.metric_type === metricType)
    }

    return filtered
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit)
  }

  // Get performance summary
  async getPerformanceSummary(timeRangeHours: number = 24): Promise<{
    database_avg_response_time: number
    realtime_uptime_percentage: number
    edge_functions_avg_response_time: number
    total_errors: number
    health_score: number
  }> {
    const cutoffTime = new Date(Date.now() - timeRangeHours * 60 * 60 * 1000).toISOString()

    try {
      // Query database for historical metrics
      const { data: metrics } = await supabase
        .from('performance_metrics')
        .select('*')
        .gte('timestamp', cutoffTime)

      if (!metrics || metrics.length === 0) {
        return {
          database_avg_response_time: 0,
          realtime_uptime_percentage: 100,
          edge_functions_avg_response_time: 0,
          total_errors: 0,
          health_score: 100
        }
      }

      // Calculate database metrics
      const dbResponseTimes = metrics
        .filter(m => m.metric_type === 'database_response_time')
        .map(m => m.metric_value)
      
      const avgDbResponseTime = dbResponseTimes.length > 0 ?
        dbResponseTimes.reduce((sum, time) => sum + time, 0) / dbResponseTimes.length : 0

      // Calculate real-time uptime
      const realtimeStatuses = metrics
        .filter(m => m.metric_type === 'realtime_connection_status')
        .map(m => m.metric_value)
      
      const realtimeUptime = realtimeStatuses.length > 0 ?
        (realtimeStatuses.filter(status => status === 1).length / realtimeStatuses.length) * 100 : 100

      // Calculate edge functions performance
      const edgeFunctionTimes = metrics
        .filter(m => m.metric_type === 'edge_function_response_time')
        .map(m => m.metric_value)
      
      const avgEdgeFunctionTime = edgeFunctionTimes.length > 0 ?
        edgeFunctionTimes.reduce((sum, time) => sum + time, 0) / edgeFunctionTimes.length : 0

      // Count errors
      const totalErrors = metrics
        .filter(m => m.metric_type.includes('error'))
        .length

      // Calculate overall health score
      const dbScore = avgDbResponseTime < 500 ? 100 : Math.max(0, 100 - (avgDbResponseTime - 500) / 10)
      const realtimeScore = realtimeUptime
      const edgeScore = avgEdgeFunctionTime < 1000 ? 100 : Math.max(0, 100 - (avgEdgeFunctionTime - 1000) / 20)
      const errorScore = Math.max(0, 100 - totalErrors * 5)

      const healthScore = (dbScore + realtimeScore + edgeScore + errorScore) / 4

      return {
        database_avg_response_time: avgDbResponseTime,
        realtime_uptime_percentage: realtimeUptime,
        edge_functions_avg_response_time: avgEdgeFunctionTime,
        total_errors: totalErrors,
        health_score: healthScore
      }
    } catch (error) {
      console.error('Error calculating performance summary:', error)
      return {
        database_avg_response_time: 0,
        realtime_uptime_percentage: 0,
        edge_functions_avg_response_time: 0,
        total_errors: 1,
        health_score: 0
      }
    }
  }

  // Create alert for performance issues
  async createPerformanceAlert(
    severity: 'info' | 'warning' | 'error' | 'critical',
    title: string,
    message: string,
    metadata?: any
  ): Promise<void> {
    try {
      await supabaseAdmin
        .from('alerts')
        .insert({
          user_id: '00000000-0000-0000-0000-000000000000', // System alerts
          title,
          message,
          severity,
          entity_type: 'system',
          entity_id: 'performance_monitor',
          metadata: metadata || {}
        })
    } catch (error) {
      console.error('Failed to create performance alert:', error)
    }
  }

  // Check for performance anomalies
  async checkPerformanceAnomalies(): Promise<void> {
    const summary = await this.getPerformanceSummary(1) // Last hour

    // Check database response time
    if (summary.database_avg_response_time > 2000) {
      await this.createPerformanceAlert(
        'warning',
        'High Database Response Time',
        `Average database response time is ${summary.database_avg_response_time.toFixed(0)}ms (threshold: 2000ms)`,
        { response_time: summary.database_avg_response_time }
      )
    }

    // Check real-time uptime
    if (summary.realtime_uptime_percentage < 95) {
      await this.createPerformanceAlert(
        'error',
        'Real-time Service Degradation',
        `Real-time uptime is ${summary.realtime_uptime_percentage.toFixed(1)}% (threshold: 95%)`,
        { uptime_percentage: summary.realtime_uptime_percentage }
      )
    }

    // Check overall health score
    if (summary.health_score < 70) {
      await this.createPerformanceAlert(
        summary.health_score < 50 ? 'critical' : 'warning',
        'System Health Degradation',
        `Overall system health score is ${summary.health_score.toFixed(0)}/100`,
        summary
      )
    }

    // Check error rate
    if (summary.total_errors > 10) {
      await this.createPerformanceAlert(
        'warning',
        'High Error Rate',
        `${summary.total_errors} errors detected in the last hour`,
        { error_count: summary.total_errors }
      )
    }
  }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor()
export default performanceMonitor