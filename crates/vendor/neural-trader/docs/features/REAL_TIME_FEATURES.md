# Real-time Features Documentation

Comprehensive guide to the real-time capabilities of the Neural Trading Platform using Supabase.

## Overview

The platform provides real-time data streaming for:
- Market data updates
- Trading signals generation
- Bot status monitoring
- Performance metrics
- Risk alerts
- Neural network training progress

## Architecture

```
Market Data Sources → Supabase Database → Real-time Subscriptions → Client Applications
                                    ↓
                            Edge Functions (Processing)
                                    ↓
                            E2B Sandboxes (Bot Execution)
```

## Real-time Channels

### 1. Market Data Channel

**Purpose**: Stream live market data updates

```typescript
import { RealtimeChannelManager } from './src/supabase/real-time/channels'

const channelManager = new RealtimeChannelManager(userId)

channelManager.setHandlers({
  onMarketUpdate: (data) => {
    console.log('Market update:', {
      symbol: data.symbols?.symbol,
      price: data.close,
      volume: data.volume,
      timestamp: data.timestamp
    })
    
    // Update UI with new price
    updatePriceDisplay(data.symbols.symbol, data.close)
    
    // Trigger technical analysis
    if (data.timeframe === '1m') {
      triggerTechnicalAnalysis(data)
    }
  }
})

// Subscribe to specific symbols
channelManager.subscribeToMarketData(['AAPL', 'GOOGL', 'TSLA'])
```

**Data Structure**:
```typescript
interface MarketDataUpdate {
  id: string
  symbol_id: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  timeframe: string
  symbols?: {
    symbol: string
    name: string
  }
}
```

### 2. Trading Signals Channel

**Purpose**: Receive AI-generated trading signals in real-time

```typescript
channelManager.setHandlers({
  onSignalGenerated: (signal) => {
    console.log('New trading signal:', {
      symbol: signal.symbols?.symbol,
      action: signal.action,
      strength: signal.signal_strength,
      reasoning: signal.reasoning
    })
    
    // Display signal in UI
    displayTradingSignal(signal)
    
    // Execute if auto-trading enabled
    if (autoTradingEnabled && signal.signal_strength > 0.7) {
      executeTrade(signal)
    }
  }
})

channelManager.subscribeToTradingSignals([botId])
```

**Data Structure**:
```typescript
interface TradingSignal {
  id: string
  bot_id: string
  symbol_id: string
  action: 'buy' | 'sell' | 'hold'
  signal_strength: number
  reasoning: string
  executed_at: string
  metadata: any
  symbols?: {
    symbol: string
    name: string
  }
}
```

### 3. Bot Status Channel

**Purpose**: Monitor trading bot health and status

```typescript
channelManager.setHandlers({
  onBotStatusChange: (bot) => {
    console.log('Bot status changed:', {
      name: bot.name,
      status: bot.status,
      performance: bot.performance_metrics
    })
    
    // Update bot dashboard
    updateBotStatus(bot.id, bot.status)
    
    // Send notifications for critical status changes
    if (bot.status === 'error') {
      showNotification('Bot Error', `${bot.name} encountered an error`)
    }
  }
})

channelManager.subscribeToBotStatus([botId1, botId2])
```

### 4. Alerts Channel

**Purpose**: Receive real-time risk and system alerts

```typescript
channelManager.setHandlers({
  onAlertTriggered: (alert) => {
    console.log('New alert:', alert)
    
    // Show notification based on severity
    switch (alert.severity) {
      case 'critical':
        showCriticalAlert(alert)
        break
      case 'warning':
        showWarningAlert(alert)
        break
      default:
        showInfoAlert(alert)
    }
  }
})

channelManager.subscribeToAlerts()
```

### 5. Performance Metrics Channel

**Purpose**: Real-time performance monitoring

```typescript
channelManager.setHandlers({
  onPerformanceUpdate: (metric) => {
    console.log('Performance update:', metric)
    
    // Update charts and dashboards
    updatePerformanceChart(metric.entity_type, metric.metric_type, metric.metric_value)
    
    // Check for performance alerts
    if (metric.metric_type === 'portfolio_var' && metric.metric_value > riskLimit) {
      triggerRiskAlert(metric)
    }
  }
})

channelManager.subscribeToPerformanceMetrics()
```

### 6. Neural Training Channel

**Purpose**: Monitor neural network training progress

```typescript
channelManager.setHandlers({
  onTrainingProgress: (progress) => {
    console.log('Training progress:', {
      model_id: progress.model_id,
      epoch: progress.epoch,
      loss: progress.loss,
      accuracy: progress.accuracy
    })
    
    // Update training progress bar
    updateTrainingProgress(progress.model_id, {
      epoch: progress.epoch,
      loss: progress.loss,
      accuracy: progress.accuracy
    })
    
    // Check if training completed
    if (progress.status === 'completed') {
      showTrainingCompleteNotification(progress.model_id)
    }
  }
})

channelManager.subscribeToNeuralTraining([modelId])
```

## Real-time Processing Pipeline

### 1. Data Ingestion

```typescript
// Market data processor edge function
export default async function marketDataProcessor(req: Request) {
  const marketData = await req.json()
  
  // Validate and process data
  const processed = await processMarketData(marketData)
  
  // Store in database (triggers real-time notification)
  await supabase
    .from('market_data')
    .insert(processed)
  
  // Process for signals if needed
  if (processed.timeframe === '1m') {
    await generateTradingSignals(processed)
  }
  
  return new Response(JSON.stringify({ success: true }))
}
```

### 2. Signal Generation

```typescript
// Automated signal generation with real-time updates
async function generateTradingSignals(marketData: MarketData) {
  // Get active bots for this symbol
  const { data: bots } = await supabase
    .from('trading_bots')
    .select('*')
    .eq('status', 'active')
    .contains('symbols', [marketData.symbol])
  
  for (const bot of bots || []) {
    // Generate signal using neural models
    const signal = await generateSignal(bot, marketData)
    
    if (signal.action !== 'hold') {
      // Store signal (triggers real-time notification)
      await supabase
        .from('bot_executions')
        .insert({
          bot_id: bot.id,
          symbol_id: marketData.symbol_id,
          action: signal.action,
          signal_strength: signal.strength,
          reasoning: signal.reasoning
        })
    }
  }
}
```

### 3. Risk Monitoring

```typescript
// Real-time risk monitoring with automatic alerts
async function monitorRisk(positionUpdate: PositionUpdate) {
  // Calculate updated portfolio risk
  const risk = await calculatePortfolioRisk(positionUpdate.account_id)
  
  // Store risk metric (triggers real-time notification)
  await supabase
    .from('performance_metrics')
    .insert({
      entity_type: 'account',
      entity_id: positionUpdate.account_id,
      metric_type: 'portfolio_var',
      metric_value: risk.var_95
    })
  
  // Create alert if risk exceeds limits
  if (risk.var_95 > risk.limits.max_var) {
    await supabase
      .from('alerts')
      .insert({
        user_id: positionUpdate.user_id,
        title: 'Risk Limit Exceeded',
        message: `Portfolio VaR (${risk.var_95}) exceeds limit (${risk.limits.max_var})`,
        severity: 'warning',
        entity_type: 'account',
        entity_id: positionUpdate.account_id
      })
  }
}
```

## Client Implementation Examples

### React Hook for Real-time Data

```typescript
import { useEffect, useState } from 'react'
import { RealtimeChannelManager } from './src/supabase/real-time/channels'

export function useRealTimeMarketData(symbols: string[]) {
  const [marketData, setMarketData] = useState<MarketDataUpdate[]>([])
  const [channelManager] = useState(() => new RealtimeChannelManager())
  
  useEffect(() => {
    channelManager.setHandlers({
      onMarketUpdate: (data) => {
        setMarketData(prev => {
          const updated = prev.filter(d => d.symbol_id !== data.symbol_id)
          return [data, ...updated].slice(0, 1000) // Keep last 1000 updates
        })
      }
    })
    
    channelManager.subscribeToMarketData(symbols)
    
    return () => {
      channelManager.unsubscribeAll()
    }
  }, [symbols])
  
  return marketData
}
```

### Vue Composition API

```typescript
import { ref, onMounted, onUnmounted } from 'vue'
import { RealtimeChannelManager } from './src/supabase/real-time/channels'

export function useRealTimeTradingSignals(botIds: string[]) {
  const signals = ref<TradingSignal[]>([])
  const channelManager = new RealtimeChannelManager()
  
  onMounted(() => {
    channelManager.setHandlers({
      onSignalGenerated: (signal) => {
        signals.value.unshift(signal)
        // Keep only recent signals
        if (signals.value.length > 100) {
          signals.value = signals.value.slice(0, 100)
        }
      }
    })
    
    channelManager.subscribeToTradingSignals(botIds)
  })
  
  onUnmounted(() => {
    channelManager.unsubscribeAll()
  })
  
  return { signals }
}
```

### Angular Service

```typescript
import { Injectable } from '@angular/core'
import { BehaviorSubject } from 'rxjs'
import { RealtimeChannelManager } from './src/supabase/real-time/channels'

@Injectable({ providedIn: 'root' })
export class RealTimeService {
  private channelManager = new RealtimeChannelManager()
  private botStatusSubject = new BehaviorSubject<BotStatus[]>([])
  
  public botStatus$ = this.botStatusSubject.asObservable()
  
  subscribeToBotStatus(botIds: string[]) {
    this.channelManager.setHandlers({
      onBotStatusChange: (bot) => {
        const current = this.botStatusSubject.value
        const updated = current.filter(b => b.id !== bot.id)
        this.botStatusSubject.next([...updated, bot])
      }
    })
    
    this.channelManager.subscribeToBotStatus(botIds)
  }
  
  unsubscribe() {
    this.channelManager.unsubscribeAll()
  }
}
```

## Performance Optimization

### 1. Connection Management

```typescript
// Efficient connection pooling
class OptimizedRealtimeManager {
  private static instance: RealtimeChannelManager
  private connectionPool: Map<string, RealtimeChannel> = new Map()
  
  static getInstance(userId?: string): RealtimeChannelManager {
    if (!this.instance) {
      this.instance = new RealtimeChannelManager(userId)
    }
    return this.instance
  }
  
  // Reuse connections for similar subscriptions
  getOrCreateChannel(channelKey: string, factory: () => RealtimeChannel): RealtimeChannel {
    if (!this.connectionPool.has(channelKey)) {
      this.connectionPool.set(channelKey, factory())
    }
    return this.connectionPool.get(channelKey)!
  }
}
```

### 2. Data Filtering

```typescript
// Filter data at the database level
channelManager.subscribeToMarketData(['AAPL'], {
  filter: 'timeframe=eq.1m' // Only 1-minute data
})

// Client-side filtering for complex conditions
channelManager.setHandlers({
  onMarketUpdate: (data) => {
    // Only process significant price changes
    if (Math.abs(data.close - data.open) / data.open > 0.01) {
      processSignificantPriceChange(data)
    }
  }
})
```

### 3. Batching Updates

```typescript
// Batch updates to reduce UI re-renders
class BatchedUpdateManager {
  private updateQueue: MarketDataUpdate[] = []
  private batchTimer?: NodeJS.Timeout
  
  addUpdate(update: MarketDataUpdate) {
    this.updateQueue.push(update)
    
    if (this.batchTimer) {
      clearTimeout(this.batchTimer)
    }
    
    this.batchTimer = setTimeout(() => {
      this.processBatch()
    }, 100) // Batch updates every 100ms
  }
  
  private processBatch() {
    if (this.updateQueue.length > 0) {
      // Group by symbol and keep latest
      const latestBySymbol = new Map<string, MarketDataUpdate>()
      
      this.updateQueue.forEach(update => {
        latestBySymbol.set(update.symbol_id, update)
      })
      
      // Process batch
      Array.from(latestBySymbol.values()).forEach(update => {
        updateUI(update)
      })
      
      this.updateQueue = []
    }
  }
}
```

## Error Handling

### 1. Connection Recovery

```typescript
channelManager.setHandlers({
  onError: (error) => {
    console.error('Real-time error:', error)
    
    // Attempt reconnection with exponential backoff
    let retryDelay = 1000
    const maxRetries = 5
    let retryCount = 0
    
    const retry = () => {
      if (retryCount < maxRetries) {
        setTimeout(() => {
          retryCount++
          retryDelay *= 2
          
          // Attempt to reconnect
          channelManager.subscribeToAll()
        }, retryDelay)
      } else {
        // Show persistent error message
        showPersistentError('Lost connection to real-time services')
      }
    }
    
    retry()
  }
})
```

### 2. Data Validation

```typescript
function validateMarketData(data: any): data is MarketDataUpdate {
  return (
    typeof data.symbol_id === 'string' &&
    typeof data.close === 'number' &&
    typeof data.volume === 'number' &&
    !isNaN(data.close) &&
    !isNaN(data.volume) &&
    data.close > 0 &&
    data.volume >= 0
  )
}

channelManager.setHandlers({
  onMarketUpdate: (data) => {
    if (!validateMarketData(data)) {
      console.warn('Invalid market data received:', data)
      return
    }
    
    processMarketData(data)
  }
})
```

## Security Considerations

### 1. RLS Policy Integration

Real-time subscriptions automatically respect RLS policies:

```sql
-- Only receive alerts for your own account
CREATE POLICY "Users can view own alerts" ON alerts
  FOR SELECT USING (auth.user_id() = user_id);
  
-- Only receive trading signals from your own bots
CREATE POLICY "Users can view own bot executions" ON bot_executions
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM trading_bots 
      WHERE id = bot_executions.bot_id 
      AND user_id = auth.user_id()
    )
  );
```

### 2. Rate Limiting

```typescript
class RateLimitedChannelManager extends RealtimeChannelManager {
  private updateCounts = new Map<string, number>()
  private resetInterval = 60000 // 1 minute
  
  private checkRateLimit(channelType: string): boolean {
    const count = this.updateCounts.get(channelType) || 0
    const limit = this.getRateLimit(channelType)
    
    if (count >= limit) {
      console.warn(`Rate limit exceeded for ${channelType}`)
      return false
    }
    
    this.updateCounts.set(channelType, count + 1)
    return true
  }
  
  private getRateLimit(channelType: string): number {
    const limits = {
      'market_data': 1000,
      'trading_signals': 100,
      'bot_status': 50,
      'alerts': 20
    }
    return limits[channelType] || 10
  }
}
```

## Monitoring and Debugging

### 1. Real-time Metrics

```typescript
import { performanceMonitor } from './src/supabase/monitoring/performance-monitor'

// Monitor real-time performance
performanceMonitor.setRealtimeMetrics({
  onMessageReceived: (channel, latency) => {
    console.log(`Message received on ${channel} with ${latency}ms latency`)
  },
  onConnectionStatus: (status) => {
    console.log(`Real-time connection status: ${status}`)
  }
})
```

### 2. Debug Logging

```typescript
// Enable debug mode for troubleshooting
const channelManager = new RealtimeChannelManager(userId, {
  debug: true,
  logLevel: 'info'
})

channelManager.setHandlers({
  onSubscribed: (status) => {
    console.log('Subscription status:', status)
  },
  onError: (error) => {
    console.error('Channel error:', error)
    
    // Send error to monitoring service
    sendErrorToMonitoring('realtime_error', error)
  }
})
```

## Best Practices

1. **Connection Management**: Use singleton pattern for channel managers
2. **Data Filtering**: Filter at database level when possible
3. **Batch Updates**: Group frequent updates to reduce UI thrashing
4. **Error Recovery**: Implement exponential backoff for reconnections
5. **Performance Monitoring**: Track latency and connection health
6. **Security**: Always validate received data
7. **Resource Cleanup**: Unsubscribe from channels when components unmount
8. **Rate Limiting**: Implement client-side rate limiting for high-frequency data

## Troubleshooting

### Common Issues

1. **Connection Drops**: Check network connectivity and Supabase status
2. **Missing Data**: Verify RLS policies and user authentication
3. **High Latency**: Check database performance and edge function response times
4. **Memory Leaks**: Ensure proper cleanup of subscriptions

### Debug Commands

```bash
# Check Supabase real-time logs
supabase logs realtime

# Monitor database connections
SELECT * FROM pg_stat_activity WHERE application_name LIKE '%realtime%';

# Test real-time connectivity
curl -X GET 'https://your-project.supabase.co/realtime/v1/health'
```

---

This completes the real-time features documentation. The system provides comprehensive real-time capabilities for all aspects of the neural trading platform.