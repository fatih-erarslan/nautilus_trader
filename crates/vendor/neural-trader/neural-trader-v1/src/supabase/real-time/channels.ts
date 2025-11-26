/**
 * Real-time Channels for Neural Trading Platform
 * Comprehensive real-time data streaming setup
 */

import { RealtimeChannel, RealtimePostgresChangesPayload } from '@supabase/supabase-js'
import { supabase, realtimeChannels, realtimeEvents, realtimeFilters } from '../supabase.config'
import { Database } from '../types/database.types'

type Tables = Database['public']['Tables']
type MarketDataRow = Tables['market_data']['Row']
type TradingSignal = Tables['bot_executions']['Row']
type BotStatus = Tables['trading_bots']['Row']
type AlertRow = Tables['alerts']['Row']
type PerformanceMetric = Tables['performance_metrics']['Row']
type TrainingRun = Tables['training_runs']['Row']

// Real-time event handlers
export interface RealtimeHandlers {
  onMarketUpdate?: (payload: MarketDataRow) => void
  onSignalGenerated?: (payload: TradingSignal) => void
  onBotStatusChange?: (payload: BotStatus) => void
  onAlertTriggered?: (payload: AlertRow) => void
  onPerformanceUpdate?: (payload: PerformanceMetric) => void
  onTrainingProgress?: (payload: TrainingRun) => void
  onOrderExecuted?: (payload: any) => void
  onPositionUpdate?: (payload: any) => void
  onError?: (error: Error) => void
  onSubscribed?: (status: string) => void
}

// Channel manager class
export class RealtimeChannelManager {
  private channels: Map<string, RealtimeChannel> = new Map()
  private handlers: RealtimeHandlers = {}
  private userId?: string

  constructor(userId?: string) {
    this.userId = userId
  }

  // Set event handlers
  setHandlers(handlers: RealtimeHandlers): void {
    this.handlers = { ...this.handlers, ...handlers }
  }

  // Subscribe to market data updates
  subscribeToMarketData(symbols?: string[]): RealtimeChannel {
    const channelName = `${realtimeChannels.marketData}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'market_data',
          filter: symbols ? `symbol_id=in.(${symbols.join(',')})` : undefined,
        },
        (payload: RealtimePostgresChangesPayload<MarketDataRow>) => {
          if (this.handlers.onMarketUpdate && payload.new) {
            this.handlers.onMarketUpdate(payload.new)
          }
        }
      )
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'market_data',
          filter: symbols ? `symbol_id=in.(${symbols.join(',')})` : undefined,
        },
        (payload: RealtimePostgresChangesPayload<MarketDataRow>) => {
          if (this.handlers.onMarketUpdate && payload.new) {
            this.handlers.onMarketUpdate(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Market data: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to trading signals
  subscribeToTradingSignals(botIds?: string[]): RealtimeChannel {
    const channelName = `${realtimeChannels.tradingSignals}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'bot_executions',
          filter: botIds ? `bot_id=in.(${botIds.join(',')})` : (this.userId ? realtimeFilters.userSpecific(this.userId) : undefined),
        },
        (payload: RealtimePostgresChangesPayload<TradingSignal>) => {
          if (this.handlers.onSignalGenerated && payload.new) {
            this.handlers.onSignalGenerated(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Trading signals: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to bot status changes
  subscribeToBotStatus(botIds?: string[]): RealtimeChannel {
    const channelName = `${realtimeChannels.botStatus}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'trading_bots',
          filter: botIds ? `id=in.(${botIds.join(',')})` : (this.userId ? realtimeFilters.userSpecific(this.userId) : undefined),
        },
        (payload: RealtimePostgresChangesPayload<BotStatus>) => {
          if (this.handlers.onBotStatusChange && payload.new) {
            this.handlers.onBotStatusChange(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Bot status: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to alerts
  subscribeToAlerts(): RealtimeChannel {
    if (!this.userId) {
      throw new Error('User ID required for alert subscription')
    }

    const channelName = `${realtimeChannels.alerts}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'alerts',
          filter: realtimeFilters.userSpecific(this.userId),
        },
        (payload: RealtimePostgresChangesPayload<AlertRow>) => {
          if (this.handlers.onAlertTriggered && payload.new) {
            this.handlers.onAlertTriggered(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Alerts: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to performance metrics
  subscribeToPerformanceMetrics(entityIds?: string[]): RealtimeChannel {
    const channelName = `${realtimeChannels.performance}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'performance_metrics',
          filter: entityIds ? `entity_id=in.(${entityIds.join(',')})` : undefined,
        },
        (payload: RealtimePostgresChangesPayload<PerformanceMetric>) => {
          if (this.handlers.onPerformanceUpdate && payload.new) {
            this.handlers.onPerformanceUpdate(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Performance: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to neural network training progress
  subscribeToNeuralTraining(modelIds?: string[]): RealtimeChannel {
    const channelName = `${realtimeChannels.neuralTraining}_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'training_runs',
          filter: modelIds ? `model_id=in.(${modelIds.join(',')})` : (this.userId ? realtimeFilters.userSpecific(this.userId) : undefined),
        },
        (payload: RealtimePostgresChangesPayload<TrainingRun>) => {
          if (this.handlers.onTrainingProgress && payload.new) {
            this.handlers.onTrainingProgress(payload.new)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Neural training: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to order executions
  subscribeToOrderExecutions(accountIds?: string[]): RealtimeChannel {
    const channelName = `orders_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'orders',
          filter: accountIds ? `account_id=in.(${accountIds.join(',')})` : undefined,
        },
        (payload) => {
          if (this.handlers.onOrderExecuted) {
            this.handlers.onOrderExecuted(payload)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Orders: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to position updates
  subscribeToPositionUpdates(accountIds?: string[]): RealtimeChannel {
    const channelName = `positions_${Date.now()}`
    
    let channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'positions',
          filter: accountIds ? `account_id=in.(${accountIds.join(',')})` : undefined,
        },
        (payload) => {
          if (this.handlers.onPositionUpdate) {
            this.handlers.onPositionUpdate(payload)
          }
        }
      )

    channel.subscribe((status) => {
      if (this.handlers.onSubscribed) {
        this.handlers.onSubscribed(`Positions: ${status}`)
      }
    })

    this.channels.set(channelName, channel)
    return channel
  }

  // Subscribe to all user-relevant channels
  subscribeToAll(): void {
    if (!this.userId) {
      throw new Error('User ID required for full subscription')
    }

    this.subscribeToAlerts()
    this.subscribeToBotStatus()
    this.subscribeToNeuralTraining()
    this.subscribeToPerformanceMetrics()
  }

  // Unsubscribe from a specific channel
  unsubscribe(channelName: string): void {
    const channel = this.channels.get(channelName)
    if (channel) {
      channel.unsubscribe()
      this.channels.delete(channelName)
    }
  }

  // Unsubscribe from all channels
  unsubscribeAll(): void {
    this.channels.forEach((channel, name) => {
      channel.unsubscribe()
    })
    this.channels.clear()
  }

  // Get active channels
  getActiveChannels(): string[] {
    return Array.from(this.channels.keys())
  }

  // Check if channel is subscribed
  isSubscribed(channelName: string): boolean {
    return this.channels.has(channelName)
  }
}

// Custom hooks for React integration (if needed)
export const createRealtimeHooks = () => {
  let channelManager: RealtimeChannelManager | null = null

  const useRealtimeSubscription = (userId: string, handlers: RealtimeHandlers) => {
    if (!channelManager) {
      channelManager = new RealtimeChannelManager(userId)
    }
    
    channelManager.setHandlers(handlers)
    return channelManager
  }

  const cleanupRealtimeSubscriptions = () => {
    if (channelManager) {
      channelManager.unsubscribeAll()
      channelManager = null
    }
  }

  return {
    useRealtimeSubscription,
    cleanupRealtimeSubscriptions,
  }
}

// Export default instance
export const realtimeManager = new RealtimeChannelManager()
export default RealtimeChannelManager