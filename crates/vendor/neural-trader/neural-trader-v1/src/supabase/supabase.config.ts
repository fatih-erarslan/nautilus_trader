/**
 * Supabase Configuration for Neural Trading Platform
 * Comprehensive setup for database, real-time, and authentication
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js'
import { Database } from './types/database.types'

// Environment configuration
const supabaseUrl = process.env.SUPABASE_URL || 'https://your-project.supabase.co'
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY || 'your-anon-key'
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || 'your-service-key'

// Client configurations
export const supabaseConfig = {
  url: supabaseUrl,
  anonKey: supabaseAnonKey,
  serviceKey: supabaseServiceKey,
  options: {
    auth: {
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: true,
    },
    realtime: {
      params: {
        eventsPerSecond: 100,
      },
    },
    global: {
      headers: {
        'x-application-name': 'neural-trader',
      },
    },
  },
}

// Main client for authenticated operations
export const supabase: SupabaseClient<Database> = createClient(
  supabaseUrl,
  supabaseAnonKey,
  supabaseConfig.options
)

// Service role client for admin operations
export const supabaseAdmin: SupabaseClient<Database> = createClient(
  supabaseUrl,
  supabaseServiceKey,
  {
    ...supabaseConfig.options,
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
)

// Real-time channel configurations
export const realtimeChannels = {
  marketData: 'market_data_channel',
  tradingSignals: 'trading_signals_channel',
  botStatus: 'bot_status_channel',
  alerts: 'alerts_channel',
  performance: 'performance_channel',
  neuralTraining: 'neural_training_channel',
} as const

// Real-time event types
export const realtimeEvents = {
  marketUpdate: 'market_update',
  signalGenerated: 'signal_generated',
  botStatusChange: 'bot_status_change',
  alertTriggered: 'alert_triggered',
  performanceUpdate: 'performance_update',
  trainingProgress: 'training_progress',
  orderExecuted: 'order_executed',
  positionUpdate: 'position_update',
} as const

// Database table names
export const tables = {
  profiles: 'profiles',
  symbols: 'symbols',
  marketData: 'market_data',
  newsData: 'news_data',
  tradingAccounts: 'trading_accounts',
  positions: 'positions',
  orders: 'orders',
  neuralModels: 'neural_models',
  trainingRuns: 'training_runs',
  modelPredictions: 'model_predictions',
  tradingBots: 'trading_bots',
  botExecutions: 'bot_executions',
  sandboxDeployments: 'sandbox_deployments',
  performanceMetrics: 'performance_metrics',
  alerts: 'alerts',
  auditLogs: 'audit_logs',
} as const

// Real-time subscription filters
export const realtimeFilters = {
  userSpecific: (userId: string) => `user_id=eq.${userId}`,
  symbolSpecific: (symbol: string) => `symbol=eq.${symbol}`,
  botSpecific: (botId: string) => `bot_id=eq.${botId}`,
  accountSpecific: (accountId: string) => `account_id=eq.${accountId}`,
  modelSpecific: (modelId: string) => `model_id=eq.${modelId}`,
}

// Connection health check
export const checkSupabaseConnection = async (): Promise<boolean> => {
  try {
    const { data, error } = await supabase.from('profiles').select('id').limit(1)
    return !error
  } catch (error) {
    console.error('Supabase connection check failed:', error)
    return false
  }
}

// Initialize database schema (for development)
export const initializeDatabase = async (): Promise<void> => {
  try {
    // This would typically be run as migrations in production
    console.log('Database initialization should be handled via Supabase migrations')
  } catch (error) {
    console.error('Database initialization failed:', error)
    throw error
  }
}

export default supabase