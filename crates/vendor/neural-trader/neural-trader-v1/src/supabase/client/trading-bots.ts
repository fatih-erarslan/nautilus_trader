/**
 * Trading Bots Client
 * Manages trading bot persistence, configuration, and execution tracking
 */

import { supabase, supabaseAdmin } from '../supabase.config'
import { Database } from '../types/database.types'

type Tables = Database['public']['Tables']
type TradingBot = Tables['trading_bots']['Row']
type BotExecution = Tables['bot_executions']['Row']
type SandboxDeployment = Tables['sandbox_deployments']['Row']

export interface CreateBotRequest {
  name: string
  account_id: string
  strategy_type: string
  configuration: any
  model_ids?: string[]
  symbols: string[]
  max_position_size?: number
  risk_limit?: number
}

export interface UpdateBotRequest {
  name?: string
  configuration?: any
  model_ids?: string[]
  symbols?: string[]
  max_position_size?: number
  risk_limit?: number
  status?: 'active' | 'paused' | 'stopped' | 'error' | 'training'
}

export interface BotExecutionRequest {
  bot_id: string
  symbol_id: string
  action: 'buy' | 'sell' | 'hold'
  signal_strength?: number
  reasoning?: string
  order_id?: string
  metadata?: any
}

export interface DeployBotToSandboxRequest {
  bot_id: string
  sandbox_name: string
  template?: string
  cpu_count?: number
  memory_mb?: number
  timeout_seconds?: number
  configuration?: any
}

export class TradingBotsClient {

  // Create a new trading bot
  async createBot(userId: string, botData: CreateBotRequest): Promise<{ bot: TradingBot; error?: string }> {
    try {
      // Verify account exists and belongs to user
      const { data: account, error: accountError } = await supabase
        .from('trading_accounts')
        .select('id')
        .eq('id', botData.account_id)
        .eq('user_id', userId)
        .single()

      if (accountError || !account) {
        return { bot: null as any, error: 'Trading account not found or access denied' }
      }

      const { data, error } = await supabase
        .from('trading_bots')
        .insert({
          user_id: userId,
          account_id: botData.account_id,
          name: botData.name,
          strategy_type: botData.strategy_type,
          configuration: botData.configuration,
          model_ids: botData.model_ids || [],
          symbols: botData.symbols,
          max_position_size: botData.max_position_size || 1000,
          risk_limit: botData.risk_limit || 0.05,
          status: 'paused',
          performance_metrics: {}
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to create trading bot:', error)
        return { bot: null as any, error: error.message }
      }

      return { bot: data }
    } catch (error) {
      console.error('Error creating trading bot:', error)
      return { bot: null as any, error: 'Failed to create bot' }
    }
  }

  // Get user's trading bots
  async getUserBots(userId: string, filters?: {
    status?: string
    account_id?: string
    strategy_type?: string
    limit?: number
  }): Promise<{ bots: TradingBot[]; error?: string }> {
    try {
      let query = supabase
        .from('trading_bots')
        .select(`
          *,
          trading_accounts (name, broker)
        `)
        .eq('user_id', userId)
        .order('created_at', { ascending: false })

      if (filters?.status) {
        query = query.eq('status', filters.status)
      }

      if (filters?.account_id) {
        query = query.eq('account_id', filters.account_id)
      }

      if (filters?.strategy_type) {
        query = query.eq('strategy_type', filters.strategy_type)
      }

      if (filters?.limit) {
        query = query.limit(filters.limit)
      }

      const { data, error } = await query

      if (error) {
        console.error('Failed to fetch trading bots:', error)
        return { bots: [], error: error.message }
      }

      return { bots: data || [] }
    } catch (error) {
      console.error('Error fetching trading bots:', error)
      return { bots: [], error: 'Failed to fetch bots' }
    }
  }

  // Update trading bot
  async updateBot(botId: string, updates: UpdateBotRequest): Promise<{ bot: TradingBot; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('trading_bots')
        .update({
          ...updates,
          updated_at: new Date().toISOString()
        })
        .eq('id', botId)
        .select()
        .single()

      if (error) {
        console.error('Failed to update trading bot:', error)
        return { bot: null as any, error: error.message }
      }

      return { bot: data }
    } catch (error) {
      console.error('Error updating trading bot:', error)
      return { bot: null as any, error: 'Failed to update bot' }
    }
  }

  // Start/stop trading bot
  async setBotStatus(botId: string, status: 'active' | 'paused' | 'stopped'): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('trading_bots')
        .update({ 
          status,
          updated_at: new Date().toISOString() 
        })
        .eq('id', botId)

      if (error) {
        console.error('Failed to update bot status:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error updating bot status:', error)
      return { success: false, error: 'Failed to update bot status' }
    }
  }

  // Record bot execution
  async recordExecution(executionData: BotExecutionRequest): Promise<{ execution: BotExecution; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('bot_executions')
        .insert({
          bot_id: executionData.bot_id,
          symbol_id: executionData.symbol_id,
          action: executionData.action,
          signal_strength: executionData.signal_strength,
          reasoning: executionData.reasoning,
          order_id: executionData.order_id,
          metadata: executionData.metadata || {}
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to record bot execution:', error)
        return { execution: null as any, error: error.message }
      }

      return { execution: data }
    } catch (error) {
      console.error('Error recording bot execution:', error)
      return { execution: null as any, error: 'Failed to record execution' }
    }
  }

  // Get bot execution history
  async getBotExecutions(
    botId: string,
    filters?: {
      symbol_id?: string
      action?: string
      start_date?: string
      end_date?: string
      limit?: number
    }
  ): Promise<{ executions: BotExecution[]; error?: string }> {
    try {
      let query = supabase
        .from('bot_executions')
        .select(`
          *,
          symbols (symbol, name),
          orders (status, filled_quantity, average_fill_price)
        `)
        .eq('bot_id', botId)
        .order('executed_at', { ascending: false })

      if (filters?.symbol_id) {
        query = query.eq('symbol_id', filters.symbol_id)
      }

      if (filters?.action) {
        query = query.eq('action', filters.action)
      }

      if (filters?.start_date) {
        query = query.gte('executed_at', filters.start_date)
      }

      if (filters?.end_date) {
        query = query.lte('executed_at', filters.end_date)
      }

      if (filters?.limit) {
        query = query.limit(filters.limit)
      }

      const { data, error } = await query

      if (error) {
        console.error('Failed to fetch bot executions:', error)
        return { executions: [], error: error.message }
      }

      return { executions: data || [] }
    } catch (error) {
      console.error('Error fetching bot executions:', error)
      return { executions: [], error: 'Failed to fetch executions' }
    }
  }

  // Deploy bot to E2B sandbox
  async deployToSandbox(deploymentData: DeployBotToSandboxRequest): Promise<{ deployment: SandboxDeployment; error?: string }> {
    try {
      // Get bot information
      const { data: bot, error: botError } = await supabase
        .from('trading_bots')
        .select('user_id, name')
        .eq('id', deploymentData.bot_id)
        .single()

      if (botError || !bot) {
        return { deployment: null as any, error: 'Bot not found' }
      }

      // Generate unique sandbox ID
      const sandboxId = `sandbox_${deploymentData.bot_id}_${Date.now()}`

      const { data, error } = await supabase
        .from('sandbox_deployments')
        .insert({
          user_id: bot.user_id,
          bot_id: deploymentData.bot_id,
          sandbox_id: sandboxId,
          name: deploymentData.sandbox_name,
          template: deploymentData.template || 'base',
          configuration: deploymentData.configuration || {},
          cpu_count: deploymentData.cpu_count || 1,
          memory_mb: deploymentData.memory_mb || 512,
          timeout_seconds: deploymentData.timeout_seconds || 300,
          status: 'pending'
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to deploy bot to sandbox:', error)
        return { deployment: null as any, error: error.message }
      }

      return { deployment: data }
    } catch (error) {
      console.error('Error deploying bot to sandbox:', error)
      return { deployment: null as any, error: 'Failed to deploy to sandbox' }
    }
  }

  // Get bot sandbox deployments
  async getBotDeployments(botId: string): Promise<{ deployments: SandboxDeployment[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('sandbox_deployments')
        .select('*')
        .eq('bot_id', botId)
        .order('created_at', { ascending: false })

      if (error) {
        console.error('Failed to fetch bot deployments:', error)
        return { deployments: [], error: error.message }
      }

      return { deployments: data || [] }
    } catch (error) {
      console.error('Error fetching bot deployments:', error)
      return { deployments: [], error: 'Failed to fetch deployments' }
    }
  }

  // Update sandbox deployment status
  async updateDeploymentStatus(
    deploymentId: string,
    status: 'pending' | 'running' | 'stopped' | 'failed' | 'terminated',
    resourceUsage?: any,
    logs?: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const updateData: any = {
        status,
        updated_at: new Date().toISOString()
      }

      if (status === 'running' && !resourceUsage) {
        updateData.started_at = new Date().toISOString()
      }

      if (status === 'stopped' || status === 'terminated') {
        updateData.stopped_at = new Date().toISOString()
      }

      if (resourceUsage) {
        updateData.resource_usage = resourceUsage
      }

      if (logs) {
        updateData.logs = logs
      }

      const { error } = await supabase
        .from('sandbox_deployments')
        .update(updateData)
        .eq('id', deploymentId)

      if (error) {
        console.error('Failed to update deployment status:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error updating deployment status:', error)
      return { success: false, error: 'Failed to update deployment status' }
    }
  }

  // Calculate bot performance metrics
  async calculateBotPerformance(botId: string, timeRange?: {
    start_date?: string
    end_date?: string
  }): Promise<{ metrics: any; error?: string }> {
    try {
      // Get bot details
      const { data: bot, error: botError } = await supabase
        .from('trading_bots')
        .select('account_id, created_at')
        .eq('id', botId)
        .single()

      if (botError || !bot) {
        return { metrics: {}, error: 'Bot not found' }
      }

      const startDate = timeRange?.start_date || bot.created_at
      const endDate = timeRange?.end_date || new Date().toISOString()

      // Get executions in time range
      const { data: executions } = await supabase
        .from('bot_executions')
        .select(`
          *,
          orders (filled_quantity, average_fill_price, commission)
        `)
        .eq('bot_id', botId)
        .gte('executed_at', startDate)
        .lte('executed_at', endDate)

      // Calculate metrics
      const totalExecutions = executions?.length || 0
      const buyExecutions = executions?.filter(e => e.action === 'buy').length || 0
      const sellExecutions = executions?.filter(e => e.action === 'sell').length || 0
      const holdDecisions = executions?.filter(e => e.action === 'hold').length || 0

      // Calculate profit/loss from completed trades
      const filledOrders = executions?.filter(e => e.orders && e.orders.length > 0) || []
      const totalPnL = filledOrders.reduce((sum, execution) => {
        const order = execution.orders[0]
        if (order && order.filled_quantity && order.average_fill_price) {
          const value = order.filled_quantity * order.average_fill_price
          return sum + (execution.action === 'buy' ? -value : value) - (order.commission || 0)
        }
        return sum
      }, 0)

      // Calculate average signal strength
      const avgSignalStrength = executions && executions.length > 0 ?
        executions.reduce((sum, e) => sum + (e.signal_strength || 0), 0) / executions.length : 0

      const metrics = {
        total_executions: totalExecutions,
        buy_executions: buyExecutions,
        sell_executions: sellExecutions,
        hold_decisions: holdDecisions,
        total_pnl: totalPnL,
        avg_signal_strength: avgSignalStrength,
        execution_rate: totalExecutions > 0 ? (buyExecutions + sellExecutions) / totalExecutions : 0,
        period_start: startDate,
        period_end: endDate,
        calculated_at: new Date().toISOString()
      }

      // Update bot performance metrics
      await supabase
        .from('trading_bots')
        .update({ 
          performance_metrics: metrics,
          updated_at: new Date().toISOString()
        })
        .eq('id', botId)

      return { metrics }
    } catch (error) {
      console.error('Error calculating bot performance:', error)
      return { metrics: {}, error: 'Failed to calculate performance' }
    }
  }

  // Get bot portfolio performance
  async getBotPortfolioPerformance(botId: string): Promise<{ performance: any; error?: string }> {
    try {
      const { data: bot, error: botError } = await supabase
        .from('trading_bots')
        .select('account_id')
        .eq('id', botId)
        .single()

      if (botError || !bot) {
        return { performance: {}, error: 'Bot not found' }
      }

      // Use the database function to calculate portfolio performance
      const { data, error } = await supabase.rpc('calculate_portfolio_performance', {
        account_id_param: bot.account_id
      })

      if (error) {
        console.error('Failed to get portfolio performance:', error)
        return { performance: {}, error: error.message }
      }

      return { performance: data && data.length > 0 ? data[0] : {} }
    } catch (error) {
      console.error('Error getting bot portfolio performance:', error)
      return { performance: {}, error: 'Failed to get portfolio performance' }
    }
  }

  // Delete trading bot
  async deleteBot(botId: string): Promise<{ success: boolean; error?: string }> {
    try {
      // Delete in order due to foreign key constraints
      await supabase.from('sandbox_deployments').delete().eq('bot_id', botId)
      await supabase.from('bot_executions').delete().eq('bot_id', botId)
      const { error } = await supabase.from('trading_bots').delete().eq('id', botId)

      if (error) {
        console.error('Failed to delete trading bot:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error deleting trading bot:', error)
      return { success: false, error: 'Failed to delete bot' }
    }
  }

  // Clone trading bot
  async cloneBot(botId: string, newName: string): Promise<{ bot: TradingBot; error?: string }> {
    try {
      const { data: originalBot, error: fetchError } = await supabase
        .from('trading_bots')
        .select('*')
        .eq('id', botId)
        .single()

      if (fetchError || !originalBot) {
        return { bot: null as any, error: 'Original bot not found' }
      }

      const { data, error } = await supabase
        .from('trading_bots')
        .insert({
          user_id: originalBot.user_id,
          account_id: originalBot.account_id,
          name: newName,
          strategy_type: originalBot.strategy_type,
          configuration: originalBot.configuration,
          model_ids: originalBot.model_ids,
          symbols: originalBot.symbols,
          max_position_size: originalBot.max_position_size,
          risk_limit: originalBot.risk_limit,
          status: 'paused', // Always start cloned bots as paused
          performance_metrics: {}
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to clone trading bot:', error)
        return { bot: null as any, error: error.message }
      }

      return { bot: data }
    } catch (error) {
      console.error('Error cloning trading bot:', error)
      return { bot: null as any, error: 'Failed to clone bot' }
    }
  }
}

// Export singleton instance
export const tradingBotsClient = new TradingBotsClient()
export default tradingBotsClient