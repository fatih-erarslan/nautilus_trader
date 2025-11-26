/**
 * E2B Sandbox Integration Client
 * Manages E2B sandbox deployments for isolated bot execution
 */

import { supabase } from '../supabase.config'
import { tradingBotsClient } from './trading-bots'

export interface E2BSandboxConfig {
  name: string
  template?: string
  cpu_count?: number
  memory_mb?: number
  timeout_seconds?: number
  environment_variables?: Record<string, string>
}

export interface BotDeploymentConfig {
  bot_id: string
  sandbox_config: E2BSandboxConfig
  trading_config?: {
    symbols?: string[]
    risk_limits?: Record<string, number>
    execution_frequency?: number
  }
}

export class SandboxIntegrationClient {

  // Deploy bot to E2B sandbox with full configuration
  async deployBotToSandbox(userId: string, config: BotDeploymentConfig): Promise<{
    deployment: any
    sandbox_id: string
    error?: string
  }> {
    try {
      // Get bot information
      const { data: bot, error: botError } = await supabase
        .from('trading_bots')
        .select(`
          *,
          trading_accounts (*)
        `)
        .eq('id', config.bot_id)
        .eq('user_id', userId)
        .single()

      if (botError || !bot) {
        return { deployment: null, sandbox_id: '', error: 'Bot not found or access denied' }
      }

      // Generate unique sandbox ID
      const sandbox_id = `neural-trader-${config.bot_id}-${Date.now()}`

      // Prepare sandbox configuration
      const sandboxConfig = {
        ...config.sandbox_config,
        name: config.sandbox_config.name || `Bot-${bot.name}`,
        template: config.sandbox_config.template || 'neural-trader-base',
        environment_variables: {
          ...config.sandbox_config.environment_variables,
          BOT_ID: config.bot_id,
          SUPABASE_URL: process.env.SUPABASE_URL || '',
          SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY || '',
          USER_ID: userId,
          ACCOUNT_ID: bot.account_id
        }
      }

      // Create E2B sandbox using MCP
      const e2bResponse = await this.createE2BSandbox(sandbox_id, sandboxConfig)
      if (e2bResponse.error) {
        return { deployment: null, sandbox_id: '', error: e2bResponse.error }
      }

      // Record deployment in database
      const { deployment, error } = await tradingBotsClient.deployToSandbox({
        bot_id: config.bot_id,
        sandbox_name: sandboxConfig.name,
        template: sandboxConfig.template,
        cpu_count: sandboxConfig.cpu_count,
        memory_mb: sandboxConfig.memory_mb,
        timeout_seconds: sandboxConfig.timeout_seconds,
        configuration: {
          sandbox_id,
          trading_config: config.trading_config,
          environment_variables: sandboxConfig.environment_variables
        }
      })

      if (error) {
        // Cleanup sandbox if database record failed
        await this.terminateE2BSandbox(sandbox_id)
        return { deployment: null, sandbox_id: '', error }
      }

      // Start the bot in the sandbox
      const startupResult = await this.startBotInSandbox(sandbox_id, {
        bot_id: config.bot_id,
        bot_config: bot,
        trading_config: config.trading_config
      })

      if (startupResult.error) {
        await tradingBotsClient.updateDeploymentStatus(deployment.id, 'failed', null, startupResult.error)
        return { deployment, sandbox_id, error: startupResult.error }
      }

      // Update deployment status to running
      await tradingBotsClient.updateDeploymentStatus(deployment.id, 'running')

      return { deployment, sandbox_id }
    } catch (error) {
      console.error('Error deploying bot to sandbox:', error)
      return { deployment: null, sandbox_id: '', error: 'Failed to deploy bot to sandbox' }
    }
  }

  // Create E2B sandbox
  private async createE2BSandbox(sandbox_id: string, config: E2BSandboxConfig): Promise<{
    success: boolean
    error?: string
  }> {
    try {
      // This would integrate with E2B MCP server
      // For now, we'll simulate the creation process
      
      const response = await fetch('/api/e2b/create-sandbox', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sandbox_id,
          name: config.name,
          template: config.template,
          cpu_count: config.cpu_count,
          memory_mb: config.memory_mb,
          timeout: config.timeout_seconds,
          env: config.environment_variables
        })
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `E2B creation failed: ${error}` }
      }

      return { success: true }
    } catch (error) {
      console.error('E2B sandbox creation error:', error)
      return { success: false, error: 'Failed to create E2B sandbox' }
    }
  }

  // Start bot execution in sandbox
  private async startBotInSandbox(sandbox_id: string, config: {
    bot_id: string
    bot_config: any
    trading_config?: any
  }): Promise<{ success: boolean; error?: string }> {
    try {
      // Prepare bot startup script
      const startupScript = this.generateBotStartupScript(config)

      // Execute startup script in sandbox
      const response = await fetch('/api/e2b/execute-command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sandbox_id,
          command: 'python',
          args: ['-c', startupScript],
          capture_output: true
        })
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `Bot startup failed: ${error}` }
      }

      const result = await response.json()
      if (result.exit_code !== 0) {
        return { success: false, error: `Bot startup error: ${result.stderr}` }
      }

      return { success: true }
    } catch (error) {
      console.error('Bot startup error:', error)
      return { success: false, error: 'Failed to start bot in sandbox' }
    }
  }

  // Generate Python script to run bot in sandbox
  private generateBotStartupScript(config: {
    bot_id: string
    bot_config: any
    trading_config?: any
  }): string {
    return `
import os
import json
import asyncio
from neural_trader_bot import NeuralTradingBot
from supabase_client import create_supabase_client

async def main():
    # Initialize Supabase client
    supabase = create_supabase_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_ANON_KEY')
    )
    
    # Bot configuration
    bot_config = ${JSON.stringify(config.bot_config)}
    trading_config = ${JSON.stringify(config.trading_config || {})}
    
    # Initialize trading bot
    bot = NeuralTradingBot(
        bot_id='${config.bot_id}',
        config=bot_config,
        trading_config=trading_config,
        supabase_client=supabase
    )
    
    # Start bot execution loop
    print(f"Starting bot {bot_config['name']} in sandbox...")
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
`
  }

  // Monitor sandbox health and performance
  async checkSandboxHealth(sandbox_id: string): Promise<{
    status: string
    metrics?: any
    error?: string
  }> {
    try {
      const response = await fetch(`/api/e2b/sandbox-status/${sandbox_id}`)
      
      if (!response.ok) {
        return { status: 'error', error: 'Failed to get sandbox status' }
      }

      const data = await response.json()
      return {
        status: data.status,
        metrics: data.metrics
      }
    } catch (error) {
      console.error('Error checking sandbox health:', error)
      return { status: 'error', error: 'Failed to check sandbox health' }
    }
  }

  // Get sandbox logs
  async getSandboxLogs(sandbox_id: string, lines?: number): Promise<{
    logs: string
    error?: string
  }> {
    try {
      const response = await fetch(`/api/e2b/sandbox-logs/${sandbox_id}?lines=${lines || 100}`)
      
      if (!response.ok) {
        return { logs: '', error: 'Failed to get sandbox logs' }
      }

      const data = await response.json()
      return { logs: data.logs }
    } catch (error) {
      console.error('Error getting sandbox logs:', error)
      return { logs: '', error: 'Failed to get sandbox logs' }
    }
  }

  // Terminate E2B sandbox
  async terminateE2BSandbox(sandbox_id: string): Promise<{
    success: boolean
    error?: string
  }> {
    try {
      const response = await fetch(`/api/e2b/terminate-sandbox/${sandbox_id}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `Failed to terminate sandbox: ${error}` }
      }

      return { success: true }
    } catch (error) {
      console.error('Error terminating sandbox:', error)
      return { success: false, error: 'Failed to terminate sandbox' }
    }
  }

  // Scale sandbox resources
  async scaleSandbox(sandbox_id: string, scaling: {
    cpu_count?: number
    memory_mb?: number
  }): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`/api/e2b/scale-sandbox/${sandbox_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scaling)
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `Failed to scale sandbox: ${error}` }
      }

      return { success: true }
    } catch (error) {
      console.error('Error scaling sandbox:', error)
      return { success: false, error: 'Failed to scale sandbox' }
    }
  }

  // Get all user sandboxes
  async getUserSandboxes(userId: string): Promise<{
    deployments: any[]
    error?: string
  }> {
    try {
      const { data, error } = await supabase
        .from('sandbox_deployments')
        .select(`
          *,
          trading_bots (name, strategy_type, status)
        `)
        .eq('user_id', userId)
        .order('created_at', { ascending: false })

      if (error) {
        console.error('Failed to fetch user sandboxes:', error)
        return { deployments: [], error: error.message }
      }

      // Enrich with real-time status from E2B
      const enrichedDeployments = await Promise.all(
        (data || []).map(async (deployment) => {
          const health = await this.checkSandboxHealth(deployment.sandbox_id)
          return {
            ...deployment,
            live_status: health.status,
            live_metrics: health.metrics
          }
        })
      )

      return { deployments: enrichedDeployments }
    } catch (error) {
      console.error('Error fetching user sandboxes:', error)
      return { deployments: [], error: 'Failed to fetch sandboxes' }
    }
  }

  // Stop sandbox (but keep it for later restart)
  async stopSandbox(deployment_id: string): Promise<{
    success: boolean
    error?: string
  }> {
    try {
      // Get deployment info
      const { data: deployment, error: fetchError } = await supabase
        .from('sandbox_deployments')
        .select('sandbox_id')
        .eq('id', deployment_id)
        .single()

      if (fetchError || !deployment) {
        return { success: false, error: 'Deployment not found' }
      }

      // Stop the sandbox
      const response = await fetch(`/api/e2b/stop-sandbox/${deployment.sandbox_id}`, {
        method: 'POST'
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `Failed to stop sandbox: ${error}` }
      }

      // Update deployment status
      await tradingBotsClient.updateDeploymentStatus(deployment_id, 'stopped')

      return { success: true }
    } catch (error) {
      console.error('Error stopping sandbox:', error)
      return { success: false, error: 'Failed to stop sandbox' }
    }
  }

  // Restart stopped sandbox
  async restartSandbox(deployment_id: string): Promise<{
    success: boolean
    error?: string
  }> {
    try {
      // Get deployment info
      const { data: deployment, error: fetchError } = await supabase
        .from('sandbox_deployments')
        .select('*')
        .eq('id', deployment_id)
        .single()

      if (fetchError || !deployment) {
        return { success: false, error: 'Deployment not found' }
      }

      // Restart the sandbox
      const response = await fetch(`/api/e2b/restart-sandbox/${deployment.sandbox_id}`, {
        method: 'POST'
      })

      if (!response.ok) {
        const error = await response.text()
        return { success: false, error: `Failed to restart sandbox: ${error}` }
      }

      // Update deployment status
      await tradingBotsClient.updateDeploymentStatus(deployment_id, 'running')

      return { success: true }
    } catch (error) {
      console.error('Error restarting sandbox:', error)
      return { success: false, error: 'Failed to restart sandbox' }
    }
  }

  // Execute command in sandbox
  async executeSandboxCommand(sandbox_id: string, command: string, args?: string[]): Promise<{
    output: string
    error_output: string
    exit_code: number
    error?: string
  }> {
    try {
      const response = await fetch('/api/e2b/execute-command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sandbox_id,
          command,
          args: args || [],
          capture_output: true
        })
      })

      if (!response.ok) {
        const error = await response.text()
        return {
          output: '',
          error_output: error,
          exit_code: -1,
          error: 'Failed to execute command'
        }
      }

      const result = await response.json()
      return {
        output: result.stdout || '',
        error_output: result.stderr || '',
        exit_code: result.exit_code || 0
      }
    } catch (error) {
      console.error('Error executing sandbox command:', error)
      return {
        output: '',
        error_output: '',
        exit_code: -1,
        error: 'Failed to execute command'
      }
    }
  }

  // Cleanup inactive sandboxes
  async cleanupInactiveSandboxes(userId: string, maxAge: number = 24 * 60 * 60 * 1000): Promise<{
    cleaned: number
    error?: string
  }> {
    try {
      const cutoffDate = new Date(Date.now() - maxAge).toISOString()

      // Get inactive deployments
      const { data: inactiveDeployments } = await supabase
        .from('sandbox_deployments')
        .select('id, sandbox_id')
        .eq('user_id', userId)
        .in('status', ['stopped', 'failed'])
        .lt('updated_at', cutoffDate)

      let cleaned = 0
      for (const deployment of inactiveDeployments || []) {
        try {
          await this.terminateE2BSandbox(deployment.sandbox_id)
          await tradingBotsClient.updateDeploymentStatus(deployment.id, 'terminated')
          cleaned++
        } catch (error) {
          console.error(`Failed to cleanup sandbox ${deployment.sandbox_id}:`, error)
        }
      }

      return { cleaned }
    } catch (error) {
      console.error('Error cleaning up sandboxes:', error)
      return { cleaned: 0, error: 'Failed to cleanup sandboxes' }
    }
  }
}

// Export singleton instance
export const sandboxIntegrationClient = new SandboxIntegrationClient()
export default sandboxIntegrationClient