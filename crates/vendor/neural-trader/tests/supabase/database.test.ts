/**
 * Supabase Database Tests
 * Comprehensive tests for database schema, RLS, functions, and triggers
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from '@jest/testing-library'
import { createClient, SupabaseClient } from '@supabase/supabase-js'
import { Database } from '../../src/supabase/types/database.types'

describe('Supabase Database Integration', () => {
  let supabase: SupabaseClient<Database>
  let testUserId: string
  let testAccountId: string
  let testSymbolId: string
  let testModelId: string
  let testBotId: string

  beforeAll(async () => {
    // Initialize Supabase client with service role for testing
    supabase = createClient(
      process.env.SUPABASE_URL || 'http://localhost:54321',
      process.env.SUPABASE_SERVICE_ROLE_KEY || 'test-service-key'
    )

    // Create test user
    const { data: user, error } = await supabase.auth.admin.createUser({
      email: 'test@neuraltrade.ai',
      password: 'test-password-123',
      email_confirm: true
    })

    if (error) throw error
    testUserId = user.user.id

    // Create test data
    await setupTestData()
  })

  afterAll(async () => {
    // Cleanup test data
    await cleanupTestData()
  })

  beforeEach(async () => {
    // Reset any state between tests if needed
  })

  describe('Database Schema', () => {
    test('should have all required tables', async () => {
      const tables = [
        'profiles', 'symbols', 'market_data', 'news_data',
        'trading_accounts', 'positions', 'orders',
        'neural_models', 'training_runs', 'model_predictions',
        'trading_bots', 'bot_executions', 'sandbox_deployments',
        'performance_metrics', 'alerts', 'audit_logs'
      ]

      for (const table of tables) {
        const { data, error } = await supabase
          .from(table as any)
          .select('*')
          .limit(1)

        expect(error).toBeNull()
      }
    })

    test('should enforce foreign key constraints', async () => {
      // Try to create position with invalid account_id
      const { error } = await supabase
        .from('positions')
        .insert({
          account_id: '00000000-0000-0000-0000-000000000000',
          symbol_id: testSymbolId,
          side: 'long',
          quantity: 100,
          entry_price: 150.00
        })

      expect(error).not.toBeNull()
      expect(error?.message).toContain('foreign key')
    })

    test('should validate enum constraints', async () => {
      // Try to create position with invalid side
      const { error } = await supabase
        .from('positions')
        .insert({
          account_id: testAccountId,
          symbol_id: testSymbolId,
          side: 'invalid_side' as any,
          quantity: 100,
          entry_price: 150.00
        })

      expect(error).not.toBeNull()
    })
  })

  describe('Row Level Security', () => {
    test('should enforce user isolation for profiles', async () => {
      // Create another test user
      const { data: otherUser } = await supabase.auth.admin.createUser({
        email: 'other@neuraltrade.ai',
        password: 'test-password-123',
        email_confirm: true
      })

      // Switch to first user context
      await supabase.auth.signInWithPassword({
        email: 'test@neuraltrade.ai',
        password: 'test-password-123'
      })

      // Should only see own profile
      const { data: profiles } = await supabase
        .from('profiles')
        .select('*')

      expect(profiles).toHaveLength(1)
      expect(profiles![0].id).toBe(testUserId)

      // Cleanup
      await supabase.auth.admin.deleteUser(otherUser!.user.id)
    })

    test('should enforce account access for trading data', async () => {
      // Switch to user context
      await supabase.auth.signInWithPassword({
        email: 'test@neuraltrade.ai',
        password: 'test-password-123'
      })

      // Should only see own trading accounts
      const { data: accounts } = await supabase
        .from('trading_accounts')
        .select('*')

      expect(accounts?.every(account => account.user_id === testUserId)).toBe(true)

      // Should only see positions from own accounts
      const { data: positions } = await supabase
        .from('positions')
        .select('*')

      for (const position of positions || []) {
        const { data: account } = await supabase
          .from('trading_accounts')
          .select('user_id')
          .eq('id', position.account_id)
          .single()

        expect(account?.user_id).toBe(testUserId)
      }
    })

    test('should allow public read access to market data', async () => {
      // Even without authentication, should be able to read market data
      await supabase.auth.signOut()

      const { data: marketData, error } = await supabase
        .from('market_data')
        .select('*')
        .limit(1)

      expect(error).toBeNull()
      expect(Array.isArray(marketData)).toBe(true)
    })
  })

  describe('Database Functions', () => {
    test('calculate_portfolio_performance function', async () => {
      const { data, error } = await supabase.rpc('calculate_portfolio_performance', {
        account_id_param: testAccountId
      })

      expect(error).toBeNull()
      expect(Array.isArray(data)).toBe(true)
      
      if (data && data.length > 0) {
        const performance = data[0]
        expect(typeof performance.total_return).toBe('number')
        expect(typeof performance.realized_pnl).toBe('number')
        expect(typeof performance.unrealized_pnl).toBe('number')
        expect(typeof performance.win_rate).toBe('number')
      }
    })

    test('update_model_performance function', async () => {
      const { data, error } = await supabase.rpc('update_model_performance', {
        model_id_param: testModelId,
        predictions_count: 50
      })

      expect(error).toBeNull()
      expect(typeof data).toBe('object')
      expect(data).toHaveProperty('accuracy')
      expect(data).toHaveProperty('mae')
      expect(data).toHaveProperty('rmse')
    })

    test('generate_trading_signal function', async () => {
      const { data, error } = await supabase.rpc('generate_trading_signal', {
        symbol_param: 'AAPL',
        model_ids: [testModelId],
        lookback_periods: 20
      })

      expect(error).toBeNull()
      expect(typeof data).toBe('object')
      expect(data).toHaveProperty('symbol')
      expect(data).toHaveProperty('signal')
      expect(data).toHaveProperty('strength')
      expect(data).toHaveProperty('confidence')
    })

    test('calculate_position_risk function', async () => {
      const { data, error } = await supabase.rpc('calculate_position_risk', {
        account_id_param: testAccountId,
        symbol_param: 'AAPL',
        position_size: 1000
      })

      expect(error).toBeNull()
      expect(typeof data).toBe('object')
      expect(data).toHaveProperty('account_balance')
      expect(data).toHaveProperty('var_95')
      expect(data).toHaveProperty('risk_percentage')
      expect(data).toHaveProperty('risk_level')
    })
  })

  describe('Database Triggers', () => {
    test('should update updated_at timestamp on record update', async () => {
      // Get initial timestamp
      const { data: initialBot } = await supabase
        .from('trading_bots')
        .select('updated_at')
        .eq('id', testBotId)
        .single()

      const initialTimestamp = new Date(initialBot!.updated_at)

      // Wait a moment and update
      await new Promise(resolve => setTimeout(resolve, 1000))

      await supabase
        .from('trading_bots')
        .update({ name: 'Updated Test Bot' })
        .eq('id', testBotId)

      // Check updated timestamp
      const { data: updatedBot } = await supabase
        .from('trading_bots')
        .select('updated_at')
        .eq('id', testBotId)
        .single()

      const updatedTimestamp = new Date(updatedBot!.updated_at)
      expect(updatedTimestamp.getTime()).toBeGreaterThan(initialTimestamp.getTime())
    })

    test('should create audit log on trading account changes', async () => {
      const initialLogCount = await getAuditLogCount('trading_accounts')

      // Update trading account
      await supabase
        .from('trading_accounts')
        .update({ balance: 15000 })
        .eq('id', testAccountId)

      const finalLogCount = await getAuditLogCount('trading_accounts')
      expect(finalLogCount).toBeGreaterThan(initialLogCount)
    })

    test('should create performance metrics on bot execution', async () => {
      const initialMetricsCount = await getPerformanceMetricsCount('bot')

      // Create bot execution
      await supabase
        .from('bot_executions')
        .insert({
          bot_id: testBotId,
          symbol_id: testSymbolId,
          action: 'buy',
          signal_strength: 0.8,
          reasoning: 'Test execution'
        })

      const finalMetricsCount = await getPerformanceMetricsCount('bot')
      expect(finalMetricsCount).toBeGreaterThan(initialMetricsCount)
    })
  })

  describe('Real-time Subscriptions', () => {
    test('should receive real-time updates for market data', (done) => {
      const channel = supabase
        .channel('test-market-data')
        .on(
          'postgres_changes',
          {
            event: 'INSERT',
            schema: 'public',
            table: 'market_data'
          },
          (payload) => {
            expect(payload.new).toBeDefined()
            expect(payload.new.symbol_id).toBe(testSymbolId)
            channel.unsubscribe()
            done()
          }
        )
        .subscribe()

      // Insert test market data after subscription
      setTimeout(async () => {
        await supabase
          .from('market_data')
          .insert({
            symbol_id: testSymbolId,
            timestamp: new Date().toISOString(),
            open: 150.00,
            high: 152.00,
            low: 149.00,
            close: 151.00,
            volume: 1000,
            timeframe: '1m'
          })
      }, 100)
    }, 10000)

    test('should receive real-time updates for bot status changes', (done) => {
      const channel = supabase
        .channel('test-bot-status')
        .on(
          'postgres_changes',
          {
            event: 'UPDATE',
            schema: 'public',
            table: 'trading_bots'
          },
          (payload) => {
            expect(payload.new).toBeDefined()
            expect(payload.new.status).toBe('active')
            channel.unsubscribe()
            done()
          }
        )
        .subscribe()

      // Update bot status after subscription
      setTimeout(async () => {
        await supabase
          .from('trading_bots')
          .update({ status: 'active' })
          .eq('id', testBotId)
      }, 100)
    }, 10000)
  })

  // Helper functions
  async function setupTestData() {
    // Create profile
    await supabase
      .from('profiles')
      .insert({
        id: testUserId,
        username: 'testuser',
        email: 'test@neuraltrade.ai',
        full_name: 'Test User'
      })

    // Create symbol
    const { data: symbol } = await supabase
      .from('symbols')
      .insert({
        symbol: 'AAPL',
        name: 'Apple Inc.',
        exchange: 'NASDAQ',
        asset_type: 'stock'
      })
      .select()
      .single()
    testSymbolId = symbol!.id

    // Create trading account
    const { data: account } = await supabase
      .from('trading_accounts')
      .insert({
        user_id: testUserId,
        name: 'Test Account',
        broker: 'Test Broker',
        balance: 10000
      })
      .select()
      .single()
    testAccountId = account!.id

    // Create neural model
    const { data: model } = await supabase
      .from('neural_models')
      .insert({
        user_id: testUserId,
        name: 'Test Model',
        model_type: 'lstm',
        architecture: { layers: [64, 32, 1] },
        status: 'trained'
      })
      .select()
      .single()
    testModelId = model!.id

    // Create trading bot
    const { data: bot } = await supabase
      .from('trading_bots')
      .insert({
        user_id: testUserId,
        account_id: testAccountId,
        name: 'Test Bot',
        strategy_type: 'neural_momentum',
        configuration: { test: true },
        symbols: ['AAPL'],
        model_ids: [testModelId]
      })
      .select()
      .single()
    testBotId = bot!.id

    // Create some market data
    await supabase
      .from('market_data')
      .insert([
        {
          symbol_id: testSymbolId,
          timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          open: 145.00,
          high: 148.00,
          low: 144.00,
          close: 147.00,
          volume: 5000,
          timeframe: '1d'
        },
        {
          symbol_id: testSymbolId,
          timestamp: new Date().toISOString(),
          open: 147.00,
          high: 150.00,
          low: 146.00,
          close: 149.00,
          volume: 3000,
          timeframe: '1d'
        }
      ])

    // Create model predictions
    await supabase
      .from('model_predictions')
      .insert([
        {
          model_id: testModelId,
          symbol_id: testSymbolId,
          prediction_timestamp: new Date().toISOString(),
          prediction_value: 0.75,
          confidence: 0.85,
          features: { price: 149.00, volume: 3000 }
        }
      ])
  }

  async function cleanupTestData() {
    // Delete in reverse order of creation due to foreign key constraints
    await supabase.from('model_predictions').delete().eq('model_id', testModelId)
    await supabase.from('market_data').delete().eq('symbol_id', testSymbolId)
    await supabase.from('bot_executions').delete().eq('bot_id', testBotId)
    await supabase.from('trading_bots').delete().eq('id', testBotId)
    await supabase.from('neural_models').delete().eq('id', testModelId)
    await supabase.from('trading_accounts').delete().eq('id', testAccountId)
    await supabase.from('symbols').delete().eq('id', testSymbolId)
    await supabase.from('profiles').delete().eq('id', testUserId)
    
    // Delete auth user
    await supabase.auth.admin.deleteUser(testUserId)
  }

  async function getAuditLogCount(entityType: string): Promise<number> {
    const { count } = await supabase
      .from('audit_logs')
      .select('*', { count: 'exact', head: true })
      .eq('entity_type', entityType)

    return count || 0
  }

  async function getPerformanceMetricsCount(entityType: string): Promise<number> {
    const { count } = await supabase
      .from('performance_metrics')
      .select('*', { count: 'exact', head: true })
      .eq('entity_type', entityType)

    return count || 0
  }
})