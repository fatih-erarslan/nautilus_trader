/**
 * Edge Function: Risk Calculator
 * Calculates portfolio risk metrics and position sizing in real-time
 */

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface RiskCalculationRequest {
  account_id: string
  proposed_trades?: ProposedTrade[]
  risk_settings?: RiskSettings
}

interface ProposedTrade {
  symbol: string
  side: 'long' | 'short'
  quantity: number
  price?: number
}

interface RiskSettings {
  max_portfolio_risk: number // Maximum portfolio risk percentage
  max_position_risk: number // Maximum single position risk percentage
  max_correlation_exposure: number // Maximum exposure to correlated assets
  var_confidence_level: number // VaR confidence level (0.95, 0.99)
}

interface RiskMetrics {
  portfolio_var: number
  portfolio_cvar: number
  max_drawdown_estimate: number
  sharpe_ratio_estimate: number
  correlation_risk: number
  concentration_risk: number
  individual_position_risks: PositionRisk[]
  recommendations: string[]
}

interface PositionRisk {
  symbol: string
  risk_percentage: number
  var_amount: number
  optimal_size: number
  current_size: number
  recommendation: 'increase' | 'decrease' | 'hold' | 'close'
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const { account_id, proposed_trades, risk_settings }: RiskCalculationRequest = await req.json()

    if (!account_id) {
      return new Response(
        JSON.stringify({ error: 'Account ID is required' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
      )
    }

    // Default risk settings
    const defaultRiskSettings: RiskSettings = {
      max_portfolio_risk: 0.05, // 5%
      max_position_risk: 0.02, // 2%
      max_correlation_exposure: 0.3, // 30%
      var_confidence_level: 0.95 // 95%
    }

    const settings = { ...defaultRiskSettings, ...risk_settings }

    // Get account information
    const { data: account, error: accountError } = await supabaseClient
      .from('trading_accounts')
      .select('*')
      .eq('id', account_id)
      .single()

    if (accountError || !account) {
      return new Response(
        JSON.stringify({ error: 'Trading account not found' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 404 }
      )
    }

    // Get current positions
    const { data: positions } = await supabaseClient
      .from('positions')
      .select(`
        *,
        symbols (symbol, name)
      `)
      .eq('account_id', account_id)
      .is('closed_at', null)

    // Get historical market data for risk calculations
    const symbolIds = positions?.map(p => p.symbol_id) || []
    const proposedSymbols = proposed_trades?.map(t => t.symbol) || []
    
    // Get symbol IDs for proposed trades
    let proposedSymbolIds: string[] = []
    if (proposedSymbols.length > 0) {
      const { data: symbolData } = await supabaseClient
        .from('symbols')
        .select('id, symbol')
        .in('symbol', proposedSymbols)
      
      proposedSymbolIds = symbolData?.map(s => s.id) || []
    }

    const allSymbolIds = [...new Set([...symbolIds, ...proposedSymbolIds])]

    // Get historical market data (last 100 days)
    const { data: marketData } = await supabaseClient
      .from('market_data')
      .select('*')
      .in('symbol_id', allSymbolIds)
      .eq('timeframe', '1d')
      .gte('timestamp', new Date(Date.now() - 100 * 24 * 60 * 60 * 1000).toISOString())
      .order('timestamp', { ascending: false })

    // Calculate risk metrics
    const riskMetrics = await calculateRiskMetrics({
      account,
      positions: positions || [],
      proposedTrades: proposed_trades || [],
      marketData: marketData || [],
      settings
    })

    // Store risk calculation in performance metrics
    await supabaseClient
      .from('performance_metrics')
      .insert({
        entity_type: 'account',
        entity_id: account_id,
        metric_type: 'portfolio_var',
        metric_value: riskMetrics.portfolio_var,
        metadata: {
          calculation_timestamp: new Date().toISOString(),
          risk_settings: settings,
          position_count: positions?.length || 0
        }
      })

    // Create alerts for high risk situations
    if (riskMetrics.portfolio_var > account.balance * settings.max_portfolio_risk) {
      await supabaseClient
        .from('alerts')
        .insert({
          user_id: account.user_id,
          title: 'Portfolio Risk Limit Exceeded',
          message: `Portfolio VaR (${(riskMetrics.portfolio_var / account.balance * 100).toFixed(1)}%) exceeds limit (${(settings.max_portfolio_risk * 100).toFixed(1)}%)`,
          severity: 'warning',
          entity_type: 'account',
          entity_id: account_id,
          metadata: {
            portfolio_var: riskMetrics.portfolio_var,
            risk_limit: settings.max_portfolio_risk,
            account_balance: account.balance
          }
        })
    }

    return new Response(
      JSON.stringify({
        success: true,
        account_balance: account.balance,
        risk_metrics: riskMetrics,
        risk_settings: settings
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Risk calculator error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})

// Main risk calculation function
async function calculateRiskMetrics(params: {
  account: any
  positions: any[]
  proposedTrades: ProposedTrade[]
  marketData: any[]
  settings: RiskSettings
}): Promise<RiskMetrics> {
  const { account, positions, proposedTrades, marketData, settings } = params

  // Group market data by symbol
  const marketDataBySymbol = groupMarketDataBySymbol(marketData)

  // Calculate returns for each symbol
  const returnsData = calculateReturns(marketDataBySymbol)

  // Calculate correlation matrix
  const correlationMatrix = calculateCorrelationMatrix(returnsData)

  // Calculate individual position risks
  const positionRisks = await calculatePositionRisks({
    positions,
    proposedTrades,
    marketDataBySymbol,
    returnsData,
    accountBalance: account.balance,
    settings
  })

  // Calculate portfolio-level risk metrics
  const portfolioRisk = calculatePortfolioRisk({
    positions,
    proposedTrades,
    returnsData,
    correlationMatrix,
    accountBalance: account.balance,
    settings
  })

  // Generate recommendations
  const recommendations = generateRiskRecommendations({
    portfolioRisk,
    positionRisks,
    settings,
    accountBalance: account.balance
  })

  return {
    portfolio_var: portfolioRisk.var,
    portfolio_cvar: portfolioRisk.cvar,
    max_drawdown_estimate: portfolioRisk.maxDrawdown,
    sharpe_ratio_estimate: portfolioRisk.sharpeRatio,
    correlation_risk: portfolioRisk.correlationRisk,
    concentration_risk: portfolioRisk.concentrationRisk,
    individual_position_risks: positionRisks,
    recommendations
  }
}

function groupMarketDataBySymbol(marketData: any[]): Record<string, any[]> {
  const grouped: Record<string, any[]> = {}
  
  for (const data of marketData) {
    if (!grouped[data.symbol_id]) {
      grouped[data.symbol_id] = []
    }
    grouped[data.symbol_id].push(data)
  }
  
  // Sort each symbol's data by timestamp
  for (const symbolId in grouped) {
    grouped[symbolId].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }
  
  return grouped
}

function calculateReturns(marketDataBySymbol: Record<string, any[]>): Record<string, number[]> {
  const returns: Record<string, number[]> = {}
  
  for (const [symbolId, data] of Object.entries(marketDataBySymbol)) {
    if (data.length < 2) continue
    
    const symbolReturns: number[] = []
    for (let i = 1; i < data.length; i++) {
      const currentPrice = data[i - 1].close // More recent price
      const previousPrice = data[i].close // Older price
      const return_ = (currentPrice - previousPrice) / previousPrice
      symbolReturns.push(return_)
    }
    
    returns[symbolId] = symbolReturns.reverse() // Reverse to have chronological order
  }
  
  return returns
}

function calculateCorrelationMatrix(returnsData: Record<string, number[]>): Record<string, Record<string, number>> {
  const symbols = Object.keys(returnsData)
  const correlations: Record<string, Record<string, number>> = {}
  
  for (const symbol1 of symbols) {
    correlations[symbol1] = {}
    for (const symbol2 of symbols) {
      if (symbol1 === symbol2) {
        correlations[symbol1][symbol2] = 1
      } else {
        correlations[symbol1][symbol2] = calculateCorrelation(
          returnsData[symbol1],
          returnsData[symbol2]
        )
      }
    }
  }
  
  return correlations
}

function calculateCorrelation(returns1: number[], returns2: number[]): number {
  const minLength = Math.min(returns1.length, returns2.length)
  if (minLength < 2) return 0
  
  const x = returns1.slice(0, minLength)
  const y = returns2.slice(0, minLength)
  
  const meanX = x.reduce((sum, val) => sum + val, 0) / x.length
  const meanY = y.reduce((sum, val) => sum + val, 0) / y.length
  
  let numerator = 0
  let sumXSquared = 0
  let sumYSquared = 0
  
  for (let i = 0; i < x.length; i++) {
    const xDiff = x[i] - meanX
    const yDiff = y[i] - meanY
    numerator += xDiff * yDiff
    sumXSquared += xDiff * xDiff
    sumYSquared += yDiff * yDiff
  }
  
  const denominator = Math.sqrt(sumXSquared * sumYSquared)
  return denominator === 0 ? 0 : numerator / denominator
}

async function calculatePositionRisks(params: {
  positions: any[]
  proposedTrades: ProposedTrade[]
  marketDataBySymbol: Record<string, any[]>
  returnsData: Record<string, number[]>
  accountBalance: number
  settings: RiskSettings
}): Promise<PositionRisk[]> {
  const { positions, proposedTrades, marketDataBySymbol, returnsData, accountBalance, settings } = params
  
  const positionRisks: PositionRisk[] = []
  
  // Calculate risk for existing positions
  for (const position of positions) {
    const symbolId = position.symbol_id
    const symbol = position.symbols?.symbol || symbolId
    
    const returns = returnsData[symbolId] || []
    if (returns.length === 0) continue
    
    const volatility = calculateVolatility(returns)
    const currentValue = Math.abs(position.quantity * (position.current_price || position.entry_price))
    
    // Calculate VaR using parametric method
    const confidenceLevel = settings.var_confidence_level
    const zScore = getZScore(confidenceLevel)
    const var_ = currentValue * volatility * zScore
    
    const riskPercentage = var_ / accountBalance
    
    // Calculate optimal position size using Kelly criterion (simplified)
    const expectedReturn = returns.length > 0 ? 
      returns.reduce((sum, r) => sum + r, 0) / returns.length : 0
    const optimalSize = expectedReturn > 0 ? 
      (expectedReturn / (volatility * volatility)) * accountBalance * settings.max_position_risk : 0
    
    let recommendation: 'increase' | 'decrease' | 'hold' | 'close'
    if (riskPercentage > settings.max_position_risk) {
      recommendation = 'decrease'
    } else if (currentValue < optimalSize * 0.8 && expectedReturn > 0) {
      recommendation = 'increase'
    } else if (expectedReturn < -0.1) { // Strong negative expectation
      recommendation = 'close'
    } else {
      recommendation = 'hold'
    }
    
    positionRisks.push({
      symbol,
      risk_percentage: riskPercentage,
      var_amount: var_,
      optimal_size: optimalSize,
      current_size: currentValue,
      recommendation
    })
  }
  
  // Calculate risk for proposed trades
  for (const trade of proposedTrades) {
    // Find symbol ID for proposed trade
    const symbolData = await findSymbolByName(trade.symbol)
    if (!symbolData) continue
    
    const returns = returnsData[symbolData.id] || []
    if (returns.length === 0) continue
    
    const volatility = calculateVolatility(returns)
    const tradeValue = Math.abs(trade.quantity * (trade.price || 0))
    
    const confidenceLevel = settings.var_confidence_level
    const zScore = getZScore(confidenceLevel)
    const var_ = tradeValue * volatility * zScore
    
    const riskPercentage = var_ / accountBalance
    
    positionRisks.push({
      symbol: trade.symbol,
      risk_percentage: riskPercentage,
      var_amount: var_,
      optimal_size: 0, // Will be calculated separately for new positions
      current_size: 0,
      recommendation: riskPercentage > settings.max_position_risk ? 'decrease' : 'hold'
    })
  }
  
  return positionRisks
}

function calculatePortfolioRisk(params: {
  positions: any[]
  proposedTrades: ProposedTrade[]
  returnsData: Record<string, number[]>
  correlationMatrix: Record<string, Record<string, number>>
  accountBalance: number
  settings: RiskSettings
}) {
  const { positions, proposedTrades, returnsData, correlationMatrix, accountBalance, settings } = params
  
  // Calculate portfolio variance using correlation matrix
  const weights: Record<string, number> = {}
  let totalValue = 0
  
  // Add existing positions
  for (const position of positions) {
    const value = Math.abs(position.quantity * (position.current_price || position.entry_price))
    weights[position.symbol_id] = value
    totalValue += value
  }
  
  // Add proposed trades
  for (const trade of proposedTrades) {
    const value = Math.abs(trade.quantity * (trade.price || 0))
    weights[trade.symbol] = (weights[trade.symbol] || 0) + value
    totalValue += value
  }
  
  // Normalize weights
  for (const symbolId in weights) {
    weights[symbolId] = weights[symbolId] / totalValue
  }
  
  // Calculate portfolio variance
  let portfolioVariance = 0
  const symbols = Object.keys(weights)
  
  for (const symbol1 of symbols) {
    for (const symbol2 of symbols) {
      const weight1 = weights[symbol1]
      const weight2 = weights[symbol2]
      const correlation = correlationMatrix[symbol1]?.[symbol2] || 0
      const vol1 = calculateVolatility(returnsData[symbol1] || [])
      const vol2 = calculateVolatility(returnsData[symbol2] || [])
      
      portfolioVariance += weight1 * weight2 * correlation * vol1 * vol2
    }
  }
  
  const portfolioVolatility = Math.sqrt(portfolioVariance)
  
  // Calculate VaR and CVaR
  const confidenceLevel = settings.var_confidence_level
  const zScore = getZScore(confidenceLevel)
  const portfolioVar = totalValue * portfolioVolatility * zScore
  
  // CVaR estimation (simplified)
  const cvarMultiplier = 1 / (1 - confidenceLevel) * Math.exp(-0.5 * zScore * zScore) / Math.sqrt(2 * Math.PI)
  const portfolioCvar = portfolioVar * cvarMultiplier
  
  // Estimate max drawdown (simplified)
  const maxDrawdown = portfolioVolatility * Math.sqrt(252) * 2 // 2 standard deviations over a year
  
  // Estimate Sharpe ratio
  const portfolioReturn = symbols.reduce((sum, symbolId) => {
    const returns = returnsData[symbolId] || []
    const avgReturn = returns.length > 0 ? returns.reduce((s, r) => s + r, 0) / returns.length : 0
    return sum + weights[symbolId] * avgReturn
  }, 0)
  
  const sharpeRatio = portfolioVolatility > 0 ? portfolioReturn / portfolioVolatility : 0
  
  // Calculate concentration risk
  const maxWeight = Math.max(...Object.values(weights))
  const concentrationRisk = maxWeight > 0.3 ? maxWeight : 0
  
  // Calculate correlation risk (average absolute correlation)
  let totalCorrelations = 0
  let correlationSum = 0
  for (const symbol1 of symbols) {
    for (const symbol2 of symbols) {
      if (symbol1 !== symbol2) {
        correlationSum += Math.abs(correlationMatrix[symbol1]?.[symbol2] || 0)
        totalCorrelations++
      }
    }
  }
  const correlationRisk = totalCorrelations > 0 ? correlationSum / totalCorrelations : 0
  
  return {
    var: portfolioVar,
    cvar: portfolioCvar,
    maxDrawdown,
    sharpeRatio,
    correlationRisk,
    concentrationRisk
  }
}

function generateRiskRecommendations(params: {
  portfolioRisk: any
  positionRisks: PositionRisk[]
  settings: RiskSettings
  accountBalance: number
}): string[] {
  const { portfolioRisk, positionRisks, settings, accountBalance } = params
  const recommendations: string[] = []
  
  // Portfolio-level recommendations
  if (portfolioRisk.var > accountBalance * settings.max_portfolio_risk) {
    recommendations.push(`Reduce portfolio risk: Current VaR (${(portfolioRisk.var / accountBalance * 100).toFixed(1)}%) exceeds limit (${(settings.max_portfolio_risk * 100).toFixed(1)}%)`)
  }
  
  if (portfolioRisk.concentrationRisk > 0.3) {
    recommendations.push(`Diversify portfolio: Single position represents ${(portfolioRisk.concentrationRisk * 100).toFixed(1)}% of portfolio`)
  }
  
  if (portfolioRisk.correlationRisk > settings.max_correlation_exposure) {
    recommendations.push(`Reduce correlation risk: Average correlation (${(portfolioRisk.correlationRisk * 100).toFixed(1)}%) is high`)
  }
  
  if (portfolioRisk.sharpeRatio < 0.5) {
    recommendations.push('Consider improving risk-adjusted returns: Current Sharpe ratio is low')
  }
  
  // Position-level recommendations
  const highRiskPositions = positionRisks.filter(p => p.risk_percentage > settings.max_position_risk)
  if (highRiskPositions.length > 0) {
    recommendations.push(`Reduce size of high-risk positions: ${highRiskPositions.map(p => p.symbol).join(', ')}`)
  }
  
  const closePositions = positionRisks.filter(p => p.recommendation === 'close')
  if (closePositions.length > 0) {
    recommendations.push(`Consider closing underperforming positions: ${closePositions.map(p => p.symbol).join(', ')}`)
  }
  
  return recommendations
}

// Helper functions
function calculateVolatility(returns: number[]): number {
  if (returns.length < 2) return 0
  
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1)
  
  return Math.sqrt(variance)
}

function getZScore(confidenceLevel: number): number {
  // Simplified z-score lookup
  const zScores: Record<string, number> = {
    '0.90': 1.282,
    '0.95': 1.645,
    '0.99': 2.326
  }
  
  return zScores[confidenceLevel.toString()] || 1.645
}

async function findSymbolByName(symbol: string): Promise<{ id: string; symbol: string } | null> {
  // This would typically query the database, but for the edge function we'll simulate
  // In a real implementation, you'd need to pass symbol mappings or make a DB call
  return { id: `symbol_${symbol}`, symbol }
}