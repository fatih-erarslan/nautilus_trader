/**
 * Edge Function: Trading Signal Generator
 * Generates trading signals based on neural model predictions and market data
 */

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface SignalRequest {
  symbol: string
  bot_id?: string
  model_ids?: string[]
  override_params?: {
    risk_tolerance?: number
    signal_threshold?: number
    time_horizon?: number
  }
}

interface TradingSignal {
  symbol: string
  action: 'buy' | 'sell' | 'hold'
  strength: number
  confidence: number
  reasoning: string[]
  risk_assessment: {
    risk_level: 'low' | 'medium' | 'high'
    max_position_size: number
    stop_loss: number
    take_profit: number
  }
  metadata: {
    generated_at: string
    model_predictions: any[]
    market_conditions: any
    technical_indicators: any
  }
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

    const { symbol, bot_id, model_ids, override_params }: SignalRequest = await req.json()

    if (!symbol) {
      return new Response(
        JSON.stringify({ error: 'Symbol is required' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
      )
    }

    // Get symbol information
    const { data: symbolData, error: symbolError } = await supabaseClient
      .from('symbols')
      .select('*')
      .eq('symbol', symbol)
      .single()

    if (symbolError || !symbolData) {
      return new Response(
        JSON.stringify({ error: 'Symbol not found' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 404 }
      )
    }

    // Get bot configuration if bot_id provided
    let botConfig: any = null
    if (bot_id) {
      const { data: bot } = await supabaseClient
        .from('trading_bots')
        .select('*')
        .eq('id', bot_id)
        .single()
      
      botConfig = bot
    }

    // Determine which models to use
    let modelsToUse = model_ids
    if (!modelsToUse && botConfig) {
      modelsToUse = botConfig.model_ids
    }

    // Get latest market data
    const { data: marketData } = await supabaseClient
      .from('market_data')
      .select('*')
      .eq('symbol_id', symbolData.id)
      .order('timestamp', { ascending: false })
      .limit(50)

    if (!marketData || marketData.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No market data available for symbol' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
      )
    }

    // Get model predictions if models specified
    let modelPredictions: any[] = []
    if (modelsToUse && modelsToUse.length > 0) {
      const { data: predictions } = await supabaseClient
        .from('model_predictions')
        .select(`
          *,
          neural_models (id, name, model_type, performance_metrics)
        `)
        .eq('symbol_id', symbolData.id)
        .in('model_id', modelsToUse)
        .gte('prediction_timestamp', new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()) // Last 2 hours
        .order('prediction_timestamp', { ascending: false })

      modelPredictions = predictions || []
    }

    // Get recent news sentiment
    const { data: newsData } = await supabaseClient
      .from('news_data')
      .select('sentiment_score, relevance_score, title')
      .contains('symbols', [symbol])
      .gte('published_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()) // Last 24 hours
      .order('published_at', { ascending: false })
      .limit(10)

    // Calculate technical indicators
    const prices = marketData.map(d => d.close)
    const technicalIndicators = calculateTechnicalIndicators(prices)

    // Analyze market conditions
    const marketConditions = analyzeMarketConditions(marketData, newsData || [])

    // Generate trading signal
    const signal = await generateTradingSignal({
      symbol,
      symbolData,
      marketData,
      modelPredictions,
      technicalIndicators,
      marketConditions,
      botConfig,
      overrideParams: override_params
    })

    // Store the signal in bot_executions if bot_id provided
    if (bot_id && signal.action !== 'hold') {
      await supabaseClient
        .from('bot_executions')
        .insert({
          bot_id,
          symbol_id: symbolData.id,
          action: signal.action,
          signal_strength: signal.strength,
          reasoning: signal.reasoning.join('; '),
          metadata: signal.metadata
        })
    }

    // Store performance metric
    await supabaseClient
      .from('performance_metrics')
      .insert({
        entity_type: 'system',
        entity_id: 'signal-generator',
        metric_type: 'signals_generated',
        metric_value: 1,
        metadata: {
          symbol,
          action: signal.action,
          strength: signal.strength,
          confidence: signal.confidence
        }
      })

    return new Response(
      JSON.stringify({
        success: true,
        signal
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Signal generator error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})

// Main signal generation logic
async function generateTradingSignal(params: {
  symbol: string
  symbolData: any
  marketData: any[]
  modelPredictions: any[]
  technicalIndicators: any
  marketConditions: any
  botConfig: any
  overrideParams?: any
}): Promise<TradingSignal> {
  const {
    symbol,
    marketData,
    modelPredictions,
    technicalIndicators,
    marketConditions,
    botConfig,
    overrideParams
  } = params

  // Default parameters
  const riskTolerance = overrideParams?.risk_tolerance || botConfig?.risk_limit || 0.05
  const signalThreshold = overrideParams?.signal_threshold || 0.6
  const timeHorizon = overrideParams?.time_horizon || 1 // hours

  const reasoning: string[] = []
  let signalStrength = 0
  let confidence = 0
  let action: 'buy' | 'sell' | 'hold' = 'hold'

  // Analyze model predictions
  if (modelPredictions.length > 0) {
    const avgPrediction = modelPredictions.reduce((sum, p) => sum + p.prediction_value, 0) / modelPredictions.length
    const avgConfidence = modelPredictions.reduce((sum, p) => sum + (p.confidence || 0.5), 0) / modelPredictions.length
    
    signalStrength += avgPrediction * 0.4 // 40% weight for model predictions
    confidence += avgConfidence * 0.4
    
    reasoning.push(`Model consensus: ${avgPrediction.toFixed(3)} (${modelPredictions.length} models)`)
  }

  // Analyze technical indicators
  const currentPrice = marketData[0].close
  const techSignal = analyzeTechnicalIndicators(technicalIndicators, currentPrice)
  signalStrength += techSignal.strength * 0.3 // 30% weight for technical analysis
  confidence += techSignal.confidence * 0.3
  reasoning.push(...techSignal.reasoning)

  // Analyze market sentiment
  const sentimentSignal = analyzeMarketSentiment(marketConditions)
  signalStrength += sentimentSignal.strength * 0.2 // 20% weight for sentiment
  confidence += sentimentSignal.confidence * 0.2
  reasoning.push(...sentimentSignal.reasoning)

  // Analyze volume and momentum
  const momentumSignal = analyzeMomentum(marketData)
  signalStrength += momentumSignal.strength * 0.1 // 10% weight for momentum
  confidence += momentumSignal.confidence * 0.1
  reasoning.push(...momentumSignal.reasoning)

  // Normalize values
  signalStrength = Math.max(-1, Math.min(1, signalStrength))
  confidence = Math.max(0, Math.min(1, confidence))

  // Determine action based on signal strength and threshold
  if (signalStrength > signalThreshold && confidence > 0.5) {
    action = 'buy'
  } else if (signalStrength < -signalThreshold && confidence > 0.5) {
    action = 'sell'
  } else {
    action = 'hold'
    reasoning.push('Signal strength below threshold or confidence too low')
  }

  // Calculate risk assessment
  const riskAssessment = calculateRiskAssessment({
    currentPrice,
    volatility: marketConditions.volatility,
    riskTolerance,
    signalStrength: Math.abs(signalStrength),
    confidence
  })

  return {
    symbol,
    action,
    strength: Math.abs(signalStrength),
    confidence,
    reasoning,
    risk_assessment: riskAssessment,
    metadata: {
      generated_at: new Date().toISOString(),
      model_predictions: modelPredictions,
      market_conditions: marketConditions,
      technical_indicators: technicalIndicators
    }
  }
}

// Technical analysis
function analyzeTechnicalIndicators(indicators: any, currentPrice: number) {
  let strength = 0
  let confidence = 0
  const reasoning: string[] = []

  if (indicators.sma_20) {
    if (currentPrice > indicators.sma_20) {
      strength += 0.2
      reasoning.push('Price above SMA(20)')
    } else {
      strength -= 0.2
      reasoning.push('Price below SMA(20)')
    }
    confidence += 0.3
  }

  if (indicators.rsi) {
    if (indicators.rsi > 70) {
      strength -= 0.3
      reasoning.push(`RSI overbought (${indicators.rsi.toFixed(1)})`)
      confidence += 0.4
    } else if (indicators.rsi < 30) {
      strength += 0.3
      reasoning.push(`RSI oversold (${indicators.rsi.toFixed(1)})`)
      confidence += 0.4
    } else {
      reasoning.push(`RSI neutral (${indicators.rsi.toFixed(1)})`)
      confidence += 0.2
    }
  }

  if (indicators.macd) {
    if (indicators.macd.histogram > 0) {
      strength += 0.1
      reasoning.push('MACD bullish')
    } else {
      strength -= 0.1
      reasoning.push('MACD bearish')
    }
    confidence += 0.2
  }

  return { strength, confidence, reasoning }
}

// Market sentiment analysis
function analyzeMarketSentiment(marketConditions: any) {
  let strength = 0
  let confidence = 0
  const reasoning: string[] = []

  if (marketConditions.news_sentiment !== undefined) {
    strength += marketConditions.news_sentiment * 0.5
    confidence += 0.4
    reasoning.push(`News sentiment: ${marketConditions.news_sentiment.toFixed(2)}`)
  }

  if (marketConditions.market_stress) {
    strength -= 0.2
    confidence += 0.3
    reasoning.push('High market stress detected')
  }

  return { strength, confidence, reasoning }
}

// Momentum analysis
function analyzeMomentum(marketData: any[]) {
  let strength = 0
  let confidence = 0
  const reasoning: string[] = []

  if (marketData.length >= 3) {
    const recent = marketData.slice(0, 3)
    const priceChange = (recent[0].close - recent[2].close) / recent[2].close
    const volumeRatio = recent[0].volume / recent[2].volume

    if (priceChange > 0.02) {
      strength += 0.3
      reasoning.push(`Strong upward momentum (+${(priceChange * 100).toFixed(1)}%)`)
    } else if (priceChange < -0.02) {
      strength -= 0.3
      reasoning.push(`Strong downward momentum (${(priceChange * 100).toFixed(1)}%)`)
    }

    if (volumeRatio > 1.5) {
      confidence += 0.3
      reasoning.push('High volume confirmation')
    }

    confidence += 0.4
  }

  return { strength, confidence, reasoning }
}

// Calculate technical indicators
function calculateTechnicalIndicators(prices: number[]) {
  // Simplified implementation - in production, use more sophisticated calculations
  const sma20 = prices.length >= 20 ? 
    prices.slice(0, 20).reduce((sum, price) => sum + price, 0) / 20 : null
  
  const rsi = calculateRSI(prices.slice(0, 14))
  
  return { sma_20: sma20, rsi }
}

function calculateRSI(prices: number[]): number {
  if (prices.length < 14) return 50

  const gains = []
  const losses = []

  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1]
    if (change > 0) {
      gains.push(change)
      losses.push(0)
    } else {
      gains.push(0)
      losses.push(Math.abs(change))
    }
  }

  const avgGain = gains.reduce((sum, gain) => sum + gain, 0) / gains.length
  const avgLoss = losses.reduce((sum, loss) => sum + loss, 0) / losses.length

  if (avgLoss === 0) return 100
  const rs = avgGain / avgLoss
  return 100 - (100 / (1 + rs))
}

// Analyze market conditions
function analyzeMarketConditions(marketData: any[], newsData: any[]) {
  const prices = marketData.slice(0, 20).map(d => d.close)
  const volatility = calculateVolatility(prices)
  
  const newsSentiment = newsData.length > 0 ?
    newsData.reduce((sum, news) => sum + (news.sentiment_score || 0), 0) / newsData.length : 0

  return {
    volatility,
    news_sentiment: newsSentiment,
    market_stress: volatility > 0.05 // High volatility threshold
  }
}

function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0
  
  const returns = []
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1])
  }
  
  const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length
  
  return Math.sqrt(variance)
}

// Risk assessment calculation
function calculateRiskAssessment(params: {
  currentPrice: number
  volatility: number
  riskTolerance: number
  signalStrength: number
  confidence: number
}) {
  const { currentPrice, volatility, riskTolerance, signalStrength, confidence } = params

  // Calculate position size based on Kelly criterion (simplified)
  const maxPositionSize = (signalStrength * confidence * riskTolerance) * 1000 // Base amount

  // Calculate stop loss and take profit levels
  const stopLossDistance = Math.max(volatility * 2, 0.02) // 2% minimum
  const takeProfitDistance = stopLossDistance * 2 // 2:1 risk-reward ratio

  const stopLoss = currentPrice * (1 - stopLossDistance)
  const takeProfit = currentPrice * (1 + takeProfitDistance)

  const riskLevel = volatility > 0.05 ? 'high' : volatility > 0.02 ? 'medium' : 'low'

  return {
    risk_level: riskLevel as 'low' | 'medium' | 'high',
    max_position_size: maxPositionSize,
    stop_loss: stopLoss,
    take_profit: takeProfit
  }
}