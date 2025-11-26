/**
 * Edge Function: Market Data Processor
 * Processes and validates incoming market data in real-time
 */

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface MarketDataPayload {
  symbol: string
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  timeframe: string
}

interface ProcessedMarketData extends MarketDataPayload {
  symbol_id: string
  technical_indicators?: {
    sma_20?: number
    sma_50?: number
    rsi?: number
    macd?: {
      macd: number
      signal: number
      histogram: number
    }
  }
  anomaly_score?: number
  validated: boolean
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const { data: requestData } = await req.json()
    const marketData: MarketDataPayload[] = Array.isArray(requestData) ? requestData : [requestData]

    const processedData: ProcessedMarketData[] = []

    for (const data of marketData) {
      try {
        // Validate data
        const validation = validateMarketData(data)
        if (!validation.isValid) {
          console.error(`Invalid market data for ${data.symbol}:`, validation.errors)
          continue
        }

        // Get symbol ID
        const { data: symbol, error: symbolError } = await supabaseClient
          .from('symbols')
          .select('id')
          .eq('symbol', data.symbol)
          .single()

        if (symbolError || !symbol) {
          console.error(`Symbol not found: ${data.symbol}`)
          continue
        }

        // Get historical data for technical indicators
        const { data: historicalData } = await supabaseClient
          .from('market_data')
          .select('close, timestamp')
          .eq('symbol_id', symbol.id)
          .eq('timeframe', data.timeframe)
          .order('timestamp', { ascending: false })
          .limit(50)

        // Calculate technical indicators
        const technicalIndicators = calculateTechnicalIndicators([
          ...historicalData?.map(d => d.close) || [],
          data.close
        ])

        // Detect anomalies
        const anomalyScore = detectAnomalies(data, historicalData || [])

        // Create processed data object
        const processed: ProcessedMarketData = {
          ...data,
          symbol_id: symbol.id,
          technical_indicators: technicalIndicators,
          anomaly_score: anomalyScore,
          validated: true
        }

        processedData.push(processed)

        // Insert into database
        const { error: insertError } = await supabaseClient
          .from('market_data')
          .insert({
            symbol_id: symbol.id,
            timestamp: data.timestamp,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume,
            timeframe: data.timeframe
          })

        if (insertError) {
          console.error(`Failed to insert market data for ${data.symbol}:`, insertError)
        }

        // Update performance metrics
        await supabaseClient
          .from('performance_metrics')
          .insert({
            entity_type: 'system',
            entity_id: 'market-data-processor',
            metric_type: 'data_processed',
            metric_value: 1,
            metadata: {
              symbol: data.symbol,
              timeframe: data.timeframe,
              anomaly_score: anomalyScore
            }
          })

        // Trigger alerts for anomalies
        if (anomalyScore > 0.8) {
          await supabaseClient
            .from('alerts')
            .insert({
              user_id: '00000000-0000-0000-0000-000000000000', // System alerts
              title: 'Market Data Anomaly Detected',
              message: `Unusual market activity detected for ${data.symbol}: anomaly score ${anomalyScore.toFixed(2)}`,
              severity: 'warning',
              entity_type: 'symbol',
              entity_id: symbol.id,
              metadata: {
                anomaly_score: anomalyScore,
                timeframe: data.timeframe,
                price_change: ((data.close - data.open) / data.open * 100).toFixed(2)
              }
            })
        }

      } catch (error) {
        console.error(`Error processing market data for ${data.symbol}:`, error)
      }
    }

    return new Response(
      JSON.stringify({
        success: true,
        processed_count: processedData.length,
        total_count: marketData.length,
        data: processedData
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Market data processor error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})

// Validation function
function validateMarketData(data: MarketDataPayload): { isValid: boolean; errors: string[] } {
  const errors: string[] = []

  if (!data.symbol || typeof data.symbol !== 'string') {
    errors.push('Invalid or missing symbol')
  }

  if (!data.timestamp || isNaN(Date.parse(data.timestamp))) {
    errors.push('Invalid or missing timestamp')
  }

  if (typeof data.open !== 'number' || data.open <= 0) {
    errors.push('Invalid open price')
  }

  if (typeof data.high !== 'number' || data.high <= 0) {
    errors.push('Invalid high price')
  }

  if (typeof data.low !== 'number' || data.low <= 0) {
    errors.push('Invalid low price')
  }

  if (typeof data.close !== 'number' || data.close <= 0) {
    errors.push('Invalid close price')
  }

  if (typeof data.volume !== 'number' || data.volume < 0) {
    errors.push('Invalid volume')
  }

  if (data.high < data.low) {
    errors.push('High price cannot be less than low price')
  }

  if (data.open > data.high || data.open < data.low) {
    errors.push('Open price must be between high and low')
  }

  if (data.close > data.high || data.close < data.low) {
    errors.push('Close price must be between high and low')
  }

  return {
    isValid: errors.length === 0,
    errors
  }
}

// Technical indicators calculation
function calculateTechnicalIndicators(prices: number[]) {
  if (prices.length < 20) return {}

  const sma20 = calculateSMA(prices, 20)
  const sma50 = prices.length >= 50 ? calculateSMA(prices, 50) : undefined
  const rsi = calculateRSI(prices, 14)
  const macd = calculateMACD(prices)

  return {
    sma_20: sma20,
    sma_50: sma50,
    rsi: rsi,
    macd: macd
  }
}

function calculateSMA(prices: number[], period: number): number {
  const relevantPrices = prices.slice(-period)
  return relevantPrices.reduce((sum, price) => sum + price, 0) / relevantPrices.length
}

function calculateRSI(prices: number[], period: number): number {
  if (prices.length < period + 1) return 50

  const changes = []
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1])
  }

  const recentChanges = changes.slice(-period)
  const gains = recentChanges.filter(change => change > 0)
  const losses = recentChanges.filter(change => change < 0).map(loss => Math.abs(loss))

  const avgGain = gains.length > 0 ? gains.reduce((sum, gain) => sum + gain, 0) / period : 0
  const avgLoss = losses.length > 0 ? losses.reduce((sum, loss) => sum + loss, 0) / period : 0

  if (avgLoss === 0) return 100
  const rs = avgGain / avgLoss
  return 100 - (100 / (1 + rs))
}

function calculateMACD(prices: number[]) {
  if (prices.length < 26) return { macd: 0, signal: 0, histogram: 0 }

  const ema12 = calculateEMA(prices, 12)
  const ema26 = calculateEMA(prices, 26)
  const macd = ema12 - ema26

  // For simplicity, using SMA instead of EMA for signal line
  const macdLine = [macd] // In real implementation, you'd have a history of MACD values
  const signal = macd // Simplified signal line
  const histogram = macd - signal

  return { macd, signal, histogram }
}

function calculateEMA(prices: number[], period: number): number {
  const multiplier = 2 / (period + 1)
  let ema = prices[0]

  for (let i = 1; i < prices.length; i++) {
    ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
  }

  return ema
}

// Anomaly detection
function detectAnomalies(currentData: MarketDataPayload, historicalData: any[]): number {
  if (historicalData.length < 10) return 0

  const recentPrices = historicalData.slice(0, 20).map(d => d.close)
  const avgPrice = recentPrices.reduce((sum, price) => sum + price, 0) / recentPrices.length
  const priceStdDev = Math.sqrt(
    recentPrices.reduce((sum, price) => sum + Math.pow(price - avgPrice, 2), 0) / recentPrices.length
  )

  const priceDeviation = Math.abs(currentData.close - avgPrice) / priceStdDev
  const volumeRatio = historicalData.length > 0 ? 
    currentData.volume / (historicalData.slice(0, 10).reduce((sum, d) => sum + (d.volume || 0), 0) / 10) : 1

  // Combine price and volume anomalies
  const priceAnomalyScore = Math.min(priceDeviation / 3, 1) // Normalize to 0-1
  const volumeAnomalyScore = Math.min(Math.abs(Math.log(volumeRatio)) / 2, 1) // Normalize to 0-1

  return (priceAnomalyScore + volumeAnomalyScore) / 2
}