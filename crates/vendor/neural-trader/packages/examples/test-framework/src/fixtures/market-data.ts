/**
 * Market data fixtures for testing
 */

import { MarketData } from '../types';

/**
 * Generate OHLCV market data
 */
export function generateMarketData(
  symbol: string,
  count: number,
  options: {
    startPrice?: number;
    volatility?: number;
    trend?: number;
    intervalMs?: number;
  } = {}
): MarketData[] {
  const {
    startPrice = 100,
    volatility = 0.02,
    trend = 0,
    intervalMs = 60000
  } = options;

  const data: MarketData[] = [];
  let currentPrice = startPrice;
  let timestamp = Date.now() - count * intervalMs;

  for (let i = 0; i < count; i++) {
    // Generate random walk with trend
    const change = (Math.random() - 0.5) * volatility * currentPrice + trend;
    currentPrice += change;

    // Generate OHLC
    const open = currentPrice;
    const high = open * (1 + Math.random() * volatility);
    const low = open * (1 - Math.random() * volatility);
    const close = low + Math.random() * (high - low);
    const volume = Math.floor(Math.random() * 1000000) + 100000;

    data.push({
      symbol,
      timestamp,
      open,
      high,
      low,
      close,
      volume
    });

    timestamp += intervalMs;
    currentPrice = close;
  }

  return data;
}

/**
 * Generate order book data
 */
export interface OrderBookLevel {
  price: number;
  size: number;
}

export interface OrderBook {
  timestamp: number;
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

export function generateOrderBook(
  symbol: string,
  midPrice: number,
  depth = 10
): OrderBook {
  const spread = midPrice * 0.001; // 0.1% spread
  const bids: OrderBookLevel[] = [];
  const asks: OrderBookLevel[] = [];

  for (let i = 0; i < depth; i++) {
    // Generate bids below mid price
    bids.push({
      price: midPrice - spread / 2 - i * spread / depth,
      size: Math.floor(Math.random() * 10000) + 1000
    });

    // Generate asks above mid price
    asks.push({
      price: midPrice + spread / 2 + i * spread / depth,
      size: Math.floor(Math.random() * 10000) + 1000
    });
  }

  return {
    timestamp: Date.now(),
    symbol,
    bids,
    asks
  };
}

/**
 * Generate tick data
 */
export interface Tick {
  timestamp: number;
  symbol: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
}

export function generateTicks(
  symbol: string,
  count: number,
  midPrice: number
): Tick[] {
  const ticks: Tick[] = [];
  let timestamp = Date.now() - count * 1000;

  for (let i = 0; i < count; i++) {
    const side = Math.random() > 0.5 ? 'buy' : 'sell';
    const price = midPrice * (1 + (Math.random() - 0.5) * 0.001);
    const size = Math.floor(Math.random() * 1000) + 10;

    ticks.push({
      timestamp,
      symbol,
      price,
      size,
      side
    });

    timestamp += 1000;
  }

  return ticks;
}

/**
 * Standard test fixtures
 */
export const TEST_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];

export const SAMPLE_MARKET_DATA = generateMarketData('AAPL', 100, {
  startPrice: 150,
  volatility: 0.02,
  trend: 0.001
});

export const SAMPLE_ORDER_BOOK = generateOrderBook('AAPL', 150, 20);

export const SAMPLE_TICKS = generateTicks('AAPL', 100, 150);
