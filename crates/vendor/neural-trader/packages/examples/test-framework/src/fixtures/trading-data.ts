/**
 * Trading data fixtures for testing
 */

import { TradingSignal } from '../types';

/**
 * Generate trading signals
 */
export function generateSignals(
  symbol: string,
  count: number,
  options: {
    buyRatio?: number;
    avgConfidence?: number;
    priceRange?: [number, number];
  } = {}
): TradingSignal[] {
  const {
    buyRatio = 0.5,
    avgConfidence = 0.7,
    priceRange = [100, 200]
  } = options;

  const signals: TradingSignal[] = [];
  let timestamp = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    const action = Math.random() < buyRatio
      ? 'buy'
      : Math.random() < 0.5 ? 'sell' : 'hold';

    const confidence = Math.min(
      1,
      Math.max(0, avgConfidence + (Math.random() - 0.5) * 0.4)
    );

    const price = priceRange[0] + Math.random() * (priceRange[1] - priceRange[0]);
    const quantity = Math.floor(Math.random() * 100) + 1;

    signals.push({
      timestamp,
      symbol,
      action,
      confidence,
      price,
      quantity
    });

    timestamp += 3600000;
  }

  return signals;
}

/**
 * Generate portfolio data
 */
export interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
}

export interface Portfolio {
  cash: number;
  positions: Position[];
  totalValue: number;
  timestamp: number;
}

export function generatePortfolio(
  symbols: string[],
  cashAmount = 100000
): Portfolio {
  const positions: Position[] = symbols.map(symbol => {
    const quantity = Math.floor(Math.random() * 100);
    const avgPrice = 100 + Math.random() * 100;
    const currentPrice = avgPrice * (1 + (Math.random() - 0.5) * 0.1);
    const unrealizedPnL = (currentPrice - avgPrice) * quantity;

    return {
      symbol,
      quantity,
      avgPrice,
      currentPrice,
      unrealizedPnL
    };
  });

  const positionsValue = positions.reduce(
    (sum, pos) => sum + pos.currentPrice * pos.quantity,
    0
  );

  return {
    cash: cashAmount,
    positions,
    totalValue: cashAmount + positionsValue,
    timestamp: Date.now()
  };
}

/**
 * Generate trade history
 */
export interface Trade {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  commission: number;
  pnl?: number;
}

export function generateTradeHistory(
  count: number,
  symbols: string[]
): Trade[] {
  const trades: Trade[] = [];
  let timestamp = Date.now() - count * 3600000;

  for (let i = 0; i < count; i++) {
    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    const side = Math.random() > 0.5 ? 'buy' : 'sell';
    const quantity = Math.floor(Math.random() * 100) + 1;
    const price = 100 + Math.random() * 100;
    const commission = price * quantity * 0.001; // 0.1% commission

    trades.push({
      id: `trade-${i}`,
      timestamp,
      symbol,
      side,
      quantity,
      price,
      commission,
      pnl: side === 'sell' ? (Math.random() - 0.5) * 1000 : undefined
    });

    timestamp += 3600000;
  }

  return trades;
}

/**
 * Standard test fixtures
 */
export const SAMPLE_SIGNALS = generateSignals('AAPL', 50, {
  buyRatio: 0.6,
  avgConfidence: 0.75
});

export const SAMPLE_PORTFOLIO = generatePortfolio(
  ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
  100000
);

export const SAMPLE_TRADES = generateTradeHistory(
  100,
  ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
);
