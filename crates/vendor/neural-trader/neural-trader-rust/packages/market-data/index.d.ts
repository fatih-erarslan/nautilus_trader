// Type definitions for @neural-trader/market-data
import type {
  Bar,
  Quote,
  MarketDataConfig,
  JsBar
} from '@neural-trader/core';

export { Bar, Quote, MarketDataConfig, JsBar };

export class MarketDataProvider {
  constructor(config: MarketDataConfig);
  connect(): Promise<boolean>;
  disconnect(): Promise<void>;
  fetchBars(symbol: string, start: string, end: string, timeframe: string): Promise<Bar[]>;
  getQuote(symbol: string): Promise<Quote>;
  subscribeQuotes(symbols: string[], callback: (quote: Quote) => void): any;
  getQuotesBatch(symbols: string[]): Promise<Quote[]>;
  isConnected(): Promise<boolean>;
}

export function fetchMarketData(
  symbol: string,
  start: string,
  end: string,
  timeframe: string
): Promise<any>;

export function listDataProviders(): string[];

export function encodeBarsToBuffer(bars: JsBar[]): any;
export function decodeBarsFromBuffer(buffer: Buffer): any;
