// Type definitions for @neural-trader/execution
import type { JsConfig, JsOrder, NapiResult } from '@neural-trader/core';

export { JsConfig, JsOrder, NapiResult };

export class NeuralTrader {
  constructor(config: JsConfig);
  start(): Promise<NapiResult>;
  stop(): Promise<NapiResult>;
  getPositions(): Promise<NapiResult>;
  placeOrder(order: JsOrder): Promise<NapiResult>;
  getBalance(): Promise<NapiResult>;
  getEquity(): Promise<NapiResult>;
}
