// Type definitions for @neural-trader/brokers
import type {
  BrokerConfig,
  OrderRequest,
  OrderResponse,
  AccountBalance,
  JsPosition
} from '@neural-trader/core';

export { BrokerConfig, OrderRequest, OrderResponse, AccountBalance, JsPosition };

export class BrokerClient {
  constructor(config: BrokerConfig);
  connect(): Promise<boolean>;
  disconnect(): Promise<void>;
  placeOrder(order: OrderRequest): Promise<OrderResponse>;
  cancelOrder(orderId: string): Promise<boolean>;
  getOrderStatus(orderId: string): Promise<OrderResponse>;
  getAccountBalance(): Promise<AccountBalance>;
  listOrders(): Promise<OrderResponse[]>;
  getPositions(): Promise<JsPosition[]>;
}

export function listBrokerTypes(): string[];
export function validateBrokerConfig(config: BrokerConfig): boolean;
