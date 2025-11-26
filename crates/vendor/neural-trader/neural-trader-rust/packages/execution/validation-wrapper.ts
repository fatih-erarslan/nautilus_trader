/**
 * Validated wrapper for NeuralTrader execution class
 * Enforces input validation before delegating to native Rust layer
 */

import { NeuralTrader as NativeNeuralTrader } from './index';
import {
  validateExecutionConfig,
  validateOrder,
  validateBatchOrders,
  ValidationError,
} from './validation';
import type { JsConfig, JsOrder, NapiResult } from '@neural-trader/core';

export class ValidatedNeuralTrader {
  private readonly native: NativeNeuralTrader;
  private isInitialized: boolean = false;

  /**
   * Initialize with validated configuration
   * @throws {ValidationError} If configuration is invalid
   */
  constructor(config: JsConfig) {
    try {
      const validatedConfig = validateExecutionConfig(config);
      this.native = new NativeNeuralTrader(validatedConfig);
      this.isInitialized = false;
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to initialize NeuralTrader: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Start the trading engine
   */
  async start(): Promise<NapiResult> {
    try {
      const result = await this.native.start();
      this.isInitialized = true;
      return result;
    } catch (error) {
      throw new ValidationError(
        `Failed to start trader: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Stop the trading engine
   */
  async stop(): Promise<NapiResult> {
    try {
      const result = await this.native.stop();
      this.isInitialized = false;
      return result;
    } catch (error) {
      throw new ValidationError(
        `Failed to stop trader: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get current positions
   */
  async getPositions(): Promise<NapiResult> {
    try {
      if (!this.isInitialized) {
        throw new ValidationError('Trader must be started before getting positions');
      }
      return await this.native.getPositions();
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to get positions: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Place a validated order
   * @throws {ValidationError} If order is invalid
   */
  async placeOrder(order: JsOrder): Promise<NapiResult> {
    try {
      if (!this.isInitialized) {
        throw new ValidationError('Trader must be started before placing orders');
      }

      const validatedOrder = validateOrder(order);
      return await this.native.placeOrder(validatedOrder);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to place order: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Place multiple validated orders
   * @throws {ValidationError} If any order is invalid
   */
  async placeBatchOrders(orders: JsOrder[]): Promise<NapiResult[]> {
    try {
      if (!this.isInitialized) {
        throw new ValidationError('Trader must be started before placing orders');
      }

      const validatedOrders = validateBatchOrders(orders);
      return await Promise.all(
        validatedOrders.map((order) => this.native.placeOrder(order))
      );
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to place batch orders: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get account balance
   */
  async getBalance(): Promise<NapiResult> {
    try {
      if (!this.isInitialized) {
        throw new ValidationError('Trader must be started before getting balance');
      }
      return await this.native.getBalance();
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to get balance: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Get account equity
   */
  async getEquity(): Promise<NapiResult> {
    try {
      if (!this.isInitialized) {
        throw new ValidationError('Trader must be started before getting equity');
      }
      return await this.native.getEquity();
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Failed to get equity: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Check if trader is initialized
   */
  isRunning(): boolean {
    return this.isInitialized;
  }
}
