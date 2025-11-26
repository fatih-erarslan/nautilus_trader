/**
 * TypeScript definitions for Rust NAPI Bridge
 */

import { EventEmitter } from 'events';

export interface RustBridgeOptions {
  /**
   * Enable stub mode (for testing without NAPI module)
   */
  stubMode?: boolean;

  /**
   * Custom path to NAPI module
   */
  modulePath?: string;
}

export interface RustBridgeStatus {
  /**
   * Bridge is ready to accept calls
   */
  ready: boolean;

  /**
   * Running in stub mode
   */
  stubMode: boolean;

  /**
   * NAPI module successfully loaded
   */
  napiLoaded: boolean;

  /**
   * Error message if module failed to load
   */
  loadError: string | null;

  /**
   * Current platform
   */
  platform: string;

  /**
   * Current architecture
   */
  arch: string;
}

/**
 * Rust NAPI Bridge
 * Provides interface to Rust-compiled trading tools via NAPI
 */
export declare class RustBridge extends EventEmitter {
  constructor(options?: RustBridgeOptions);

  /**
   * Load and initialize the Rust NAPI module
   */
  start(): Promise<void>;

  /**
   * Stop the bridge and cleanup resources
   */
  stop(): Promise<void>;

  /**
   * Call a Rust tool via NAPI
   * @param method - Tool name/method to invoke
   * @param params - Parameters for the tool
   * @returns Promise resolving to tool result
   */
  call(method: string, params?: Record<string, any>): Promise<any>;

  /**
   * Check if bridge is ready
   */
  isReady(): boolean;

  /**
   * Check if NAPI module is loaded (not in stub mode)
   */
  isNapiLoaded(): boolean;

  /**
   * Get bridge status information
   */
  getStatus(): RustBridgeStatus;
}
