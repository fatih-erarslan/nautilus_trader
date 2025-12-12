/**
 * Wolfram Module - Hybrid Local + Cloud Execution
 *
 * Exports:
 * - WolframClient: Main client class with local/cloud/hybrid modes
 * - executeWolfram: Convenience function for one-off executions
 * - getWolframClient: Singleton accessor
 */

export {
  WolframClient,
  getWolframClient,
  executeWolfram,
  type WolframMode,
  type WolframResult
} from "./client.js";
