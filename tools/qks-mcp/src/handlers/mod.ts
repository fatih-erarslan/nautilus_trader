/**
 * QKS MCP Handlers - Agentic AI Integration
 *
 * Export all handler modules for 8-layer cognitive architecture
 */

export * from './thermodynamic.js';
export * from './cognitive.js';
export * from './decision.js';
export * from './learning.js';
export * from './collective.js';
export * from './consciousness.js';
export * from './metacognition.js';
export * from './integration.js';
export * from './session.js';
export * from './streaming.js';

// Bridge interface (to be implemented)
export interface QKSBridge {
  callRust(method: string, params: any): Promise<any>;
  callPython(code: string): Promise<any>;
  callWolfram(code: string): Promise<any>;
}
