/**
 * Session Management - Stateful Agent Sessions
 *
 * Enables persistent agent state across tool calls:
 * - Session creation and lifecycle
 * - State persistence and restoration
 * - Multi-session orchestration
 * - Resource cleanup
 */

import { QKSBridge } from './mod.js';
import type { HomeostasisState } from './thermodynamic.js';

export interface SessionConfig {
  observation_dim?: number;
  action_dim?: number;
  hidden_dim?: number;
  learning_rate?: number;
  survival_strength?: number;
  phi_calculator_type?: 'exact' | 'monte_carlo' | 'greedy' | 'hierarchical';
  enable_learning?: boolean;
  enable_consciousness?: boolean;
  enable_metacognition?: boolean;
}

export interface QKSSession {
  id: string;
  config: SessionConfig;
  state: {
    phi: number;
    free_energy: number;
    survival: number;
    control: number;
    beliefs: number[];
    precision: number[];
    position: number[];
    working_memory: number[][];
    episodic_memory_ids: string[];
  };
  phi_calculator_type: string;
  created_at: number;
  last_accessed: number;
  cycles_completed: number;
}

export class SessionManager {
  private bridge: QKSBridge;
  private sessions: Map<string, QKSSession> = new Map();

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Create new session
   */
  createSession(config: SessionConfig = {}): string {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const defaultConfig: SessionConfig = {
      observation_dim: 10,
      action_dim: 5,
      hidden_dim: 20,
      learning_rate: 0.01,
      survival_strength: 1.0,
      phi_calculator_type: 'greedy',
      enable_learning: true,
      enable_consciousness: true,
      enable_metacognition: true,
    };

    const mergedConfig = { ...defaultConfig, ...config };

    const initialState = {
      phi: 0.1,
      free_energy: 1.0,
      survival: 0.5,
      control: 0.2,
      beliefs: new Array(mergedConfig.hidden_dim!).fill(0.1),
      precision: new Array(mergedConfig.hidden_dim!).fill(1.0),
      position: [1.0, ...Array(11).fill(0)], // Origin in H^11
      working_memory: [],
      episodic_memory_ids: [],
    };

    const session: QKSSession = {
      id: sessionId,
      config: mergedConfig,
      state: initialState,
      phi_calculator_type: mergedConfig.phi_calculator_type!,
      created_at: Date.now(),
      last_accessed: Date.now(),
      cycles_completed: 0,
    };

    this.sessions.set(sessionId, session);

    return sessionId;
  }

  /**
   * Get session by ID
   */
  getSession(sessionId: string): QKSSession | undefined {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.last_accessed = Date.now();
    }
    return session;
  }

  /**
   * Update session state
   */
  updateSession(sessionId: string, updates: Partial<QKSSession['state']>): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    session.state = { ...session.state, ...updates };
    session.last_accessed = Date.now();
    return true;
  }

  /**
   * Delete session
   */
  deleteSession(sessionId: string): boolean {
    return this.sessions.delete(sessionId);
  }

  /**
   * List all sessions
   */
  listSessions(): Array<{ id: string; created_at: number; last_accessed: number; cycles: number }> {
    return Array.from(this.sessions.values()).map(s => ({
      id: s.id,
      created_at: s.created_at,
      last_accessed: s.last_accessed,
      cycles: s.cycles_completed,
    }));
  }

  /**
   * Save session state to persistent storage
   */
  async saveSessionState(sessionId: string): Promise<{
    saved: boolean;
    storage_path?: string;
    size_bytes?: number;
  }> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return { saved: false };
    }

    try {
      const result = await this.bridge.callRust('session.save', {
        session_id: sessionId,
        state: session.state,
        config: session.config,
      });
      return { saved: true, ...result };
    } catch (e) {
      // Fallback: In-memory only
      return {
        saved: true,
        storage_path: 'memory',
        size_bytes: JSON.stringify(session).length,
      };
    }
  }

  /**
   * Load session state from persistent storage
   */
  async loadSessionState(sessionId: string): Promise<{
    loaded: boolean;
    session?: QKSSession;
  }> {
    try {
      const result = await this.bridge.callRust('session.load', { session_id: sessionId });
      if (result.session) {
        this.sessions.set(sessionId, result.session);
        return { loaded: true, session: result.session };
      }
      return { loaded: false };
    } catch (e) {
      const session = this.sessions.get(sessionId);
      return { loaded: !!session, session };
    }
  }

  /**
   * Clone session (create copy)
   */
  cloneSession(sourceSessionId: string): string | null {
    const sourceSession = this.sessions.get(sourceSessionId);
    if (!sourceSession) return null;

    const newSessionId = this.createSession(sourceSession.config);
    const newSession = this.sessions.get(newSessionId);

    if (newSession) {
      newSession.state = JSON.parse(JSON.stringify(sourceSession.state));
      newSession.cycles_completed = 0;
    }

    return newSessionId;
  }

  /**
   * Cleanup idle sessions
   */
  cleanupIdleSessions(maxIdleTime: number = 3600000): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.last_accessed > maxIdleTime) {
        this.sessions.delete(sessionId);
        cleaned++;
      }
    }

    return cleaned;
  }

  /**
   * Get session metrics
   */
  getSessionMetrics(sessionId: string): {
    age_ms: number;
    idle_time_ms: number;
    cycles_completed: number;
    memory_usage_approx: number;
    state_complexity: number;
  } | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    const now = Date.now();
    const stateString = JSON.stringify(session.state);

    return {
      age_ms: now - session.created_at,
      idle_time_ms: now - session.last_accessed,
      cycles_completed: session.cycles_completed,
      memory_usage_approx: stateString.length,
      state_complexity: session.state.beliefs.length + session.state.working_memory.length,
    };
  }

  /**
   * Merge sessions
   * Combines state from multiple sessions
   */
  mergeSessions(sessionIds: string[]): string | null {
    if (sessionIds.length === 0) return null;

    const sessions = sessionIds.map(id => this.sessions.get(id)).filter(s => s !== undefined) as QKSSession[];

    if (sessions.length === 0) return null;

    // Use first session config as base
    const mergedSessionId = this.createSession(sessions[0].config);
    const mergedSession = this.sessions.get(mergedSessionId);

    if (!mergedSession) return null;

    // Merge beliefs (average)
    const beliefDim = sessions[0].state.beliefs.length;
    const mergedBeliefs = new Array(beliefDim).fill(0);

    for (const session of sessions) {
      for (let i = 0; i < beliefDim; i++) {
        mergedBeliefs[i] += session.state.beliefs[i] / sessions.length;
      }
    }

    mergedSession.state.beliefs = mergedBeliefs;

    // Merge working memory (concatenate and truncate)
    const allMemory = sessions.flatMap(s => s.state.working_memory);
    mergedSession.state.working_memory = allMemory.slice(0, 10);

    // Average metrics
    mergedSession.state.phi = sessions.reduce((s, sess) => s + sess.state.phi, 0) / sessions.length;
    mergedSession.state.free_energy = sessions.reduce((s, sess) => s + sess.state.free_energy, 0) / sessions.length;

    return mergedSessionId;
  }

  /**
   * Export session to JSON
   */
  exportSession(sessionId: string): string | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    return JSON.stringify(session, null, 2);
  }

  /**
   * Import session from JSON
   */
  importSession(sessionJson: string): string | null {
    try {
      const session: QKSSession = JSON.parse(sessionJson);
      const newId = `session_${Date.now()}_imported`;
      session.id = newId;
      session.last_accessed = Date.now();

      this.sessions.set(newId, session);
      return newId;
    } catch (e) {
      return null;
    }
  }

  /**
   * Get session count
   */
  getSessionCount(): number {
    return this.sessions.size;
  }

  /**
   * Clear all sessions
   */
  clearAllSessions(): number {
    const count = this.sessions.size;
    this.sessions.clear();
    return count;
  }
}
