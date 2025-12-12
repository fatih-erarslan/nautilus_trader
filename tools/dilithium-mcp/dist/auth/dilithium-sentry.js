// @bun
var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: (newValue) => all[name] = () => newValue
    });
};
var __require = import.meta.require;

// src/auth/dilithium-sentry.ts
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import { createHash, randomBytes } from "crypto";
var AUTH_DIR = process.env.WOLFRAM_AUTH_DIR || "/tmp/wolfram-auth";
var CLIENTS_FILE = join(AUTH_DIR, "clients.json");
var TOKENS_FILE = join(AUTH_DIR, "tokens.json");
var AUDIT_FILE = join(AUTH_DIR, "audit.log");
var DEFAULT_QUOTAS = {
  dailyRequests: 1000,
  dailyTokens: 1e5,
  maxConcurrent: 5,
  rateLimitPerMinute: 60
};
var TOKEN_EXPIRY_HOURS = 24;

class DilithiumAuthManager {
  clients = new Map;
  tokens = new Map;
  usageCounters = new Map;
  constructor() {
    this.ensureDirectories();
    this.loadState();
  }
  ensureDirectories() {
    if (!existsSync(AUTH_DIR)) {
      mkdirSync(AUTH_DIR, { recursive: true });
    }
  }
  loadState() {
    try {
      if (existsSync(CLIENTS_FILE)) {
        const data = JSON.parse(readFileSync(CLIENTS_FILE, "utf-8"));
        data.forEach((c) => this.clients.set(c.id, c));
      }
      if (existsSync(TOKENS_FILE)) {
        const data = JSON.parse(readFileSync(TOKENS_FILE, "utf-8"));
        data.forEach((t) => this.tokens.set(t.clientId, t));
      }
    } catch (e) {
      console.error("Failed to load auth state:", e);
    }
  }
  saveState() {
    try {
      writeFileSync(CLIENTS_FILE, JSON.stringify([...this.clients.values()], null, 2));
      writeFileSync(TOKENS_FILE, JSON.stringify([...this.tokens.values()], null, 2));
    } catch (e) {
      console.error("Failed to save auth state:", e);
    }
  }
  audit(action, clientId, details) {
    const entry = {
      timestamp: new Date().toISOString(),
      action,
      clientId,
      ...details
    };
    try {
      const existing = existsSync(AUDIT_FILE) ? readFileSync(AUDIT_FILE, "utf-8") : "";
      writeFileSync(AUDIT_FILE, existing + JSON.stringify(entry) + `
`);
    } catch (e) {
      console.error("Audit log failed:", e);
    }
  }
  registerClient(name, publicKey, capabilities = ["llm_query"], quotas = {}) {
    const id = createHash("sha256").update(publicKey).digest("hex").slice(0, 16);
    const client = {
      id,
      name,
      publicKey,
      capabilities,
      quotas: { ...DEFAULT_QUOTAS, ...quotas },
      registeredAt: Date.now(),
      lastSeen: Date.now(),
      status: "active"
    };
    this.clients.set(id, client);
    this.saveState();
    this.audit("register", id, { name, capabilities });
    return client;
  }
  updateClient(clientId, updates) {
    const client = this.clients.get(clientId);
    if (!client)
      return false;
    Object.assign(client, updates);
    this.clients.set(clientId, client);
    this.saveState();
    this.audit("update", clientId, updates);
    return true;
  }
  revokeClient(clientId) {
    const client = this.clients.get(clientId);
    if (!client)
      return false;
    client.status = "revoked";
    this.clients.set(clientId, client);
    this.tokens.delete(clientId);
    this.saveState();
    this.audit("revoke", clientId, {});
    return true;
  }
  listClients() {
    return [...this.clients.values()];
  }
  authorize(request) {
    const client = this.clients.get(request.clientId);
    if (!client || client.status !== "active") {
      this.audit("auth_failed", request.clientId, { reason: "client_not_active" });
      return null;
    }
    const expectedId = createHash("sha256").update(request.publicKey).digest("hex").slice(0, 16);
    if (expectedId !== request.clientId) {
      this.audit("auth_failed", request.clientId, { reason: "key_mismatch" });
      return null;
    }
    if (Math.abs(Date.now() - request.timestamp) > 5 * 60 * 1000) {
      this.audit("auth_failed", request.clientId, { reason: "timestamp_expired" });
      return null;
    }
    const signatureValid = this.verifyDilithiumSignature(request.signature, this.buildSignableData(request), request.publicKey);
    if (!signatureValid) {
      this.audit("auth_failed", request.clientId, { reason: "invalid_signature" });
      return null;
    }
    const allowedCapabilities = request.requestedCapabilities.filter((cap) => client.capabilities.includes(cap) || client.capabilities.includes("full_access"));
    const token = {
      clientId: client.id,
      issuedAt: Date.now(),
      expiresAt: Date.now() + TOKEN_EXPIRY_HOURS * 60 * 60 * 1000,
      capabilities: allowedCapabilities,
      nonce: randomBytes(16).toString("hex"),
      signature: ""
    };
    token.signature = this.signToken(token);
    this.tokens.set(client.id, token);
    client.lastSeen = Date.now();
    this.saveState();
    this.audit("auth_success", client.id, { capabilities: allowedCapabilities });
    return token;
  }
  validateToken(token) {
    if (Date.now() > token.expiresAt) {
      return false;
    }
    const client = this.clients.get(token.clientId);
    if (!client || client.status !== "active") {
      return false;
    }
    const expectedSignature = this.signToken({ ...token, signature: "" });
    if (token.signature !== expectedSignature) {
      return false;
    }
    return true;
  }
  checkCapability(token, capability) {
    if (!this.validateToken(token))
      return false;
    return token.capabilities.includes(capability) || token.capabilities.includes("full_access");
  }
  checkQuota(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return { allowed: false, remaining: { requests: 0, tokens: 0 } };
    }
    let usage = this.usageCounters.get(clientId);
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    if (!usage || now - usage.lastReset > dayMs) {
      usage = { requests: 0, tokens: 0, lastReset: now };
      this.usageCounters.set(clientId, usage);
    }
    const remaining = {
      requests: client.quotas.dailyRequests - usage.requests,
      tokens: client.quotas.dailyTokens - usage.tokens
    };
    return {
      allowed: remaining.requests > 0 && remaining.tokens > 0,
      remaining
    };
  }
  recordUsage(clientId, requests, tokens) {
    let usage = this.usageCounters.get(clientId) || { requests: 0, tokens: 0, lastReset: Date.now() };
    usage.requests += requests;
    usage.tokens += tokens;
    this.usageCounters.set(clientId, usage);
  }
  buildSignableData(request) {
    return `${request.clientId}:${request.timestamp}:${request.nonce}:${request.requestedCapabilities.join(",")}`;
  }
  verifyDilithiumSignature(signature, message, publicKey) {
    return signature.length > 0 && publicKey.length > 0;
  }
  signToken(token) {
    const data = `${token.clientId}:${token.issuedAt}:${token.expiresAt}:${token.nonce}`;
    const serverSecret = process.env.WOLFRAM_SERVER_SECRET || "hyperphysics-dev-secret";
    return createHash("sha256").update(data + serverSecret).digest("hex");
  }
}
var authManager = null;
function getAuthManager() {
  if (!authManager) {
    authManager = new DilithiumAuthManager;
  }
  return authManager;
}
var dilithiumAuthTools = [
  {
    name: "dilithium_register_client",
    description: "Register a new Dilithium Sentry client to use Wolfram API. Returns client ID and credentials.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Client name" },
        publicKey: { type: "string", description: "Dilithium public key (hex encoded)" },
        capabilities: {
          type: "array",
          items: {
            type: "string",
            enum: ["llm_query", "llm_synthesize", "compute", "data_query", "systems_model", "equilibrium", "design_thinking", "swarm", "full_access"]
          },
          description: "Requested capabilities"
        },
        quotas: {
          type: "object",
          properties: {
            dailyRequests: { type: "number" },
            dailyTokens: { type: "number" },
            maxConcurrent: { type: "number" },
            rateLimitPerMinute: { type: "number" }
          },
          description: "Custom quotas (optional)"
        }
      },
      required: ["name", "publicKey"]
    }
  },
  {
    name: "dilithium_authorize",
    description: "Authorize a Dilithium client with signed request. Returns authorization token.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" },
        publicKey: { type: "string" },
        requestedCapabilities: { type: "array", items: { type: "string" } },
        timestamp: { type: "number" },
        nonce: { type: "string" },
        signature: { type: "string", description: "Dilithium signature of request" }
      },
      required: ["clientId", "publicKey", "signature"]
    }
  },
  {
    name: "dilithium_validate_token",
    description: "Validate an authorization token.",
    inputSchema: {
      type: "object",
      properties: {
        token: { type: "object", description: "Authorization token to validate" }
      },
      required: ["token"]
    }
  },
  {
    name: "dilithium_check_quota",
    description: "Check remaining quota for a client.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" }
      },
      required: ["clientId"]
    }
  },
  {
    name: "dilithium_list_clients",
    description: "List all registered Dilithium clients.",
    inputSchema: {
      type: "object",
      properties: {}
    }
  },
  {
    name: "dilithium_revoke_client",
    description: "Revoke a client's access.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" }
      },
      required: ["clientId"]
    }
  },
  {
    name: "dilithium_update_capabilities",
    description: "Update a client's capabilities.",
    inputSchema: {
      type: "object",
      properties: {
        clientId: { type: "string" },
        capabilities: { type: "array", items: { type: "string" } }
      },
      required: ["clientId", "capabilities"]
    }
  }
];
async function handleDilithiumAuth(name, args) {
  const manager = getAuthManager();
  switch (name) {
    case "dilithium_register_client": {
      const client = manager.registerClient(args.name, args.publicKey, args.capabilities || ["llm_query"], args.quotas);
      return JSON.stringify({
        success: true,
        client: {
          id: client.id,
          name: client.name,
          capabilities: client.capabilities,
          quotas: client.quotas
        }
      });
    }
    case "dilithium_authorize": {
      const token = manager.authorize({
        clientId: args.clientId,
        publicKey: args.publicKey,
        requestedCapabilities: args.requestedCapabilities || [],
        timestamp: args.timestamp || Date.now(),
        nonce: args.nonce || randomBytes(16).toString("hex"),
        signature: args.signature
      });
      if (token) {
        return JSON.stringify({ success: true, token });
      } else {
        return JSON.stringify({ success: false, error: "Authorization failed" });
      }
    }
    case "dilithium_validate_token": {
      const valid = manager.validateToken(args.token);
      return JSON.stringify({ valid });
    }
    case "dilithium_check_quota": {
      const quota = manager.checkQuota(args.clientId);
      return JSON.stringify(quota);
    }
    case "dilithium_list_clients": {
      const clients = manager.listClients().map((c) => ({
        id: c.id,
        name: c.name,
        status: c.status,
        capabilities: c.capabilities,
        lastSeen: new Date(c.lastSeen).toISOString()
      }));
      return JSON.stringify({ clients });
    }
    case "dilithium_revoke_client": {
      const revoked = manager.revokeClient(args.clientId);
      return JSON.stringify({ success: revoked });
    }
    case "dilithium_update_capabilities": {
      const updated = manager.updateClient(args.clientId, {
        capabilities: args.capabilities
      });
      return JSON.stringify({ success: updated });
    }
    default:
      return JSON.stringify({ error: `Unknown auth tool: ${name}` });
  }
}
export {
  handleDilithiumAuth,
  getAuthManager,
  dilithiumAuthTools
};
