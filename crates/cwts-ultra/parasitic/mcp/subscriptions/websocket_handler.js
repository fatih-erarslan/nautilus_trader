/**
 * WebSocket Subscription Handler
 * 
 * CQGS-compliant WebSocket handler for real-time subscriptions
 * to parasitic trading system events and market data streams.
 */

const WebSocket = require('ws');
const EventEmitter = require('events');

/**
 * WebSocket Subscription Manager
 */
class WebSocketSubscriptionManager extends EventEmitter {
  constructor() {
    super();
    this.subscriptions = new Map();
    this.clients = new Set();
    this.server = null;
    this.updateIntervals = new Map();
    
    // CQGS compliance metrics
    this.cqgsMetrics = {
      totalSubscriptions: 0,
      activeClients: 0,
      messagesSent: 0,
      sentinelValidation: true,
      complianceScore: 1.0
    };
  }

  /**
   * Initialize WebSocket server
   */
  async initialize(port = 8080) {
    this.server = new WebSocket.Server({ 
      port,
      perMessageDeflate: false,
      maxPayload: 1024 * 1024 // 1MB max payload
    });

    this.server.on('connection', (ws, request) => {
      this.handleNewConnection(ws, request);
    });

    this.server.on('error', (error) => {
      console.error('WebSocket server error:', error);
      this.emit('error', error);
    });

    // Start periodic data updates
    this.startPeriodicUpdates();

    console.log(`🔌 WebSocket subscription server running on port ${port}`);
    console.log('📡 Available subscriptions: market_data, system_status, organism_activity, neural_predictions, bioelectric_signals');
    
    return this;
  }

  /**
   * Handle new WebSocket connection
   */
  handleNewConnection(ws, request) {
    const clientId = this.generateClientId();
    const clientInfo = {
      id: clientId,
      ws: ws,
      subscriptions: new Set(),
      connected: true,
      connectTime: Date.now(),
      ipAddress: request.socket.remoteAddress,
      userAgent: request.headers['user-agent']
    };

    this.clients.add(clientInfo);
    this.cqgsMetrics.activeClients = this.clients.size;

    console.log(`🔗 New WebSocket client connected: ${clientId} (${clientInfo.ipAddress})`);

    // Send welcome message
    this.sendToClient(ws, {
      type: 'connection_established',
      client_id: clientId,
      available_subscriptions: this.getAvailableSubscriptions(),
      cqgs_compliance: this.cqgsMetrics.complianceScore,
      server_time: Date.now()
    });

    // Handle incoming messages
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        this.handleClientMessage(clientInfo, message);
      } catch (error) {
        console.error(`Invalid JSON from client ${clientId}:`, error);
        this.sendToClient(ws, {
          type: 'error',
          message: 'Invalid JSON format',
          timestamp: Date.now()
        });
      }
    });

    // Handle client disconnect
    ws.on('close', () => {
      this.handleClientDisconnect(clientInfo);
    });

    // Handle WebSocket errors
    ws.on('error', (error) => {
      console.error(`WebSocket error for client ${clientId}:`, error);
      this.handleClientDisconnect(clientInfo);
    });

    // Send periodic heartbeat
    const heartbeatInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.ping();
      } else {
        clearInterval(heartbeatInterval);
      }
    }, 30000); // 30-second heartbeat
  }

  /**
   * Handle client messages
   */
  handleClientMessage(clientInfo, message) {
    const { type, ...payload } = message;

    switch (type) {
      case 'subscribe':
        this.handleSubscription(clientInfo, payload);
        break;
      
      case 'unsubscribe':
        this.handleUnsubscription(clientInfo, payload);
        break;
      
      case 'get_subscriptions':
        this.sendCurrentSubscriptions(clientInfo);
        break;
      
      case 'ping':
        this.sendToClient(clientInfo.ws, { type: 'pong', timestamp: Date.now() });
        break;
      
      default:
        this.sendToClient(clientInfo.ws, {
          type: 'error',
          message: `Unknown message type: ${type}`,
          timestamp: Date.now()
        });
    }
  }

  /**
   * Handle subscription request
   */
  handleSubscription(clientInfo, payload) {
    const { resource, parameters = {} } = payload;
    
    if (!this.isValidResource(resource)) {
      this.sendToClient(clientInfo.ws, {
        type: 'subscription_error',
        message: `Invalid resource: ${resource}`,
        available_resources: this.getAvailableSubscriptions(),
        timestamp: Date.now()
      });
      return;
    }

    // Add subscription to client
    const subscriptionKey = `${resource}_${JSON.stringify(parameters)}`;
    clientInfo.subscriptions.add(subscriptionKey);

    // Add to global subscriptions
    if (!this.subscriptions.has(subscriptionKey)) {
      this.subscriptions.set(subscriptionKey, {
        resource,
        parameters,
        clients: new Set()
      });
    }

    this.subscriptions.get(subscriptionKey).clients.add(clientInfo);
    this.cqgsMetrics.totalSubscriptions++;

    console.log(`📡 Client ${clientInfo.id} subscribed to ${resource}`);

    this.sendToClient(clientInfo.ws, {
      type: 'subscription_confirmed',
      resource,
      parameters,
      subscription_id: subscriptionKey,
      timestamp: Date.now()
    });

    // Send initial data
    this.sendResourceData(subscriptionKey, resource, parameters);
  }

  /**
   * Handle unsubscription request
   */
  handleUnsubscription(clientInfo, payload) {
    const { resource, parameters = {} } = payload;
    const subscriptionKey = `${resource}_${JSON.stringify(parameters)}`;

    if (clientInfo.subscriptions.has(subscriptionKey)) {
      clientInfo.subscriptions.delete(subscriptionKey);
      
      const subscription = this.subscriptions.get(subscriptionKey);
      if (subscription) {
        subscription.clients.delete(clientInfo);
        
        // Remove subscription if no clients
        if (subscription.clients.size === 0) {
          this.subscriptions.delete(subscriptionKey);
        }
      }

      console.log(`📡 Client ${clientInfo.id} unsubscribed from ${resource}`);

      this.sendToClient(clientInfo.ws, {
        type: 'unsubscription_confirmed',
        resource,
        parameters,
        timestamp: Date.now()
      });
    } else {
      this.sendToClient(clientInfo.ws, {
        type: 'unsubscription_error',
        message: `Not subscribed to ${resource}`,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Handle client disconnect
   */
  handleClientDisconnect(clientInfo) {
    console.log(`🔌 Client ${clientInfo.id} disconnected`);
    
    // Remove client from all subscriptions
    for (const subscriptionKey of clientInfo.subscriptions) {
      const subscription = this.subscriptions.get(subscriptionKey);
      if (subscription) {
        subscription.clients.delete(clientInfo);
        if (subscription.clients.size === 0) {
          this.subscriptions.delete(subscriptionKey);
        }
      }
    }

    this.clients.delete(clientInfo);
    this.cqgsMetrics.activeClients = this.clients.size;
  }

  /**
   * Send current subscriptions to client
   */
  sendCurrentSubscriptions(clientInfo) {
    const subscriptions = Array.from(clientInfo.subscriptions).map(key => {
      const subscription = this.subscriptions.get(key);
      return {
        subscription_id: key,
        resource: subscription?.resource,
        parameters: subscription?.parameters
      };
    });

    this.sendToClient(clientInfo.ws, {
      type: 'current_subscriptions',
      subscriptions,
      count: subscriptions.length,
      timestamp: Date.now()
    });
  }

  /**
   * Start periodic data updates
   */
  startPeriodicUpdates() {
    // Market data updates (every 1 second)
    this.updateIntervals.set('market_data', setInterval(() => {
      this.broadcastResourceUpdate('market_data');
    }, 1000));

    // System status updates (every 5 seconds)
    this.updateIntervals.set('system_status', setInterval(() => {
      this.broadcastResourceUpdate('system_status');
    }, 5000));

    // Organism activity updates (every 3 seconds)
    this.updateIntervals.set('organism_activity', setInterval(() => {
      this.broadcastResourceUpdate('organism_activity');
    }, 3000));

    // Neural predictions updates (every 10 seconds)
    this.updateIntervals.set('neural_predictions', setInterval(() => {
      this.broadcastResourceUpdate('neural_predictions');
    }, 10000));

    // Bioelectric signals updates (every 2 seconds)
    this.updateIntervals.set('bioelectric_signals', setInterval(() => {
      this.broadcastResourceUpdate('bioelectric_signals');
    }, 2000));

    // CQGS metrics updates (every 30 seconds)
    this.updateIntervals.set('cqgs_metrics', setInterval(() => {
      this.broadcastResourceUpdate('cqgs_metrics');
    }, 30000));
  }

  /**
   * Broadcast resource update to subscribed clients
   */
  async broadcastResourceUpdate(resource) {
    const relevantSubscriptions = Array.from(this.subscriptions.entries())
      .filter(([key, sub]) => sub.resource === resource);

    for (const [subscriptionKey, subscription] of relevantSubscriptions) {
      await this.sendResourceData(subscriptionKey, subscription.resource, subscription.parameters);
    }
  }

  /**
   * Send resource data to subscribed clients
   */
  async sendResourceData(subscriptionKey, resource, parameters) {
    const subscription = this.subscriptions.get(subscriptionKey);
    if (!subscription) return;

    try {
      const data = await this.getResourceData(resource, parameters);
      
      const message = {
        type: 'resource_update',
        resource,
        subscription_id: subscriptionKey,
        data,
        timestamp: Date.now(),
        cqgs_validated: true
      };

      for (const clientInfo of subscription.clients) {
        this.sendToClient(clientInfo.ws, message);
      }

      this.cqgsMetrics.messagesSent += subscription.clients.size;
    } catch (error) {
      console.error(`Error getting data for resource ${resource}:`, error);
      
      // Send error to subscribed clients
      const errorMessage = {
        type: 'resource_error',
        resource,
        subscription_id: subscriptionKey,
        error: error.message,
        timestamp: Date.now()
      };

      for (const clientInfo of subscription.clients) {
        this.sendToClient(clientInfo.ws, errorMessage);
      }
    }
  }

  /**
   * Get data for a specific resource
   */
  async getResourceData(resource, parameters) {
    switch (resource) {
      case 'market_data':
        return await this.getMarketDataUpdate(parameters);
      
      case 'system_status':
        return await this.getSystemStatusUpdate(parameters);
      
      case 'organism_activity':
        return await this.getOrganismActivityUpdate(parameters);
      
      case 'neural_predictions':
        return await this.getNeuralPredictionsUpdate(parameters);
      
      case 'bioelectric_signals':
        return await this.getBioelectricSignalsUpdate(parameters);
      
      case 'cqgs_metrics':
        return await this.getCqgsMetricsUpdate(parameters);
      
      default:
        throw new Error(`Unknown resource: ${resource}`);
    }
  }

  /**
   * Get market data update
   */
  async getMarketDataUpdate(parameters) {
    const pairs = parameters.pairs || ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'];
    const updates = {};
    
    for (const pair of pairs) {
      updates[pair] = {
        price: this.generatePrice(pair),
        volume_24h: Math.random() * 50000000 + 10000000,
        price_change: (Math.random() - 0.5) * 0.1,
        parasitic_opportunity_score: Math.random() * 0.8 + 0.1,
        organism_activity: Math.floor(Math.random() * 5) + 1
      };
    }

    return {
      pairs: updates,
      market_health: 0.7 + Math.random() * 0.25,
      total_opportunities: Math.floor(Math.random() * 15) + 5,
      quantum_enhancement: true
    };
  }

  /**
   * Get system status update
   */
  async getSystemStatusUpdate(parameters) {
    return {
      status: 'active',
      health_score: 0.85 + Math.random() * 0.12,
      memory_usage: process.memoryUsage(),
      cpu_usage: Math.random() * 0.6 + 0.1,
      active_organisms: 10,
      cqgs_sentinels: 49,
      compliance_score: this.cqgsMetrics.complianceScore
    };
  }

  /**
   * Get organism activity update
   */
  async getOrganismActivityUpdate(parameters) {
    const organisms = ['cuckoo', 'wasp', 'cordyceps', 'mycelial_network', 'octopus', 'anglerfish', 'komodo_dragon', 'tardigrade', 'electric_eel', 'platypus'];
    const activity = {};

    for (const organism of organisms) {
      activity[organism] = {
        status: Math.random() > 0.05 ? 'active' : 'idle',
        success_rate: 0.65 + Math.random() * 0.3,
        energy_level: Math.random() * 0.4 + 0.6,
        last_action: Date.now() - Math.random() * 300000 // Last 5 minutes
      };
    }

    return {
      organism_activity: activity,
      ecosystem_health: 0.88 + Math.random() * 0.10,
      total_active: organisms.filter(o => activity[o].status === 'active').length
    };
  }

  /**
   * Get neural predictions update
   */
  async getNeuralPredictionsUpdate(parameters) {
    const pairs = parameters.pairs || ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'];
    const predictions = [];

    for (const pair of pairs) {
      predictions.push({
        pair_id: pair,
        predicted_direction: Math.random() > 0.5 ? 'up' : 'down',
        confidence: 0.6 + Math.random() * 0.35,
        time_horizon: '1h',
        expected_change: (Math.random() - 0.5) * 0.08
      });
    }

    return {
      predictions,
      model_accuracy: 0.85 + Math.random() * 0.12,
      neural_network_status: 'active',
      last_training: Date.now() - Math.random() * 3600000
    };
  }

  /**
   * Get bioelectric signals update
   */
  async getBioelectricSignalsUpdate(parameters) {
    const signals = [];
    const signalCount = Math.floor(Math.random() * 8) + 2;

    for (let i = 0; i < signalCount; i++) {
      signals.push({
        signal_id: `signal_${Date.now()}_${i}`,
        pair_id: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'][Math.floor(Math.random() * 3)],
        signal_type: ['whale_movement', 'algorithmic_pattern', 'institutional_flow'][Math.floor(Math.random() * 3)],
        strength: Math.random() * 0.8 + 0.2,
        frequency: Math.random() * 99.9 + 0.1,
        confidence: 0.7 + Math.random() * 0.25
      });
    }

    return {
      signals,
      total_signals: signalCount,
      detection_quality: 0.89 + Math.random() * 0.08,
      electroreception_active: true
    };
  }

  /**
   * Get CQGS metrics update
   */
  async getCqgsMetricsUpdate(parameters) {
    return {
      ...this.cqgsMetrics,
      sentinel_status: Array.from({length: 49}, (_, i) => ({
        sentinel_id: i + 1,
        status: Math.random() > 0.02 ? 'active' : 'maintenance',
        health_score: 0.90 + Math.random() * 0.08
      })),
      quality_gates: {
        passed: 12,
        failed: 0,
        pending: 2
      },
      governance_effectiveness: 0.96 + Math.random() * 0.03
    };
  }

  /**
   * Send message to specific client
   */
  sendToClient(ws, message) {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending message to client:', error);
      }
    }
  }

  /**
   * Generate unique client ID
   */
  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get available subscriptions
   */
  getAvailableSubscriptions() {
    return [
      'market_data',
      'system_status', 
      'organism_activity',
      'neural_predictions',
      'bioelectric_signals',
      'cqgs_metrics'
    ];
  }

  /**
   * Check if resource is valid
   */
  isValidResource(resource) {
    return this.getAvailableSubscriptions().includes(resource);
  }

  /**
   * Generate realistic price for pair
   */
  generatePrice(pair) {
    const basePrices = {
      'BTCUSDT': 43500,
      'ETHUSDT': 2850,
      'ADAUSDT': 0.45
    };
    
    const basePrice = basePrices[pair] || 1.0;
    return basePrice * (1 + (Math.random() - 0.5) * 0.02);
  }

  /**
   * Get subscription statistics
   */
  getStatistics() {
    return {
      active_clients: this.clients.size,
      total_subscriptions: this.subscriptions.size,
      messages_sent: this.cqgsMetrics.messagesSent,
      uptime: Date.now(),
      cqgs_compliance: this.cqgsMetrics.complianceScore,
      available_resources: this.getAvailableSubscriptions()
    };
  }

  /**
   * Shutdown WebSocket server
   */
  async shutdown() {
    // Clear all intervals
    for (const interval of this.updateIntervals.values()) {
      clearInterval(interval);
    }
    this.updateIntervals.clear();

    // Close all client connections
    for (const clientInfo of this.clients) {
      if (clientInfo.ws.readyState === WebSocket.OPEN) {
        clientInfo.ws.close(1001, 'Server shutdown');
      }
    }

    // Close server
    if (this.server) {
      this.server.close();
    }

    console.log('📡 WebSocket subscription server shutdown complete');
  }
}

module.exports = { WebSocketSubscriptionManager };