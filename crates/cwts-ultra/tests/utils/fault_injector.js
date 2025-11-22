/**
 * Fault Injection Utility for Chaos Engineering
 * Provides comprehensive fault injection capabilities for testing system resilience
 */

class FaultInjector {
  constructor() {
    this.activeFaults = new Map();
    this.faultHistory = [];
    this.injectionCallbacks = new Map();
  }

  /**
   * Inject database-related faults
   */
  async injectDatabaseFault(config) {
    const faultId = this.generateFaultId('db');
    const fault = {
      id: faultId,
      type: 'database',
      config,
      startTime: Date.now(),
      status: 'active'
    };

    this.activeFaults.set(faultId, fault);

    switch (config.type) {
      case 'CONNECTION_LOSS':
        await this.simulateConnectionLoss(config);
        break;
      case 'SLOW_QUERIES':
        await this.simulateSlowQueries(config);
        break;
      case 'DATA_CORRUPTION':
        await this.simulateDataCorruption(config);
        break;
      case 'TRANSACTION_FAILURES':
        await this.simulateTransactionFailures(config);
        break;
      default:
        throw new Error(`Unknown database fault type: ${config.type}`);
    }

    // Auto-recovery if specified
    if (config.duration) {
      setTimeout(() => {
        this.clearFault(faultId);
      }, config.duration);
    }

    return faultId;
  }

  /**
   * Inject network-related faults
   */
  async injectNetworkFault(config) {
    const faultId = this.generateFaultId('net');
    const fault = {
      id: faultId,
      type: 'network',
      config,
      startTime: Date.now(),
      status: 'active'
    };

    this.activeFaults.set(faultId, fault);

    switch (config.type) {
      case 'INTERMITTENT_LOSS':
        await this.simulateIntermittentLoss(config);
        break;
      case 'HIGH_LATENCY':
        await this.simulateHighLatency(config);
        break;
      case 'COMPLETE_PARTITION':
        await this.simulateNetworkPartition(config);
        break;
      case 'PACKET_CORRUPTION':
        await this.simulatePacketCorruption(config);
        break;
      default:
        throw new Error(`Unknown network fault type: ${config.type}`);
    }

    if (config.duration) {
      setTimeout(() => {
        this.clearFault(faultId);
      }, config.duration);
    }

    return faultId;
  }

  /**
   * Inject resource exhaustion faults
   */
  async injectResourceFault(config) {
    const faultId = this.generateFaultId('res');
    const fault = {
      id: faultId,
      type: 'resource',
      config,
      startTime: Date.now(),
      status: 'active'
    };

    this.activeFaults.set(faultId, fault);

    switch (config.type) {
      case 'MEMORY_EXHAUSTION':
        await this.simulateMemoryExhaustion(config);
        break;
      case 'CPU_EXHAUSTION':
        await this.simulateCpuExhaustion(config);
        break;
      case 'DISK_FULL':
        await this.simulateDiskFull(config);
        break;
      case 'FILE_DESCRIPTOR_EXHAUSTION':
        await this.simulateFdExhaustion(config);
        break;
      default:
        throw new Error(`Unknown resource fault type: ${config.type}`);
    }

    if (config.duration) {
      setTimeout(() => {
        this.clearFault(faultId);
      }, config.duration);
    }

    return faultId;
  }

  /**
   * Inject Byzantine failure behaviors
   */
  async injectByzantineFault(config) {
    const faultId = this.generateFaultId('byz');
    const fault = {
      id: faultId,
      type: 'byzantine',
      config,
      startTime: Date.now(),
      status: 'active'
    };

    this.activeFaults.set(faultId, fault);

    switch (config.behavior) {
      case 'MALICIOUS_RESPONSES':
        await this.simulateMaliciousResponses(config);
        break;
      case 'INCONSISTENT_STATE':
        await this.simulateInconsistentState(config);
        break;
      case 'SELECTIVE_FAILURES':
        await this.simulateSelectiveFailures(config);
        break;
      case 'TIMING_ATTACKS':
        await this.simulateTimingAttacks(config);
        break;
      default:
        throw new Error(`Unknown Byzantine behavior: ${config.behavior}`);
    }

    return faultId;
  }

  // Database fault simulations
  async simulateConnectionLoss(config) {
    const originalConnect = require('../../quantum_trading/core/database').connect;
    
    require('../../quantum_trading/core/database').connect = function() {
      if (Math.random() < (config.probability || 1.0)) {
        throw new Error('Database connection lost');
      }
      return originalConnect.apply(this, arguments);
    };

    this.injectionCallbacks.set('db_connect', () => {
      require('../../quantum_trading/core/database').connect = originalConnect;
    });
  }

  async simulateSlowQueries(config) {
    const originalQuery = require('../../quantum_trading/core/database').query;
    
    require('../../quantum_trading/core/database').query = async function() {
      if (Math.random() < (config.probability || 0.5)) {
        await new Promise(resolve => 
          setTimeout(resolve, config.delay || 2000 + Math.random() * 3000)
        );
      }
      return originalQuery.apply(this, arguments);
    };

    this.injectionCallbacks.set('db_query', () => {
      require('../../quantum_trading/core/database').query = originalQuery;
    });
  }

  async simulateDataCorruption(config) {
    const originalQuery = require('../../quantum_trading/core/database').query;
    
    require('../../quantum_trading/core/database').query = async function(sql, params) {
      const result = await originalQuery.call(this, sql, params);
      
      if (config.scope === 'partial' && Math.random() < 0.1) {
        // Randomly corrupt some data
        if (result && Array.isArray(result)) {
          result.forEach(row => {
            if (Math.random() < 0.05) {
              Object.keys(row).forEach(key => {
                if (typeof row[key] === 'number') {
                  row[key] = row[key] * (0.9 + Math.random() * 0.2); // ±10% corruption
                }
              });
            }
          });
        }
      }
      
      return result;
    };

    this.injectionCallbacks.set('db_corrupt', () => {
      require('../../quantum_trading/core/database').query = originalQuery;
    });
  }

  // Network fault simulations
  async simulateIntermittentLoss(config) {
    const originalRequest = require('http').request;
    
    require('http').request = function() {
      if (Math.random() < (config.probability || 0.3)) {
        const mockResponse = {
          on: (event, callback) => {
            if (event === 'error') {
              setTimeout(() => callback(new Error('Network timeout')), 100);
            }
          },
          end: () => {}
        };
        return mockResponse;
      }
      return originalRequest.apply(this, arguments);
    };

    this.injectionCallbacks.set('http_request', () => {
      require('http').request = originalRequest;
    });
  }

  async simulateHighLatency(config) {
    const originalRequest = require('http').request;
    
    require('http').request = function() {
      const req = originalRequest.apply(this, arguments);
      const originalEnd = req.end;
      
      req.end = function() {
        const delay = config.delay + (Math.random() - 0.5) * (config.jitter || 0);
        setTimeout(() => {
          originalEnd.apply(this, arguments);
        }, Math.max(0, delay));
      };
      
      return req;
    };

    this.injectionCallbacks.set('http_latency', () => {
      require('http').request = originalRequest;
    });
  }

  async simulateNetworkPartition(config) {
    const partitionedComponents = new Set(config.components || []);
    
    // Mock network calls between partitioned components
    global.NETWORK_PARTITION_ACTIVE = true;
    global.PARTITIONED_COMPONENTS = partitionedComponents;

    this.injectionCallbacks.set('network_partition', () => {
      global.NETWORK_PARTITION_ACTIVE = false;
      global.PARTITIONED_COMPONENTS = null;
    });
  }

  // Resource fault simulations
  async simulateMemoryExhaustion(config) {
    const memoryBallast = [];
    const targetUsage = config.targetUsage || 0.9;
    const currentUsage = process.memoryUsage().heapUsed / process.memoryUsage().heapTotal;
    
    if (currentUsage < targetUsage) {
      const additionalMemory = Math.floor(
        (targetUsage - currentUsage) * process.memoryUsage().heapTotal / 1024
      );
      
      // Gradually consume memory
      const chunkSize = 1024 * 1024; // 1MB chunks
      const chunks = Math.floor(additionalMemory / chunkSize);
      
      for (let i = 0; i < chunks; i++) {
        memoryBallast.push(Buffer.alloc(chunkSize));
        if (config.gradual && i % 100 === 0) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
    }

    this.injectionCallbacks.set('memory_ballast', () => {
      memoryBallast.length = 0; // Release memory
      if (global.gc) global.gc();
    });
  }

  async simulateCpuExhaustion(config) {
    const targetUsage = config.targetUsage || 0.95;
    let active = true;
    
    // Spawn CPU-intensive workers
    const workers = [];
    const numWorkers = require('os').cpus().length;
    
    for (let i = 0; i < numWorkers; i++) {
      workers.push(setInterval(() => {
        if (active) {
          const start = Date.now();
          while (Date.now() - start < 50) {
            Math.random() * Math.random();
          }
        }
      }, 100));
    }

    this.injectionCallbacks.set('cpu_exhaust', () => {
      active = false;
      workers.forEach(worker => clearInterval(worker));
    });
  }

  // Byzantine fault simulations
  async simulateMaliciousResponses(config) {
    const component = config.component;
    const responses = config.responses || {};
    
    // Inject malicious behavior into specified component
    global.BYZANTINE_ACTIVE = true;
    global.BYZANTINE_CONFIG = { component, responses, probability: config.probability };

    this.injectionCallbacks.set('byzantine_responses', () => {
      global.BYZANTINE_ACTIVE = false;
      global.BYZANTINE_CONFIG = null;
    });
  }

  async simulateInconsistentState(config) {
    global.BYZANTINE_INCONSISTENT_STATE = true;
    global.BYZANTINE_INCONSISTENCY_RATE = config.inconsistencyRate || 0.1;

    this.injectionCallbacks.set('byzantine_state', () => {
      global.BYZANTINE_INCONSISTENT_STATE = false;
      global.BYZANTINE_INCONSISTENCY_RATE = 0;
    });
  }

  // Utility methods
  generateFaultId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  clearFault(faultId) {
    const fault = this.activeFaults.get(faultId);
    if (fault) {
      fault.status = 'cleared';
      fault.endTime = Date.now();
      fault.duration = fault.endTime - fault.startTime;
      
      this.faultHistory.push(fault);
      this.activeFaults.delete(faultId);
      
      // Execute cleanup callback
      const callback = this.injectionCallbacks.get(this.getCallbackKey(fault));
      if (callback) {
        callback();
        this.injectionCallbacks.delete(this.getCallbackKey(fault));
      }
    }
  }

  getCallbackKey(fault) {
    switch (fault.type) {
      case 'database':
        switch (fault.config.type) {
          case 'CONNECTION_LOSS': return 'db_connect';
          case 'SLOW_QUERIES': return 'db_query';
          case 'DATA_CORRUPTION': return 'db_corrupt';
          default: return 'db_generic';
        }
      case 'network':
        switch (fault.config.type) {
          case 'INTERMITTENT_LOSS': return 'http_request';
          case 'HIGH_LATENCY': return 'http_latency';
          case 'COMPLETE_PARTITION': return 'network_partition';
          default: return 'net_generic';
        }
      case 'resource':
        switch (fault.config.type) {
          case 'MEMORY_EXHAUSTION': return 'memory_ballast';
          case 'CPU_EXHAUSTION': return 'cpu_exhaust';
          default: return 'res_generic';
        }
      case 'byzantine':
        return 'byzantine_responses';
      default:
        return 'generic';
    }
  }

  async clearAllFaults() {
    const faultIds = Array.from(this.activeFaults.keys());
    for (const faultId of faultIds) {
      this.clearFault(faultId);
    }
  }

  async clearResourceFaults() {
    const resourceFaults = Array.from(this.activeFaults.entries())
      .filter(([_, fault]) => fault.type === 'resource')
      .map(([id, _]) => id);
    
    for (const faultId of resourceFaults) {
      this.clearFault(faultId);
    }
  }

  async clearByzantineFaults() {
    const byzantineFaults = Array.from(this.activeFaults.entries())
      .filter(([_, fault]) => fault.type === 'byzantine')
      .map(([id, _]) => id);
    
    for (const faultId of byzantineFaults) {
      this.clearFault(faultId);
    }
  }

  getActiveFaults() {
    return Array.from(this.activeFaults.values());
  }

  getFaultHistory() {
    return [...this.faultHistory];
  }

  getFaultStatistics() {
    return {
      totalFaults: this.faultHistory.length + this.activeFaults.size,
      activeFaults: this.activeFaults.size,
      faultsByType: this.getFaultsByType(),
      averageDuration: this.getAverageFaultDuration()
    };
  }

  getFaultsByType() {
    const allFaults = [...this.faultHistory, ...Array.from(this.activeFaults.values())];
    const byType = {};
    
    allFaults.forEach(fault => {
      byType[fault.type] = (byType[fault.type] || 0) + 1;
    });
    
    return byType;
  }

  getAverageFaultDuration() {
    const completedFaults = this.faultHistory.filter(f => f.duration);
    if (completedFaults.length === 0) return 0;
    
    const totalDuration = completedFaults.reduce((sum, f) => sum + f.duration, 0);
    return totalDuration / completedFaults.length;
  }
}

module.exports = { FaultInjector };