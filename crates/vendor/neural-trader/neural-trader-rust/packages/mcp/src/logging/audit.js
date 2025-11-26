/**
 * Audit Logging (JSON Lines format)
 * MCP 2025-11 compliance requirement
 */

const fs = require('fs');
const path = require('path');

class AuditLogger {
  constructor(options = {}) {
    this.options = {
      logFile: path.join(__dirname, '../../logs/mcp-audit.jsonl'),
      enabled: true,
      ...options,
    };
    this.stream = null;

    if (this.options.enabled) {
      this.initializeLogFile();
    }
  }

  /**
   * Initialize log file
   */
  initializeLogFile() {
    try {
      const logDir = path.dirname(this.options.logFile);
      if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
      }

      this.stream = fs.createWriteStream(this.options.logFile, {
        flags: 'a', // Append mode
        encoding: 'utf8',
      });

      this.log({
        type: 'server_start',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error('Failed to initialize audit log:', error.message);
      this.options.enabled = false;
    }
  }

  /**
   * Write a log entry (JSON Lines format)
   */
  log(entry) {
    if (!this.options.enabled || !this.stream) {
      return;
    }

    try {
      const line = JSON.stringify({
        ...entry,
        timestamp: entry.timestamp || new Date().toISOString(),
      });
      this.stream.write(line + '\n');
    } catch (error) {
      console.error('Audit log write error:', error.message);
    }
  }

  /**
   * Log a tool call
   */
  logToolCall(toolName, args) {
    this.log({
      type: 'tool_call',
      tool: toolName,
      arguments: args,
    });
  }

  /**
   * Log a tool result
   */
  logToolResult(toolName, result, error, duration) {
    this.log({
      type: 'tool_result',
      tool: toolName,
      success: !error,
      error: error ? error.message : null,
      duration_ms: duration,
    });
  }

  /**
   * Log an error
   */
  logError(error, context = {}) {
    this.log({
      type: 'error',
      error: error.message,
      stack: error.stack,
      ...context,
    });
  }

  /**
   * Close the audit log
   */
  close() {
    if (this.stream) {
      this.log({
        type: 'server_stop',
      });
      this.stream.end();
      this.stream = null;
    }
  }
}

function createAuditLogger(options) {
  return new AuditLogger(options);
}

module.exports = { AuditLogger, createAuditLogger };
