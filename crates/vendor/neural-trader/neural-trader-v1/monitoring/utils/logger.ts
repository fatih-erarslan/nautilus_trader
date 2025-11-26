/**
 * Structured Logger Utility
 * Provides consistent logging across monitoring components
 */

import * as fs from 'fs/promises';
import * as path from 'path';

type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'critical';

interface LogEntry {
  timestamp: Date;
  level: LogLevel;
  component: string;
  message: string;
  data?: any;
  correlationId?: string;
}

export class Logger {
  private component: string;
  private logFile?: string;
  private minLevel: LogLevel;
  private buffer: LogEntry[] = [];
  private flushInterval: NodeJS.Timeout | null = null;

  private levelPriority: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
    critical: 4
  };

  constructor(component: string, options: {
    logFile?: string;
    minLevel?: LogLevel;
    flushIntervalMs?: number;
  } = {}) {
    this.component = component;
    this.logFile = options.logFile;
    this.minLevel = options.minLevel || 'info';

    if (options.flushIntervalMs) {
      this.flushInterval = setInterval(() => this.flush(), options.flushIntervalMs);
    }
  }

  public debug(message: string, data?: any): void {
    this.log('debug', message, data);
  }

  public info(message: string, data?: any): void {
    this.log('info', message, data);
  }

  public warn(message: string, data?: any): void {
    this.log('warn', message, data);
  }

  public error(message: string, data?: any): void {
    this.log('error', message, data);
  }

  public critical(message: string, data?: any): void {
    this.log('critical', message, data);
  }

  private log(level: LogLevel, message: string, data?: any, correlationId?: string): void {
    if (this.levelPriority[level] < this.levelPriority[this.minLevel]) {
      return; // Skip logs below minimum level
    }

    const entry: LogEntry = {
      timestamp: new Date(),
      level,
      component: this.component,
      message,
      data,
      correlationId
    };

    // Console output with colors
    const color = this.getColor(level);
    const timestamp = entry.timestamp.toISOString();
    console.log(
      `${color}[${timestamp}] [${level.toUpperCase()}] [${this.component}]${this.resetColor()} ${message}`,
      data ? `\n${JSON.stringify(data, null, 2)}` : ''
    );

    // Buffer for file output
    if (this.logFile) {
      this.buffer.push(entry);
    }
  }

  private getColor(level: LogLevel): string {
    const colors: Record<LogLevel, string> = {
      debug: '\x1b[36m',    // Cyan
      info: '\x1b[32m',     // Green
      warn: '\x1b[33m',     // Yellow
      error: '\x1b[31m',    // Red
      critical: '\x1b[35m'  // Magenta
    };
    return colors[level];
  }

  private resetColor(): string {
    return '\x1b[0m';
  }

  public async flush(): Promise<void> {
    if (!this.logFile || this.buffer.length === 0) return;

    const logs = this.buffer.splice(0, this.buffer.length);
    const logLines = logs.map(entry =>
      JSON.stringify(entry)
    ).join('\n') + '\n';

    try {
      await fs.appendFile(this.logFile, logLines);
    } catch (error) {
      console.error('Failed to write logs:', error);
    }
  }

  public async close(): Promise<void> {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    await this.flush();
  }
}

export default Logger;
