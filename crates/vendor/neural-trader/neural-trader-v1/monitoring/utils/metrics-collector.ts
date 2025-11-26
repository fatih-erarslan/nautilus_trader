/**
 * Metrics Collector Utility
 * Collects and aggregates metrics from multiple sources
 */

import { EventEmitter } from 'events';

interface MetricPoint {
  timestamp: Date;
  value: number;
  metadata?: Record<string, any>;
}

interface MetricSeries {
  name: string;
  unit: string;
  points: MetricPoint[];
  aggregations?: {
    min: number;
    max: number;
    avg: number;
    stdDev: number;
  };
}

export class MetricsCollector extends EventEmitter {
  private metrics: Map<string, MetricSeries> = new Map();
  private maxPoints: number;

  constructor(maxPoints: number = 1000) {
    super();
    this.maxPoints = maxPoints;
  }

  public record(name: string, value: number, unit: string = '', metadata?: Record<string, any>): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, {
        name,
        unit,
        points: []
      });
    }

    const series = this.metrics.get(name)!;
    series.points.push({
      timestamp: new Date(),
      value,
      metadata
    });

    // Limit points
    if (series.points.length > this.maxPoints) {
      series.points.shift();
    }

    // Update aggregations
    series.aggregations = this.calculateAggregations(series.points);

    this.emit('metric-recorded', { name, value, unit, metadata });
  }

  public get(name: string): MetricSeries | undefined {
    return this.metrics.get(name);
  }

  public getAll(): MetricSeries[] {
    return Array.from(this.metrics.values());
  }

  public getLastValue(name: string): number | undefined {
    const series = this.metrics.get(name);
    if (!series || series.points.length === 0) return undefined;
    return series.points[series.points.length - 1].value;
  }

  public getTimeSeries(name: string, duration: number): MetricPoint[] {
    const series = this.metrics.get(name);
    if (!series) return [];

    const cutoff = Date.now() - duration;
    return series.points.filter(p => p.timestamp.getTime() >= cutoff);
  }

  private calculateAggregations(points: MetricPoint[]): {
    min: number;
    max: number;
    avg: number;
    stdDev: number;
  } {
    if (points.length === 0) {
      return { min: 0, max: 0, avg: 0, stdDev: 0 };
    }

    const values = points.map(p => p.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((sum, v) => sum + v, 0) / values.length;

    const variance = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return { min, max, avg, stdDev };
  }

  public clear(name?: string): void {
    if (name) {
      this.metrics.delete(name);
    } else {
      this.metrics.clear();
    }
  }

  public export(): string {
    return JSON.stringify(
      Array.from(this.metrics.values()),
      null,
      2
    );
  }
}

export default MetricsCollector;
