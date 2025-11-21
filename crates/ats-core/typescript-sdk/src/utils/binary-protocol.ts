/**
 * Binary Protocol Handler for Ultra-Low Latency
 * 
 * Handles binary serialization/deserialization for sub-25μs latency requirements
 * using optimized binary formats and zero-copy operations where possible.
 */

import { BinaryPredictionMessage } from '../types';

/**
 * Binary message format layout (64 bytes total):
 * - msg_type: 4 bytes (u32)
 * - model_id_hash: 8 bytes (u64)  
 * - timestamp_ns: 8 bytes (u64)
 * - prediction: 8 bytes (f64)
 * - lower_bound: 8 bytes (f64)
 * - upper_bound: 8 bytes (f64)
 * - confidence: 8 bytes (f64)
 * - latency_ns: 8 bytes (u64)
 * - padding: 4 bytes (for 64-byte alignment)
 */
const BINARY_MESSAGE_SIZE = 64;

export class BinaryProtocolHandler {
  private encoder: TextEncoder;
  private decoder: TextDecoder;

  constructor() {
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
  }

  /**
   * Encode message to binary format for ultra-low latency transmission
   */
  public encode(message: any): ArrayBuffer {
    const buffer = new ArrayBuffer(BINARY_MESSAGE_SIZE);
    const view = new DataView(buffer);
    
    let offset = 0;

    // Message type (4 bytes)
    view.setUint32(offset, 1, true); // 1 = PredictionUpdate
    offset += 4;

    // Model ID hash (8 bytes) - simple hash for demo
    const modelIdHash = this.simpleHash(message.data?.model_id || '');
    view.setBigUint64(offset, BigInt(modelIdHash), true);
    offset += 8;

    // Timestamp (8 bytes) - nanoseconds since epoch
    const timestampNs = BigInt(Date.now() * 1000000); // Convert ms to ns
    view.setBigUint64(offset, timestampNs, true);
    offset += 8;

    // Prediction data (32 bytes total)
    const prediction = message.data?.prediction || {};
    
    view.setFloat64(offset, prediction.point_prediction || 0, true);
    offset += 8;
    
    view.setFloat64(offset, prediction.prediction_intervals?.[0]?.lower_bound || 0, true);
    offset += 8;
    
    view.setFloat64(offset, prediction.prediction_intervals?.[0]?.upper_bound || 0, true);
    offset += 8;
    
    view.setFloat64(offset, prediction.prediction_intervals?.[0]?.confidence || 0, true);
    offset += 8;

    // Latency (8 bytes)
    const latencyNs = BigInt((message.data?.latency_us || 0) * 1000); // Convert μs to ns
    view.setBigUint64(offset, latencyNs, true);
    offset += 8;

    // Padding (4 bytes) - reserved for future use
    view.setUint32(offset, 0, true);

    return buffer;
  }

  /**
   * Decode binary format back to structured message
   */
  public decode(buffer: ArrayBuffer): BinaryPredictionMessage {
    if (buffer.byteLength !== BINARY_MESSAGE_SIZE) {
      throw new Error(`Invalid binary message size: expected ${BINARY_MESSAGE_SIZE}, got ${buffer.byteLength}`);
    }

    const view = new DataView(buffer);
    let offset = 0;

    // Message type (4 bytes)
    const msgType = view.getUint32(offset, true);
    offset += 4;

    // Model ID hash (8 bytes)
    const modelIdHash = view.getBigUint64(offset, true);
    offset += 8;

    // Timestamp (8 bytes)
    const timestampNs = view.getBigUint64(offset, true);
    offset += 8;

    // Prediction data
    const prediction = view.getFloat64(offset, true);
    offset += 8;

    const lowerBound = view.getFloat64(offset, true);
    offset += 8;

    const upperBound = view.getFloat64(offset, true);
    offset += 8;

    const confidence = view.getFloat64(offset, true);
    offset += 8;

    // Latency (8 bytes)
    const latencyNs = view.getBigUint64(offset, true);
    offset += 8;

    return {
      msg_type: msgType,
      model_id_hash: modelIdHash,
      timestamp_ns: timestampNs,
      prediction,
      lower_bound: lowerBound,
      upper_bound: upperBound,
      confidence,
      latency_ns: latencyNs,
    };
  }

  /**
   * Fast message validation without full deserialization
   */
  public isValidBinaryMessage(buffer: ArrayBuffer): boolean {
    if (buffer.byteLength !== BINARY_MESSAGE_SIZE) {
      return false;
    }

    const view = new DataView(buffer);
    const msgType = view.getUint32(0, true);
    
    // Check if message type is supported (1 = PredictionUpdate)
    return msgType === 1;
  }

  /**
   * Extract timestamp from binary message without full decode
   */
  public extractTimestamp(buffer: ArrayBuffer): bigint {
    if (buffer.byteLength !== BINARY_MESSAGE_SIZE) {
      throw new Error('Invalid binary message size');
    }

    const view = new DataView(buffer);
    return view.getBigUint64(12, true); // timestamp is at offset 12
  }

  /**
   * Extract latency from binary message without full decode
   */
  public extractLatency(buffer: ArrayBuffer): bigint {
    if (buffer.byteLength !== BINARY_MESSAGE_SIZE) {
      throw new Error('Invalid binary message size');
    }

    const view = new DataView(buffer);
    return view.getBigUint64(56, true); // latency is at offset 56
  }

  /**
   * Create optimized batch binary format for multiple predictions
   */
  public encodeBatch(messages: any[]): ArrayBuffer {
    const batchSize = messages.length;
    const headerSize = 8; // 4 bytes count + 4 bytes reserved
    const totalSize = headerSize + (batchSize * BINARY_MESSAGE_SIZE);
    
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    
    // Header: message count
    view.setUint32(0, batchSize, true);
    view.setUint32(4, 0, true); // reserved
    
    let offset = headerSize;
    
    // Encode each message
    for (const message of messages) {
      const messageBuffer = this.encode(message);
      const messageView = new Uint8Array(messageBuffer);
      const targetView = new Uint8Array(buffer, offset, BINARY_MESSAGE_SIZE);
      
      targetView.set(messageView);
      offset += BINARY_MESSAGE_SIZE;
    }
    
    return buffer;
  }

  /**
   * Decode batch binary format
   */
  public decodeBatch(buffer: ArrayBuffer): BinaryPredictionMessage[] {
    if (buffer.byteLength < 8) {
      throw new Error('Invalid batch binary message size');
    }

    const view = new DataView(buffer);
    const messageCount = view.getUint32(0, true);
    
    const expectedSize = 8 + (messageCount * BINARY_MESSAGE_SIZE);
    if (buffer.byteLength !== expectedSize) {
      throw new Error(`Invalid batch size: expected ${expectedSize}, got ${buffer.byteLength}`);
    }

    const messages: BinaryPredictionMessage[] = [];
    let offset = 8;

    for (let i = 0; i < messageCount; i++) {
      const messageBuffer = buffer.slice(offset, offset + BINARY_MESSAGE_SIZE);
      const message = this.decode(messageBuffer);
      messages.push(message);
      offset += BINARY_MESSAGE_SIZE;
    }

    return messages;
  }

  /**
   * Simple hash function for model ID (for demo purposes)
   * In production, this would use a proper hash function
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Benchmark binary vs JSON performance
   */
  public benchmark(iterations = 10000): {
    binaryEncodeTime: number;
    binaryDecodeTime: number;
    jsonEncodeTime: number;
    jsonDecodeTime: number;
    binarySize: number;
    jsonSize: number;
    speedupRatio: number;
  } {
    const testMessage = {
      data: {
        model_id: 'test_model_123',
        prediction: {
          point_prediction: 42.5,
          prediction_intervals: [{
            lower_bound: 40.0,
            upper_bound: 45.0,
            confidence: 0.95,
          }],
        },
        timestamp: new Date().toISOString(),
        latency_us: 15,
      },
    };

    // Binary encoding benchmark
    let start = performance.now();
    let binaryData: ArrayBuffer;
    for (let i = 0; i < iterations; i++) {
      binaryData = this.encode(testMessage);
    }
    const binaryEncodeTime = performance.now() - start;

    // Binary decoding benchmark
    start = performance.now();
    for (let i = 0; i < iterations; i++) {
      this.decode(binaryData!);
    }
    const binaryDecodeTime = performance.now() - start;

    // JSON encoding benchmark
    start = performance.now();
    let jsonData: string;
    for (let i = 0; i < iterations; i++) {
      jsonData = JSON.stringify(testMessage);
    }
    const jsonEncodeTime = performance.now() - start;

    // JSON decoding benchmark
    start = performance.now();
    for (let i = 0; i < iterations; i++) {
      JSON.parse(jsonData!);
    }
    const jsonDecodeTime = performance.now() - start;

    const binarySize = binaryData!.byteLength;
    const jsonSize = new Blob([jsonData!]).size;
    const binaryTotalTime = binaryEncodeTime + binaryDecodeTime;
    const jsonTotalTime = jsonEncodeTime + jsonDecodeTime;
    const speedupRatio = jsonTotalTime / binaryTotalTime;

    return {
      binaryEncodeTime,
      binaryDecodeTime,
      jsonEncodeTime,
      jsonDecodeTime,
      binarySize,
      jsonSize,
      speedupRatio,
    };
  }

  /**
   * Get binary protocol statistics
   */
  public getStats() {
    return {
      messageSize: BINARY_MESSAGE_SIZE,
      supportedMessageTypes: ['PredictionUpdate'],
      endianness: 'little-endian',
      alignment: '8-byte',
    };
  }
}