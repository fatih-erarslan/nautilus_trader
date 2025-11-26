/**
 * Stream Processor
 * Memory-efficient processing of large datasets
 */

import { Readable, Transform } from 'stream';
import { pipeline } from 'stream/promises';

export interface StreamOptions {
  batchSize?: number;
  concurrency?: number;
  onProgress?: (processed: number) => void;
  onError?: (error: Error, item: any) => void;
}

/**
 * Process large datasets in streaming fashion
 */
export class StreamProcessor<TIn, TOut> {
  private options: Required<Omit<StreamOptions, 'onProgress' | 'onError'>> & Pick<StreamOptions, 'onProgress' | 'onError'>;

  constructor(options: StreamOptions = {}) {
    this.options = {
      batchSize: options.batchSize || 1000,
      concurrency: options.concurrency || 10,
      onProgress: options.onProgress,
      onError: options.onError,
    };
  }

  /**
   * Process items in batches using streaming
   */
  async processBatches(
    items: TIn[],
    processor: (batch: TIn[]) => Promise<TOut[]>
  ): Promise<TOut[]> {
    const results: TOut[] = [];
    let processed = 0;

    // Create readable stream from array
    const source = Readable.from(this.chunkArray(items, this.options.batchSize));

    // Create transform stream for processing
    const transform = new Transform({
      objectMode: true,
      async transform(batch: TIn[], encoding, callback) {
        try {
          const result = await processor(batch);
          results.push(...result);
          processed += batch.length;

          if (this.options.onProgress) {
            this.options.onProgress(processed);
          }

          callback();
        } catch (error) {
          callback(error instanceof Error ? error : new Error(String(error)));
        }
      }.bind(this),
    });

    // Process stream
    await pipeline(source, transform);

    return results;
  }

  /**
   * Process items individually with concurrency control
   */
  async processIndividual(
    items: TIn[],
    processor: (item: TIn) => Promise<TOut>
  ): Promise<TOut[]> {
    const results: TOut[] = [];
    const errors: Array<{ item: TIn; error: Error }> = [];

    // Process in chunks to control memory usage
    for (let i = 0; i < items.length; i += this.options.concurrency) {
      const chunk = items.slice(i, i + this.options.concurrency);

      const chunkResults = await Promise.allSettled(
        chunk.map(item => processor(item))
      );

      chunkResults.forEach((result, idx) => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        } else {
          const error = result.reason instanceof Error
            ? result.reason
            : new Error(String(result.reason));

          errors.push({ item: chunk[idx], error });

          if (this.options.onError) {
            this.options.onError(error, chunk[idx]);
          }
        }
      });

      if (this.options.onProgress) {
        this.options.onProgress(Math.min(i + this.options.concurrency, items.length));
      }
    }

    if (errors.length > 0 && !this.options.onError) {
      console.warn(`${errors.length} items failed processing`);
    }

    return results;
  }

  /**
   * Transform stream with mapping function
   */
  async transformStream(
    items: TIn[],
    mapper: (item: TIn) => Promise<TOut>
  ): Promise<TOut[]> {
    return this.processIndividual(items, mapper);
  }

  /**
   * Filter stream with predicate
   */
  async filterStream(
    items: TIn[],
    predicate: (item: TIn) => Promise<boolean>
  ): Promise<TIn[]> {
    const results: TIn[] = [];

    for (let i = 0; i < items.length; i += this.options.concurrency) {
      const chunk = items.slice(i, i + this.options.concurrency);

      const predicateResults = await Promise.all(
        chunk.map(item => predicate(item).then(pass => ({ item, pass })))
      );

      predicateResults.forEach(({ item, pass }) => {
        if (pass) {
          results.push(item);
        }
      });

      if (this.options.onProgress) {
        this.options.onProgress(Math.min(i + this.options.concurrency, items.length));
      }
    }

    return results;
  }

  /**
   * Chunk array into smaller arrays
   */
  private *chunkArray(array: TIn[], size: number): Generator<TIn[]> {
    for (let i = 0; i < array.length; i += size) {
      yield array.slice(i, i + size);
    }
  }
}

/**
 * Helper function to create a stream processor
 */
export function createStreamProcessor<TIn, TOut>(options?: StreamOptions): StreamProcessor<TIn, TOut> {
  return new StreamProcessor<TIn, TOut>(options);
}
