import type { AnomalyPoint } from '../index';

interface IsolationTree {
  feature?: number;
  splitValue?: number;
  left?: IsolationTree;
  right?: IsolationTree;
  size: number;
}

/**
 * Isolation Forest algorithm for anomaly detection
 *
 * Based on the principle that anomalies are easier to isolate (require fewer splits)
 * in a random decision tree than normal points.
 */
export class IsolationForest {
  private trees: IsolationTree[] = [];
  private numTrees = 100;
  private subsampleSize = 256;
  private maxDepth = 8;

  constructor(private featureDimensions: number) {}

  /**
   * Train isolation forest
   */
  async train(data: AnomalyPoint[]): Promise<void> {
    this.trees = [];

    for (let i = 0; i < this.numTrees; i++) {
      // Random subsample
      const subsample = this.randomSubsample(data, this.subsampleSize);

      // Build isolation tree
      const tree = this.buildTree(subsample, 0);
      this.trees.push(tree);
    }
  }

  /**
   * Build a single isolation tree recursively
   */
  private buildTree(data: AnomalyPoint[], depth: number): IsolationTree {
    const size = data.length;

    // Terminal conditions
    if (depth >= this.maxDepth || size <= 1) {
      return { size };
    }

    // Random split
    const feature = Math.floor(Math.random() * this.featureDimensions);
    const values = data.map(p => p.features[feature]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);

    if (minVal === maxVal) {
      return { size };
    }

    const splitValue = minVal + Math.random() * (maxVal - minVal);

    // Split data
    const leftData = data.filter(p => p.features[feature] < splitValue);
    const rightData = data.filter(p => p.features[feature] >= splitValue);

    return {
      feature,
      splitValue,
      left: this.buildTree(leftData, depth + 1),
      right: this.buildTree(rightData, depth + 1),
      size,
    };
  }

  /**
   * Predict anomaly score (0-1, higher = more anomalous)
   */
  predict(point: AnomalyPoint): number {
    const avgPathLength = this.trees.reduce((sum, tree) =>
      sum + this.pathLength(point, tree, 0), 0
    ) / this.trees.length;

    // Normalize using expected path length
    const c = this.expectedPathLength(this.subsampleSize);
    const score = Math.pow(2, -avgPathLength / c);

    return score;
  }

  /**
   * Compute path length to leaf for a point
   */
  private pathLength(point: AnomalyPoint, tree: IsolationTree, currentDepth: number): number {
    // Leaf node
    if (tree.left === undefined || tree.right === undefined) {
      return currentDepth + this.expectedPathLength(tree.size);
    }

    const feature = tree.feature!;
    const splitValue = tree.splitValue!;

    if (point.features[feature] < splitValue) {
      return this.pathLength(point, tree.left!, currentDepth + 1);
    } else {
      return this.pathLength(point, tree.right!, currentDepth + 1);
    }
  }

  /**
   * Expected path length for a BST of size n
   */
  private expectedPathLength(n: number): number {
    if (n <= 1) return 0;
    if (n === 2) return 1;

    const H = Math.log(n - 1) + 0.5772156649; // Euler's constant
    return 2 * H - (2 * (n - 1) / n);
  }

  /**
   * Random subsample without replacement
   */
  private randomSubsample(data: AnomalyPoint[], size: number): AnomalyPoint[] {
    const shuffled = [...data].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(size, data.length));
  }
}
