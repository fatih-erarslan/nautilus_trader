/**
 * Merkle Proof System
 * Generate tamper-evident audit trails
 */

import crypto from 'crypto';
import { Transaction } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';

export interface MerkleNode {
  hash: string;
  left?: MerkleNode;
  right?: MerkleNode;
  data?: any;
}

export interface MerkleProof {
  transactionId: string;
  rootHash: string;
  proof: string[];
  index: number;
  timestamp: Date;
}

export class MerkleTreeService {
  /**
   * Build Merkle tree from transactions
   */
  buildTree(transactions: Transaction[]): MerkleNode {
    if (transactions.length === 0) {
      throw new Error('Cannot build Merkle tree from empty transaction list');
    }

    // Create leaf nodes
    let nodes: MerkleNode[] = transactions.map(tx => ({
      hash: this.hashTransaction(tx),
      data: tx
    }));

    // Build tree bottom-up
    while (nodes.length > 1) {
      const parentNodes: MerkleNode[] = [];

      for (let i = 0; i < nodes.length; i += 2) {
        const left = nodes[i];
        const right = i + 1 < nodes.length ? nodes[i + 1] : left;

        const parentHash = this.hashPair(left.hash, right.hash);
        parentNodes.push({
          hash: parentHash,
          left,
          right
        });
      }

      nodes = parentNodes;
    }

    logger.debug('Merkle tree built', {
      rootHash: nodes[0].hash,
      transactions: transactions.length
    });

    return nodes[0]; // Root node
  }

  /**
   * Generate Merkle proof for a transaction
   */
  generateProof(transactions: Transaction[], transactionId: string): MerkleProof {
    const index = transactions.findIndex(tx => tx.id === transactionId);
    if (index === -1) {
      throw new Error(`Transaction ${transactionId} not found in transaction list`);
    }

    const proof: string[] = [];
    const tree = this.buildTree(transactions);

    // Generate proof by walking up the tree
    let currentIndex = index;
    let levelNodes = transactions.length;

    // Rebuild tree to get proof path
    let nodes: string[] = transactions.map(tx => this.hashTransaction(tx));

    while (nodes.length > 1) {
      const newNodes: string[] = [];

      for (let i = 0; i < nodes.length; i += 2) {
        if (i === currentIndex) {
          // Found our node, add sibling to proof
          const siblingIndex = i % 2 === 0 ? i + 1 : i - 1;
          if (siblingIndex < nodes.length) {
            proof.push(nodes[siblingIndex]);
          }
          currentIndex = Math.floor(i / 2);
        } else if (i + 1 === currentIndex) {
          // Our node is the right sibling
          proof.push(nodes[i]);
          currentIndex = Math.floor(i / 2);
        }

        const left = nodes[i];
        const right = i + 1 < nodes.length ? nodes[i + 1] : left;
        newNodes.push(this.hashPair(left, right));
      }

      nodes = newNodes;
    }

    return {
      transactionId,
      rootHash: tree.hash,
      proof,
      index,
      timestamp: new Date()
    };
  }

  /**
   * Verify Merkle proof
   */
  verifyProof(
    transaction: Transaction,
    proof: MerkleProof,
    expectedRootHash: string
  ): boolean {
    let hash = this.hashTransaction(transaction);
    let index = proof.index;

    // Reconstruct root hash using proof
    for (const siblingHash of proof.proof) {
      if (index % 2 === 0) {
        hash = this.hashPair(hash, siblingHash);
      } else {
        hash = this.hashPair(siblingHash, hash);
      }
      index = Math.floor(index / 2);
    }

    const isValid = hash === expectedRootHash;

    logger.debug('Merkle proof verification', {
      transactionId: transaction.id,
      isValid,
      computedRoot: hash,
      expectedRoot: expectedRootHash
    });

    return isValid;
  }

  /**
   * Hash a transaction
   */
  private hashTransaction(transaction: Transaction): string {
    const data = JSON.stringify({
      id: transaction.id,
      timestamp: transaction.timestamp.toISOString(),
      type: transaction.type,
      asset: transaction.asset,
      quantity: transaction.quantity,
      price: transaction.price,
      fees: transaction.fees
    });

    return crypto.createHash('sha256').update(data).digest('hex');
  }

  /**
   * Hash a pair of nodes
   */
  private hashPair(left: string, right: string): string {
    return crypto
      .createHash('sha256')
      .update(left + right)
      .digest('hex');
  }

  /**
   * Get root hash of transaction set
   */
  getRootHash(transactions: Transaction[]): string {
    const tree = this.buildTree(transactions);
    return tree.hash;
  }

  /**
   * Batch generate proofs for all transactions
   */
  generateAllProofs(transactions: Transaction[]): Map<string, MerkleProof> {
    const proofs = new Map<string, MerkleProof>();

    for (const tx of transactions) {
      const proof = this.generateProof(transactions, tx.id);
      proofs.set(tx.id, proof);
    }

    logger.info(`Generated ${proofs.size} Merkle proofs`);
    return proofs;
  }

  /**
   * Verify batch of proofs
   */
  verifyAllProofs(
    transactions: Transaction[],
    proofs: Map<string, MerkleProof>,
    expectedRootHash: string
  ): Map<string, boolean> {
    const results = new Map<string, boolean>();

    for (const tx of transactions) {
      const proof = proofs.get(tx.id);
      if (proof) {
        const isValid = this.verifyProof(tx, proof, expectedRootHash);
        results.set(tx.id, isValid);
      } else {
        results.set(tx.id, false);
      }
    }

    return results;
  }
}
