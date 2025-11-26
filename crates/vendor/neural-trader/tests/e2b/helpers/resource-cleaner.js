/**
 * Resource Cleaner Helper
 *
 * Manages cleanup of E2B resources and sandboxes
 */

class ResourceCleaner {
  constructor(credentials) {
    this.credentials = credentials;
    this.cleanedResources = [];
    this.failedCleanups = [];
  }

  /**
   * Clean up a sandbox
   * @param {string} sandboxId - Sandbox ID to clean
   * @returns {Promise<boolean>} True if successful
   */
  async cleanupSandbox(sandboxId) {
    try {
      console.log(`🧹 Cleaning up sandbox: ${sandboxId}`);

      // In real implementation, call E2B API to terminate sandbox
      // For now, simulate cleanup
      await this.simulateCleanup(sandboxId);

      this.cleanedResources.push({
        sandboxId,
        timestamp: new Date().toISOString(),
        success: true,
      });

      console.log(`✅ Successfully cleaned up: ${sandboxId}`);
      return true;
    } catch (error) {
      console.error(`❌ Failed to cleanup ${sandboxId}:`, error.message);

      this.failedCleanups.push({
        sandboxId,
        timestamp: new Date().toISOString(),
        error: error.message,
      });

      return false;
    }
  }

  /**
   * Simulate cleanup delay
   * @param {string} sandboxId - Sandbox ID
   */
  async simulateCleanup(sandboxId) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Simulate 95% success rate
    if (Math.random() < 0.05) {
      throw new Error('Simulated cleanup failure');
    }
  }

  /**
   * Clean up multiple sandboxes
   * @param {Array<string>} sandboxIds - Array of sandbox IDs
   * @param {number} delayMs - Delay between cleanups
   * @returns {Promise<Object>} Cleanup results
   */
  async cleanupBatch(sandboxIds, delayMs = 2000) {
    const results = {
      total: sandboxIds.length,
      successful: 0,
      failed: 0,
      details: [],
    };

    for (const sandboxId of sandboxIds) {
      const success = await this.cleanupSandbox(sandboxId);

      if (success) {
        results.successful++;
      } else {
        results.failed++;
      }

      results.details.push({ sandboxId, success });

      // Wait between cleanups to avoid rate limiting
      if (delayMs > 0 && sandboxId !== sandboxIds[sandboxIds.length - 1]) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    return results;
  }

  /**
   * Get cleanup statistics
   * @returns {Object} Cleanup statistics
   */
  getCleanupStats() {
    return {
      totalCleaned: this.cleanedResources.length,
      totalFailed: this.failedCleanups.length,
      successRate: this.cleanedResources.length > 0
        ? (this.cleanedResources.length / (this.cleanedResources.length + this.failedCleanups.length)) * 100
        : 0,
      cleanedResources: this.cleanedResources,
      failedCleanups: this.failedCleanups,
    };
  }

  /**
   * Verify resource cleanup
   * @param {string} sandboxId - Sandbox ID to verify
   * @returns {boolean} True if cleaned
   */
  isResourceCleaned(sandboxId) {
    return this.cleanedResources.some(r => r.sandboxId === sandboxId);
  }

  /**
   * Retry failed cleanups
   * @returns {Promise<Object>} Retry results
   */
  async retryFailedCleanups() {
    if (this.failedCleanups.length === 0) {
      return { message: 'No failed cleanups to retry' };
    }

    console.log(`🔄 Retrying ${this.failedCleanups.length} failed cleanups...`);

    const failedIds = this.failedCleanups.map(f => f.sandboxId);
    this.failedCleanups = []; // Clear failed list

    return await this.cleanupBatch(failedIds, 3000);
  }

  /**
   * Force cleanup all tracked resources
   * @returns {Promise<Object>} Cleanup results
   */
  async forceCleanupAll() {
    const allIds = [
      ...this.cleanedResources.map(r => r.sandboxId),
      ...this.failedCleanups.map(f => f.sandboxId),
    ];

    console.log(`⚠️  Force cleaning ${allIds.length} resources...`);

    // Reset tracking
    this.cleanedResources = [];
    this.failedCleanups = [];

    return await this.cleanupBatch(allIds, 1000);
  }

  /**
   * Generate cleanup report
   * @returns {Object} Cleanup report
   */
  generateReport() {
    const stats = this.getCleanupStats();

    return {
      summary: {
        totalResources: stats.totalCleaned + stats.totalFailed,
        successful: stats.totalCleaned,
        failed: stats.totalFailed,
        successRate: `${stats.successRate.toFixed(2)}%`,
        generatedAt: new Date().toISOString(),
      },
      details: {
        cleaned: this.cleanedResources,
        failed: this.failedCleanups,
      },
    };
  }

  /**
   * Clear cleanup history
   */
  clearHistory() {
    this.cleanedResources = [];
    this.failedCleanups = [];
  }
}

module.exports = { ResourceCleaner };
