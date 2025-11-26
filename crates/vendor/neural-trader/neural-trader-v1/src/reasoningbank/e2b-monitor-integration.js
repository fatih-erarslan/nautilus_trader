/**
 * E2BMonitor Integration with ReasoningBank
 *
 * Adds learning metrics to monitoring and health checks
 *
 * @module reasoningbank/e2b-monitor-integration
 */

/**
 * Add ReasoningBank metrics to E2BMonitor
 *
 * @param {E2BMonitor} monitor - E2BMonitor instance
 * @param {SwarmCoordinator} coordinator - SwarmCoordinator with learning
 * @returns {E2BMonitor} Enhanced monitor
 */
function addLearningMetrics(monitor, coordinator) {
  if (!coordinator.reasoningBank) {
    console.warn('⚠️  SwarmCoordinator does not have ReasoningBank enabled');
    return monitor;
  }

  /**
   * Enhanced health report with learning metrics
   */
  const originalGenerateHealthReport = monitor.generateHealthReport.bind(monitor);
  monitor.generateHealthReport = async function() {
    const report = await originalGenerateHealthReport();

    // Add learning metrics
    if (coordinator.reasoningBank?.isInitialized) {
      const learningStats = coordinator.getLearningStats();

      report.learning = {
        enabled: true,
        mode: learningStats.learningMode || 'unknown',
        stats: {
          totalDecisions: learningStats.totalDecisions,
          totalOutcomes: learningStats.totalOutcomes,
          avgVerdictScore: learningStats.avgVerdictScore,
          patternsLearned: learningStats.patternsLearned,
          adaptationEvents: learningStats.adaptationEvents,
          learningRate: learningStats.learningRate,
          currentEpisode: learningStats.currentEpisode,
          totalEpisodes: learningStats.totalEpisodes
        },
        components: {
          trajectoryTracker: {
            total: learningStats.trajectoryStats?.total || 0,
            pending: learningStats.trajectoryStats?.pending || 0,
            completed: learningStats.trajectoryStats?.completed || 0,
            judged: learningStats.trajectoryStats?.judged || 0,
            learned: learningStats.trajectoryStats?.learned || 0
          },
          verdictJudge: {
            totalEvaluations: learningStats.verdictStats?.totalEvaluations || 0,
            avgScore: learningStats.verdictStats?.avgScore || 0,
            qualityDistribution: learningStats.verdictStats?.qualityDistribution || {}
          },
          memoryDistiller: {
            totalDistilled: learningStats.distillerStats?.totalDistilled || 0,
            totalPruned: learningStats.distillerStats?.totalPruned || 0,
            totalPatterns: learningStats.distillerStats?.totalPatterns || 0
          },
          patternRecognizer: {
            totalPatterns: learningStats.recognizerStats?.totalPatterns || 0,
            totalQueries: learningStats.recognizerStats?.totalQueries || 0,
            avgQueryTime: learningStats.recognizerStats?.avgQueryTime || 0,
            cacheHitRate: learningStats.recognizerStats?.cacheHitRate || 0
          }
        },
        timestamp: new Date().toISOString()
      };

      // Add learning-based recommendations
      report.recommendations = report.recommendations || [];

      // Low verdict score warning
      if (learningStats.avgVerdictScore < 0.5 && learningStats.totalOutcomes > 10) {
        report.recommendations.push({
          type: 'learning',
          severity: 'warning',
          message: `Low average verdict score: ${learningStats.avgVerdictScore.toFixed(3)}`,
          action: 'Review trading strategies and adapt agents based on learnings'
        });
      }

      // Low adaptation rate
      if (learningStats.adaptationRate < 0.1 && learningStats.uptime > 300) {
        report.recommendations.push({
          type: 'learning',
          severity: 'info',
          message: 'Low adaptation rate detected',
          action: 'Consider enabling more frequent agent adaptation'
        });
      }

      // Insufficient data warning
      if (learningStats.totalDecisions < 20 && learningStats.uptime > 300) {
        report.recommendations.push({
          type: 'learning',
          severity: 'info',
          message: 'Insufficient trading data for effective learning',
          action: 'Wait for more trading activity or increase trading frequency'
        });
      }

      // Pattern recognition performance
      const avgQueryTime = learningStats.recognizerStats?.avgQueryTime || 0;
      if (avgQueryTime > 100 && learningStats.recognizerStats?.totalQueries > 10) {
        report.recommendations.push({
          type: 'learning',
          severity: 'info',
          message: `High pattern recognition query time: ${avgQueryTime.toFixed(2)}ms`,
          action: 'Consider enabling HNSW indexing or quantization for faster retrieval'
        });
      }

    } else {
      report.learning = {
        enabled: false,
        message: 'ReasoningBank learning not initialized'
      };
    }

    return report;
  };

  /**
   * Monitor learning health
   */
  monitor.checkLearningHealth = function() {
    if (!coordinator.reasoningBank?.isInitialized) {
      return {
        status: 'disabled',
        issues: []
      };
    }

    const stats = coordinator.getLearningStats();
    const issues = [];

    // Check for stale trajectories
    const pendingRate = stats.trajectoryStats?.pending / Math.max(1, stats.trajectoryStats?.total) || 0;
    if (pendingRate > 0.5 && stats.trajectoryStats?.total > 10) {
      issues.push({
        type: 'stale_trajectories',
        severity: 'warning',
        message: `${(pendingRate * 100).toFixed(1)}% of trajectories pending outcomes`
      });
    }

    // Check verdict quality distribution
    const qualityDist = stats.verdictStats?.qualityDistribution || {};
    const terribleRate = qualityDist.terrible / Math.max(1, stats.verdictStats?.totalEvaluations) || 0;
    if (terribleRate > 0.3) {
      issues.push({
        type: 'poor_quality',
        severity: 'critical',
        message: `${(terribleRate * 100).toFixed(1)}% of decisions rated as terrible`
      });
    }

    // Check pattern learning
    if (stats.patternsLearned === 0 && stats.totalOutcomes > 20) {
      issues.push({
        type: 'no_patterns',
        severity: 'warning',
        message: 'No patterns learned despite sufficient data'
      });
    }

    const status = issues.some(i => i.severity === 'critical') ? 'critical' :
                   issues.length > 0 ? 'degraded' : 'healthy';

    return {
      status,
      issues,
      stats: {
        avgVerdictScore: stats.avgVerdictScore,
        patternsLearned: stats.patternsLearned,
        adaptationEvents: stats.adaptationEvents,
        totalDecisions: stats.totalDecisions
      },
      timestamp: new Date().toISOString()
    };
  };

  console.log('✅ E2BMonitor enhanced with ReasoningBank metrics');

  return monitor;
}

module.exports = {
  addLearningMetrics
};
