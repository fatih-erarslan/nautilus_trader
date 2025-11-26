/**
 * E2B Template Manager Helper
 *
 * Manages E2B template deployments, configurations, and operations
 */

class E2BTemplateManager {
  constructor(credentials) {
    this.credentials = credentials;
    this.templates = new Map();
    this.deploymentHistory = [];
  }

  /**
   * Get template configuration
   * @param {string} templateName - Template name
   * @returns {Object} Template configuration
   */
  getTemplateConfig(templateName) {
    const configs = {
      base: {
        name: 'base',
        description: 'Base Ubuntu environment',
        features: ['nodejs', 'python3', 'git'],
        estimatedBootTime: 5000,
      },
      nodejs: {
        name: 'nodejs',
        description: 'Node.js development environment',
        features: ['nodejs', 'npm', 'yarn'],
        estimatedBootTime: 7000,
      },
      python: {
        name: 'python',
        description: 'Python development environment',
        features: ['python3', 'pip', 'virtualenv'],
        estimatedBootTime: 8000,
      },
      react: {
        name: 'react',
        description: 'React development environment',
        features: ['nodejs', 'npm', 'react', 'webpack'],
        estimatedBootTime: 10000,
      },
      rust: {
        name: 'rust',
        description: 'Rust development environment',
        features: ['rustc', 'cargo'],
        estimatedBootTime: 12000,
      },
      golang: {
        name: 'golang',
        description: 'Go development environment',
        features: ['go', 'gofmt'],
        estimatedBootTime: 9000,
      },
    };

    return configs[templateName] || configs.base;
  }

  /**
   * Register deployed template
   * @param {Object} sandbox - Sandbox information
   */
  registerTemplate(sandbox) {
    this.templates.set(sandbox.sandboxId, {
      ...sandbox,
      deployedAt: new Date().toISOString(),
    });

    this.deploymentHistory.push({
      sandboxId: sandbox.sandboxId,
      template: sandbox.template,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Get all deployed templates
   * @returns {Array} List of deployed templates
   */
  getDeployedTemplates() {
    return Array.from(this.templates.values());
  }

  /**
   * Get templates by type
   * @param {string} templateType - Template type
   * @returns {Array} Matching templates
   */
  getTemplatesByType(templateType) {
    return this.getDeployedTemplates().filter(t => t.template === templateType);
  }

  /**
   * Get deployment statistics
   * @returns {Object} Deployment statistics
   */
  getDeploymentStats() {
    const stats = {
      total: this.deploymentHistory.length,
      byTemplate: {},
    };

    this.deploymentHistory.forEach(deployment => {
      const template = deployment.template;
      stats.byTemplate[template] = (stats.byTemplate[template] || 0) + 1;
    });

    return stats;
  }

  /**
   * Validate template configuration
   * @param {string} templateName - Template name
   * @returns {boolean} True if valid
   */
  validateTemplate(templateName) {
    const validTemplates = ['base', 'nodejs', 'python', 'react', 'rust', 'golang'];
    return validTemplates.includes(templateName);
  }

  /**
   * Get recommended template for use case
   * @param {string} useCase - Use case description
   * @returns {string} Recommended template
   */
  getRecommendedTemplate(useCase) {
    const useCaseMap = {
      'trading': 'nodejs',
      'ml': 'python',
      'analysis': 'python',
      'web': 'react',
      'api': 'nodejs',
      'data': 'python',
      'blockchain': 'rust',
      'backend': 'golang',
    };

    const normalized = useCase.toLowerCase();

    for (const [key, template] of Object.entries(useCaseMap)) {
      if (normalized.includes(key)) {
        return template;
      }
    }

    return 'base';
  }

  /**
   * Clear deployment history
   */
  clearHistory() {
    this.deploymentHistory = [];
    this.templates.clear();
  }
}

module.exports = { E2BTemplateManager };
