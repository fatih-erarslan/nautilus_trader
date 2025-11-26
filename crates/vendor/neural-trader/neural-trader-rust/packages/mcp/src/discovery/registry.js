/**
 * Tool Registry
 * Manages tool discovery and metadata
 */

const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class ToolRegistry {
  constructor(options = {}) {
    this.options = {
      toolsDir: path.join(__dirname, '../../tools'),
      cacheEnabled: true,
      ...options,
    };
    this.tools = new Map();
    this.schemaCache = new Map();
    this.etags = new Map();
  }

  /**
   * Load all tool schemas from /tools directory
   */
  async loadTools() {
    const toolsDir = this.options.toolsDir;

    try {
      await fs.access(toolsDir);
    } catch {
      console.error(`Tools directory not found: ${toolsDir}`);
      return;
    }

    const files = await fs.readdir(toolsDir);
    const jsonFiles = files.filter(f => f.endsWith('.json'));

    for (const file of jsonFiles) {
      try {
        const filePath = path.join(toolsDir, file);
        const content = await fs.readFile(filePath, 'utf8');
        const schema = JSON.parse(content);

        const toolName = path.basename(file, '.json');
        this.tools.set(toolName, {
          schema,
          file: filePath,
          etag: this.generateETag(content),
        });

        console.error(`Loaded tool: ${toolName}`);
      } catch (error) {
        console.error(`Error loading tool ${file}:`, error.message);
      }
    }

    console.error(`Loaded ${this.tools.size} tools`);
  }

  /**
   * Generate ETag for content (SHA-256)
   */
  generateETag(content) {
    return crypto
      .createHash('sha256')
      .update(content)
      .digest('hex'); // Full 64-character SHA-256 hash
  }

  /**
   * Get tool schema by name
   */
  getToolSchema(toolName) {
    const tool = this.tools.get(toolName);
    return tool ? tool.schema : null;
  }

  /**
   * Get tool ETag for caching
   */
  getToolETag(toolName) {
    const tool = this.tools.get(toolName);
    return tool ? tool.etag : null;
  }

  /**
   * List all available tools
   */
  listTools() {
    return Array.from(this.tools.keys()).map(name => {
      const tool = this.tools.get(name);
      return {
        name,
        description: tool.schema.description || '',
        category: tool.schema.category || 'general',
        cost: tool.schema.metadata?.cost || 'low',
        latency: tool.schema.metadata?.latency || 'fast',
        etag: tool.etag,
      };
    });
  }

  /**
   * Get tool metadata
   */
  getToolMetadata(toolName) {
    const tool = this.tools.get(toolName);
    if (!tool) {
      return null;
    }

    return {
      name: toolName,
      description: tool.schema.description || '',
      category: tool.schema.category || 'general',
      input_schema: tool.schema.input_schema || {},
      output_schema: tool.schema.output_schema || {},
      metadata: tool.schema.metadata || {},
      etag: tool.etag,
    };
  }

  /**
   * Check if tool exists
   */
  hasTool(toolName) {
    return this.tools.has(toolName);
  }

  /**
   * Get tools by category
   */
  getToolsByCategory(category) {
    // Map common category aliases
    const categoryMap = {
      'Trading': ['trading', 'strategy', 'analysis'],
      'Neural Networks': ['neural', 'ml'],
      'News Trading': ['news'],
      'Portfolio & Risk': ['portfolio', 'risk'],
      'Sports Betting': ['sports'],
      'Prediction Markets': ['prediction'],
      'Syndicates': ['syndicate'],
      'E2B Cloud': ['e2b']
    };

    const matchCategories = categoryMap[category] || [category.toLowerCase()];

    return Array.from(this.tools.entries())
      .filter(([, tool]) => {
        const toolCategory = (tool.schema.category || '').toLowerCase();
        return matchCategories.some(cat => toolCategory.includes(cat));
      })
      .map(([name]) => name);
  }

  /**
   * Search tools by query
   */
  searchTools(query) {
    const lowerQuery = query.toLowerCase();
    return Array.from(this.tools.entries())
      .filter(([name, tool]) => {
        return name.toLowerCase().includes(lowerQuery) ||
               (tool.schema.description || '').toLowerCase().includes(lowerQuery);
      })
      .map(([name]) => name);
  }
}

module.exports = { ToolRegistry };
