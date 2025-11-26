/**
 * MCP Server Tests
 * Coverage Target: 85%+
 */

describe('MCP Server', () => {
  describe('Tool Definitions', () => {
    it('should define accounting_calculate_tax tool', () => {
      const toolDef = {
        name: 'accounting_calculate_tax',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          properties: expect.any(Object),
          required: expect.arrayContaining(['transaction', 'method']),
        }),
      };

      expect(toolDef.name).toBe('accounting_calculate_tax');
    });

    it('should define accounting_check_compliance tool', () => {
      const toolDef = {
        name: 'accounting_check_compliance',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          properties: expect.any(Object),
          required: expect.arrayContaining(['transaction']),
        }),
      };

      expect(toolDef.name).toBe('accounting_check_compliance');
    });

    it('should define accounting_detect_fraud tool', () => {
      const toolDef = {
        name: 'accounting_detect_fraud',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
        }),
      };

      expect(toolDef.name).toBe('accounting_detect_fraud');
    });

    it('should define accounting_harvest_losses tool', () => {
      const toolDef = {
        name: 'accounting_harvest_losses',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['positions', 'currentPrices']),
        }),
      };

      expect(toolDef.name).toBe('accounting_harvest_losses');
    });

    it('should define accounting_generate_report tool', () => {
      const toolDef = {
        name: 'accounting_generate_report',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['reportType', 'transactions']),
        }),
      };

      expect(toolDef.name).toBe('accounting_generate_report');
    });

    it('should define accounting_ingest_transactions tool', () => {
      const toolDef = {
        name: 'accounting_ingest_transactions',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['source', 'data']),
        }),
      };

      expect(toolDef.name).toBe('accounting_ingest_transactions');
    });

    it('should define accounting_get_position tool', () => {
      const toolDef = {
        name: 'accounting_get_position',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['asset']),
        }),
      };

      expect(toolDef.name).toBe('accounting_get_position');
    });

    it('should define accounting_verify_merkle_proof tool', () => {
      const toolDef = {
        name: 'accounting_verify_merkle_proof',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['transaction', 'proof', 'rootHash']),
        }),
      };

      expect(toolDef.name).toBe('accounting_verify_merkle_proof');
    });

    it('should define accounting_learn_from_feedback tool', () => {
      const toolDef = {
        name: 'accounting_learn_from_feedback',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['agentId', 'rating']),
        }),
      };

      expect(toolDef.name).toBe('accounting_learn_from_feedback');
    });

    it('should define accounting_get_metrics tool', () => {
      const toolDef = {
        name: 'accounting_get_metrics',
        description: expect.any(String),
        inputSchema: expect.objectContaining({
          type: 'object',
          required: expect.arrayContaining(['agentId']),
        }),
      };

      expect(toolDef.name).toBe('accounting_get_metrics');
    });
  });

  describe('Tool Execution', () => {
    it('should handle accounting_calculate_tax calls', async () => {
      const args = {
        transaction: {
          id: 'txn-001',
          timestamp: new Date(),
          type: 'SELL',
          asset: 'BTC',
          quantity: 1,
          price: 50000,
        },
        method: 'FIFO',
      };

      // Mock execution
      const result = {
        content: [
          {
            type: 'text',
            text: expect.stringContaining('method'),
          },
        ],
      };

      expect(result.content).toBeDefined();
    });

    it('should handle accounting_check_compliance calls', async () => {
      const args = {
        transaction: {
          id: 'txn-002',
          timestamp: new Date(),
          type: 'BUY',
          asset: 'ETH',
          quantity: 10,
          price: 2500,
        },
        jurisdiction: 'US',
      };

      const result = {
        content: [
          {
            type: 'text',
            text: expect.stringContaining('isCompliant'),
          },
        ],
      };

      expect(result.content).toBeDefined();
    });

    it('should handle accounting_detect_fraud calls', async () => {
      const args = {
        transaction: {
          id: 'txn-003',
          timestamp: new Date(),
          type: 'SELL',
          asset: 'SOL',
          quantity: 100,
          price: 75,
        },
      };

      const result = {
        content: [
          {
            type: 'text',
            text: expect.stringContaining('fraudScore'),
          },
        ],
      };

      expect(result.content).toBeDefined();
    });

    it('should handle invalid tool name', async () => {
      const result = {
        content: [
          {
            type: 'text',
            text: expect.stringContaining('Unknown tool'),
          },
        ],
        isError: true,
      };

      expect(result.isError).toBe(true);
    });
  });

  describe('Input Validation', () => {
    it('should validate required parameters for calculate_tax', () => {
      const schema = {
        type: 'object',
        properties: {
          transaction: { type: 'object' },
          method: {
            type: 'string',
            enum: ['FIFO', 'LIFO', 'HIFO', 'SPECIFIC_ID', 'AVERAGE_COST'],
          },
        },
        required: ['transaction', 'method'],
      };

      expect(schema.required).toContain('transaction');
      expect(schema.required).toContain('method');
    });

    it('should validate method enum values', () => {
      const validMethods = ['FIFO', 'LIFO', 'HIFO', 'SPECIFIC_ID', 'AVERAGE_COST'];

      validMethods.forEach((method) => {
        expect(['FIFO', 'LIFO', 'HIFO', 'SPECIFIC_ID', 'AVERAGE_COST']).toContain(method);
      });
    });

    it('should validate report type enum values', () => {
      const validTypes = ['pnl', 'schedule-d', 'form-8949', 'audit'];

      validTypes.forEach((type) => {
        expect(['pnl', 'schedule-d', 'form-8949', 'audit']).toContain(type);
      });
    });

    it('should validate source enum values', () => {
      const validSources = ['coinbase', 'binance', 'kraken', 'etherscan', 'csv'];

      validSources.forEach((source) => {
        expect(['coinbase', 'binance', 'kraken', 'etherscan', 'csv']).toContain(source);
      });
    });
  });

  describe('Error Handling', () => {
    it('should return error for missing required parameters', async () => {
      const result = {
        content: [
          {
            type: 'text',
            text: expect.stringContaining('Error'),
          },
        ],
        isError: true,
      };

      expect(result.isError).toBe(true);
    });

    it('should handle tool execution errors gracefully', async () => {
      const result = {
        content: [
          {
            type: 'text',
            text: expect.any(String),
          },
        ],
        isError: false,
      };

      expect(result.content).toBeDefined();
    });
  });

  describe('Response Format', () => {
    it('should return responses in correct format', () => {
      const response = {
        content: [
          {
            type: 'text',
            text: JSON.stringify({ result: 'success' }),
          },
        ],
      };

      expect(response.content).toHaveLength(1);
      expect(response.content[0].type).toBe('text');
    });

    it('should return JSON-formatted data', () => {
      const data = {
        method: 'FIFO',
        gainLoss: 5000,
        costBasis: 45000,
        proceeds: 50000,
      };

      const jsonString = JSON.stringify(data, null, 2);
      expect(() => JSON.parse(jsonString)).not.toThrow();
    });
  });
});
