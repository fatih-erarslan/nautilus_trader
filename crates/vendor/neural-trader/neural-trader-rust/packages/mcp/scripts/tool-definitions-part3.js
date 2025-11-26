/**
 * Neural Trader MCP Tool Definitions - Part 3
 * Syndicate Investment and E2B Cloud Sandbox Tools
 */

module.exports = {
  // ============================================================================
  // SYNDICATE INVESTMENT TOOLS (17 tools)
  // ============================================================================

  create_syndicate_tool: {
    title: 'create_syndicate_tool',
    description: 'Create a new investment syndicate for collaborative trading.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string', description: 'Unique syndicate identifier' },
        name: { type: 'string', description: 'Syndicate name' },
        description: { type: 'string', default: '', description: 'Syndicate description' }
      },
      required: ['syndicate_id', 'name']
    },
    output_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        name: { type: 'string' },
        created_at: { type: 'string', format: 'date-time' },
        status: { type: 'string', enum: ['active', 'pending'] },
        member_count: { type: 'integer' },
        total_capital: { type: 'number' }
      },
      required: ['syndicate_id', 'name', 'created_at', 'status']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  add_syndicate_member: {
    title: 'add_syndicate_member',
    description: 'Add a new member to an investment syndicate.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        name: { type: 'string' },
        email: { type: 'string', format: 'email' },
        role: { type: 'string', enum: ['admin', 'manager', 'member', 'observer'] },
        initial_contribution: { type: 'number', minimum: 0 }
      },
      required: ['syndicate_id', 'name', 'email', 'role', 'initial_contribution']
    },
    output_schema: {
      type: 'object',
      properties: {
        member_id: { type: 'string' },
        syndicate_id: { type: 'string' },
        name: { type: 'string' },
        role: { type: 'string' },
        capital_contributed: { type: 'number' },
        equity_share: { type: 'number' },
        joined_at: { type: 'string', format: 'date-time' }
      },
      required: ['member_id', 'syndicate_id', 'capital_contributed']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_syndicate_status_tool: {
    title: 'get_syndicate_status_tool',
    description: 'Get current status and statistics for a syndicate.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' }
      },
      required: ['syndicate_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        name: { type: 'string' },
        status: { type: 'string' },
        member_count: { type: 'integer' },
        total_capital: { type: 'number' },
        available_capital: { type: 'number' },
        deployed_capital: { type: 'number' },
        total_returns: { type: 'number' },
        roi: { type: 'number' },
        active_positions: { type: 'integer' },
        performance_metrics: {
          type: 'object',
          properties: {
            sharpe_ratio: { type: 'number' },
            win_rate: { type: 'number' },
            total_bets: { type: 'integer' }
          }
        }
      },
      required: ['syndicate_id', 'total_capital', 'roi']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  allocate_syndicate_funds: {
    title: 'allocate_syndicate_funds',
    description: 'Allocate syndicate funds across betting opportunities.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        opportunities: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              market_id: { type: 'string' },
              odds: { type: 'number' },
              probability: { type: 'number' },
              max_stake: { type: 'number' }
            },
            required: ['market_id', 'odds', 'probability'],
            additionalProperties: true
          }
        },
        strategy: { type: 'string', default: 'kelly_criterion', enum: ['kelly_criterion', 'equal_weight', 'proportional', 'risk_adjusted'] }
      },
      required: ['syndicate_id', 'opportunities']
    },
    output_schema: {
      type: 'object',
      properties: {
        allocation_id: { type: 'string' },
        syndicate_id: { type: 'string' },
        strategy: { type: 'string' },
        allocations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              market_id: { type: 'string' },
              allocated_amount: { type: 'number' },
              expected_value: { type: 'number' },
              risk_score: { type: 'number' }
            }
          }
        },
        total_allocated: { type: 'number' },
        expected_return: { type: 'number' },
        risk_metrics: {
          type: 'object',
          properties: {
            portfolio_var: { type: 'number' },
            kelly_fraction: { type: 'number' }
          }
        }
      },
      required: ['allocation_id', 'allocations', 'total_allocated']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  distribute_syndicate_profits: {
    title: 'distribute_syndicate_profits',
    description: 'Distribute profits among syndicate members.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        total_profit: { type: 'number' },
        model: { type: 'string', default: 'hybrid', enum: ['proportional', 'equal', 'hybrid', 'performance_based'] }
      },
      required: ['syndicate_id', 'total_profit']
    },
    output_schema: {
      type: 'object',
      properties: {
        distribution_id: { type: 'string' },
        syndicate_id: { type: 'string' },
        total_profit: { type: 'number' },
        model: { type: 'string' },
        distributions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              member_id: { type: 'string' },
              member_name: { type: 'string' },
              share: { type: 'number' },
              amount: { type: 'number' }
            }
          }
        },
        timestamp: { type: 'string', format: 'date-time' }
      },
      required: ['distribution_id', 'distributions', 'total_profit']
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  },

  process_syndicate_withdrawal: {
    title: 'process_syndicate_withdrawal',
    description: 'Process a member withdrawal request.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        member_id: { type: 'string' },
        amount: { type: 'number', minimum: 0 },
        is_emergency: { type: 'boolean', default: false }
      },
      required: ['syndicate_id', 'member_id', 'amount']
    },
    output_schema: {
      type: 'object',
      properties: {
        withdrawal_id: { type: 'string' },
        status: { type: 'string', enum: ['approved', 'pending', 'rejected'] },
        amount: { type: 'number' },
        fee: { type: 'number' },
        net_amount: { type: 'number' },
        reason: { type: 'string' },
        processed_at: { type: 'string', format: 'date-time' }
      },
      required: ['withdrawal_id', 'status', 'amount']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_syndicate_member_performance: {
    title: 'get_syndicate_member_performance',
    description: 'Get detailed performance metrics for a member.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        member_id: { type: 'string' }
      },
      required: ['syndicate_id', 'member_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        member_id: { type: 'string' },
        member_name: { type: 'string' },
        capital_contributed: { type: 'number' },
        current_value: { type: 'number' },
        total_returns: { type: 'number' },
        roi: { type: 'number' },
        equity_share: { type: 'number' },
        performance_rank: { type: 'integer' }
      },
      required: ['member_id', 'capital_contributed', 'roi']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  create_syndicate_vote: {
    title: 'create_syndicate_vote',
    description: 'Create a new vote for syndicate decisions.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        vote_type: { type: 'string', enum: ['strategy', 'withdrawal', 'member', 'allocation', 'other'] },
        proposal: { type: 'string' },
        options: { type: 'array', items: { type: 'string' }, minItems: 2 },
        duration_hours: { type: 'integer', default: 48, minimum: 1, maximum: 168 }
      },
      required: ['syndicate_id', 'vote_type', 'proposal', 'options']
    },
    output_schema: {
      type: 'object',
      properties: {
        vote_id: { type: 'string' },
        syndicate_id: { type: 'string' },
        proposal: { type: 'string' },
        status: { type: 'string', enum: ['open', 'closed'] },
        expires_at: { type: 'string', format: 'date-time' },
        options: { type: 'array', items: { type: 'string' } }
      },
      required: ['vote_id', 'proposal', 'status']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  cast_syndicate_vote: {
    title: 'cast_syndicate_vote',
    description: 'Cast a vote on a syndicate proposal.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        vote_id: { type: 'string' },
        member_id: { type: 'string' },
        option: { type: 'string' }
      },
      required: ['syndicate_id', 'vote_id', 'member_id', 'option']
    },
    output_schema: {
      type: 'object',
      properties: {
        vote_id: { type: 'string' },
        member_id: { type: 'string' },
        option: { type: 'string' },
        weight: { type: 'number' },
        timestamp: { type: 'string', format: 'date-time' },
        current_results: {
          type: 'object',
          additionalProperties: { type: 'number' }
        }
      },
      required: ['vote_id', 'option', 'timestamp']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_syndicate_allocation_limits: {
    title: 'get_syndicate_allocation_limits',
    description: 'Get allocation limits and risk constraints.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' }
      },
      required: ['syndicate_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        max_single_bet: { type: 'number' },
        max_total_exposure: { type: 'number' },
        max_correlated_exposure: { type: 'number' },
        min_bankroll_reserve: { type: 'number' },
        kelly_multiplier: { type: 'number' }
      },
      required: ['syndicate_id', 'max_single_bet', 'max_total_exposure']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  update_syndicate_member_contribution: {
    title: 'update_syndicate_member_contribution',
    description: "Update a member's capital contribution.",
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        member_id: { type: 'string' },
        additional_amount: { type: 'number' }
      },
      required: ['syndicate_id', 'member_id', 'additional_amount']
    },
    output_schema: {
      type: 'object',
      properties: {
        member_id: { type: 'string' },
        previous_contribution: { type: 'number' },
        additional_contribution: { type: 'number' },
        total_contribution: { type: 'number' },
        new_equity_share: { type: 'number' }
      },
      required: ['member_id', 'total_contribution', 'new_equity_share']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_syndicate_profit_history: {
    title: 'get_syndicate_profit_history',
    description: 'Get profit distribution history.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        days: { type: 'integer', default: 30, minimum: 1, maximum: 365 }
      },
      required: ['syndicate_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        distributions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              distribution_id: { type: 'string' },
              date: { type: 'string', format: 'date' },
              total_profit: { type: 'number' },
              member_count: { type: 'integer' }
            }
          }
        },
        total_distributed: { type: 'number' },
        average_per_distribution: { type: 'number' }
      },
      required: ['syndicate_id', 'distributions']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  simulate_syndicate_allocation: {
    title: 'simulate_syndicate_allocation',
    description: 'Simulate fund allocation across strategies.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        opportunities: { type: 'array', items: { type: 'object', additionalProperties: true } },
        test_strategies: { type: 'array', items: { type: 'string' } }
      },
      required: ['syndicate_id', 'opportunities']
    },
    output_schema: {
      type: 'object',
      properties: {
        simulation_id: { type: 'string' },
        results: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              strategy: { type: 'string' },
              expected_return: { type: 'number' },
              risk_score: { type: 'number' },
              sharpe_ratio: { type: 'number' }
            }
          }
        },
        recommended_strategy: { type: 'string' }
      },
      required: ['simulation_id', 'results']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_syndicate_withdrawal_history: {
    title: 'get_syndicate_withdrawal_history',
    description: 'Get withdrawal history for syndicate or member.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        member_id: { type: 'string', description: 'Optional: filter by member' }
      },
      required: ['syndicate_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        withdrawals: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              withdrawal_id: { type: 'string' },
              member_id: { type: 'string' },
              amount: { type: 'number' },
              status: { type: 'string' },
              date: { type: 'string', format: 'date-time' }
            }
          }
        },
        total_withdrawn: { type: 'number' }
      },
      required: ['withdrawals']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  update_syndicate_allocation_strategy: {
    title: 'update_syndicate_allocation_strategy',
    description: 'Update allocation strategy parameters.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        strategy_config: {
          type: 'object',
          additionalProperties: true,
          description: 'Strategy configuration parameters'
        }
      },
      required: ['syndicate_id', 'strategy_config']
    },
    output_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        updated_config: { type: 'object' },
        effective_date: { type: 'string', format: 'date-time' }
      },
      required: ['syndicate_id', 'updated_config']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_syndicate_member_list: {
    title: 'get_syndicate_member_list',
    description: 'Get list of all syndicate members.',
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        active_only: { type: 'boolean', default: true }
      },
      required: ['syndicate_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        members: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              member_id: { type: 'string' },
              name: { type: 'string' },
              role: { type: 'string' },
              equity_share: { type: 'number' },
              status: { type: 'string' }
            }
          }
        },
        total_members: { type: 'integer' }
      },
      required: ['members', 'total_members']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  calculate_syndicate_tax_liability: {
    title: 'calculate_syndicate_tax_liability',
    description: "Calculate estimated tax liability for member's earnings.",
    category: 'syndicate',
    input_schema: {
      type: 'object',
      properties: {
        syndicate_id: { type: 'string' },
        member_id: { type: 'string' },
        jurisdiction: { type: 'string', default: 'US' }
      },
      required: ['syndicate_id', 'member_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        member_id: { type: 'string' },
        gross_earnings: { type: 'number' },
        estimated_tax: { type: 'number' },
        net_earnings: { type: 'number' },
        tax_rate: { type: 'number' },
        jurisdiction: { type: 'string' }
      },
      required: ['member_id', 'gross_earnings', 'estimated_tax']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  // ============================================================================
  // E2B CLOUD SANDBOX TOOLS (10 tools)
  // ============================================================================

  create_e2b_sandbox: {
    title: 'create_e2b_sandbox',
    description: 'Create a new E2B sandbox for isolated execution.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Sandbox name' },
        template: { type: 'string', default: 'base', enum: ['base', 'python', 'nodejs', 'trading'] },
        memory_mb: { type: 'integer', default: 512, minimum: 256, maximum: 4096 },
        cpu_count: { type: 'integer', default: 1, minimum: 1, maximum: 4 },
        timeout: { type: 'integer', default: 300, description: 'Timeout in seconds' }
      },
      required: ['name']
    },
    output_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        name: { type: 'string' },
        status: { type: 'string', enum: ['creating', 'running', 'stopped'] },
        template: { type: 'string' },
        resources: {
          type: 'object',
          properties: {
            memory_mb: { type: 'integer' },
            cpu_count: { type: 'integer' }
          }
        },
        created_at: { type: 'string', format: 'date-time' },
        endpoint: { type: 'string' }
      },
      required: ['sandbox_id', 'name', 'status']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: false
    }
  },

  run_e2b_agent: {
    title: 'run_e2b_agent',
    description: 'Run a trading agent in E2B sandbox.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        agent_type: { type: 'string', enum: ['trader', 'analyzer', 'monitor', 'optimizer'] },
        symbols: { type: 'array', items: { type: 'string' } },
        strategy_params: { type: 'object', additionalProperties: true },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['sandbox_id', 'agent_type', 'symbols']
    },
    output_schema: {
      type: 'object',
      properties: {
        execution_id: { type: 'string' },
        sandbox_id: { type: 'string' },
        agent_type: { type: 'string' },
        status: { type: 'string', enum: ['running', 'completed', 'failed'] },
        results: { type: 'object' },
        execution_time: { type: 'string' }
      },
      required: ['execution_id', 'status']
    },
    metadata: {
      cost: 'high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  execute_e2b_process: {
    title: 'execute_e2b_process',
    description: 'Execute a process in E2B sandbox.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        command: { type: 'string' },
        args: { type: 'array', items: { type: 'string' } },
        capture_output: { type: 'boolean', default: true },
        timeout: { type: 'integer', default: 60 }
      },
      required: ['sandbox_id', 'command']
    },
    output_schema: {
      type: 'object',
      properties: {
        process_id: { type: 'string' },
        exit_code: { type: 'integer' },
        stdout: { type: 'string' },
        stderr: { type: 'string' },
        execution_time: { type: 'number' }
      },
      required: ['process_id', 'exit_code']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  list_e2b_sandboxes: {
    title: 'list_e2b_sandboxes',
    description: 'List all E2B sandboxes with optional filtering.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        status_filter: { type: 'string', enum: ['running', 'stopped', 'all'] }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        sandboxes: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              sandbox_id: { type: 'string' },
              name: { type: 'string' },
              status: { type: 'string' },
              template: { type: 'string' },
              created_at: { type: 'string' }
            }
          }
        },
        total_count: { type: 'integer' }
      },
      required: ['sandboxes', 'total_count']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  terminate_e2b_sandbox: {
    title: 'terminate_e2b_sandbox',
    description: 'Terminate E2B sandbox and cleanup resources.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        force: { type: 'boolean', default: false }
      },
      required: ['sandbox_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        status: { type: 'string', enum: ['terminated', 'terminating', 'failed'] },
        message: { type: 'string' },
        terminated_at: { type: 'string', format: 'date-time' }
      },
      required: ['sandbox_id', 'status']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_e2b_sandbox_status: {
    title: 'get_e2b_sandbox_status',
    description: 'Get detailed status and metrics for sandbox.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' }
      },
      required: ['sandbox_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        status: { type: 'string' },
        uptime: { type: 'number' },
        resource_usage: {
          type: 'object',
          properties: {
            cpu_percent: { type: 'number' },
            memory_mb: { type: 'number' },
            disk_mb: { type: 'number' }
          }
        },
        processes_running: { type: 'integer' }
      },
      required: ['sandbox_id', 'status']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  deploy_e2b_template: {
    title: 'deploy_e2b_template',
    description: 'Deploy a pre-configured E2B template.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        template_name: { type: 'string' },
        category: { type: 'string', enum: ['trading', 'analytics', 'ml', 'custom'] },
        configuration: { type: 'object', additionalProperties: true }
      },
      required: ['template_name', 'category', 'configuration']
    },
    output_schema: {
      type: 'object',
      properties: {
        deployment_id: { type: 'string' },
        sandbox_id: { type: 'string' },
        template: { type: 'string' },
        status: { type: 'string' }
      },
      required: ['deployment_id', 'sandbox_id']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: false
    }
  },

  scale_e2b_deployment: {
    title: 'scale_e2b_deployment',
    description: 'Scale E2B deployment to multiple instances.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        deployment_id: { type: 'string' },
        instance_count: { type: 'integer', minimum: 1, maximum: 10 },
        auto_scale: { type: 'boolean', default: false }
      },
      required: ['deployment_id', 'instance_count']
    },
    output_schema: {
      type: 'object',
      properties: {
        deployment_id: { type: 'string' },
        instances: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              sandbox_id: { type: 'string' },
              status: { type: 'string' }
            }
          }
        },
        total_instances: { type: 'integer' }
      },
      required: ['deployment_id', 'instances']
    },
    metadata: {
      cost: 'very_high',
      latency: 'slow',
      gpu_capable: false
    }
  },

  monitor_e2b_health: {
    title: 'monitor_e2b_health',
    description: 'Monitor health and performance of E2B infrastructure.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        include_all_sandboxes: { type: 'boolean', default: false }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        overall_health: { type: 'string', enum: ['healthy', 'degraded', 'unhealthy'] },
        total_sandboxes: { type: 'integer' },
        running_sandboxes: { type: 'integer' },
        average_cpu: { type: 'number' },
        average_memory: { type: 'number' },
        issues: { type: 'array', items: { type: 'string' } }
      },
      required: ['overall_health', 'total_sandboxes']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  export_e2b_template: {
    title: 'export_e2b_template',
    description: 'Export sandbox configuration as reusable template.',
    category: 'e2b',
    input_schema: {
      type: 'object',
      properties: {
        sandbox_id: { type: 'string' },
        template_name: { type: 'string' },
        include_data: { type: 'boolean', default: false }
      },
      required: ['sandbox_id', 'template_name']
    },
    output_schema: {
      type: 'object',
      properties: {
        template_id: { type: 'string' },
        template_name: { type: 'string' },
        exported_at: { type: 'string', format: 'date-time' },
        size_mb: { type: 'number' }
      },
      required: ['template_id', 'template_name']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  // ============================================================================
  // E2B TRADING SWARM TOOLS (8 tools)
  // ============================================================================

  init_e2b_swarm: {
    title: 'init_e2b_swarm',
    description: 'Initialize E2B trading swarm with specified topology and configuration. Creates a distributed trading system with multiple coordinated agents.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        topology: {
          type: 'string',
          enum: ['mesh', 'hierarchical', 'ring', 'star'],
          description: 'Network topology for swarm coordination'
        },
        maxAgents: { type: 'number', default: 5, minimum: 1, maximum: 50 },
        strategy: { type: 'string', default: 'balanced', enum: ['balanced', 'aggressive', 'conservative', 'adaptive'] },
        sharedMemory: { type: 'boolean', default: true },
        autoScale: { type: 'boolean', default: false }
      },
      required: ['topology']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        topology: { type: 'string' },
        max_agents: { type: 'integer' },
        status: { type: 'string', enum: ['initializing', 'active', 'error'] },
        created_at: { type: 'string', format: 'date-time' }
      },
      required: ['swarm_id', 'topology', 'status']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  deploy_trading_agent: {
    title: 'deploy_trading_agent',
    description: 'Deploy a specialized trading agent to the E2B swarm with specific role and capabilities.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string', description: 'Unique identifier of the swarm' },
        agent_type: {
          type: 'string',
          enum: ['market_maker', 'trend_follower', 'arbitrage', 'risk_manager', 'coordinator']
        },
        symbols: { type: 'array', items: { type: 'string' } },
        strategy_params: { type: 'object', additionalProperties: true },
        resources: {
          type: 'object',
          properties: {
            memory_mb: { type: 'integer', default: 512 },
            cpu_count: { type: 'integer', default: 1 },
            gpu_enabled: { type: 'boolean', default: false }
          }
        }
      },
      required: ['swarm_id', 'agent_type', 'symbols']
    },
    output_schema: {
      type: 'object',
      properties: {
        agent_id: { type: 'string' },
        swarm_id: { type: 'string' },
        agent_type: { type: 'string' },
        sandbox_id: { type: 'string' },
        status: { type: 'string', enum: ['deploying', 'running', 'error'] },
        deployed_at: { type: 'string', format: 'date-time' }
      },
      required: ['agent_id', 'swarm_id', 'status']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  get_swarm_status: {
    title: 'get_swarm_status',
    description: 'Get comprehensive status and health metrics for an E2B trading swarm.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        include_metrics: { type: 'boolean', default: true },
        include_agents: { type: 'boolean', default: true }
      },
      required: ['swarm_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        status: { type: 'string', enum: ['active', 'degraded', 'stopped'] },
        topology: { type: 'string' },
        agent_count: { type: 'integer' },
        active_agents: { type: 'integer' },
        total_trades: { type: 'integer' },
        uptime_seconds: { type: 'number' },
        agents: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              agent_id: { type: 'string' },
              agent_type: { type: 'string' },
              status: { type: 'string' },
              trades: { type: 'integer' },
              pnl: { type: 'number' }
            }
          }
        },
        metrics: {
          type: 'object',
          properties: {
            avg_latency_ms: { type: 'number' },
            success_rate: { type: 'number' },
            total_pnl: { type: 'number' }
          }
        }
      },
      required: ['swarm_id', 'status', 'agent_count']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  scale_swarm: {
    title: 'scale_swarm',
    description: 'Scale E2B trading swarm by adding or removing agents based on demand.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        target_agents: { type: 'integer', minimum: 1, maximum: 50 },
        scale_mode: { type: 'string', enum: ['immediate', 'gradual', 'adaptive'], default: 'gradual' },
        preserve_state: { type: 'boolean', default: true }
      },
      required: ['swarm_id', 'target_agents']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        previous_agents: { type: 'integer' },
        target_agents: { type: 'integer' },
        current_agents: { type: 'integer' },
        status: { type: 'string', enum: ['scaling', 'completed', 'failed'] },
        estimated_completion: { type: 'string', format: 'date-time' }
      },
      required: ['swarm_id', 'status', 'target_agents']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  execute_swarm_strategy: {
    title: 'execute_swarm_strategy',
    description: 'Execute a coordinated trading strategy across all agents in the swarm.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        strategy: { type: 'string' },
        parameters: { type: 'object', additionalProperties: true },
        coordination: { type: 'string', enum: ['parallel', 'sequential', 'adaptive'], default: 'parallel' },
        timeout: { type: 'integer', default: 300 }
      },
      required: ['swarm_id', 'strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        execution_id: { type: 'string' },
        swarm_id: { type: 'string' },
        strategy: { type: 'string' },
        status: { type: 'string', enum: ['executing', 'completed', 'failed', 'timeout'] },
        agents_executed: { type: 'integer' },
        total_trades: { type: 'integer' },
        total_pnl: { type: 'number' },
        started_at: { type: 'string', format: 'date-time' },
        completed_at: { type: 'string', format: 'date-time' }
      },
      required: ['execution_id', 'swarm_id', 'status']
    },
    metadata: {
      cost: 'high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  monitor_swarm_health: {
    title: 'monitor_swarm_health',
    description: 'Monitor E2B swarm health with real-time metrics and alerts.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        interval: { type: 'integer', default: 60, minimum: 10 },
        alerts: {
          type: 'object',
          properties: {
            failure_threshold: { type: 'number', default: 0.2 },
            latency_threshold: { type: 'number', default: 1000 },
            error_rate_threshold: { type: 'number', default: 0.05 }
          }
        },
        include_system_metrics: { type: 'boolean', default: true }
      },
      required: ['swarm_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        health_status: { type: 'string', enum: ['healthy', 'degraded', 'critical'] },
        timestamp: { type: 'string', format: 'date-time' },
        metrics: {
          type: 'object',
          properties: {
            cpu_usage: { type: 'number' },
            memory_usage: { type: 'number' },
            network_latency_ms: { type: 'number' },
            error_rate: { type: 'number' },
            uptime_seconds: { type: 'number' }
          }
        },
        alerts: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              severity: { type: 'string' },
              message: { type: 'string' },
              timestamp: { type: 'string', format: 'date-time' }
            }
          }
        },
        agent_health: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              agent_id: { type: 'string' },
              status: { type: 'string' },
              last_heartbeat: { type: 'string', format: 'date-time' }
            }
          }
        }
      },
      required: ['swarm_id', 'health_status', 'timestamp']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_swarm_metrics: {
    title: 'get_swarm_metrics',
    description: 'Get detailed performance metrics for E2B trading swarm.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        time_range: { type: 'string', enum: ['1h', '6h', '24h', '7d', '30d'], default: '24h' },
        metrics: {
          type: 'array',
          items: { type: 'string', enum: ['latency', 'throughput', 'error_rate', 'success_rate', 'pnl', 'trades', 'all'] },
          default: ['all']
        },
        aggregation: { type: 'string', enum: ['avg', 'min', 'max', 'sum', 'p50', 'p95', 'p99'], default: 'avg' }
      },
      required: ['swarm_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        time_range: { type: 'string' },
        metrics: {
          type: 'object',
          properties: {
            latency_ms: { type: 'number' },
            throughput_tps: { type: 'number' },
            error_rate: { type: 'number' },
            success_rate: { type: 'number' },
            total_pnl: { type: 'number' },
            total_trades: { type: 'integer' },
            avg_trade_size: { type: 'number' },
            win_rate: { type: 'number' }
          }
        },
        per_agent_metrics: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              agent_id: { type: 'string' },
              trades: { type: 'integer' },
              pnl: { type: 'number' },
              success_rate: { type: 'number' }
            }
          }
        }
      },
      required: ['swarm_id', 'time_range', 'metrics']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  shutdown_swarm: {
    title: 'shutdown_swarm',
    description: 'Gracefully shutdown E2B trading swarm and cleanup resources.',
    category: 'e2b_swarm',
    input_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        grace_period: { type: 'integer', default: 60 },
        save_state: { type: 'boolean', default: true },
        force: { type: 'boolean', default: false }
      },
      required: ['swarm_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        swarm_id: { type: 'string' },
        status: { type: 'string', enum: ['shutting_down', 'stopped', 'error'] },
        agents_stopped: { type: 'integer' },
        state_saved: { type: 'boolean' },
        shutdown_at: { type: 'string', format: 'date-time' },
        final_metrics: {
          type: 'object',
          properties: {
            total_runtime_seconds: { type: 'number' },
            total_trades: { type: 'integer' },
            total_pnl: { type: 'number' }
          }
        }
      },
      required: ['swarm_id', 'status', 'agents_stopped']
    },
    metadata: {
      cost: 'low',
      latency: 'medium',
      gpu_capable: false
    }
  }
};
