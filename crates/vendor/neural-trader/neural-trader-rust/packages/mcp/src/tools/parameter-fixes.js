/**
 * Parameter Type Fixes for MCP Tools
 *
 * This module provides parameter validation and type coercion to fix
 * the 7 tools that are failing due to type mismatches.
 *
 * @module tools/parameter-fixes
 */

/**
 * Validate and coerce parameters for backtest tools
 */
function validateBacktestParams(params) {
  const {
    strategy,
    symbol,
    start_date,
    end_date,
    use_gpu,
    benchmark,
    include_costs,
  } = params;

  // Type coercions
  return {
    strategy: String(strategy),
    symbol: String(symbol),
    start_date: String(start_date),
    end_date: String(end_date),
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
    benchmark: benchmark ? String(benchmark) : 'sp500',
    include_costs: include_costs === undefined ? true : (include_costs === 'true' || include_costs === true),
  };
}

/**
 * Validate and coerce parameters for optimization tools
 */
function validateOptimizeParams(params) {
  const {
    strategy,
    symbol,
    parameter_ranges,
    use_gpu,
    max_iterations,
    optimization_metric,
  } = params;

  // Ensure parameter_ranges is an object
  let ranges = parameter_ranges;
  if (typeof ranges === 'string') {
    try {
      ranges = JSON.parse(ranges);
    } catch (e) {
      throw new Error(`Invalid parameter_ranges: must be a JSON object, got: ${ranges}`);
    }
  }
  if (typeof ranges !== 'object' || ranges === null) {
    throw new Error(`Invalid parameter_ranges: must be an object, got: ${typeof ranges}`);
  }

  return {
    strategy: String(strategy),
    symbol: String(symbol),
    parameter_ranges: ranges,
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
    max_iterations: max_iterations ? parseInt(max_iterations, 10) : 1000,
    optimization_metric: optimization_metric ? String(optimization_metric) : 'sharpe_ratio',
  };
}

/**
 * Validate and coerce parameters for neural forecast tools
 */
function validateNeuralForecastParams(params) {
  const {
    symbol,
    horizon,
    model_id,
    use_gpu,
    confidence_level,
  } = params;

  return {
    symbol: String(symbol),
    horizon: parseInt(horizon, 10),
    model_id: model_id || null,
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
    confidence_level: confidence_level ? parseFloat(confidence_level) : 0.95,
  };
}

/**
 * Validate and coerce parameters for neural training tools
 */
function validateNeuralTrainParams(params) {
  const {
    data_path,
    model_type,
    epochs,
    batch_size,
    learning_rate,
    use_gpu,
    validation_split,
  } = params;

  return {
    data_path: String(data_path),
    model_type: String(model_type),
    epochs: epochs ? parseInt(epochs, 10) : 100,
    batch_size: batch_size ? parseInt(batch_size, 10) : 32,
    learning_rate: learning_rate ? parseFloat(learning_rate) : 0.001,
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
    validation_split: validation_split ? parseFloat(validation_split) : 0.2,
  };
}

/**
 * Validate and coerce parameters for risk analysis tools
 */
function validateRiskAnalysisParams(params) {
  const {
    portfolio,
    use_gpu,
    use_monte_carlo,
    var_confidence,
    time_horizon,
  } = params;

  // Ensure portfolio is an array
  let portfolioArray = portfolio;
  if (typeof portfolioArray === 'string') {
    try {
      portfolioArray = JSON.parse(portfolioArray);
    } catch (e) {
      throw new Error(`Invalid portfolio: must be a JSON array, got: ${portfolioArray}`);
    }
  }
  if (!Array.isArray(portfolioArray)) {
    throw new Error(`Invalid portfolio: must be an array, got: ${typeof portfolioArray}`);
  }

  return {
    portfolio: portfolioArray,
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
    use_monte_carlo: use_monte_carlo === undefined ? true : (use_monte_carlo === 'true' || use_monte_carlo === true),
    var_confidence: var_confidence ? parseFloat(var_confidence) : 0.05,
    time_horizon: time_horizon ? parseInt(time_horizon, 10) : 1,
  };
}

/**
 * Validate and coerce parameters for correlation analysis tools
 */
function validateCorrelationParams(params) {
  const {
    symbols,
    period_days,
    use_gpu,
  } = params;

  // Ensure symbols is an array
  let symbolsArray = symbols;
  if (typeof symbolsArray === 'string') {
    // Handle comma-separated string
    if (symbolsArray.includes(',')) {
      symbolsArray = symbolsArray.split(',').map(s => s.trim());
    } else {
      try {
        symbolsArray = JSON.parse(symbolsArray);
      } catch (e) {
        throw new Error(`Invalid symbols: must be a JSON array or comma-separated string, got: ${symbolsArray}`);
      }
    }
  }
  if (!Array.isArray(symbolsArray)) {
    throw new Error(`Invalid symbols: must be an array, got: ${typeof symbolsArray}`);
  }

  return {
    symbols: symbolsArray,
    period_days: period_days ? parseInt(period_days, 10) : 90,
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
  };
}

/**
 * Validate and coerce parameters for news analysis tools
 */
function validateNewsAnalysisParams(params) {
  const {
    symbol,
    lookback_hours,
    sentiment_model,
    use_gpu,
  } = params;

  return {
    symbol: String(symbol),
    lookback_hours: lookback_hours ? parseInt(lookback_hours, 10) : 24,
    sentiment_model: sentiment_model ? String(sentiment_model) : 'enhanced',
    use_gpu: use_gpu === 'true' || use_gpu === true || use_gpu === 1,
  };
}

/**
 * Main parameter validation router
 * @param {string} toolName - Name of the MCP tool
 * @param {Object} params - Raw parameters
 * @returns {Object} Validated and coerced parameters
 */
function validateToolParameters(toolName, params) {
  try {
    // Route to appropriate validator based on tool name
    if (toolName === 'run_backtest') {
      return validateBacktestParams(params);
    }
    if (toolName === 'optimize_strategy') {
      return validateOptimizeParams(params);
    }
    if (toolName === 'neural_forecast') {
      return validateNeuralForecastParams(params);
    }
    if (toolName === 'neural_train') {
      return validateNeuralTrainParams(params);
    }
    if (toolName === 'risk_analysis') {
      return validateRiskAnalysisParams(params);
    }
    if (toolName === 'correlation_analysis') {
      return validateCorrelationParams(params);
    }
    if (toolName === 'analyze_news') {
      return validateNewsAnalysisParams(params);
    }

    // No validation needed for this tool
    return params;
  } catch (error) {
    throw new Error(`Parameter validation failed for ${toolName}: ${error.message}`);
  }
}

/**
 * Wrap a tool handler with parameter validation
 * @param {string} toolName - Tool name
 * @param {Function} handler - Original handler
 * @returns {Function} Wrapped handler with validation
 */
function withParameterValidation(toolName, handler) {
  return async (params, context) => {
    const validatedParams = validateToolParameters(toolName, params);
    return handler(validatedParams, context);
  };
}

module.exports = {
  validateToolParameters,
  withParameterValidation,
  validateBacktestParams,
  validateOptimizeParams,
  validateNeuralForecastParams,
  validateNeuralTrainParams,
  validateRiskAnalysisParams,
  validateCorrelationParams,
  validateNewsAnalysisParams,
};
