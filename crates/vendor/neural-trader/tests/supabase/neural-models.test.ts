/**
 * Neural Models Client Tests
 * Tests for neural network model persistence and training workflows
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from '@jest/testing-library'
import { neuralModelsClient } from '../../src/supabase/client/neural-models'
import { createClient } from '@supabase/supabase-js'

describe('Neural Models Client', () => {
  let testUserId: string
  let testModelId: string
  let testSymbolId: string
  let supabase: any

  beforeAll(async () => {
    supabase = createClient(
      process.env.SUPABASE_URL || 'http://localhost:54321',
      process.env.SUPABASE_SERVICE_ROLE_KEY || 'test-service-key'
    )

    // Create test user
    const { data: user } = await supabase.auth.admin.createUser({
      email: 'neuraltest@neuraltrade.ai',
      password: 'test-password-123',
      email_confirm: true
    })
    testUserId = user.user.id

    // Create test symbol
    const { data: symbol } = await supabase
      .from('symbols')
      .insert({
        symbol: 'TSLA',
        name: 'Tesla Inc.',
        exchange: 'NASDAQ',
        asset_type: 'stock'
      })
      .select()
      .single()
    testSymbolId = symbol.id

    // Create test profile
    await supabase
      .from('profiles')
      .insert({
        id: testUserId,
        username: 'neuraltest',
        email: 'neuraltest@neuraltrade.ai'
      })
  })

  afterAll(async () => {
    // Cleanup
    if (testModelId) {
      await neuralModelsClient.deleteModel(testModelId)
    }
    await supabase.from('symbols').delete().eq('id', testSymbolId)
    await supabase.from('profiles').delete().eq('id', testUserId)
    await supabase.auth.admin.deleteUser(testUserId)
  })

  describe('Model Creation and Management', () => {
    test('should create a new neural model', async () => {
      const modelData = {
        name: 'Test LSTM Model',
        model_type: 'lstm' as const,
        architecture: {
          layers: [
            { type: 'lstm', units: 64, return_sequences: true },
            { type: 'lstm', units: 32 },
            { type: 'dense', units: 1, activation: 'sigmoid' }
          ]
        },
        parameters: {
          learning_rate: 0.001,
          batch_size: 32,
          epochs: 100
        }
      }

      const { model, error } = await neuralModelsClient.createModel(testUserId, modelData)

      expect(error).toBeUndefined()
      expect(model).toBeDefined()
      expect(model.name).toBe(modelData.name)
      expect(model.model_type).toBe(modelData.model_type)
      expect(model.user_id).toBe(testUserId)
      expect(model.status).toBe('training')

      testModelId = model.id
    })

    test('should retrieve user models', async () => {
      const { models, error } = await neuralModelsClient.getUserModels(testUserId)

      expect(error).toBeUndefined()
      expect(Array.isArray(models)).toBe(true)
      expect(models.length).toBeGreaterThan(0)
      expect(models[0].user_id).toBe(testUserId)
    })

    test('should filter models by type and status', async () => {
      const { models, error } = await neuralModelsClient.getUserModels(testUserId, {
        model_type: 'lstm',
        status: 'training',
        limit: 5
      })

      expect(error).toBeUndefined()
      expect(Array.isArray(models)).toBe(true)
      
      models.forEach(model => {
        expect(model.model_type).toBe('lstm')
        expect(model.status).toBe('training')
      })
    })
  })

  describe('Training Management', () => {
    test('should start model training', async () => {
      const trainingData = {
        model_id: testModelId,
        hyperparameters: {
          learning_rate: 0.001,
          batch_size: 32,
          epochs: 50,
          validation_split: 0.2
        }
      }

      const { trainingRun, error } = await neuralModelsClient.startTraining(trainingData)

      expect(error).toBeUndefined()
      expect(trainingRun).toBeDefined()
      expect(trainingRun.model_id).toBe(testModelId)
      expect(trainingRun.status).toBe('running')
      expect(trainingRun.epoch).toBe(0)
    })

    test('should update training progress', async () => {
      const { trainingRuns } = await neuralModelsClient.getTrainingHistory(testModelId)
      const trainingRunId = trainingRuns[0].id

      const progress = {
        epoch: 10,
        loss: 0.045,
        accuracy: 0.892,
        validation_loss: 0.052,
        validation_accuracy: 0.875,
        metrics: {
          precision: 0.88,
          recall: 0.85,
          f1_score: 0.865
        },
        logs: 'Epoch 10/50 - loss: 0.045 - accuracy: 0.892'
      }

      const { success, error } = await neuralModelsClient.updateTrainingProgress(trainingRunId, progress)

      expect(success).toBe(true)
      expect(error).toBeUndefined()

      // Verify update
      const { trainingRuns: updatedRuns } = await neuralModelsClient.getTrainingHistory(testModelId)
      const updatedRun = updatedRuns.find(run => run.id === trainingRunId)
      
      expect(updatedRun!.epoch).toBe(10)
      expect(updatedRun!.loss).toBe(0.045)
      expect(updatedRun!.accuracy).toBe(0.892)
    })

    test('should complete training', async () => {
      const { trainingRuns } = await neuralModelsClient.getTrainingHistory(testModelId)
      const trainingRunId = trainingRuns[0].id

      const finalMetrics = {
        final_loss: 0.028,
        final_accuracy: 0.915,
        validation_loss: 0.032,
        validation_accuracy: 0.898,
        training_time_minutes: 45,
        total_epochs: 50
      }

      const { success, error } = await neuralModelsClient.completeTraining(
        trainingRunId,
        finalMetrics,
        '/models/test_lstm_model.h5'
      )

      expect(success).toBe(true)
      expect(error).toBeUndefined()

      // Verify model status updated
      const { models } = await neuralModelsClient.getUserModels(testUserId)
      const model = models.find(m => m.id === testModelId)
      
      expect(model!.status).toBe('trained')
      expect(model!.model_path).toBe('/models/test_lstm_model.h5')
    })

    test('should get training history', async () => {
      const { trainingRuns, error } = await neuralModelsClient.getTrainingHistory(testModelId)

      expect(error).toBeUndefined()
      expect(Array.isArray(trainingRuns)).toBe(true)
      expect(trainingRuns.length).toBeGreaterThan(0)
      expect(trainingRuns[0].model_id).toBe(testModelId)
    })
  })

  describe('Predictions Management', () => {
    test('should store model prediction', async () => {
      const predictionData = {
        model_id: testModelId,
        symbol_id: testSymbolId,
        prediction_value: 0.78,
        confidence: 0.92,
        features: {
          price: 245.50,
          volume: 1500000,
          rsi: 68.5,
          sma_20: 242.30,
          volatility: 0.045
        },
        prediction_timestamp: new Date().toISOString()
      }

      const { prediction, error } = await neuralModelsClient.storePrediction(predictionData)

      expect(error).toBeUndefined()
      expect(prediction).toBeDefined()
      expect(prediction.model_id).toBe(testModelId)
      expect(prediction.symbol_id).toBe(testSymbolId)
      expect(prediction.prediction_value).toBe(0.78)
      expect(prediction.confidence).toBe(0.92)
    })

    test('should retrieve model predictions', async () => {
      const { predictions, error } = await neuralModelsClient.getModelPredictions(testModelId, {
        symbol_id: testSymbolId,
        limit: 10
      })

      expect(error).toBeUndefined()
      expect(Array.isArray(predictions)).toBe(true)
      expect(predictions.length).toBeGreaterThan(0)
      expect(predictions[0].model_id).toBe(testModelId)
      expect(predictions[0].symbol_id).toBe(testSymbolId)
    })

    test('should update prediction with actual value', async () => {
      const { predictions } = await neuralModelsClient.getModelPredictions(testModelId, { limit: 1 })
      const predictionId = predictions[0].id

      const { success, error } = await neuralModelsClient.updatePredictionActual(predictionId, 0.82)

      expect(success).toBe(true)
      expect(error).toBeUndefined()

      // Verify update
      const { predictions: updatedPredictions } = await neuralModelsClient.getModelPredictions(testModelId, { limit: 1 })
      const updatedPrediction = updatedPredictions.find(p => p.id === predictionId)
      
      expect(updatedPrediction!.actual_value).toBe(0.82)
    })
  })

  describe('Model Performance', () => {
    test('should update model performance metrics', async () => {
      const { success, error } = await neuralModelsClient.updateModelPerformance(testModelId)

      expect(success).toBe(true)
      expect(error).toBeUndefined()

      // Verify performance metrics were updated
      const { models } = await neuralModelsClient.getUserModels(testUserId)
      const model = models.find(m => m.id === testModelId)
      
      expect(model!.performance_metrics).toBeDefined()
      expect(typeof model!.performance_metrics.accuracy).toBe('number')
      expect(typeof model!.performance_metrics.mae).toBe('number')
      expect(typeof model!.performance_metrics.rmse).toBe('number')
    })

    test('should deploy model', async () => {
      const { success, error } = await neuralModelsClient.deployModel(testModelId)

      expect(success).toBe(true)
      expect(error).toBeUndefined()

      // Verify model status
      const { models } = await neuralModelsClient.getUserModels(testUserId)
      const model = models.find(m => m.id === testModelId)
      
      expect(model!.status).toBe('deployed')
    })

    test('should compare multiple models', async () => {
      // Create another model for comparison
      const { model: model2 } = await neuralModelsClient.createModel(testUserId, {
        name: 'Test Transformer Model',
        model_type: 'transformer',
        architecture: {
          layers: [
            { type: 'attention', heads: 8, dim: 64 },
            { type: 'dense', units: 1, activation: 'sigmoid' }
          ]
        }
      })

      const { comparison, error } = await neuralModelsClient.compareModels([testModelId, model2.id])

      expect(error).toBeUndefined()
      expect(Array.isArray(comparison)).toBe(true)
      expect(comparison).toHaveLength(2)
      
      comparison.forEach(model => {
        expect(model).toHaveProperty('id')
        expect(model).toHaveProperty('name')
        expect(model).toHaveProperty('model_type')
        expect(model).toHaveProperty('performance_metrics')
        expect(model).toHaveProperty('recent_accuracy')
        expect(model).toHaveProperty('prediction_count')
      })

      // Cleanup
      await neuralModelsClient.deleteModel(model2.id)
    })
  })

  describe('Error Handling', () => {
    test('should handle invalid model creation', async () => {
      const invalidModelData = {
        name: '', // Invalid empty name
        model_type: 'invalid_type' as any,
        architecture: null as any
      }

      const { model, error } = await neuralModelsClient.createModel(testUserId, invalidModelData)

      expect(error).toBeDefined()
      expect(model).toBeNull()
    })

    test('should handle unauthorized access', async () => {
      const unauthorizedUserId = '00000000-0000-0000-0000-000000000000'
      
      const { models, error } = await neuralModelsClient.getUserModels(unauthorizedUserId)

      // Should return empty array for non-existent user (RLS enforced)
      expect(error).toBeUndefined()
      expect(models).toHaveLength(0)
    })

    test('should handle training non-existent model', async () => {
      const { trainingRun, error } = await neuralModelsClient.startTraining({
        model_id: '00000000-0000-0000-0000-000000000000',
        hyperparameters: {}
      })

      expect(error).toBeDefined()
      expect(trainingRun).toBeNull()
    })
  })
})