/**
 * Neural Models Client
 * Manages neural network model persistence and training workflows
 */

import { supabase, supabaseAdmin } from '../supabase.config'
import { Database } from '../types/database.types'

type Tables = Database['public']['Tables']
type NeuralModel = Tables['neural_models']['Row']
type TrainingRun = Tables['training_runs']['Row']
type ModelPrediction = Tables['model_predictions']['Row']

export interface CreateModelRequest {
  name: string
  model_type: 'lstm' | 'transformer' | 'cnn' | 'ensemble'
  architecture: any
  parameters?: any
  training_data_hash?: string
}

export interface StartTrainingRequest {
  model_id: string
  hyperparameters: any
  training_data_path?: string
}

export interface ModelPredictionRequest {
  model_id: string
  symbol_id: string
  features: any
  prediction_timestamp?: string
}

export interface TrainingProgress {
  epoch: number
  loss: number
  accuracy?: number
  validation_loss?: number
  validation_accuracy?: number
  metrics?: any
  logs?: string
}

export class NeuralModelsClient {
  
  // Create a new neural model
  async createModel(userId: string, modelData: CreateModelRequest): Promise<{ model: NeuralModel; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('neural_models')
        .insert({
          user_id: userId,
          name: modelData.name,
          model_type: modelData.model_type,
          architecture: modelData.architecture,
          parameters: modelData.parameters || {},
          training_data_hash: modelData.training_data_hash,
          status: 'training',
          version: 1
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to create neural model:', error)
        return { model: null as any, error: error.message }
      }

      return { model: data }
    } catch (error) {
      console.error('Error creating neural model:', error)
      return { model: null as any, error: 'Failed to create model' }
    }
  }

  // Get user's models
  async getUserModels(userId: string, filters?: {
    status?: string
    model_type?: string
    limit?: number
  }): Promise<{ models: NeuralModel[]; error?: string }> {
    try {
      let query = supabase
        .from('neural_models')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })

      if (filters?.status) {
        query = query.eq('status', filters.status)
      }

      if (filters?.model_type) {
        query = query.eq('model_type', filters.model_type)
      }

      if (filters?.limit) {
        query = query.limit(filters.limit)
      }

      const { data, error } = await query

      if (error) {
        console.error('Failed to fetch neural models:', error)
        return { models: [], error: error.message }
      }

      return { models: data || [] }
    } catch (error) {
      console.error('Error fetching neural models:', error)
      return { models: [], error: 'Failed to fetch models' }
    }
  }

  // Start training a model
  async startTraining(trainingData: StartTrainingRequest): Promise<{ trainingRun: TrainingRun; error?: string }> {
    try {
      // First verify model exists and user has access
      const { data: model, error: modelError } = await supabase
        .from('neural_models')
        .select('*')
        .eq('id', trainingData.model_id)
        .single()

      if (modelError || !model) {
        return { trainingRun: null as any, error: 'Model not found or access denied' }
      }

      // Create training run
      const { data, error } = await supabase
        .from('training_runs')
        .insert({
          model_id: trainingData.model_id,
          status: 'running',
          hyperparameters: trainingData.hyperparameters,
          epoch: 0
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to start training run:', error)
        return { trainingRun: null as any, error: error.message }
      }

      // Update model status to training
      await supabase
        .from('neural_models')
        .update({ status: 'training' })
        .eq('id', trainingData.model_id)

      return { trainingRun: data }
    } catch (error) {
      console.error('Error starting training:', error)
      return { trainingRun: null as any, error: 'Failed to start training' }
    }
  }

  // Update training progress
  async updateTrainingProgress(
    trainingRunId: string,
    progress: TrainingProgress
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('training_runs')
        .update({
          epoch: progress.epoch,
          loss: progress.loss,
          accuracy: progress.accuracy,
          validation_loss: progress.validation_loss,
          validation_accuracy: progress.validation_accuracy,
          metrics: progress.metrics,
          logs: progress.logs,
          updated_at: new Date().toISOString()
        })
        .eq('id', trainingRunId)

      if (error) {
        console.error('Failed to update training progress:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error updating training progress:', error)
      return { success: false, error: 'Failed to update training progress' }
    }
  }

  // Complete training
  async completeTraining(
    trainingRunId: string,
    finalMetrics: any,
    modelPath?: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // Update training run
      const { data: trainingRun, error: updateError } = await supabase
        .from('training_runs')
        .update({
          status: 'completed',
          completed_at: new Date().toISOString(),
          metrics: finalMetrics
        })
        .eq('id', trainingRunId)
        .select()
        .single()

      if (updateError) {
        console.error('Failed to complete training run:', updateError)
        return { success: false, error: updateError.message }
      }

      // Update model status and performance metrics
      const updateData: any = {
        status: 'trained',
        performance_metrics: finalMetrics
      }

      if (modelPath) {
        updateData.model_path = modelPath
      }

      const { error: modelError } = await supabase
        .from('neural_models')
        .update(updateData)
        .eq('id', trainingRun.model_id)

      if (modelError) {
        console.error('Failed to update model after training:', modelError)
        return { success: false, error: modelError.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error completing training:', error)
      return { success: false, error: 'Failed to complete training' }
    }
  }

  // Store model prediction
  async storePrediction(predictionData: ModelPredictionRequest & {
    prediction_value: number
    confidence?: number
    actual_value?: number
  }): Promise<{ prediction: ModelPrediction; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('model_predictions')
        .insert({
          model_id: predictionData.model_id,
          symbol_id: predictionData.symbol_id,
          prediction_timestamp: predictionData.prediction_timestamp || new Date().toISOString(),
          prediction_value: predictionData.prediction_value,
          confidence: predictionData.confidence,
          actual_value: predictionData.actual_value,
          features: predictionData.features,
          metadata: {}
        })
        .select()
        .single()

      if (error) {
        console.error('Failed to store prediction:', error)
        return { prediction: null as any, error: error.message }
      }

      return { prediction: data }
    } catch (error) {
      console.error('Error storing prediction:', error)
      return { prediction: null as any, error: 'Failed to store prediction' }
    }
  }

  // Get model predictions
  async getModelPredictions(
    modelId: string,
    filters?: {
      symbol_id?: string
      start_date?: string
      end_date?: string
      limit?: number
    }
  ): Promise<{ predictions: ModelPrediction[]; error?: string }> {
    try {
      let query = supabase
        .from('model_predictions')
        .select('*')
        .eq('model_id', modelId)
        .order('prediction_timestamp', { ascending: false })

      if (filters?.symbol_id) {
        query = query.eq('symbol_id', filters.symbol_id)
      }

      if (filters?.start_date) {
        query = query.gte('prediction_timestamp', filters.start_date)
      }

      if (filters?.end_date) {
        query = query.lte('prediction_timestamp', filters.end_date)
      }

      if (filters?.limit) {
        query = query.limit(filters.limit)
      }

      const { data, error } = await query

      if (error) {
        console.error('Failed to fetch predictions:', error)
        return { predictions: [], error: error.message }
      }

      return { predictions: data || [] }
    } catch (error) {
      console.error('Error fetching predictions:', error)
      return { predictions: [], error: 'Failed to fetch predictions' }
    }
  }

  // Update model performance metrics
  async updateModelPerformance(modelId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const { data, error } = await supabase.rpc('update_model_performance', {
        model_id_param: modelId,
        predictions_count: 100
      })

      if (error) {
        console.error('Failed to update model performance:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error updating model performance:', error)
      return { success: false, error: 'Failed to update model performance' }
    }
  }

  // Deploy model (mark as deployed)
  async deployModel(modelId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('neural_models')
        .update({ status: 'deployed' })
        .eq('id', modelId)

      if (error) {
        console.error('Failed to deploy model:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error deploying model:', error)
      return { success: false, error: 'Failed to deploy model' }
    }
  }

  // Get training history for a model
  async getTrainingHistory(modelId: string): Promise<{ trainingRuns: TrainingRun[]; error?: string }> {
    try {
      const { data, error } = await supabase
        .from('training_runs')
        .select('*')
        .eq('model_id', modelId)
        .order('started_at', { ascending: false })

      if (error) {
        console.error('Failed to fetch training history:', error)
        return { trainingRuns: [], error: error.message }
      }

      return { trainingRuns: data || [] }
    } catch (error) {
      console.error('Error fetching training history:', error)
      return { trainingRuns: [], error: 'Failed to fetch training history' }
    }
  }

  // Update actual values for predictions (for performance tracking)
  async updatePredictionActual(
    predictionId: string,
    actualValue: number
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await supabase
        .from('model_predictions')
        .update({
          actual_value: actualValue,
          error: null // Will be calculated by trigger
        })
        .eq('id', predictionId)

      if (error) {
        console.error('Failed to update prediction actual value:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error updating prediction actual value:', error)
      return { success: false, error: 'Failed to update prediction actual value' }
    }
  }

  // Delete model and all associated data
  async deleteModel(modelId: string): Promise<{ success: boolean; error?: string }> {
    try {
      // Delete in order due to foreign key constraints
      await supabase.from('model_predictions').delete().eq('model_id', modelId)
      await supabase.from('training_runs').delete().eq('model_id', modelId)
      const { error } = await supabase.from('neural_models').delete().eq('id', modelId)

      if (error) {
        console.error('Failed to delete model:', error)
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      console.error('Error deleting model:', error)
      return { success: false, error: 'Failed to delete model' }
    }
  }

  // Compare model performance
  async compareModels(modelIds: string[]): Promise<{
    comparison: any[];
    error?: string
  }> {
    try {
      const { data: models, error } = await supabase
        .from('neural_models')
        .select(`
          id,
          name,
          model_type,
          performance_metrics,
          created_at,
          status
        `)
        .in('id', modelIds)

      if (error) {
        console.error('Failed to fetch models for comparison:', error)
        return { comparison: [], error: error.message }
      }

      // Get recent predictions for each model
      const comparisons = await Promise.all(
        (models || []).map(async (model) => {
          const { data: predictions } = await supabase
            .from('model_predictions')
            .select('prediction_value, actual_value, confidence, created_at')
            .eq('model_id', model.id)
            .not('actual_value', 'is', null)
            .order('created_at', { ascending: false })
            .limit(100)

          const accuracy = predictions && predictions.length > 0 ? 
            predictions.filter(p => 
              Math.abs((p.prediction_value - p.actual_value) / p.actual_value) <= 0.05
            ).length / predictions.length : 0

          return {
            ...model,
            recent_accuracy: accuracy,
            prediction_count: predictions?.length || 0
          }
        })
      )

      return { comparison: comparisons }
    } catch (error) {
      console.error('Error comparing models:', error)
      return { comparison: [], error: 'Failed to compare models' }
    }
  }
}

// Export singleton instance
export const neuralModelsClient = new NeuralModelsClient()
export default neuralModelsClient