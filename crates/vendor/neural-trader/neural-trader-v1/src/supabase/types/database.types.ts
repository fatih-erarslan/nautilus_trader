/**
 * Database Types for Neural Trading Platform
 * Auto-generated types for type-safe database operations
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          username: string
          email: string
          full_name: string | null
          avatar_url: string | null
          tier: string
          api_quota: number
          api_usage: number
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          username: string
          email: string
          full_name?: string | null
          avatar_url?: string | null
          tier?: string
          api_quota?: number
          api_usage?: number
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          username?: string
          email?: string
          full_name?: string | null
          avatar_url?: string | null
          tier?: string
          api_quota?: number
          api_usage?: number
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }
      symbols: {
        Row: {
          id: string
          symbol: string
          name: string
          exchange: string
          asset_type: string
          is_active: boolean
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          symbol: string
          name: string
          exchange: string
          asset_type: string
          is_active?: boolean
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          symbol?: string
          name?: string
          exchange?: string
          asset_type?: string
          is_active?: boolean
          metadata?: Json
          created_at?: string
        }
        Relationships: []
      }
      market_data: {
        Row: {
          id: string
          symbol_id: string
          timestamp: string
          open: number
          high: number
          low: number
          close: number
          volume: number
          timeframe: string
          created_at: string
        }
        Insert: {
          id?: string
          symbol_id: string
          timestamp: string
          open: number
          high: number
          low: number
          close: number
          volume: number
          timeframe: string
          created_at?: string
        }
        Update: {
          id?: string
          symbol_id?: string
          timestamp?: string
          open?: number
          high?: number
          low?: number
          close?: number
          volume?: number
          timeframe?: string
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "market_data_symbol_id_fkey"
            columns: ["symbol_id"]
            referencedRelation: "symbols"
            referencedColumns: ["id"]
          }
        ]
      }
      news_data: {
        Row: {
          id: string
          title: string
          content: string | null
          source: string
          url: string | null
          published_at: string
          symbols: string[]
          sentiment_score: number | null
          relevance_score: number | null
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          title: string
          content?: string | null
          source: string
          url?: string | null
          published_at: string
          symbols?: string[]
          sentiment_score?: number | null
          relevance_score?: number | null
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          title?: string
          content?: string | null
          source?: string
          url?: string | null
          published_at?: string
          symbols?: string[]
          sentiment_score?: number | null
          relevance_score?: number | null
          metadata?: Json
          created_at?: string
        }
        Relationships: []
      }
      trading_accounts: {
        Row: {
          id: string
          user_id: string
          name: string
          broker: string
          account_type: string
          balance: number
          equity: number
          margin_used: number
          margin_available: number
          api_credentials: Json | null
          is_active: boolean
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          broker: string
          account_type?: string
          balance?: number
          equity?: number
          margin_used?: number
          margin_available?: number
          api_credentials?: Json | null
          is_active?: boolean
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          broker?: string
          account_type?: string
          balance?: number
          equity?: number
          margin_used?: number
          margin_available?: number
          api_credentials?: Json | null
          is_active?: boolean
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "trading_accounts_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          }
        ]
      }
      positions: {
        Row: {
          id: string
          account_id: string
          symbol_id: string
          side: string
          quantity: number
          entry_price: number
          current_price: number | null
          unrealized_pnl: number
          realized_pnl: number
          opened_at: string
          closed_at: string | null
          metadata: Json
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          account_id: string
          symbol_id: string
          side: string
          quantity: number
          entry_price: number
          current_price?: number | null
          unrealized_pnl?: number
          realized_pnl?: number
          opened_at?: string
          closed_at?: string | null
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          account_id?: string
          symbol_id?: string
          side?: string
          quantity?: number
          entry_price?: number
          current_price?: number | null
          unrealized_pnl?: number
          realized_pnl?: number
          opened_at?: string
          closed_at?: string | null
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "positions_account_id_fkey"
            columns: ["account_id"]
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "positions_symbol_id_fkey"
            columns: ["symbol_id"]
            referencedRelation: "symbols"
            referencedColumns: ["id"]
          }
        ]
      }
      orders: {
        Row: {
          id: string
          account_id: string
          symbol_id: string
          order_type: string
          side: string
          quantity: number
          price: number | null
          stop_price: number | null
          status: string
          filled_quantity: number
          average_fill_price: number | null
          commission: number
          external_order_id: string | null
          placed_at: string
          filled_at: string | null
          cancelled_at: string | null
          metadata: Json
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          account_id: string
          symbol_id: string
          order_type: string
          side: string
          quantity: number
          price?: number | null
          stop_price?: number | null
          status?: string
          filled_quantity?: number
          average_fill_price?: number | null
          commission?: number
          external_order_id?: string | null
          placed_at?: string
          filled_at?: string | null
          cancelled_at?: string | null
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          account_id?: string
          symbol_id?: string
          order_type?: string
          side?: string
          quantity?: number
          price?: number | null
          stop_price?: number | null
          status?: string
          filled_quantity?: number
          average_fill_price?: number | null
          commission?: number
          external_order_id?: string | null
          placed_at?: string
          filled_at?: string | null
          cancelled_at?: string | null
          metadata?: Json
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "orders_account_id_fkey"
            columns: ["account_id"]
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "orders_symbol_id_fkey"
            columns: ["symbol_id"]
            referencedRelation: "symbols"
            referencedColumns: ["id"]
          }
        ]
      }
      neural_models: {
        Row: {
          id: string
          user_id: string
          name: string
          model_type: string
          architecture: Json
          parameters: Json
          status: string
          version: number
          performance_metrics: Json
          training_data_hash: string | null
          model_path: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          model_type: string
          architecture: Json
          parameters?: Json
          status?: string
          version?: number
          performance_metrics?: Json
          training_data_hash?: string | null
          model_path?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          model_type?: string
          architecture?: Json
          parameters?: Json
          status?: string
          version?: number
          performance_metrics?: Json
          training_data_hash?: string | null
          model_path?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "neural_models_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          }
        ]
      }
      training_runs: {
        Row: {
          id: string
          model_id: string
          started_at: string
          completed_at: string | null
          status: string
          epoch: number
          loss: number | null
          accuracy: number | null
          validation_loss: number | null
          validation_accuracy: number | null
          hyperparameters: Json
          metrics: Json
          logs: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          model_id: string
          started_at?: string
          completed_at?: string | null
          status?: string
          epoch?: number
          loss?: number | null
          accuracy?: number | null
          validation_loss?: number | null
          validation_accuracy?: number | null
          hyperparameters?: Json
          metrics?: Json
          logs?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          model_id?: string
          started_at?: string
          completed_at?: string | null
          status?: string
          epoch?: number
          loss?: number | null
          accuracy?: number | null
          validation_loss?: number | null
          validation_accuracy?: number | null
          hyperparameters?: Json
          metrics?: Json
          logs?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "training_runs_model_id_fkey"
            columns: ["model_id"]
            referencedRelation: "neural_models"
            referencedColumns: ["id"]
          }
        ]
      }
      model_predictions: {
        Row: {
          id: string
          model_id: string
          symbol_id: string
          prediction_timestamp: string
          prediction_value: number
          confidence: number | null
          actual_value: number | null
          error: number | null
          features: Json
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          model_id: string
          symbol_id: string
          prediction_timestamp: string
          prediction_value: number
          confidence?: number | null
          actual_value?: number | null
          error?: number | null
          features?: Json
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          model_id?: string
          symbol_id?: string
          prediction_timestamp?: string
          prediction_value?: number
          confidence?: number | null
          actual_value?: number | null
          error?: number | null
          features?: Json
          metadata?: Json
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "model_predictions_model_id_fkey"
            columns: ["model_id"]
            referencedRelation: "neural_models"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "model_predictions_symbol_id_fkey"
            columns: ["symbol_id"]
            referencedRelation: "symbols"
            referencedColumns: ["id"]
          }
        ]
      }
      trading_bots: {
        Row: {
          id: string
          user_id: string
          account_id: string
          name: string
          strategy_type: string
          configuration: Json
          model_ids: string[]
          symbols: string[]
          status: string
          max_position_size: number
          risk_limit: number
          is_active: boolean
          performance_metrics: Json
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          account_id: string
          name: string
          strategy_type: string
          configuration: Json
          model_ids?: string[]
          symbols: string[]
          status?: string
          max_position_size?: number
          risk_limit?: number
          is_active?: boolean
          performance_metrics?: Json
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          account_id?: string
          name?: string
          strategy_type?: string
          configuration?: Json
          model_ids?: string[]
          symbols?: string[]
          status?: string
          max_position_size?: number
          risk_limit?: number
          is_active?: boolean
          performance_metrics?: Json
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "trading_bots_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "trading_bots_account_id_fkey"
            columns: ["account_id"]
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          }
        ]
      }
      bot_executions: {
        Row: {
          id: string
          bot_id: string
          symbol_id: string
          action: string
          signal_strength: number | null
          reasoning: string | null
          order_id: string | null
          executed_at: string
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          bot_id: string
          symbol_id: string
          action: string
          signal_strength?: number | null
          reasoning?: string | null
          order_id?: string | null
          executed_at?: string
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          bot_id?: string
          symbol_id?: string
          action?: string
          signal_strength?: number | null
          reasoning?: string | null
          order_id?: string | null
          executed_at?: string
          metadata?: Json
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "bot_executions_bot_id_fkey"
            columns: ["bot_id"]
            referencedRelation: "trading_bots"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "bot_executions_symbol_id_fkey"
            columns: ["symbol_id"]
            referencedRelation: "symbols"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "bot_executions_order_id_fkey"
            columns: ["order_id"]
            referencedRelation: "orders"
            referencedColumns: ["id"]
          }
        ]
      }
      sandbox_deployments: {
        Row: {
          id: string
          user_id: string
          bot_id: string | null
          sandbox_id: string
          name: string
          template: string
          configuration: Json
          status: string
          cpu_count: number
          memory_mb: number
          timeout_seconds: number
          started_at: string | null
          stopped_at: string | null
          resource_usage: Json
          logs: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          bot_id?: string | null
          sandbox_id: string
          name: string
          template?: string
          configuration?: Json
          status?: string
          cpu_count?: number
          memory_mb?: number
          timeout_seconds?: number
          started_at?: string | null
          stopped_at?: string | null
          resource_usage?: Json
          logs?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          bot_id?: string | null
          sandbox_id?: string
          name?: string
          template?: string
          configuration?: Json
          status?: string
          cpu_count?: number
          memory_mb?: number
          timeout_seconds?: number
          started_at?: string | null
          stopped_at?: string | null
          resource_usage?: Json
          logs?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "sandbox_deployments_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "sandbox_deployments_bot_id_fkey"
            columns: ["bot_id"]
            referencedRelation: "trading_bots"
            referencedColumns: ["id"]
          }
        ]
      }
      performance_metrics: {
        Row: {
          id: string
          entity_type: string
          entity_id: string
          metric_type: string
          metric_value: number
          timestamp: string
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          entity_type: string
          entity_id: string
          metric_type: string
          metric_value: number
          timestamp?: string
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          entity_type?: string
          entity_id?: string
          metric_type?: string
          metric_value?: number
          timestamp?: string
          metadata?: Json
          created_at?: string
        }
        Relationships: []
      }
      alerts: {
        Row: {
          id: string
          user_id: string
          title: string
          message: string
          severity: string
          entity_type: string | null
          entity_id: string | null
          is_read: boolean
          expires_at: string | null
          metadata: Json
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          title: string
          message: string
          severity?: string
          entity_type?: string | null
          entity_id?: string | null
          is_read?: boolean
          expires_at?: string | null
          metadata?: Json
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          title?: string
          message?: string
          severity?: string
          entity_type?: string | null
          entity_id?: string | null
          is_read?: boolean
          expires_at?: string | null
          metadata?: Json
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "alerts_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          }
        ]
      }
      audit_logs: {
        Row: {
          id: string
          user_id: string | null
          action: string
          entity_type: string | null
          entity_id: string | null
          old_values: Json | null
          new_values: Json | null
          ip_address: string | null
          user_agent: string | null
          created_at: string
        }
        Insert: {
          id?: string
          user_id?: string | null
          action: string
          entity_type?: string | null
          entity_id?: string | null
          old_values?: Json | null
          new_values?: Json | null
          ip_address?: string | null
          user_agent?: string | null
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string | null
          action?: string
          entity_type?: string | null
          entity_id?: string | null
          old_values?: Json | null
          new_values?: Json | null
          ip_address?: string | null
          user_agent?: string | null
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "audit_logs_user_id_fkey"
            columns: ["user_id"]
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          }
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      calculate_portfolio_performance: {
        Args: {
          account_id_param: string
          start_date?: string
          end_date?: string
        }
        Returns: {
          total_return: number
          realized_pnl: number
          unrealized_pnl: number
          win_rate: number
          sharpe_ratio: number
          max_drawdown: number
        }[]
      }
      update_model_performance: {
        Args: {
          model_id_param: string
          predictions_count?: number
        }
        Returns: Json
      }
      generate_trading_signal: {
        Args: {
          symbol_param: string
          model_ids: string[]
          lookback_periods?: number
        }
        Returns: Json
      }
      calculate_position_risk: {
        Args: {
          account_id_param: string
          symbol_param: string
          position_size: number
        }
        Returns: Json
      }
      cleanup_old_data: {
        Args: {}
        Returns: number
      }
    }
    Enums: {
      bot_status: 'active' | 'paused' | 'stopped' | 'error' | 'training'
      order_status: 'pending' | 'filled' | 'cancelled' | 'rejected' | 'partial'
      order_type: 'market' | 'limit' | 'stop' | 'stop_limit'
      position_side: 'long' | 'short'
      model_status: 'training' | 'trained' | 'deployed' | 'deprecated'
      alert_severity: 'info' | 'warning' | 'error' | 'critical'
      deployment_status: 'pending' | 'running' | 'stopped' | 'failed' | 'terminated'
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}