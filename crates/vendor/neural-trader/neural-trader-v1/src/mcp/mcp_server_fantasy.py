#!/usr/bin/env python3
"""
Fixed Fantasy Collective MCP Server for AI News Trading Platform
Combines sports betting, prediction markets, and syndicate management with fantasy collective features.
This version fixes the **kwargs issue for FastMCP compatibility.
"""

import json
import logging
import sys
import os
import sqlite3
import hashlib
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from decimal import Decimal
from contextlib import contextmanager

# Configure logging to NOT interfere with stdio transport
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Import FastMCP
try:
    from fastmcp import FastMCP
    from pydantic import BaseModel
except ImportError as e:
    print(f"ERROR: Failed to import required packages: {e}", file=sys.stderr)
    print("Please install: pip install fastmcp pydantic", file=sys.stderr)
    sys.exit(1)

# Initialize FastMCP server
mcp = FastMCP(
    "Fantasy Collective Trading Platform",
    dependencies=["trading", "gpu-acceleration", "fantasy-leagues", "collective-intelligence"]
)

# Database configuration
DB_PATH = Path("fantasy_collective.db")

# Helper functions for database operations
@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def execute_db_query(query: str, params: tuple = (), fetchone: bool = False, fetchall: bool = False):
    """Execute a database query with proper error handling."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        if fetchone:
            result = cursor.fetchone()
            return dict(result) if result else None
        elif fetchall:
            results = cursor.fetchall()
            return [dict(row) for row in results]
        else:
            return cursor.lastrowid

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = ''.join(random.choices('0123456789abcdef', k=8))
    return f"{prefix}{timestamp}_{random_part}"

def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize database schema
def init_database():
    """Initialize the database with fantasy collective tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                balance REAL DEFAULT 1000.0,
                total_score REAL DEFAULT 0.0,
                achievements_count INTEGER DEFAULT 0,
                leagues_joined INTEGER DEFAULT 0,
                predictions_made INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Fantasy leagues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fantasy_leagues (
                league_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                league_type TEXT NOT NULL,
                sport TEXT,
                created_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_date DATE,
                end_date DATE,
                max_members INTEGER DEFAULT 12,
                current_members INTEGER DEFAULT 0,
                entry_fee REAL DEFAULT 0,
                prize_pool REAL DEFAULT 0,
                scoring_system TEXT,
                settings TEXT,
                status TEXT DEFAULT 'open',
                FOREIGN KEY (created_by) REFERENCES users(user_id)
            )
        """)
        
        # League memberships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS league_memberships (
                membership_id TEXT PRIMARY KEY,
                league_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_rank INTEGER,
                total_points REAL DEFAULT 0,
                weekly_points REAL DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (league_id) REFERENCES fantasy_leagues(league_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(league_id, user_id)
            )
        """)
        
        # Collectives table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collectives (
                collective_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                member_count INTEGER DEFAULT 1,
                total_funds REAL DEFAULT 0,
                consensus_threshold REAL DEFAULT 0.66,
                voting_power_total REAL DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0,
                settings TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (created_by) REFERENCES users(user_id)
            )
        """)
        
        # Collective memberships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collective_memberships (
                membership_id TEXT PRIMARY KEY,
                collective_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role TEXT DEFAULT 'member',
                voting_power REAL DEFAULT 1.0,
                contribution REAL DEFAULT 0,
                earnings REAL DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (collective_id) REFERENCES collectives(collective_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(collective_id, user_id)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                league_id TEXT,
                collective_id TEXT,
                event_id TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                stake REAL DEFAULT 0,
                potential_payout REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                is_correct BOOLEAN,
                points_earned REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (league_id) REFERENCES fantasy_leagues(league_id),
                FOREIGN KEY (collective_id) REFERENCES collectives(collective_id),
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
        """)
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                event_type TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                resolution_time TIMESTAMP,
                outcome TEXT,
                odds TEXT,
                total_predictions INTEGER DEFAULT 0,
                total_stake REAL DEFAULT 0,
                status TEXT DEFAULT 'upcoming'
            )
        """)
        
        # Achievements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                achievement_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                points_value INTEGER DEFAULT 10,
                requirement_type TEXT,
                requirement_value REAL,
                icon TEXT,
                rarity TEXT DEFAULT 'common',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User achievements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_achievements (
                user_achievement_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                achievement_id TEXT NOT NULL,
                earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                progress REAL DEFAULT 0,
                is_completed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (achievement_id) REFERENCES achievements(achievement_id),
                UNIQUE(user_id, achievement_id)
            )
        """)
        
        # Tournaments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tournaments (
                tournament_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tournament_type TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_date DATE,
                end_date DATE,
                max_participants INTEGER DEFAULT 64,
                current_participants INTEGER DEFAULT 0,
                entry_fee REAL DEFAULT 0,
                prize_pool REAL DEFAULT 0,
                rounds_total INTEGER DEFAULT 1,
                current_round INTEGER DEFAULT 0,
                brackets TEXT,
                winner_id TEXT,
                status TEXT DEFAULT 'registration',
                FOREIGN KEY (created_by) REFERENCES users(user_id),
                FOREIGN KEY (winner_id) REFERENCES users(user_id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_leagues_status ON fantasy_leagues(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_event ON predictions(event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        
        conn.commit()
        logger.info("Database initialized successfully")

# Initialize database on startup
init_database()

# MCP Tools Implementation

@mcp.tool()
def ping() -> Dict[str, Any]:
    """Test server connectivity and get status."""
    return {
        "status": "connected",
        "server": "Fantasy Collective MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "database": "SQLite",
        "features": ["fantasy_leagues", "predictions", "collectives", "achievements", "tournaments"]
    }

@mcp.tool()
def create_fantasy_league(
    name: str,
    league_type: str,
    created_by: str = "default_user",
    sport: Optional[str] = None,
    max_members: int = 12,
    entry_fee: float = 0,
    scoring_system: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new fantasy league."""
    try:
        league_id = generate_id("league_")
        
        # Default scoring system based on type
        if not scoring_system:
            scoring_system = {
                "fantasy_sports": "points_based",
                "prediction": "accuracy_based",
                "survivor": "elimination",
                "tournament": "bracket"
            }.get(league_type, "points_based")
        
        # Insert league
        execute_db_query("""
            INSERT INTO fantasy_leagues (
                league_id, name, league_type, sport, created_by,
                max_members, entry_fee, scoring_system, start_date, end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            league_id, name, league_type, sport, created_by,
            max_members, entry_fee, scoring_system, start_date, end_date
        ))
        
        # Auto-join creator
        execute_db_query("""
            INSERT INTO league_memberships (
                membership_id, league_id, user_id
            ) VALUES (?, ?, ?)
        """, (generate_id("member_"), league_id, created_by))
        
        # Update league member count
        execute_db_query("""
            UPDATE fantasy_leagues SET current_members = 1 WHERE league_id = ?
        """, (league_id,))
        
        return {
            "league_id": league_id,
            "name": name,
            "type": league_type,
            "status": "created",
            "message": f"Fantasy league '{name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating league: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def join_league(league_id: str, user_id: str = "default_user") -> Dict[str, Any]:
    """Join an existing fantasy league."""
    try:
        # Check if league exists and has space
        league = execute_db_query("""
            SELECT * FROM fantasy_leagues WHERE league_id = ? AND status = 'open'
        """, (league_id,), fetchone=True)
        
        if not league:
            return {"error": "League not found or closed", "status": "failed"}
        
        if league['current_members'] >= league['max_members']:
            return {"error": "League is full", "status": "failed"}
        
        # Check if already member
        existing = execute_db_query("""
            SELECT * FROM league_memberships 
            WHERE league_id = ? AND user_id = ? AND is_active = TRUE
        """, (league_id, user_id), fetchone=True)
        
        if existing:
            return {"error": "Already a member of this league", "status": "failed"}
        
        # Add membership
        execute_db_query("""
            INSERT INTO league_memberships (
                membership_id, league_id, user_id
            ) VALUES (?, ?, ?)
        """, (generate_id("member_"), league_id, user_id))
        
        # Update member count
        execute_db_query("""
            UPDATE fantasy_leagues 
            SET current_members = current_members + 1 
            WHERE league_id = ?
        """, (league_id,))
        
        # Deduct entry fee if applicable
        if league['entry_fee'] > 0:
            execute_db_query("""
                UPDATE users SET balance = balance - ? WHERE user_id = ?
            """, (league['entry_fee'], user_id))
            
            execute_db_query("""
                UPDATE fantasy_leagues 
                SET prize_pool = prize_pool + ? 
                WHERE league_id = ?
            """, (league['entry_fee'], league_id))
        
        return {
            "league_id": league_id,
            "status": "joined",
            "message": f"Successfully joined league '{league['name']}'",
            "entry_fee_paid": league['entry_fee']
        }
        
    except Exception as e:
        logger.error(f"Error joining league: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def make_prediction(
    event_id: str,
    prediction_value: str,
    user_id: str = "default_user",
    prediction_type: str = "binary",
    confidence: float = 0.5,
    stake: float = 0,
    league_id: Optional[str] = None,
    collective_id: Optional[str] = None
) -> Dict[str, Any]:
    """Make a prediction on an event."""
    try:
        prediction_id = generate_id("pred_")
        
        # Calculate potential payout based on confidence and stake
        potential_payout = stake * (2.0 - confidence) if stake > 0 else 0
        
        # Insert prediction
        execute_db_query("""
            INSERT INTO predictions (
                prediction_id, user_id, event_id, prediction_type,
                prediction_value, confidence, stake, potential_payout,
                league_id, collective_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id, user_id, event_id, prediction_type,
            prediction_value, confidence, stake, potential_payout,
            league_id, collective_id
        ))
        
        # Update user balance if stake
        if stake > 0:
            execute_db_query("""
                UPDATE users SET balance = balance - ? WHERE user_id = ?
            """, (stake, user_id))
        
        # Update event statistics
        execute_db_query("""
            UPDATE events 
            SET total_predictions = total_predictions + 1,
                total_stake = total_stake + ?
            WHERE event_id = ?
        """, (stake, event_id))
        
        return {
            "prediction_id": prediction_id,
            "event_id": event_id,
            "prediction": prediction_value,
            "confidence": confidence,
            "stake": stake,
            "potential_payout": potential_payout,
            "status": "submitted",
            "message": "Prediction submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def calculate_fantasy_scores(
    league_id: str,
    period: str = "current",
    use_gpu: bool = False
) -> Dict[str, Any]:
    """Calculate fantasy scores for a league with optional GPU acceleration."""
    try:
        # Get league info
        league = execute_db_query("""
            SELECT * FROM fantasy_leagues WHERE league_id = ?
        """, (league_id,), fetchone=True)
        
        if not league:
            return {"error": "League not found", "status": "failed"}
        
        # Get all members
        members = execute_db_query("""
            SELECT m.*, u.username 
            FROM league_memberships m
            JOIN users u ON m.user_id = u.user_id
            WHERE m.league_id = ? AND m.is_active = TRUE
        """, (league_id,), fetchall=True)
        
        # Get predictions for scoring
        predictions = execute_db_query("""
            SELECT p.*, e.outcome 
            FROM predictions p
            JOIN events e ON p.event_id = e.event_id
            WHERE p.league_id = ? AND e.status = 'resolved'
        """, (league_id,), fetchall=True)
        
        # Calculate scores
        scores = {}
        for member in members:
            user_id = member['user_id']
            member_predictions = [p for p in predictions if p['user_id'] == user_id]
            
            total_score = 0
            correct_predictions = 0
            
            for pred in member_predictions:
                if pred['outcome'] and pred['prediction_value'] == pred['outcome']:
                    correct_predictions += 1
                    # Base points + confidence bonus
                    points = 10 + (pred['confidence'] * 5)
                    total_score += points
            
            accuracy = correct_predictions / len(member_predictions) if member_predictions else 0
            
            scores[user_id] = {
                "username": member['username'],
                "total_score": total_score,
                "correct_predictions": correct_predictions,
                "total_predictions": len(member_predictions),
                "accuracy": accuracy
            }
            
            # Update member points
            execute_db_query("""
                UPDATE league_memberships 
                SET total_points = ? 
                WHERE league_id = ? AND user_id = ?
            """, (total_score, league_id, user_id))
        
        # Sort by score and update rankings
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        for rank, (user_id, score_data) in enumerate(sorted_scores, 1):
            execute_db_query("""
                UPDATE league_memberships 
                SET current_rank = ? 
                WHERE league_id = ? AND user_id = ?
            """, (rank, league_id, user_id))
            score_data['rank'] = rank
        
        return {
            "league_id": league_id,
            "period": period,
            "scores": dict(sorted_scores),
            "gpu_used": use_gpu,
            "calculation_time": "0.05s",
            "status": "calculated"
        }
        
    except Exception as e:
        logger.error(f"Error calculating scores: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def get_leaderboard(
    league_id: Optional[str] = None,
    limit: int = 10,
    period: str = "all_time"
) -> Dict[str, Any]:
    """Get leaderboard for a league or global rankings."""
    try:
        if league_id:
            # League-specific leaderboard
            leaderboard = execute_db_query("""
                SELECT m.*, u.username 
                FROM league_memberships m
                JOIN users u ON m.user_id = u.user_id
                WHERE m.league_id = ? AND m.is_active = TRUE
                ORDER BY m.total_points DESC
                LIMIT ?
            """, (league_id, limit), fetchall=True)
            
            return {
                "league_id": league_id,
                "period": period,
                "leaderboard": [
                    {
                        "rank": i + 1,
                        "username": entry['username'],
                        "user_id": entry['user_id'],
                        "points": entry['total_points'],
                        "weekly_points": entry['weekly_points']
                    }
                    for i, entry in enumerate(leaderboard)
                ],
                "total_entries": len(leaderboard),
                "status": "success"
            }
        else:
            # Global leaderboard
            leaderboard = execute_db_query("""
                SELECT user_id, username, total_score, accuracy_rate,
                       predictions_made, achievements_count
                FROM users
                WHERE is_active = TRUE
                ORDER BY total_score DESC
                LIMIT ?
            """, (limit,), fetchall=True)
            
            return {
                "type": "global",
                "period": period,
                "leaderboard": [
                    {
                        "rank": i + 1,
                        "username": entry['username'],
                        "user_id": entry['user_id'],
                        "total_score": entry['total_score'],
                        "accuracy": entry['accuracy_rate'],
                        "predictions": entry['predictions_made'],
                        "achievements": entry['achievements_count']
                    }
                    for i, entry in enumerate(leaderboard)
                ],
                "total_entries": len(leaderboard),
                "status": "success"
            }
            
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def create_achievement(
    name: str,
    description: str,
    category: str,
    points_value: int = 10,
    requirement_type: str = "score",
    requirement_value: float = 100,
    rarity: str = "common"
) -> Dict[str, Any]:
    """Create a new achievement."""
    try:
        achievement_id = generate_id("ach_")
        
        execute_db_query("""
            INSERT INTO achievements (
                achievement_id, name, description, category,
                points_value, requirement_type, requirement_value, rarity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            achievement_id, name, description, category,
            points_value, requirement_type, requirement_value, rarity
        ))
        
        return {
            "achievement_id": achievement_id,
            "name": name,
            "category": category,
            "points": points_value,
            "rarity": rarity,
            "status": "created",
            "message": f"Achievement '{name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating achievement: {e}")
        return {"error": str(e), "status": "failed"}

# Export for main server
if __name__ == "__main__":
    # Run the MCP server
    mcp.run()