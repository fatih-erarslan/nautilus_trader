"""
Portfolio management handler for Beefy Finance
"""

import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import aiohttp

logger = logging.getLogger(__name__)

class PortfolioHandler:
    """Handle portfolio management and tracking"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Ensure tables exist
            with open('/workspaces/ai-news-trader/src/crypto_trading/database/schema.sql', 'r') as f:
                schema = f.read()
                self.conn.executescript(schema)
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def add_position(
        self,
        vault_id: str,
        amount: float,
        tx_hash: str
    ) -> int:
        """Add a new position to the portfolio"""
        
        try:
            # Fetch current vault data
            vault_data = await self._fetch_vault_data(vault_id)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO vault_positions 
                (vault_id, vault_name, chain, amount_deposited, shares_owned, 
                 current_value, entry_price, entry_apy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vault_id,
                vault_data['name'],
                vault_data['chain'],
                amount,
                vault_data['estimated_shares'],
                amount,  # Initial value equals deposit
                vault_data['price_per_share'],
                vault_data['apy'],
                'active'
            ))
            
            position_id = cursor.lastrowid
            
            # Log transaction
            cursor.execute("""
                INSERT INTO crypto_transactions
                (transaction_type, vault_id, chain, amount, tx_hash, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                'deposit',
                vault_id,
                vault_data['chain'],
                amount,
                tx_hash,
                'confirmed'
            ))
            
            self.conn.commit()
            logger.info(f"Added position {position_id} for vault {vault_id}")
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            self.conn.rollback()
            raise
    
    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM active_positions_summary
            ORDER BY current_value DESC
        """)
        
        positions = []
        for row in cursor.fetchall():
            position = dict(row)
            
            # Update current value from API
            current_data = await self._fetch_vault_data(position['vault_id'])
            position['current_value'] = position['shares_owned'] * current_data['price_per_share']
            position['current_apy'] = current_data['apy']
            
            positions.append(position)
        
        return positions
    
    async def get_harvestable_yields(
        self,
        vault_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get harvestable yields for vaults"""
        
        yields = {}
        
        # Get positions
        if vault_ids:
            placeholders = ','.join('?' * len(vault_ids))
            query = f"""
                SELECT * FROM vault_positions 
                WHERE vault_id IN ({placeholders}) AND status = 'active'
            """
            cursor = self.conn.cursor()
            cursor.execute(query, vault_ids)
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM vault_positions WHERE status = 'active'")
        
        for row in cursor.fetchall():
            position = dict(row)
            vault_id = position['vault_id']
            
            # Calculate yields
            current_data = await self._fetch_vault_data(vault_id)
            current_value = position['shares_owned'] * current_data['price_per_share']
            
            # Get total earned from yield history
            cursor.execute("""
                SELECT SUM(earned_amount) as total_earned
                FROM yield_history
                WHERE position_id = ?
            """, (position['id'],))
            
            result = cursor.fetchone()
            total_earned = result['total_earned'] or 0
            
            # Calculate harvestable (current value - deposited - already harvested)
            harvestable = current_value - position['amount_deposited'] - total_earned
            
            yields[vault_id] = {
                "position_id": position['id'],
                "harvestable": max(0, harvestable),
                "total_earned": total_earned,
                "current_value": current_value,
                "apy": current_data['apy']
            }
        
        return yields
    
    async def record_harvest(
        self,
        vault_id: str,
        amount: float,
        tx_hash: str
    ) -> None:
        """Record a yield harvest"""
        
        try:
            # Get position
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id FROM vault_positions 
                WHERE vault_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """, (vault_id,))
            
            position = cursor.fetchone()
            if not position:
                raise ValueError(f"No active position for vault {vault_id}")
            
            # Get current vault data
            vault_data = await self._fetch_vault_data(vault_id)
            
            # Record yield
            cursor.execute("""
                INSERT INTO yield_history
                (vault_id, position_id, earned_amount, apy_snapshot, 
                 tvl_snapshot, price_per_share)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                vault_id,
                position['id'],
                amount,
                vault_data['apy'],
                vault_data['tvl'],
                vault_data['price_per_share']
            ))
            
            # Log transaction
            cursor.execute("""
                INSERT INTO crypto_transactions
                (transaction_type, vault_id, chain, amount, tx_hash, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                'claim',
                vault_id,
                vault_data['chain'],
                amount,
                tx_hash,
                'confirmed'
            ))
            
            self.conn.commit()
            logger.info(f"Recorded harvest of {amount} from vault {vault_id}")
            
        except Exception as e:
            logger.error(f"Error recording harvest: {e}")
            self.conn.rollback()
            raise
    
    async def update_position(
        self,
        vault_id: str,
        action: Dict[str, Any]
    ) -> None:
        """Update a position after rebalancing"""
        
        try:
            cursor = self.conn.cursor()
            
            if action['type'] == 'withdraw':
                # Reduce position
                cursor.execute("""
                    UPDATE vault_positions
                    SET shares_owned = shares_owned - ?,
                        current_value = current_value - ?
                    WHERE vault_id = ? AND status = 'active'
                """, (
                    action['shares'],
                    action['amount'],
                    vault_id
                ))
            else:  # deposit
                # Increase position or create new one
                cursor.execute("""
                    SELECT id FROM vault_positions
                    WHERE vault_id = ? AND status = 'active'
                    LIMIT 1
                """, (vault_id,))
                
                position = cursor.fetchone()
                if position:
                    # Update existing
                    cursor.execute("""
                        UPDATE vault_positions
                        SET shares_owned = shares_owned + ?,
                            amount_deposited = amount_deposited + ?
                        WHERE id = ?
                    """, (
                        action['shares'],
                        action['amount'],
                        position['id']
                    ))
                else:
                    # Create new position
                    await self.add_position(
                        vault_id,
                        action['amount'],
                        action.get('tx_hash', '0x0')
                    )
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            self.conn.rollback()
            raise
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        
        positions = await self.get_all_positions()
        
        total_value = sum(p['current_value'] for p in positions)
        total_deposited = sum(p['amount_deposited'] for p in positions)
        total_earned = sum(p['total_earned'] for p in positions)
        
        # Calculate weighted average APY
        weighted_apy = 0
        if total_value > 0:
            weighted_apy = sum(
                p['current_apy'] * p['current_value'] / total_value 
                for p in positions
            )
        
        # Get unique chains
        chains = list(set(p['chain'] for p in positions))
        
        # Store summary
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio_summary
            (total_value_usd, total_yield_earned, average_apy, 
             chains_active, vaults_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            total_value,
            total_earned,
            weighted_apy,
            json.dumps(chains),
            len(positions)
        ))
        self.conn.commit()
        
        return {
            "total_value": total_value,
            "total_deposited": total_deposited,
            "total_earned": total_earned,
            "unrealized_pnl": total_value - total_deposited,
            "average_apy": weighted_apy,
            "active_vaults": len(positions),
            "chains": chains,
            "positions": positions
        }
    
    async def _fetch_vault_data(self, vault_id: str) -> Dict[str, Any]:
        """Fetch current vault data from API"""
        
        async with aiohttp.ClientSession() as session:
            # Get vault info
            async with session.get("https://api.beefy.finance/vaults") as response:
                vaults = await response.json()
                vault_info = vaults.get(vault_id, {})
            
            # Get APY
            async with session.get("https://api.beefy.finance/apy") as response:
                apy_data = await response.json()
                apy = apy_data.get(vault_id, {}).get('totalApy', 0) * 100
            
            # Get TVL
            async with session.get("https://api.beefy.finance/tvl") as response:
                tvl_data = await response.json()
                tvl = tvl_data.get(vault_id, 0)
            
            # Simulate price per share (would come from contract call)
            price_per_share = 1.15  # Example
            
            # Estimate shares based on standard calculation
            estimated_shares = 100  # Simplified
            
            return {
                "name": vault_info.get('name', vault_id),
                "chain": vault_info.get('chain', 'bsc'),
                "apy": apy,
                "tvl": tvl,
                "price_per_share": price_per_share,
                "estimated_shares": estimated_shares
            }
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()