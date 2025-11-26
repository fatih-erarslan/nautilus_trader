#!/usr/bin/env python3
"""
Swarm Streaming API for real-time swarm updates
Provides WebSocket and SSE endpoints for live monitoring
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
import json
import uuid
from datetime import datetime
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/swarm/stream", tags=["Swarm Streaming"])

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Swarm event queue
event_queues: Dict[str, asyncio.Queue] = {}

class ConnectionManager:
    """Manages WebSocket connections for swarm streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time swarm updates.
    Connect to receive live updates from a swarm session.
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    # Create event queue for this client
    event_queue = asyncio.Queue()
    event_queues[client_id] = event_queue
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "session_id": session_id,
            "client_id": client_id,
            "message": "Connected to swarm stream",
            "timestamp": datetime.now().isoformat()
        })
        
        # Simulate swarm updates
        while True:
            # Check for queued events
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Simulate swarm status update every 5 seconds
            await asyncio.sleep(5)
            await websocket.send_json({
                "type": "swarm_update",
                "session_id": session_id,
                "status": "running",
                "agents_active": 5,
                "progress": 0.65,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if client_id in event_queues:
            del event_queues[client_id]

@router.get("/sse/{session_id}")
async def server_sent_events(session_id: str):
    """
    Server-Sent Events endpoint for swarm updates.
    Alternative to WebSocket for simpler streaming.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events"""
        client_id = str(uuid.uuid4())
        event_queue = asyncio.Queue()
        event_queues[client_id] = event_queue
        
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'session_id': session_id, 'timestamp': datetime.now().isoformat()})}\n\n"
            
            while True:
                # Check for queued events
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Simulate periodic updates
                await asyncio.sleep(10)
                yield f"data: {json.dumps({'type': 'swarm_update', 'session_id': session_id, 'status': 'running', 'timestamp': datetime.now().isoformat()})}\n\n"
                
        except asyncio.CancelledError:
            if client_id in event_queues:
                del event_queues[client_id]
            logger.info(f"SSE client {client_id} disconnected")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/broadcast/{session_id}")
async def broadcast_event(session_id: str, event: Dict[str, Any]):
    """
    Broadcast an event to all connected clients watching a session.
    Used internally by swarm orchestrators.
    """
    message = json.dumps({
        "type": "broadcast",
        "session_id": session_id,
        "event": event,
        "timestamp": datetime.now().isoformat()
    })
    
    # Send to all WebSocket clients
    await manager.broadcast(message)
    
    # Queue for SSE clients
    for queue in event_queues.values():
        await queue.put({
            "type": "broadcast",
            "session_id": session_id,
            "event": event,
            "timestamp": datetime.now().isoformat()
        })
    
    return {
        "status": "broadcast_sent",
        "session_id": session_id,
        "clients_notified": len(manager.active_connections) + len(event_queues)
    }

@router.get("/connections")
async def get_active_connections():
    """
    Get information about active streaming connections.
    """
    return {
        "websocket_clients": len(manager.active_connections),
        "sse_clients": len(event_queues),
        "total_connections": len(manager.active_connections) + len(event_queues),
        "timestamp": datetime.now().isoformat()
    }