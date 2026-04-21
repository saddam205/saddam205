"""
WebSocket routes for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import random
from datetime import datetime

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send market update
            data = {
                "type": "market_update",
                "timestamp": datetime.now().isoformat(),
                "price": round(50000 * (1 + random.uniform(-0.02, 0.02)), 2),
                "signal": random.choice(["BUY", "SELL", "HOLD"]),
                "confidence": round(random.uniform(0.6, 0.9), 2)
            }
            await manager.broadcast(data)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
