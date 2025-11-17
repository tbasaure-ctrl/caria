"""
WebSocket Chat Service - Implementación según documento de auditoría.
Solución de 3 partes:
1. Autenticación JWT en handshake
2. Heartbeat Ping/Pong (ping_interval=25, ping_timeout=60)
3. Recuperación de historial en reconexión
"""

import logging
import os
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import socketio
from fastapi import HTTPException

from caria.services.auth_service import AuthService

LOGGER = logging.getLogger("caria.api.websocket")

# Create SocketIO server with heartbeat configuration per audit document
# ping_interval=25: Server sends PING every 25 seconds
# ping_timeout=60: Server waits 60 seconds for PONG response
sio = socketio.AsyncServer(
    cors_allowed_origins="*",  # Configure per environment
    ping_interval=25,  # Per audit document - Problem #2
    ping_timeout=60,   # Per audit document - Problem #2
    async_mode='asgi'
)

# Store for chat messages (in production, use Redis or database)
# Format: {user_id: [messages]}
chat_history: dict[str, list[dict]] = {}

# Store for user sessions
# Format: {session_id: user_id}
user_sessions: dict[str, str] = {}


def get_auth_service() -> AuthService:
    """Get AuthService instance with database connection."""
    import psycopg2
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),  # Use 'postgres' for Docker
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=password,
        database=os.getenv("POSTGRES_DB", "caria"),
    )
    return AuthService(conn)


@sio.on('connect')
async def handle_connect(sid: str, environ: dict, auth: Optional[dict]):
    """
    Handle WebSocket connection with JWT authentication per audit document.
    Problem #1: Authentication in handshake
    """
    LOGGER.info(f"WebSocket connection attempt from session {sid}")
    
    # Extract token from auth dict (sent by client)
    token = None
    if auth:
        token = auth.get('token')
    
    # If no token in auth, try to get from query string (fallback)
    if not token:
        query_string = environ.get('QUERY_STRING', '')
        if 'token=' in query_string:
            token = query_string.split('token=')[1].split('&')[0]
    
    if not token:
        LOGGER.warning(f"WebSocket connection rejected: No token provided for session {sid}")
        await sio.emit('connect_error', {'message': 'Authentication required'}, room=sid)
        return False  # Reject connection
    
    # Validate token using AuthService
    try:
        auth_service = get_auth_service()
        token_payload = AuthService.decode_token(token)
        user_id = UUID(token_payload.sub)
        
        # Verify user exists and is active
        user = auth_service.get_user_by_id(user_id)
        if not user:
            LOGGER.warning(f"WebSocket connection rejected: User not found for token")
            await sio.emit('connect_error', {'message': 'Invalid token'}, room=sid)
            return False
        
        # Store session mapping
        user_sessions[sid] = str(user_id)
        
        # Initialize chat history for user if not exists
        if str(user_id) not in chat_history:
            chat_history[str(user_id)] = []
        
        LOGGER.info(f"WebSocket connected: session {sid} -> user {user.username} ({user_id})")
        
        # Emit connection success
        await sio.emit('connect', {'message': 'Connected and authenticated'}, room=sid)
        
        return True  # Accept connection
        
    except ValueError as e:
        LOGGER.warning(f"WebSocket connection rejected: Invalid token - {e}")
        await sio.emit('connect_error', {'message': 'Invalid token'}, room=sid)
        return False
    except Exception as e:
        LOGGER.exception(f"WebSocket connection error: {e}")
        await sio.emit('connect_error', {'message': 'Authentication failed'}, room=sid)
        return False


@sio.on('disconnect')
async def handle_disconnect(sid: str):
    """Handle WebSocket disconnection."""
    user_id = user_sessions.pop(sid, None)
    if user_id:
        LOGGER.info(f"WebSocket disconnected: session {sid} -> user {user_id}")
    else:
        LOGGER.info(f"WebSocket disconnected: session {sid} (no user mapping)")


@sio.on('chat_message')
async def handle_chat_message(sid: str, data: dict):
    """
    Handle incoming chat message.
    Store message in history for recovery on reconnection (Problem #3).
    """
    user_id = user_sessions.get(sid)
    if not user_id:
        await sio.emit('error', {'message': 'Not authenticated'}, room=sid)
        return
    
    message_text = data.get('message', '').strip()
    if not message_text:
        return
    
    # Create message record
    message_record = {
        'id': str(uuid4()),
        'user_id': user_id,
        'message': message_text,
        'timestamp': datetime.utcnow().isoformat(),
        'session_id': sid,
    }
    
    # Store in history
    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append(message_record)
    
    # Keep only last 100 messages per user
    if len(chat_history[user_id]) > 100:
        chat_history[user_id] = chat_history[user_id][-100:]
    
    LOGGER.debug(f"Chat message from user {user_id}: {message_text[:50]}...")
    
    # Echo message back (in production, process with AI/LLM)
    await sio.emit('chat_message', {
        'id': message_record['id'],
        'message': f"Received: {message_text}",  # Placeholder - replace with actual AI response
        'timestamp': message_record['timestamp'],
        'role': 'assistant',
    }, room=sid)


def get_chat_history(user_id: str, since_timestamp: Optional[str] = None) -> list[dict]:
    """
    Get chat history for user since a timestamp.
    Used for recovery on reconnection (Problem #3).
    """
    user_messages = chat_history.get(user_id, [])
    
    if since_timestamp:
        # Filter messages after the timestamp
        try:
            since_dt = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
            filtered = [
                msg for msg in user_messages
                if datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')) > since_dt
            ]
            return filtered
        except Exception as e:
            LOGGER.warning(f"Error parsing timestamp {since_timestamp}: {e}")
            return user_messages[-50:]  # Return last 50 if parsing fails
    
    # Return all messages (or last 50)
    return user_messages[-50:]

