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
    from urllib.parse import urlparse, parse_qs
    
    # Try DATABASE_URL first (Cloud SQL format)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)
            
            # Check for Unix socket (Cloud SQL)
            unix_socket_host = query_params.get('host', [None])[0]
            
            if unix_socket_host:
                # Use Cloud SQL Unix socket
                conn = psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                # Use normal TCP connection
                conn = psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            else:
                raise ValueError("Invalid DATABASE_URL format")
        except Exception as e:
            LOGGER.warning(f"Error using DATABASE_URL: {e}. Falling back to individual vars...")
            database_url = None
    
    # Fallback to individual environment variables
    if not database_url:
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "caria_user"),
            password=password,
            database=os.getenv("POSTGRES_DB", "caria"),
        )
    
    return AuthService(conn)



# Global LLM Service instance
llm_service = None


def set_llm_service(service):
    """Set LLM service instance."""
    global llm_service
    llm_service = service


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
    Uses LLMService for RAG + Generation.
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
    
    # Generate AI response
    ai_response_text = "I'm sorry, I'm having trouble connecting to my brain right now."
    
    if llm_service:
        try:
            # 1. Get RAG context
            evidence_text, chunks = llm_service.get_rag_context(message_text)
            
            # 2. Build prompt
            system_prompt = (
                "You are Caria, an elite AI investment assistant designed for sophisticated investors. "
                "Your goal is to provide data-driven, objective financial analysis. "
                "Use the provided RAG context to answer the user's question with precision. "
                "If the context is insufficient, leverage your general financial knowledge but explicitly state that the answer is not based on the retrieved documents. "
                "Always maintain a professional, institutional-grade tone. "
                "Highlight risks where appropriate. "
                "Do not provide financial advice; instead, provide analysis to support decision-making."
            )
            
            prompt = f"""Context from knowledge base:
{evidence_text}

User Question:
{message_text}

Answer:"""
            
            # 3. Call LLM
            # Note: LLMService is synchronous, but we are in async function.
            # For high load, this should be run in executor, but for now direct call is acceptable
            # or we can use run_in_executor if we want to be strictly non-blocking.
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm_service.call_llm, prompt, system_prompt)
            
            if response:
                ai_response_text = response
            else:
                ai_response_text = "I couldn't generate a response at this time."
                
        except Exception as e:
            LOGGER.exception(f"Error generating AI response: {e}")
            ai_response_text = "An error occurred while processing your request."
    else:
        LOGGER.warning("LLMService not initialized in WebSocket chat")
    
    # Emit response
    response_record = {
        'id': str(uuid4()),
        'message': ai_response_text,
        'timestamp': datetime.utcnow().isoformat(),
        'role': 'assistant',
    }
    
    # Store response in history too
    chat_history[user_id].append({
        **response_record,
        'user_id': 'ai_assistant',
        'session_id': sid
    })
    
    await sio.emit('chat_message', response_record, room=sid)


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


