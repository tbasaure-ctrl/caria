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

from api.dependencies import open_db_connection
from caria.services.auth_service import AuthService

LOGGER = logging.getLogger("caria.api.websocket")

# Create SocketIO server with heartbeat configuration per audit document
# ping_interval=25: Server sends PING every 25 seconds
# ping_timeout=60: Server waits 60 seconds for PONG response
sio = socketio.AsyncServer(
    cors_allowed_origins="*",  # Configure per environment
    ping_interval=25,  # Per audit document - Problem #2
    ping_timeout=60,   # Per audit document - Problem #2
    async_mode='asgi',
    logger=True,  # Enable logging for debugging
    engineio_logger=True,  # Enable engine.io logging
    allow_upgrades=True,  # Allow transport upgrades
    transports=['polling', 'websocket'],  # Supported transports
)

# Store for chat messages (in production, use Redis or database)
# Format: {user_id: [messages]}
chat_history: dict[str, list[dict]] = {}

# Store for user sessions
# Format: {session_id: user_id}
user_sessions: dict[str, str] = {}


def get_auth_service() -> AuthService:
    """Get AuthService instance with database connection."""
    conn = open_db_connection()
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
    
    try:
        # Extract token from auth dict (sent by client)
        token = None
        if auth and isinstance(auth, dict):
            token = auth.get('token')
        
        # If no token in auth, try to get from query string (fallback)
        if not token:
            query_string = environ.get('QUERY_STRING', '')
            if 'token=' in query_string:
                token = query_string.split('token=')[1].split('&')[0]
        
        if not token:
            LOGGER.warning(f"WebSocket connection rejected: No token provided for session {sid}")
            # Don't emit error here - just reject the connection
            # Socket.IO will handle the rejection properly
            return False  # Reject connection
    
        # Validate token using AuthService
        auth_service = get_auth_service()
        token_payload = AuthService.decode_token(token)
        user_id = UUID(token_payload.sub)
        
        # Verify user exists and is active
        user = auth_service.get_user_by_id(user_id)
        if not user:
            LOGGER.warning(f"WebSocket connection rejected: User not found for token")
            return False  # Reject connection - don't emit error, Socket.IO handles it
        
        # Store session mapping
        user_sessions[sid] = str(user_id)
        
        # Initialize chat history for user if not exists
        if str(user_id) not in chat_history:
            chat_history[str(user_id)] = []
        
        LOGGER.info(f"WebSocket connected: session {sid} -> user {user.username} ({user_id})")
        
        # Connection is successful - Socket.IO automatically emits 'connect' event to client
        return True  # Accept connection
        
    except ValueError as e:
        LOGGER.warning(f"WebSocket connection rejected: Invalid token - {e}")
        # Don't emit error - just reject, Socket.IO handles it
        return False
    except Exception as e:
        LOGGER.exception(f"WebSocket connection error: {e}")
        # Don't emit error - just reject, Socket.IO handles it
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
            # 1. Detect language and companies mentioned
            user_language = _detect_language(message_text)
            mentioned_companies = _extract_companies(message_text)
            
            # 2. Get conversation history for context
            conversation_context = _build_conversation_context(chat_history.get(user_id, []), max_messages=10)
            
            # 3. Get RAG context (enhanced with company-specific queries if companies detected)
            rag_query = message_text
            if mentioned_companies:
                # Enhance query with company context
                rag_query = f"{message_text} [Companies mentioned: {', '.join(mentioned_companies)}]"
            
            evidence_text, chunks = llm_service.get_rag_context(rag_query)
            
            # 4. Build Socratic, conversational prompt
            system_prompt = _build_socratic_system_prompt(user_language, mentioned_companies)
            
            # 5. Build conversational prompt with context
            prompt = _build_conversational_prompt(
                message_text,
                evidence_text,
                conversation_context,
                mentioned_companies,
                user_language
            )
            
            # 6. Call LLM
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm_service.call_llm, prompt, system_prompt)
            
            if response:
                ai_response_text = response.strip()
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


def _detect_language(text: str) -> str:
    """Detect user's language from message text."""
    # Simple heuristic: check for Spanish indicators
    spanish_indicators = ['qué', 'cómo', 'cuál', 'dónde', 'por qué', 'tiene', 'crees', 'va a']
    text_lower = text.lower()
    if any(indicator in text_lower for indicator in spanish_indicators):
        return "es"
    return "en"


def _extract_companies(text: str) -> list[str]:
    """Extract potential company tickers/names from text."""
    import re
    # Common ticker pattern: 1-5 uppercase letters
    tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
    # Filter out common words that aren't tickers
    common_words = {'THE', 'AND', 'OR', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
    tickers = [t for t in tickers if t not in common_words and len(t) >= 1]
    
    # Also look for common company names
    company_names = ['apple', 'microsoft', 'google', 'amazon', 'nvidia', 'meta', 'tesla', 'nvidia']
    found_names = []
    text_lower = text.lower()
    for name in company_names:
        if name in text_lower:
            found_names.append(name.upper())
    
    return list(set(tickers + found_names))


def _build_conversation_context(messages: list[dict], max_messages: int = 10) -> str:
    """Build conversation context from recent messages."""
    if not messages:
        return ""
    
    recent_messages = messages[-max_messages:]
    context_parts = []
    for msg in recent_messages:
        role = msg.get('role', 'user')
        content = msg.get('message', '')
        if role == 'user':
            context_parts.append(f"User: {content}")
        elif role == 'assistant':
            context_parts.append(f"Assistant: {content}")
    
    return "\n".join(context_parts)


def _build_socratic_system_prompt(language: str, companies: list[str]) -> str:
    """Build a Socratic, conversational system prompt with 13-point framework and thesis filter."""
    lang_instruction = "Responde en español." if language == "es" else "Respond in English."
    
    company_context = ""
    if companies:
        company_context = f"\nThe user has mentioned: {', '.join(companies)}. If relevant, reference specific metrics or facts about these companies naturally in conversation."
    
    return f"""You are Caria, an institutional-grade AI investment partner. Your goal is to help the user refine their thinking through precise, high-value dialogue.

Core Directives:
1. **Language Strictness**: You MUST respond in the SAME language as the user. If they speak Spanish, respond in Spanish. If English, English. Do not mix languages.
   - User Language Detected: {language.upper()}
   - Your Output Language: {language.upper()}

2. **The "Thesis Filter" (CRITICAL)**:
   - When a user starts a conversation about a stock/position, check if they have a "thesis" (a reason for buying/selling).
   - **IF NO THESIS**: If the user provides NO reason or argument (e.g., "Should I buy AAPL?", "What do you think of NVDA?"), you MUST politely but firmly END the conversation with a message similar to:
     "If you can't provide a single argument on why you are considering this position, my advice is not to. It is easy for me to provide data or give you my opinion, but if you don't understand why you are buying, you are the one facing the downside risk. Come back more informed, and I'll be happy to help you reach a conclusion that allows you to sleep well at night knowing you own a great business."
   - **IF THESIS EXISTS**: Proceed to challenge and refine it using the 13-Point Framework.

3. **Conversation Flow (3-4 Interactions)**:
   - Do not drag on. Aim to reach a conclusion within 3-4 interactions.
   - **Goal**: Help the user decide whether to "back up the investment thesis" or "expose points on why not to".
   - Be polite but efficient.

4. **The 13-Point Framework**:
   Use this framework to analyze the stock and guide your questions/conclusions. Do not list these points explicitly unless asked, but use them to structure your thinking and challenges.
   
   1. **What They Sell and Who Buys**: Products, target customers, pain points.
   2. **How They Make Money**: Revenue model, pricing, recurring vs one-time.
   3. **Revenue Quality**: Predictability, diversification, concentration.
   4. **Cost Structure**: COGS, labor, margins, scalability.
   5. **Capital Intensity**: Capex, working capital, cash conversion.
   6. **Growth Drivers**: Volume, pricing, mix, structural vs cyclical.
   7. **Competitive Edge**: Moat (brand, cost, switching costs), durability.
   8. **Industry Structure**: Value chain, market share, pricing power.
   9. **Unit Economics**: CAC, LTV, churn, ARPU.
   10. **Capital Allocation**: Buybacks, dividends, debt, balance sheet strength.
   11. **Risks**: Competitive, regulatory, macro, failure modes.
   12. **Valuation**: P/E, FCF yield, bear/base/bull cases.
   13. **Catalysts**: Near/medium term events, time horizon.

5. **Tone & Style**:
   - Analytical, neutral, precise. No filler or marketing language.
   - Professional, like a senior analyst.
   - **Opinionated**: Don't just ask questions. If the data suggests a bad investment, say so (politely).

{company_context}

Remember: You are a partner, not a search engine. Apply the Thesis Filter first. If they pass, guide them to a conclusion in 3-4 steps using the 13 points. Keep it brief and high-impact."""


def _build_conversational_prompt(
    user_message: str,
    rag_context: str,
    conversation_history: str,
    companies: list[str],
    language: str
) -> str:
    """Build a natural, conversational prompt."""
    parts = []
    
    if conversation_history:
        parts.append(f"Previous conversation:\n{conversation_history}\n")
    
    if rag_context and rag_context.strip():
        parts.append(f"Relevant context:\n{rag_context}\n")
    
    if companies:
        parts.append(f"Companies mentioned: {', '.join(companies)}\n")
    
    parts.append(f"User's current message:\n{user_message}\n")
    
    parts.append("\nRespond naturally and conversationally. Use the Socratic method—ask questions, challenge thinking, guide the user. Be engaging, not robotic.")
    
    return "\n".join(parts)


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


