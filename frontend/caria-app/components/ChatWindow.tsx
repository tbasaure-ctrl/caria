import React, { useEffect, useState, useRef } from 'react';
import { API_BASE_URL, getToken } from '../services/apiService';

// Import socket.io-client - if it fails, the app will still work without chat
import { io as socketIO, Socket } from 'socket.io-client';

interface Message {
    id: string;
    message: string;
    timestamp: string;
    role: 'user' | 'assistant';
}

interface ChatWindowProps {
    onClose?: () => void;
    initialMessage?: string;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ onClose, initialMessage }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [lastMessageTimestamp, setLastMessageTimestamp] = useState<string | null>(null);
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'reconnecting' | 'error' | 'unauthenticated'>(
        'connecting'
    );
    const [connectionMessage, setConnectionMessage] = useState('Connecting to Caria...');
    const [showDebug, setShowDebug] = useState(false);
    const [debugInfo, setDebugInfo] = useState<any>(null);
    const socketRef = useRef<Socket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const reconnectAttemptsRef = useRef(0);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Initialize WebSocket connection with JWT authentication
    useEffect(() => {
        const connectSocket = async () => {
            try {
                const token = getToken();
                if (!token) {
                    console.warn('No auth token available for WebSocket connection');
                    setConnectionStatus('unauthenticated');
                    setConnectionMessage('Please log in to chat with Caria.');
                    return;
                }

                setConnectionStatus('connecting');
                setConnectionMessage('Connecting to Caria...');

                // Connect with JWT token in auth object (per audit document Problem #1)
                // Socket.IO connects to the base URL (without /api), e.g., http://localhost:8000
                let socketBaseUrl = API_BASE_URL;
                if (socketBaseUrl.includes('/api')) {
                    socketBaseUrl = socketBaseUrl.replace('/api', '');
                }
                if (!socketBaseUrl || socketBaseUrl === '') {
                    socketBaseUrl = 'http://localhost:8000';
                }

                // Ensure URL doesn't have trailing slash
                socketBaseUrl = socketBaseUrl.replace(/\/$/, '');

                const socket = socketIO(socketBaseUrl, {
                    auth: {
                        token: token
                    },
                    path: '/socket.io/',
                    transports: ['polling', 'websocket'], // Try polling first, then upgrade to websocket
                    reconnection: true,
                    reconnectionAttempts: 5,
                    reconnectionDelay: 1000,
                    reconnectionDelayMax: 6000,
                    randomizationFactor: 0.5,
                    timeout: 20000, // 20 second timeout
                    forceNew: false, // Reuse existing connection if available
                    autoConnect: true,
                });

                socketRef.current = socket;

                // Handle connection success
                socket.on('connect', async () => {
                    console.log('WebSocket connected');
                    setConnectionStatus('connected');
                    setConnectionMessage('Connected');
                    reconnectAttemptsRef.current = 0;

                    // Problem #3: Recover lost messages on reconnection
                    try {
                        // Fix: API_BASE_URL already contains the base, just append the path
                        // Avoid double /api by checking if API_BASE_URL already ends with /api
                        const chatHistoryUrl = API_BASE_URL.endsWith('/api') 
                            ? `${API_BASE_URL}/chat/history${lastMessageTimestamp ? `?since=${lastMessageTimestamp}` : ''}`
                            : `${API_BASE_URL}/api/chat/history${lastMessageTimestamp ? `?since=${lastMessageTimestamp}` : ''}`;
                        
                        const response = await fetch(chatHistoryUrl, {
                            headers: {
                                'Authorization': `Bearer ${token}`
                            }
                        });
                        if (response.ok) {
                            const data = await response.json();
                            if (data.messages && data.messages.length > 0) {
                                const recoveredMessages: Message[] = data.messages.map((msg: any) => ({
                                    id: msg.id,
                                    message: msg.message,
                                    timestamp: msg.timestamp,
                                    role: msg.role || 'assistant'
                                }));
                                setMessages(prev => [...prev, ...recoveredMessages]);
                                // Update last timestamp
                                if (recoveredMessages.length > 0) {
                                    setLastMessageTimestamp(recoveredMessages[recoveredMessages.length - 1].timestamp);
                                }
                            }
                        }
                    } catch (error) {
                        console.error('Error recovering chat history:', error);
                    }
                });

                // Handle connection error
                socket.on('connect_error', (error: any) => {
                    console.error('WebSocket connection error:', error);
                    const errorMessage = error?.message || 'Connection error';
                    
                    // Check if it's an authentication error
                    if (errorMessage.includes('Authentication') || errorMessage.includes('Invalid token') || errorMessage.includes('401')) {
                        setConnectionStatus('error');
                        setConnectionMessage('Authentication failed. Please log in again.');
                        // Don't attempt to reconnect if auth failed
                        socket.disconnect();
                        return;
                    }
                    
                    setConnectionStatus('reconnecting');
                    setConnectionMessage(`Connection error: ${errorMessage}. Retrying...`);
                });

                // Handle disconnection
                socket.on('disconnect', (reason: Socket.DisconnectReason) => {
                    console.log('WebSocket disconnected:', reason);
                    
                    // Don't reconnect if server disconnected us (likely auth issue)
                    if (reason === 'io server disconnect') {
                        setConnectionStatus('error');
                        setConnectionMessage('Disconnected by server. Please refresh the page.');
                        return;
                    }
                    
                    // Don't reconnect if client disconnected intentionally
                    if (reason === 'io client disconnect') {
                        setConnectionStatus('error');
                        setConnectionMessage('Connection closed.');
                        return;
                    }
                    
                    // For other reasons (network issues, etc.), try to reconnect
                    setConnectionStatus('reconnecting');
                    setConnectionMessage('Connection lost. Reconnecting...');
                });

                socket.io.on('reconnect_attempt', (attempt: number) => {
                    reconnectAttemptsRef.current = attempt;
                    setConnectionStatus('reconnecting');
                    setConnectionMessage(`Reconnecting (${attempt}/5)...`);
                });

                socket.io.on('reconnect_failed', () => {
                    setConnectionStatus('error');
                    setConnectionMessage('Could not reconnect. Please refresh the page or check your connection.');
                });

                // Handle parse errors (common Socket.IO issue)
                socket.io.on('error', (error: any) => {
                    console.error('Socket.IO error:', error);
                    if (error?.message?.includes('parse')) {
                        setConnectionStatus('error');
                        setConnectionMessage('Connection error. Please refresh the page.');
                    }
                });

                // Handle incoming chat messages
                socket.on('chat_message', (data: any) => {
                    const msg: Message = {
                        id: data.id || Date.now().toString(),
                        message: data.message || data.response || '',
                        timestamp: data.timestamp || new Date().toISOString(),
                        role: data.role || 'assistant'
                    };
                    setMessages(prev => {
                        const newMessages = [...prev, msg];
                        setLastMessageTimestamp(msg.timestamp);
                        return newMessages;
                    });
                    // Store debug info if available
                    if (data.debug) {
                        setDebugInfo({
                            request: data.debug.request,
                            response: data.debug.response,
                            latency_ms: data.debug.latency_ms,
                            tokens_used: data.debug.tokens_used,
                            timestamp: msg.timestamp
                        });
                    }
                });

                // Handle errors
                socket.on('error', (error: any) => {
                    console.error('WebSocket error:', error);
                });

            } catch (error) {
                console.error('Error initializing WebSocket:', error);
                setConnectionStatus('error');
                setConnectionMessage('Could not initialize connection. Chat service may be unavailable.');
            }
        };

        // Only connect if component is mounted
        connectSocket();

        // Cleanup on unmount
        return () => {
            if (socketRef.current) {
                try {
                    socketRef.current.disconnect();
                } catch (error) {
                    console.error('Error disconnecting socket:', error);
                }
                socketRef.current = null;
            }
        };
    }, []);

    // Send initial message if provided and connected
    useEffect(() => {
        if (initialMessage && connectionStatus === 'connected' && socketRef.current) {
            // Small delay to ensure socket is ready
            const timer = setTimeout(() => {
                // Check if we haven't already sent this message (simple check)
                const alreadySent = messages.some(m => m.role === 'user' && m.message === initialMessage);
                if (!alreadySent) {
                    setInputValue(initialMessage);
                    // Optionally auto-send: handleSendMessage();
                }
            }, 500);
            return () => clearTimeout(timer);
        }
    }, [initialMessage, connectionStatus]);

    const handleSendMessage = async () => {
        const isReady = connectionStatus === 'connected';
        if (!inputValue.trim() || !socketRef.current || !isReady) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            message: inputValue.trim(),
            timestamp: new Date().toISOString(),
            role: 'user'
        };

        // Add user message immediately
        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);

        try {
            // Send message via WebSocket
            socketRef.current.emit('chat_message', {
                message: userMessage.message
            });
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const handleManualReconnect = () => {
        if (socketRef.current) {
            setConnectionStatus('connecting');
            setConnectionMessage('Reconnecting...');
            reconnectAttemptsRef.current = 0;
            socketRef.current.connect();
        }
    };

    const isReady = connectionStatus === 'connected';
    const statusLabel =
        connectionStatus === 'connected'
            ? 'Connected'
            : connectionStatus === 'connecting'
                ? 'Connecting...'
                : connectionStatus === 'reconnecting'
                    ? 'Reconnecting...'
                    : connectionStatus === 'unauthenticated'
                        ? 'Disconnected'
                        : 'Disconnected';
    const statusColor =
        connectionStatus === 'connected'
            ? 'bg-green-500'
            : connectionStatus === 'error' || connectionStatus === 'unauthenticated'
                ? 'bg-red-500'
                : 'bg-yellow-400';

    if (connectionStatus === 'unauthenticated') {
        return (
            <div className="flex flex-col h-full bg-gray-950 border border-slate-800 rounded-lg justify-center items-center p-6 text-center">
                <h3 className="text-lg font-bold text-slate-200 mb-2">Start a conversation with Caria</h3>
                <p className="text-sm text-slate-400 mb-6">Please log in to chat with Caria.</p>
                <div className="flex gap-3">
                    <button
                        onClick={() => window.location.href = '/?login=true'}
                        className="bg-slate-700 text-white font-bold px-4 py-2 rounded-md hover:bg-slate-600 transition-all text-sm"
                    >
                        Log In
                    </button>
                    <button
                        onClick={() => window.location.href = '/?register=true'}
                        className="border border-slate-600 text-slate-300 font-bold px-4 py-2 rounded-md hover:bg-slate-800 transition-all text-sm"
                    >
                        Create Account
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full bg-gray-950 border border-slate-800 rounded-lg">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
                <h3 className="text-lg font-bold text-slate-200">Chat with Caria</h3>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${statusColor}`}></div>
                    <span className="text-xs text-slate-400">{statusLabel}</span>
                    {connectionStatus !== 'connected' && connectionStatus !== 'unauthenticated' && (
                        <button
                            onClick={handleManualReconnect}
                            className="text-xs text-slate-400 border border-slate-600 rounded px-2 py-1 hover:text-slate-100"
                        >
                            Retry
                        </button>
                    )}
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="text-slate-400 hover:text-slate-200 transition-colors"
                        >
                            ✕
                        </button>
                    )}
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-slate-500 py-8">
                        <p>Start a conversation with Caria</p>
                        <p className="text-xs mt-2">
                            {connectionStatus === 'connected'
                                ? 'Ask about investments, market analysis, or portfolio advice'
                                : connectionMessage}
                        </p>
                    </div>
                )}
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user'
                                    ? 'bg-slate-700 text-slate-100'
                                    : 'bg-slate-800 text-slate-200'
                                }`}
                        >
                            <p className="text-sm whitespace-pre-wrap">{msg.message}</p>
                            <p className="text-xs text-slate-400 mt-1">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-slate-800 rounded-lg p-3">
                            <div className="flex gap-1">
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-slate-800">
                {connectionMessage && connectionStatus !== 'connected' && (
                    <div className="text-xs text-slate-400 mb-2">{connectionMessage}</div>
                )}
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={isReady ? 'Type your message...' : connectionMessage}
                        disabled={!isReady || isLoading}
                        className="flex-1 bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 text-sm text-slate-200 disabled:opacity-50"
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!isReady || isLoading || !inputValue.trim()}
                        className="bg-slate-700 text-white font-bold px-4 rounded-md hover:bg-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                    >
                        Send
                    </button>
                </div>

                {/* Debug Panel Toggle */}
                <button
                    onClick={() => setShowDebug(!showDebug)}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors mt-2"
                >
                    Debug {showDebug ? '▴' : '▾'}
                </button>

                {/* Debug Panel */}
                {showDebug && debugInfo && (
                    <div className="mt-3 p-3 bg-gray-900 border border-slate-800 rounded-md text-xs space-y-2">
                        <div className="text-slate-400 font-semibold">Debug Information</div>
                        <div>
                            <span className="text-slate-500">Latency:</span>
                            <span className="text-slate-300 ml-2">{debugInfo.latency_ms || 'N/A'} ms</span>
                        </div>
                        <div>
                            <span className="text-slate-500">Tokens Used:</span>
                            <span className="text-slate-300 ml-2">
                                {debugInfo.tokens_used ? `${debugInfo.tokens_used.prompt || 0} + ${debugInfo.tokens_used.completion || 0}` : 'N/A'}
                            </span>
                        </div>
                        <div>
                            <span className="text-slate-500">Timestamp:</span>
                            <span className="text-slate-300 ml-2">{debugInfo.timestamp ? new Date(debugInfo.timestamp).toLocaleString() : 'N/A'}</span>
                        </div>
                        {debugInfo.request && (
                            <div>
                                <div className="text-slate-500 mb-1">Request Payload:</div>
                                <pre className="bg-gray-950 p-2 rounded overflow-x-auto text-slate-400">
                                    {JSON.stringify(debugInfo.request, null, 2)}
                                </pre>
                            </div>
                        )}
                        {debugInfo.response && (
                            <div>
                                <div className="text-slate-500 mb-1">Raw LLM Response:</div>
                                <pre className="bg-gray-950 p-2 rounded overflow-x-auto text-slate-400 max-h-40 overflow-y-auto">
                                    {JSON.stringify(debugInfo.response, null, 2)}
                                </pre>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};
