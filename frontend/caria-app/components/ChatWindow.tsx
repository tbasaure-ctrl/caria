import React, { useEffect, useState, useRef } from 'react';
import { API_BASE_URL } from '../services/apiService';
import { getAuthToken } from '../services/apiService';

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
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ onClose }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [lastMessageTimestamp, setLastMessageTimestamp] = useState<string | null>(null);
    const socketRef = useRef<Socket | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Initialize WebSocket connection with JWT authentication
    useEffect(() => {
        const connectSocket = async () => {
            try {
                const token = getAuthToken();
                if (!token) {
                    console.warn('No auth token available for WebSocket connection');
                    setIsConnected(false);
                    return;
                }

                // Connect with JWT token in auth object (per audit document Problem #1)
                // Socket.IO connects to the base URL (without /api), e.g., http://localhost:8000
                let socketBaseUrl = API_BASE_URL;
                if (socketBaseUrl.includes('/api')) {
                    socketBaseUrl = socketBaseUrl.replace('/api', '');
                }
                if (!socketBaseUrl || socketBaseUrl === '') {
                    socketBaseUrl = 'http://localhost:8000';
                }
                
                const socket = socketIO(socketBaseUrl, {
                    auth: {
                        token: token
                    },
                    path: '/socket.io/',
                    transports: ['websocket', 'polling'],
                    reconnection: true,
                    reconnectionAttempts: 5,
                    reconnectionDelay: 1000,
                });

                socketRef.current = socket;

                // Handle connection success
                socket.on('connect', async () => {
                    console.log('WebSocket connected');
                    setIsConnected(true);

                    // Problem #3: Recover lost messages on reconnection
                    try {
                        const response = await fetch(`${API_BASE_URL}/api/chat/history${lastMessageTimestamp ? `?since=${lastMessageTimestamp}` : ''}`, {
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
                    setIsConnected(false);
                });

                // Handle disconnection
                socket.on('disconnect', () => {
                    console.log('WebSocket disconnected');
                    setIsConnected(false);
                });

                // Handle incoming chat messages
                socket.on('chat_message', (data: Message) => {
                    setMessages(prev => {
                        const newMessages = [...prev, data];
                        setLastMessageTimestamp(data.timestamp);
                        return newMessages;
                    });
                });

                // Handle errors
                socket.on('error', (error: any) => {
                    console.error('WebSocket error:', error);
                });

            } catch (error) {
                console.error('Error initializing WebSocket:', error);
                setIsConnected(false);
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

    const handleSendMessage = async () => {
        if (!inputValue.trim() || !socketRef.current || !isConnected) return;

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

    return (
        <div className="flex flex-col h-full bg-gray-950 border border-slate-800 rounded-lg">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
                <h3 className="text-lg font-bold text-slate-200">Chat with Caria</h3>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className="text-xs text-slate-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
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
                        <p className="text-xs mt-2">Ask about investments, market analysis, or portfolio advice</p>
                    </div>
                )}
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[80%] rounded-lg p-3 ${
                                msg.role === 'user'
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
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={isConnected ? "Type your message..." : "Connecting..."}
                        disabled={!isConnected || isLoading}
                        className="flex-1 bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 text-sm text-slate-200 disabled:opacity-50"
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!isConnected || isLoading || !inputValue.trim()}
                        className="bg-slate-700 text-white font-bold px-4 rounded-md hover:bg-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
    );
};
