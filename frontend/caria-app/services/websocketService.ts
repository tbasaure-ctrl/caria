/**
 * WebSocket Chat Service - Frontend implementation per audit document.
 * Solution of 3 parts:
 * 1. JWT Authentication in handshake
 * 2. Heartbeat Ping/Pong (handled by server with ping_interval=25, ping_timeout=60)
 * 3. History recovery on reconnection
 */

import { io, Socket } from 'socket.io-client';
import { getToken } from './apiService';
import { WS_BASE_URL } from './apiConfig';

export interface ChatMessage {
    id: string;
    message: string;
    timestamp: string;
    role: 'user' | 'assistant' | 'error';
}

export type ChatMessageHandler = (message: ChatMessage) => void;
export type ErrorHandler = (error: string) => void;

class WebSocketService {
    private socket: Socket | null = null;
    private messageHandlers: ChatMessageHandler[] = [];
    private errorHandlers: ErrorHandler[] = [];
    private lastMessageTimestamp: string | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    /**
     * Connect to WebSocket server with JWT authentication per audit document.
     * Problem #1: Authentication in handshake
     */
    connect(): void {
        if (this.socket?.connected) {
            console.warn('WebSocket already connected');
            return;
        }

        const token = getToken();
        if (!token) {
            console.error('No authentication token available. Please log in first.');
            this.notifyError('Authentication required. Please log in.');
            return;
        }

        console.log(`Connecting to WebSocket at ${WS_BASE_URL}...`);

        // Create Socket.IO client with authentication in handshake
        // Per audit document: token must be sent in the initial handshake
        this.socket = io(WS_BASE_URL, {
            auth: {
                token: token,  // Problem #1: JWT token in handshake
            },
            transports: ['websocket', 'polling'], // Fallback to polling if WebSocket fails
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
        });

        // Handle connection success
        this.socket.on('connect', async () => {
            console.log('WebSocket connected successfully');
            this.reconnectAttempts = 0;

            // Problem #3: Recover lost messages on reconnection
            await this.recoverChatHistory();
        });

        // Handle connection error
        this.socket.on('connect_error', (error: { message: string }) => {
            console.error('WebSocket connection error:', error);
            this.reconnectAttempts++;
            
            if (error.message.includes('Authentication') || error.message.includes('Invalid token')) {
                this.notifyError('Authentication failed. Please log in again.');
                // Don't attempt to reconnect if auth failed
                this.disconnect();
            } else if (this.reconnectAttempts < this.maxReconnectAttempts) {
                console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            } else {
                this.notifyError('Failed to connect to chat server. Please refresh the page.');
            }
        });

        // Handle disconnection
        this.socket.on('disconnect', (reason: string) => {
            console.log('WebSocket disconnected:', reason);
            if (reason === 'io server disconnect') {
                // Server disconnected us, try to reconnect
                this.socket?.connect();
            }
        });

        // Handle incoming chat messages
        this.socket.on('chat_message', (data: ChatMessage) => {
            console.log('Received chat message:', data);
            this.lastMessageTimestamp = data.timestamp;
            this.notifyMessage(data);
        });

        // Handle errors
        this.socket.on('error', (error: { message: string }) => {
            console.error('WebSocket error:', error);
            this.notifyError(error.message || 'An error occurred');
        });
    }

    /**
     * Recover chat history on reconnection per audit document.
     * Problem #3: Recovery of lost messages
     */
    private async recoverChatHistory(): Promise<void> {
        if (!this.lastMessageTimestamp) {
            // First connection, get last 50 messages
            await this.fetchChatHistory();
        } else {
            // Reconnection, get messages since last timestamp
            await this.fetchChatHistory(this.lastMessageTimestamp);
        }
    }

    /**
     * Fetch chat history from REST API endpoint.
     * Per audit document Problem #3: This endpoint is called in the 'connect' event.
     */
    private async fetchChatHistory(since?: string): Promise<void> {
        try {
            const token = getToken();
            if (!token) {
                return;
            }

            const { API_BASE_URL } = await import('./apiConfig');
            const url = new URL(`${API_BASE_URL}/api/chat/history`);
            if (since) {
                url.searchParams.append('since', since);
            }

            const response = await fetch(url.toString(), {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                console.error('Failed to fetch chat history:', response.statusText);
                return;
            }

            const data = await response.json();
            const messages: ChatMessage[] = data.messages || [];

            console.log(`Recovered ${messages.length} chat messages`);

            // Notify handlers of recovered messages
            for (const message of messages) {
                this.notifyMessage(message);
                // Update last timestamp
                if (!this.lastMessageTimestamp || message.timestamp > this.lastMessageTimestamp) {
                    this.lastMessageTimestamp = message.timestamp;
                }
            }
        } catch (error) {
            console.error('Error fetching chat history:', error);
        }
    }

    /**
     * Send a chat message via WebSocket.
     */
    sendMessage(message: string): void {
        if (!this.socket?.connected) {
            this.notifyError('Not connected to chat server. Please wait...');
            return;
        }

        this.socket.emit('chat_message', { message });
    }

    /**
     * Disconnect from WebSocket server.
     */
    disconnect(): void {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }

    /**
     * Check if WebSocket is connected.
     */
    isConnected(): boolean {
        return this.socket?.connected ?? false;
    }

    /**
     * Register a message handler.
     */
    onMessage(handler: ChatMessageHandler): void {
        this.messageHandlers.push(handler);
    }

    /**
     * Unregister a message handler.
     */
    offMessage(handler: ChatMessageHandler): void {
        this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    }

    /**
     * Register an error handler.
     */
    onError(handler: ErrorHandler): void {
        this.errorHandlers.push(handler);
    }

    /**
     * Unregister an error handler.
     */
    offError(handler: ErrorHandler): void {
        this.errorHandlers = this.errorHandlers.filter(h => h !== handler);
    }

    private notifyMessage(message: ChatMessage): void {
        this.messageHandlers.forEach(handler => {
            try {
                handler(message);
            } catch (error) {
                console.error('Error in message handler:', error);
            }
        });
    }

    private notifyError(error: string): void {
        this.errorHandlers.forEach(handler => {
            try {
                handler(error);
            } catch (err) {
                console.error('Error in error handler:', err);
            }
        });
    }
}

// Export singleton instance
export const websocketService = new WebSocketService();

