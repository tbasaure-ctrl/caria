import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { XIcon, SendIcon } from './Icons';
import { API_BASE_URL, fetchWithAuth } from '../services/apiService';

// LocalStorage Helpers
const getChatHistory = (): ChatMessage[] => {
    try {
        const saved = localStorage.getItem('cariaChatHistory');
        return saved ? JSON.parse(saved) : [];
    } catch (e) {
        console.warn('Failed to parse chat history', e);
        return [];
    }
};

const saveChatHistory = (messages: ChatMessage[]) => {
    localStorage.setItem('cariaChatHistory', JSON.stringify(messages));
};

// Chat Bubble Component
const ChatMessageBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
    const isModel = message.role === 'model';
    const isError = message.role === 'error';

    return (
        <div className={`w-full flex ${isModel ? 'justify-start' : 'justify-end'}`}>
            <div 
                className="max-w-2xl p-4 rounded-xl"
                style={{
                    backgroundColor: isModel 
                        ? 'var(--color-bg-tertiary)' 
                        : isError 
                        ? 'var(--color-negative-muted)' 
                        : 'var(--color-accent-primary)',
                    color: isModel 
                        ? 'var(--color-text-primary)' 
                        : isError 
                        ? 'var(--color-negative)' 
                        : '#FFFFFF',
                    border: isModel ? '1px solid var(--color-border-subtle)' : 'none',
                }}
            >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                    {message.content}
                </div>
            </div>
        </div>
    );
};

// Ticker Extraction
const extractTicker = (text: string): string | null => {
    const upper = text.toUpperCase();
    const tickerRegex = /\b([A-Z]{3,5}(?:\.[A-Z])?)\b/g;
    const matches = upper.match(tickerRegex);
    if (matches) return matches[0];

    const companyMap: Record<string, string> = {
        'nvidia': 'NVDA',
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'tesla': 'TSLA',
        'meta': 'META',
        'facebook': 'META',
        'netflix': 'NFLX',
    };

    const lower = text.toLowerCase();
    for (const [name, ticker] of Object.entries(companyMap)) {
        if (lower.includes(name)) return ticker;
    }

    return null;
};

export const AnalysisTool: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const [messages, setMessages] = useState<ChatMessage[]>(getChatHistory);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const chatEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => saveChatHistory(messages), [messages]);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [input]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || isLoading) return;

        const userMessage: ChatMessage = { role: 'user', content: text };

        if (text.length < 3) {
            const err: ChatMessage = {
                role: 'error',
                content: 'Por favor escribe al menos 3 caracteres.'
            };
            setMessages(prev => [...prev, userMessage, err]);
            setInput('');
            return;
        }

        // Extract ticker if present, but don't require it
        const ticker = extractTicker(text);

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // Build conversation history from current messages
            const conversationHistory = messages
                .filter(msg => msg.role === 'user' || msg.role === 'model')
                .map(msg => ({
                    role: msg.role === 'user' ? 'user' : 'assistant',
                    content: msg.content
                }));

            const response = await fetchWithAuth(`${API_BASE_URL}/api/analysis/challenge`, {
                method: 'POST',
                body: JSON.stringify({ 
                    thesis: text, 
                    ticker: ticker || undefined, 
                    top_k: 5,
                    conversation_history: conversationHistory
                })
            });
            
            const data = await response.json();

            // New response format: just a simple conversational response
            const responseText = data.response || 'Lo siento, no pude generar una respuesta.';

            const modelMessage: ChatMessage = { role: 'model', content: responseText };
            setMessages(prev => [...prev, modelMessage]);

        } catch (err: any) {
            console.error('Chat error:', err);
            const errorMsg: ChatMessage = {
                role: 'error',
                content: err?.message || 'No se pudo completar el análisis. Por favor intenta de nuevo.',
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)' }}
            onClick={onClose}
        >
            <div
                className="relative flex flex-col h-full max-h-[90vh] w-full max-w-4xl rounded-xl overflow-hidden"
                style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    border: '1px solid var(--color-border-default)',
                }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <header 
                    className="px-6 py-5 flex items-center justify-between border-b"
                    style={{ 
                        backgroundColor: 'var(--color-bg-secondary)',
                        borderColor: 'var(--color-border-subtle)'
                    }}
                >
                    <div>
                        <h1 
                            className="text-lg font-semibold"
                            style={{ 
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-text-primary)'
                            }}
                        >
                            Chat con Caria
                        </h1>
                        <p 
                            className="text-xs"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Conversa sobre tus ideas de inversión
                        </p>
                    </div>
                    <button 
                        onClick={onClose} 
                        className="w-10 h-10 rounded-lg flex items-center justify-center transition-colors"
                        style={{ 
                            color: 'var(--color-text-muted)',
                            backgroundColor: 'var(--color-bg-surface)'
                        }}
                    >
                        <XIcon className="w-5 h-5" />
                    </button>
                </header>

                {/* Chat Area */}
                <main 
                    className="flex-1 overflow-y-auto p-6 space-y-4"
                    style={{ backgroundColor: 'var(--color-bg-primary)' }}
                >
                    {messages.length === 0 && (
                        <div className="text-center py-12">
                            <div 
                                className="w-16 h-16 mx-auto mb-5 rounded-xl flex items-center justify-center"
                                style={{ backgroundColor: 'rgba(46, 124, 246, 0.12)' }}
                            >
                                <svg className="w-8 h-8" style={{ color: 'var(--color-accent-primary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                            </div>
                            <h3 
                                className="text-lg font-semibold mb-2"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                Habla con Caria
                            </h3>
                            <p 
                                className="text-sm max-w-md mx-auto"
                                style={{ color: 'var(--color-text-secondary)' }}
                            >
                                Conversa con Caria sobre tus ideas de inversión. Haz preguntas, comparte tus tesis, o pide su opinión sobre empresas. Caria te ayudará a pensar más profundamente.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <ChatMessageBubble key={i} message={msg} />
                    ))}

                    {isLoading && (
                        <div className="w-full flex justify-start">
                            <div 
                                className="px-4 py-3 rounded-xl flex items-center gap-2"
                                style={{ 
                                    backgroundColor: 'var(--color-bg-tertiary)',
                                    border: '1px solid var(--color-border-subtle)'
                                }}
                            >
                                <div className="flex gap-1">
                                    <div 
                                        className="w-2 h-2 rounded-full animate-pulse"
                                        style={{ backgroundColor: 'var(--color-accent-primary)', animationDelay: '0ms' }}
                                    />
                                    <div 
                                        className="w-2 h-2 rounded-full animate-pulse"
                                        style={{ backgroundColor: 'var(--color-accent-primary)', animationDelay: '150ms' }}
                                    />
                                    <div 
                                        className="w-2 h-2 rounded-full animate-pulse"
                                        style={{ backgroundColor: 'var(--color-accent-primary)', animationDelay: '300ms' }}
                                    />
                                </div>
                                <span 
                                    className="text-sm"
                                    style={{ color: 'var(--color-text-muted)' }}
                                >
                                    Caria está pensando...
                                </span>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </main>

                {/* Input Area */}
                <footer 
                    className="p-4 border-t"
                    style={{ 
                        backgroundColor: 'var(--color-bg-secondary)',
                        borderColor: 'var(--color-border-subtle)'
                    }}
                >
                    <form onSubmit={handleSubmit} className="flex items-end gap-3 max-w-3xl mx-auto">
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmit(e as any);
                                }
                            }}
                            placeholder="Escribe tu mensaje... (ej: 'Me interesa comprar ASTS', '¿Qué opinas de NVDA?')"
                            className="flex-1 px-4 py-3 rounded-xl resize-none focus:outline-none focus:ring-2 text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            rows={1}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="p-3 rounded-xl transition-colors disabled:opacity-50"
                            style={{
                                backgroundColor: 'var(--color-accent-primary)',
                                color: '#FFFFFF',
                            }}
                        >
                            <SendIcon className="w-5 h-5" />
                        </button>
                    </form>
                </footer>
            </div>
        </div>
    );
};
