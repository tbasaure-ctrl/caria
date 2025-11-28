import React, { useState, useRef, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { SendIcon, ThesisIcon, CariaLogoIcon } from '../Icons'; // Usando tus iconos nativos
import { getToken, API_BASE_URL } from '../../services/apiService';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    id?: string;
}

export const ThesisArena: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [threadId, setThreadId] = useState<string | null>(null);

    // Auto-scroll al último mensaje
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim() || loading) return;

        const userMsg: Message = { role: 'user', content: input };
        const newMessages = [...messages, userMsg];
        setMessages(newMessages);
        setInput('');
        setLoading(true);

        try {
            const token = getToken();
            const response = await fetch(`${API_BASE_URL}/api/thesis/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    message: userMsg.content,
                    ticker: ticker || undefined,
                    thread_id: threadId
                })
            });

            if (!response.ok) throw new Error('Failed to connect');
            if (!response.body) throw new Error('No stream body');

            // Preparar mensaje del asistente
            const assistantMsg: Message = { role: 'assistant', content: '' };
            setMessages([...newMessages, assistantMsg]);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            let currentContent = '';

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value, { stream: !done });
                currentContent += chunkValue;
                
                // Actualizar estado en tiempo real
                setMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = { ...assistantMsg, content: currentContent };
                    return updated;
                });
            }

        } catch (error) {
            console.error('Chat error:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Connection error. Please try again.' }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <WidgetCard
            title="Thesis Arena"
            tooltip="Debate your investment thesis with Caria Senior Partner (Socratic AI)."
            className="h-full min-h-[350px] sm:min-h-[450px] lg:min-h-[550px] flex flex-col"
        >
            <div className="flex flex-col h-full">
                {/* Header Controls - Responsive */}
                <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 mb-3 sm:mb-4 border-b border-white/5 pb-3 sm:pb-4">
                    <div className="relative flex-1">
                        <input
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="TICKER (Optional)"
                            className="w-full bg-bg-primary border border-white/10 rounded px-3 py-2 text-xs text-white focus:border-accent-primary outline-none font-mono tracking-wider"
                        />
                    </div>
                    <div className="text-[10px] text-text-muted flex items-center px-2 justify-end sm:justify-start">
                        <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                        <span className="hidden xs:inline">Senior Partner</span> Active
                    </div>
                </div>

                {/* Chat Area - Responsive */}
                <div className="flex-1 overflow-y-auto space-y-3 sm:space-y-4 pr-1 sm:pr-2 custom-scrollbar mb-3 sm:mb-4 min-h-0">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center text-text-muted opacity-50 px-4">
                            <CariaLogoIcon className="w-10 h-10 sm:w-12 sm:h-12 mb-3 sm:mb-4 text-accent-cyan" />
                            <p className="text-xs sm:text-sm">Present your thesis.</p>
                            <p className="text-[10px] sm:text-xs">I will challenge it.</p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`
                                    max-w-[90%] sm:max-w-[85%] rounded-2xl px-3 sm:px-5 py-2 sm:py-3 text-xs sm:text-sm leading-relaxed
                                    ${msg.role === 'user'
                                        ? 'bg-accent-primary text-white rounded-br-none shadow-glow-sm'
                                        : 'bg-bg-tertiary text-text-secondary rounded-bl-none border border-white/5'
                                    }
                                `}
                            >
                                {msg.role === 'assistant' && (
                                    <div className="flex items-center gap-2 mb-1 sm:mb-2 border-b border-white/5 pb-1">
                                        <span className="text-[10px] sm:text-xs font-bold text-accent-cyan uppercase tracking-wider">Caria</span>
                                    </div>
                                )}
                                <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                            </div>
                        </div>
                    ))}

                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-bg-tertiary rounded-2xl rounded-bl-none px-3 sm:px-4 py-2 sm:py-3 border border-white/5 flex items-center gap-1">
                                <div className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-1.5 h-1.5 bg-text-muted rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area - Responsive */}
                <div className="flex gap-2 sm:gap-3 pt-2">
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                        placeholder="Type your thesis..."
                        className="flex-1 bg-bg-primary border border-white/10 rounded-lg px-3 sm:px-4 py-2 sm:py-3 text-xs sm:text-sm text-white focus:border-accent-primary outline-none transition-colors placeholder:text-text-subtle"
                        disabled={loading}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={!input.trim() || loading}
                        className={`
                            px-3 sm:px-4 rounded-lg flex items-center justify-center transition-all
                            ${!input.trim() || loading
                                ? 'bg-bg-tertiary text-text-muted cursor-not-allowed'
                                : 'bg-accent-primary text-white hover:bg-accent-primary/90 shadow-glow-sm'
                            }
                        `}
                    >
                        <SendIcon className="w-4 h-4 sm:w-5 sm:h-5" />
                    </button>
                </div>
            </div>
        </WidgetCard>
    );
};
