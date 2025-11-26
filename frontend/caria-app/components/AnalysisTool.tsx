import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { CariaLogoIcon, XIcon, SendIcon } from './Icons';
import { API_BASE_URL, fetchWithAuth } from '../services/apiService';


// -----------------------------------------------
// LocalStorage Helpers
// -----------------------------------------------
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


// -----------------------------------------------
// Chat Bubble Component
// -----------------------------------------------
const ChatMessageBubble: React.FC<{ message: ChatMessage }> = ({ message }) => {
    const isModel = message.role === 'model';
    const isError = message.role === 'error';

    const container = `w-full flex ${isModel ? 'justify-start' : 'justify-end'}`;
    const bubble =
        `max-w-2xl p-4 rounded-lg text-left ` +
        (isModel
            ? 'bg-gray-800 text-slate-200'
            : isError
            ? 'bg-red-900/50 text-red-200'
            : 'bg-slate-700 text-white'
        );

    return (
        <div className={container}>
            <div className={bubble}>
                <div className="whitespace-pre-wrap font-sans">{message.content}</div>
            </div>
        </div>
    );
};


// -----------------------------------------------
// Ticker Extraction Utility
// -----------------------------------------------
// Only accept valid tickers: 3–5 uppercase characters, optional ".X"
const extractTicker = (text: string): string | null => {
    const upper = text.toUpperCase();
    const tickerRegex = /\b([A-Z]{3,5}(?:\.[A-Z])?)\b/g;
    const matches = upper.match(tickerRegex);
    if (matches) return matches[0];

    // Fallback: company name → ticker
    const companyMap: Record<string, string> = {
        'nvidia': 'NVDA',
        'nvidia corporation': 'NVDA',
        'apple': 'AAPL',
        'apple inc': 'AAPL',
        'microsoft': 'MSFT',
        'microsoft corporation': 'MSFT',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'amazon.com': 'AMZN',
        'tesla': 'TSLA',
        'tesla inc': 'TSLA',
        'meta': 'META',
        'facebook': 'META',
        'netflix': 'NFLX',
        'netflix inc': 'NFLX'
    };

    const lower = text.toLowerCase();
    for (const [name, ticker] of Object.entries(companyMap)) {
        if (lower.includes(name)) return ticker;
    }

    return null;
};


// -----------------------------------------------
// Main Component
// -----------------------------------------------
export const AnalysisTool: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const [messages, setMessages] = useState<ChatMessage[]>(getChatHistory);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const chatEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Persist chat history
    useEffect(() => saveChatHistory(messages), [messages]);

    // Auto-scroll
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    // Resize textarea dynamically
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [input]);

    // Escape closes modal
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);


    // -------------------------------------------
    // Submit handler
    // -------------------------------------------
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || isLoading) return;

        const userMessage: ChatMessage = { role: 'user', content: text };

        // 1) Validate thesis length (backend requires ≥ 10 chars)
        if (text.length < 10) {
            const err: ChatMessage = {
                role: 'error',
                content: 'Your thesis is too short. Please write at least 10 characters.'
            };
            setMessages(prev => [...prev, userMessage, err]);
            setInput('');
            return;
        }

        // 2) Extract ticker
        const ticker = extractTicker(text);
        if (!ticker) {
            const err: ChatMessage = {
                role: 'error',
                content: 'Please include a valid ticker (e.g., AAPL, TSLA, NVDA).'
            };
            setMessages(prev => [...prev, userMessage, err]);
            setInput('');
            return;
        }

        // Add user message to chat
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            // Call Railway backend API endpoint
            const response = await fetchWithAuth(`${API_BASE_URL}/api/analysis/challenge`, {
                method: 'POST',
                body: JSON.stringify({
                    thesis: text,
                    ticker: ticker,
                    top_k: 5
                })
            });
            
            const data = await response.json();

            const critical = data.critical_analysis || 'No analysis provided.';
            const biases = Array.isArray(data.identified_biases) ? data.identified_biases : [];
            const recs = Array.isArray(data.recommendations) ? data.recommendations : [];

            // More conversational, Socratic format
            let formatted = critical;
            
            if (biases.length) {
                formatted += `\n\n**🤔 Things to Question:**\n`;
                formatted += biases.map(b => `- ${b}`).join('\n');
            }
            
            if (recs.length) {
                formatted += `\n\n**💭 Food for Thought:**\n`;
                formatted += recs.map(r => `- ${r}`).join('\n');
            }

            const modelMessage: ChatMessage = { role: 'model', content: formatted };
            setMessages(prev => [...prev, modelMessage]);

        } catch (err: any) {
            console.error('Chat error:', err);
            const friendlyMessage =
                err?.message ||
                'No se pudo completar el análisis en este momento. Intenta nuevamente en unos segundos.';
            const errorMsg: ChatMessage = {
                role: 'error',
                content: friendlyMessage,
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setIsLoading(false);
        }
    };


    // -------------------------------------------
    // Render
    // -------------------------------------------
    return (
        <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={onClose}
        >
            <div
                className="relative flex flex-col h-full max-h-[90vh] w-full max-w-4xl bg-gray-950 text-slate-200 rounded-2xl border border-slate-800/50 overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* HEADER */}
                <header className="bg-gray-900/80 p-4 border-b border-slate-800/50 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <CariaLogoIcon className="w-7 h-7 text-slate-100" />
                        <h1 className="text-xl font-bold">Caria Analysis</h1>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">
                        <XIcon className="w-6 h-6" />
                    </button>
                </header>

                {/* CHAT */}
                <main className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6">
                    {messages.map((msg, i) => (
                        <ChatMessageBubble key={i} message={msg} />
                    ))}

                    {isLoading && (
                        <div className="w-full flex justify-start">
                            <div className="max-w-2xl p-4 rounded-lg bg-gray-800 text-slate-400 flex gap-2">
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse" />
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
                                <div className="w-2 h-2 bg-slate-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </main>

                {/* INPUT */}
                <footer className="p-4 bg-gray-950/80 border-t border-slate-800/50">
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
                            placeholder="Share your investment thesis... What makes you think this is a good opportunity?"
                            className="flex-1 bg-gray-800 border border-slate-700 rounded-lg py-2 px-3 resize-none focus:outline-none focus:ring-2 focus:ring-slate-600 text-base max-h-40"
                            rows={1}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="bg-slate-700 text-white font-bold p-3 rounded-lg hover:bg-slate-600 disabled:opacity-50"
                        >
                            <SendIcon className="w-5 h-5" />
                        </button>
                    </form>
                </footer>
            </div>
        </div>
    );
};
