import React, { useState, useEffect } from 'react';
import { CariaLogoIcon, XIcon } from './Icons';

interface LoginModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onClose, onSuccess, onSwitchToRegister }) => {
    const [username, setUsername] = useState('user'); // Default for demo
    const [password, setPassword] = useState('pass'); // Default for demo
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            // Use centralized API_BASE_URL per audit document (must be absolute URL)
            const { API_BASE_URL } = await import('../services/apiService');
            // Remove /api if API_BASE_URL already includes it to avoid double slash or path issues
            // But usually API_BASE_URL is just the domain.
            // Let's ensure we are using the correct endpoint.
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                let errorMessage = 'Invalid username or password';
                try {
                    const errData = await response.json();
                    errorMessage = errData.detail || errData.message || errorMessage;
                } catch {
                    // Si no se puede parsear JSON, usar el status text
                    errorMessage = `Login failed: ${response.status} ${response.statusText}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            if (data.token && data.token.access_token) {
                // Guardar ambos tokens
                const { saveToken } = await import('../services/apiService');
                saveToken(data.token.access_token, data.token.refresh_token);
                onSuccess(data.token.access_token);
            } else {
                throw new Error('Login response did not include a token.');
            }
        } catch (err: any) {
            // Manejar diferentes tipos de errores per audit document
            if (err.message === 'Failed to fetch' || err.name === 'TypeError' || err.message.includes('Failed to connect')) {
                const { API_BASE_URL } = await import('../services/apiService');
                setError(`Unable to connect to the server. Please check that the API is running at ${API_BASE_URL}`);
            } else {
                setError(err.message || 'Login failed. Please try again.');
            }
            console.error('Login error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div 
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 fade-in"
            style={{animationDuration: '0.3s'}}
            onClick={onClose}
            role="dialog"
            aria-modal="true"
            aria-labelledby="login-modal-title"
        >
            <div 
                className="relative w-full max-w-sm bg-gray-950 text-slate-200 rounded-2xl border border-slate-800/50 overflow-hidden modal-fade-in" 
                onClick={(e) => e.stopPropagation()}
            >
                <header className="p-4 border-b border-slate-800/50 flex items-center justify-between">
                    <h1 id="login-modal-title" className="text-xl font-bold text-[#E0E1DD]">Welcome Back</h1>
                    <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors" aria-label="Close login modal">
                        <XIcon className="w-6 h-6" />
                    </button>
                </header>

                <main className="p-6">
                    <div className="flex flex-col items-center text-center mb-6">
                        <CariaLogoIcon className="w-12 h-12 text-slate-400 mb-3"/>
                        <p className="text-slate-400 text-sm">Log in to access your dashboard.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label htmlFor="username" className="text-sm font-bold text-slate-400">Username</label>
                            <input
                                id="username"
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="password-input" className="text-sm font-bold text-slate-400">Password</label>
                            <input
                                id="password-input"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                required
                            />
                        </div>

                        {error && <p className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">{error}</p>}
                        
                        <button type="submit" disabled={isLoading} className="w-full bg-slate-800 text-white font-bold py-3 px-5 rounded-md hover:bg-slate-700 transition-all disabled:opacity-50 disabled:bg-slate-800 disabled:cursor-not-allowed">
                            {isLoading ? 'Signing In...' : 'Sign In'}
                        </button>
                    </form>

                    {onSwitchToRegister && (
                        <div className="mt-4 text-center">
                            <p className="text-sm text-slate-400">
                                Don't have an account?{' '}
                                <button 
                                    onClick={onSwitchToRegister}
                                    className="text-slate-300 hover:text-white underline"
                                >
                                    Sign up
                                </button>
                            </p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};
