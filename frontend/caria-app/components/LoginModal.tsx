import React, { useState, useEffect } from 'react';
import { CariaLogoIcon, XIcon } from './Icons';

interface LoginModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToRegister?: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onClose, onSuccess, onSwitchToRegister }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
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
            const { API_BASE_URL } = await import('../services/apiService');
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
                    errorMessage = `Login failed: ${response.status} ${response.statusText}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            if (data.token && data.token.access_token) {
                const { saveToken } = await import('../services/apiService');
                saveToken(data.token.access_token, data.token.refresh_token);
                onSuccess(data.token.access_token);
            } else {
                throw new Error('Login response did not include a token.');
            }
        } catch (err: any) {
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
            style={{ animationDuration: '0.3s' }}
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
                        <CariaLogoIcon className="w-12 h-12 text-slate-400 mb-3" />
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
                                placeholder="Enter your username"
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="password-input" className="text-sm font-bold text-slate-400">Password</label>
                            <div className="relative mt-1">
                                <input
                                    id="password-input"
                                    type={showPassword ? 'text' : 'password'}
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 pr-10 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                    placeholder="Enter your password"
                                    required
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-all duration-200"
                                    title={showPassword ? 'Hide password' : 'Show password'}
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    {showPassword ? (
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                                        </svg>
                                    ) : (
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                        </svg>
                                    )}
                                </button>
                            </div>
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
