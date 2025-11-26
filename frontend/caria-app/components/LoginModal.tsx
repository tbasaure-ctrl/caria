import React, { useState } from 'react';
import { API_BASE_URL } from '../services/apiService';

interface LoginModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToRegister: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onClose, onSuccess, onSwitchToRegister }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: email,
                    password: password,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Login failed' }));
                throw new Error(errorData.detail || 'Login failed');
            }

            const data = await response.json();
            // Backend returns { user: {...}, token: { access_token: "...", ... } }
            if (data.token && data.token.access_token) {
                onSuccess(data.token.access_token);
            } else if (data.access_token) {
                // Fallback for different response format
                onSuccess(data.access_token);
            } else {
                throw new Error('Invalid response format from server');
            }
        } catch (err: any) {
            setError(err.message || 'An error occurred during login');
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
                className="w-full max-w-md rounded-xl overflow-hidden"
                style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    border: '1px solid var(--color-border-default)',
                }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div 
                    className="px-6 py-5 flex items-center justify-between border-b"
                    style={{ borderColor: 'var(--color-border-subtle)' }}
                >
                    <div>
                        <h2 
                            className="text-xl font-semibold"
                            style={{
                                fontFamily: 'var(--font-display)',
                                color: 'var(--color-text-primary)',
                            }}
                        >
                            Welcome Back
                        </h2>
                        <p 
                            className="text-sm mt-0.5"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Sign in to your Caria account
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-8 h-8 rounded-lg flex items-center justify-center text-xl"
                        style={{ 
                            color: 'var(--color-text-muted)',
                            backgroundColor: 'var(--color-bg-surface)'
                        }}
                    >
                        ×
                    </button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="p-6 space-y-5">
                    {error && (
                        <div 
                            className="px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-negative-muted)',
                                color: 'var(--color-negative)',
                                border: '1px solid var(--color-negative)',
                            }}
                        >
                            {error}
                        </div>
                    )}

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Username or Email
                        </label>
                        <input
                            type="text"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="username or email@example.com"
                        />
                    </div>

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="••••••••"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full py-3 rounded-lg font-semibold text-sm transition-all duration-200 disabled:opacity-50"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                        }}
                    >
                        {isLoading ? 'Signing In...' : 'Sign In'}
                    </button>

                    <div className="text-center">
                        <span 
                            className="text-sm"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Don't have an account?{' '}
                        </span>
                        <button
                            type="button"
                            onClick={onSwitchToRegister}
                            className="text-sm font-medium transition-colors"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            Create one
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};
