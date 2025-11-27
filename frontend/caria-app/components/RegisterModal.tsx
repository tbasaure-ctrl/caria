import React, { useState } from 'react';
import { API_BASE_URL } from '../services/apiService';

interface RegisterModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToLogin: () => void;
}

export const RegisterModal: React.FC<RegisterModalProps> = ({ onClose, onSuccess, onSwitchToLogin }) => {
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Auto-generate username from email when email changes
    React.useEffect(() => {
        if (email && !username) {
            // Extract username from email (part before @)
            const emailUsername = email.split('@')[0];
            // Clean it to match username requirements: alphanumeric, underscore, hyphen only
            const cleanUsername = emailUsername.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase();
            if (cleanUsername.length >= 3) {
                setUsername(cleanUsername);
            }
        }
    }, [email, username]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        // Validate email
        if (!email || !email.trim()) {
            setError('Email is required');
            return;
        }

        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            setError('Please enter a valid email address');
            return;
        }

        // Validate password
        if (!password || password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }

        // Validate password match
        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        // Validate username
        const trimmedUsername = username.trim();
        if (!trimmedUsername || trimmedUsername.length < 3) {
            setError('Username must be at least 3 characters');
            return;
        }
        if (!/^[a-zA-Z0-9_-]+$/.test(trimmedUsername)) {
            setError('Username can only contain letters, numbers, underscores, and hyphens');
            return;
        }

        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    email: email.trim(),
                    username: trimmedUsername,
                    password
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Registration failed');
            }

            // Auto-login after registration
            const loginResponse = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: trimmedUsername,
                    password: password,
                }),
            });

            if (!loginResponse.ok) {
                throw new Error('Registration successful, but auto-login failed. Please sign in manually.');
            }

            const loginData = await loginResponse.json();
            // Backend returns { user: {...}, token: { access_token: "...", ... } }
            if (loginData.token && loginData.token.access_token) {
                onSuccess(loginData.token.access_token);
            } else if (loginData.access_token) {
                // Fallback for different response format
                onSuccess(loginData.access_token);
            } else {
                throw new Error('Invalid response format from server');
            }
        } catch (err: any) {
            setError(err.message || 'An error occurred during registration');
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
                            Create Account
                        </h2>
                        <p 
                            className="text-sm mt-0.5"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Create an account to track your portfolio and get personalized insights
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
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="you@example.com"
                        />
                    </div>

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Username
                        </label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="username"
                            pattern="[a-zA-Z0-9_-]+"
                            minLength={3}
                            maxLength={50}
                        />
                        <p className="text-xs mt-1" style={{ color: 'var(--color-text-subtle)' }}>
                            Letters, numbers, underscores, and hyphens only (3-50 characters)
                        </p>
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
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="At least 8 characters"
                        />
                    </div>

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Confirm Password
                        </label>
                        <input
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="••••••••"
                        />
                    </div>

                    <div 
                        className="px-4 py-3 rounded-lg text-xs"
                        style={{
                            backgroundColor: 'rgba(74, 144, 226, 0.1)',
                            border: '1px solid rgba(74, 144, 226, 0.2)',
                            color: 'var(--color-text-secondary)',
                        }}
                    >
                        <p className="mb-1 font-medium" style={{ color: 'var(--color-accent-primary)' }}>
                            Why create an account?
                        </p>
                        <p>
                            An account is required to add holdings and track your portfolio. This allows Caria to provide you with personalized insights and follow your investment journey. All other features are available without an account.
                        </p>
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
                        {isLoading ? 'Creating Account...' : 'Create Account'}
                    </button>

                    <div className="text-center">
                        <span 
                            className="text-sm"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Already have an account?{' '}
                        </span>
                        <button
                            type="button"
                            onClick={onSwitchToLogin}
                            className="text-sm font-medium transition-colors"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            Sign in
                        </button>
                    </div>

                    <p 
                        className="text-xs text-center"
                        style={{ color: 'var(--color-text-subtle)' }}
                    >
                        By creating an account, you agree to our Terms of Service and Privacy Policy.
                    </p>
                </form>
            </div>
        </div>
    );
};
