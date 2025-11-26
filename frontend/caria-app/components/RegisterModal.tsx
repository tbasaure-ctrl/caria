import React, { useState } from 'react';
import { API_BASE_URL } from '../services/apiService';

interface RegisterModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToLogin: () => void;
}

export const RegisterModal: React.FC<RegisterModalProps> = ({ onClose, onSuccess, onSwitchToLogin }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [occupation, setOccupation] = useState('');
    const [selfDescription, setSelfDescription] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        if (password.length < 8) {
            setError('Password must be at least 8 characters');
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
                    email, 
                    password,
                    occupation: occupation || undefined,
                    self_description: selfDescription || undefined
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Registration failed');
            }

            // Auto-login after registration
            const formData = new URLSearchParams();
            formData.append('username', email);
            formData.append('password', password);

            const loginResponse = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData.toString(),
            });

            if (!loginResponse.ok) {
                throw new Error('Registration successful, but auto-login failed. Please sign in manually.');
            }

            const loginData = await loginResponse.json();
            onSuccess(loginData.access_token);
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
                            Join Caria for professional-grade research
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
                            required
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

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Occupation <span className="text-xs normal-case" style={{ color: 'var(--color-text-subtle)' }}>(optional)</span>
                        </label>
                        <input
                            type="text"
                            value={occupation}
                            onChange={(e) => setOccupation(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="What do you do for a living?"
                        />
                    </div>

                    <div>
                        <label 
                            className="block text-xs font-medium tracking-wider uppercase mb-2"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            Tell us about yourself <span className="text-xs normal-case" style={{ color: 'var(--color-text-subtle)' }}>(optional)</span>
                        </label>
                        <textarea
                            value={selfDescription}
                            onChange={(e) => setSelfDescription(e.target.value)}
                            rows={3}
                            className="w-full px-4 py-3 rounded-lg text-sm resize-none"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                border: '1px solid var(--color-border-subtle)',
                                color: 'var(--color-text-primary)',
                            }}
                            placeholder="Tell us a bit about yourself..."
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
