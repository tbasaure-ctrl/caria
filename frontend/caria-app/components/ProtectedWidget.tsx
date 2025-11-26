import React, { useState } from 'react';
import { getToken, saveToken } from '../services/apiService';
import { API_BASE_URL } from '../services/apiService';

interface ProtectedWidgetProps {
    children: React.ReactNode;
    featureName: string;
    description?: string;
}

// Feature descriptions to intrigue users
const featureDescriptions: Record<string, string> = {
    'Portfolio Management': 'Add your holdings to track your investments, analyze performance, and optimize your portfolio allocation in real-time. An account is required to save and follow your portfolio.',
};

// Personalized unlock messages for each feature
const unlockMessages: Record<string, string> = {
    'Portfolio Management': 'Add your first position',
    'Portfolio Analytics': 'View your portfolio analysis',
    'Alpha Stock Picker': 'Discover AI stock picks',
    'Hidden Gems Screener': 'Find hidden opportunities',
    'Investment Thesis Analysis': 'Challenge your thesis',
    'Valuation Tool': 'Run valuation models',
    'Crisis Simulator': 'Stress test your portfolio',
    'Community': 'Join the community',
    'Regime Test': 'Test market regimes',
    'Industry Research': 'Explore industry insights',
};

export const ProtectedWidget: React.FC<ProtectedWidgetProps> = ({ children, featureName, description }) => {
    const token = getToken();
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [isLoginMode, setIsLoginMode] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Only protect Portfolio Management - all other features are open
    const shouldProtect = featureName === 'Portfolio Management';
    
    const featureDescription = description || featureDescriptions[featureName] || `An account is required to add holdings and track your portfolio. This allows Caria to provide you with personalized insights and follow your investment journey.`;

    const handleAuth = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            const endpoint = isLoginMode ? '/api/auth/login' : '/api/auth/register';
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: email,
                    password: password,
                    ...(isLoginMode ? {} : { email: email }),
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `${isLoginMode ? 'Login' : 'Registration'} failed` }));
                throw new Error(errorData.detail || `${isLoginMode ? 'Login' : 'Registration'} failed`);
            }

            const data = await response.json();
            const authToken = data.token?.access_token || data.access_token;
            
            if (!authToken) {
                throw new Error('Invalid response format from server');
            }

            saveToken(authToken);
            setShowAuthModal(false);
            // Force reload to refresh the widget
            window.location.reload();
        } catch (err: any) {
            setError(err.message || `An error occurred during ${isLoginMode ? 'login' : 'registration'}`);
        } finally {
            setIsLoading(false);
        }
    };

    // If feature doesn't need protection, render children directly
    if (!shouldProtect) {
        return <>{children}</>;
    }

    if (!token) {
        return (
            <>
                <div 
                    className="rounded-xl p-8 text-center transition-all duration-300 cursor-pointer group"
                    style={{
                        backgroundColor: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border-subtle)',
                    }}
                    onClick={() => setShowAuthModal(true)}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border-emphasis)';
                        e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                        e.currentTarget.style.transform = 'translateY(0)';
                    }}
                >
                    <div 
                        className="w-16 h-16 mx-auto mb-4 rounded-xl flex items-center justify-center transition-transform duration-300 group-hover:scale-110"
                        style={{ backgroundColor: 'rgba(46, 124, 246, 0.12)' }}
                    >
                        <svg 
                            className="w-8 h-8" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                            style={{ color: 'var(--color-accent-primary)' }}
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h3 
                        className="text-lg font-semibold mb-2"
                        style={{ 
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-text-primary)' 
                        }}
                    >
                        {featureName}
                    </h3>
                    <p 
                        className="text-sm mb-6 leading-relaxed max-w-md mx-auto"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        {featureDescription}
                    </p>
                    <div 
                        className="px-4 py-3 rounded-lg text-xs mb-4 max-w-md mx-auto"
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
                        onClick={(e) => {
                            e.stopPropagation();
                            setShowAuthModal(true);
                        }}
                        className="px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200"
                        style={{
                            backgroundColor: 'var(--color-accent-primary)',
                            color: '#FFFFFF',
                            fontFamily: 'var(--font-body)'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--color-accent-secondary)';
                            e.currentTarget.style.transform = 'translateY(-1px)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(46, 124, 246, 0.3)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--color-accent-primary)';
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = 'none';
                        }}
                    >
                        Create Account to Add Holdings →
                    </button>
                </div>

                {/* Auth Modal */}
                {showAuthModal && (
                    <div
                        className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)', backdropFilter: 'blur(4px)' }}
                        onClick={() => setShowAuthModal(false)}
                    >
                        <div
                            className="w-full max-w-md rounded-xl overflow-hidden animate-fade-in"
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
                                        {isLoginMode ? 'Welcome Back' : 'Create Account'}
                                    </h2>
                                    <p 
                                        className="text-sm mt-0.5"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        {isLoginMode ? 'Sign in to unlock ' + featureName : 'Get started with ' + featureName}
                                    </p>
                                </div>
                                <button
                                    onClick={() => setShowAuthModal(false)}
                                    className="w-8 h-8 rounded-lg flex items-center justify-center text-xl transition-colors"
                                    style={{ 
                                        color: 'var(--color-text-muted)',
                                        backgroundColor: 'var(--color-bg-surface)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
                                    }}
                                >
                                    ×
                                </button>
                            </div>

                            {/* Form */}
                            <form onSubmit={handleAuth} className="p-6 space-y-5">
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
                                        {isLoginMode ? 'Username or Email' : 'Email'}
                                    </label>
                                    <input
                                        type="text"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                        className="w-full px-4 py-3 rounded-lg text-sm transition-all"
                                        style={{
                                            backgroundColor: 'var(--color-bg-tertiary)',
                                            border: '1px solid var(--color-border-subtle)',
                                            color: 'var(--color-text-primary)',
                                        }}
                                        placeholder={isLoginMode ? "username or email@example.com" : "email@example.com"}
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
                                        className="w-full px-4 py-3 rounded-lg text-sm transition-all"
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
                                    {isLoading ? (isLoginMode ? 'Signing In...' : 'Creating Account...') : (isLoginMode ? 'Sign In' : 'Create Account')}
                                </button>

                                <div className="text-center">
                                    <span 
                                        className="text-sm"
                                        style={{ color: 'var(--color-text-muted)' }}
                                    >
                                        {isLoginMode ? "Don't have an account? " : "Already have an account? "}
                                    </span>
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setIsLoginMode(!isLoginMode);
                                            setError(null);
                                        }}
                                        className="text-sm font-medium transition-colors"
                                        style={{ color: 'var(--color-accent-primary)' }}
                                    >
                                        {isLoginMode ? 'Create one' : 'Sign in'}
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}
            </>
        );
    }

    return <>{children}</>;
};
