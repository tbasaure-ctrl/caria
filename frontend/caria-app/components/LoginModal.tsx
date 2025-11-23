import React, { useState, useEffect } from 'react';
import { CariaLogoIcon, XIcon } from './Icons';
import { loginWithGoogle, getIdToken } from '../src/firebase/auth';

interface LoginModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToRegister?: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onClose, onSuccess, onSwitchToRegister }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [usernameFocused, setUsernameFocused] = useState(false);
    const [passwordFocused, setPasswordFocused] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loginMethod, setLoginMethod] = useState<'username' | 'google' | null>(null);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const handleGoogleLogin = async () => {
        setIsLoading(true);
        setError(null);
        setLoginMethod('google');

        try {
            // Login with Google via Firebase
            const userCredential = await loginWithGoogle();
            console.log('User logged in with Google:', userCredential.user);

            // Get Firebase token
            const firebaseToken = await getIdToken();
            
            if (!firebaseToken) {
                throw new Error('Could not obtain Firebase token');
            }

            // Try to get JWT token from backend
            try {
                const { API_BASE_URL } = await import('../services/apiConfig');
                const response = await fetch(`${API_BASE_URL}/api/auth/firebase/verify`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ firebase_token: firebaseToken }),
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.access_token) {
                        const { saveToken } = await import('../services/apiService');
                        saveToken(data.access_token, data.refresh_token);
                        onSuccess(data.access_token);
                    } else {
                        // If backend doesn't have Firebase endpoint, use Firebase token directly
                        const { saveToken } = await import('../services/apiService');
                        saveToken(firebaseToken, firebaseToken);
                        onSuccess(firebaseToken);
                    }
                } else {
                    // If backend doesn't have Firebase endpoint, use Firebase token directly
                    const { saveToken } = await import('../services/apiService');
                    saveToken(firebaseToken, firebaseToken);
                    onSuccess(firebaseToken);
                }
            } catch (backendError) {
                // If backend fails, use Firebase token directly
                console.warn('Error connecting to backend, using Firebase token only:', backendError);
                const { saveToken } = await import('../services/apiService');
                saveToken(firebaseToken, firebaseToken);
                onSuccess(firebaseToken);
            }

        } catch (err: any) {
            console.error('Google login error:', err);
            
            let errorMessage = 'Error signing in with Google';
            
            if (err.code === 'auth/popup-closed-by-user') {
                errorMessage = 'Google sign-in window was closed';
            } else if (err.code === 'auth/popup-blocked') {
                errorMessage = 'Popup blocked. Please allow popups for this site';
            } else if (err.message) {
                errorMessage = err.message;
            }
            
            setError(errorMessage);
        } finally {
            setIsLoading(false);
            setLoginMethod(null);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        setLoginMethod('username');

        try {
            // Use centralized API_BASE_URL per audit document (must be absolute URL)
            const { API_BASE_URL } = await import('../services/apiConfig');
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
                const { API_BASE_URL } = await import('../services/apiConfig');
                setError(`Unable to connect to the server. Please check that the API is running at ${API_BASE_URL}`);
            } else {
                setError(err.message || 'Login failed. Please try again.');
            }
            console.error('Login error:', err);
        } finally {
            setIsLoading(false);
            setLoginMethod(null);
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

                    {/* Google Sign In Button */}
                    <button
                        type="button"
                        onClick={handleGoogleLogin}
                        disabled={isLoading}
                        className="w-full mb-4 bg-white text-gray-900 font-bold py-3 px-5 rounded-md hover:bg-gray-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {isLoading && loginMethod === 'google' ? (
                            <>
                                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Signing in...
                            </>
                        ) : (
                            <>
                                <svg className="w-5 h-5" viewBox="0 0 24 24">
                                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                                </svg>
                                Sign in with Google
                            </>
                        )}
                    </button>

                    <div className="relative my-4">
                        <div className="absolute inset-0 flex items-center">
                            <div className="w-full border-t border-slate-700"></div>
                        </div>
                        <div className="relative flex justify-center text-sm">
                            <span className="px-2 bg-gray-950 text-slate-400">Or continue with username</span>
                        </div>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label htmlFor="username" className="text-sm font-bold text-slate-400">Username</label>
                            <input
                                id="username"
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                onFocus={() => setUsernameFocused(true)}
                                onBlur={() => setUsernameFocused(false)}
                                placeholder={usernameFocused || username ? '' : 'username'}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow placeholder:text-slate-500"
                                required
                                autoComplete="username"
                            />
                        </div>
                        <div>
                            <label htmlFor="password-input" className="text-sm font-bold text-slate-400">Password</label>
                            <input
                                id="password-input"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                onFocus={() => setPasswordFocused(true)}
                                onBlur={() => setPasswordFocused(false)}
                                placeholder={passwordFocused || password ? '' : 'password'}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow placeholder:text-slate-500"
                                required
                                autoComplete="current-password"
                            />
                        </div>

                        {error && <p className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">{error}</p>}
                        
                        <button 
                            type="submit" 
                            disabled={isLoading} 
                            className="w-full bg-slate-800 text-white font-bold py-3 px-5 rounded-md hover:bg-slate-700 transition-all disabled:opacity-50 disabled:bg-slate-800 disabled:cursor-not-allowed"
                        >
                            {isLoading && loginMethod === 'username' ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Signing In...
                                </span>
                            ) : (
                                'Sign In'
                            )}
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
