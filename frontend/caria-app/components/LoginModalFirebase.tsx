import React, { useState, useEffect } from 'react';
import { CariaLogoIcon, XIcon } from './Icons';
import { loginWithEmail, loginWithGoogle, getIdToken } from '../src/firebase';

interface LoginModalFirebaseProps {
    onClose: () => void;
    onSuccess: (token: string, firebaseToken?: string) => void;
    onSwitchToRegister?: () => void;
}

export const LoginModalFirebase: React.FC<LoginModalFirebaseProps> = ({ 
    onClose, 
    onSuccess, 
    onSwitchToRegister 
}) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loginMethod, setLoginMethod] = useState<'email' | 'google' | null>(null);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const handleEmailLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        setLoginMethod('email');

        try {
            // Login con Firebase
            const userCredential = await loginWithEmail(email, password);
            console.log('Usuario logueado con Firebase:', userCredential.user);

            // Obtener token de Firebase
            const firebaseToken = await getIdToken();
            
            if (!firebaseToken) {
                throw new Error('No se pudo obtener el token de Firebase');
            }

            // Opción 1: Usar solo Firebase token
            // onSuccess(firebaseToken, firebaseToken);

            // Opción 2: Enviar token a tu backend para obtener token JWT
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
                        onSuccess(data.access_token, firebaseToken);
                    } else {
                        // Si el backend no tiene endpoint de Firebase, usar solo Firebase token
                        onSuccess(firebaseToken, firebaseToken);
                    }
                } else {
                    // Si el backend no tiene endpoint de Firebase, usar solo Firebase token
                    console.warn('Backend no tiene endpoint de Firebase, usando solo Firebase token');
                    onSuccess(firebaseToken, firebaseToken);
                }
            } catch (backendError) {
                // Si falla el backend, usar solo Firebase token
                console.warn('Error conectando con backend, usando solo Firebase token:', backendError);
                onSuccess(firebaseToken, firebaseToken);
            }

        } catch (err: any) {
            console.error('Error de login con Firebase:', err);
            
            // Manejar errores específicos de Firebase
            let errorMessage = 'Error al iniciar sesión';
            
            if (err.code === 'auth/user-not-found') {
                errorMessage = 'Usuario no encontrado';
            } else if (err.code === 'auth/wrong-password') {
                errorMessage = 'Contraseña incorrecta';
            } else if (err.code === 'auth/invalid-email') {
                errorMessage = 'Email inválido';
            } else if (err.code === 'auth/network-request-failed') {
                errorMessage = 'Error de conexión. Verifica tu internet';
            } else if (err.message) {
                errorMessage = err.message;
            }
            
            setError(errorMessage);
        } finally {
            setIsLoading(false);
            setLoginMethod(null);
        }
    };

    const handleGoogleLogin = async () => {
        setIsLoading(true);
        setError(null);
        setLoginMethod('google');

        try {
            // Login con Google
            const userCredential = await loginWithGoogle();
            console.log('Usuario logueado con Google:', userCredential.user);

            // Obtener token de Firebase
            const firebaseToken = await getIdToken();
            
            if (!firebaseToken) {
                throw new Error('No se pudo obtener el token de Firebase');
            }

            // Intentar obtener token JWT del backend
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
                        onSuccess(data.access_token, firebaseToken);
                    } else {
                        onSuccess(firebaseToken, firebaseToken);
                    }
                } else {
                    onSuccess(firebaseToken, firebaseToken);
                }
            } catch (backendError) {
                console.warn('Error conectando con backend, usando solo Firebase token:', backendError);
                onSuccess(firebaseToken, firebaseToken);
            }

        } catch (err: any) {
            console.error('Error de login con Google:', err);
            
            let errorMessage = 'Error al iniciar sesión con Google';
            
            if (err.code === 'auth/popup-closed-by-user') {
                errorMessage = 'Ventana de Google cerrada';
            } else if (err.code === 'auth/popup-blocked') {
                errorMessage = 'Popup bloqueado. Permite popups para este sitio';
            } else if (err.message) {
                errorMessage = err.message;
            }
            
            setError(errorMessage);
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

                    {/* Botón de Google */}
                    <button
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
                            <span className="px-2 bg-gray-950 text-slate-400">Or continue with email</span>
                        </div>
                    </div>

                    <form onSubmit={handleEmailLogin} className="space-y-4">
                        <div>
                            <label htmlFor="email" className="text-sm font-bold text-slate-400">Email</label>
                            <input
                                id="email"
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                placeholder="tu@email.com"
                                required
                                disabled={isLoading}
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
                                placeholder="••••••••"
                                required
                                disabled={isLoading}
                            />
                        </div>

                        {error && (
                            <div className="text-sm text-red-400 bg-red-900/30 p-3 rounded-md border border-red-800/50">
                                {error}
                            </div>
                        )}
                        
                        <button 
                            type="submit" 
                            disabled={isLoading} 
                            className="w-full bg-slate-800 text-white font-bold py-3 px-5 rounded-md hover:bg-slate-700 transition-all disabled:opacity-50 disabled:bg-slate-800 disabled:cursor-not-allowed"
                        >
                            {isLoading && loginMethod === 'email' ? (
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
                                    disabled={isLoading}
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

