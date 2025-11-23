import React, { useState, useEffect } from 'react';
import { CariaLogoIcon, XIcon } from './Icons';

interface RegisterModalProps {
    onClose: () => void;
    onSuccess: (token: string) => void;
    onSwitchToLogin?: () => void;
}

export const RegisterModal: React.FC<RegisterModalProps> = ({ onClose, onSuccess, onSwitchToLogin }) => {
    const [email, setEmail] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [fullName, setFullName] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [passwordStrength, setPasswordStrength] = useState(0);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    // Calculate password strength
    useEffect(() => {
        let strength = 0;
        if (password.length >= 8) strength++;
        if (password.length >= 12) strength++;
        if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++;
        if (/\d/.test(password)) strength++;
        if (/[^a-zA-Z\d]/.test(password)) strength++;
        setPasswordStrength(strength);
    }, [password]);

    const getPasswordStrengthColor = () => {
        if (passwordStrength < 2) return 'bg-red-500';
        if (passwordStrength < 4) return 'bg-yellow-500';
        return 'bg-green-500';
    };

    const getPasswordStrengthText = () => {
        if (passwordStrength < 2) return 'Weak';
        if (passwordStrength < 4) return 'Medium';
        return 'Strong';
    };

    // Función para limpiar contraseña de caracteres invisibles
    const cleanPassword = (pwd: string): string => {
        // Eliminar caracteres invisibles comunes (zero-width spaces, etc.)
        return pwd
            .replace(/[\u200B-\u200D\uFEFF]/g, '') // Zero-width spaces
            .replace(/[\u202A-\u202E]/g, '') // Directional formatting
            .replace(/[\u2060-\u206F]/g, '') // Word joiners
            .trim(); // Espacios al inicio/final
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        // Limpiar la contraseña de caracteres invisibles
        const cleanedPassword = cleanPassword(password);
        
        // Si la contraseña cambió después de limpiar, actualizar el estado
        if (cleanedPassword !== password) {
            console.warn('Password contained invisible characters. Cleaned from', password.length, 'to', cleanedPassword.length, 'characters');
            setPassword(cleanedPassword);
        }

        // Validation
        if (cleanedPassword.length < 8) {
            setError('Password must be at least 8 characters long');
            setIsLoading(false);
            return;
        }

        // Validar longitud de bytes (bcrypt tiene límite de 72 bytes)
        const passwordBytes = new TextEncoder().encode(cleanedPassword);
        if (passwordBytes.length > 72) {
            // Debug: mostrar información detallada
            console.error('Password validation failed:', {
                originalLength: password.length,
                cleanedLength: cleanedPassword.length,
                passwordBytes: passwordBytes.length,
                passwordPreview: cleanedPassword.substring(0, 20) + (cleanedPassword.length > 20 ? '...' : ''),
                passwordCharCodes: Array.from(cleanedPassword.substring(0, 50)).map(c => `${c}:${c.charCodeAt(0)}`),
            });
            setError(`Password is too long. Maximum length is 72 bytes when encoded. Your password has ${cleanedPassword.length} characters but is ${passwordBytes.length} bytes when encoded. Please use a shorter password (maximum ~50 characters for safety).`);
            setIsLoading(false);
            return;
        }

        if (username.length < 3) {
            setError('Username must be at least 3 characters long');
            setIsLoading(false);
            return;
        }

        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            setError('Please enter a valid email address');
            setIsLoading(false);
            return;
        }

        try {
            // Use centralized API_BASE_URL per audit document (must be absolute URL)
            const { API_BASE_URL } = await import('../services/apiConfig');
            const registerUrl = `${API_BASE_URL}/api/auth/register`;
            
            // Detailed diagnostic logging
            console.group('🔍 Registration Request Diagnostics');
            console.log('API_BASE_URL:', API_BASE_URL);
            console.log('VITE_API_URL from env:', import.meta.env.VITE_API_URL);
            console.log('Register URL:', registerUrl);
            console.log('Current origin:', window.location.origin);
            console.log('Request will be sent from:', window.location.href);
            console.groupEnd();
            
            const response = await fetch(registerUrl, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    email, 
                    username, 
                    password: cleanedPassword, // Usar la contraseña limpia
                    full_name: fullName || undefined,
                }),
            });
            
            console.log('Response status:', response.status);
            console.log('Response headers:', Object.fromEntries(response.headers.entries()));

            if (!response.ok) {
                let errorMessage = 'Registration failed';
                try {
                    const errData = await response.json();
                    errorMessage = errData.detail || errData.message || errorMessage;
                } catch {
                    // Si no se puede parsear JSON, usar el status text
                    errorMessage = `Registration failed: ${response.status} ${response.statusText}`;
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
                throw new Error('Registration response did not include a token.');
            }
        } catch (err: any) {
            // Enhanced error logging with CORS diagnostics
            console.group('❌ Registration Error Details');
            console.error('Error message:', err.message);
            console.error('Error name:', err.name);
            console.error('Error stack:', err.stack);
            console.error('Error cause:', err.cause);
            
            const { API_BASE_URL } = await import('../services/apiConfig');
            console.error('API_BASE_URL:', API_BASE_URL);
            console.error('Current origin:', window.location.origin);
            console.error('Full error object:', err);
            
            // Check for CORS-specific errors
            const isCorsError = err.message.includes('CORS') || 
                               err.message.includes('cross-origin') ||
                               err.message.includes('Access-Control') ||
                               err.name === 'TypeError' && err.message.includes('Failed to fetch');
            
            if (isCorsError) {
                console.error('🚨 CORS Error Detected');
                console.error('This usually means:');
                console.error('1. The backend CORS configuration does not allow this origin');
                console.error('2. The VITE_API_URL might be incorrect');
                console.error('3. The backend might not be responding to OPTIONS requests');
                console.error('Origin:', window.location.origin);
                console.error('Expected backend:', API_BASE_URL);
            }
            console.groupEnd();

            if (err.message === 'Failed to fetch' || err.name === 'TypeError' || err.message.includes('Failed to connect') || err.message.includes('NetworkError')) {
                // Check if it's a CORS error specifically
                if (isCorsError) {
                    setError(
                        `CORS Error: The API at ${API_BASE_URL} is not allowing requests from ${window.location.origin}. ` +
                        `Please check CORS configuration. Open browser console for details.`
                    );
                } else {
                    setError(
                        `Unable to connect to the server at ${API_BASE_URL}. ` +
                        `Please check that the API is running. Error: ${err.message}. ` +
                        `Open browser console for details.`
                    );
                }
            } else {
                setError(err.message || 'Registration failed. Please try again.');
            }
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
            aria-labelledby="register-modal-title"
        >
            <div 
                className="relative w-full max-w-sm bg-gray-950 text-slate-200 rounded-2xl border border-slate-800/50 overflow-hidden modal-fade-in" 
                onClick={(e) => e.stopPropagation()}
            >
                <header className="p-4 border-b border-slate-800/50 flex items-center justify-between">
                    <h1 id="register-modal-title" className="text-xl font-bold text-[#E0E1DD]">Create Account</h1>
                    <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors" aria-label="Close register modal">
                        <XIcon className="w-6 h-6" />
                    </button>
                </header>

                <main className="p-6">
                    <div className="flex flex-col items-center text-center mb-6">
                        <CariaLogoIcon className="w-12 h-12 text-slate-400 mb-3"/>
                        <p className="text-slate-400 text-sm">Create an account to access your dashboard.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label htmlFor="email" className="text-sm font-bold text-slate-400">Email</label>
                            <input
                                id="email"
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                required
                                autoComplete="email"
                            />
                        </div>
                        <div>
                            <label htmlFor="username-register" className="text-sm font-bold text-slate-400">Username</label>
                            <input
                                id="username-register"
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                required
                                minLength={3}
                                maxLength={50}
                                pattern="[a-zA-Z0-9_-]+"
                                autoComplete="username"
                            />
                            <p className="text-xs text-slate-500 mt-1">3-50 characters, alphanumeric, underscores, hyphens</p>
                        </div>
                        <div>
                            <label htmlFor="password-register" className="text-sm font-bold text-slate-400">Password</label>
                            <input
                                id="password-register"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                required
                                minLength={8}
                                autoComplete="new-password"
                            />
                            {password && (
                                <div className="mt-2">
                                    <div className="flex items-center gap-2 mb-1">
                                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                                            <div 
                                                className={`h-full ${getPasswordStrengthColor()} transition-all`}
                                                style={{ width: `${(passwordStrength / 5) * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-xs text-slate-400">{getPasswordStrengthText()}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                        <div>
                            <label htmlFor="full-name" className="text-sm font-bold text-slate-400">Full Name (Optional)</label>
                            <input
                                id="full-name"
                                type="text"
                                value={fullName}
                                onChange={(e) => setFullName(e.target.value)}
                                className="mt-1 w-full bg-gray-800 border border-slate-700 rounded-md py-2 px-3 focus:outline-none focus:ring-1 focus:ring-slate-600 transition-shadow"
                                autoComplete="name"
                            />
                        </div>

                        {error && <p className="text-sm text-red-400 bg-red-900/30 p-2 rounded-md">{error}</p>}
                        
                        <button 
                            type="submit" 
                            disabled={isLoading || passwordStrength < 2} 
                            className="w-full bg-slate-800 text-white font-bold py-3 px-5 rounded-md hover:bg-slate-700 transition-all disabled:opacity-50 disabled:bg-slate-800 disabled:cursor-not-allowed"
                        >
                            {isLoading ? 'Creating Account...' : 'Create Account'}
                        </button>
                    </form>

                    {onSwitchToLogin && (
                        <div className="mt-4 text-center">
                            <p className="text-sm text-slate-400">
                                Already have an account?{' '}
                                <button 
                                    onClick={onSwitchToLogin}
                                    className="text-slate-300 hover:text-white underline"
                                >
                                    Sign in
                                </button>
                            </p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

