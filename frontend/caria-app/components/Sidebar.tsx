import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { CariaLogoIcon, LogoutIcon, DashboardIcon, CommunityIcon, ThesisIcon } from './Icons';
import { API_BASE_URL, getToken, saveToken } from '../services/apiService';

interface SidebarProps {
    onLogout: () => void;
}

interface UserProfileFormProps {
    onClose: () => void;
}

const UserProfileForm: React.FC<UserProfileFormProps> = ({ onClose }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [occupation, setOccupation] = useState('');
    const [selfDescription, setSelfDescription] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

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
                    occupation,
                    self_description: selfDescription
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
            saveToken(loginData.access_token);
            setSuccess(true);
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        } catch (err: any) {
            setError(err.message || 'An error occurred during registration');
        } finally {
            setIsLoading(false);
        }
    };

    const token = getToken();
    if (token) {
        return (
            <div className="text-center">
                <div 
                    className="w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto"
                    style={{ backgroundColor: 'var(--color-bg-surface)' }}
                >
                    <svg 
                        className="w-8 h-8" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                </div>
                <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
                    You are logged in
                </p>
            </div>
        );
    }

    if (success) {
        return (
            <div className="text-center">
                <div 
                    className="w-16 h-16 rounded-full flex items-center justify-center mb-4 mx-auto"
                    style={{ backgroundColor: 'var(--color-positive-muted)' }}
                >
                    <svg 
                        className="w-8 h-8" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                        style={{ color: 'var(--color-positive)' }}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                </div>
                <p className="text-sm font-semibold" style={{ color: 'var(--color-positive)' }}>
                    Account created successfully!
                </p>
                <p className="text-xs mt-2" style={{ color: 'var(--color-text-secondary)' }}>
                    Redirecting...
                </p>
            </div>
        );
    }

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                <h3 
                    className="text-base font-semibold mb-4"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)' 
                    }}
                >
                    Create Account
                </h3>
            </div>

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
                    className="w-full px-4 py-2.5 rounded-lg text-sm"
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
                    className="w-full px-4 py-2.5 rounded-lg text-sm"
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
                    className="w-full px-4 py-2.5 rounded-lg text-sm"
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
                    className="w-full px-4 py-2.5 rounded-lg text-sm"
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
                    className="w-full px-4 py-2.5 rounded-lg text-sm resize-none"
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
                className="w-full py-2.5 rounded-lg font-semibold text-sm transition-all duration-200 disabled:opacity-50"
                style={{
                    backgroundColor: 'var(--color-accent-primary)',
                    color: '#FFFFFF',
                }}
            >
                {isLoading ? 'Creating Account...' : 'Create Account'}
            </button>
        </form>
    );
};

export const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
    const [showProfile, setShowProfile] = useState(false);

    const navItems = [
        { to: '/dashboard', icon: DashboardIcon, label: 'Terminal', description: 'Main dashboard' },
        { to: '/community', icon: CommunityIcon, label: 'Community', description: 'Discussions & ideas' },
        { to: '/resources', icon: ThesisIcon, label: 'Research', description: 'Learning resources' },
    ];

    const getLinkClass = ({ isActive }: { isActive: boolean }) => `
        relative flex items-center justify-center w-12 h-12 rounded-lg transition-all duration-200
        ${isActive
            ? 'bg-accent-primary/20 text-accent-primary'
            : 'text-text-muted hover:bg-bg-surface hover:text-text-primary'
        }
    `;

    return (
        <>
            {/* Desktop Sidebar - Terminal Style */}
            <aside 
                className="hidden md:flex flex-col w-[72px] h-screen shrink-0 border-r"
                style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    borderColor: 'var(--color-border-subtle)'
                }}
            >
                {/* Logo Section */}
                <div className="flex flex-col items-center py-5 border-b" style={{ borderColor: 'var(--color-border-subtle)' }}>
                    <NavLink 
                        to="/dashboard" 
                        aria-label="Caria Home"
                        className="group relative"
                    >
                        <div 
                            className="w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 group-hover:scale-105"
                            style={{ 
                                backgroundColor: 'var(--color-bg-surface)',
                                border: '1px solid var(--color-border-subtle)'
                            }}
                        >
                            <CariaLogoIcon 
                                className="w-6 h-6" 
                                style={{ color: 'var(--color-accent-primary)' }} 
                            />
                        </div>
                    </NavLink>
                </div>

                {/* Main Navigation */}
                <nav className="flex-1 py-4">
                    <ul className="flex flex-col items-center gap-2 px-3">
                        {navItems.map((item) => (
                            <li key={item.to} className="w-full">
                                <NavLink
                                    to={item.to}
                                    className={getLinkClass}
                                    title={item.label}
                                >
                                    {({ isActive }) => (
                                        <>
                                            {/* Active Indicator */}
                                            {isActive && (
                                                <div 
                                                    className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 rounded-r-full"
                                                    style={{ backgroundColor: 'var(--color-accent-primary)' }}
                                                />
                                            )}
                                            <item.icon className="w-5 h-5" />
                                        </>
                                    )}
                                </NavLink>
                            </li>
                        ))}
                    </ul>
                </nav>

                {/* Bottom Section */}
                <div className="py-4 px-3 border-t" style={{ borderColor: 'var(--color-border-subtle)' }}>
                    <ul className="flex flex-col items-center gap-2">
                        {/* Profile Button */}
                        <li className="w-full">
                            <button
                                onClick={() => setShowProfile(!showProfile)}
                                className={`
                                    w-full flex items-center justify-center h-12 rounded-lg transition-all duration-200
                                    ${showProfile ? 'bg-bg-surface text-text-primary' : 'text-text-muted hover:bg-bg-surface hover:text-text-primary'}
                                `}
                                title="Profile & Settings"
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                            </button>
                        </li>

                        {/* Logout Button */}
                        <li className="w-full">
                            <button
                                onClick={onLogout}
                                className="w-full flex items-center justify-center h-12 rounded-lg text-text-muted hover:bg-negative-muted hover:text-negative transition-all duration-200"
                                title="Sign Out"
                            >
                                <LogoutIcon className="w-5 h-5" />
                            </button>
                        </li>
                    </ul>
                </div>

                {/* Version Tag */}
                <div 
                    className="text-center py-3 border-t"
                    style={{ borderColor: 'var(--color-border-subtle)' }}
                >
                    <span 
                        className="text-[9px] font-mono tracking-wider uppercase"
                        style={{ color: 'var(--color-text-subtle)' }}
                    >
                        v1.0
                    </span>
                </div>
            </aside>

            {/* Mobile Bottom Navigation */}
            <nav 
                className="md:hidden fixed bottom-0 left-0 right-0 z-50 border-t"
                style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    borderColor: 'var(--color-border-subtle)'
                }}
            >
                <div className="flex justify-around items-center h-16 px-4">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.to}
                            to={item.to}
                            className={({ isActive }) => `
                                flex flex-col items-center justify-center gap-1 px-4 py-2 rounded-lg transition-colors
                                ${isActive ? 'text-accent-primary' : 'text-text-muted'}
                            `}
                        >
                            <item.icon className="w-5 h-5" />
                            <span className="text-[10px] font-medium">{item.label}</span>
                        </NavLink>
                    ))}
                    <button
                        onClick={onLogout}
                        className="flex flex-col items-center justify-center gap-1 px-4 py-2 text-text-muted"
                    >
                        <LogoutIcon className="w-5 h-5" />
                        <span className="text-[10px] font-medium">Sign Out</span>
                    </button>
                </div>
            </nav>

            {/* Profile Panel */}
            {showProfile && (
                <div 
                    className="fixed z-50 w-72 rounded-xl shadow-xl overflow-hidden animate-fade-in-scale"
                    style={{
                        left: '84px',
                        bottom: '100px',
                        backgroundColor: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border-default)',
                    }}
                >
                    {/* Panel Header */}
                    <div 
                        className="flex justify-between items-center px-5 py-4 border-b"
                        style={{ borderColor: 'var(--color-border-subtle)' }}
                    >
                        <h3 
                            className="text-sm font-semibold"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Profile & Settings
                        </h3>
                        <button 
                            onClick={() => setShowProfile(false)}
                            className="w-6 h-6 rounded flex items-center justify-center transition-colors hover:bg-bg-surface"
                            style={{ color: 'var(--color-text-muted)' }}
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    {/* Panel Content */}
                    <div className="p-5">
                        <UserProfileForm onClose={() => setShowProfile(false)} />
                    </div>
                </div>
            )}
        </>
    );
};
