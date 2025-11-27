import React, { useState, useEffect } from 'react';
import { NavLink, useSearchParams, useLocation } from 'react-router-dom';
import { CariaLogoIcon, LogoutIcon, PortfolioIcon, ChartIcon, ThesisIcon } from './Icons';
import { API_BASE_URL, getToken, saveToken, fetchWithAuth } from '../services/apiService';

interface SidebarProps {
    onLogout: () => void;
}

interface UserProfileFormProps {
    onClose: () => void;
}

interface UserProfile {
    id: string;
    email: string;
    username: string;
    full_name: string | null;
    is_active: boolean;
    is_verified: boolean;
    created_at: string;
    last_login: string | null;
}

const UserProfileForm: React.FC<UserProfileFormProps> = ({ onClose }) => {
    const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
    const [fullName, setFullName] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingProfile, setIsLoadingProfile] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

    // Fetch user profile on mount
    useEffect(() => {
        const fetchProfile = async () => {
            const token = getToken();
            if (!token) {
                setIsLoadingProfile(false);
                return;
            }

            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/api/auth/me`);
                if (response.ok) {
                    const profile = await response.json();
                    setUserProfile(profile);
                    setFullName(profile.full_name || '');
                }
            } catch (err: any) {
                console.error('Error fetching user profile:', err);
                setError('Failed to load profile');
            } finally {
                setIsLoadingProfile(false);
            }
        };

        fetchProfile();
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setSuccess(false);

        const token = getToken();
        if (!token) {
            setError('You must be logged in to update your profile');
            return;
        }

        setIsLoading(true);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/auth/me`, {
                method: 'PUT',
                body: JSON.stringify({
                    full_name: fullName.trim() || null
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to update profile');
            }

            const updatedProfile = await response.json();
            setUserProfile(updatedProfile);
            setSuccess(true);
            setTimeout(() => {
                setSuccess(false);
            }, 2000);
        } catch (err: any) {
            setError(err.message || 'An error occurred while updating your profile');
        } finally {
            setIsLoading(false);
        }
    };

    const token = getToken();
    if (!token) {
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
                    Please log in to view your profile
                </p>
            </div>
        );
    }

    if (isLoadingProfile) {
        return (
            <div className="text-center py-8">
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                    Loading profile...
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
                    Profile updated successfully!
                </p>
            </div>
        );
    }

    if (!userProfile) {
        return (
            <div className="text-center py-8">
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                    Unable to load profile
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
                    Profile & Settings
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
                    value={userProfile.email}
                    disabled
                    className="w-full px-4 py-2.5 rounded-lg text-sm opacity-60 cursor-not-allowed"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)',
                        color: 'var(--color-text-primary)',
                    }}
                />
                <p className="text-xs mt-1" style={{ color: 'var(--color-text-subtle)' }}>
                    Email cannot be changed
                </p>
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
                    value={userProfile.username}
                    disabled
                    className="w-full px-4 py-2.5 rounded-lg text-sm opacity-60 cursor-not-allowed"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)',
                        color: 'var(--color-text-primary)',
                    }}
                />
                <p className="text-xs mt-1" style={{ color: 'var(--color-text-subtle)' }}>
                    Username cannot be changed
                </p>
            </div>

            <div>
                <label 
                    className="block text-xs font-medium tracking-wider uppercase mb-2"
                    style={{ color: 'var(--color-text-muted)' }}
                >
                    Full Name <span className="text-xs normal-case" style={{ color: 'var(--color-text-subtle)' }}>(optional)</span>
                </label>
                <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-lg text-sm"
                    style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        border: '1px solid var(--color-border-subtle)',
                        color: 'var(--color-text-primary)',
                    }}
                    placeholder="Your full name"
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
                {isLoading ? 'Updating Profile...' : 'Update Profile'}
            </button>
        </form>
    );
};

export const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
    const [showProfile, setShowProfile] = useState(false);
    const [searchParams] = useSearchParams();
    const location = useLocation();

    const navItems = [
        { to: '/dashboard?tab=portfolio', icon: PortfolioIcon, label: 'Portfolio', description: 'Portfolio management' },
        { to: '/dashboard?tab=analysis', icon: ChartIcon, label: 'Analysis', description: 'Stock analysis & valuation' },
        { to: '/dashboard?tab=research', icon: ThesisIcon, label: 'Research', description: 'Research & resources' },
    ];

    const getLinkClass = ({ isActive }: { isActive: boolean }) => `
        relative flex items-center justify-center w-12 h-12 rounded-lg transition-all duration-200
        ${isActive
            ? 'bg-accent-primary/20 text-accent-primary'
            : 'text-text-muted hover:bg-bg-surface hover:text-text-primary'
        }
    `;

    // Check if a nav item is active based on the tab parameter
    const isNavItemActive = (to: string) => {
        const currentPath = location.pathname;
        const currentTab = searchParams.get('tab');
        const itemTab = new URLSearchParams(to.split('?')[1] || '').get('tab');
        
        if (currentPath === '/dashboard' && itemTab) {
            return currentTab === itemTab || (!currentTab && itemTab === 'portfolio');
        }
        return false;
    };

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
                        to="/dashboard?tab=portfolio" 
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
                        {navItems.map((item) => {
                            const isActive = isNavItemActive(item.to);
                            return (
                                <li key={item.to} className="w-full">
                                    <NavLink
                                        to={item.to}
                                        className={getLinkClass({ isActive })}
                                        title={item.label}
                                    >
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
                                    </NavLink>
                                </li>
                            );
                        })}
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
                    {navItems.map((item) => {
                        const isActive = isNavItemActive(item.to);
                        return (
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className={`
                                    flex flex-col items-center justify-center gap-1 px-4 py-2 rounded-lg transition-colors
                                    ${isActive ? 'text-accent-primary' : 'text-text-muted'}
                                `}
                            >
                                <item.icon className="w-5 h-5" />
                                <span className="text-[10px] font-medium">{item.label}</span>
                            </NavLink>
                        );
                    })}
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
