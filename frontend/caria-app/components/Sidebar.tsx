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
    if (!token) return null;

    if (isLoadingProfile) {
        return <div className="text-center py-4 text-xs text-text-muted">Loading...</div>;
    }

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div>
                <h3 className="text-sm font-display text-white mb-4 tracking-wide">Profile</h3>
            </div>

            {error && (
                <div className="px-3 py-2 rounded bg-negative-muted border border-negative text-negative text-xs">
                    {error}
                </div>
            )}
            
            {success && (
                <div className="px-3 py-2 rounded bg-positive-muted border border-positive text-positive text-xs">
                    Updated successfully
                </div>
            )}

            <div className="space-y-3">
                <div>
                    <label className="block text-[10px] uppercase tracking-wider text-text-muted mb-1">Email</label>
                    <div className="text-xs text-text-secondary font-mono">{userProfile?.email}</div>
                </div>
                <div>
                    <label className="block text-[10px] uppercase tracking-wider text-text-muted mb-1">Username</label>
                    <div className="text-xs text-text-secondary font-mono">@{userProfile?.username}</div>
                </div>
                <div>
                    <label className="block text-[10px] uppercase tracking-wider text-text-muted mb-1">Full Name</label>
                    <input
                        type="text"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1.5 text-xs text-white focus:border-accent-primary outline-none"
                        placeholder="Enter name"
                    />
                </div>
            </div>

            <button
                type="submit"
                disabled={isLoading}
                className="w-full py-2 rounded bg-accent-primary hover:bg-accent-primary/90 text-white text-xs font-bold tracking-wider transition-colors disabled:opacity-50"
            >
                {isLoading ? 'Saving...' : 'Save Changes'}
            </button>
        </form>
    );
};

export const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
    const [showProfile, setShowProfile] = useState(false);
    const [searchParams] = useSearchParams();
    const location = useLocation();

    const navItems = [
        { to: '/dashboard?tab=portfolio', icon: PortfolioIcon, label: 'Portfolio' },
        { to: '/dashboard?tab=analysis', icon: ChartIcon, label: 'Analysis' },
        { to: '/dashboard?tab=research', icon: ThesisIcon, label: 'Research' },
    ];

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
            {/* Desktop Sidebar - Ultra Minimal Dock */}
            <aside 
                className="hidden md:flex flex-col w-16 h-screen shrink-0 border-r border-white/5 bg-bg-primary z-50 items-center py-6"
            >
                {/* Logo */}
                <div className="mb-10">
                    <NavLink 
                        to="/dashboard?tab=portfolio" 
                        className="block w-10 h-10 text-accent-cyan hover:text-white transition-colors duration-300"
                    >
                        <CariaLogoIcon className="w-full h-full" />
                    </NavLink>
                </div>

                {/* Nav Icons */}
                <nav className="flex-1 flex flex-col gap-6 w-full items-center">
                    {navItems.map((item) => {
                        const isActive = isNavItemActive(item.to);
                        return (
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className={`
                                    relative w-10 h-10 flex items-center justify-center rounded-xl transition-all duration-300 group
                                    ${isActive 
                                        ? 'text-accent-cyan bg-accent-cyan/10 shadow-[0_0_15px_rgba(34,211,238,0.2)]' 
                                        : 'text-text-muted hover:text-white hover:bg-white/5'
                                    }
                                `}
                                title={item.label}
                            >
                                <item.icon className="w-5 h-5" />
                                
                                {/* Hover Label Tooltip */}
                                <span className="absolute left-full ml-4 px-2 py-1 bg-bg-secondary border border-white/10 rounded text-[10px] uppercase tracking-wider text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                                    {item.label}
                                </span>
                                
                                {/* Active Indicator Line */}
                                {isActive && (
                                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent-cyan rounded-r-full" />
                                )}
                            </NavLink>
                        );
                    })}
                </nav>

                {/* Bottom Actions */}
                <div className="flex flex-col gap-4 mt-auto w-full items-center">
                    <button
                        onClick={() => setShowProfile(!showProfile)}
                        className={`
                            w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200
                            ${showProfile ? 'text-white bg-white/10' : 'text-text-muted hover:text-white hover:bg-white/5'}
                        `}
                        title="Profile"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                    </button>

                    <button
                        onClick={onLogout}
                        className="w-10 h-10 rounded-xl flex items-center justify-center text-text-muted hover:text-negative hover:bg-negative/10 transition-all duration-200"
                        title="Sign Out"
                    >
                        <LogoutIcon className="w-5 h-5" />
                    </button>
                </div>
            </aside>

            {/* Profile Panel Popout */}
            {showProfile && (
                <div 
                    className="fixed z-50 w-72 rounded-xl shadow-2xl overflow-hidden animate-fade-in-scale bg-bg-secondary border border-white/10"
                    style={{ left: '70px', bottom: '20px' }}
                >
                    <div className="flex justify-between items-center px-5 py-4 border-b border-white/5 bg-bg-tertiary/50">
                        <h3 className="text-sm font-display text-white">Settings</h3>
                        <button onClick={() => setShowProfile(false)} className="text-text-muted hover:text-white">×</button>
                    </div>
                    <div className="p-5">
                        <UserProfileForm onClose={() => setShowProfile(false)} />
                    </div>
                </div>
            )}

            {/* Mobile Bottom Nav */}
            <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-bg-primary/95 backdrop-blur border-t border-white/10">
                <div className="flex justify-around items-center h-16">
                    {navItems.map((item) => {
                        const isActive = isNavItemActive(item.to);
                        return (
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className={`flex flex-col items-center p-2 ${isActive ? 'text-accent-cyan' : 'text-text-muted'}`}
                            >
                                <item.icon className="w-5 h-5" />
                                <span className="text-[9px] mt-1 uppercase tracking-wide">{item.label}</span>
                            </NavLink>
                        );
                    })}
                    <button onClick={onLogout} className="flex flex-col items-center p-2 text-text-muted">
                        <LogoutIcon className="w-5 h-5" />
                        <span className="text-[9px] mt-1 uppercase tracking-wide">Exit</span>
                    </button>
                </div>
            </nav>
        </>
    );
};
