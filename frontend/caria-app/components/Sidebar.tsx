import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { CariaLogoIcon, LogoutIcon, DashboardIcon, CommunityIcon, ThesisIcon } from './Icons';

interface SidebarProps {
    onLogout: () => void;
}

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
                        <div className="flex flex-col items-center text-center">
                            <div 
                                className="w-16 h-16 rounded-full flex items-center justify-center mb-4"
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
                                User profile settings
                            </p>
                            <span 
                                className="text-xs mt-2 px-3 py-1 rounded-full"
                                style={{ 
                                    backgroundColor: 'var(--color-bg-surface)',
                                    color: 'var(--color-text-muted)'
                                }}
                            >
                                Coming soon
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};
