import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { CariaLogoIcon, LogoutIcon, DashboardIcon, CommunityIcon, ThesisIcon } from './Icons';

interface SidebarProps {
    onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
    const [showProfile, setShowProfile] = useState(false);

    const getLinkClass = ({ isActive }: { isActive: boolean }) => `
        p-3 rounded-lg block transition-all duration-200
        ${isActive
            ? 'bg-[var(--color-primary)] text-[var(--color-cream)]'
            : 'text-[var(--color-text-muted)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-cream)]'
        }
    `;

    return (
        <>
            <aside className="hidden md:flex flex-col w-20 p-4 items-center justify-between shrink-0"
                   style={{
                     backgroundColor: 'var(--color-bg-secondary)',
                     borderRight: '1px solid var(--color-bg-tertiary)'
                   }}>
                <div className="flex flex-col items-center gap-8">
                    <NavLink to="/dashboard" aria-label="Caria Home" className="transition-transform hover:scale-110">
                        <CariaLogoIcon className="w-9 h-9" style={{color: 'var(--color-secondary)'}} />
                    </NavLink>
                    <nav aria-label="Main navigation">
                        <ul className="flex flex-col gap-4">
                            <li>
                                <NavLink
                                   to="/dashboard"
                                   className={getLinkClass}
                                   title="Dashboard"
                                >
                                    <DashboardIcon className="w-6 h-6" />
                                </NavLink>
                            </li>
                             <li>
                                <NavLink
                                   to="/community"
                                   className={getLinkClass}
                                   title="Community"
                                >
                                    <CommunityIcon className="w-6 h-6" />
                                </NavLink>
                            </li>
                             <li>
                                <NavLink
                                   to="/resources"
                                   className={getLinkClass}
                                   title="Resources"
                                >
                                    <ThesisIcon className="w-6 h-6" />
                                </NavLink>
                            </li>
                            <li>
                                <button
                                    onClick={() => setShowProfile(!showProfile)}
                                    className="p-3 rounded-lg transition-all duration-200"
                                    style={{
                                      backgroundColor: showProfile ? 'var(--color-bg-tertiary)' : 'transparent',
                                      color: showProfile ? 'var(--color-cream)' : 'var(--color-text-muted)'
                                    }}
                                    onMouseEnter={(e) => {
                                      if (!showProfile) {
                                        e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
                                        e.currentTarget.style.color = 'var(--color-cream)';
                                      }
                                    }}
                                    onMouseLeave={(e) => {
                                      if (!showProfile) {
                                        e.currentTarget.style.backgroundColor = 'transparent';
                                        e.currentTarget.style.color = 'var(--color-text-muted)';
                                      }
                                    }}
                                    title="Profile & Settings"
                                    aria-label="Open Profile"
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                    </svg>
                                </button>
                            </li>
                        </ul>
                    </nav>
                </div>
                <button
                    onClick={onLogout}
                    className="p-3 rounded-lg transition-all duration-200"
                    aria-label="Log Out"
                    title="Log Out"
                    style={{color: 'var(--color-text-muted)'}}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'var(--color-primary)';
                      e.currentTarget.style.color = 'var(--color-cream)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.color = 'var(--color-text-muted)';
                    }}
                >
                    <LogoutIcon className="w-6 h-6" />
                </button>
            </aside>
            {/* Profile Panel - Coming Soon */}
            {showProfile && (
                <div className="fixed bottom-4 right-4 w-80 z-50 md:right-24 bg-gray-900 rounded-lg border border-slate-700 shadow-xl p-6">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-semibold text-slate-200">Profile & Settings</h3>
                        <button onClick={() => setShowProfile(false)} className="text-slate-400 hover:text-white">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    <div className="text-center py-8">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                            <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                        </div>
                        <p className="text-slate-400 text-sm">User profile and settings</p>
                        <p className="text-slate-500 text-xs mt-2">Coming soon</p>
                    </div>
                </div>
            )}
        </>
    );
};