import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { CariaLogoIcon, LogoutIcon, DashboardIcon, CommunityIcon, ThesisIcon, ChartIcon } from './Icons';
import { ChatWindow } from './ChatWindow';

interface SidebarProps {
    onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
    const [showChat, setShowChat] = useState(false);

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
                                    onClick={() => setShowChat(!showChat)}
                                    className="p-3 rounded-lg transition-all duration-200"
                                    style={{
                                      backgroundColor: showChat ? 'var(--color-bg-tertiary)' : 'transparent',
                                      color: showChat ? 'var(--color-cream)' : 'var(--color-text-muted)'
                                    }}
                                    onMouseEnter={(e) => {
                                      if (!showChat) {
                                        e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)';
                                        e.currentTarget.style.color = 'var(--color-cream)';
                                      }
                                    }}
                                    onMouseLeave={(e) => {
                                      if (!showChat) {
                                        e.currentTarget.style.backgroundColor = 'transparent';
                                        e.currentTarget.style.color = 'var(--color-text-muted)';
                                      }
                                    }}
                                    title="Chat"
                                    aria-label="Open Chat"
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
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
            {showChat && (
                <div className="fixed bottom-4 right-4 w-96 h-[600px] z-50 md:right-24">
                    <ChatWindow onClose={() => setShowChat(false)} />
                </div>
            )}
        </>
    );
};