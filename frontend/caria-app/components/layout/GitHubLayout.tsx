import React, { useState } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { Bell, Menu, X, User, Search, LogIn, UserPlus } from 'lucide-react';
import { getToken, removeToken } from '../../services/apiService';
import { CariaLogoIcon } from '../Icons';

const NavItem: React.FC<{
    label: string;
    path: string;
    isActive: boolean;
    onClick: (path: string) => void
}> = ({ label, path, isActive, onClick }) => (
    <button
        onClick={() => onClick(path)}
        className={`
            px-4 py-2 text-sm font-medium transition-all duration-200
            ${isActive
                ? 'text-white border-b-2 border-accent-primary'
                : 'text-text-muted hover:text-text-secondary hover:bg-white/5 rounded-md'
            }
        `}
    >
        {label}
    </button>
);

export const TopNav: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const isAuthenticated = !!getToken();

    const navItems = [
        { label: 'Portfolio', path: '/portfolio' },
        { label: 'Analysis', path: '/analysis' },
        { label: 'World Economies', path: '/world-economies' },
        { label: 'League', path: '/league' },
        { label: 'Research', path: '/research' },
        { label: 'About Us', path: '/about' },
    ];

    const handleNavClick = (path: string) => {
        navigate(path);
        setIsMobileMenuOpen(false);
    };

    const handleLogout = () => {
        removeToken();
        localStorage.removeItem('cariaChatHistory');
        navigate('/');
    };

    return (
        <nav className="w-full bg-[#020408] border-b border-white/10 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">

                    {/* Left: Logo & Desktop Nav */}
                    <div className="flex items-center gap-8">
                        <div
                            className="flex-shrink-0 cursor-pointer flex items-center gap-2"
                            onClick={() => navigate('/')}
                        >
                            <CariaLogoIcon className="w-8 h-8 text-accent-cyan shrink-0" />
                            <span className="font-display font-bold text-white text-lg tracking-wide hidden sm:block">CARIA</span>
                        </div>

                        <div className="hidden md:flex items-center gap-2">
                            {navItems.map((item) => (
                                <NavItem
                                    key={item.path}
                                    label={item.label}
                                    path={item.path}
                                    isActive={location.pathname.startsWith(item.path)}
                                    onClick={handleNavClick}
                                />
                            ))}
                        </div>
                    </div>

                    {/* Right: Search & Profile */}
                    <div className="flex items-center gap-4">
                        <div className="relative hidden sm:block">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                            <input
                                type="text"
                                placeholder="Search ticker..."
                                className="bg-bg-secondary border border-white/10 rounded-full pl-9 pr-4 py-1.5 text-sm text-white focus:border-accent-primary focus:outline-none w-64 transition-all"
                            />
                        </div>

                        {isAuthenticated ? (
                            <>
                                <button className="text-text-muted hover:text-white transition-colors relative">
                                    <Bell className="w-5 h-5" />
                                    <span className="absolute top-0 right-0 w-2 h-2 bg-accent-primary rounded-full"></span>
                                </button>

                                <div
                                    onClick={handleLogout}
                                    className="w-8 h-8 rounded-full bg-bg-tertiary border border-white/10 flex items-center justify-center text-text-secondary cursor-pointer hover:border-white/30 transition-colors group relative"
                                    title="Log Out"
                                >
                                    <User className="w-4 h-4" />
                                    <div className="absolute top-full right-0 mt-2 hidden group-hover:block bg-black border border-white/10 px-2 py-1 rounded text-xs text-white whitespace-nowrap">Log Out</div>
                                </div>
                            </>
                        ) : (
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => window.location.href = '/?login=true'}
                                    className="text-sm font-medium text-text-secondary hover:text-white px-3 py-1.5 transition-colors flex items-center gap-1"
                                >
                                    <LogIn className="w-4 h-4" /> <span className="hidden sm:inline">Log In</span>
                                </button>
                                <button
                                    onClick={() => window.location.href = '/?register=true'}
                                    className="text-sm font-bold text-black bg-white hover:bg-gray-200 px-3 py-1.5 rounded transition-colors flex items-center gap-1"
                                >
                                    <UserPlus className="w-4 h-4" /> <span className="hidden sm:inline">Sign Up</span>
                                </button>
                            </div>
                        )}

                        {/* Mobile Menu Button */}
                        <button
                            className="md:hidden text-text-muted hover:text-white"
                            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                        >
                            {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {isMobileMenuOpen && (
                <div className="md:hidden bg-bg-secondary border-b border-white/10 animate-fade-in">
                    <div className="px-4 pt-2 pb-4 space-y-2">
                        {navItems.map((item) => (
                            <button
                                key={item.path}
                                onClick={() => handleNavClick(item.path)}
                                className={`
                                    block w-full text-left px-4 py-3 rounded-lg text-base font-medium transition-colors
                                    ${location.pathname.startsWith(item.path)
                                        ? 'bg-white/10 text-white'
                                        : 'text-text-muted hover:text-white hover:bg-white/5'
                                    }
                                `}
                            >
                                {item.label}
                            </button>
                        ))}
                        {!isAuthenticated && (
                            <div className="mt-4 flex flex-col gap-3 pt-4 border-t border-white/10">
                                <button
                                    onClick={() => window.location.href = '/?login=true'}
                                    className="w-full text-center py-3 border border-white/20 rounded-lg text-white font-medium hover:bg-white/5 transition-colors"
                                >
                                    Log In
                                </button>
                                <button
                                    onClick={() => window.location.href = '/?register=true'}
                                    className="w-full text-center py-3 bg-white text-black rounded-lg font-bold hover:bg-gray-200 transition-colors"
                                >
                                    Sign Up
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </nav>
    );
};

export const GitHubLayout: React.FC = () => {
    return (
        <div className="min-h-screen bg-[#020408] text-text-primary font-sans selection:bg-accent-primary/30">
            <TopNav />
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-8">
                <Outlet />
            </div>
        </div>
    );
};
