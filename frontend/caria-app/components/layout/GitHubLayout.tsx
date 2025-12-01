import React, { useState } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { Bell, Menu, X, User, Search } from 'lucide-react';

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

    const navItems = [
        { label: 'Portfolio', path: '/portfolio' },
        { label: 'Analysis', path: '/analysis' },
        { label: 'Research', path: '/research' },
        { label: 'About Us', path: '/about' },
    ];

    const handleNavClick = (path: string) => {
        navigate(path);
        setIsMobileMenuOpen(false);
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
                            <div className="w-8 h-8 rounded bg-white text-black flex items-center justify-center font-display font-bold text-xl">C</div>
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
                        
                        <button className="text-text-muted hover:text-white transition-colors relative">
                            <Bell className="w-5 h-5" />
                            <span className="absolute top-0 right-0 w-2 h-2 bg-accent-primary rounded-full"></span>
                        </button>
                        
                        <div className="w-8 h-8 rounded-full bg-bg-tertiary border border-white/10 flex items-center justify-center text-text-secondary cursor-pointer hover:border-white/30 transition-colors">
                            <User className="w-4 h-4" />
                        </div>

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
                <div className="md:hidden bg-bg-secondary border-b border-white/10">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        {navItems.map((item) => (
                            <button
                                key={item.path}
                                onClick={() => handleNavClick(item.path)}
                                className={`
                                    block w-full text-left px-3 py-2 rounded-md text-base font-medium
                                    ${location.pathname.startsWith(item.path)
                                        ? 'bg-white/10 text-white'
                                        : 'text-text-muted hover:text-white hover:bg-white/5'
                                    }
                                `}
                            >
                                {item.label}
                            </button>
                        ))}
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
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <Outlet />
            </div>
        </div>
    );
};

