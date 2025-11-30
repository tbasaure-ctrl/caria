import React, { useState, useEffect } from 'react';
import { getToken } from '../services/apiService';
import { isGuestBannerDismissed, dismissGuestBanner, hasGuestData } from '../services/guestStorageService';

/**
 * GuestModeBanner - Non-intrusive notification for guest users
 * 
 * Shows a subtle banner encouraging account creation without blocking functionality.
 * Dismissible and respects 24-hour re-show interval.
 */
export const GuestModeBanner: React.FC = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [isAnimatingOut, setIsAnimatingOut] = useState(false);

    useEffect(() => {
        // Only show for guest users who haven't dismissed recently
        const token = getToken();
        const shouldShow = !token && !isGuestBannerDismissed();
        
        // Small delay to not flash on page load
        const timer = setTimeout(() => {
            setIsVisible(shouldShow);
        }, 1500);

        return () => clearTimeout(timer);
    }, []);

    const handleDismiss = () => {
        setIsAnimatingOut(true);
        dismissGuestBanner();
        
        // Wait for animation to complete
        setTimeout(() => {
            setIsVisible(false);
        }, 300);
    };

    const handleCreateAccount = () => {
        // Navigate to login with account creation intent
        window.location.href = '/?login=true';
    };

    if (!isVisible) return null;

    return (
        <div 
            className={`
                fixed bottom-20 md:bottom-6 left-4 right-4 md:left-auto md:right-6 md:max-w-md z-40
                transition-all duration-300 ease-out
                ${isAnimatingOut ? 'opacity-0 translate-y-4' : 'opacity-100 translate-y-0'}
            `}
        >
            <div className="glass-card rounded-xl p-4 shadow-xl border border-accent-cyan/20">
                <div className="flex items-start gap-3">
                    {/* Icon */}
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-accent-cyan/10 flex items-center justify-center">
                        <svg className="w-5 h-5 text-accent-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium text-white mb-1">
                            Guest Mode Active
                        </h4>
                        <p className="text-xs text-text-secondary leading-relaxed">
                            Your data is saved locally on this device. Create a free account to sync across devices and unlock all features.
                        </p>
                        
                        {/* Actions */}
                        <div className="flex items-center gap-3 mt-3">
                            <button
                                onClick={handleCreateAccount}
                                className="px-3 py-1.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-xs font-medium hover:bg-accent-cyan/30 transition-colors"
                            >
                                Create Account
                            </button>
                            <button
                                onClick={handleDismiss}
                                className="text-xs text-text-muted hover:text-text-secondary transition-colors"
                            >
                                Maybe later
                            </button>
                        </div>
                    </div>

                    {/* Close button */}
                    <button
                        onClick={handleDismiss}
                        className="flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center text-text-muted hover:text-white hover:bg-white/10 transition-colors"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default GuestModeBanner;

