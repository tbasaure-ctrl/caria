import React from 'react';
import { getToken } from '../services/apiService';

interface ProtectedWidgetProps {
    children: React.ReactNode;
    featureName: string;
}

export const ProtectedWidget: React.FC<ProtectedWidgetProps> = ({ children, featureName }) => {
    const token = getToken();

    if (!token) {
        return (
            <div 
                className="rounded-xl p-8 text-center"
                style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    border: '1px solid var(--color-border-subtle)',
                }}
            >
                <div 
                    className="w-16 h-16 mx-auto mb-4 rounded-xl flex items-center justify-center"
                    style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
                >
                    <svg 
                        className="w-8 h-8" 
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                        style={{ color: 'var(--color-text-muted)' }}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                </div>
                <h3 
                    className="text-lg font-semibold mb-2"
                    style={{ 
                        fontFamily: 'var(--font-display)',
                        color: 'var(--color-text-primary)' 
                    }}
                >
                    Authentication Required
                </h3>
                <p 
                    className="text-sm mb-4"
                    style={{ color: 'var(--color-text-secondary)' }}
                >
                    Please create an account to access {featureName}. Click the user icon in the sidebar to get started.
                </p>
            </div>
        );
    }

    return <>{children}</>;
};
