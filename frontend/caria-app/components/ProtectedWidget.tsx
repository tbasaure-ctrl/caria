import React from 'react';

interface ProtectedWidgetProps {
    children: React.ReactNode;
    featureName: string;
    description?: string;
}

/**
 * ProtectedWidget - Now a pass-through component for Guest Mode support.
 * 
 * All widgets are now accessible without login. Guest mode stores data
 * in localStorage. Users see a non-intrusive banner encouraging account creation.
 * 
 * Previously this would block access and show a login prompt.
 * Now it simply renders children directly, enabling full Guest Mode functionality.
 */
export const ProtectedWidget: React.FC<ProtectedWidgetProps> = ({ children }) => {
    // Guest Mode: All widgets are now accessible
    // Data persistence is handled by individual widgets using guestStorageService
    return <>{children}</>;
};
