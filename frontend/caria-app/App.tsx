import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { LandingPage } from './components/LandingPage';
import { Sidebar } from './components/Sidebar';
import { LoginModal } from './components/LoginModal';
import { RegisterModal } from './components/RegisterModal';
import { DashboardPage } from './components/pages/DashboardPage';
import { CommunityPage } from './components/pages/CommunityPage';
import { ResourcesPage } from './components/pages/ResourcesPage';
import { getToken, saveToken, removeToken } from './services/apiService';

// Protected Route Component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const token = getToken();
  if (!token) {
    return <Navigate to="/" replace />;
  }
  return <>{children}</>;
};

// Dashboard Layout (Sidebar + Content)
const DashboardLayout: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
  return (
    <div className="flex h-screen w-full bg-[var(--color-bg-primary)]">
      <Sidebar onLogout={onLogout} />
      <div className="flex-1 overflow-y-auto bg-[var(--color-bg-primary)]">
        <Outlet />
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [authToken, setAuthToken] = useState<string | null>(getToken());
  const [isLoginModalOpen, setLoginModalOpen] = useState(false);
  const [isRegisterModalOpen, setRegisterModalOpen] = useState(false);

  const handleShowLogin = () => {
    setLoginModalOpen(true);
    setRegisterModalOpen(false);
  };

  const handleShowRegister = () => {
    setRegisterModalOpen(true);
    setLoginModalOpen(false);
  };

  // Effect to check for token on initial load
  useEffect(() => {
    const token = getToken();
    if (token) {
      setAuthToken(token);
    }
    
    // Check for login query parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('login') === 'true' && !token) {
      handleShowLogin();
      // Clean up URL
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  const handleLoginSuccess = (token: string) => {
    saveToken(token);
    setAuthToken(token);
    setLoginModalOpen(false);
    // Force reload to ensure everything syncs or just navigate?
    // Navigation will happen automatically if we are on landing page and now have token?
    // Actually, LandingPage is at "/", Dashboard is at "/dashboard".
    // We need to navigate to dashboard.
    window.location.href = '/dashboard';
  };

  const handleLogout = () => {
    removeToken();
    localStorage.removeItem('cariaChatHistory');
    setAuthToken(null);
    window.location.href = '/';
  };

  return (
    <div className="min-h-screen text-[var(--color-text-primary)] font-sans antialiased" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={
            authToken ? <Navigate to="/dashboard" replace /> : (
              <>
                <LandingPage onLogin={handleShowLogin} onRegister={handleShowRegister} />
                {isLoginModalOpen && (
                  <LoginModal
                    onClose={() => setLoginModalOpen(false)}
                    onSuccess={handleLoginSuccess}
                    onSwitchToRegister={handleShowRegister}
                  />
                )}
                {isRegisterModalOpen && (
                  <RegisterModal
                    onClose={() => setRegisterModalOpen(false)}
                    onSuccess={handleLoginSuccess}
                    onSwitchToLogin={handleShowLogin}
                  />
                )}
              </>
            )
          } />

          {/* Dashboard Routes - accessible without login, but widgets may require auth */}
          <Route element={<DashboardLayout onLogout={handleLogout} />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/community" element={<CommunityPage />} />
            <Route path="/resources" element={<ResourcesPage />} />
            {/* Fallback for unknown routes */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Routes>
      </Router>
    </div>
  );
};

export default App;
