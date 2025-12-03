import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Analytics } from '@vercel/analytics/react';
import { LandingPage } from './components/LandingPage';
import { LoginModal } from './components/LoginModal';
import { RegisterModal } from './components/RegisterModal';
import { DashboardPage } from './components/pages/DashboardPage';
import { PortfolioPage } from './components/pages/PortfolioPage';
import { AnalysisPage } from './components/pages/AnalysisPage';
import { CommunityPage } from './components/pages/CommunityPage';
import { ResourcesPage } from './components/pages/ResourcesPage';
import { EconomicHealthPage } from './components/pages/EconomicHealthPage';
import { GitHubLayout } from './components/layout/GitHubLayout';
import { getToken, saveToken, removeToken } from './services/apiService';
import LeaguePage from './src/pages/LeaguePage';

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
    window.location.href = '/portfolio';
  };

  return (
    <div className="min-h-screen text-[var(--color-text-primary)] font-sans antialiased bg-[var(--color-bg-primary)]">
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={
            authToken ? <Navigate to="/portfolio" replace /> : (
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



          {/* GitHub Layout Routes - Replaces DashboardLayout */}
          <Route element={<GitHubLayout />}>
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/world-economies" element={<EconomicHealthPage />} />
            <Route path="/league" element={<LeaguePage />} />
            <Route path="/research" element={<CommunityPage />} /> {/* Mapping Research to Community for now, or can create dedicated */}
            <Route path="/about" element={<ResourcesPage />} />

            {/* Fallback redirects */}
            <Route path="/dashboard" element={<Navigate to="/portfolio" replace />} />
            <Route path="*" element={<Navigate to="/portfolio" replace />} />
          </Route>
        </Routes>
      </Router>
      <Analytics />
    </div>
  );
};

export default App;
