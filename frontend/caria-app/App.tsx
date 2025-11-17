import React, { useState, useEffect } from 'react';
import { LandingPage } from './components/LandingPage';
import { AnalysisTool } from './components/AnalysisTool';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { LoginModal } from './components/LoginModal';
import { RegisterModal } from './components/RegisterModal';
import { getToken, saveToken, removeToken } from './services/apiService';

const App: React.FC = () => {
  const [authToken, setAuthToken] = useState<string | null>(getToken());
  const [isLoginModalOpen, setLoginModalOpen] = useState(false);
  const [isRegisterModalOpen, setRegisterModalOpen] = useState(false);
  const [isAnalysisOpen, setAnalysisOpen] = useState(false);
  
  // Effect to check for token on initial load
  useEffect(() => {
    const token = getToken();
    if (token) {
      setAuthToken(token);
    }
  }, []);

  const handleShowLogin = () => {
    setLoginModalOpen(true);
    setRegisterModalOpen(false);
  };

  const handleShowRegister = () => {
    setRegisterModalOpen(true);
    setLoginModalOpen(false);
  };
  
  const handleLoginSuccess = (token: string) => {
    saveToken(token);
    setAuthToken(token);
    setLoginModalOpen(false);
  };

  const handleLogout = () => {
    removeToken();
    localStorage.removeItem('cariaChatHistory');
    setAuthToken(null);
    setAnalysisOpen(false); // Close modal on logout
  };

  return (
    <div className="min-h-screen text-[var(--color-text-primary)] font-sans antialiased" style={{backgroundColor: 'var(--color-bg-primary)'}}>
      {authToken ? (
        <div className="flex h-screen w-full">
          <Sidebar onLogout={handleLogout} />
          <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />
          {isAnalysisOpen && <AnalysisTool onClose={() => setAnalysisOpen(false)} />}
        </div>
      ) : (
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
      )}
    </div>
  );
};

export default App;
