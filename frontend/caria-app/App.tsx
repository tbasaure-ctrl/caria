import React, { useState, useEffect, Component, ErrorInfo, ReactNode } from 'react';
import { LandingPage } from './components/LandingPage';
import { AnalysisTool } from './components/AnalysisTool';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { LoginModal } from './components/LoginModal';
import { RegisterModal } from './components/RegisterModal';
import { getToken, saveToken, removeToken } from './services/apiService';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center" style={{backgroundColor: 'var(--color-bg-primary)'}}>
          <div className="text-center p-8">
            <h1 className="text-2xl font-bold mb-4" style={{color: 'var(--color-cream)'}}>Something went wrong</h1>
            <p className="mb-4" style={{color: 'var(--color-text-secondary)'}}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-4 py-2 rounded"
              style={{
                backgroundColor: 'var(--color-primary)',
                color: 'var(--color-cream)'
              }}
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

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
    <ErrorBoundary>
      <div className="min-h-screen text-[var(--color-text-primary)] font-sans antialiased" style={{backgroundColor: 'var(--color-bg-primary)'}}>
        {authToken ? (
          <div className="flex h-screen w-full">
            <ErrorBoundary>
              <Sidebar onLogout={handleLogout} />
            </ErrorBoundary>
            <ErrorBoundary>
              <Dashboard onStartAnalysis={() => setAnalysisOpen(true)} />
            </ErrorBoundary>
            {isAnalysisOpen && (
              <ErrorBoundary>
                <AnalysisTool onClose={() => setAnalysisOpen(false)} />
              </ErrorBoundary>
            )}
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
    </ErrorBoundary>
  );
};

export default App;
