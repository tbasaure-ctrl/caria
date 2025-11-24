import React, { Component, ErrorInfo, ReactNode } from 'react';

interface SafeWidgetProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface SafeWidgetState {
  hasError: boolean;
}

/**
 * SafeWidget - Wraps a widget to catch rendering errors without breaking the entire app.
 * This prevents hooks violations by ensuring components always render consistently.
 */
export class SafeWidget extends Component<SafeWidgetProps, SafeWidgetState> {
  constructor(props: SafeWidgetProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): SafeWidgetState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Widget rendering error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 rounded-lg" style={{ 
          backgroundColor: 'var(--color-bg-secondary)', 
          border: '1px solid var(--color-bg-tertiary)',
          minHeight: '100px'
        }}>
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            Widget unavailable
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
