import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
    errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
        errorInfo: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error, errorInfo: null };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
        this.setState({ errorInfo });
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white p-4">
                    <div className="max-w-xl w-full bg-gray-800 rounded-lg shadow-xl p-8 border border-red-500/30">
                        <h1 className="text-2xl font-bold text-red-400 mb-4">Something went wrong</h1>
                        <p className="mb-4 text-gray-300">The application encountered an unexpected error.</p>

                        {this.state.error && (
                            <div className="mb-4 p-4 bg-black/50 rounded overflow-auto max-h-40">
                                <p className="font-mono text-sm text-red-300">{this.state.error.toString()}</p>
                            </div>
                        )}

                        {this.state.errorInfo && (
                            <details className="mb-4">
                                <summary className="cursor-pointer text-blue-400 hover:text-blue-300">Component Stack</summary>
                                <pre className="mt-2 p-4 bg-black/50 rounded overflow-auto max-h-60 text-xs text-gray-400">
                                    {this.state.errorInfo.componentStack}
                                </pre>
                            </details>
                        )}

                        <button
                            onClick={() => window.location.reload()}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
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
