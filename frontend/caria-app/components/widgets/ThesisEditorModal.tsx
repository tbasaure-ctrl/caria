/**
 * ThesisEditorModal - Rich text editor for creating/editing community posts.
 * Supports validation and Arena pre-fill.
 */

import React, { useState, useEffect } from 'react';
import { fetchWithAuth, API_BASE_URL } from '../../services/apiService';

interface ThesisEditorModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess?: () => void;
    prefillData?: {
        title?: string;
        thesis_preview?: string;
        full_thesis?: string;
        ticker?: string | null;
        arena_thread_id?: string | null;
        arena_round_id?: string | null;
        arena_community?: string | null;
    };
}

interface ValidationResult {
    is_valid: boolean;
    quality_score: number;
    feedback: string[];
    recommendation: string;
}

export const ThesisEditorModal: React.FC<ThesisEditorModalProps> = ({
    isOpen,
    onClose,
    onSuccess,
    prefillData,
}) => {
    const [title, setTitle] = useState('');
    const [thesisPreview, setThesisPreview] = useState('');
    const [fullThesis, setFullThesis] = useState('');
    const [ticker, setTicker] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showFullEditor, setShowFullEditor] = useState(false);

    useEffect(() => {
        if (isOpen && prefillData) {
            setTitle(prefillData.title || '');
            setThesisPreview(prefillData.thesis_preview || '');
            setFullThesis(prefillData.full_thesis || prefillData.thesis_preview || '');
            setTicker(prefillData.ticker || '');
            setShowFullEditor(!!prefillData.full_thesis);
        } else if (isOpen) {
            // Reset form when opening without prefill
            setTitle('');
            setThesisPreview('');
            setFullThesis('');
            setTicker('');
            setShowFullEditor(false);
        }
        setValidationResult(null);
        setError(null);
    }, [isOpen, prefillData]);

    const handleValidate = async () => {
        if (!title.trim() || !thesisPreview.trim()) {
            setError('Title and preview are required');
            return;
        }

        setIsValidating(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/community/posts/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title.trim(),
                    thesis_preview: thesisPreview.trim(),
                    full_thesis: fullThesis.trim() || thesisPreview.trim(),
                }),
            });

            if (!response.ok) {
                throw new Error('Validation failed');
            }

            const result: ValidationResult = await response.json();
            setValidationResult(result);
        } catch (err: any) {
            console.error('Error validating post:', err);
            setError('Failed to validate post. You can still submit.');
        } finally {
            setIsValidating(false);
        }
    };

    const handleSubmit = async () => {
        if (!title.trim() || !thesisPreview.trim()) {
            setError('Title and preview are required');
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/community/posts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title.trim(),
                    thesis_preview: thesisPreview.trim(),
                    full_thesis: fullThesis.trim() || thesisPreview.trim(),
                    ticker: ticker.trim() || null,
                    arena_thread_id: prefillData?.arena_thread_id || null,
                    arena_round_id: prefillData?.arena_round_id || null,
                    arena_community: prefillData?.arena_community || null,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to create post' }));
                throw new Error(errorData.detail || 'Failed to create post');
            }

            if (typeof window !== 'undefined') {
                window.dispatchEvent(new CustomEvent('caria-community-refresh'));
            }
            if (onSuccess) {
                onSuccess();
            }
            onClose();
        } catch (err: any) {
            console.error('Error creating post:', err);
            setError(err?.message || 'No pudimos publicar tu tesis. Intenta nuevamente.');
        } finally {
            setIsSubmitting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ backgroundColor: 'rgba(0, 0, 0, 0.7)' }}
            onClick={onClose}
        >
            <div
                className="bg-gray-900 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6"
                onClick={(e) => e.stopPropagation()}
                style={{
                    backgroundColor: 'var(--color-bg-primary)',
                    border: '1px solid var(--color-bg-tertiary)',
                }}
            >
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                    <h2
                        className="text-2xl font-bold"
                        style={{
                            fontFamily: 'var(--font-display)',
                            color: 'var(--color-cream)',
                        }}
                    >
                        Share Your Thesis
                    </h2>
                    <button
                        onClick={onClose}
                        className="text-2xl font-bold hover:opacity-70 transition-opacity"
                        style={{ color: 'var(--color-text-secondary)' }}
                    >
                        ×
                    </button>
                </div>

                {/* Form */}
                <div className="space-y-4">
                    {/* Title */}
                    <div>
                        <label
                            htmlFor="title"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Title *
                        </label>
                        <input
                            id="title"
                            type="text"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder="e.g., Why I'm bullish on AAPL"
                            maxLength={255}
                            className="w-full px-3 py-2 rounded-lg text-sm"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        />
                        <div className="text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                            {title.length}/255 characters
                        </div>
                    </div>

                    {/* Ticker */}
                    <div>
                        <label
                            htmlFor="ticker"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Ticker (Optional)
                        </label>
                        <input
                            id="ticker"
                            type="text"
                            value={ticker}
                            onChange={(e) => setTicker(e.target.value.toUpperCase())}
                            placeholder="AAPL"
                            maxLength={10}
                            className="w-full px-3 py-2 rounded-lg text-sm font-mono"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        />
                    </div>

                    {/* Preview */}
                    <div>
                        <label
                            htmlFor="preview"
                            className="block text-sm font-medium mb-2"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Preview (Short Summary) *
                        </label>
                        <textarea
                            id="preview"
                            value={thesisPreview}
                            onChange={(e) => setThesisPreview(e.target.value)}
                            placeholder="A brief summary that will appear in the feed..."
                            maxLength={500}
                            rows={3}
                            className="w-full px-3 py-2 rounded-lg text-sm resize-none"
                            style={{
                                backgroundColor: 'var(--color-bg-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                            }}
                        />
                        <label
                            htmlFor="show-full"
                            className="text-sm cursor-pointer"
                            style={{ color: 'var(--color-text-primary)' }}
                        >
                            Add full thesis (optional)
                        </label>
                    </div>

                    {/* Full Thesis */}
                    {showFullEditor && (
                        <div>
                            <label
                                htmlFor="full-thesis"
                                className="block text-sm font-medium mb-2"
                                style={{ color: 'var(--color-text-primary)' }}
                            >
                                Full Thesis
                            </label>
                            <textarea
                                id="full-thesis"
                                value={fullThesis}
                                onChange={(e) => setFullThesis(e.target.value)}
                                placeholder="Your complete investment thesis..."
                                maxLength={5000}
                                rows={8}
                                className="w-full px-3 py-2 rounded-lg text-sm resize-none"
                                style={{
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    border: '1px solid var(--color-bg-tertiary)',
                                    color: 'var(--color-text-primary)',
                                }}
                            />
                            <div className="text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
                                {fullThesis.length}/5000 characters
                            </div>
                        </div>
                    )}

                    {/* Arena Badge */}
                    {prefillData?.arena_thread_id && (
                        <div
                            className="p-3 rounded-lg flex items-center gap-2"
                            style={{
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                border: '1px solid rgba(59, 130, 246, 0.3)',
                            }}
                        >
                            <span className="text-lg">🏛️</span>
                            <div className="text-sm" style={{ color: 'var(--color-text-primary)' }}>
                                This post is linked to your Thesis Arena thread
                            </div>
                        </div>
                    )}

                    {/* Validation Result */}
                    {validationResult && (
                        <div
                            className={`p-3 rounded-lg ${validationResult.is_valid
                                ? 'bg-green-900/20 border-green-500/30'
                                : 'bg-yellow-900/20 border-yellow-500/30'
                                }`}
                            style={{ border: '1px solid' }}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span
                                    className="text-sm font-semibold"
                                    style={{
                                        color: validationResult.is_valid ? '#10b981' : '#f59e0b',
                                    }}
                                >
                                    Quality Score: {(validationResult.quality_score * 100).toFixed(0)}%
                                </span>
                                <span
                                    className="text-xs px-2 py-1 rounded"
                                    style={{
                                        backgroundColor: validationResult.is_valid
                                            ? 'rgba(16, 185, 129, 0.2)'
                                            : 'rgba(245, 158, 11, 0.2)',
                                        color: validationResult.is_valid ? '#10b981' : '#f59e0b',
                                    }}
                                >
                                    {validationResult.recommendation}
                                </span>
                            </div>
                            {validationResult.feedback.length > 0 && (
                                <ul className="text-xs space-y-1 mt-2" style={{ color: 'var(--color-text-secondary)' }}>
                                    {validationResult.feedback.map((fb, idx) => (
                                        <li key={idx}>• {fb}</li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    )}

                    {/* Error */}
                    {error && (
                        <div
                            className="p-3 rounded-lg"
                            style={{
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                color: '#ef4444',
                            }}
                        >
                            {error}
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-3 pt-4">
                        <button
                            onClick={handleValidate}
                            disabled={isValidating || !title.trim() || !thesisPreview.trim()}
                            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity"
                            style={{
                                backgroundColor: 'var(--color-bg-tertiary)',
                                color: 'var(--color-text-primary)',
                                opacity:
                                    isValidating || !title.trim() || !thesisPreview.trim() ? 0.5 : 1,
                                cursor:
                                    isValidating || !title.trim() || !thesisPreview.trim()
                                        ? 'not-allowed'
                                        : 'pointer',
                            }}
                        >
                            {isValidating ? 'Validating...' : 'Validate Quality'}
                        </button>
                        <div className="flex-1" />
                        <button
                            onClick={onClose}
                            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity"
                            style={{
                                backgroundColor: 'transparent',
                                color: 'var(--color-text-secondary)',
                                border: '1px solid var(--color-bg-tertiary)',
                            }}
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting || !title.trim() || !thesisPreview.trim()}
                            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity"
                            style={{
                                backgroundColor:
                                    isSubmitting || !title.trim() || !thesisPreview.trim()
                                        ? 'var(--color-bg-tertiary)'
                                        : 'var(--color-primary)',
                                color: 'var(--color-cream)',
                                opacity:
                                    isSubmitting || !title.trim() || !thesisPreview.trim() ? 0.5 : 1,
                                cursor:
                                    isSubmitting || !title.trim() || !thesisPreview.trim()
                                        ? 'not-allowed'
                                        : 'pointer',
                            }}
                        >
                            {isSubmitting ? 'Publishing...' : 'Publish to Feed'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

