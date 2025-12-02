import React, { useState } from 'react';
import { Shield, Eye, EyeOff, Trophy } from 'lucide-react';
import { API_BASE_URL, getToken } from '../../services/apiService';

interface JoinLeagueProps {
    onJoinSuccess: () => void;
}

export const JoinLeague: React.FC<JoinLeagueProps> = ({ onJoinSuccess }) => {
    const [isAnonymous, setIsAnonymous] = useState(false);
    const [displayName, setDisplayName] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleJoin = async () => {
        setIsSubmitting(true);
        try {
            const token = getToken();
            const headers: HeadersInit = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = `Bearer ${token}`;

            const response = await fetch(`${API_BASE_URL}/api/league/join`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    is_anonymous: isAnonymous,
                    display_name: isAnonymous && displayName ? displayName : null,
                }),
            });

            if (response.ok) {
                onJoinSuccess();
            } else {
                console.error('Failed to join league');
            }
        } catch (error) {
            console.error('Error joining league:', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-[60vh] flex items-center justify-center px-4">
            <div className="max-w-2xl w-full bg-white/5 border border-white/10 rounded-xl p-8 sm:p-12">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-yellow-500/10 rounded-full mb-4">
                        <Trophy className="w-8 h-8 text-yellow-500" />
                    </div>
                    <h2 className="text-3xl font-bold text-white mb-3">Join the Global League</h2>
                    <p className="text-white/60">
                        Compete with other investors and prove your discipline. Rankings are based on risk-adjusted
                        performance, not just returns.
                    </p>
                </div>

                <div className="space-y-6">
                    {/* Anonymity Toggle */}
                    <div className="bg-black/20 border border-white/5 rounded-lg p-6">
                        <div className="flex items-start gap-4">
                            <div className="flex-shrink-0 mt-1">
                                <button
                                    onClick={() => setIsAnonymous(!isAnonymous)}
                                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${isAnonymous ? 'bg-blue-500' : 'bg-white/10'
                                        }`}
                                >
                                    <span
                                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${isAnonymous ? 'translate-x-6' : 'translate-x-1'
                                            }`}
                                    />
                                </button>
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                    {isAnonymous ? (
                                        <EyeOff className="w-4 h-4 text-blue-400" />
                                    ) : (
                                        <Eye className="w-4 h-4 text-white/50" />
                                    )}
                                    <span className="text-white font-medium">
                                        {isAnonymous ? 'Participate Anonymously' : 'Show My Username'}
                                    </span>
                                </div>
                                <p className="text-sm text-white/50">
                                    {isAnonymous
                                        ? 'Your username will be hidden. You can optionally provide a display name below.'
                                        : 'Your real username will be visible to other league participants.'}
                                </p>
                            </div>
                        </div>

                        {isAnonymous && (
                            <div className="mt-4">
                                <label className="block text-sm text-white/70 mb-2">
                                    Display Name (Optional)
                                </label>
                                <input
                                    type="text"
                                    value={displayName}
                                    onChange={(e) => setDisplayName(e.target.value)}
                                    placeholder="Leave empty for 'Anonymous Investor'"
                                    className="w-full bg-black/30 border border-white/10 rounded px-4 py-2 text-white placeholder-white/30 focus:border-blue-500 focus:outline-none"
                                    maxLength={30}
                                />
                            </div>
                        )}
                    </div>

                    {/* Privacy Notice */}
                    <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                            <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                            <div className="text-sm text-white/70">
                                <strong className="text-white">Privacy Notice:</strong> By joining, your portfolio
                                performance metrics (Sharpe Ratio, CAGR, etc.) will be visible to other participants.
                                Your actual holdings remain private.
                            </div>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-4">
                        <button
                            onClick={handleJoin}
                            disabled={isSubmitting}
                            className="flex-1 bg-yellow-500 hover:bg-yellow-600 disabled:bg-yellow-500/50 text-black font-bold py-3 px-6 rounded-lg transition-colors"
                        >
                            {isSubmitting ? 'Joining...' : 'Join League'}
                        </button>
                    </div>

                    <p className="text-xs text-center text-white/40">
                        Rankings are updated daily at midnight UTC based on your portfolio performance.
                    </p>
                </div>
            </div>
        </div>
    );
};
