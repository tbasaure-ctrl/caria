import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, Shield, Clock, BarChart3 } from 'lucide-react';
import { fetchWithAuth, API_BASE_URL, getToken } from '../../services/apiService';

interface LeagueProfileProps {
    userId?: string; // If undefined, show current user
    initialHasJoined?: boolean;
}

const LeagueProfile: React.FC<LeagueProfileProps> = ({ userId, initialHasJoined = false }) => {
    const [hasJoined, setHasJoined] = useState(initialHasJoined);
    const [loading, setLoading] = useState(!initialHasJoined); // If we know they joined, we can skip initial loading check or just load stats
    const [showUsernameModal, setShowUsernameModal] = useState(false);
    const [usernameChoice, setUsernameChoice] = useState<'current' | 'new'>('current');
    const [newUsername, setNewUsername] = useState('');
    const [currentUsername, setCurrentUsername] = useState('');
    const [leagueUsername, setLeagueUsername] = useState('');
    const [stats, setStats] = useState<any>(null);

    const [currentUserId, setCurrentUserId] = useState<string | undefined>(userId);

    useEffect(() => {
        const checkJoinStatus = async () => {
            try {
                const token = getToken();
                if (!token) {
                    setLoading(false);
                    return;
                }

                // Check if user has joined
                // If initialHasJoined is true, we trust it. Otherwise check localStorage.
                const joined = initialHasJoined || localStorage.getItem('caria_league_joined');

                // Always fetch user info to get ID and username
                const userResp = await fetchWithAuth(`${API_BASE_URL}/api/auth/me`);
                let fetchedUserId = userId;

                if (userResp.ok) {
                    const userData = await userResp.json();
                    setCurrentUsername(userData.username || '');
                    setCurrentUserId(userData.id);
                    fetchedUserId = userData.id;

                    if (joined) {
                        setLeagueUsername(localStorage.getItem('caria_league_username') || userData.username || '');
                    }
                }

                if (joined) {
                    setHasJoined(true);
                    // Fetch real stats now that we have the ID
                    if (fetchedUserId) await fetchLeagueStats(fetchedUserId);
                }
            } catch (err) {
                console.error('Error checking league status:', err);
            } finally {
                setLoading(false);
            }
        };
        checkJoinStatus();
    }, [userId, initialHasJoined]);

    const fetchLeagueStats = async (uid?: string) => {
        const targetId = uid || currentUserId || userId;
        if (!targetId) return;

        try {
            const resp = await fetchWithAuth(`${API_BASE_URL}/api/league/profile/${targetId}`);
            if (resp.ok) {
                const data = await resp.json();
                if (data.current) {
                    setStats(data.current);
                } else {
                    // User joined but no stats yet (calculation pending)
                    setStats({
                        rank: null,
                        score: null,
                        percentile: null,
                        sharpe: null,
                        cagr: null,
                        drawdown: null,
                        diversification: null,
                        accountAge: null,
                    });
                }
            }
        } catch (err) {
            console.error('Error fetching league stats:', err);
        }
    };

    const handleJoinClick = () => {
        setShowUsernameModal(true);
    };

    const handleJoinConfirm = async () => {
        try {
            const finalUsername = usernameChoice === 'current'
                ? currentUsername
                : newUsername.trim();

            if (!finalUsername) {
                alert('Please enter a username');
                return;
            }

            // Store join status and username
            localStorage.setItem('caria_league_joined', 'true');
            localStorage.setItem('caria_league_username', finalUsername);
            setLeagueUsername(finalUsername);
            setHasJoined(true);
            setShowUsernameModal(false);

            // TODO: Call backend API to register user in league
            // await fetchWithAuth(`${API_BASE_URL}/api/league/join`, {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({ display_username: finalUsername }),
            // });

            // Fetch stats after joining
            await fetchLeagueStats();
        } catch (err) {
            console.error('Error joining league:', err);
            alert('Failed to join league. Please try again.');
        }
    };

    if (loading) {
        return (
            <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-white/10 rounded-xl p-8 text-center mb-8">
                <div className="text-white/50">Loading...</div>
            </div>
        );
    }

    if (!hasJoined) {
        return (
            <>
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-white/10 rounded-xl p-8 text-center mb-8">
                    <div className="max-w-2xl mx-auto">
                        <Shield className="w-16 h-16 text-blue-400 mx-auto mb-6 opacity-80" />
                        <h2 className="text-2xl font-bold text-white mb-4">Join the Global Portfolio League</h2>
                        <p className="text-white/70 mb-8 leading-relaxed">
                            Test your investment discipline against the best.
                            <br />
                            <span className="text-white/50 text-sm block mt-4 bg-black/20 p-4 rounded-lg border border-white/5">
                                <span className="text-blue-400 font-bold block mb-1">🔒 Privacy Guarantee</span>
                                Joining the league does <strong>NOT</strong> publicly disclose your holdings, portfolio value, or personal identity.
                                We only display your <strong>Rank Score</strong> and an anonymous username (or alias).
                                Your financial data remains strictly private and encrypted.
                            </span>
                        </p>
                        <button
                            onClick={handleJoinClick}
                            className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-8 rounded-lg transition-all transform hover:scale-105 shadow-lg shadow-blue-900/20"
                        >
                            Join League
                        </button>
                    </div>
                </div>

                {/* Username Selection Modal */}
                {showUsernameModal && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fade-in">
                        <div className="w-full max-w-md bg-[#050A14] border border-accent-gold/30 rounded-xl shadow-2xl p-6">
                            <h3 className="text-xl font-display text-white mb-4">Choose Your Display Name</h3>
                            <p className="text-sm text-text-secondary mb-6">
                                This username will be displayed in the rankings. Your privacy is protected.
                            </p>

                            <div className="space-y-4 mb-6">
                                <label className="flex items-center gap-3 p-4 rounded-lg border cursor-pointer hover:border-accent-cyan/50 transition-colors"
                                    style={{
                                        borderColor: usernameChoice === 'current' ? 'var(--color-accent-cyan)' : 'var(--color-bg-tertiary)',
                                        backgroundColor: usernameChoice === 'current' ? 'rgba(6, 182, 212, 0.1)' : 'transparent'
                                    }}>
                                    <input
                                        type="radio"
                                        name="username"
                                        value="current"
                                        checked={usernameChoice === 'current'}
                                        onChange={() => setUsernameChoice('current')}
                                        className="w-4 h-4"
                                    />
                                    <div className="flex-1">
                                        <div className="text-white font-medium">Join with your current username</div>
                                        <div className="text-xs text-text-muted mt-1">
                                            {currentUsername || 'Loading...'}
                                        </div>
                                    </div>
                                </label>

                                <label className="flex items-center gap-3 p-4 rounded-lg border cursor-pointer hover:border-accent-cyan/50 transition-colors"
                                    style={{
                                        borderColor: usernameChoice === 'new' ? 'var(--color-accent-cyan)' : 'var(--color-bg-tertiary)',
                                        backgroundColor: usernameChoice === 'new' ? 'rgba(6, 182, 212, 0.1)' : 'transparent'
                                    }}>
                                    <input
                                        type="radio"
                                        name="username"
                                        value="new"
                                        checked={usernameChoice === 'new'}
                                        onChange={() => setUsernameChoice('new')}
                                        className="w-4 h-4"
                                    />
                                    <div className="flex-1">
                                        <div className="text-white font-medium">Join with a new username</div>
                                        {usernameChoice === 'new' && (
                                            <input
                                                type="text"
                                                value={newUsername}
                                                onChange={(e) => setNewUsername(e.target.value)}
                                                placeholder="Enter display name"
                                                className="w-full mt-2 px-3 py-2 rounded bg-bg-tertiary border border-white/10 text-white placeholder-text-muted focus:border-accent-cyan focus:outline-none"
                                            />
                                        )}
                                    </div>
                                </label>
                            </div>

                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowUsernameModal(false)}
                                    className="flex-1 px-4 py-2 rounded-lg border border-white/10 text-white hover:bg-white/5 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleJoinConfirm}
                                    disabled={usernameChoice === 'new' && !newUsername.trim()}
                                    className="flex-1 px-4 py-2 rounded-lg bg-accent-primary text-black font-bold hover:bg-accent-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Join League
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 animate-fade-in">
            {/* Main Score Card */}
            <div className="col-span-1 md:col-span-1 bg-gradient-to-br from-blue-900/40 to-purple-900/40 border border-white/10 rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10">
                    <TrophyIcon className="w-32 h-32" />
                </div>

                <h3 className="text-white/70 font-medium mb-2">Current Rank</h3>
                <div className="flex items-baseline gap-2 mb-4">
                    {stats?.rank ? (
                        <>
                            <span className="text-4xl font-bold text-white">#{stats.rank}</span>
                            {stats.percentile && (
                                <span className="text-green-400 text-sm font-medium">Top {100 - stats.percentile}%</span>
                            )}
                        </>
                    ) : (
                        <span className="text-white/50 text-sm">Calculating rank...</span>
                    )}
                </div>

                {leagueUsername && (
                    <div className="text-xs text-white/50 mb-4">
                        Display name: <span className="text-white font-mono">{leagueUsername}</span>
                    </div>
                )}

                <div className="space-y-4">
                    {stats?.score ? (
                        <>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-white/50">League Score</span>
                                    <span className="text-white font-mono">{stats.score}</span>
                                </div>
                                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                                        style={{ width: `${Math.min((stats.score / 10), 100)}%` }}
                                    />
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="text-xs text-white/50">Rank score will appear after your first portfolio analysis</div>
                    )}
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="col-span-1 md:col-span-2 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <MetricCard
                    label="Risk-Adj Return"
                    value={stats?.sharpe ? stats.sharpe.toFixed(2) : '--'}
                    subLabel="Sharpe Ratio"
                    icon={<Activity className="w-4 h-4 text-blue-400" />}
                />
                <MetricCard
                    label="Growth"
                    value={stats?.cagr ? `${(stats.cagr * 100).toFixed(1)}%` : '--'}
                    subLabel="CAGR"
                    icon={<TrendingUp className="w-4 h-4 text-green-400" />}
                />
                <MetricCard
                    label="Safety"
                    value={stats?.drawdown ? `-${(stats.drawdown * 100).toFixed(1)}%` : '--'}
                    subLabel="Max Drawdown"
                    icon={<Shield className="w-4 h-4 text-red-400" />}
                />
                <MetricCard
                    label="Experience"
                    value={stats?.accountAge ? `${stats.accountAge}d` : '--'}
                    subLabel="Active Days"
                    icon={<Clock className="w-4 h-4 text-yellow-400" />}
                />
            </div>
        </div>
    );
};

const MetricCard = ({ label, value, subLabel, icon }: any) => (
    <div className="bg-[#0A0A0A] border border-white/10 rounded-xl p-4 flex flex-col justify-between hover:border-white/20 transition-colors">
        <div className="flex justify-between items-start mb-2">
            <span className="text-white/50 text-xs font-medium uppercase">{label}</span>
            {icon}
        </div>
        <div>
            <div className="text-xl font-bold text-white font-mono">{value}</div>
            <div className="text-white/30 text-xs mt-1">{subLabel}</div>
        </div>
    </div>
);

const TrophyIcon = ({ className }: { className?: string }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        className={className}
    >
        <path fillRule="evenodd" d="M5.166 2.621v.858c-1.035.148-2.059.33-3.071.543a.75.75 0 00-.584.859 6.753 6.753 0 006.138 5.6 6.73 6.73 0 002.743 1.346A6.707 6.707 0 019.279 15H8.54c-1.036 0-1.875.84-1.875 1.875V19.5h-.75a2.25 2.25 0 00-2.25 2.25c0 .414.336.75.75.75h14.25c.414 0 .75-.336.75-.75a2.25 2.25 0 00-2.25-2.25h-.75v-2.625c0-1.036-.84-1.875-1.875-1.875h-.739a6.706 6.706 0 01-1.612-3.13 6.73 6.73 0 002.743-1.347 6.753 6.753 0 006.139-5.6.75.75 0 00-.585-.858 47.077 47.077 0 00-3.07-.543V2.62a.75.75 0 00-.658-.744 49.22 49.22 0 00-6.093-.377c-2.063 0-4.096.128-6.093.377a.75.75 0 00-.657.744zm0 2.629c0 1.196.312 2.32.857 3.294A5.266 5.266 0 013.16 5.337a45.6 45.6 0 012.006-.348zm13.668 8.031a5.255 5.255 0 01-5.545-3.094 5.241 5.241 0 002.929-2.544 5.254 5.254 0 012.616 5.638zM15.75 9a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" clipRule="evenodd" />
    </svg>
);

export default LeagueProfile;
