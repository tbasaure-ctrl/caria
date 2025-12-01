import React from 'react';
import { TrendingUp, Activity, Shield, Clock, BarChart3 } from 'lucide-react';

interface LeagueProfileProps {
    userId?: string; // If undefined, show current user
}

const LeagueProfile: React.FC<LeagueProfileProps> = ({ userId }) => {
    const [hasJoined, setHasJoined] = React.useState(false);

    React.useEffect(() => {
        const joined = localStorage.getItem('caria_league_joined');
        if (joined) setHasJoined(true);
    }, []);

    const handleJoin = () => {
        localStorage.setItem('caria_league_joined', 'true');
        setHasJoined(true);
    };

    // Mock data for now
    const stats = {
        rank: 42,
        score: 750,
        percentile: 85,
        sharpe: 1.8,
        cagr: 0.28,
        drawdown: 0.14,
        diversification: 78,
        accountAge: 124,
        bestMonth: 0.12,
        worstMonth: -0.05
    };

    if (!hasJoined) {
        return (
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
                        onClick={handleJoin}
                        className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-8 rounded-lg transition-all transform hover:scale-105 shadow-lg shadow-blue-900/20"
                    >
                        Join League (Anonymous)
                    </button>
                </div>
            </div>
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
                <div className="flex items-baseline gap-2 mb-6">
                    <span className="text-4xl font-bold text-white">#{stats.rank}</span>
                    <span className="text-green-400 text-sm font-medium">Top {100 - stats.percentile}%</span>
                </div>

                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/50">League Score</span>
                            <span className="text-white font-mono">{stats.score}</span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                                style={{ width: `${stats.score / 10}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="col-span-1 md:col-span-2 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <MetricCard
                    label="Risk-Adj Return"
                    value={stats.sharpe.toFixed(2)}
                    subLabel="Sharpe Ratio"
                    icon={<Activity className="w-4 h-4 text-blue-400" />}
                />
                <MetricCard
                    label="Growth"
                    value={`${(stats.cagr * 100).toFixed(1)}%`}
                    subLabel="CAGR"
                    icon={<TrendingUp className="w-4 h-4 text-green-400" />}
                />
                <MetricCard
                    label="Safety"
                    value={`-${(stats.drawdown * 100).toFixed(1)}%`}
                    subLabel="Max Drawdown"
                    icon={<Shield className="w-4 h-4 text-red-400" />}
                />
                <MetricCard
                    label="Experience"
                    value={`${stats.accountAge}d`}
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
