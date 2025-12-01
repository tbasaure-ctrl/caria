import React, { useState, useEffect } from 'react';
import { Trophy, TrendingUp, Shield, Activity, Users } from 'lucide-react';

interface LeagueEntry {
    rank: number;
    user_id: string;
    username: string;
    score: number;
    sharpe_ratio: number | null;
    cagr: number | null;
    max_drawdown: number | null;
    diversification_score: number | null;
    account_age_days: number | null;
}

const Leaderboard: React.FC = () => {
    const [entries, setEntries] = useState<LeagueEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<'global' | 'friends'>('global');

    useEffect(() => {
        fetchLeaderboard();
    }, [filter]);

    const fetchLeaderboard = async () => {
        setLoading(true);
        try {
            // In a real app, use the API URL from config
            const response = await fetch('http://localhost:8000/api/league/leaderboard?limit=50');
            if (response.ok) {
                const data = await response.json();
                setEntries(data);
            } else {
                console.error('Failed to fetch leaderboard');
                setEntries([]);
            }
        } catch (error) {
            console.error('Error fetching leaderboard:', error);
            setEntries([]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-[#0A0A0A] border border-white/10 rounded-xl overflow-hidden">
            <div className="p-6 border-b border-white/10 flex justify-between items-center">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Trophy className="w-5 h-5 text-yellow-500" />
                        Global League
                    </h2>
                    <p className="text-white/50 text-sm mt-1">Top investors ranked by risk-adjusted performance</p>
                </div>

                <div className="flex bg-white/5 rounded-lg p-1">
                    <button
                        onClick={() => setFilter('global')}
                        className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${filter === 'global' ? 'bg-white/10 text-white' : 'text-white/50 hover:text-white'
                            }`}
                    >
                        Global
                    </button>
                    <button
                        onClick={() => setFilter('friends')}
                        className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${filter === 'friends' ? 'bg-white/10 text-white' : 'text-white/50 hover:text-white'
                            }`}
                    >
                        Friends
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr className="bg-white/5 text-left">
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider w-16">Rank</th>
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider">Investor</th>
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider text-right">Score</th>
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider text-right">Sharpe</th>
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider text-right">CAGR</th>
                            <th className="py-4 px-6 text-xs font-medium text-white/50 uppercase tracking-wider text-right">Drawdown</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {loading ? (
                            [...Array(5)].map((_, i) => (
                                <tr key={i} className="animate-pulse">
                                    <td className="py-4 px-6"><div className="h-4 w-8 bg-white/10 rounded"></div></td>
                                    <td className="py-4 px-6"><div className="h-4 w-32 bg-white/10 rounded"></div></td>
                                    <td className="py-4 px-6"><div className="h-4 w-16 bg-white/10 rounded ml-auto"></div></td>
                                    <td className="py-4 px-6"><div className="h-4 w-12 bg-white/10 rounded ml-auto"></div></td>
                                    <td className="py-4 px-6"><div className="h-4 w-12 bg-white/10 rounded ml-auto"></div></td>
                                    <td className="py-4 px-6"><div className="h-4 w-12 bg-white/10 rounded ml-auto"></div></td>
                                </tr>
                            ))
                        ) : entries.length === 0 ? (
                            <tr>
                                <td colSpan={6} className="py-12 text-center text-white/30">
                                    <div className="flex flex-col items-center gap-3">
                                        <Users className="w-12 h-12 opacity-20" />
                                        <p>No rankings available yet.</p>
                                        <p className="text-xs">Rankings are updated daily based on portfolio performance.</p>
                                    </div>
                                </td>
                            </tr>
                        ) : (
                            entries.map((entry) => (
                                <tr key={entry.user_id} className="hover:bg-white/5 transition-colors group">
                                    <td className="py-4 px-6">
                                        <div className={`
                                            w-8 h-8 flex items-center justify-center rounded-full font-bold text-sm
                                            ${entry.rank === 1 ? 'bg-yellow-500/20 text-yellow-500' :
                                                entry.rank === 2 ? 'bg-gray-400/20 text-gray-400' :
                                                    entry.rank === 3 ? 'bg-orange-700/20 text-orange-700' :
                                                        'text-white/50'}
                                        `}>
                                            {entry.rank}
                                        </div>
                                    </td>
                                    <td className="py-4 px-6">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-xs font-bold text-white">
                                                {entry.username.substring(0, 2).toUpperCase()}
                                            </div>
                                            <span className="text-white font-medium group-hover:text-blue-400 transition-colors">
                                                {entry.username}
                                            </span>
                                            {entry.rank <= 10 && (
                                                <Shield className="w-3 h-3 text-yellow-500 fill-yellow-500/20" />
                                            )}
                                        </div>
                                    </td>
                                    <td className="py-4 px-6 text-right">
                                        <span className="text-white font-bold font-mono text-lg">
                                            {entry.score.toFixed(0)}
                                        </span>
                                    </td>
                                    <td className="py-4 px-6 text-right">
                                        <span className={`font-mono ${getSharpeColor(entry.sharpe_ratio)}`}>
                                            {entry.sharpe_ratio?.toFixed(2) || '-'}
                                        </span>
                                    </td>
                                    <td className="py-4 px-6 text-right">
                                        <span className="text-green-400 font-mono">
                                            {entry.cagr ? `${(entry.cagr * 100).toFixed(1)}%` : '-'}
                                        </span>
                                    </td>
                                    <td className="py-4 px-6 text-right">
                                        <span className="text-red-400 font-mono">
                                            {entry.max_drawdown ? `-${(entry.max_drawdown * 100).toFixed(1)}%` : '-'}
                                        </span>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

const getSharpeColor = (sharpe: number | null) => {
    if (!sharpe) return 'text-white/30';
    if (sharpe >= 2) return 'text-green-400';
    if (sharpe >= 1) return 'text-yellow-400';
    return 'text-red-400';
};

export default Leaderboard;
