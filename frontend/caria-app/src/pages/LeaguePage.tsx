import React, { useState, useEffect } from 'react';
import Leaderboard from '../../components/league/Leaderboard';
import LeagueProfile from '../../components/league/LeagueProfile';
import { JoinLeague } from '../../components/league/JoinLeague';
import { API_BASE_URL, getToken } from '../../services/apiService';

const LeaguePage: React.FC = () => {
    const [hasJoined, setHasJoined] = useState<boolean | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        checkParticipationStatus();
    }, []);

    const checkParticipationStatus = async () => {
        setLoading(true);
        try {
            const token = getToken();
            const headers: HeadersInit = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = `Bearer ${token}`;

            const response = await fetch(`${API_BASE_URL}/api/league/participation-status`, { headers });
            if (response.ok) {
                const data = await response.json();
                setHasJoined(data.has_joined);
            } else {
                setHasJoined(false);
            }
        } catch (error) {
            console.error('Error checking participation status:', error);
            setHasJoined(false);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-black text-white p-8 flex items-center justify-center">
                <div className="text-white/50">Loading...</div>
            </div>
        );
    }

    if (!hasJoined) {
        return (
            <div className="min-h-screen bg-black text-white p-8">
                <div className="max-w-7xl mx-auto">
                    <JoinLeague onJoinSuccess={() => setHasJoined(true)} />
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-black text-white p-8">
            <div className="max-w-7xl mx-auto space-y-8">

                {/* Header */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-yellow-400 to-orange-500">
                            Caria Global League
                        </h1>
                        <p className="text-white/50 mt-1">
                            Compete with the best investors. Prove your discipline.
                        </p>
                    </div>

                </div>

                {/* Ranking Criteria Explanation */}
                <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                        <span className="text-blue-400">ℹ️</span> How Rankings Work
                    </h3>
                    <p className="text-sm text-white/70 mb-4">
                        The Caria Rank Score is a composite metric designed to reward long-term, risk-adjusted performance rather than short-term luck.
                        We prioritize consistency and discipline over volatility.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 text-sm">
                        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                            <div className="text-green-400 font-bold mb-1">35% Sharpe Ratio</div>
                            <div className="text-xs text-white/50">Risk-adjusted return. Rewards high returns with low volatility.</div>
                        </div>
                        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                            <div className="text-blue-400 font-bold mb-1">30% CAGR</div>
                            <div className="text-xs text-white/50">Compound Annual Growth Rate. Pure performance velocity.</div>
                        </div>
                        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                            <div className="text-red-400 font-bold mb-1">15% Max Drawdown</div>
                            <div className="text-xs text-white/50">Capital preservation. Penalizes deep losses and poor risk management.</div>
                        </div>
                        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                            <div className="text-purple-400 font-bold mb-1">10% Diversification</div>
                            <div className="text-xs text-white/50">Portfolio health. Penalizes extreme concentration.</div>
                        </div>
                        <div className="bg-black/20 p-3 rounded-lg border border-white/5">
                            <div className="text-yellow-400 font-bold mb-1">10% Account Age</div>
                            <div className="text-xs text-white/50">Longevity. Rewards time in the market and experience.</div>
                        </div>
                    </div>
                </div>

                {/* User Stats Summary */}
                <LeagueProfile initialHasJoined={true} />

                {/* Main Leaderboard */}
                <Leaderboard />

                {/* Footer / Motivation */}
                <div className="text-center py-12 text-white/30 text-sm">
                    "The stock market is a device for transferring money from the impatient to the patient." — Warren Buffett
                </div>
            </div>
        </div>
    );
};

export default LeaguePage;
