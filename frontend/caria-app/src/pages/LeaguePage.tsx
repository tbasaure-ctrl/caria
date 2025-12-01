import Leaderboard from '../../components/league/Leaderboard';
import LeagueProfile from '../../components/league/LeagueProfile';

const LeaguePage: React.FC = () => {
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

                    <div className="flex items-center gap-2 text-sm text-white/50 bg-white/5 px-4 py-2 rounded-full border border-white/10">
                        <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                        Season 1 Active • Ends Dec 31
                    </div>
                </div>

                {/* User Stats Summary */}
                <LeagueProfile />

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
