import React from 'react';
import { IndustryResearch } from '../widgets/IndustryResearch';
import { WeeklyMedia } from '../widgets/WeeklyMedia';
import { OpportunityRadar } from '../widgets/OpportunityRadar';
import { CommunityFeed } from '../widgets/CommunityFeed';
import { Resources } from '../widgets/Resources';
import { RankingsWidget } from '../widgets/RankingsWidget';
import { ProtectedWidget } from '../ProtectedWidget';
import { RedditSentiment } from '../widgets/RedditSentiment';

export const CommunityPage: React.FC = () => {
    return (
        <div className="animate-fade-in space-y-8 pb-20">
            {/* Hero Section: Industry & Media */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                <div className="col-span-1 xl:col-span-8 min-h-[600px]">
                    <ProtectedWidget featureName="Industry Research">
                        <IndustryResearch />
                    </ProtectedWidget>
                </div>
                <div className="col-span-1 xl:col-span-4 flex flex-col gap-6">
                    <OpportunityRadar />
                    <WeeklyMedia compact={false} />
                </div>
            </div>

            {/* Social Sentiment & Community */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="flex flex-col gap-6">
                    <div className="h-[400px]">
                        <ProtectedWidget featureName="Social Sentiment">
                            <RedditSentiment />
                        </ProtectedWidget>
                    </div>
                    <div className="h-[400px]">
                        <ProtectedWidget featureName="Community">
                            <CommunityFeed />
                        </ProtectedWidget>
                    </div>
                </div>
                <div className="flex flex-col gap-6">
                    <div className="h-[400px]">
                        <Resources />
                    </div>
                    <div className="h-[400px]">
                        <RankingsWidget />
                    </div>
                </div>
            </div>
        </div>
    );
};
