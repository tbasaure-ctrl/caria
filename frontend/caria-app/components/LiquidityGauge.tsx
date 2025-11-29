import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus, Droplets } from 'lucide-react';

interface LiquidityStatus {
  score: number;
  state: string;
  net_liquidity_bn: number;
  yield_curve_10y2y: number;
  last_updated: string;
  trend_roc_4w?: number;
}

export default function LiquidityGauge() {
  const [status, setStatus] = useState<LiquidityStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLiquidityStatus();
    const interval = setInterval(fetchLiquidityStatus, 4 * 60 * 60 * 1000); // 4 hours
    return () => clearInterval(interval);
  }, []);

  const fetchLiquidityStatus = async () => {
    try {
      const response = await fetch('/api/liquidity/status');
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Error fetching liquidity status:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Droplets className="h-5 w-5" />
            Hydraulic Core
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">Loading...</div>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return null;
  }

  const getStateColor = (state: string) => {
    if (state.includes('EXPANSION')) return 'bg-green-500';
    if (state.includes('CONTRACTION')) return 'bg-red-500';
    return 'bg-yellow-500';
  };

  const getStateIcon = (state: string) => {
    if (state.includes('EXPANSION')) return <TrendingUp className="h-4 w-4" />;
    if (state.includes('CONTRACTION')) return <TrendingDown className="h-4 w-4" />;
    return <Minus className="h-4 w-4" />;
  };

  const getScoreGradient = (score: number) => {
    // Red -> Yellow -> Green gradient
    if (score >= 60) {
      const greenIntensity = Math.min(100, ((score - 60) / 40) * 100);
      return `bg-gradient-to-r from-yellow-400 to-green-${Math.floor(greenIntensity / 20) * 100 + 500}`;
    } else if (score <= 40) {
      const redIntensity = Math.min(100, ((40 - score) / 40) * 100);
      return `bg-gradient-to-r from-red-${Math.floor(redIntensity / 20) * 100 + 500} to-yellow-400`;
    } else {
      return 'bg-yellow-400';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Droplets className="h-5 w-5" />
          Hydraulic Core
        </CardTitle>
        <CardDescription>Fed Liquidity Gatekeeper</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Score Gauge */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Hydraulic Score</span>
            <Badge className={getStateColor(status.state)}>
              {getStateIcon(status.state)}
              <span className="ml-1">{status.state}</span>
            </Badge>
          </div>
          <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`absolute h-full transition-all duration-500 ${getScoreGradient(status.score)}`}
              style={{ width: `${status.score}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-sm font-bold text-gray-900 dark:text-white">
                {status.score}/100
              </span>
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-muted-foreground">Net Liquidity</div>
            <div className="font-semibold">${(status.net_liquidity_bn / 1000).toFixed(2)}T</div>
          </div>
          <div>
            <div className="text-muted-foreground">Yield Curve</div>
            <div className="font-semibold">{status.yield_curve_10y2y?.toFixed(2)}%</div>
          </div>
        </div>

        {/* Last Updated */}
        <div className="text-xs text-muted-foreground">
          Updated: {new Date(status.last_updated).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  );
}
