import React, { useEffect, useState } from 'react';
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
      <div className="w-full rounded-xl p-4 bg-slate-900/50 border border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <Droplets className="h-5 w-5 text-cyan-400" />
          <h3 className="text-base font-bold text-white">Hydraulic Core</h3>
        </div>
        <div className="text-sm text-gray-400">Loading...</div>
      </div>
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

  return (
    <div className="w-full rounded-xl p-4 bg-slate-900/50 border border-white/10 backdrop-blur-sm">
      {/* Header */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <Droplets className="h-5 w-5 text-cyan-400" />
          <h3 className="text-base font-bold text-white">Hydraulic Core</h3>
        </div>
        <p className="text-xs text-gray-400">Fed Liquidity Gatekeeper</p>
      </div>

      {/* Score Gauge */}
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-300">Hydraulic Score</span>
          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold ${getStateColor(status.state)} text-white`}>
            {getStateIcon(status.state)}
            <span>{status.state}</span>
          </span>
        </div>
        <div className="relative h-8 bg-gray-700/50 rounded-full overflow-hidden">
          <div
            className={`absolute h-full transition-all duration-500 ${status.score >= 60 ? 'bg-green-500' :
                status.score <= 40 ? 'bg-red-500' : 'bg-yellow-500'
              }`}
            style={{ width: `${status.score}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm font-bold text-white">
              {status.score}/100
            </span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 text-sm mb-3">
        <div>
          <div className="text-gray-400 text-xs">Net Liquidity</div>
          <div className="font-semibold text-white">${(status.net_liquidity_bn / 1000).toFixed(2)}T</div>
        </div>
        <div>
          <div className="text-gray-400 text-xs">Yield Curve</div>
          <div className="font-semibold text-white">{status.yield_curve_10y2y?.toFixed(2)}%</div>
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-xs text-gray-500">
        Updated: {new Date(status.last_updated).toLocaleString()}
      </div>
    </div>
  );
}
