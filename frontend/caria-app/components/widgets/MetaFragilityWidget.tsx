import React, { useState, useEffect } from 'react';

interface FragilityData {
  value: number;
  status: 'NORMAL' | 'WARNING' | 'CRITICAL';
  color: string;
  percentile: number;
  thresholds: {
    warning: number;
    critical: number;
  };
}

interface Signal {
  id: string;
  name: string;
  description: string;
  value: number;
  weight: number;
  is_key: boolean;
}

export const MetaFragilityWidget: React.FC = () => {
  const [fragility, setFragility] = useState<FragilityData | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [fragRes, sigRes] = await Promise.all([
          fetch('/api/fragility/current'),
          fetch('/api/fragility/signals')
        ]);

        if (fragRes.ok) {
          const fragData = await fragRes.json();
          setFragility(fragData);
        }

        if (sigRes.ok) {
          const sigData = await sigRes.json();
          setSignals(sigData.signals || []);
        }
      } catch (error) {
        console.error('Error fetching fragility data:', error);
        // Demo data fallback
        setFragility({
          value: 42,
          status: 'NORMAL',
          color: '#10b981',
          percentile: 65,
          thresholds: { warning: 60, critical: 85 }
        });
        setSignals([
          { id: 'sync', name: 'Global Sync', description: '156d lead', value: 35, weight: 0.59, is_key: true },
          { id: 'cf', name: 'Crisis Factor', description: '88d lead', value: 48, weight: 0.34, is_key: true }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="meta-fragility-widget loading">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!fragility) return null;

  const getStatusIcon = () => {
    switch (fragility.status) {
      case 'CRITICAL': return '🔴';
      case 'WARNING': return '🟡';
      default: return '🟢';
    }
  };

  const getGaugeRotation = () => {
    // Map 0-100 to -90 to 90 degrees
    return -90 + (fragility.value * 1.8);
  };

  return (
    <div className="meta-fragility-widget">
      {/* Header */}
      <div className="widget-header">
        <div className="widget-title">
          <span className="icon">🌋</span>
          <span>Caria Multiscale Fragility Index</span>
        </div>
        <div className="status-badge" style={{ backgroundColor: fragility.color }}>
          {getStatusIcon()} {fragility.status}
        </div>
      </div>

      {/* Main Gauge */}
      <div className="gauge-container">
        <svg viewBox="0 0 200 120" className="gauge">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#2d3748"
            strokeWidth="12"
            strokeLinecap="round"
          />
          {/* Warning zone */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${fragility.value * 2.51} 251`}
          />
          {/* Needle */}
          <g transform={`rotate(${getGaugeRotation()}, 100, 100)`}>
            <line x1="100" y1="100" x2="100" y2="35" stroke="white" strokeWidth="3" />
            <circle cx="100" cy="100" r="8" fill="white" />
          </g>
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="60%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#dc2626" />
            </linearGradient>
          </defs>
        </svg>

        <div className="gauge-value">
          <span className="value">{fragility.value.toFixed(1)}</span>
          <span className="unit">%</span>
        </div>
        <div className="gauge-label">
          {fragility.percentile.toFixed(0)}th percentile
        </div>
      </div>

      {/* Signals breakdown */}
      <div className="signals-section">
        <div
          className="signals-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span>🔬 Key Signals</span>
          <span className="expand-icon">{expanded ? '▲' : '▼'}</span>
        </div>

        {expanded && (
          <div className="signals-list">
            {signals.filter(s => s.is_key).map(signal => (
              <div key={signal.id} className="signal-row">
                <div className="signal-info">
                  <span className="signal-name">{signal.name}</span>
                  <span className="signal-weight">({(signal.weight * 100).toFixed(0)}% weight)</span>
                </div>
                <div className="signal-bar-container">
                  <div
                    className="signal-bar"
                    style={{ width: `${signal.value}%` }}
                  />
                  <span className="signal-value">{signal.value}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Thresholds legend */}
      <div className="thresholds-legend">
        <div className="threshold">
          <span className="dot" style={{ backgroundColor: '#10b981' }}></span>
          <span>Normal &lt;{fragility.thresholds.warning}%</span>
        </div>
        <div className="threshold">
          <span className="dot" style={{ backgroundColor: '#f59e0b' }}></span>
          <span>Warning &lt;{fragility.thresholds.critical}%</span>
        </div>
        <div className="threshold">
          <span className="dot" style={{ backgroundColor: '#dc2626' }}></span>
          <span>Critical ≥{fragility.thresholds.critical}%</span>
        </div>
      </div>

      <style>{`
        .meta-fragility-widget {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border-radius: 16px;
          padding: 20px;
          color: white;
          font-family: 'Inter', sans-serif;
          box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        .widget-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .widget-title {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 18px;
          font-weight: 600;
        }

        .widget-title .icon {
          font-size: 24px;
        }

        .status-badge {
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
        }

        .gauge-container {
          position: relative;
          text-align: center;
          margin: 20px 0;
        }

        .gauge {
          width: 100%;
          max-width: 200px;
        }

        .gauge-value {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          font-size: 32px;
          font-weight: 700;
        }

        .gauge-value .unit {
          font-size: 16px;
          opacity: 0.7;
        }

        .gauge-label {
          color: #a0aec0;
          font-size: 12px;
          margin-top: 8px;
        }

        .signals-section {
          background: rgba(255,255,255,0.05);
          border-radius: 12px;
          overflow: hidden;
        }

        .signals-header {
          padding: 12px 16px;
          display: flex;
          justify-content: space-between;
          cursor: pointer;
          transition: background 0.2s;
        }

        .signals-header:hover {
          background: rgba(255,255,255,0.05);
        }

        .signals-list {
          padding: 0 16px 16px;
        }

        .signal-row {
          margin-bottom: 12px;
        }

        .signal-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: 4px;
        }

        .signal-name {
          font-weight: 500;
        }

        .signal-weight {
          color: #a0aec0;
          font-size: 12px;
        }

        .signal-bar-container {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .signal-bar-container {
          flex: 1;
          height: 8px;
          background: rgba(255,255,255,0.1);
          border-radius: 4px;
          overflow: hidden;
          position: relative;
        }

        .signal-bar {
          height: 100%;
          background: linear-gradient(90deg, #10b981, #f59e0b);
          border-radius: 4px;
          transition: width 0.3s;
        }

        .signal-value {
          position: absolute;
          right: 8px;
          font-size: 11px;
          font-weight: 600;
        }

        .thresholds-legend {
          display: flex;
          justify-content: space-around;
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid rgba(255,255,255,0.1);
        }

        .threshold {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: #a0aec0;
        }

        .threshold .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .loading {
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 200px;
        }

        .loading-spinner {
          width: 32px;
          height: 32px;
          border: 3px solid rgba(255,255,255,0.1);
          border-top-color: #10b981;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default MetaFragilityWidget;
