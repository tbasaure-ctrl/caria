
import React, { useState } from 'react';

interface WidgetCardProps {
  title: string;
  children: React.ReactNode;
  id?: string;
  className?: string;
  tooltip?: string;
}

export const WidgetCard: React.FC<WidgetCardProps> = ({ title, children, id, className = '', tooltip }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div id={id}
         className={`rounded-lg p-5 transition-all duration-300 ${className}`}
         style={{
           backgroundColor: 'var(--color-bg-secondary)',
           border: '1px solid var(--color-bg-tertiary)',
           boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
         }}>
        <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold tracking-widest uppercase"
                    style={{
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--color-text-muted)',
                      letterSpacing: '0.1em'
                    }}>
                    {title}
                </h3>
                {tooltip && (
                  <div className="relative">
                    <button
                      className="w-4 h-4 rounded-full flex items-center justify-center text-xs transition-all duration-200 hover:scale-110"
                      style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        color: 'var(--color-text-secondary)',
                        border: '1px solid var(--color-text-muted)'
                      }}
                      onMouseEnter={() => setShowTooltip(true)}
                      onMouseLeave={() => setShowTooltip(false)}
                      aria-label="Information"
                    >
                      i
                    </button>
                    {showTooltip && (
                      <div
                        className="absolute left-0 top-6 z-50 w-64 p-3 rounded-lg shadow-lg"
                        style={{
                          backgroundColor: 'var(--color-bg-primary)',
                          border: '1px solid var(--color-primary)',
                          color: 'var(--color-text-primary)',
                          fontSize: '0.875rem',
                          lineHeight: '1.4'
                        }}
                      >
                        {tooltip}
                      </div>
                    )}
                  </div>
                )}
            </div>
            {/* Decorative accent */}
            <div className="h-px w-8" style={{backgroundColor: 'var(--color-primary)'}}></div>
        </div>
        <div>{children}</div>
    </div>
  );
};
