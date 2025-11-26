
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
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      id={id}
      className={`rounded-xl p-6 transition-all duration-300 ${className}`}
      style={{
        backgroundColor: 'rgba(19, 23, 28, 0.8)',
        border: '1px solid rgba(74, 144, 226, 0.15)',
        boxShadow: isHovered 
          ? '0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(74, 144, 226, 0.2)' 
          : '0 4px 16px rgba(0,0,0,0.2)',
        backdropFilter: 'blur(10px)',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >

      {/* Widget header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3
            className="text-xs font-bold tracking-widest uppercase"
            style={{
              fontFamily: 'var(--font-mono)',
              color: 'var(--color-text-muted)',
              letterSpacing: '0.1em',
            }}
          >
            {title}
          </h3>
          {tooltip && (
            <div className="relative group">
              <button
                className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-200 hover:scale-110 cursor-help"
                style={{
                  backgroundColor: 'var(--color-primary)',
                  color: 'var(--color-cream)',
                  border: '1px solid var(--color-primary)',
                }}
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                aria-label="Information"
              >
                i
              </button>
              {showTooltip && (
                <div
                  className="fixed z-[9999] w-80 p-4 rounded-lg shadow-2xl animate-fadeIn"
                  style={{
                    backgroundColor: 'var(--color-bg-primary)',
                    border: '2px solid var(--color-primary)',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.875rem',
                    lineHeight: '1.5',
                    maxWidth: '90vw',
                    left: '50%',
                    top: '50%',
                    transform: 'translate(-50%, -50%)',
                    boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
                  }}
                >
                  <div className="flex justify-between items-start mb-2">
                    <strong
                      style={{
                        color: 'var(--color-primary)',
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                      }}
                    >
                      Information
                    </strong>
                    <button
                      onClick={() => setShowTooltip(false)}
                      className="text-sm font-bold"
                      style={{ color: 'var(--color-text-secondary)' }}
                    >
                      ×
                    </button>
                  </div>
                  {tooltip}
                </div>
              )}
            </div>
          )}
        </div>
        {/* Decorative accent */}
        <div className="h-px w-8" style={{ backgroundColor: 'var(--color-primary)' }}></div>
      </div>

      {/* Widget content */}
      <div>{children}</div>
    </div>
  );
};
