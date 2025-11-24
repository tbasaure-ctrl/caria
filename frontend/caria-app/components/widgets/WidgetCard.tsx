
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
      className={`rounded-2xl p-6 md:p-7 transition-all duration-500 relative overflow-hidden group ${className}`}
      style={{
        backgroundColor: 'rgba(28, 33, 39, 0.6)',
        border: '1px solid rgba(74, 144, 226, 0.15)',
        backdropFilter: 'blur(10px)',
        boxShadow: isHovered
          ? '0 20px 60px rgba(74, 144, 226, 0.25), 0 0 0 1px rgba(74, 144, 226, 0.3)'
          : '0 8px 32px rgba(0, 0, 0, 0.25)',
        transform: isHovered ? 'translateY(-4px)' : 'translateY(0)',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Animated gradient background on hover */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity duration-500"
        style={{
          background: 'radial-gradient(circle at top right, var(--color-blue-light), transparent 70%)',
        }}
      />

      {/* Widget header */}
      <div className="flex items-center justify-between mb-6 relative z-10">
        <div className="flex items-center gap-3">
          <h3
            className="text-xs md:text-sm font-bold tracking-widest uppercase"
            style={{
              fontFamily: 'var(--font-mono)',
              color: 'rgba(74, 144, 226, 0.9)',
              letterSpacing: '0.15em',
            }}
          >
            {title}
          </h3>
          {tooltip && (
            <div className="relative">
              <button
                className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 hover:scale-110 cursor-help"
                style={{
                  backgroundColor: 'rgba(74, 144, 226, 0.2)',
                  color: 'var(--color-blue-light)',
                  border: '1px solid rgba(74, 144, 226, 0.4)',
                }}
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                aria-label="Information"
              >
                i
              </button>
              {showTooltip && (
                <div
                  className="fixed z-[9999] w-80 p-5 rounded-xl shadow-2xl animate-fadeIn"
                  style={{
                    backgroundColor: 'rgba(11, 14, 17, 0.98)',
                    border: '2px solid var(--color-blue-light)',
                    color: 'var(--color-text-primary)',
                    fontSize: '0.9rem',
                    lineHeight: '1.6',
                    maxWidth: '90vw',
                    left: '50%',
                    top: '50%',
                    transform: 'translate(-50%, -50%)',
                    boxShadow: '0 25px 70px rgba(74, 144, 226, 0.3)',
                    backdropFilter: 'blur(20px)',
                  }}
                >
                  <div className="flex justify-between items-start mb-3">
                    <strong
                      style={{
                        color: 'var(--color-blue-light)',
                        fontSize: '0.75rem',
                        textTransform: 'uppercase',
                        letterSpacing: '0.1em',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      Information
                    </strong>
                    <button
                      onClick={() => setShowTooltip(false)}
                      className="text-xl font-bold hover:text-red-400 transition-colors"
                      style={{ color: 'var(--color-text-secondary)' }}
                    >
                      ×
                    </button>
                  </div>
                  <p style={{ fontFamily: "'Crimson Pro', Georgia, serif" }}>{tooltip}</p>
                </div>
              )}
            </div>
          )}
        </div>
        {/* Decorative accent line with animation */}
        <div
          className="h-[2px] transition-all duration-500"
          style={{
            width: isHovered ? '60px' : '40px',
            background: 'linear-gradient(90deg, var(--color-blue-light), transparent)',
          }}
        ></div>
      </div>

      {/* Widget content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
};
