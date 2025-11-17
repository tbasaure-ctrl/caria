
import React from 'react';

export const WidgetCard: React.FC<{ title: string; children: React.ReactNode; id?: string; className?: string }> = ({ title, children, id, className = '' }) => (
    <div id={id}
         className={`rounded-lg p-5 transition-all duration-300 ${className}`}
         style={{
           backgroundColor: 'var(--color-bg-secondary)',
           border: '1px solid var(--color-bg-tertiary)',
           boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
         }}>
        <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-bold tracking-widest uppercase"
                style={{
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-text-muted)',
                  letterSpacing: '0.1em'
                }}>
                {title}
            </h3>
            {/* Decorative accent */}
            <div className="h-px w-8" style={{backgroundColor: 'var(--color-primary)'}}></div>
        </div>
        <div>{children}</div>
    </div>
);
