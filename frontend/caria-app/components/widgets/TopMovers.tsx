
import React, { useState, useEffect } from 'react';
import { topMovers as initialTopMovers } from '../../data/mockData';
import { WidgetCard } from './WidgetCard';

type Mover = {
    ticker: string;
    name: string;
    price: number;
    change: number;
    positive: boolean;
};

export const TopMovers: React.FC = () => {
    const [movers, setMovers] = useState<Mover[]>(initialTopMovers);

    useEffect(() => {
        const interval = setInterval(() => {
            setMovers(currentMovers => 
                currentMovers.map(mover => {
                    const priceChange = (Math.random() - 0.5) * (mover.price * 0.01);
                    const newPrice = mover.price + priceChange;
                    const changePercent = (priceChange / mover.price) * 100;
                    
                    // In a real app, you'd calculate change based on opening price
                    // Here we just accumulate it for visual effect
                    const newChange = mover.change + changePercent;

                    return {
                        ...mover,
                        price: newPrice,
                        change: newChange,
                        positive: newChange >= 0,
                    };
                })
            );
        }, 3000);

        return () => clearInterval(interval);
    }, []);


    return (
        <WidgetCard title="TOP MOVERS (LIVE SIMULATION)">
            <div className="space-y-3">
                {movers.map((mover, index) => (
                    <div key={mover.ticker}
                         className="flex justify-between items-center p-3 rounded-lg transition-all duration-300"
                         style={{
                           backgroundColor: 'var(--color-bg-tertiary)',
                           border: '1px solid transparent',
                           animationDelay: `${index * 0.1}s`
                         }}
                         onMouseEnter={(e) => {
                           e.currentTarget.style.borderColor = 'var(--color-primary)';
                           e.currentTarget.style.transform = 'translateX(4px)';
                         }}
                         onMouseLeave={(e) => {
                           e.currentTarget.style.borderColor = 'transparent';
                           e.currentTarget.style.transform = 'translateX(0)';
                         }}>
                        <div>
                            <span className="font-bold text-sm"
                                  style={{
                                    fontFamily: 'var(--font-mono)',
                                    color: 'var(--color-cream)'
                                  }}>
                                {mover.ticker}
                            </span>
                            <p className="text-xs truncate max-w-[120px]"
                               style={{
                                 fontFamily: 'var(--font-body)',
                                 color: 'var(--color-text-muted)'
                               }}>
                                {mover.name}
                            </p>
                        </div>
                        <div className="text-right">
                             <span className="font-mono block text-sm mb-1"
                                   style={{
                                     fontFamily: 'var(--font-mono)',
                                     color: 'var(--color-text-secondary)'
                                   }}>
                                ${mover.price.toFixed(2)}
                             </span>
                            <p className="font-semibold text-sm transition-all duration-500"
                               style={{
                                 fontFamily: 'var(--font-mono)',
                                 color: mover.positive ? 'var(--color-accent)' : 'var(--color-primary)'
                               }}>
                                {mover.positive ? '↑' : '↓'} {mover.positive ? '+' : ''}{mover.change.toFixed(2)}%
                            </p>
                        </div>
                    </div>
                ))}
            </div>
        </WidgetCard>
    );
};
