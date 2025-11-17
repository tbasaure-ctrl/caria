import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-20"
            style={{
              backgroundColor: 'var(--color-bg-primary)',
              borderTop: '1px solid var(--color-bg-tertiary)'
            }}>
      <div className="container mx-auto px-6 py-6 text-center"
           style={{
             fontFamily: 'var(--font-body)',
             color: 'var(--color-text-muted)'
           }}>
        <p>&copy; {new Date().getFullYear()} Caria. All rights reserved.</p>
      </div>
    </footer>
  );
};
