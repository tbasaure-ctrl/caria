import React from 'react';

export interface Feature {
  icon: React.FC<React.SVGProps<SVGSVGElement>>;
  title: string;
  description: string;
  visual?: React.ReactNode;
}

export interface ChatMessage {
    role: 'user' | 'model' | 'error';
    content: string;
}
