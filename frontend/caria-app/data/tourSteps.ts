
export interface TourStep {
  elementId: string;
  title: string;
  content: string;
  position?: 'bottom' | 'top' | 'left' | 'right';
}

/**
 * Onboarding Tour Steps - Optimized for 4.5 minute target (Chime benchmark).
 * Per audit document (4.2): Progressive disclosure, concise content.
 */
export const tourSteps: TourStep[] = [
  {
    elementId: 'market-bar-widget',
    title: 'Market Overview',
    content: 'Real-time global market indices. Updated continuously.',
    position: 'bottom',
  },
  {
    elementId: 'portfolio-widget',
    title: 'Portfolio',
    content: 'Track performance, allocation, and trends. Add holdings to get started.',
    position: 'right',
  },
  {
    elementId: 'analysis-cta-widget',
    title: 'AI Analysis',
    content: 'Test investment ideas. Uncover biases and strengthen your thesis.',
    position: 'left',
  },
];
