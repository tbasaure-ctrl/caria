
import React, { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { tourSteps, TourStep } from '../data/tourSteps';
import { uxTrackingService } from '../services/uxTrackingService';

const TOUR_STORAGE_KEY = 'caria-onboarding-complete';
const TARGET_ONBOARDING_SECONDS = 270; // 4.5 minutes per audit document (4.2)

const calculatePopoverPosition = (rect: DOMRect, step: TourStep): React.CSSProperties => {
    const baseStyle: React.CSSProperties = {
        position: 'absolute',
        transition: 'opacity 0.3s ease, transform 0.3s ease',
    };

    switch (step.position) {
        case 'left':
            return { ...baseStyle, top: rect.top, left: rect.left - 320, transform: `translateY(${(rect.height / 2) - 50}px)` };
        case 'right':
             return { ...baseStyle, top: rect.top, left: rect.right + 20, transform: `translateY(${(rect.height / 2) - 50}px)` };
        case 'top':
            return { ...baseStyle, top: rect.top - 160, left: rect.left };
        case 'bottom':
        default:
            return { ...baseStyle, top: rect.bottom + 20, left: rect.left };
    }
};


export const OnboardingTour: React.FC = () => {
    const [isTourActive, setTourActive] = useState(false);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
    const tourStartTime = useRef<number | null>(null);
    const stepStartTime = useRef<number | null>(null);
    const totalClicks = useRef<number>(0);

    useEffect(() => {
        const isTourCompleted = localStorage.getItem(TOUR_STORAGE_KEY);
        if (!isTourCompleted) {
            setTourActive(true);
            tourStartTime.current = Date.now();
            stepStartTime.current = Date.now();
            uxTrackingService.startTask('onboarding_complete');
        }
    }, []);

    useLayoutEffect(() => {
        if (!isTourActive) return;

        const step = tourSteps[currentStepIndex];
        const element = document.getElementById(step.elementId);
        
        const updatePosition = () => {
            if (element) {
                setTargetRect(element.getBoundingClientRect());
            } else {
                // If element is not found, maybe advance or end tour.
                handleNext();
            }
        };

        updatePosition();
        
        window.addEventListener('resize', updatePosition);
        return () => window.removeEventListener('resize', updatePosition);

    }, [currentStepIndex, isTourActive]);

    const handleNext = async () => {
        totalClicks.current += 1;
        uxTrackingService.incrementClicks('onboarding_complete');

        // Track current step completion
        if (stepStartTime.current) {
            const stepSeconds = (Date.now() - stepStartTime.current) / 1000;
            const step = tourSteps[currentStepIndex];
            await uxTrackingService.trackOnboardingStep(
                step.elementId,
                stepSeconds,
                1 // 1 click to advance
            );
        }

        if (currentStepIndex < tourSteps.length - 1) {
            setCurrentStepIndex(prev => prev + 1);
            stepStartTime.current = Date.now(); // Reset step timer
        } else {
            await finishTour();
        }
    };

    const finishTour = async () => {
        // Track complete onboarding
        if (tourStartTime.current) {
            const totalSeconds = (Date.now() - tourStartTime.current) / 1000;
            await uxTrackingService.completeTask('onboarding_complete', {
                total_steps: tourSteps.length,
                total_seconds: totalSeconds,
                total_clicks: totalClicks.current,
                target_met: totalSeconds <= TARGET_ONBOARDING_SECONDS,
                target_seconds: TARGET_ONBOARDING_SECONDS,
            });
        }

        localStorage.setItem(TOUR_STORAGE_KEY, 'true');
        setTargetRect(null); // Hide highlight
        setTourActive(false);
    };

    if (!isTourActive || !targetRect) return null;

    const currentStep = tourSteps[currentStepIndex];
    const popoverStyle = calculatePopoverPosition(targetRect, currentStep);

    return (
        <div className="fixed inset-0 z-[1000]">
            {/* Highlight Box */}
            <div
                className="fixed border-2 border-blue-500 rounded-lg shadow-[0_0_0_9999px_rgba(15,23,42,0.8)] z-[1001] transition-all duration-300 pointer-events-none"
                style={{
                    left: targetRect.left - 6,
                    top: targetRect.top - 6,
                    width: targetRect.width + 12,
                    height: targetRect.height + 12,
                }}
            />

            {/* Popover */}
            <div 
                style={popoverStyle}
                className="w-72 bg-gray-900 border border-slate-700 p-5 rounded-lg shadow-2xl z-[1002] modal-fade-in"
                role="dialog"
                aria-labelledby="tour-title"
                aria-describedby="tour-content"
            >
                <h3 id="tour-title" className="text-lg font-bold text-[#E0E1DD] mb-2">{currentStep.title}</h3>
                <p id="tour-content" className="text-slate-300 text-sm mb-4">{currentStep.content}</p>
                
                <div className="flex justify-between items-center">
                    <button onClick={finishTour} className="text-xs text-slate-500 hover:text-white transition-colors">Skip Tour</button>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400">{currentStepIndex + 1} / {tourSteps.length}</span>
                        <button 
                            onClick={handleNext} 
                            className="bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:bg-slate-600 transition-all text-sm"
                        >
                            {currentStepIndex === tourSteps.length - 1 ? 'Finish' : 'Next'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
