/**
 * UX Tracking Service per audit document (4.2).
 * Tracks user journeys: clicks and seconds per task.
 */

import { fetchWithAuth } from './apiService';
import { API_BASE_URL } from './apiConfig';

export interface TaskTracking {
    task_name: string;
    clicks: number;
    seconds: number;
    metadata?: Record<string, any>;
}

class UXTrackingService {
    private taskStartTime: Map<string, number> = new Map();
    private taskClicks: Map<string, number> = new Map();

    /**
     * Start tracking a task.
     */
    startTask(taskName: string): void {
        this.taskStartTime.set(taskName, Date.now());
        this.taskClicks.set(taskName, 0);
    }

    /**
     * Increment click count for a task.
     */
    incrementClicks(taskName: string): void {
        const current = this.taskClicks.get(taskName) || 0;
        this.taskClicks.set(taskName, current + 1);
    }

    /**
     * Complete and track a task.
     */
    async completeTask(taskName: string, metadata?: Record<string, any>): Promise<void> {
        const startTime = this.taskStartTime.get(taskName);
        if (!startTime) {
            console.warn(`Task ${taskName} was not started`);
            return;
        }

        const seconds = (Date.now() - startTime) / 1000;
        const clicks = this.taskClicks.get(taskName) || 0;

        try {
            await fetchWithAuth(`${API_BASE_URL}/api/ux/track`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_name: taskName,
                    clicks,
                    seconds,
                    metadata,
                }),
            });
        } catch (error) {
            console.error('Error tracking task:', error);
            // Don't throw - tracking failures shouldn't break the app
        } finally {
            // Clean up
            this.taskStartTime.delete(taskName);
            this.taskClicks.delete(taskName);
        }
    }

    /**
     * Track onboarding completion.
     * Per audit document (4.2): Target 4.5 minutes (270 seconds).
     */
    async trackOnboardingStep(stepName: string, stepSeconds: number, stepClicks: number): Promise<void> {
        await this.completeTask(`onboarding_${stepName}`, {
            step_name: stepName,
            step_seconds: stepSeconds,
            step_clicks: stepClicks,
        });
    }
}

export const uxTrackingService = new UXTrackingService();

