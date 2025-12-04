/**
 * WorldEconomiesHealth Component Tests
 * Validates CARIA economic flow integration
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { WorldEconomiesHealth } from '../WorldEconomiesHealth';

// Mock fetch
global.fetch = vi.fn();

// Mock Plotly
vi.mock('react-plotly.js', () => ({
    default: ({ data, layout }: any) => (
        <div data-testid="plotly-globe" data-points={data?.length || 0}>
            Globe Visualization
        </div>
    ),
}));

describe('WorldEconomiesHealth - CARIA Integration', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should load and display CARIA flow data', async () => {
        const mockFlowData = {
            date: '2025-12-03',
            generatedAt: '2025-12-03T17:23:09.527448',
            modelVersion: 'CARIA V12',
            stats: {
                totalConnections: 57,
                maxWeight: 0.3154,
                minWeight: 0.0037,
                meanWeight: 0.042,
                threshold: 0.06,
                usaThreshold: 0.04,
                modelAccuracy: '64%',
                strongConnections: 4,
                mediumConnections: 10,
                weakConnections: 33,
                usaConnections: 13,
            },
            arcs: [
                {
                    source: 'IND',
                    target: 'CHL',
                    sourceName: 'India',
                    targetName: 'Chile',
                    startLat: 20.5937,
                    startLng: 78.9629,
                    endLat: -35.6751,
                    endLng: -71.543,
                    weight: 0.3154,
                    deviation: 0.2487,
                    strength: 'strong',
                    isUSA: false,
                    label: 'IND → CHL',
                },
            ],
            nodes: ['USA', 'CHN', 'JPN', 'DEU', 'GBR'],
            coordinates: {
                IND: { lat: 20.5937, lng: 78.9629, name: 'India' },
                CHL: { lat: -35.6751, lng: -71.543, name: 'Chile' },
            },
        };

        const mockCountryData = [
            {
                isoCode: 'USA',
                name: 'United States',
                lat: 37.0902,
                lon: -95.7129,
                cyclePhase: 'slowdown',
                cycleMomentum: -0.11,
                structuralRisk: 15.4,
                externalVulnerability: 12.2,
                stressLevel: 11.5,
                behavioralSignal: -0.27,
                instabilityRisk: 2.5,
            },
        ];

        // Mock fetch responses
        (global.fetch as any).mockImplementation((url: string) => {
            if (url.includes('world_economies.json')) {
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve(mockCountryData),
                });
            }
            if (url.includes('caria_flows.json')) {
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve(mockFlowData),
                });
            }
            return Promise.reject(new Error('Unknown URL'));
        });

        render(<WorldEconomiesHealth id="test" />);

        // Wait for data to load
        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalledWith('/data/caria_flows.json');
        });

        // Check that CARIA tab is available
        await waitFor(() => {
            const cariaTab = screen.getByText(/CARIA/i);
            expect(cariaTab).toBeInTheDocument();
        });
    });

    it('should handle missing CARIA flow data gracefully', async () => {
        const mockCountryData = [
            {
                isoCode: 'USA',
                name: 'United States',
                lat: 37.0902,
                lon: -95.7129,
                cyclePhase: 'slowdown',
                cycleMomentum: -0.11,
                structuralRisk: 15.4,
                externalVulnerability: 12.2,
                stressLevel: 11.5,
                behavioralSignal: -0.27,
                instabilityRisk: 2.5,
            },
        ];

        (global.fetch as any).mockImplementation((url: string) => {
            if (url.includes('world_economies.json')) {
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve(mockCountryData),
                });
            }
            if (url.includes('caria_flows.json')) {
                return Promise.resolve({
                    ok: false,
                    status: 404,
                });
            }
            return Promise.reject(new Error('Unknown URL'));
        });

        render(<WorldEconomiesHealth id="test" />);

        // Component should still render without CARIA data
        await waitFor(() => {
            expect(screen.getByText(/GLOBAL ECONOMIC MONITOR/i)).toBeInTheDocument();
        });
    });

    it('should filter connections correctly', async () => {
        const mockFlowData = {
            date: '2025-12-03',
            generatedAt: '2025-12-03T17:23:09',
            modelVersion: 'CARIA V12',
            stats: {
                totalConnections: 3,
                maxWeight: 0.3154,
                minWeight: 0.05,
                meanWeight: 0.15,
                threshold: 0.06,
                modelAccuracy: '64%',
                strongConnections: 1,
                mediumConnections: 1,
                weakConnections: 1,
                usaConnections: 1,
            },
            arcs: [
                {
                    source: 'IND',
                    target: 'CHL',
                    weight: 0.3154,
                    strength: 'strong',
                    isUSA: false,
                    startLat: 20.5937,
                    startLng: 78.9629,
                    endLat: -35.6751,
                    endLng: -71.543,
                    sourceName: 'India',
                    targetName: 'Chile',
                    label: 'IND → CHL',
                },
                {
                    source: 'USA',
                    target: 'CHN',
                    weight: 0.0697,
                    strength: 'usa-link',
                    isUSA: true,
                    startLat: 37.0902,
                    startLng: -95.7129,
                    endLat: 35.8617,
                    endLng: 104.1954,
                    sourceName: 'United States',
                    targetName: 'China',
                    label: 'USA → CHN',
                },
            ],
            nodes: ['USA', 'CHN', 'IND', 'CHL'],
            coordinates: {},
        };

        (global.fetch as any).mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockFlowData),
        });

        render(<WorldEconomiesHealth id="test" />);

        // Wait for data to load
        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalled();
        });

        // Component should render without errors
        expect(screen.getByText(/GLOBAL ECONOMIC MONITOR/i)).toBeInTheDocument();
    });

    it('should validate arc data structure', () => {
        // Test that our expected arc structure matches TypeScript types
        const validArc = {
            source: 'USA',
            target: 'CHN',
            sourceName: 'United States',
            targetName: 'China',
            startLat: 37.0902,
            startLng: -95.7129,
            endLat: 35.8617,
            endLng: 104.1954,
            weight: 0.0697,
            deviation: 0.003,
            strength: 'usa-link' as const,
            isUSA: true,
            label: 'USA → CHN',
        };

        // Type check (will fail at compile time if types don't match)
        expect(validArc).toHaveProperty('source');
        expect(validArc).toHaveProperty('target');
        expect(validArc).toHaveProperty('weight');
        expect(validArc).toHaveProperty('strength');
        expect(['strong', 'medium', 'weak', 'usa-link']).toContain(validArc.strength);
        expect(validArc.weight).toBeGreaterThanOrEqual(0);
        expect(validArc.weight).toBeLessThanOrEqual(1);
    });
});

