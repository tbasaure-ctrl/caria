export type CyclePhase = 'expansion' | 'slowdown' | 'recession' | 'recovery';

export interface CountryState {
    isoCode: string;          // ISO-2 or ISO-3
    name: string;
    lat: number;
    lon: number;

    // Core macro layer
    cyclePhase: CyclePhase;   // expansion / slowdown / recession / recovery
    cycleMomentum: number;    // -1 to +1 (decelerating → accelerating)

    // Risk layers (0–100)
    structuralRisk: number;        // debt, credit gap, twin deficits
    externalVulnerability: number; // reserves, FX, terms of trade
    stressLevel: number;           // combined “Global Stress” score

    // Behavior / complexity
    behavioralSignal: number;      // -1 (optimistic) to +1 (fearful)
    instabilityRisk: number;       // 0–100 (probability of regime shift)

    // Optional history for tail trails
    history?: {
        date: string;                // ISO date
        cyclePhase: CyclePhase;
        cycleMomentum: number;
        stressLevel: number;
    }[];

    // Raw metrics for display
    metrics?: {
        gdpGrowth?: number;
        inflation?: number;
        debtToGdp?: number;
        unemployment?: number;
        currencyChange?: number; // % change vs USD 6m
    }
}
