export type CyclePhase = 'expansion' | 'slowdown' | 'recession' | 'recovery';

// CARIA Economic Graph Types
export interface EconomicArc {
    source: string;         // ISO code of influencer
    target: string;         // ISO code of influenced
    sourceName: string;
    targetName: string;
    startLat: number;
    startLng: number;
    endLat: number;
    endLng: number;
    weight: number;         // 0 to 1 (connection strength)
    deviation: number;      // deviation from uniform
    strength: 'strong' | 'medium' | 'weak' | 'usa-link';
    isUSA?: boolean;        // true if connection involves USA
    label: string;
}

export interface EconomicFlowData {
    date: string;
    generatedAt: string;
    modelVersion: string;
    stats: {
        totalConnections: number;
        maxWeight: number;
        minWeight: number;
        meanWeight: number;
        threshold: number;
        usaThreshold?: number;
        modelAccuracy: string;
        strongConnections: number;
        mediumConnections: number;
        weakConnections: number;
        usaConnections?: number;
    };
    arcs: EconomicArc[];
    nodes: string[];
    usaAnalysis?: {
        influencedBy: { country: string; name: string; weight: number }[];
        influences: { country: string; name: string; weight: number }[];
    };
}

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

// Direction Prediction Types
export interface DirectionPrediction {
    country: string;           // ISO code
    countryName: string;       // Full name
    direction: 'UP' | 'DOWN';  // Predicted direction
    confidence: number;        // Confidence score (magnitude)
    prediction: number;        // Raw prediction value
}

export interface DirectionPredictionsData {
    predictionDate?: string;
    modelVersion: string;
    modelAccuracy: string;
    countries: string[];
    countryNames: string[];
    predictions: number[];     // Raw predictions
    directions: ('UP' | 'DOWN')[];
    confidences: number[];     // Absolute values
    normalizedConfidences: number[];
    summary: {
        totalCountries: number;
        upPredictions: number;
        downPredictions: number;
        avgConfidence: number;
        maxConfidence: number;
        minConfidence: number;
    };
}
