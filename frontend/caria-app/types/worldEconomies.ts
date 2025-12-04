export type CyclePhase = 'expansion' | 'slowdown' | 'recession' | 'recovery';

// CARIA V22 Economic Graph Types
export interface EconomicConnection {
    from: string;            // ISO code of influencer
    to: string;              // ISO code of influenced
    weight: number;          // 0 to 1 (connection strength)
    raw: number;             // Raw adjacency score
    strength: 'strong' | 'medium' | 'weak';
    label: string;
}

// Transformed arc with coordinates for rendering
export interface EconomicArc {
    source: string;
    target: string;
    sourceName: string;
    targetName: string;
    startLat: number;
    startLng: number;
    endLat: number;
    endLng: number;
    weight: number;
    raw: number;
    strength: 'strong' | 'medium' | 'weak';
    label: string;
}

export interface CountryCoordinate {
    name: string;
    lat: number;
    lon: number;
}

export interface EconomicFlowData {
    modelVersion: string;
    generatedAt: string;
    directionAccuracy?: number;      // V22: ~59.6%
    directionNote?: string;          // V22: "Discovered as auxiliary signal"
    countries: string[];
    coordinates: Record<string, CountryCoordinate>;
    connections: EconomicConnection[];
    stats: {
        totalCountries: number;
        totalConnections: number;
        avgWeight: number;
        maxWeight: number;
        threshold: number;
    };
    // Computed fields (added after loading)
    arcs?: EconomicArc[];
    usaAnalysis?: {
        influencedBy: { country: string; name: string; weight: number }[];
        influences: { country: string; name: string; weight: number }[];
    };
}

export interface CountryState {
    isoCode: string;
    name: string;
    lat: number;
    lon: number;

    // Core macro layer
    cyclePhase: CyclePhase;
    cycleMomentum: number;

    // Risk layers
    structuralRisk: number;
    externalVulnerability: number;
    stressLevel: number;

    // Behavior / complexity
    behavioralSignal: number;
    instabilityRisk: number;

    // History
    history?: {
        date: string;
        cyclePhase: CyclePhase;
        cycleMomentum: number;
        stressLevel: number;
    }[];

    // Metrics
    metrics?: {
        gdpGrowth?: number;
        inflation?: number;
        debtToGdp?: number;
        unemployment?: number;
        currencyChange?: number;
    }
}

// Direction Prediction Types
export interface DirectionPrediction {
    country: string;
    countryName: string;
    direction: 'UP' | 'DOWN';
    confidence: number;
    prediction: number;
}

export interface DirectionPredictionsData {
    predictionDate?: string;
    modelVersion: string;
    modelAccuracy: string;
    countries: string[];
    countryNames: string[];
    predictions: number[];
    directions: ('UP' | 'DOWN')[];
    confidences: number[];
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
