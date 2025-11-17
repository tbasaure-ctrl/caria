/**
 * Firebase Functions Service
 * Servicio para llamar a las Firebase Functions serverless
 */

// URLs de las Firebase Functions desplegadas
export const FIREBASE_FUNCTIONS = {
  CHALLENGE_THESIS: 'https://us-central1-caria-9b633.cloudfunctions.net/challenge_thesis',
  ANALYZE_WITH_GEMINI: 'https://us-central1-caria-9b633.cloudfunctions.net/analyze_with_gemini',
} as const;

export interface ChallengeThesisRequest {
  thesis: string;
  ticker?: string;
  top_k?: number;
}

export interface ChallengeThesisResponse {
  thesis: string;
  retrieved_chunks: Array<{
    id: string;
    score: number;
    title?: string;
    source?: string;
    content?: string;
    metadata: Record<string, any>;
  }>;
  critical_analysis: string;
  identified_biases: string[];
  recommendations: string[];
  confidence_score: number;
}

export interface AnalyzeWithGeminiRequest {
  prompt: string;
}

export interface AnalyzeWithGeminiResponse {
  raw_text: string;
}

/**
 * Llama a la Firebase Function challenge_thesis
 * Reemplaza: POST /api/analysis/challenge
 */
export const challengeThesis = async (
  request: ChallengeThesisRequest
): Promise<ChallengeThesisResponse> => {
  const response = await fetch(FIREBASE_FUNCTIONS.CHALLENGE_THESIS, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      thesis: request.thesis,
      ticker: request.ticker,
      top_k: request.top_k || 5,
    }),
  });

  if (!response.ok) {
    let errorMessage = 'Error al procesar la tesis';
    try {
      const errorData = await response.json();
      errorMessage = errorData.error || errorData.detail || errorMessage;
    } catch {
      errorMessage = `HTTP error! status: ${response.status}`;
    }
    throw new Error(errorMessage);
  }

  return response.json();
};

/**
 * Llama a la Firebase Function analyze_with_gemini
 * Para análisis directo con Gemini (sin RAG)
 */
export const analyzeWithGemini = async (
  request: AnalyzeWithGeminiRequest
): Promise<AnalyzeWithGeminiResponse> => {
  const response = await fetch(FIREBASE_FUNCTIONS.ANALYZE_WITH_GEMINI, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: request.prompt,
    }),
  });

  if (!response.ok) {
    let errorMessage = 'Error al analizar con Gemini';
    try {
      const errorData = await response.json();
      errorMessage = errorData.error || errorData.detail || errorMessage;
    } catch {
      errorMessage = `HTTP error! status: ${response.status}`;
    }
    throw new Error(errorMessage);
  }

  return response.json();
};

