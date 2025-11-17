/**
 * Ejemplo de Vercel Functions (Serverless Functions)
 * 
 * Si quieres migrar algunos endpoints a Vercel Functions en lugar de Firebase,
 * puedes crear funciones aquí. Vercel automáticamente las detectará.
 * 
 * Estructura:
 * - api/          → Funciones serverless
 *   - challenge.ts → POST /api/challenge
 *   - analyze.ts   → POST /api/analyze
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Ejemplo: Endpoint para challenge thesis
 * Accesible en: https://tu-dominio.vercel.app/api/challenge
 */
export default async function handler(
  req: VercelRequest,
  res: VercelResponse,
) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { thesis, ticker, top_k = 5 } = req.body;

    if (!thesis || thesis.length < 10) {
      return res.status(400).json({ error: 'Tesis debe tener al menos 10 caracteres' });
    }

    // Llamar a Firebase Function o tu backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    
    // Obtener chunks RAG
    const ragResponse = await fetch(`${backendUrl}/api/analysis/wisdom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `${thesis}\nTicker: ${ticker || ''}`,
        top_k,
        page: 1,
        page_size: top_k
      }),
    });

    const ragData = await ragResponse.json();
    const retrievedChunks = ragData.results || [];

    // Llamar a Gemini
    const geminiApiKey = process.env.GEMINI_API_KEY;
    if (!geminiApiKey) {
      return res.status(500).json({ error: 'GEMINI_API_KEY no configurada' });
    }

    const geminiUrl = process.env.GEMINI_API_URL || 
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';

    // Construir prompt con RAG
    const evidenceLines = retrievedChunks.slice(0, 5).map((chunk: any) => {
      const snippet = (chunk.content || '').replace('\n', ' ').substring(0, 400);
      return `- [${chunk.source || 'doc'}] ${chunk.title || ''}: ${snippet}`;
    });
    const evidenceText = evidenceLines.join('\n') || 'No se encontraron fragmentos relevantes.';

    const prompt = `You are CARIA, a rational investment sparring partner.

User:
- Thesis: "${thesis}"
- Ticker: ${ticker || 'N/A'}

Evidence from historical wisdom:
${evidenceText}

Tasks:
1. Critically analyze the thesis
2. Identify cognitive biases
3. Give practical recommendations
4. Estimate confidence_score (0-1)

Respond ONLY in valid JSON:
{
  "critical_analysis": "string",
  "identified_biases": ["string"],
  "recommendations": ["string"],
  "confidence_score": 0.0-1.0
}`;

    const geminiResponse = await fetch(`${geminiUrl}?key=${geminiApiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }]
      }),
    });

    const geminiData = await geminiResponse.json();
    const text = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || '';

    // Parsear respuesta
    let analysisData;
    try {
      analysisData = JSON.parse(text);
    } catch {
      analysisData = {
        critical_analysis: text,
        identified_biases: [],
        recommendations: [],
        confidence_score: 0.6
      };
    }

    return res.status(200).json({
      thesis,
      retrieved_chunks: retrievedChunks,
      critical_analysis: analysisData.critical_analysis || text,
      identified_biases: analysisData.identified_biases || [],
      recommendations: analysisData.recommendations || [],
      confidence_score: analysisData.confidence_score || 0.6,
    });

  } catch (error: any) {
    console.error('Error en challenge:', error);
    return res.status(500).json({ error: error.message || 'Error interno' });
  }
}

