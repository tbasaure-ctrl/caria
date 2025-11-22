## Groq LLM Setup

This repo now targets Groq's OpenAI-compatible endpoint for both the chat widget and the thesis challenge flow. To switch to the new `meta-llama/llama-4-scout-17b-16e-instruct` model (with a safety fallback), configure the backend with the following env vars:

```env
# Required
LLAMA_API_KEY=sk_your_groq_key
LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Optional fallback if the primary model returns an empty response or errors
LLAMA_FALLBACK_MODEL=llama-3.1-70b-instruct
```

### Testing from the CLI

```bash
curl https://api.groq.com/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLAMA_API_KEY" \
  -d '{
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
          {"role": "system","content": "You are Caria, an institutional-grade investment mentor."},
          {"role": "user","content": "Summarize the secular thesis for ASML."}
        ],
        "temperature": 0.5
      }'
```

You should receive a JSON payload with `choices[0].message.content`. If not, double-check the key, endpoint, and model string.

### RAG Integration

No additional changes are required for RAG—the `LLMService` already retrieves the vector-store context and forwards it to the selected Groq model. The fallback logic ensures we still return a structured answer even if the primary model is temporarily unavailable.

