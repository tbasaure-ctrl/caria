# LLM Provider Setup Guide

This guide explains how to configure the LLM service with Groq (Llama 3.1 70B) as default and Claude 3.5 Sonnet as fallback.

## Overview

The LLM service now supports:
- **Primary**: Llama 3.1 70B via Groq (ultra-low latency, cost-effective)
- **Fallback**: Claude 3.5 Sonnet via Anthropic (excellent for structured JSON)

## Configuration

### 1. Groq Setup (Primary Provider)

1. **Get API Key**:
   - Sign up at https://console.groq.com/
   - Create an API key from the dashboard

2. **Set Environment Variables**:
   ```bash
   LLAMA_API_KEY=your_groq_api_key_here
   LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
   LLAMA_MODEL=llama-3.1-70b-versatile  # Optional: defaults to 70B
   ```

3. **Available Models**:
   - `llama-3.1-70b-versatile` (default, recommended)
   - `llama-3.1-70b-instruct`
   - `llama-3-70b-8192`
   - `llama-3.1-8b-instruct` (smaller, faster)

### 2. Claude Setup (Fallback Provider)

1. **Get API Key**:
   - Sign up at https://console.anthropic.com/
   - Create an API key

2. **Set Environment Variable**:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## How It Works

### Default Behavior
- **Normal prompts**: Uses Groq (Llama 3.1 70B) for ultra-low latency
- **Structured JSON prompts**: Automatically detects and prefers Claude 3.5 Sonnet
- **Fallback**: If Groq fails, automatically falls back to Claude

### Automatic Detection
The service automatically detects structured JSON prompts by looking for keywords:
- "json", "structured", "format", "schema", "parse", "extract"

### Manual Override
You can force Claude usage:
```python
# Force Claude for structured output
response = llm_service.call_llm(prompt, system_prompt, use_fallback=True)

# Or use the JSON-optimized method
response = llm_service.call_llm_with_json_fallback(prompt, system_prompt)
```

## Railway/Production Setup

Add these environment variables in your Railway project:

1. Go to your Railway project → Variables
2. Add:
   - `LLAMA_API_KEY` = your Groq API key
   - `ANTHROPIC_API_KEY` = your Anthropic API key (optional but recommended)
   - `LLAMA_MODEL` = `llama-3.1-70b-versatile` (optional)

## Testing

Test the setup:
```python
from api.services.llm_service import LLMService

llm = LLMService()

# Test Groq (default)
response = llm.call_llm("What is 2+2?")
print(f"Groq response: {response}")

# Test Claude fallback for JSON
json_prompt = "Extract the following data as JSON: Name: John, Age: 30"
response = llm.call_llm_with_json_fallback(json_prompt)
print(f"Claude response: {response}")
```

## Cost Considerations

- **Groq**: Very affordable, pay-per-use pricing
- **Claude**: More expensive but excellent for structured outputs
- **Fallback**: Only used when Groq fails or for JSON prompts, minimizing costs

## Troubleshooting

### Groq not working
- Check `LLAMA_API_KEY` is set correctly
- Verify API key has credits/quota
- Check logs for specific error messages

### Claude not working
- Check `ANTHROPIC_API_KEY` is set
- Verify API key is valid
- Claude is optional - service will work with just Groq

### Model not found
- Verify model name matches Groq's available models
- Check Groq console for latest model names
- Default model should work without configuration
