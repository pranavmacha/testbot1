# Ollama-Powered Fake News Detection Bot

A local LLM-based bot for Bot Arena that uses Ollama to detect fake news.

## Requirements

1. **Ollama installed** - https://ollama.ai
2. **A model pulled** - Run `ollama pull llama3.2` (or `mistral`, `gemma`, etc.)

## Setup

```bash
# Make sure Ollama is running
ollama serve

# Pull a model
ollama pull llama3.2

# Test the bot
python bot.py
```

## How It Works

1. Takes article title and content
2. Sends to local Ollama model with a prompt
3. Parses LLM response to detect fake/real
4. Returns prediction with confidence

## Configuration

Edit `bot.py` to change:
- `OLLAMA_MODEL` - Model to use (default: llama3.2)
- `OLLAMA_URL` - Ollama API URL (default: http://localhost:11434)
