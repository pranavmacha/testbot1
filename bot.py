"""
Ollama-Powered Fake News Detection Bot
Uses local LLM via Ollama API to analyze news articles.
"""
import requests
import json
import re

# Configuration
OLLAMA_URL = "http://ollama:11434"
OLLAMA_MODEL = "llama3.2"  # Change to your preferred model: mistral, gemma, etc.


def call_ollama(prompt: str) -> str:
    """Call the Ollama API and return the response text."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "num_predict": 150   # Limit response length for speed
                }
            },
            timeout=25  # Leave buffer for the 30s competition timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


def parse_response(response: str) -> dict:
    """Parse the LLM response to extract is_fake and confidence."""
    response_lower = response.lower()
    
    # Look for explicit FAKE or REAL markers
    if "verdict: fake" in response_lower or "classification: fake" in response_lower:
        is_fake = True
    elif "verdict: real" in response_lower or "classification: real" in response_lower:
        is_fake = False
    # Fallback to keyword detection
    elif "this is fake" in response_lower or "appears to be fake" in response_lower:
        is_fake = True
    elif "this is real" in response_lower or "appears to be real" in response_lower or "legitimate" in response_lower:
        is_fake = False
    else:
        # Count fake vs real indicators
        fake_count = response_lower.count("fake") + response_lower.count("misinformation") + response_lower.count("false")
        real_count = response_lower.count("real") + response_lower.count("legitimate") + response_lower.count("credible")
        is_fake = fake_count > real_count
    
    # Try to extract confidence from response
    confidence = 0.75  # Default confidence
    
    # Look for percentage patterns
    confidence_match = re.search(r'(\d{1,3})%?\s*(confident|confidence|certain)', response_lower)
    if confidence_match:
        confidence = min(int(confidence_match.group(1)) / 100, 1.0)
    elif "highly likely" in response_lower or "very confident" in response_lower:
        confidence = 0.9
    elif "likely" in response_lower or "probably" in response_lower:
        confidence = 0.75
    elif "possibly" in response_lower or "might be" in response_lower:
        confidence = 0.6
    elif "uncertain" in response_lower:
        confidence = 0.5
    
    return {"is_fake": is_fake, "confidence": confidence}


def analyze(article_title: str, article_content: str) -> dict:
    """
    Analyze a news article using Ollama LLM.
    
    Args:
        article_title: The headline of the article
        article_content: The full text content
    
    Returns:
        dict with is_fake (bool) and confidence (float 0-1)
    """
    
    # Build a focused prompt for fake news detection
    prompt = f"""You are a fake news detection expert. Analyze this news article and determine if it's FAKE or REAL.

TITLE: {article_title}

CONTENT: {article_content[:1500]}

Analyze for:
1. Sensationalist language (shocking, unbelievable, miracle)
2. Lack of credible sources
3. Emotional manipulation
4. Logical inconsistencies
5. Implausible claims

Respond with:
- VERDICT: FAKE or REAL
- CONFIDENCE: percentage (e.g., 85%)
- REASON: one sentence explanation

Keep your response brief and focused."""

    # Call Ollama
    response = call_ollama(prompt)
    
    if not response:
        # Fallback to simple heuristics if Ollama fails
        return fallback_analyze(article_title, article_content)
    
    # Parse the LLM response
    result = parse_response(response)
    return result


def fallback_analyze(article_title: str, article_content: str) -> dict:
    """Fallback analysis when Ollama is unavailable."""
    fake_keywords = ['shocking', 'unbelievable', 'miracle', 'secret', 'conspiracy']
    text = (article_title + ' ' + article_content).lower()
    
    score = sum(1 for k in fake_keywords if k in text)
    is_fake = score >= 2
    
    return {"is_fake": is_fake, "confidence": min(0.5 + score * 0.1, 0.9)}


# Testing
if __name__ == "__main__":
    # Test with sample articles
    
    print("=" * 50)
    print("Testing Ollama Fake News Bot")
    print("=" * 50)
    
    # Test 1: Obvious fake news
    fake_title = "SHOCKING: Scientists Discover Miracle Cure They Don't Want You To Know!"
    fake_content = """
    In an unbelievable revelation, a secret group of researchers has discovered a miracle 
    cure for all diseases. The conspiracy to hide this from the public has been exposed. 
    You won't believe what happens next! Share before it gets deleted!
    """
    
    print("\n[TEST 1] Obvious Fake:")
    print(f"Title: {fake_title}")
    result = analyze(fake_title, fake_content)
    print(f"Result: {result}")
    print(f"Prediction: {'FAKE ❌' if result['is_fake'] else 'REAL ✓'}")
    
    # Test 2: Likely real news
    real_title = "Federal Reserve Announces Interest Rate Decision"
    real_content = """
    The Federal Reserve announced today that it will maintain current interest rates, 
    citing stable inflation data. Fed Chair Jerome Powell stated that the committee 
    will continue to monitor economic indicators. Markets responded with modest gains.
    """
    
    print("\n[TEST 2] Likely Real:")
    print(f"Title: {real_title}")
    result = analyze(real_title, real_content)
    print(f"Result: {result}")
    print(f"Prediction: {'FAKE ❌' if result['is_fake'] else 'REAL ✓'}")
