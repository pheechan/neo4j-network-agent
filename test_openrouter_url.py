"""Test script to verify OpenRouter configuration"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check configuration
print("=" * 60)
print("OpenRouter Configuration Check")
print("=" * 60)

openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
openrouter_api_base = os.getenv("OPENROUTER_API_BASE")
openai_api_base = os.getenv("OPENAI_API_BASE")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

print(f"OPENROUTER_BASE_URL: {openrouter_base_url}")
print(f"OPENROUTER_API_BASE: {openrouter_api_base}")
print(f"OPENAI_API_BASE: {openai_api_base}")
print(f"OPENROUTER_API_KEY: {'***' + openrouter_key[-10:] if openrouter_key else 'NOT SET'}")
print("=" * 60)

# Determine which URL will be used
if openrouter_base_url:
    url_to_use = openrouter_base_url
elif openrouter_api_base:
    url_to_use = openrouter_api_base
elif openai_api_base:
    url_to_use = openai_api_base
else:
    url_to_use = "https://openrouter.ai/api/v1"  # default

print(f"\n✅ URL that will be used: {url_to_use}")

# Check if it's the correct format
if "api.openrouter.ai" in url_to_use:
    print("❌ ERROR: Using old URL format 'api.openrouter.ai'")
    print("   Should be: 'openrouter.ai/api/v1'")
elif "openrouter.ai" in url_to_use:
    print("✅ Correct URL format!")
else:
    print("⚠️  Unknown URL format")

print("=" * 60)
