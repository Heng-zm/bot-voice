import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)
for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or ()
    if not actions or "generateContent" in actions:
        print(model.name)
