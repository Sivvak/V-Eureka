import os

from google import generativeai as genai


def configure_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")

    genai.configure(api_key=api_key)
