import os
import torch
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from .env file."""
    try:
        load_dotenv()
        print("Loaded environment variables from .env file")
    except ImportError:
        print("python-dotenv not installed. Install with: pip install python-dotenv")
        print("Or set environment variables manually")

def setup_api_clients():
    """Initialize API clients and authentication."""
    # HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(hf_token, add_to_git_credential=True)
            print("HuggingFace authenticated")
        except Exception as e:
            print(f"HuggingFace authentication failed: {e}")
    
    # OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            print("OpenAI authenticated")
            return openai_client
        except Exception as e:
            print(f"OpenAI authentication failed: {e}")
    
    # Google Gemini
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            print("Google Gemini authenticated")
        except Exception as e:
            print(f"Google Gemini authentication failed: {e}")

def get_device():
    """Get the device to use for torch."""
    return "cuda" if torch.cuda.is_available() else "cpu"
