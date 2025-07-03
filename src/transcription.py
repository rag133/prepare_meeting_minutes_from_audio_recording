import os
import google.generativeai as genai
from openai import OpenAI

def transcribe_openai_api(openai_client, audio_file, progress=None):
    """Transcribe using OpenAI Whisper API"""
    if not openai_client:
        raise ValueError("OpenAI API key not configured")
    
    if progress:
        progress(0.1, "Starting OpenAI Whisper transcription...")
    
    try:
        with open(audio_file, "rb") as file:
            if progress:
                progress(0.5, "Uploading audio to OpenAI...")
            
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                response_format="text"
            )
            
            if progress:
                progress(1.0, "OpenAI transcription completed!")
            
            return transcription
    except Exception as e:
        raise Exception(f"OpenAI transcription failed: {str(e)}")

def transcribe_with_gemini(audio_file, progress=None):
    """Transcribe using Google Gemini"""
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError("Google Gemini API key not configured")

    if progress:
        progress(0.1, "Uploading audio to Google...")
    
    try:
        audio_file_uploaded = genai.upload_file(path=audio_file)
        if progress:
            progress(0.5, "Audio uploaded, starting Gemini transcription...")

        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(["Please transcribe this audio.", audio_file_uploaded])
        
        if progress:
            progress(0.9, "Gemini transcription completed, cleaning up...")

        genai.delete_file(audio_file_uploaded.name)

        if progress:
            progress(1.0, "Transcription complete!")

        return response.text
    except Exception as e:
        raise Exception(f"Google Gemini transcription failed: {str(e)}")
