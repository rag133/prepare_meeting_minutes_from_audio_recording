import os
import torch
import requests
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def summarize_with_llama(transcription, context, device, progress=None):
    """Summarize using Llama model"""
    if progress:
        progress(0.1, "Loading Llama model...")
    
    try:
        LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        # Quantization config for CUDA
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        ) if device == "cuda" else None
        
        llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        
        llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        if progress:
            progress(0.4, "Llama model loaded, generating summary...")
    except Exception as e:
        raise Exception(f"Failed to load Llama model: {str(e)}")
    
    try:
        system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
        user_prompt = f"Below is a transcript of a meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n\n{transcription}"
        if context:
            user_prompt += f"\n\nAdditional Context: {context}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        inputs = llama_tokenizer.apply_chat_template(messages, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to("cuda")
        
        if progress:
            progress(0.6, "Generating meeting minutes...")
        
        with torch.no_grad():
            outputs = llama_model.generate(
                inputs,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llama_tokenizer.eos_token_id
            )
        
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = llama_tokenizer.decode(inputs[0], skip_special_tokens=True)
        generated_text = response[len(input_text):].strip()
        
        if progress:
            progress(1.0, "Llama summarization completed!")
        
        return generated_text
    except Exception as e:
        raise Exception(f"Llama summarization failed: {str(e)}")

def summarize_with_ollama(transcription, context, progress=None):
    """Summarize using Ollama with qwen2.5:latest"""
    if progress:
        progress(0.1, "Connecting to Ollama...")
    
    try:
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
        system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
        user_prompt = f"Below is a transcript of a meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n\n{transcription}"
        if context:
            user_prompt += f"\n\nAdditional Context: {context}"
        
        payload = {
            "model": "qwen2.5:latest",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        if progress:
            progress(0.3, "Sending request to Ollama...")
        
        response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if progress:
                progress(1.0, "Ollama summarization completed!")
            return result["message"]["content"]
        else:
            raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        raise Exception(f"Ollama summarization failed: {str(e)}")

def summarize_with_gemini(transcription, context, progress=None):
    """Summarize using Google Gemini 2.5 Flash"""
    if progress:
        progress(0.1, "Connecting to Google Gemini...")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown.

Below is a transcript of a meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.

{transcription}
"""
        if context:
            prompt += f"Additional Context: {context}"
        
        if progress:
            progress(0.3, "Generating with Gemini...")
        
        response = model.generate_content(prompt)
        
        if progress:
            progress(1.0, "Gemini summarization completed!")
        
        return response.text
    except Exception as e:
        raise Exception(f"Gemini summarization failed: {str(e)}")
