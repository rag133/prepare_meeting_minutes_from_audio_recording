import gradio as gr
from .transcription import transcribe_openai_api, transcribe_with_gemini
from .summarization import summarize_with_llama, summarize_with_ollama, summarize_with_gemini
from .config import get_device, setup_api_clients

class MeetingMinutesGenerator:
    def __init__(self):
        self.device = get_device()
        self.openai_client = setup_api_clients()

    def process_meeting(self, audio_file, transcription_model, summarization_model, progress=gr.Progress()):
        """Main processing function"""
        try:
            # Step 1: Transcription
            progress(0, "Starting transcription...")
            
            if transcription_model == "OpenAI Whisper API":
                transcription = transcribe_openai_api(self.openai_client, audio_file, 
                    lambda p, desc: progress(p * 0.4, desc))
            elif transcription_model == "Google Gemini":
                transcription = transcribe_with_gemini(audio_file,
                    lambda p, desc: progress(p * 0.4, desc))
            else:
                raise ValueError(f"Unknown transcription model: {transcription_model}")
            
            progress(0.4, "Transcription completed. Starting summarization...")
            
            # Step 2: Summarization
            if summarization_model == "Llama 3.1 8B (Local)":
                summary = summarize_with_llama(transcription, self.device,
                    lambda p, desc: progress(0.4 + p * 0.6, desc))
            elif summarization_model == "Ollama (qwen2.5:latest)":
                summary = summarize_with_ollama(transcription,
                    lambda p, desc: progress(0.4 + p * 0.6, desc))
            elif summarization_model == "Google Gemini 2.0 Flash":
                summary = summarize_with_gemini(transcription,
                    lambda p, desc: progress(0.4 + p * 0.6, desc))
            else:
                raise ValueError(f"Unknown summarization model: {summarization_model}")
            
            progress(1.0, "Meeting minutes generated successfully!")
            
            return summary, transcription
            
        except Exception as e:
            progress(1.0, f"Error: {str(e)}")
            return f"**Error:** {str(e)}", ""

def create_interface():
    """Create Gradio interface"""
    generator = MeetingMinutesGenerator()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .progress-bar {
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Meeting Minutes Generator") as interface:
        gr.Markdown("""
        # 🎤 Meeting Minutes Generator
        
        Upload an audio file and generate professional meeting minutes using AI models.
        
        **Setup Requirements:**
        - For OpenAI: Set `OPENAI_API_KEY` in your .env file
        - For Gemini: Set `GEMINI_API_KEY` in your .env file  
        - For HuggingFace models: Set `HF_TOKEN` in your .env file
        - For Ollama: Ensure Ollama is running with qwen2.5:latest model
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Meeting Audio",
                    type="filepath"
                )
                
                transcription_model = gr.Dropdown(
                    choices=["Google Gemini", "OpenAI Whisper API"],
                    value="Google Gemini",
                    label="Transcription Model",
                    info="Choose how to convert speech to text"
                )
                
                summarization_model = gr.Dropdown(
                    choices=[
                        "Llama 3.1 8B (Local)",
                        "Ollama (qwen2.5:latest)", 
                        "Google Gemini 2.0 Flash"
                    ],
                    value="Google Gemini 2.0 Flash",
                    label="Summarization Model",
                    info="Choose which model to generate meeting minutes"
                )
                
                process_btn = gr.Button(
                    "🚀 Generate Meeting Minutes", 
                    variant="primary"
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📋 Meeting Minutes"):
                        minutes_output = gr.Markdown(
                            label="Generated Meeting Minutes",
                            value="Upload an audio file and click 'Generate Meeting Minutes' to see results here."
                        )
                    
                    with gr.TabItem("📝 Full Transcription"):
                        transcription_output = gr.Textbox(
                            label="Full Transcription",
                            value="The complete transcription will appear here.",
                            lines=15,
                            max_lines=20
                        )
        
        # Event handler
        process_btn.click(
            fn=generator.process_meeting,
            inputs=[audio_input, transcription_model, summarization_model],
            outputs=[minutes_output, transcription_output],
            show_progress=True
        )
        
        gr.Markdown("""
        ---
        ### 💡 Tips:
        - **Audio Quality**: Better audio quality = better transcription accuracy
        - **File Size**: Large files may take longer to process
        - **Models**: Try different combinations to find what works best for your use case
        - **API Keys**: Make sure your API keys are properly configured in the .env file
        """)
    
    return interface
