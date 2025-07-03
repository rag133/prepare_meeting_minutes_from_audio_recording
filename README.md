# Meeting Minutes Generator

This application transcribes an audio file of a meeting and generates professional meeting minutes using a variety of AI models. 

## Features

- **Audio Transcription:** Convert spoken words from your meeting recordings into text.
- **Meeting Summarization:** Generate concise and informative meeting minutes, including a summary, key discussion points, takeaways, and action items.
- **Model Selection:** Choose from a range of transcription and summarization models to find the best fit for your needs.

## Models

This application utilizes the following models:

### Transcription Models

- **Google Gemini:** A powerful and versatile model from Google, offering fast and accurate transcription.
- **OpenAI Whisper:** A state-of-the-art speech recognition model from OpenAI.

### Summarization Models

- **Llama 3.1 8B (Local):** A powerful, open-source model from Meta AI that can be run locally.
- **Ollama (qwen2.5:latest):** A flexible option that allows you to use cái model from Ollama for summarization.
- **Google Gemini 2.0 Flash:** A fast and efficient model from Google for generating high-quality summaries.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd prepare_minutes_from_meeting_recording
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file:**

    Create a `.env` file in the root of the project and add your API keys:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    GEMINI_API_KEY="your-gemini-api-key"
    HF_TOKEN="your-huggingface-token"
    OLLAMA_URL="http://localhost:11434" # Or your Ollama URL
    ```

## Usage

Run the application with the following command:

```bash
python src/main.py
```

This will launch a Gradio interface in your web browser. You can then upload an audio file, select the transcription and summarization models, and generate the meeting minutes.

## Credits
This project was inspired by the LLM Engineering course by Edward Donnor.