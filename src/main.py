import torch
from ui import create_interface
from config import load_environment_variables, get_device

def main():
    """Main function to launch the application"""
    load_environment_variables()
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        share=False,
        debug=True,
        inbrowser=True  # Automatically open browser
    )

if __name__ == "__main__":
    main()
