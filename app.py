"""
Gradio web application for dysarthric voice command classification.
Optimized for Hugging Face Spaces deployment.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

from src.inference.predictor import VoiceCommandPredictor


# HuggingFace Spaces paths (files will be in the root of the Space)
# For HF Spaces, use HuggingFace Hub model ID directly
HUBERT_MODEL_ID = "facebook/hubert-large-ls960-ft"

# Download model checkpoint from HuggingFace Hub (stored in separate model repo)
print("Downloading model checkpoint from HuggingFace Hub...")
CHECKPOINT_PATH = hf_hub_download(
    repo_id="DNE58293/dysarthric-voice-model",
    filename="phase_b_curriculum_trained.pt"
)
print(f"Model downloaded to: {CHECKPOINT_PATH}")

# Label mapping (small file, kept in Space repo)
LABEL_MAPPING_PATH = "label_mapping.json"


def create_app():
    """Create and configure the Gradio app."""
    
    # Initialize predictor
    print("Loading model...")
    predictor = VoiceCommandPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        label_mapping_path=LABEL_MAPPING_PATH,
        model_cache_dir=HUBERT_MODEL_ID  # HF Hub will auto-download
    )
    print("Model loaded!")
    
    def classify_audio(audio):
        """
        Classify the recorded audio.
        
        Args:
            audio: Tuple of (sample_rate, audio_array) from Gradio
            
        Returns:
            Tuple of (label_dict, top5_text)
        """
        if audio is None:
            return None, "No audio recorded. Please record your voice."
        
        sample_rate, audio_array = audio
        
        # Convert to float32 and normalize
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        
        # Handle stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Get prediction
        result = predictor.predict(audio_array, sample_rate)
        
        # Format for Gradio Label component
        label_dict = result['probabilities']
        
        # Format top-5 predictions
        top5 = list(result['probabilities'].items())[:5]
        top5_text = "\n".join([
            f"**{i+1}. {label}**: {prob*100:.1f}%" 
            for i, (label, prob) in enumerate(top5)
        ])
        
        return label_dict, top5_text
    
    # Define command categories for display
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'forward', 'back', 'select', 'menu']
    
    # Custom CSS for a distinctive look
    custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #e94560 !important;
        margin: 0 !important;
    }
    .main-header p {
        color: #eee !important;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(
        title="Dysarthric Voice Command Classifier",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="rose",
            secondary_hue="slate",
        )
    ) as app:
        with gr.Column(elem_classes="main-header"):
            gr.Markdown("""
            # 🎤 Dysarthric Voice Command Classifier
            
            This application uses a fine-tuned **HuBERT** model to recognize voice commands,
            specifically designed to work with **dysarthric speech patterns**.
            """)
        
        gr.Markdown("""
        ## 📋 Instructions
        Use the audio box below to **record** (microphone icon) or **upload** a file (upload icon). 
        The audio will be automatically classified.
        
        ## 🗣️ Supported Commands
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"**Digits:** {', '.join(digits)}")
            with gr.Column():
                gr.Markdown(f"**Commands:** {', '.join(commands)}")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎙️ Record or Upload Audio"
                )
            
            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Predicted Command",
                    num_top_classes=5
                )
                
                top5_output = gr.Markdown(
                    label="Top 5 Predictions",
                    value="Record or upload audio to see predictions..."
                )
        
        # Auto-classify when recording stops
        audio_input.stop_recording(
            fn=classify_audio,
            inputs=[audio_input],
            outputs=[output_label, top5_output]
        )
        
        # Auto-classify when file is uploaded
        audio_input.change(
            fn=classify_audio,
            inputs=[audio_input],
            outputs=[output_label, top5_output]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About the Model
        
        This model was trained on the **TORGO dataset** using **curriculum learning**:
        1. **Phase A**: Pre-training on control (non-dysarthric) speakers
        2. **Phase B**: Fine-tuning on dysarthric speakers
        3. **Phase C**: Leave-one-speaker-out evaluation
        
        The model achieves **~87% accuracy** on dysarthric voice commands.
        
        ---
        
        *Built with [Gradio](https://gradio.app) • Model: [HuBERT](https://huggingface.co/facebook/hubert-large-ls960-ft)*
        """)
    
    return app


# Create and launch the app
app = create_app()

if __name__ == "__main__":
    app.launch()
