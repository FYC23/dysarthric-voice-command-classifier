"""
Gradio web application for dysarthric voice command classification.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np

from src.inference.predictor import VoiceCommandPredictor


# Default paths (adjust as needed)
DEFAULT_CHECKPOINT = "outputs/phase_b_curriculum_trained.pt"
DEFAULT_LABEL_MAPPING = "outputs/label_mapping.json"
DEFAULT_MODEL_CACHE = "model_cache/facebook/hubert-large-ls960-ft"


def create_app(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    label_mapping_path: str = DEFAULT_LABEL_MAPPING,
    model_cache_dir: str = DEFAULT_MODEL_CACHE
):
    """Create and configure the Gradio app."""
    
    # Initialize predictor
    print("Loading model...")
    predictor = VoiceCommandPredictor(
        checkpoint_path=checkpoint_path,
        label_mapping_path=label_mapping_path,
        model_cache_dir=model_cache_dir
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
    
    # Create Gradio interface
    with gr.Blocks(
        title="Dysarthric Voice Command Classifier"
    ) as app:
        gr.Markdown("""
        # 🎤 Dysarthric Voice Command Classifier
        
        This application uses a fine-tuned HuBERT model to recognize voice commands,
        specifically designed to work with dysarthric speech patterns.
        
        ## Instructions
        Use the audio box below to **record** (microphone icon) or **upload** a file (upload icon). The audio will be automatically classified. 
        
        ## Supported Commands
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
        ### About
        
        This model was trained on the TORGO dataset using curriculum learning:
        1. **Phase A**: Pre-training on control (non-dysarthric) speakers
        2. **Phase B**: Fine-tuning on dysarthric speakers
        3. **Phase C**: Leave-one-speaker-out evaluation
        
        The model achieves ~87% accuracy on dysarthric voice commands.
        """)
    
    return app


def main():
    """Run the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dysarthric Voice Command Classifier")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--labels", type=str, default=DEFAULT_LABEL_MAPPING,
                        help="Path to label mapping JSON")
    parser.add_argument("--model-cache", type=str, default=DEFAULT_MODEL_CACHE,
                        help="Path to HuBERT model cache")
    parser.add_argument("--share", action="store_true",
                        help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on")
    
    args = parser.parse_args()
    
    app = create_app(
        checkpoint_path=args.checkpoint,
        label_mapping_path=args.labels,
        model_cache_dir=args.model_cache
    )
    
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()