# Future Work & Possibilities

This document outlines potential future directions and improvements for the dysarthric voice command system.

## 1. Dysarthria Detection & Adaptive ASR

Train a dysarthria classifier to detect when the speaker has dysarthria, and use adaptive ASR based on the detection:

- **Approach**: Develop a binary classifier that can determine whether a speaker exhibits dysarthric speech patterns
- **Benefit**: System can automatically route audio to the appropriate model:
  - Standard ASR for typical speech
  - Dysarthria-specific voice command classifier for dysarthric speech
- **Advantages**:
  - More efficient processing for non-dysarthric speakers
  - Specialized handling for dysarthric speakers
  - Could be deployed as a general-purpose voice command system
- **Considerations**:
  - Need labeled data for dysarthria vs. non-dysarthria speech
  - Real-time detection requirements
  - Threshold tuning for classification confidence

## 2. ESPNet Integration / Larger ASR System

Explore using ESPNet or training a larger-scale ASR system:

- **ESPNet**: End-to-End Speech Processing Toolkit
  - State-of-the-art speech recognition framework
  - Supports various architectures (Transformer, Conformer, etc.)
  - Pre-trained models available for fine-tuning
- **Benefits**:
  - Better generalization across speakers
  - More robust to acoustic variability
  - Potential for full-vocabulary speech recognition (beyond limited commands)
- **Approaches**:
  - Fine-tune existing ESPNet models on dysarthric speech datasets
  - Train from scratch with dysarthric + standard speech data
  - Use ESPNet's data augmentation and training recipes
- **Considerations**:
  - Computational requirements for larger models
  - Dataset size requirements
  - Deployment constraints (model size, inference speed)

## Next Steps

- [ ] Collect/annotate data for dysarthria detection task
- [ ] Research ESPNet architectures suitable for command recognition
- [ ] Evaluate compute resources needed for larger models
- [ ] Consider hybrid approaches combining both ideas
