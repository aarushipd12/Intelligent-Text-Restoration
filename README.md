0### (Restormer-ML-in-Signal-Processing)
# Restormer-LLM: Intelligent Text Restoration
​A multi-stage signal processing framework leveraging Restoration Transformers and LLMs for high-fidelity text recovery from degraded imagery. This project addresses the limitations of standard CNNs and Transformers in processing high-resolution, motion-blurred text images.  
​## Overview
​High-frequency textual details are often lost due to motion blur and camera shake, rendering traditional OCR engines ineffective as they rely on pixel-perfect edges. Our pipeline restores these signals through three specialized layers:  
​Deblur: Implements the Restormer architecture to restore high-frequency textual signals.  
​Digitize: Extracts raw text via a high-precision Tesseract OCR layer.  
​Refine: Leverages the Groq LLM (Llama 3.3 70B) to semantically correct and polish the output while preserving original structure.  
​## Architecture & Technical Novelty
​Our implementation utilizes a multiscale hierarchical design with efficient Transformer blocks.  
​Core Components
#​## Multi-Dconv Head Transposed Attention (MDTA): Unlike standard self-attention with O(H^2W^2) complexity, MDTA operates across the channel dimension to achieve linear complexity, making it ideal for high-resolution images. 
### Gated-Dconv Feed-Forward Network (GDFN): Enhances the standard FFN by adding a gating mechanism for selective information flow and depthwise convolutions for spatial awareness of edges and textures.  
​## Getting Started
​Prerequisites
​Python 3.x
​Tesseract OCR engine
​Groq API Key (for the LLM refinement layer)

git clone https://github.com/aarushipd12/Restormer-ML-in-Signal-Processing.git
cd Restormer-ML-in-Signal-Processing
pip install -r requirements.txt

## Running the Pipeline
​The project is designed to run in Google Colab environment.  
### 1.Preprocessing: 
Images are normalized and padded to 8 \times 8 multiples.  
### 2.Model Inference: 
Load the pre-trained weights from HuggingFace (VaishV/RestormerForTextDeblurring). 
### 3.Execution:

# Upload your blurred image
# The pipeline will automatically:
 1. Deblur using Restormer
  
 2. Extract text using pytesseract
    
 3. Refine text using Groq LLM
