
<div align="center">

#  Restormer-LLM: Intelligent Text Restoration

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Weights-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/VaishV/RestormerForTextDeblurring)
[![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3_70B-F55036?style=for-the-badge)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

### A 3-stage AI pipeline that recovers readable text from motion-blurred images —<br>combining Restoration Transformers, OCR, and LLM-based semantic refinement.

[Website Link](https://intelligent-text-restoration.lovable.app/)

</div>

---

##  The Problem

Motion blur and camera shake destroy the **high-frequency edge information** that OCR engines depend on. Standard CNNs smooth over fine details; standard Vision Transformers scale at **O(H²W²)** complexity — making them impractical for high-resolution text images. Neither approach reliably restores blurred text for downstream extraction.

---

##  Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: Blurred Image                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Restormer Deblur                                     │
│                                                                 │
│  • MDTA: channel-wise attention → linear complexity O(HW·C²)    │
│  • GDFN: gated depthwise convolutions for edge/texture detail   │
│  • Multiscale hierarchical encoder-decoder with skip connects   │
│                                                                 │
│  Weights: VaishV/RestormerForTextDeblurring (HuggingFace)       │
└───────────────────────────────┬─────────────────────────────────┘
                                │  Restored Image
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Tesseract OCR                                        │
│                                                                 │
│  • High-precision text extraction on restored pixel data        │
│  • Images preprocessed: normalized + padded to 8×8 multiples    │
└───────────────────────────────┬─────────────────────────────────┘
                                │  Raw OCR Text
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Groq LLM Semantic Refinement                         │
│                                                                 │
│  • Llama 3.3 70B corrects OCR errors contextually               │
│  • Recovers word meaning even when pixel restoration is noisy   │
│  • Structure-preserving: maintains original text layout         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                      Clean, Readable Text
```

---

##  Key Technical Contributions

### Multi-Dconv Head Transposed Attention (MDTA)
Standard self-attention computes similarity across spatial tokens — **O(H²W²)** for an image of height H and width W, making it infeasible at high resolutions. MDTA instead operates across the **channel dimension**, achieving **linear complexity** while still capturing long-range dependencies critical for blur removal.

### Gated-Dconv Feed-Forward Network (GDFN)
Replaces the standard FFN with a **learned gating mechanism** that selectively suppresses irrelevant features, combined with **depthwise convolutions** for local spatial awareness of edges and fine textures — the exact features needed for text deblurring.

### LLM Semantic Correction Layer
Even after strong deblurring, OCR output on imperfectly restored images often contains plausible-but-wrong characters (*"rn"* vs *"m"*, *"cl"* vs *"d"*). A Groq-hosted **Llama 3.3 70B** pass resolves these contextually, recovering intended meaning at the semantic level rather than the pixel level.

---

##  Visual Results

| | Blurred Input | Restored Output |
|---|---|---|
| **Image** | * <img width="510" height="392" alt="image" src="https://github.com/user-attachments/assets/19db9a03-58da-4201-a42a-7637bbc69535" />* | * <img width="515" height="380" alt="image" src="https://github.com/user-attachments/assets/ffcf2ff3-86e8-4cac-9828-c828fbd0fc47" /> * |
| **OCR Text** | ` ` | `'y the time-scale on which we b er 100 sites) is greater than the ‘er duplication [5S, 6]. Second,  set of residues and can not be mn of non-synonymous change  ` |

>  *before/after image pairs to `assets/results/` and embed here.*

---

## Evaluation Results & Performance

The model was trained for **55,000 iterations** on a subset (~5,500 sharp/blur image pairs + 100 validation pairs) of the [Text Deblurring Dataset (Kaggle)](https://www.kaggle.com/datasets/anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr), using a single GPU.

| Metric | Score | What it measures |
|--------|-------|-----------------|
| **PSNR** | 23.97 dB | Pixel-level reconstruction fidelity (based on MSE) |
| **SSIM** | **0.9470** ✅ | Structural similarity — textures, contrast, and image structure |

> **Achieved SSIM of 0.9470 is competitive with SOTA methods** on motion deblurring benchmarks (RealBlur-J, HIDE).  
> PSNR reflects our custom text-focused dataset and single-GPU training constraints.

### Achieved SOTA SSIM value:
Images can look visually very similar and preserve structure perfectly, yet differ slightly at the pixel level — SSIM captures this. PSNR penalizes every small pixel mismatch, making SSIM the more meaningful metric for perceptual text clarity.

### Comparison with SOTA (HIDE Dataset results — from original Restormer paper)

| Method | PSNR | SSIM |
|--------|------|------|
| DeblurGAN (2017) | 24.51 | 0.871 |
| DeblurGAN-v2 | 26.61 | 0.875 |
| SRN | 28.36 | 0.915 |
| DBGAN | 28.94 | 0.915 |
| MT-RNN | 29.15 | 0.918 |
| MPRNet | 30.96 | 0.939 |
| **Restormer** | **31.22** | **0.942** |

> *Note: The above comparison uses the original Restormer paper's HIDE Dataset results. Our training was conducted on a domain-specific text deblurring dataset for OCR-oriented restoration — a different task and evaluation context.*

---

## Training Setup

### Dataset
- **Source:** [Text Deblurring Dataset with PSF for OCR (Kaggle)](https://www.kaggle.com/datasets/anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr)
- **Training subset:** ~5,500 sharp/blur image pairs
- **Validation subset:** 100 sharp/blur test pairs

### Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight decay=1e⁻⁴) |
| Loss Function | L1 Loss |
| Total Iterations | 55,000 |
| Patch Size | 128×128 |
| Batch Size | 2 |
| GPU | 1 |
| Gradient Clipping | Enabled |

### Learning Rate Schedule
- **Initial LR:** 3×10⁻⁴
- **Scheduler:** Cosine Annealing Restart Cyclic
  - Phase 1 (15,000 iters): LR stays at 3×10⁻⁴
  - Phase 2 (55,000 iters): LR decays to 1×10⁻⁶
- **Validated every:** 1,000 iterations

---

##  Quick Start

### Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- A [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
git clone https://github.com/aarushipd12/Intelligent-Text-Restoration.git
cd Intelligent-Text-Restoration
pip install -r requirements.txt
```

### Running the Pipeline

> The project is designed to run in **Google Colab** for GPU access.

1. Open `Image_Deblurring_&_Text_Extraction_Pipeline.ipynb` in Colab
2. Add your `GROQ_API_KEY` when prompted
3. Upload a motion-blurred image
4. The pipeline automatically runs all 3 stages end-to-end

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Deblurring Model | Restormer (CVPR 2022) via HuggingFace |
| OCR | Tesseract + pytesseract |
| LLM Refinement | Groq API — Llama 3.3 70B |
| Framework | PyTorch |
| Image Processing | OpenCV, PIL |
| Environment | Google Colab (GPU) |

---

##  Repository Structure

```
Intelligent-Text-Restoration/
├── Image_Deblurring_&_Text_Extraction_Pipeline.ipynb  # Full inference pipeline
├── RESTORMER_Training.ipynb                            # Model training notebook
├── assets/                                             # Demo GIFs, result images
│   └── results/
│       ├── blurred_input.png
│       └── restored_output.png
└── README.md
```

---

##  References

- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881) — Zamir et al., CVPR 2022
- Pre-trained weights: [`VaishV/RestormerForTextDeblurring`](https://huggingface.co/VaishV/RestormerForTextDeblurring) on HuggingFace
- [Groq API](https://console.groq.com) — Llama 3.3 70B inference

---

## Acknowledgements

The Restormer fine-tuning and initial OCR pipeline were developed in collaboration 
with [Vaish-Ventra](https://github.com/Vaish-Ventra) as part of a team project.

<div align="center">

Aarushi Pandey | IIT Indore

</div>
