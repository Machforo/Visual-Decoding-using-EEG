# 🧠 Visual Decoding from EEG

This repository contains the implementation and results of the project titled **"Visual Decoding from EEG"**, conducted by **Atharv Kumar (B21038)** under the guidance of **Prof. Arnav Bhaskar** and **Prof. Padmanabhan**.

## 📌 Overview

This project explores the decoding of brain activity (EEG signals) to reconstruct corresponding visual stimuli. By integrating multimodal deep learning techniques, it aims to enable "thought-to-image" synthesis — transforming raw EEG signals into descriptive text and eventually reconstructing the original visual stimuli.

---

## 🧠 Objectives

- **EEG-to-Textual Encoding**: Extract semantic textual information from EEG signals using a VAE + CLIP pipeline.
- **Multimodal Integration**: Align EEG embeddings with image captions using BLIP-2 and train a CLIP model for EEG-to-text conversion.
- **Image Reconstruction**: Reconstruct high-quality images from generated text and EEG-derived depth maps using the Stable Diffusion model.
- **"Thought-to-Image" Synthesis**: Bypass text as an intermediate and directly convert EEG to visual content.

---

## 🧩 Model Architecture

![Overall Model Architecture](./assets/overall_model.png)

> **Components:**
> - **EEG Input**: 16740 EEG samples (17 channels, 100 timepoints) across 10 subjects
> - **VAE Model**: Encodes EEG into latent embeddings
> - **BLIP-2**: Generates captions for paired images
> - **CLIP**: Aligns EEG embeddings with text embeddings; trained to predict text directly from EEG
> - **Depth Extraction**: GAT model used to infer spatial depth from EEG
> - **Stable Diffusion**: Generates final image using the predicted caption and depth map

---

## 🗃️ Dataset

- EEG signals (16740 samples, 10 subjects)
- Paired images shown to subjects
- Semantic labels (for caption generation and embedding alignment)

---

## 🛠️ Methodology

### 🔁 VAE for EEG Encoding
- Trained to minimize reconstruction loss + KL divergence
- Captures latent embeddings from EEG signals

### 🧾 Caption Generation with BLIP-2
- Fine-tuned on dataset to produce rich image captions
- Outputs used for training the text prediction model and image generation

### 🔀 Cross Modal Alignment (CLIP)
- CLIP trained to bring EEG and text embeddings into a shared latent space
- ROUGE score used to evaluate similarity between actual and predicted captions

### 🖼️ Image Reconstruction with Stable Diffusion
- Combines text and depth as prompt to generate images
- Stable Diffusion 2.1 base model used
- SSIM used for image quality evaluation

### 📏 Depth Estimation from EEG
- GAT-based model predicts depth maps from EEG embeddings
- Combined with generated text to enrich image reconstruction

---

## 🧪 Results

| Metric | Caption 1 | Caption 2 |
|--------|-----------|-----------|
| ROUGE-1 F1 | 0.44 | 0.52 |
| SSIM (Image 1) | 11.02% |
| SSIM (Image 2) | 14.32% |

Example 1:
- **EEG-derived Caption**: *a baby armadillo in its enclosure at the zoo*
- **BLIP-2 Caption**: *a small armadillo walking on the dirt*

Example 2:
- **EEG-derived Caption**: *a group of people riding in an airboat*
- **BLIP-2 Caption**: *a group of people riding in a boat on the water*

---

## 🔮 Future Work

- Improve EEG embedding quality using masked CLIP
- Explore direct EEG-to-text generation without CLIP (e.g., DeCap)
- Experiment with hvEEGNet for enhanced encoding
- Benchmark across multiple subjects and datasets
- Final submission and report by **November**

---

## 📚 References

- [CLIP - Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [BLIP-2 - Bootstrapping Language-Image Pretraining](https://arxiv.org/abs/2301.12597)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

---

## 🙏 Acknowledgments

Special thanks to Prof. Arnav Bhaskar and Prof. Padmanabhan for their constant guidance and feedback throughout this project.

---

## 📁 Directory Structure

```bash
.
├── assets/
│   └── overall_model.png  # Architecture diagram
├── data/
│   ├── eeg_signals.npy
│   ├── image_data/
│   └── labels.csv
├── src/
│   ├── vae_model.py
│   ├── blip_finetune.py
│   ├── clip_training.py
│   ├── depth_from_eeg.py
│   └── stable_diffusion_infer.py
├── results/
│   ├── reconstructed_images/
│   └── metrics/
├── README.md
