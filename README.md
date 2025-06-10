# 🧠 Visual Decoding from EEG

**Author**
- Atharv Kumar (atharvkumar43@gmail.com)

**Faculty Guides:**  
Prof. Arnav Bhaskar

---

## 📝 Overview

This project explores decoding **EEG (Electroencephalogram)** signals to reconstruct the **visual stimuli** experienced by the brain — using **text and image generation models**. It leverages **deep learning** and **multi-modal alignment** to generate high-fidelity image reconstructions from brain signals.

This work opens new pathways in **brain-computer interfaces**, **neuroscience**, and **thought-driven AI systems**.

---

## 🎯 Objectives

- **EEG-based Textual Encoding**: Extract meaningful embeddings from EEG data.
- **Image Reconstruction**: Use captions (via BLIP-2) and images (via Stable Diffusion) to reconstruct what the subject saw.
- **Direct Thought-to-Image**: Create an end-to-end pipeline from EEG → Text → Image.

---

## 🧠 Dataset

- **EEG Signals**: 16,740 EEG samples (17 channels, 100 timepoints each).
- **Images**: Each of the 16,740 images shown to 10 subjects.
- **Labels**: For supervised and aligned training.

---

## 🔧 Methodology

### 🔹 Step 1: EEG Embedding (VAE)
- VAE trained on DEAP dataset to extract EEG embeddings.
- Ensures compact, meaningful signal representation.

### 🔹 Step 2: Caption Generation (BLIP-2)
- BLIP-2 generates captions from the original image.

> 🧾 _"A small armadillo walking on the dirt"_

### 🔹 Step 3: Cross-Modal Alignment (CLIP / Masked CLIP)
- Align EEG and text embeddings via CLIP.
- Trained to bring both into a common latent space.

### 🔹 Step 4: Text Generation (GPT-2)
- GPT-2 decodes EEG → Text via autoregressive generation.

> 🧠 ➡️ GPT-2 ➡️ _"A baby armadillo in its enclosure at the zoo"_

### 🔹 Step 5: Depth Estimation (GCNN/GAT)
- Graph CNN captures spatial relations in EEG for image depth features.

### 🔹 Step 6: Image Reconstruction (Stable Diffusion)
- Prompt + Depth Map → Stable Diffusion (v2.1 base) to synthesize visual output.

---

## 🧩 Model Architecture

![Model Architecture](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20150550.png)

---

## 📊 Results

### ✅ Caption Alignment Results

| EEG Caption (GPT-2)                          | BLIP Caption                                  | ROUGE Score |
|---------------------------------------------|-----------------------------------------------|-------------|
| "a man holding an accordion..."              | "a person playing an accordion..."            | 0.44        |
| "a floral air mattress..."                   | "an air mattress with a floral pattern..."    | 0.52        |

---

### ✅ Image Reconstruction Results

| EEG Signal | Original Image | Caption | Generated Text | Reconstructed Image | SSIM |
|------------|----------------|---------|----------------|----------------------|------|
| ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151516.png) | ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151526.png) | "a small armadillo walking in the dirt" | "a baby armadillo enclosure at the zoo" | ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151536.png) | 11.02% |
| ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151752.png) | ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151758.png) | "a group of people riding on a boat" | "a group of people in an airboat" | ![](https://github.com/Machforo/Visual-Decoding-using-EEG/blob/main/Screenshot%202025-06-10%20151809.png) | 14.32% |

---

## 🔬 Quantitative Analysis

- **CLIP Loss**: Dropped from 3.48 to 0.12 (30 epochs).
- **Cosine Similarity Matrix**: Strong diagonals (high EEG-text alignment).
- **ROUGE Scores**: ROUGE-1 between 0.44–0.52.
- **SSIM**: Image similarity remains low (~10–15%), but semantically accurate.



## 📚 References

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [CLIP: Contrastive Language–Image Pretraining](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [GPT-2 by OpenAI](https://openai.com/research/better-language-models)
- [GCNNs for EEG](https://arxiv.org/abs/1805.08768)

---

## 🙏 Acknowledgements

Special thanks to our guides **Prof. Arnav Bhaskar**  for their constant support and insights.

---
