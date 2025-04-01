# FoodLens

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/) [![Colab](https://img.shields.io/badge/Google%20Colab-Pro-green.svg)](https://colab.research.google.com/)

## Overview

**FoodLens** is an innovative machine learning project that combines **Content-Based Image Retrieval (CBIR)** and **diet-level classification** to analyze a dataset of approximately 45,000 recipe images. Each image, sized at 256x256 pixels, is labeled with one of six dietary categories: Low-Calorie High-Fiber (LCHFib), Junk, Balanced, High-Protein Low-Carb (HPLC), Low-Carb High-Fat (LCHF), and High-Carb Low-Fat (HCLF). The system leverages a hybrid autoencoder to (1) retrieve visually similar recipes and (2) classify their dietary profiles, offering a dual-purpose tool for culinary and health-focused applications.

Built on PyTorch and accelerated with Google Colab Pro, FoodLens uses `CBIRCAutoEncoder_v2`, a custom autoencoder with a pretrained ResNet18 encoder, to extract a 128-dimensional latent representation. This powers both CBIR (via Euclidean distance or FAISS) and classification (6-class logits). Initial training with 2 epochs yielded 27% accuracy, but ongoing efforts target 60–70% with 20 epochs. The project also integrates reconstruction error metrics inspired by "Class-wise Autoencoders Measure Classification Difficulty And Detect Label Mistakes" to diagnose class difficulty and detect potential mislabels.

## Features
- **CBIR**: Retrieve the top-5 visually similar recipe images from the dataset.
- **Diet Classification**: Predict dietary categories with a hybrid autoencoder.
- **Pretrained Backbone**: ResNet18 encoder for robust feature extraction.
- **Scalable**: Handles 45K images with Colab Pro’s GPU (T4/P100) and 32 GB RAM.
- **Diagnostics**: Reconstruction error analysis for performance insights.

## Getting Started
1. **Clone the Repo**: `git clone `
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run in Colab**: Upload the Jupyter notebook and dataset to Google Colab Pro.
4. **Train**: Execute the training loop (20 epochs recommended).
5. **Evaluate**: Check classification accuracy and CBIR results.

## Dataset
- **Size**: ~45,582 RGB images, 256x256 resolution.
- **Classes**: 6 balanced diet categories.
- **Source**: Custom recipe image collection (stored at `/MyDrive/project/data.recipe.csv`).

## Progress
- **Training**: 2 epochs completed (27% accuracy); 20 epochs in progress.
- **Next Steps**: Full inference, CBIR visualization, and report finalization (due in 3 days!).

## License
MIT License—feel free to fork and tweak!

## Acknowledgments
- Built with ❤️ in a 3-day sprint.
- Inspired by "Class-wise Autoencoders" paper for error metrics.
- Powered by Colab Pro’s GPU muscle.