# FoodLens: Dual-Model Recipe Recommendation and Diet Classification System

**FoodLens** is an AI-powered system that combines Content-Based Image Retrieval (CBIR) and diet classification to recommend visually similar recipes and categorize them into dietary profiles. Built within a 3-day timeline, it leverages a dataset of 45,582 recipes, a ResNet18-based autoencoder for CBIR, and a custom convolutional model for classifying recipes into six diet categories: Low-Calorie High-Fiber (LCHFib), High-Sugar Low-Nutrient (Junk), Balanced, High-Protein Low-Carb (HPLC), Low-Carb High-Fat (LCHF), and High-Carb Low-Fat (HCLF). The system integrates FAISS for efficient similarity search and provides a user-friendly interface for recipe discovery and nutritional analysis.

## Features

- **Image-Based Recipe Retrieval**: Upload a food image to retrieve visually similar recipes using a ResNet18-based autoencoder (`cbir_autoencoder_V2.pth`).
- **Diet Classification**: Automatically classify recipes into six dietary categories based on nutritional data using a custom model (`classifier_model_keras`).
- **Nutritional Insights**: View detailed nutritional breakdowns (calories, carbs, sugars, fat, protein) for each recipe.
- **Recipe Details**: Access ingredients, preparation instructions, and original recipe links for recommended dishes.
- **Efficient Search**: Utilizes FAISS for fast nearest-neighbor search in a 128-D latent space (`embeddings.npy`).

## Project Structure

food-lens/
├── data/                   # Datasets
│   ├── images/             # 45K+ recipe images
│   ├── recipes.csv         # Raw recipe data
│   └── recipes_classified.csv # Labeled dataset
│
├── foodlens/models/        # ML models
│   ├── cbir_autoencoder_V2.pth  # CBIR model
│   ├── classifier_model_keras   # Diet classifier  
│   └── embeddings.npy      # Precomputed embeddings
│
├── src/
│   ├── back-end/           # FastAPI code
│   └── front-end/          # Web interface
│
└── notebooks/              # Jupyter notebooks
    ├── cbir_training.ipynb
    ├── diet_classifier.ipynb
    └── downstream.ipynb

## Prerequisites

- **Hardware**: A GPU (e.g., NVIDIA T4, available on Google Colab Pro) is recommended for training and inference.
- **Software**:
  - Python 3.8+
  - PyTorch 2.0+ (for CBIR model)
  - TensorFlow/Keras (for nutrition classifier)
  - FastAPI (for backend)
  - JavaScript (for frontend)
  - Bootstrap 5 (for styling, included in `index.html`)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/food-lens.git
cd food-lens
```
### 2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
Install the required dependencies:
```
```bash
pip install -r requirements.txt
```
Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The FastAPI backend should now be running at http://127.0.0.1:8000.

Frontend
Navigate to the frontend folder (if applicable):

```bash
cd frontend
Open the index.html file in a browser:
```
This is a simple static page that allows you to upload an image and get similar recipe recommendations.
