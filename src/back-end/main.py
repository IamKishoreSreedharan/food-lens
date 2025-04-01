# main.py
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from io import BytesIO
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
app = FastAPI()

# Define base path (move two levels up)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print(BASE_DIR)
# Define the base directory dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print(f"BASE_DIR: {BASE_DIR}")

# Use BASE_DIR to dynamically set paths
MODEL_PATH = os.path.join(BASE_DIR, 'models/cbir_autoencoder_V2.pth')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'models/embeddings.npy')
IMAGE_LIST_PATH = os.path.join(BASE_DIR, 'data/image_list.csv')
class ConvAutoencoder_v2(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_v2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Print paths to verify
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"EMBEDDINGS_PATH: {EMBEDDINGS_PATH}")
print(f"IMAGE_LIST_PATH: {IMAGE_LIST_PATH}")
# Load embeddings and image list
embeddings = np.load(EMBEDDINGS_PATH)

df_image_list = pd.read_csv(IMAGE_LIST_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize model and load weights
model = ConvAutoencoder_v2().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transformation for input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Transformation for input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Device configuration

# Function to preprocess the image and get the embedding
def get_image_embedding(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encoder(image).flatten(start_dim=1)
    return embedding.cpu().numpy()

# Endpoint to upload image
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Load the image from the uploaded file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Get the embedding for the uploaded image
    query_embedding = get_image_embedding(image)

    # Perform similarity search
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]

    # Get the similar images
    similar_images = df_image_list.iloc[top_indices]

    # Return the top similar images
    results = similar_images[['filename', 'full_path']].to_dict(orient='records')
    return {"similar_images": results}
