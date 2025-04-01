# main.py
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, io
from io import BytesIO
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
# import keras
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to your frontend URL for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Define the base directory dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print(f"BASE_DIR: {BASE_DIR}")

# Use BASE_DIR to dynamically set paths
MODEL_PATH = os.path.join(BASE_DIR, 'models/cbir_autoencoder_V2.pth')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'models/embeddings.npy')
IMAGE_LIST_PATH = os.path.join(BASE_DIR, 'data/image_list.csv')
RECIPES_PATH = os.path.join(BASE_DIR, "data/recipes_classified.csv")
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models/classifier_autoencoder.keras")
IMAGE_DIR = os.path.join(BASE_DIR, "data/images")

df_recipes = pd.read_csv(RECIPES_PATH)
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
# classifier_model = keras.models.load_model(CLASSIFIER_MODEL_PATH)
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
    print("[INFO] Loading image...")
    print(f"[DEBUG] File name: {file.filename}")

    # image_data = await file.read()
    # Construct the full path
    image_path = os.path.join(IMAGE_DIR, file.filename)


    print(f"[INFO] Loading image from: {image_path}")

    # Read the image from disk
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    print("[DEBUG] Image data received")
    image = Image.open(io.BytesIO(image_data))
    print("[INFO] Image uploaded successfully")
    # Get the embedding for the uploaded image
    query_embedding = get_image_embedding(image)

    # Perform similarity search
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]

    # Get the similar images
    similar_images = df_image_list.iloc[top_indices]

    # Define the correct image directory
    
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    
    # Construct the full image paths
    results = [
        {
            "filename": row["filename"],
            "full_path": os.path.join(IMAGE_DIR, row["filename"])  # Ensure correct local path
        }
        for _, row in similar_images.iterrows()
        if os.path.exists(os.path.join(IMAGE_DIR, row["filename"]))  # Check if file exists
    ]
    # Get the first image filename (without extension)
    first_image_filename = results[0]["filename"].replace(".jpg", "")

    # Fetch the corresponding recipe
    df_recipe_gotten = df_recipes[df_recipes["recipe_id"] == first_image_filename]
    # features = ['calories', 'carbohydrates_g', 'sugars_g', 'fat_g', 'protein_g']
    # class_list = ['Balanced', 'HCLF', 'HPLC', 'Junk', 'LCHF', 'LCHFib']
    # # Recreate MinMaxScaler using df_recipe_list (assuming similar range as training data)
    # # Recreate MinMaxScaler using df_recipe_list (assuming similar range as training data)
    # scaler = MinMaxScaler()
    # X_normalized = scaler.fit_transform(df_recipe_gotten[features])  # Fit on available data

    # # Reshape for CNN input (samples, height=1, width=5, channels=1)
    # X_image = X_normalized.reshape(-1, 1, 5, 1)
    # le = LabelEncoder()
    # le.fit(class_list)  # Fit with known labels


    # # Make predictions
    # predictions = model.predict(X_image)

    # # Convert softmax probabilities to class labels
    # predicted_classes = le.inverse_transform(predictions.argmax(axis=1))

    # # Add predictions to df_recipe_list
    # df_recipe_gotten['predicted_diet'] = predicted_classes
    print("[INFO] Similar images and recipe details fetched successfully")
    return {
        "similar_images": results,
        "recipe_details": df_recipe_gotten.to_dict(orient="records")  # Return recipe details
    }
