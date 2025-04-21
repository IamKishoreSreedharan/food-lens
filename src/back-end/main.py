import os
import io
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model

# ---------------------- CONFIGURATION ---------------------- #

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, 'models/cbir_autoencoder_V2.pth')
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, 'models/classifier_autoencoder.keras')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'models/embeddings.npy')
IMAGE_LIST_PATH = os.path.join(BASE_DIR, 'data/image_list.csv')
RECIPES_PATH = os.path.join(BASE_DIR, "data/recipes_classified.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data/images")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- MODEL DEFINITIONS ---------------------- #

class ConvAutoencoder(nn.Module):
    """
    Deep convolutional autoencoder for image embeddings
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2),
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),
            nn.ConvTranspose2d(32, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------- LOAD MODELS & DATA ---------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder_model = ConvAutoencoder().to(device)
autoencoder_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
autoencoder_model.eval()

classifier_model = load_model(CLASSIFIER_MODEL_PATH)
logger.info("Both models loaded successfully.")

embeddings = np.load(EMBEDDINGS_PATH)
df_image_list = pd.read_csv(IMAGE_LIST_PATH)
df_recipes = pd.read_csv(RECIPES_PATH)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------------- FASTAPI SETUP ---------------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- CORE FUNCTIONS ---------------------- #

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Extract image embedding using encoder
    """
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = autoencoder_model.encoder(image).flatten(start_dim=1)
    return embedding.cpu().numpy()

def predict_diet(nutrition_data: pd.DataFrame) -> str:
    """
    Predict diet category using classifier model and nutrition values
    """
    features = ['calories', 'total_fat', 'sugar', 'sodium', 'protein']
    class_list = ['Balanced', 'HCLF', 'HPLC', 'Junk', 'LCHF', 'LCHFib']

    scaler = MinMaxScaler()
    le = LabelEncoder()
    le.fit(class_list)

    X = nutrition_data[features]
    X_scaled = scaler.fit_transform(X)  # Assuming small sample or similar scale
    X_reshaped = X_scaled.reshape(-1, 1, 5, 1)

    prediction = classifier_model.predict(X_reshaped)
    predicted_label = le.inverse_transform([prediction.argmax(axis=1)[0]])[0]

    return predicted_label

# ---------------------- API ROUTE ---------------------- #

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Process uploaded image, find top 5 similar images, return recipe and diet prediction
    """
    try:
        logger.info(f"Processing file: {file.filename}")
        image_path = os.path.join(IMAGE_DIR, file.filename)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        query_embedding = get_image_embedding(image)

        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-5:][::-1]

        similar_images = df_image_list.iloc[top_indices]
        results = [
            {
                "filename": row["filename"],
                "full_path": os.path.join(IMAGE_DIR, row["filename"])
            }
            for _, row in similar_images.iterrows()
        ]

        recipe_id = results[0]["filename"].replace(".jpg", "")
        matched_recipe = df_recipes[df_recipes["recipe_id"] == recipe_id]

        predicted_diet = predict_diet(matched_recipe)
        matched_recipe["predicted_diet"] = predicted_diet

        return {
            "similar_images": results,
            "recipe_details": matched_recipe.to_dict(orient="records")
        }

    except Exception as e:
        logger.exception("Error processing image.")
        raise HTTPException(status_code=500, detail="Internal server error")
