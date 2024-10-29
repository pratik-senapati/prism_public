from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
import io
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
import joblib
import uvicorn
from scipy.spatial.distance import euclidean, cityblock

app = FastAPI()

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the PCA model
pca = joblib.load("pca_model_ookpik.pkl")

def categorize_score(score):
    if score >= 900:
        return "Excellent"
    elif score >= 800:
        return "Very Good"
    elif score >= 700:
        return "Good"
    elif score >= 600:
        return "Average"
    elif score >= 500:
        return "Bad"
    elif score >= 400:
        return "Very Bad"
    elif score >= 300:
        return "Poor"
    else:
        return "Very Poor"

def normalize_score(cosine_similarity_score):
    # Define the range of cosine similarity scores and the corresponding range of normalized scores
    min_cosine = 0.1
    max_cosine = 0.35
    min_normalized = 100
    max_normalized = 1000
    
    # Apply linear transformation
    normalized_score = ((cosine_similarity_score - min_cosine) / (max_cosine - min_cosine)) * (max_normalized - min_normalized) + min_normalized
    return normalized_score

@app.post("/compare")
async def compare_image_caption(image: UploadFile = File(...), caption: str = Form(...)):
    # Trim the caption to 77 characters
    caption = caption[:77]
    
    # Load the uploaded image
    image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    
    # Resize and normalize the image
    image = image.resize((224, 224))  # Resize to the model's expected input size
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1] range
    
    # Ensure the image is a 3D array (height, width, channels)
    if image_np.ndim == 2:  # Grayscale image
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:  # RGBA image, remove alpha channel
        image_np = image_np[..., :3]
    
    # Convert numpy array back to PIL Image
    image = Image.fromarray((image_np * 255).astype(np.uint8))
    
    # Preprocess the image and text with padding
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    
    # Extract features
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
    
    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
    
    # Combine features
    combined_features = torch.cat((image_features, text_features), dim=0).cpu().numpy()
    
    # Apply PCA to reduce dimensionality
    reduced_features = pca.transform(combined_features)
    
    # Split the reduced features back into image and text features
    reduced_image_features = torch.tensor(reduced_features[:image_features.shape[0]])
    reduced_text_features = torch.tensor(reduced_features[image_features.shape[0]:])
    
    # Flatten the tensors for distance calculations
    reduced_image_features_flat = reduced_image_features.flatten().numpy()
    reduced_text_features_flat = reduced_text_features.flatten().numpy()
    
    # Calculate cosine similarity
    cosine_similarity_score = torch.nn.functional.cosine_similarity(reduced_image_features, reduced_text_features).item()
    
    # Calculate Euclidean distance
    euclidean_distance_score = euclidean(reduced_image_features_flat, reduced_text_features_flat)
    
    # Calculate Manhattan distance
    manhattan_distance_score = cityblock(reduced_image_features_flat, reduced_text_features_flat)
    
    # Calculate dot product
    dot_product_score = torch.dot(reduced_image_features.squeeze(), reduced_text_features.squeeze()).item()
    
    # Normalize the cosine similarity score to a scale of 100 to 1000
    normalized_score = normalize_score(cosine_similarity_score)
    
    # Categorize the score
    category = categorize_score(normalized_score)
    
    return {
        "caption": caption,
        "cosine_similarity_score": float(cosine_similarity_score),
        "euclidean_distance_score": float(euclidean_distance_score),
        "manhattan_distance_score": float(manhattan_distance_score),
        "dot_product_score": float(dot_product_score),
        "normalized_score": float(normalized_score),
        "category": category
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)