from fastapi import UploadFile
from PIL import Image
import io
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
import asyncio
from scipy.spatial.distance import euclidean, cityblock

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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

def normalize_score(value, min_value, max_value, scale_min=0, scale_max=1000):
    return ((value - min_value) / (max_value - min_value)) * (scale_max - scale_min) + scale_min

async def test_compare_image_caption():
    # Load an image
    image_path = "animals.jpeg"
    with open(image_path, "rb") as image_file:
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    
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
    
    # Use a placeholder caption
    placeholder_caption = "a photo of a cat and dogs and birds and farm animals like cows and pigs and sheep in a barn on a sunny day"
    
    # Preprocess the image and text with padding
    inputs = clip_processor(text=[placeholder_caption], images=image, return_tensors="pt", padding=True)
    
    # Extract features
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
    
    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
    
    # Flatten the tensors for distance calculations
    reduced_image_features_flat = image_features.flatten().numpy()
    reduced_text_features_flat = text_features.flatten().numpy()
    
    # Calculate cosine similarity
    cosine_similarity_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    
    # Calculate Euclidean distance
    euclidean_distance_score = euclidean(reduced_image_features_flat, reduced_text_features_flat)
    
    # Calculate Manhattan distance
    manhattan_distance_score = cityblock(reduced_image_features_flat, reduced_text_features_flat)
    
    # Calculate dot product
    dot_product_score = torch.dot(image_features.squeeze(), text_features.squeeze()).item()
    
    # Normalize the scores to a scale of 0 to 1000
    cosine_similarity_score_normalized = normalize_score(cosine_similarity_score, -1, 1, 0, 1000)
    euclidean_distance_score_normalized = normalize_score(euclidean_distance_score, 0, 10, 0, 1000)  # Adjust max distance based on your data
    manhattan_distance_score_normalized = normalize_score(manhattan_distance_score, 0, 50, 0, 1000)  # Adjust max distance based on your data
    dot_product_score_normalized = normalize_score(dot_product_score, -1, 1, 0, 1000)
    
    # Categorize the score
    category = categorize_score(cosine_similarity_score_normalized)
    
    print({
        "placeholder_caption": placeholder_caption,
        "cosine_similarity_score": cosine_similarity_score,
        "cosine_similarity_score_normalized": cosine_similarity_score_normalized,
        "euclidean_distance_score": euclidean_distance_score,
        "euclidean_distance_score_normalized": euclidean_distance_score_normalized,
        "manhattan_distance_score": manhattan_distance_score,
        "manhattan_distance_score_normalized": manhattan_distance_score_normalized,
        "dot_product_score": dot_product_score,
        "dot_product_score_normalized": dot_product_score_normalized,
        "normalized_score": cosine_similarity_score_normalized,
        "category": category
    })

# Run the test function
asyncio.run(test_compare_image_caption())