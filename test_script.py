from fastapi import UploadFile
from PIL import Image
import io
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import asyncio

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

def normalize_score(cosine_similarity_score):
    # Define the range of cosine similarity scores and the corresponding range of normalized scores
    min_cosine = 0.1
    max_cosine = 0.35
    min_normalized = 100
    max_normalized = 1000
    
    # Apply linear transformation
    normalized_score = ((cosine_similarity_score - min_cosine) / (max_cosine - min_cosine)) * (max_normalized - min_normalized) + min_normalized
    return normalized_score

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
    placeholder_caption = "Experience the tranquility of the countryside through this captivating artwork that brings to life the charm of farm animals in their natural habitat."
    
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
    
    # Calculate cosine similarity
    cosine_similarity_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    
    # Normalize the score to a scale of 100 to 1000
    normalized_score = normalize_score(cosine_similarity_score)
    
    # Categorize the score
    category = categorize_score(normalized_score)
    
    print({
        "placeholder_caption": placeholder_caption,
        "cosine_similarity_score": cosine_similarity_score,
        "normalized_score": normalized_score,
        "category": category
    })

# Run the test function
asyncio.run(test_compare_image_caption())