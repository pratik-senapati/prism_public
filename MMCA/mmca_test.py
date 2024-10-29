import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from mmca.main import MultiModalCausalAttention

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the MMCA model
attn = MultiModalCausalAttention(dim=512, heads=8)

# Define the path to the sample image and caption
image_path = "animals.jpeg"
accurate_caption = "Step into a serene countryside where vibrant farm animals roam freely in a lush green field. This painting captures the essence of rural life. 🐄🌾🐑 #FarmLife #ArtInspiration #CountrysideBeauty"
inaccurate_caption = "A bunch of random nonsense."

# Function to extract and normalize features using CLIP
def extract_features(image_path, caption):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    # Ensure the image is a 3D array (height, width, channels)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[..., :3]

    # Convert numpy array back to PIL Image
    image = Image.fromarray((image_np * 255).astype(np.uint8))

    # Extract features using CLIP
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    return image_features, text_features

# Extract features for accurate and inaccurate captions
image_features_accurate, text_features_accurate = extract_features(image_path, accurate_caption)
image_features_inaccurate, text_features_inaccurate = extract_features(image_path, inaccurate_caption)

# Function to apply MMCA and compute similarity
def compute_similarity_with_mmca(image_features, text_features):
    # Ensure the features have the same dimension
    image_features = image_features.view(1, -1, 512)  # Reshape if necessary
    text_features = text_features.view(1, -1, 512)  # Reshape if necessary

    # Apply MMCA
    img_attended, text_attended = attn(image_features, text_features)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(img_attended, text_attended, dim=-1)
    return similarity.item()

# Compute similarities
similarity_accurate = compute_similarity_with_mmca(image_features_accurate, text_features_accurate)
similarity_inaccurate = compute_similarity_with_mmca(image_features_inaccurate, text_features_inaccurate)

print("Cosine Similarity with MMCA (Accurate Caption):", similarity_accurate)
print("Cosine Similarity with MMCA (Inaccurate Caption):", similarity_inaccurate)

# Compare raw features without MMCA
raw_similarity_accurate = torch.nn.functional.cosine_similarity(image_features_accurate, text_features_accurate, dim=-1).item()
raw_similarity_inaccurate = torch.nn.functional.cosine_similarity(image_features_inaccurate, text_features_inaccurate, dim=-1).item()

print("Raw Cosine Similarity (Accurate Caption):", raw_similarity_accurate)
print("Raw Cosine Similarity (Inaccurate Caption):", raw_similarity_inaccurate)

# Verify feature dimensions
print("Image Features Shape:", image_features_accurate.shape)
print("Text Features Shape:", text_features_accurate.shape)

# Experiment with different MMCA configurations
attn_alternative = MultiModalCausalAttention(dim=512, heads=4)  # Try different number of heads

def compute_similarity_with_alternative_mmca(image_features, text_features):
    # Ensure the features have the same dimension
    image_features = image_features.view(1, -1, 512)  # Reshape if necessary
    text_features = text_features.view(1, -1, 512)  # Reshape if necessary

    # Apply alternative MMCA
    img_attended, text_attended = attn_alternative(image_features, text_features)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(img_attended, text_attended, dim=-1)
    return similarity.item()

# Compute similarities with alternative MMCA
similarity_accurate_alternative = compute_similarity_with_alternative_mmca(image_features_accurate, text_features_accurate)
similarity_inaccurate_alternative = compute_similarity_with_alternative_mmca(image_features_inaccurate, text_features_inaccurate)

print("Cosine Similarity with Alternative MMCA (Accurate Caption):", similarity_accurate_alternative)
print("Cosine Similarity with Alternative MMCA (Inaccurate Caption):", similarity_inaccurate_alternative)