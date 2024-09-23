import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from chan import CrossModalHierarchicalAttentionNetwork  # Import the simplified CHAN model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the CHAN model
chan_model = CrossModalHierarchicalAttentionNetwork(dim=512, heads=8)  # Using 512 dimensions

# Define the path to the sample image and caption
image_path = "image.jpg"
accurate_caption = "Embracing the elegance of simplicity, this image captures the essence of serene beauty. 🌿 #EleganceInSimplicity"
inaccurate_caption = "Several men and women dancing on a stage with bright lights and colorful costumes."

# Function to extract features using CLIP
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

# Ensure the features have the correct dimensions
def ensure_correct_dimensions(features, target_dim):
    if features.shape[-1] != target_dim:
        features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(0), target_dim).squeeze(0)
    return features

image_features_accurate = ensure_correct_dimensions(image_features_accurate, 512)
text_features_accurate = ensure_correct_dimensions(text_features_accurate, 512)
image_features_inaccurate = ensure_correct_dimensions(image_features_inaccurate, 512)
text_features_inaccurate = ensure_correct_dimensions(text_features_inaccurate, 512)

# Function to apply CHAN and compute similarity
def compute_similarity_with_chan(image_features, text_features, alpha=1.0, c=1.0, d=3):
    # Ensure the features have the same dimension
    image_features = image_features.view(1, -1, 512)  # Reshape if necessary
    text_features = text_features.view(1, -1, 512)  # Reshape if necessary

    # Apply CHAN
    img_attended, text_attended = chan_model(image_features, text_features)

    # Compute cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(img_attended, text_attended, dim=-1)
    
    # Compute polynomial kernel similarity
    poly_similarity = (alpha * cosine_similarity + c) ** d
    
    return cosine_similarity.item(), poly_similarity.item()

# Function to scale similarity scores to a range from 0 to 1000 with exponential transformation
def scale_score(score, scale_factor=1000):
    return (torch.exp(torch.tensor(score)) - 1) / (torch.exp(torch.tensor(1.0)) - 1) * scale_factor

# Compute similarities
cosine_similarity_accurate, poly_similarity_accurate = compute_similarity_with_chan(image_features_accurate, text_features_accurate)
cosine_similarity_inaccurate, poly_similarity_inaccurate = compute_similarity_with_chan(image_features_inaccurate, text_features_inaccurate)

# Scale the similarities
scaled_cosine_similarity_accurate = scale_score(cosine_similarity_accurate)
scaled_poly_similarity_accurate = scale_score(poly_similarity_accurate)
scaled_cosine_similarity_inaccurate = scale_score(cosine_similarity_inaccurate)
scaled_poly_similarity_inaccurate = scale_score(poly_similarity_inaccurate)

print("Cosine Similarity with CHAN (Accurate Caption):", cosine_similarity_accurate)
print("Scaled Cosine Similarity with CHAN (Accurate Caption):", scaled_cosine_similarity_accurate.item())
print("Polynomial Kernel Similarity with CHAN (Accurate Caption):", poly_similarity_accurate)
print("Scaled Polynomial Kernel Similarity with CHAN (Accurate Caption):", scaled_poly_similarity_accurate.item())
print("Cosine Similarity with CHAN (Inaccurate Caption):", cosine_similarity_inaccurate)
print("Scaled Cosine Similarity with CHAN (Inaccurate Caption):", scaled_cosine_similarity_inaccurate.item())
print("Polynomial Kernel Similarity with CHAN (Inaccurate Caption):", poly_similarity_inaccurate)
print("Scaled Polynomial Kernel Similarity with CHAN (Inaccurate Caption):", scaled_poly_similarity_inaccurate.item())