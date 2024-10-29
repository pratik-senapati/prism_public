import torch
from PIL import Image
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from chan_test.chan import CrossModalHierarchicalAttentionNetwork  # Import the simplified CHAN model
from tqdm import tqdm  # For progress bar
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the CHAN model
chan_model = CrossModalHierarchicalAttentionNetwork(dim=512, heads=8)  # Using 512 dimensions

# Function to extract features using CLIP
def extract_features(image_path, caption, max_length=77):
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
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    return image_features, text_features

# Ensure the features have the correct dimensions
def ensure_correct_dimensions(features, target_dim):
    if features.shape[-1] != target_dim:
        features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(0), target_dim).squeeze(0)
    return features

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

# Load OOKPIK test dataset
def load_ookpik_test_data(test_data_file):
    with open(test_data_file, 'r') as f:
        test_data = [json.loads(line) for line in f]
    return test_data

# Main function to process the dataset and store results
def process_ookpik_test_data(test_data_file, images_dir, output_file):
    test_data = load_ookpik_test_data(test_data_file)
    results = []

    for item in tqdm(test_data, desc="Processing OOKPIK Test Data"):
        image_path = os.path.join(images_dir, item['img_local_path'])
        caption1 = item['caption1']
        caption2 = item['caption2']
        
        image_features, text_features1 = extract_features(image_path, caption1)
        image_features, text_features2 = extract_features(image_path, caption2)
        
        image_features = ensure_correct_dimensions(image_features, 512)
        text_features1 = ensure_correct_dimensions(text_features1, 512)
        text_features2 = ensure_correct_dimensions(text_features2, 512)
        
        cosine_similarity1, poly_similarity1 = compute_similarity_with_chan(image_features, text_features1)
        cosine_similarity2, poly_similarity2 = compute_similarity_with_chan(image_features, text_features2)
        
        scaled_cosine_similarity1 = scale_score(cosine_similarity1).item()
        scaled_poly_similarity1 = scale_score(poly_similarity1).item()
        scaled_cosine_similarity2 = scale_score(cosine_similarity2).item()
        scaled_poly_similarity2 = scale_score(poly_similarity2).item()
        
        results.append({
            'image_path': image_path,
            'caption1': caption1,
            'caption2': caption2,
            'cosine_similarity1': cosine_similarity1,
            'scaled_cosine_similarity1': scaled_cosine_similarity1,
            'poly_similarity1': poly_similarity1,
            'scaled_poly_similarity1': scaled_poly_similarity1,
            'cosine_similarity2': cosine_similarity2,
            'scaled_cosine_similarity2': scaled_cosine_similarity2,
            'poly_similarity2': poly_similarity2,
            'scaled_poly_similarity2': scaled_poly_similarity2,
            'context_label': item.get('context_label', -1)  # Use a default value of -1 if 'context_label' is missing
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Define paths
test_data_file = "C:/Users/Pratik Senapati/Desktop/Code/Cybersecurity project/OOKPIK-Dataset/test_data.json"
images_dir = "C:/Users/Pratik Senapati/Desktop/Code/Cybersecurity project/OOKPIK-Dataset"
output_file = "ookpik_test_results.json"

# Process the dataset and store results
process_ookpik_test_data(test_data_file, images_dir, output_file)