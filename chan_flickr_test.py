import torch
from PIL import Image
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from chan import CrossModalHierarchicalAttentionNetwork  # Import the simplified CHAN model
from tqdm import tqdm  # For progress bar

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the CHAN model
chan_model = CrossModalHierarchicalAttentionNetwork(dim=512, heads=8)  # Using 512 dimensions

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

# Load Flickr8k test dataset
def load_flickr8k_test_data(test_images_file, captions_file):
    # Load test image filenames
    with open(test_images_file, 'r') as f:
        test_images = set(line.strip() for line in f.readlines())
    
    # Load captions
    captions = {}
    with open(captions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                filename, caption = parts
                filename = filename.split('#')[0]
                if filename in test_images:
                    if filename not in captions:
                        captions[filename] = []
                    captions[filename].append(caption)
    
    test_data = []
    for filename, caption_list in captions.items():
        for caption in caption_list:
            image_path = f"{flickr_images_dir}/{filename}"
            test_data.append((image_path, caption))
    
    return test_data

# Main function to process the dataset and store results
def process_flickr8k_test_data(test_images_file, captions_file, output_file):
    test_data = load_flickr8k_test_data(test_images_file, captions_file)
    results = []

    for image_path, caption in tqdm(test_data, desc="Processing Flickr8k Test Data"):
        image_features, text_features = extract_features(image_path, caption)
        image_features = ensure_correct_dimensions(image_features, 512)
        text_features = ensure_correct_dimensions(text_features, 512)
        
        cosine_similarity, poly_similarity = compute_similarity_with_chan(image_features, text_features)
        
        scaled_cosine_similarity = scale_score(cosine_similarity).item()
        scaled_poly_similarity = scale_score(poly_similarity).item()
        
        results.append({
            'image_path': image_path,
            'caption': caption,
            'cosine_similarity': cosine_similarity,
            'scaled_cosine_similarity': scaled_cosine_similarity,
            'poly_similarity': poly_similarity,
            'scaled_poly_similarity': scaled_poly_similarity
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Define paths
flickr_images_dir = "C:/Users/Pratik Senapati/Downloads/Flickr8k_Dataset/Flicker8k_Dataset"
flickr_captions_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr8k.token.txt"
test_images_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr_8k.testImages.txt"
output_file = "flickr8k_test_results.json"

# Process the dataset and store results
process_flickr8k_test_data(test_images_file, flickr_captions_file, output_file)