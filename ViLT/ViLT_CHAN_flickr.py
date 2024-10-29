import torch
from PIL import Image
import numpy as np
import json
from transformers import ViltProcessor, ViltModel
from tqdm import tqdm  # For progress bar
import csv

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the ViLT model and processor
vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Function to extract features using ViLT
def extract_features(image_path, caption):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = vilt_processor(text=[caption], images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = vilt_model(**inputs)
        embeddings = outputs.last_hidden_state

    # Separate image and text embeddings
    image_features = embeddings[:, :inputs['pixel_values'].shape[1], :]
    text_features = embeddings[:, inputs['pixel_values'].shape[1]:, :]

    # Normalize features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    return image_features, text_features

# Function to compute similarity using mean pooling
def compute_similarity_with_pooling(image_features, text_features, alpha=1.0, c=1.0, d=3):
    # Apply mean pooling
    image_features_pooled = image_features.mean(dim=1)
    text_features_pooled = text_features.mean(dim=1)

    # Compute cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features_pooled, text_features_pooled, dim=-1).item()
    
    # Compute polynomial kernel similarity
    poly_similarity = (alpha * cosine_similarity + c) ** d
    
    return cosine_similarity, poly_similarity

# Function to compute polynomial similarity
def compute_polynomial_similarity(cosine_similarity, alpha=1.0, c=1.0, d=3):
    return (alpha * cosine_similarity + c) ** d

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
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                filename, caption = row
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
        
        # Compute similarities using ViLT
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features.mean(dim=1), text_features.mean(dim=1), dim=-1).item()
        poly_similarity = compute_polynomial_similarity(cosine_similarity)
        
        scaled_cosine_similarity = scale_score(cosine_similarity).item()
        scaled_poly_similarity = scale_score(poly_similarity).item()
        
        # Compute similarities using ViLT with mean pooling
        pooled_cosine_similarity, pooled_poly_similarity = compute_similarity_with_pooling(image_features, text_features)
        
        scaled_pooled_cosine_similarity = scale_score(pooled_cosine_similarity).item()
        scaled_pooled_poly_similarity = scale_score(pooled_poly_similarity).item()
        
        results.append({
            'image_path': image_path,
            'caption': caption,
            'cosine_similarity': cosine_similarity,
            'scaled_cosine_similarity': scaled_cosine_similarity,
            'poly_similarity': poly_similarity,
            'scaled_poly_similarity': scaled_poly_similarity,
            'pooled_cosine_similarity': pooled_cosine_similarity,
            'scaled_pooled_cosine_similarity': scaled_pooled_cosine_similarity,
            'pooled_poly_similarity': pooled_pooled_poly_similarity,
            'scaled_pooled_poly_similarity': scaled_pooled_poly_similarity
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Define paths
flickr_images_dir = "C:/Users/Pratik Senapati/Downloads/Flickr8k_Dataset/Flicker8k_Dataset"
flickr_captions_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr8k.token.txt"
test_images_file = "C:/Users/Pratik Senapati/Downloads/Flickr8k_text/Flickr_8k.testImages.txt"
output_file = "flickr8k_test_results_vilt_pooling.json"

# Process the dataset and store results
process_flickr8k_test_data(test_images_file, flickr_captions_file, output_file)