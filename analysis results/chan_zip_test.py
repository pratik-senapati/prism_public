from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from chan_test.chan import CrossModalHierarchicalAttentionNetwork  # Import the simplified CHAN model
import os
import zipfile
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACT_FOLDER'] = 'extracted'

# Ensure the upload and extract folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)

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
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[..., :3]

    image = Image.fromarray((image_np * 255).astype(np.uint8))
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    return image_features, text_features

# Function to compute similarity
def compute_similarity(image_features, text_features):
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)
    return cosine_similarity.item()

# Function to normalize and scale scores
def normalize_and_scale(score, avg, std, scale_factor=1000):
    normalized_score = (score - avg) / std
    scaled_score = (normalized_score + 3) / 6 * scale_factor  # Assuming a range of -3 to 3 for normalized scores
    return max(0, min(scaled_score, scale_factor))

# Function to determine grade based on scaled score
def determine_grade(scaled_score):
    if scaled_score >= 900:
        return "A"
    elif scaled_score >= 800:
        return "B"
    elif scaled_score >= 700:
        return "C"
    elif scaled_score >= 600:
        return "D"
    else:
        return "F"

# Define average and standard deviation for polynomial similarities
avg_poly_similarity = 2.3241
std_poly_similarity = 0.2892

@app.route('/')
def index():
    return render_template('index_zip.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'zipfile' not in request.files:
        return jsonify({'error': 'No zip file provided'}), 400

    zip_file = request.files['zipfile']
    if zip_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
    zip_file.save(zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(app.config['EXTRACT_FOLDER'])

    # Load the captions from the JSON file
    captions_path = os.path.join(app.config['EXTRACT_FOLDER'], 'captions.json')
    with open(captions_path, 'r') as f:
        captions = json.load(f)

    results = []

    # Process each image and its corresponding caption
    for image_name, caption in captions.items():
        image_path = os.path.join(app.config['EXTRACT_FOLDER'], 'images', image_name)

        # Extract features
        image_features, text_features = extract_features(image_path, caption)

        # Compute similarities
        cosine_similarity = compute_similarity(image_features, text_features)
        poly_similarity = (cosine_similarity * 2 + 1) ** 3  # Example polynomial similarity calculation

        # Normalize and scale similarities
        scaled_poly_similarity = normalize_and_scale(poly_similarity, avg_poly_similarity, std_poly_similarity)

        # Determine grade
        grade = determine_grade(scaled_poly_similarity)

        results.append({
            'image': image_name,
            'caption': caption,
            'poly_similarity': poly_similarity,
            'scaled_poly_similarity': scaled_poly_similarity,
            'grade': grade
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)