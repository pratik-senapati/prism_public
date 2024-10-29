from flask import Flask, request, jsonify, render_template
from transformers import ViltProcessor, ViltModel
import torch
from PIL import Image
import os
import zipfile
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the ViLT model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

# Function to extract features using ViLT
def extract_features_vilt(image_path, caption):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        image_text_features = outputs.last_hidden_state

    return image_text_features

# Function to compute similarity using ViLT
def compute_similarity_vilt(features1, features2):
    similarity = cosine_similarity(features1.mean(dim=1).numpy(), features2.mean(dim=1).numpy())
    return similarity.item()

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

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

    # Read captions
    captions_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captions.txt')
    with open(captions_path, 'r') as f:
        captions = f.readlines()

    results = []
    images_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    for i, caption in enumerate(captions):
        image_path = os.path.join(images_folder, f'image{i+1}.jpg')
        if not os.path.exists(image_path):
            continue

        # Extract features using ViLT
        image_text_features = extract_features_vilt(image_path, caption.strip())

        # Compute similarities
        poly_similarity = (compute_similarity_vilt(image_text_features, image_text_features) * 2 + 1) ** 3  # Example polynomial similarity calculation

        # Normalize and scale similarities
        scaled_poly_similarity = normalize_and_scale(poly_similarity, avg_poly_similarity, std_poly_similarity)

        # Determine grade
        grade = determine_grade(scaled_poly_similarity)

        results.append({
            'image': f'image{i+1}.jpg',
            'poly_similarity': poly_similarity,
            'scaled_poly_similarity': scaled_poly_similarity,
            'grade': grade
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)