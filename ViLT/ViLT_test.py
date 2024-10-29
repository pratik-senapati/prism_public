from flask import Flask, request, jsonify, render_template
from transformers import ViltProcessor, ViltModel
import torch
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

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
        embeddings = outputs.last_hidden_state

    # Separate image and text embeddings
    image_features = embeddings[:, :inputs['pixel_values'].shape[1], :]
    text_features = embeddings[:, inputs['pixel_values'].shape[1]:, :]

    return image_features, text_features

# Function to compute cosine similarity using ViLT
def compute_cosine_similarity(image_features, text_features):
    cosine_sim = sklearn_cosine_similarity(image_features.mean(dim=1).numpy(), text_features.mean(dim=1).numpy())
    return cosine_sim.item()

# Function to compute polynomial similarity using ViLT
def compute_polynomial_similarity(cosine_sim):
    poly_similarity = (cosine_sim * 2 + 1) ** 3  # Example polynomial similarity calculation
    return poly_similarity

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

# Define average and standard deviation for cosine and polynomial similarities
avg_cosine_similarity = 0.3223
std_cosine_similarity = 0.0553
avg_poly_similarity = 2.3241
std_poly_similarity = 0.2892

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    caption = request.form['caption']

    # Extract features using ViLT
    image_features, text_features = extract_features_vilt(image_path, caption)

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(image_features, text_features)
    print(f"Cosine Similarity: {cosine_sim}")  # Debug print

    # Compute polynomial similarity
    poly_similarity = compute_polynomial_similarity(cosine_sim)
    print(f"Polynomial Similarity: {poly_similarity}")  # Debug print

    # Normalize and scale similarities
    scaled_cosine_similarity = normalize_and_scale(cosine_sim, avg_cosine_similarity, std_cosine_similarity)
    scaled_poly_similarity = normalize_and_scale(poly_similarity, avg_poly_similarity, std_poly_similarity)

    # Determine grade based on scaled polynomial similarity
    grade = determine_grade(scaled_poly_similarity)

    result = {
        'cosine_similarity': cosine_sim,
        'scaled_cosine_similarity': scaled_cosine_similarity,
        'poly_similarity': poly_similarity,
        'scaled_poly_similarity': scaled_poly_similarity,
        'grade': grade
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5500)