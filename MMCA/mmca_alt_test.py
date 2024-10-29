import torch
from PIL import Image
import numpy as np
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from mmca.main import MultiModalCausalAttention

# Load the ResNet model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Load the BERT model and tokenizer
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model.eval()

# Initialize the MMCA model
attn = MultiModalCausalAttention(dim=768, heads=8)

# Define the path to the sample image and caption
image_path = "animals.jpeg"
accurate_caption = "Breathe in the tranquility of farm life through this stunning painting of happy animals frolicking in a sun-kissed field. Every brush stroke tells a story! 🎨🐖🌞 #AnimalArt #NatureLovers #ArtisticExpression"
inaccurate_caption = "several farm animals like pigs and horses and dogs and cows that are happy and playing in a field"

# Function to extract image features using ResNet
def extract_image_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet_model(image)
    return features

# Function to extract text features using BERT
def extract_text_features(caption):
    inputs = bert_tokenizer(caption, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return features

# Extract features for accurate and inaccurate captions
image_features_accurate = extract_image_features(image_path)
text_features_accurate = extract_text_features(accurate_caption)
image_features_inaccurate = extract_image_features(image_path)
text_features_inaccurate = extract_text_features(inaccurate_caption)

# Ensure the features have the correct dimensions
def ensure_correct_dimensions(features, target_dim):
    if features.shape[-1] != target_dim:
        features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(0), target_dim).squeeze(0)
    return features

image_features_accurate = ensure_correct_dimensions(image_features_accurate, 768)
text_features_accurate = ensure_correct_dimensions(text_features_accurate, 768)
image_features_inaccurate = ensure_correct_dimensions(image_features_inaccurate, 768)
text_features_inaccurate = ensure_correct_dimensions(text_features_inaccurate, 768)

# Function to apply MMCA and compute similarity
def compute_similarity_with_mmca(image_features, text_features):
    # Ensure the features have the same dimension
    image_features = image_features.view(1, -1, 768)  # Reshape if necessary
    text_features = text_features.view(1, -1, 768)  # Reshape if necessary

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