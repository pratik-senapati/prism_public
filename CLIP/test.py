from fastapi import UploadFile
from PIL import Image
import io
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import asyncio

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the BLIP model and processor for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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
    
    # Generate a caption using the BLIP model
    blip_inputs = blip_processor(image, return_tensors="pt")
    blip_outputs = blip_model.generate(**blip_inputs)
    generated_caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
    
    print(f"Generated Caption: {generated_caption}")
    
    # Preprocess the image and text with padding
    inputs = clip_processor(text=[generated_caption], images=image, return_tensors="pt", padding=True)
    
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
    
    print({
        "generated_caption": generated_caption,
        "cosine_similarity_score": cosine_similarity_score
    })

# Run the test function
asyncio.run(test_compare_image_caption())