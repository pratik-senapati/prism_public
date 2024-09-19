from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/compare")
async def compare_image_caption(file: UploadFile = File(...), reference_caption: str = ""):
    image = Image.open(io.BytesIO(await file.read()))
    
    # Preprocess the image and text
    inputs = processor(text=[reference_caption], images=image, return_tensors="pt", padding=True).to("cuda")
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
    
    # Calculate cosine similarity
    similarity_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()
    
    return {
        "reference_caption": reference_caption,
        "similarity_score": similarity_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
