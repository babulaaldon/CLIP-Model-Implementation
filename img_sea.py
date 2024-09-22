import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set the path to the specific image file
image_path = r"C:\searcg engine\imgs\IMG_6321.JPG"  # Updated path

# Function to preprocess and embed a single image
def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

# Function to embed text query
def get_text_embedding(text):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features.cpu().numpy()

# Create an image embedding for the single image
def create_single_image_embedding(image_path):
    image_embedding = get_image_embedding(image_path)
    return image_embedding, image_path

# Function to search for similar images given a text query
def search_image(query, image_embedding, image_path):
    text_embedding = get_text_embedding(query)
    
    # Compute cosine similarity
    similarity = cosine_similarity(text_embedding, image_embedding)[0][0]
    
    return [(image_path, similarity)]

# Example usage:
if __name__ == "__main__":
    # Create image embedding for the specific image
    image_embedding, image_path = create_single_image_embedding(image_path)

    # Search for similar images given a query
    query = "a cat on a table"  # Update this query as needed
    results = search_image(query, image_embedding, image_path)

    # Display results
    for path, score in results:
        print(f"Image: {path}, Similarity: {score}")
