import os
import time
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
folder = os.path.join(os.path.dirname(__file__), "ResearchImages", "UniqueImages")

# --- CONFIG ---
IMAGE_FOLDER = folder
INDEX_NAME = "image-embeds"
# ---------------

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Wait until index is ready
while not pc.describe_index(INDEX_NAME).status["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(2)

index = pc.Index(INDEX_NAME)

# Load SigLIP model
model = SentenceTransformer("google/siglip-base-patch16-256-multilingual")

# ---------------------------
# Utility to add one image
# ---------------------------
def add_image_to_pinecone(image_path: str):
    """
    Encodes an image and adds it to Pinecone with metadata:
    - filename
    - date added
    - image size (width x height)
    """
    if not os.path.exists(image_path):
        print(f"[!] File not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    embedding = model.encode(image).tolist()
    width, height = image.size
    date_added = datetime.utcnow().isoformat()

    metadata = {
        "file": os.path.basename(image_path),
        "date_added": date_added,
        "size": f"{width}x{height}"
    }

    index.upsert([
        {
            "id": os.path.basename(image_path),
            "values": embedding,
            "metadata": metadata
        }
    ])
    print(f"✅ Added {image_path} to Pinecone with metadata {metadata}")


# ---------------------------
# Search function
# ---------------------------
def search_images(query_text, top_k=5):
    query_vector = model.encode(query_text).tolist()
    results = index.query(
        queries=[query_vector],
        top_k=top_k,
        include_metadata=True
    )

    print(f"\nTop {top_k} matches for: '{query_text}'\n")
    for match in results['results'][0]['matches']:
        score = round(match['score'], 4)
        filename = match['metadata']['file']
        date_added = match['metadata'].get("date_added", "N/A")
        size = match['metadata'].get("size", "N/A")
        print(f"{score}  ->  {filename} | Added: {date_added} | Size: {size}")

    return results


# ---------------------------
# Add all images in folder
# ---------------------------
for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(("png", "jpg", "jpeg", "webp")):
        continue

    path = os.path.join(IMAGE_FOLDER, filename)
    add_image_to_pinecone(path)

print("✅ Uploaded all images in folder to Pinecone with metadata!")
