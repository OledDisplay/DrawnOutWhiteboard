import os
import time
from PIL import Image
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
IMAGE_FOLDER = "/ResearchImages/UniqueImages"
INDEX_NAME = "image-embeds"
# ---------------

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

while not pc.describe_index(INDEX_NAME).status["ready"]:
    print("Waiting for index to be ready...")
    time.sleep(2)

index = pc.Index(INDEX_NAME)

model = SentenceTransformer("google/siglip-base-patch16-256-multilingual")

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
        print(f"{score}  ->  {filename}")

    return results

# Sanity check: SigLIP outputs 768-dimensional vectors
dimension = model.get_sentence_embedding_dimension()
print("Embedding dimension:", dimension)
# Make sure your Pinecone index uses "dimension = 768"

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(("png", "jpg", "jpeg", "webp")):
        continue

    path = os.path.join(IMAGE_FOLDER, filename)

    image = Image.open(path).convert("RGB")

    embedding = model.encode(image).tolist()

    index.upsert([
        {
            "id": filename,
            "values": embedding,
            "metadata": {"file": filename}
        }
    ])

print("Uploaded all SigLIP embeddings to Pinecone!")
