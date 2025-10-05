from chonkie import TokenChunker
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, CollectionStatus

import uuid

# STEP 1 — Extracted text (assume you already used Docling)
# Replace this with: `docling_output.get_full_text()` or similar
text = """Your full document text extracted by Docling goes here."""

# STEP 2 — Chunk the text
chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(text)

# STEP 3 — Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # Or your preferred embedding model

texts = [chunk.text for chunk in chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# STEP 4 — Connect to Qdrant
client = QdrantClient("http://localhost:6333")  # Or remote URL / API key

COLLECTION_NAME = "docling_chunks"

# STEP 5 — Create collection if it doesn't exist
if COLLECTION_NAME not in client.get_collections().collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

# STEP 6 — Upload chunks to Qdrant
points = []
for i, chunk in enumerate(chunks):
    point = PointStruct(
        id=str(uuid.uuid4()),  # or use chunk index
        vector=embeddings[i],
        payload={
            "text": chunk.text,
            "token_count": chunk.token_count,
            "chunk_index": i,
        }
    )
    points.append(point)

client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ Uploaded {len(points)} chunks to Qdrant collection '{COLLECTION_NAME}'")
