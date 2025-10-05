import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Config
COLLECTION_NAME = "docling_chunks"
QDRANT_URL = "http://localhost:6333"  # Or your Qdrant cloud URL
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

model = load_model()

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL)

# Streamlit UI
st.title("📚 Document QA with Qdrant + Chonkie + Docling")

query = st.text_input("Enter your question or query:")
top_k = st.slider("Top K Results", 1, 10, 3)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        # Embed the query
        query_vector = model.encode(query).tolist()

        # Perform similarity search
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

        if results:
            st.success(f"Found {len(results)} relevant chunks")
            for i, res in enumerate(results):
                st.markdown(f"### Chunk #{i+1} (Score: {res.score:.2f})")
                st.write(res.payload['text'])
        else:
            st.warning("No relevant chunks found.")
