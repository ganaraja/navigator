# app.py
import os
import tempfile
from typing import List, Dict, Any

import streamlit as st

# Try the expected libs; provide helpful messages if missing.
try:
    import docling
except Exception:
    docling = None

try:
    import chonkie
except Exception:
    chonkie = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
except Exception:
    QdrantClient = None
    rest_models = None

# Fallback parsing if docling is not available
from io import BytesIO
from typing import Tuple

# PDF fallback
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Basic config
st.set_page_config(page_title="Doc QA (Docling + Chonkie + Qdrant)", layout="wide")

st.title("📄 Document QA — Docling + Chonkie + Local Embeddings + Qdrant")
st.caption("Upload a PDF, parse it, chunk it, embed locally, store in Qdrant, and ask questions.")

# Sidebar: connection settings for Qdrant
st.sidebar.header("Qdrant & settings")

QDRANT_HOST = st.sidebar.text_input("Qdrant Host", value=os.getenv("QDRANT_HOST", "localhost"))
QDRANT_PORT = st.sidebar.text_input("Qdrant Port", value=os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = st.sidebar.text_input("Qdrant API Key (if any)", value=os.getenv("QDRANT_API_KEY", ""))
COLLECTION_NAME = st.sidebar.text_input("Collection name", value=os.getenv("QDRANT_COLLECTION", "documents"))
EMBEDDING_MODEL_NAME = st.sidebar.text_input("SentenceTransformer model", value=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
TOP_K = st.sidebar.slider("Top K results", 1, 10, 4)

# Helper messages about login
st.sidebar.markdown(
    """
    **Note:** I can't log into your Qdrant instance from here.
    - If Qdrant runs locally (Docker), use host `localhost` and port `6333`.
    - If Qdrant Cloud, put host and API key here.
    """
)

# Caching heavy objects
@st.cache_resource(show_spinner=False)
def load_embedding_model(name: str):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. See requirements.txt")
    model = SentenceTransformer(name)
    return model

@st.cache_resource(show_spinner=False)
def get_qdrant_client(host: str, port: str, api_key: str = None):
    if QdrantClient is None:
        raise ImportError("qdrant-client is not installed. See requirements.txt")
    # Try different connection modes depending on whether API key provided.
    if api_key:
        client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
    else:
        # local / no auth
        client = QdrantClient(url=f"http://{host}:{port}")
    return client

# Fallback chunker (if chonkie not available)
def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Parse PDF using docling if available, else use pdfplumber
def parse_pdf_bytes(file_bytes: bytes) -> str:
    # Prefer docling
    if docling is not None:
        try:
            # Many docling versions may expose different API; try common patterns
            try:
                # If docling has a parse_pdf_bytes or Document.from_bytes
                if hasattr(docling, "parse_pdf_bytes"):
                    text = docling.parse_pdf_bytes(file_bytes)
                    return text
                if hasattr(docling, "Document") and hasattr(docling.Document, "from_bytes"):
                    doc = docling.Document.from_bytes(file_bytes)
                    return doc.text
                # some docling accept bytes -> parser
                if hasattr(docling, "parse"):
                    doc = docling.parse(file_bytes)
                    if isinstance(doc, str):
                        return doc
                    if hasattr(doc, "text"):
                        return doc.text
            except Exception:
                # Fall through to other attempts
                pass
        except Exception:
            pass

    # Fallback: pdfplumber
    if pdfplumber is None:
        raise ImportError("Neither docling nor pdfplumber are installed. See requirements.txt")

    text_pages = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text()
            if txt:
                text_pages.append(txt)
    return "\n\n".join(text_pages)


def chunk_text_with_chonkie(full_text: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: [{"id": str, "text": "...", "meta": {...}}]
    Uses chonkie if available, otherwise simple chunker.
    """
    if chonkie is not None:
        try:
            # Try typical chonkie usage pattern (hypothetical)
            # Example: chonkie.Chunker(chunk_size=..., overlap=...).chunk(text)
            if hasattr(chonkie, "Chunker"):
                chunker = chonkie.Chunker(chunk_size=500, overlap=50)
                chunks = chunker.chunk(full_text)
                result = []
                for i, c in enumerate(chunks):
                    result.append({"id": f"chunk_{i}", "text": c, "meta": {"chunk_index": i}})
                return result
            # else try chonkie.chunk_text
            if hasattr(chonkie, "chunk_text"):
                chunks = chonkie.chunk_text(full_text, size=500, overlap=50)
                return [{"id": f"chunk_{i}", "text": c, "meta": {"chunk_index": i}} for i, c in enumerate(chunks)]
        except Exception:
            # fallback to simple
            pass

    # fallback
    chunks = simple_chunk_text(full_text, chunk_size=500, overlap=50)
    return [{"id": f"chunk_{i}", "text": c, "meta": {"chunk_index": i}} for i, c in enumerate(chunks)]


def upsert_chunks_to_qdrant(client: QdrantClient, collection_name: str, embeddings: List[List[float]], chunks: List[Dict[str, Any]], distance: str = "Cosine"):
    """
    Create collection (if not exists) and upsert vectors with metadata.
    """
    vector_size = len(embeddings[0])
    # Create collection if not exists
    try:
        # collection schema
        if rest_models is None:
            raise ImportError("qdrant-client models missing")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE)
        )
    except Exception as e:
        # If recreate_collection fails (because it exists and you don't want to recreate), try create
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE)
            )
        except Exception:
            # ignore if exists
            pass

    # Prepare points
    points = []
    for emb, chunk in zip(embeddings, chunks):
        points.append(
            rest_models.PointStruct(
                id=chunk["id"],
                vector=emb,
                payload={"text": chunk["text"], "meta": chunk.get("meta", {})}
            )
        )
    # Upsert
    client.upsert(collection_name=collection_name, points=points)


def search_qdrant(client: QdrantClient, collection_name: str, query_embedding: List[float], top_k: int = 4):
    res = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
    )
    # res is a list of ScoredPoint
    out = []
    for item in res:
        payload = item.payload or {}
        text = payload.get("text") or ""
        meta = payload.get("meta") or {}
        out.append({"id": item.id, "score": float(item.score), "text": text, "meta": meta})
    return out


# Begin UI interaction
uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to get started. Example: academic paper, report, or contract.")
else:
    st.session_state.setdefault("last_file_name", uploaded.name)
    st.write(f"**File:** {uploaded.name} — size: {uploaded.size} bytes")

    parse_button = st.button("Parse & Index PDF into Qdrant")
    if parse_button:
        with st.spinner("Parsing PDF..."):
            try:
                file_bytes = uploaded.read()
                full_text = parse_pdf_bytes(file_bytes)
                if not full_text or not full_text.strip():
                    st.error("No text could be extracted from the PDF.")
                    raise RuntimeError("Empty extracted text")
            except Exception as e:
                st.exception(f"Error parsing PDF: {e}")
                st.stop()

        st.success("PDF parsed. Chunking...")
        chunks = chunk_text_with_chonkie(full_text)
        st.write(f"Produced {len(chunks)} chunks (approx). Showing first 3 chunks below:")
        for c in chunks[:3]:
            st.write(f"**{c['id']}** — {c['meta']}")
            st.write(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""))

        # Load embedding model
        try:
            with st.spinner(f"Loading embedding model {EMBEDDING_MODEL_NAME}..."):
                embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        except Exception as e:
            st.exception(f"Error loading embedding model: {e}")
            st.stop()

        # Compute embeddings in batches
        texts = [c["text"] for c in chunks]
        try:
            with st.spinner("Computing embeddings..."):
                embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                # Convert to list of floats
                embeddings_list = [emb.tolist() for emb in embeddings]
        except Exception as e:
            st.exception(f"Embedding error: {e}")
            st.stop()

        # Connect to Qdrant
        try:
            with st.spinner("Connecting to Qdrant..."):
                client = get_qdrant_client(QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY or None)
        except Exception as e:
            st.exception(f"Error connecting to Qdrant: {e}")
            st.stop()

        # Upsert to Qdrant
        try:
            with st.spinner("Upserting vectors to Qdrant..."):
                upsert_chunks_to_qdrant(client, COLLECTION_NAME, embeddings_list, chunks)
        except Exception as e:
            st.exception(f"Upsert error: {e}")
            st.stop()

        st.success(f"Indexed {len(chunks)} chunks into Qdrant collection '{COLLECTION_NAME}'.")

    st.divider()
    st.header("Ask questions about the document")
    query = st.text_input("Type your question here")
    ask_button = st.button("Search & Retrieve")

    if ask_button and query:
        # connect to qdrant and model again
        try:
            client = get_qdrant_client(QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY or None)
        except Exception as e:
            st.exception(f"Can't connect to Qdrant: {e}")
            st.stop()

        try:
            embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        except Exception as e:
            st.exception(f"Can't load embedding model: {e}")
            st.stop()

        with st.spinner("Embedding the query..."):
            q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

        with st.spinner("Searching Qdrant for relevant chunks..."):
            results = search_qdrant(client, COLLECTION_NAME, q_emb, top_k=TOP_K)

        if not results:
            st.warning("No results found — maybe the PDF wasn't indexed, or collection name differs.")
        else:
            st.success(f"Found {len(results)} relevant chunks (top {TOP_K} shown).")
            for i, r in enumerate(results):
                st.markdown(f"**Result #{i+1} — score: {r['score']:.4f} — id: {r['id']}**")
                st.write(r["text"])
                st.caption(f"meta: {r.get('meta')}")

            # Optionally: provide a combined context for external LLM (not included).
            st.divider()
            st.subheader("Combined context (concatenated top chunks)")
            combined = "\n\n---\n\n".join([r["text"] for r in results])
            st.write(combined[:4000] + ("..." if len(combined) > 4000 else ""))

st.sidebar.markdown("---")
st.sidebar.write("Troubleshooting / tips:")
st.sidebar.write("""
- If you get connection errors with Qdrant, verify Qdrant is running and that `QDRANT_HOST` & `QDRANT_PORT` are correct.
- To run Qdrant locally quickly:
  `docker run -p 6333:6333 -p 6334:6334 -it qdrant/qdrant`
- If `docling` or `chonkie` are not installed/fail, the app will fall back to `pdfplumber` and a simple chunker.
- For large PDFs, consider increasing memory and using batching for embeddings.
""")
