# app.py
import os
import tempfile
import uuid
from typing import List, Dict, Any
import json
import csv
import html
import re

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
st.set_page_config(page_title="Document Navigator", layout="wide")

# Inject custom CSS (fonts, background, cards)
def inject_custom_css():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Merriweather:wght@300;400;700&display=swap');

    :root{
      --bg: #f3f3f3;
      --panel: #ffffff;
      --muted: #6b6b6b;
      --accent: #ff9900; /* Amazon-like orange */
      --accent-dark: #e58900;
      --border: #e6e6e6;
      --card-shadow: 0 4px 10px rgba(0,0,0,0.06);
      --text: #111827;
    }

    body, .stApp {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Make the main block feel like a centered page on Amazon */
    .block-container {
      max-width: 1100px;
      margin: 20px auto;
      padding: 24px;
      background: var(--panel);
      border-radius: 6px;
      box-shadow: var(--card-shadow);
      border: 1px solid var(--border);
    }

    /* Header */
    .app-header {
      display:flex;
      gap:16px;
      align-items:center;
      padding:12px 16px;
      border-radius:6px;
      background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.4));
      border: 1px solid var(--border);
      margin-bottom:16px;
    }
    .logo {
      width:64px;
      height:40px;
      border-radius:4px;
      background: linear-gradient(90deg,var(--accent),var(--accent-dark));
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:800;
      color:#111;
      font-size:20px;
      letter-spacing: -0.5px;
    }
    .app-title {
      font-weight:700;
      font-size:18px;
      color:var(--text);
    }

    /* Cards */
    .card {
      background: #fff;
      padding: 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
      margin-bottom:12px;
    }

    /* Preview text uses a comfortable serif */
    .preview {
      font-family: 'Merriweather', Georgia, serif;
      color: #0f172a;
      line-height:1.5;
      font-size:14.5px;
      white-space:pre-wrap;
    }

    .meta {
      color: var(--muted);
      font-size:13px;
      margin-top:8px;
    }

    .combined {
      background: #fff;
      padding:12px;
      border-radius:6px;
      color:var(--text);
      border:1px solid var(--border);
    }

    /* Primary buttons — visible and Amazon-like */
    .stButton > button, button[kind="primary"] {
      background: var(--accent) !important;
      color: #111 !important;
      border: none !important;
      padding: 8px 14px !important;
      font-weight: 700 !important;
      border-radius: 4px !important;
      box-shadow: none !important;
      opacity: 1 !important;
    }
    .stButton > button:hover, button[kind="primary"]:hover {
      background: var(--accent-dark) !important;
    }

    /* Secondary appearance */
    .stButton > .stButton>button[variant="secondary"] {
      background: #fff !important;
      color: var(--text) !important;
      border: 1px solid var(--border) !important;
    }

    /* Make inputs and sidebar light and crisp */
    .css-1lcbmhc e1fqkh3o0, .stTextInput>div>div>input {
      background: #fff;
    }
    .sidebar .stBlock {
      background: transparent;
    }

    /* Small helper styles */
    .result-score {
      color: var(--accent);
      font-weight:700;
      margin-right:8px;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

inject_custom_css()

st.title("📄 Document Navigator")
st.caption("Upload a PDF, parse it, chunk it, embed locally, store in Qdrant, and ask questions.")

# Sidebar: connection settings for Qdrant
st.sidebar.header("Qdrant & settings")

QDRANT_HOST = st.sidebar.text_input("Qdrant Host", value=os.getenv("QDRANT_HOST", "localhost"))
QDRANT_PORT = st.sidebar.text_input("Qdrant Port", value=os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# st.sidebar.text_input("Qdrant API Key (if any)", value=os.getenv("QDRANT_API_KEY", ""))
# # Do not prefill or display the API key from environment — keep the field empty and masked.
# QDRANT_API_KEY = st.sidebar.text_input(
#     "Qdrant API Key (optional)",
#     value="",
#     type="password",
#     placeholder="Set via QDRANT_API_KEY env var or paste here (hidden)"
# )
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
        client = QdrantClient(url=f"https://{host}:{port}", api_key=api_key)
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


def parse_text_file(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Parse TXT or TSV/CSV bytes into chunks:
    - For TSV/CSV: each row becomes one chunk with meta containing the row dict (using header if present).
    - For TXT: split into chunks (using chonkie if available or simple chunker).
    Returns list of {"id": str, "text": str, "meta": {...}}
    """
    text_chunks: List[Dict[str, Any]] = []
    name = (filename or "").lower()

    # CSV / TSV
    if name.endswith((".tsv", ".csv")):
        try:
            s = file_bytes.decode("utf-8")
        except Exception:
            s = file_bytes.decode("latin-1", errors="ignore")
        delim = "\t" if name.endswith(".tsv") else ","
        lines = s.splitlines()
        # Try DictReader first (works if header present)
        try:
            reader = csv.DictReader(lines, delimiter=delim)
            for i, row in enumerate(reader):
                # row can be OrderedDict; stringify values
                row_text = " \n ".join([f"{k}: {v}" for k, v in row.items() if v is not None and v != ""])
                meta = {"row_index": i, "row": row}
                text_chunks.append({"id": f"row_{i}", "text": row_text or "", "meta": meta})
        except Exception:
            text_chunks = []

        # If DictReader produced nothing (no header) or failed, fallback to simple reader
        if not text_chunks:
            try:
                reader2 = csv.reader(lines, delimiter=delim)
                for i, row in enumerate(reader2):
                    row_text = " \n ".join([str(c) for c in row if c is not None and c != ""])
                    meta = {"row_index": i}
                    text_chunks.append({"id": f"row_{i}", "text": row_text or "", "meta": meta})
            except Exception:
                # last resort: entire file as one chunk
                text_chunks = [{"id": "text_0", "text": s, "meta": {"source": filename}}]

        return text_chunks

    # Plain text (.txt) or fallback
    try:
        s = file_bytes.decode("utf-8")
    except Exception:
        s = file_bytes.decode("latin-1", errors="ignore")

    if name.endswith(".txt") or not name:
        # If small file, keep as single chunk
        if len(s) < 4000:
            return [{"id": "text_0", "text": s, "meta": {"source": filename}}]
        # else chunk
        chunk_texts = chunk_text_with_chonkie(s)
        return [{"id": f"text_{i}", "text": c, "meta": {"chunk_index": i, "source": filename}} for i, c in enumerate(chunk_texts)]

    # fallback: whole content
    return [{"id": "text_0", "text": s, "meta": {"source": filename}}]


def index_chunks_and_upsert(chunks: List[Dict[str, Any]]):
    """
    Given prepared chunks (id/text/meta), compute embeddings and upsert to Qdrant.
    Uses cached model and client functions and shows Streamlit spinners/errors.
    """
    if not chunks:
        st.warning("No chunks to index.")
        return

    texts = [c.get("text", "") for c in chunks]

    # Load embedding model
    try:
        with st.spinner(f"Loading embedding model {EMBEDDING_MODEL_NAME}..."):
            embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.exception(f"Error loading embedding model: {e}")
        st.stop()

    # Compute embeddings (in-memory)
    try:
        with st.spinner("Computing embeddings..."):
            # sentence-transformers returns numpy array when convert_to_numpy=True
            embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
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

    # Upsert
    try:
        with st.spinner("Upserting vectors to Qdrant..."):
            upsert_chunks_to_qdrant(client, COLLECTION_NAME, embeddings_list, chunks)
    except Exception as e:
        st.exception(f"Upsert error: {e}")
        st.stop()

    st.success(f"Indexed {len(chunks)} chunks into Qdrant collection '{COLLECTION_NAME}'.")
# ...existing code...

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


# Add this function so render_chunk_card is available before use
def render_chunk_card(chunk: Dict[str, Any]):
    """
    Render a single chunk as a styled card in Streamlit.
    Safely handle meta (ensure dict), escape text, and show id / chunk index.
    """
    # Safely get meta
    meta = chunk.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {"meta": str(meta)}

    # Safely get chunk index and id
    chunk_index = meta.get("chunk_index", meta.get("row_index", "-"))
    chunk_id = chunk.get("id", "-")

    # Safely get and slice text
    text = chunk.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    preview = text[:1200] if text else ""

    # Escape HTML-sensitive chars
    preview_escaped = preview.replace("<", "&lt;").replace(">", "&gt;")
    
    # Stringify meta for display
    try:
        meta_str = json.dumps(meta, ensure_ascii=False)
    except Exception:
        meta_str = str(meta)

    html = f'''
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div style="font-weight:700">{chunk_id}</div>
          <div style="color:var(--muted);font-size:13px">chunk #{chunk_index}</div>
        </div>
        <div class="preview" style="margin-top:8px">{preview_escaped}</div>
        <div class="meta">Meta: {meta_str}</div>
      </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def upsert_chunks_to_qdrant(client: QdrantClient, collection_name: str, embeddings: List[List[float]], chunks: List[Dict[str, Any]], distance: str = "Cosine"):
    """
    Create collection (if not exists) and upsert vectors with metadata.
    Raises on error so callers (Streamlit UI) can show the exception.
    """
    if not embeddings:
        raise ValueError("No embeddings provided to upsert.")

    if rest_models is None:
        raise RuntimeError("qdrant_client.http.models (rest_models) not available")

    vector_size = len(embeddings[0])

    # Ensure collection exists
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Qdrant collection '{collection_name}': {e}")

    # Prepare points with valid IDs (unsigned int or UUID string)
    points = []
    for emb, chunk in zip(embeddings, chunks):
        raw_id = chunk.get("id") or chunk.get("meta", {}).get("chunk_index")
        # Determine a valid Qdrant ID: prefer integer if numeric, accept UUID strings, otherwise generate UUID4
        if raw_id is None:
            pid = str(uuid.uuid4())
        else:
            pid = None
            # try integer
            try:
                pid_int = int(raw_id)
                if pid_int >= 0:
                    pid = pid_int
            except Exception:
                pass
            if pid is None:
                # try valid UUID string
                try:
                    uuid.UUID(str(raw_id))
                    pid = str(raw_id)
                except Exception:
                    pid = str(uuid.uuid4())

        points.append(
            rest_models.PointStruct(
                id=pid,
                vector=list(emb),
                payload={"text": chunk.get("text", ""), "meta": chunk.get("meta", {})}
            )
        )

    # Upsert and raise on failure
    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as e:
        raise RuntimeError(f"Qdrant upsert failed: {e}")


def search_qdrant(client: QdrantClient, collection_name: str, query_embedding: List[float], top_k: int = 4):
    """
    Use Qdrant's query_points API with proper parameters to ensure results are returned.
    Returns: [{"id": str, "score": float, "text": str, "meta": dict}, ...]
    """
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
    except Exception as e:
        # If search fails, try query_points with specific parameters
        try:
            results = client.query_points(
                collection_name=collection_name,
                query_vector=query_embedding,
                with_payload=True,  # ensure we get text content
                with_vectors=False,  # we don't need vectors back
                limit=top_k
            )
        except Exception as e2:
            raise RuntimeError(f"Qdrant search failed: {e2}") from e2

    # Normalize results to our expected format
    out = []
    if not results:
        return out

    for point in results:
        # Handle both dict and object responses
        if isinstance(point, dict):
            point_id = point.get('id')
            score = point.get('score')
            payload = point.get('payload', {})
        else:
            point_id = getattr(point, 'id', None)
            score = getattr(point, 'score', None)
            payload = getattr(point, 'payload', {})

        # Ensure payload is a dict
        if not isinstance(payload, dict):
            payload = {}

        # Extract text and metadata
        text = payload.get('text', '')
        meta = payload.get('meta', {})

        # Build normalized result
        out.append({
            'id': str(point_id) if point_id is not None else '',
            'score': float(score) if score is not None else None,
            'text': text,
            'meta': meta
        })

    return out


# Begin UI interaction
uploaded = st.file_uploader("Upload a file (PDF / TSV / CSV / TXT)", type=["pdf", "tsv", "csv", "txt"])

if uploaded is None:
    st.info("Upload a PDF to get started. Example: academic paper, report, or contract.")
else:
    st.session_state.setdefault("last_file_name", uploaded.name)
    st.write(f"**File:** {uploaded.name} — size: {uploaded.size} bytes")

    # New: detect text/tsv/csv and provide a Parse & Index action
    parse_button = st.button("Parse & Index file into Qdrant")

    if parse_button:
        filename = uploaded.name or "uploaded"
        file_bytes = uploaded.read()
        if filename.lower().endswith(".pdf"):
            # existing PDF path
            with st.spinner("Parsing PDF..."):
                try:
                    full_text = parse_pdf_bytes(file_bytes)
                    if not full_text or not full_text.strip():
                        st.error("No text could be extracted from the PDF.")
                        raise RuntimeError("Empty extracted text")
                except Exception as e:
                    st.exception(f"Error parsing PDF: {e}")
                    st.stop()

            st.success("PDF parsed. Chunking...")
            chunks = chunk_text_with_chonkie(full_text)
            chunks = [{"id": f"chunk_{i}", "text": c, "meta": {"chunk_index": i}} for i, c in enumerate(chunks)]
            st.write(f"Produced {len(chunks)} chunks (approx). Showing first 3 chunks below:")
            for c in chunks[:3]:
                render_chunk_card(c)

            # index and upsert
            index_chunks_and_upsert(chunks)

        else:
            # TSV/CSV/TXT path
            try:
                parsed_chunks = parse_text_file(file_bytes, filename)
                if not parsed_chunks:
                    st.error("No usable content found in the uploaded text file.")
                    st.stop()
            except Exception as e:
                st.exception(f"Error parsing text file: {e}")
                st.stop()

            st.success("File parsed. Showing first 3 parsed items:")
            for c in parsed_chunks[:3]:
                render_chunk_card(c)

            # index and upsert
            index_chunks_and_upsert(parsed_chunks)

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
            st.subheader("Search Results")
            st.write(f"Top {min(len(results), TOP_K)} results from collection '{COLLECTION_NAME}':")

            def render_result_card(r: Dict[str, Any], idx: int):
                """
                Render a search result using Streamlit native components with improved text formatting.
                Shows score, text in readable paragraphs, and metadata.
                """
                # Extract and clean the text
                raw_text = r.get('text', '') or ''
                
                # Clean HTML and normalize whitespace
                clean_text = html.unescape(raw_text)
                clean_text = re.sub(r'<[^>]+>', '', clean_text)
                
                # Normalize line endings
                clean_text = re.sub(r'\r\n|\r', '\n', clean_text)
                
                # Format paragraphs:
                # 1. Split on multiple newlines
                paragraphs = re.split(r'\n\s*\n', clean_text)
                # 2. Clean each paragraph
                paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in paragraphs if p.strip()]
                
                # Format vectors/lists (lines starting with numbers, bullets, etc.)
                formatted_paragraphs = []
                for p in paragraphs:
                    # If paragraph contains list-like items, split them
                    if re.search(r'^[\d\-\•\*]\s|[\n][\d\-\•\*]\s', p):
                        items = re.split(r'(?m)^[\d\-\•\*]\s', p)
                        items = [item.strip() for item in items if item.strip()]
                        formatted_paragraphs.append('\n'.join(f"• {item}" for item in items))
                    else:
                        formatted_paragraphs.append(p)
                
                # Get score and ID
                score = r.get('score')
                score_str = f"{score:.4f}" if score is not None else "n/a"
                rid = r.get('id', '')

                # Render the card
                st.markdown(f"### Result #{idx+1}")
                
                # Score and ID in columns
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("**Score:**")
                    st.markdown(f"_{score_str}_")
                with col2:
                    st.markdown(f"**ID:** {rid}")
                
                # Show formatted text
                with st.expander("Show Text", expanded=True):
                    # Display each paragraph with spacing
                    for i, para in enumerate(formatted_paragraphs):
                        if i > 0:
                            st.markdown("---")  # visual separator between paragraphs
                        if para.startswith('•'):
                            # Format as a list
                            st.markdown(para)
                        else:
                            # Format as a regular paragraph
                            st.write(para)
                
                # Show metadata if present
                if r.get('meta'):
                    with st.expander("Metadata", expanded=False):
                        st.json(r['meta'])
                
                # Visual separator between results
                st.markdown("---")

            for i, r in enumerate(results[:TOP_K]):
                render_result_card(r, i)

            # Combined context
            combined = "\n\n---\n\n".join([r["text"] for r in results[:TOP_K]])
            st.divider()
            st.subheader("Combined context (top results)")
            st.markdown(f"<div class='combined'>{combined[:8000].replace('<','&lt;').replace('>','&gt;')}{'...' if len(combined)>8000 else ''}</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.write("Troubleshooting / tips:")
st.sidebar.write("""
- If you get connection errors with Qdrant, verify Qdrant is running and that `QDRANT_HOST` & `QDRANT_PORT` are correct.
- To run Qdrant locally quickly:
  `docker run -p 6333:6333 -p 6334:6334 -it qdrant/qdrant`
- If `docling` or `chonkie` are not installed/fail, the app will fall back to `pdfplumber` and a simple chunker.
- For large PDFs, consider increasing memory and using batching for embeddings.
""")

