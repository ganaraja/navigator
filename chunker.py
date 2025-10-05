from chonkie import TokenChunker, RecursiveChunker, SemanticChunker
# or import whichever chunker you want

def chunk_text(text: str, chunker):
    """
    text: full document text
    chunker: a Chonkie chunker instance
    returns: list of chunk objects (or chunk.text)
    """
    chunks = chunker.chunk(text)
    return chunks

# Suppose you have docling output:
# docling_output = parser.parse_bytes(...)  
# and you get combined text:
text = docling_output.get_full_text()  # or join all pages

# Choose a chunker
# e.g. simple token-based
token_chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
token_chunks = chunk_text(text, token_chunker)

for c in token_chunks:
    print("Chunk:", c.text)
    print("Token count:", c.token_count)

# Or use RecursiveChunker
rec_chunker = RecursiveChunker()
rec_chunks = chunk_text(text, rec_chunker)

# Or semantic (if installed)
sem_chunker = SemanticChunker(embedding_model="minishlab/potion-base-8M", threshold=0.5, chunk_size=512)
sem_chunks = chunk_text(text, sem_chunker)

for c in sem_chunks:
    print("Chunk:", c.text)
    print("Tokens:", c.token_count)
    # you might also get properties like c.sentences, etc. :contentReference[oaicite:2]{index=2}
