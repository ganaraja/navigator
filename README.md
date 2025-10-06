Navigator
Could you clarify exactly what you mean by “books from the Open Library AI”? Do you mean:

books indexed in the Open Library (openlibrary.org)

“AI‑generated books” (books produced by AI)

books retrieved via an Open Library API that you want to parse with Docling

Assuming you meant: fetch metadata or content of books via Open Library APIs (or via their data dumps) and then use Docling to parse documents (PDFs, epubs, etc.), here’s how you could do that:

Outline of Approach

Fetch book metadata / identifiers from Open Library Use Open Library’s APIs (Search, Works / Editions, etc.) to get identifiers (e.g. editions, download links, URLs). Open Library +2 Open Library +2

Retrieve the actual document files If Open Library (or Internet Archive) hosts the book’s full text (PDF, ePub, scanned), download the file.

Use Docling to parse those document files Feed the files to Docling (or its API / library) to extract structured content (text, tables, images). ds4sd.github.io +3 docling.ai +3 GitHub +3

Combine metadata + parsed content You can then join metadata from Open Library with parsed content (chapters, sections) from Docling for your downstream AI or RAG pipeline.

Caveats & Considerations

Not all books are freely available to download via Open Library. Many are only metadata.

Docling’s strength is in converting document files (PDF, DOCX, etc.) into structured representation. It’s not directly a metadata parser.

For large scale, handle rate limits, batching, caching, and error cases.

What is Chonkie & chunkers

Chonkie is a lightweight text‑chunking library for RAG use cases. PyPI +2 Chonkie +2

It offers different chunkers, e.g.:

TokenChunker — chunk based on token count

RecursiveChunker — hierarchical splitting

SemanticChunker — chunking based on semantic similarity (requires embedding) Chonkie +1

So depending on how “smart” you want your chunk boundaries to be, you can pick one.

Putting it into a full pipeline with Docling

Use Docling to parse PDF / DOCX → structured pages, text, etc.

Concatenate or appropriately concatenate page texts into a single continuous document string (or maintain page separators).

Feed the text into Chonkie chunker.

Get a list of chunks with metadata (token counts, original offsets, etc.).

Use those chunks for embedding + indexing / retrieval.

To complete your pipeline — from Docling → Chonkie → embeddings → Qdrant VectorDB — here's a full Python example that:

✅ Overview

Parses documents (from PDF or similar) using Docling

Chunks the text using Chonkie

Creates embeddings using a model (e.g., sentence-transformers, Hugging Face, or OpenAI)

Stores chunks and their embeddings in a Qdrant collection (local or remote)

✅ Install the required packages

Make sure you have the required packages installed:

pip install chonkie qdrant-client sentence-transformers docling

If you're using embeddings from OpenAI, also install:

pip install openai

############

Notes, limitations, and optional improvements
This app generates embeddings locally using sentence-transformers. That keeps data local and fast for moderate-sized documents.

The app stores vectors in Qdrant. The upsert currently tries to recreate the collection—customize that logic if you want to append to an existing collection instead of recreating it.

The code shows robust fallbacks if docling or chonkie aren't available. If the real docling/chonkie APIs differ, you may need to tweak the small adapter functions parse_pdf_bytes and chunk_text_with_chonkie.

I didn't wire a generative LLM to produce summarized answers; the app returns the relevant chunks. If you want model-based answers (e.g., synthesize an answer from the chunks), we can add local LLM integration (e.g., llama.cpp bindings) or an API call to an LLM you choose.

If the PDF is scanned images (no selectable text), you'll need OCR (Tesseract) before parsing.

If you'd like, I can:

Add an option to append instead of recreate collection.

Add a simple built-in extractive QA step that ranks sentences within top chunks (or uses a local LLM to synthesize an answer).

Produce a Dockerfile to run the Streamlit app together with Qdrant.

Tell me which of those you'd like and I’ll add it directly into the code.Navigator
Could you clarify exactly what you mean by “books from the Open Library AI”? Do you mean:

books indexed in the Open Library (openlibrary.org)

“AI‑generated books” (books produced by AI)

books retrieved via an Open Library API that you want to parse with Docling

Assuming you meant: fetch metadata or content of books via Open Library APIs (or via their data dumps) and then use Docling to parse documents (PDFs, epubs, etc.), here’s how you could do that:

Outline of Approach

Fetch book metadata / identifiers from Open Library Use Open Library’s APIs (Search, Works / Editions, etc.) to get identifiers (e.g. editions, download links, URLs). Open Library +2 Open Library +2

Retrieve the actual document files If Open Library (or Internet Archive) hosts the book’s full text (PDF, ePub, scanned), download the file.

Use Docling to parse those document files Feed the files to Docling (or its API / library) to extract structured content (text, tables, images). ds4sd.github.io +3 docling.ai +3 GitHub +3

Combine metadata + parsed content You can then join metadata from Open Library with parsed content (chapters, sections) from Docling for your downstream AI or RAG pipeline.

Caveats & Considerations

Not all books are freely available to download via Open Library. Many are only metadata.

Docling’s strength is in converting document files (PDF, DOCX, etc.) into structured representation. It’s not directly a metadata parser.

For large scale, handle rate limits, batching, caching, and error cases.

What is Chonkie & chunkers

Chonkie is a lightweight text‑chunking library for RAG use cases. PyPI +2 Chonkie +2

It offers different chunkers, e.g.:

TokenChunker — chunk based on token count

RecursiveChunker — hierarchical splitting

SemanticChunker — chunking based on semantic similarity (requires embedding) Chonkie +1

So depending on how “smart” you want your chunk boundaries to be, you can pick one.

Putting it into a full pipeline with Docling

Use Docling to parse PDF / DOCX → structured pages, text, etc.

Concatenate or appropriately concatenate page texts into a single continuous document string (or maintain page separators).

Feed the text into Chonkie chunker.

Get a list of chunks with metadata (token counts, original offsets, etc.).

Use those chunks for embedding + indexing / retrieval.

To complete your pipeline — from Docling → Chonkie → embeddings → Qdrant VectorDB — here's a full Python example that:

✅ Overview

Parses documents (from PDF or similar) using Docling

Chunks the text using Chonkie

Creates embeddings using a model (e.g., sentence-transformers, Hugging Face, or OpenAI)

Stores chunks and their embeddings in a Qdrant collection (local or remote)

✅ Install the required packages

Make sure you have the required packages installed:

pip install chonkie qdrant-client sentence-transformers docling

If you're using embeddings from OpenAI, also install:

pip install openai

############

Notes, limitations, and optional improvements
This app generates embeddings locally using sentence-transformers. That keeps data local and fast for moderate-sized documents.

The app stores vectors in Qdrant. The upsert currently tries to recreate the collection—customize that logic if you want to append to an existing collection instead of recreating it.

The code shows robust fallbacks if docling or chonkie aren't available. If the real docling/chonkie APIs differ, you may need to tweak the small adapter functions parse_pdf_bytes and chunk_text_with_chonkie.

I didn't wire a generative LLM to produce summarized answers; the app returns the relevant chunks. If you want model-based answers (e.g., synthesize an answer from the chunks), we can add local LLM integration (e.g., llama.cpp bindings) or an API call to an LLM you choose.

If the PDF is scanned images (no selectable text), you'll need OCR (Tesseract) before parsing.

If you'd like, I can:

Add an option to append instead of recreate collection.

Add a simple built-in extractive QA step that ranks sentences within top chunks (or uses a local LLM to synthesize an answer).

Produce a Dockerfile to run the Streamlit app together with Qdrant.

Tell me which of those you'd like and I’ll add it directly into the code.
