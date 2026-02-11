# JSON RAG with ChromaDB

RAG system for structured book content using ChromaDB and section-based chunking.

## Features

- Section-based semantic chunking
- ChromaDB local vector storage
- Metadata filtering by chapter
- OpenAI embeddings

## Technical Details

- Chunking Strategy: Section-based (not character-based)
- Vector DB: ChromaDB
- Embeddings: OpenAI text-embedding-ada-002
- Metadata: chapter_num, chapter_title, section_title, page

## Results

Query: "What is the difference between Qdrant and Pinecone?"
- Found correct sections from Chapter 2
- Accurate retrieval with metadata

Metadata Filtering Test:
- Query: "chunking" + Filter: Chapter 3
- Only returned results from Chapter 3
- Precise filtering working correctly

