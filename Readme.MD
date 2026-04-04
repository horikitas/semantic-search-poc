# semantic-search-poc

A minimal proof-of-concept for semantic search using OpenAI embeddings and Qdrant Cloud.
This project shows how to convert text into vector embeddings, store them in a vector database, and perform semantic search without depending on an LLM chat flow.

## Introduction

This repository demonstrates a simple, production-friendly semantic search pipeline:

- `embed.py` generates semantic vectors from text using OpenAI embeddings.
- `qdrant.py` manages the vector store and communicates with Qdrant Cloud.
- `main.py` ingests data, creates embeddings, and stores them in Qdrant.
- `search.py` performs similarity search against the stored vectors.

The goal is to provide a lightweight implementation for semantic retrieval and ranking of text data.

## Flow

1. Collect or prepare text documents.
2. Generate embeddings for each document using OpenAI.
3. Upload the embeddings into Qdrant Cloud.
4. Receive a query and convert it to an embedding.
5. Search the Qdrant vector index for the nearest documents.
6. Return the most relevant text results.

## Flow Diagram

```mermaid
flowchart LR
    A[Text Documents] --> B[embed.py]
    B --> C[Qdrant Cloud]
    D[Search Query] --> E[embed.py]
    E --> F[Qdrant Cloud Search]
    F --> G[Relevant Results]
    C --> F
```

## Implementation

The implementation is organized as a small pipeline:

- `embed.py`
  - Creates and formats text embeddings via the OpenAI API.
  - Exposes reusable functions for embedding both documents and queries.

- `qdrant.py`
  - Initializes and configures the Qdrant client.
  - Creates or updates the vector collection.
  - Inserts vectors and metadata into Qdrant.
  - Performs nearest-neighbor search for query embeddings.

- `main.py`
  - Loads sample content or source data.
  - Uses `embed.py` to compute embeddings.
  - Saves embedding vectors in Qdrant through `qdrant.py`.

- `search.py`
  - Reads a search query from the user or a preset value.
  - Embeds the query text.
  - Queries Qdrant for the most semantically similar documents.
  - Displays search results.

## Quick Start

```shell
python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate        # On Windows
pip install -r requirements.txt
python main.py
```

### First time run

```shell
python main.py
```

### Subsequent search only

```shell
python search.py
```
