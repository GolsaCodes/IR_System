# LLM-Based Query Expansion for Dense Retrieval

The project uses the **SciFact** benchmark to compare the retrieval process before and after expanding a user's original query with an LLM (in this case, Gemini Flash 3.5)

## Overview

In this project, the pipeline is:

```text
Original Query
      │
      ▼
LLM-based Query Expansion
      │
      ▼
Expanded Query
      │
      ▼
Dense Embedding Model
      │
      ▼
Vector Retrieval
      │
      ▼
Retrieved Documents
```

## Dataset

I used the **SciFact** dataset from the BEIR benchmark.

## Method

### 1. Original Queries

The original SciFact queries are used as the starting point.

### 2. LLM Query Expansion

Each original query is provided to an LLM to generate a concise expansion containing additional relevant terminology and concepts.

Conceptually:

The LLM component is implemented using the Gemini Flash 3.5 API.

> The model is used only for query expansion. It is not responsible for document retrieval or relevance scoring.

### 3. Dense Representation

The expanded query is converted into a dense vector representation using a BGE embedding model (BAAI/bge-base-en-v1.5).

The resulting embeddings are normalized before retrieval.

```text
Expanded Query
      ↓
BGE Encoder
      ↓
Dense Vector
```

### 4. Dense Retrieval

The query embedding is then compared against pre-computed document embeddings using **FAISS** for efficient vector similarity search.

The current implementation retrieves the top-k documents for each query.

```text
Query Embedding
       │
       ▼
    FAISS
       │
       ▼
Top-k Retrieved Documents
```

## Technologies

* Python
* Pandas
* NumPy
* PyTorch
* Sentence Transformers
* BGE embedding models
* FAISS
* Gemini API
* SciFact / BEIR
