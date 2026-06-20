# Real World RAG System — Implementation Report
### Capstone Project | IIIT Hyderabad
**Supervisor:** Dr. Manish Shrivastava | **Mentors:** Gopichand, Lokesh

---

## Table of Contents

1. [What is RAG? — A Layman's Introduction](#1-what-is-rag)
2. [Project Goal](#2-project-goal)
3. [Dataset — RAGBench](#3-dataset--ragbench)
4. [System Architecture — The Full Pipeline](#4-system-architecture)
5. [Component 1 — Data Ingestion](#5-component-1--data-ingestion)
6. [Component 2 — Preprocessing](#6-component-2--preprocessing)
7. [Component 3 — Chunking](#7-component-3--chunking)
8. [Component 4 — Embeddings](#8-component-4--embeddings)
9. [Component 5 — Vector Database](#9-component-5--vector-database)
10. [Component 6 — Query Classification](#10-component-6--query-classification)
11. [Component 7 — Retrieval](#11-component-7--retrieval)
12. [Component 8 — Reranking](#12-component-8--reranking)
13. [Component 9 — Answer Generation](#13-component-9--answer-generation)
14. [Component 10 — Evaluation](#14-component-10--evaluation)
15. [Problems Found and Fixed](#15-problems-found-and-fixed)
16. [Results — Improvement Journey](#16-results--improvement-journey)
17. [Code Structure](#17-code-structure)
18. [Glossary](#18-glossary)

---

## 1. What is RAG?

### The Problem RAG Solves

Imagine you ask a doctor a question about a rare disease. The doctor has two options:

**Option A (Pure LLM):** Answer from memory alone. The doctor answers based on what they studied years ago — but medical knowledge changes, and they may have forgotten details or may hallucinate a wrong answer.

**Option B (RAG):** Before answering, the doctor quickly looks up the latest medical journals, finds the relevant pages, reads them, and then answers based on what they just read. Much more accurate.

RAG — **Retrieval Augmented Generation** — is Option B for AI systems.

### Simple Analogy

Think of RAG like an open-book exam:

```
Without RAG (Closed Book Exam):
  Student → answers from memory alone → may be wrong

With RAG (Open Book Exam):
  Student → finds relevant pages → reads them → answers from the book → accurate
```

### The Three Steps

Every RAG system does three things:

```
Step 1: RETRIEVE
  "Given this question, find the most relevant text from a large document library"

Step 2: AUGMENT
  "Add the retrieved text to the question as context for the AI"

Step 3: GENERATE
  "Ask the AI to answer using ONLY the provided context"
```

---

## 2. Project Goal

This project builds a **Real World RAG System** evaluated on the RAGBench benchmark dataset. The system:

- Handles **5 industry domains**: Finance, Legal, Biomedical, Customer Support, General Knowledge
- Uses **Advanced RAG** techniques: query classification, dense retrieval, cross-encoder reranking
- Evaluates answers using **4 RAGBench metrics**: Context Relevance, Context Utilization, Completeness, Adherence
- Targets **industry-grade** implementation quality

---

## 3. Dataset — RAGBench

### What is RAGBench?

RAGBench is a large-scale benchmark dataset created by Galileo AI containing **100,000 examples** spanning 5 industry domains. It is publicly available on HuggingFace.

**Link:** `rungalileo/ragbench`

### Finance Domain (Our Focus)

The finance domain contains two sub-datasets:

| Dataset | Description | Examples |
|---|---|---|
| `finqa` | Financial question answering over earnings reports | ~2,294 |
| `tatqa` | Table and text combined financial QA | ~1,669 |

### RAGBench Row Structure

Each row in RAGBench contains:

| Column | Type | Description |
|---|---|---|
| `question` | string | The query/question |
| `documents` | list of strings | Pre-retrieved context documents |
| `response` | string | Gold (correct) answer |
| `relevance_score` | float | Gold context relevance score (0-1) |
| `utilization_score` | float | Gold context utilization score (0-1) |
| `completeness_score` | float | Gold completeness score (0-1) |
| `adherence_score` | float | Gold adherence score (0-1) |

**Important:** The column is `response` not `answer` — a common mistake when first working with this dataset.

### The 4 Evaluation Metrics

These metrics measure different aspects of RAG quality:

```
1. CONTEXT RELEVANCE
   Question: "Were the chunks retrieved actually about the question?"
   Formula: relevant_chunks / total_chunks
   Example: If 3 out of 5 retrieved chunks were about the question → 0.60

2. CONTEXT UTILIZATION
   Question: "Did the LLM actually use what was retrieved?"
   Formula: used_context_facts / total_context_facts
   Example: If LLM used 4 out of 8 facts in the context → 0.50

3. COMPLETENESS
   Question: "Did the answer cover all the relevant information?"
   Formula: covered_relevant_facts / total_relevant_facts
   Example: If answer covered 6 out of 10 required facts → 0.60

4. ADHERENCE
   Question: "Is every statement in the answer backed by the context?"
   Formula: supported_statements / total_statements
   Example: If 7 out of 10 answer statements are grounded → 0.70
```

**Why these 4 metrics matter:**
- Low Context Relevance → Retrieval is broken (wrong chunks fetched)
- Low Context Utilization → LLM is ignoring the retrieved context
- Low Completeness → Answer is missing important facts
- Low Adherence → LLM is hallucinating (making up facts not in context)

---

## 4. System Architecture

### The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (Run Once)                   │
│                                                                     │
│  RAGBench Dataset                                                   │
│       │                                                             │
│       ▼                                                             │
│  [1. Load Documents]  ← HuggingFace datasets library               │
│       │                                                             │
│       ▼                                                             │
│  [2. Preprocess]      ← Clean text, convert JSON tables to text    │
│       │                                                             │
│       ▼                                                             │
│  [3. Chunk]           ← Split into smaller pieces                  │
│       │                                                             │
│       ▼                                                             │
│  [4. Embed]           ← Convert text → 384-dim vectors             │
│       │                                                             │
│       ▼                                                             │
│  [5. Store in ChromaDB] ← Persistent vector database on disk       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   QUERY PIPELINE (Runs Per Question)                │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  [6. Classify Query]  ← Does this need retrieval or direct LLM?    │
│       │                                                             │
│       ├─── No retrieval needed ──→ LLM answers directly            │
│       │                                                             │
│       ▼ Retrieval needed                                            │
│  [7. Retrieve]        ← Embed query → search ChromaDB → top-10     │
│       │                                                             │
│       ▼                                                             │
│  [8. Rerank]          ← Cross-encoder scores → keep top-5          │
│       │                                                             │
│       ▼                                                             │
│  [9. Generate Answer] ← LLM answers using retrieved context        │
│       │                                                             │
│       ▼                                                             │
│  [10. Evaluate]       ← LLM-as-judge scores the 4 metrics         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Simple RAG vs Advanced RAG

| Feature | Simple RAG | Our Advanced RAG |
|---|---|---|
| Retrieval | Basic keyword search | Dense vector search |
| Filtering | None | Cross-encoder reranking |
| Query handling | Always retrieves | Classifies first |
| Evaluation | Manual | Automated LLM-as-judge |
| Domains | Single | Multi-domain (5 domains) |

---

## 5. Component 1 — Data Ingestion

### What It Does

Downloads documents from the RAGBench HuggingFace dataset and prepares them for storage in ChromaDB.

### Key File

`src/ingestion/build_vectordb.py` — `VectorDBBuilder_RAGbench` class

### How It Works

```python
# Load finance datasets
ds = load_dataset("rungalileo/ragbench", "finqa", split="test")

# Each row has a 'documents' field — a list of context strings
# These are the documents we store in ChromaDB
for doc_text in ds['documents']:
    for single_doc in doc_text:
        # deduplicate, preprocess, chunk, store
```

### The `is_empty()` Check — Why It Matters

The `initialize()` method checks if the collection already has data before ingesting:

```python
def initialize(self):
    if not self.is_empty():
        print("Collection already has data. Skipping ingestion.")
        return
    # Only runs if empty — first time only
```

**Why this is important (production design):**
- Ingestion is slow (embeds millions of tokens)
- You only want to do it once, or when data changes
- Every subsequent run just loads the existing DB — milliseconds instead of hours

### Deduplication

`src/ingestion/dedup_docs.py` removes duplicate documents before storing:

```python
def dedup_docs(subset_data):
    seen_docs = set()
    for doc_text in subset_data['documents']:
        for docs in doc_text:
            if docs not in seen_docs:
                seen_docs.add(docs)
                raw_documents.append(docs)
```

**Why deduplicate?** RAGBench has many questions sharing the same source documents. Without deduplication, the same chunk would be stored multiple times, wasting storage and skewing retrieval results.

---

## 6. Component 2 — Preprocessing

### What It Does

Cleans raw document text before it gets chunked and stored. This is critical for data quality.

### Key File

`src/ingestion/preprocess_text.py`

### Evolution of Preprocessing

#### Version 1 (Original)
```python
def preprocess_documents(document):
    return document.replace('\n\n', '\n')
```
Simple — just removes double newlines. Fast but misses a critical issue.

#### Version 2 (With Table Conversion — Our Fix)
```python
def preprocess_documents(document):
    document = document.replace('\n\n', '\n')
    document = _convert_tables(document)
    return document
```

### The Table Problem

RAGBench finance documents often contain tables stored as raw JSON strings:

```
BEFORE (raw JSON in ChromaDB):
[["", "1/2/2010", "1/1/2011", "12/31/2011"],
 ["cadence design systems inc.", "100.00", "137.90", "173.62"],
 ["nasdaq composite", "100.00", "117.61", "118.70"]]
```

An LLM reading this sees bracket characters and numbers but struggles to understand the table structure. It doesn't know which number belongs to which company and which date.

```
AFTER (readable text in ChromaDB):
cadence design systems inc. — 1/2/2010: 100.00, 1/1/2011: 137.90, 12/31/2011: 173.62
nasdaq composite — 1/2/2010: 100.00, 1/1/2011: 117.61, 12/31/2011: 118.70
```

Now the LLM can clearly see: "Cadence's value on 1/1/2011 was 137.90."

### The Table Conversion Function

```python
def _table_to_text(table: list) -> str:
    headers = table[0]           # First row = column headers
    rows = table[1:]             # Remaining rows = data

    lines = []
    for row in rows:
        label = row[0]           # Company/category name
        parts = []
        for col_idx, value in enumerate(row[1:], start=1):
            if col_idx < len(headers) and value:
                parts.append(f"{headers[col_idx]}: {value}")
        lines.append(f"{label} — " + ", ".join(parts))

    return "\n".join(lines)
```

**How to read this:**
1. Take the header row: `["", "1/2/2010", "1/1/2011", "12/31/2011"]`
2. For each data row: `["cadence design systems inc.", "100.00", "137.90", "173.62"]`
3. Pair each value with its column header
4. Format as: `"Company — date: value, date: value, ..."`

---

## 7. Component 3 — Chunking

### What Is Chunking and Why Is It Needed?

Imagine a 50-page annual report. You can't give all 50 pages to the LLM at once because:
1. LLMs have a context window limit (maximum text they can process)
2. Searching 50 pages for one answer is inefficient
3. Irrelevant pages add noise to the answer

Chunking breaks large documents into smaller, manageable pieces.

```
Full Document (50,000 characters)
          │
          ▼
   [Chunking Algorithm]
          │
          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Chunk 1 │ │Chunk 2 │ │Chunk 3 │ │Chunk 4 │  ... 50 chunks
│1000 ch │ │1000 ch │ │1000 ch │ │1000 ch │
└────────┘ └────────┘ └────────┘ └────────┘
```

### Version 1 — Recursive Character Splitting (Original)

**File:** `src/chunking/text_chunking.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
```

**How it works:**
- Tries to split on `\n\n` (paragraph breaks) first
- If still too large, splits on `\n` (line breaks)
- If still too large, splits on spaces
- Last resort: splits on individual characters
- Always respects the 1000 character limit
- `chunk_overlap=100` means 100 characters are repeated between consecutive chunks (helps maintain context continuity)

**Problem:** It splits purely on size, not meaning. A financial table could be cut in half right between a company name and its values.

### Version 2 — Semantic Chunking (Improved)

**The Core Idea:** Instead of splitting every 1000 characters, split where the *meaning changes*.

```
Sentence 1: "The company reported strong earnings in Q3."
Sentence 2: "Revenue grew by 15% year over year."
Sentence 3: "The board approved a new dividend policy."  ← meaning changes here
Sentence 4: "Shareholders will receive $0.50 per share."
```

Sentences 1-2 are about earnings (keep together). Sentences 3-4 are about dividends (keep together). A character-based splitter might put sentences 2-3 in the same chunk — breaking the semantic coherence.

**How semantic chunking works:**

```
Step 1: Split document into sentences

Step 2: Convert each sentence into a vector (embed it)
   Sentence 1 → [0.2, -0.1, 0.8, ...]   (384 numbers)
   Sentence 2 → [0.3, -0.2, 0.7, ...]   (384 numbers)
   Sentence 3 → [-0.5, 0.6, 0.1, ...]   (384 numbers)  ← very different!

Step 3: Compare consecutive sentence vectors
   similarity(S1, S2) = 0.85  → high → keep together
   similarity(S2, S3) = 0.31  → low  → SPLIT HERE

Step 4: Group sentences between split points into chunks
```

**The Code:**

```python
def _semantic_chunk(text, breakpoint_threshold=0.4, max_chunk_chars=1000):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+|\n', text.strip())

    # Embed all sentences at once (efficient)
    model = _get_semantic_model()  # loaded once, reused
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        # Calculate similarity between consecutive sentences
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])

        # Split if meaning changes OR chunk is getting too large
        if similarity < 0.4 or len_of_current > max_chunk_chars:
            chunks.append(' '.join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    return chunks
```

**Key Design Decision — Singleton Model:**
```python
_semantic_model = None

def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer(EMBED_MODEL)
    return _semantic_model
```

The embedding model is loaded ONCE and reused for every document. Without this, the model would reload (~2 minutes) for every document — making ingestion impossibly slow.

### Chunking Comparison

| Aspect | Recursive (Old) | Semantic (New) |
|---|---|---|
| Split logic | Character count | Meaning change |
| Financial tables | May cut mid-row | Keeps table together |
| Speed | Very fast | Slower (needs embedding) |
| Quality | Generic | Finance-aware |
| Chunk size | Fixed 1000 chars | Variable (content-driven) |

---

## 8. Component 4 — Embeddings

### What Are Embeddings?

An embedding is a way to represent text as a list of numbers (a vector) such that similar texts have similar numbers.

```
"The stock price went up"    → [0.2, -0.1, 0.8, 0.3, ...]  (384 numbers)
"Shares increased in value"  → [0.3, -0.2, 0.7, 0.4, ...]  (similar numbers!)
"The weather is sunny"       → [-0.5, 0.6, 0.1, -0.3, ...] (very different)
```

This allows us to do **math-based similarity search** instead of keyword matching.

### The Model Used

**Model:** `BAAI/bge-small-en-v1.5`
- Made by Beijing Academy of Artificial Intelligence (BAAI)
- "bge" = BAAI General Embedding
- "small" = lightweight version (fast, lower memory)
- Produces 384-dimensional vectors
- Free, runs locally (no API costs)

### Key File

`src/embeddings/embedder.py` (created during implementation):

```python
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_MODEL

def get_embedding_function():
    """For ChromaDB — used when building/querying the vector DB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

def get_embedder():
    """Raw SentenceTransformer — for inspecting vectors directly."""
    return SentenceTransformer(EMBED_MODEL)
```

**Why two functions?**
- `get_embedding_function()` wraps the model in ChromaDB's format — ChromaDB calls it automatically when you add or query documents
- `get_embedder()` gives you the raw model — useful for inspecting embedding vectors, semantic chunking, and manual query embedding

### What a Vector Looks Like

When we ran Step 4 in the notebook:
```
Embedding vector shape  : (384,)
Vector dimension        : 384
First 10 values         : [-0.0513  -0.0231  -0.0122  -0.0053   0.0221  -0.0049  -0.0257  ...]
Min value               : -0.3098
Max value               : 0.3418
Cosine similarity between two finance chunks: 0.7668
```

The 0.7668 similarity means two chunks from the same financial document are 76.68% similar — makes intuitive sense.

### Cosine Similarity Explained

```
Cosine Similarity = 1.0  → Identical meaning
Cosine Similarity = 0.8  → Very similar (same topic)
Cosine Similarity = 0.5  → Somewhat related
Cosine Similarity = 0.0  → Completely unrelated
Cosine Similarity = -1.0 → Opposite meaning
```

Formula: `similarity = (A · B) / (|A| × |B|)`
Where A and B are the embedding vectors.

---

## 9. Component 5 — Vector Database

### What is a Vector Database?

A regular database stores text and lets you search by exact keywords. A vector database stores text + its embedding vector and lets you search by **meaning similarity**.

```
Regular DB search:
  Query: "rate of return"
  Finds: documents containing EXACTLY "rate of return"
  Misses: "investment performance", "percentage gain"

Vector DB search:
  Query: "rate of return"
  Finds: documents semantically similar to "rate of return"
  Also finds: "investment performance", "percentage gain", "ROI"
```

### ChromaDB

**ChromaDB** is the vector database used in this project. It is:
- Open source and free
- Runs locally (no cloud needed)
- Stores vectors in an HNSW index for fast search
- Supports persistent storage (data survives session restarts)

### HNSW Index — How Fast Search Works

HNSW = **Hierarchical Navigable Small World**

Think of it like a highway system:
- Level 2 (Highways): Few connections, long distances — quickly gets you to the right region
- Level 1 (Roads): More connections — narrows down to the right neighborhood
- Level 0 (Streets): All connections — finds the exact answer

Without HNSW: Compare query against every single vector (slow — O(n))
With HNSW: Navigate the graph hierarchy (fast — O(log n))

### Two Modes of ChromaDB

| Mode | Code | Persists? | Use Case |
|---|---|---|---|
| In-memory | `EphemeralClient()` | No — lost when Python stops | Quick experiments |
| Persistent | `PersistentClient(path)` | Yes — saved to disk | Production |

**Our setup:** `PersistentClient` with path `./vectordb/chroma_db`

### Collections

ChromaDB organizes data into **collections** — one per domain:

```
chroma_db/
├── finance          ← finqa + tatqa chunks
├── legal            ← cuad chunks
├── biological       ← pubmedqa + covidqa chunks
├── customer_support ← delucionqa + emanual + techqa chunks
└── general_knowledge ← hotpotqa + msmarco + hagrid + expertqa chunks
```

Finance collection: **3,565 chunks** after rebuilding with semantic chunking + table preprocessing.

---

## 10. Component 6 — Query Classification

### What It Does

Before retrieving, the system classifies the query to decide the best strategy:

```
"What is the rate of return for Cadence from 2010 to 2011?"
              │
              ▼
       [Query Classifier]
              │
    ┌─────────┴──────────┐
    │                    │
retrieval_required=True  retrieval_required=False
    │                    │
    ▼                    ▼
RAG Pipeline        LLM answers directly
(retrieve+generate)  (no ChromaDB needed)
```

### Why Classify?

Not every question needs retrieval. Example:
- "What is 2 + 2?" → LLM knows this, no retrieval needed
- "What was Apple's revenue in Q3 2023?" → Needs retrieval from stored documents

Skipping unnecessary retrieval saves time and reduces noise.

### How It Works

**File:** `src/classification/query_classifier.py`

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def build_classifier(llm):
    chain = prompt | llm | JsonOutputParser()
    return chain

# Usage:
classification = classifier.invoke({"query": user_question})
# Returns: {"domain": "finance", "retrieval_required": True, "reasoning": "..."}
```

**The classification schema** (`src/classification/query_classification_schema.py`):
```python
class QueryClassification(BaseModel):
    domain: str              # e.g., "finance"
    dataset: str             # e.g., "finqa"
    retrieval_required: bool # True or False
    reasoning: str           # Why this decision was made
```

**The strategy** (`src/classification/query_strategy.py`):
```python
def determine_strategy(classification):
    if not classification["retrieval_required"]:
        return {"retrieve": False, "strategy": "direct_llm"}
    return {"retrieve": True, "strategy": "rag"}
```

---

## 11. Component 7 — Retrieval

### What It Does

Given a user query, find the most relevant chunks from ChromaDB.

### How It Works

```
User Query: "What is the rate of return for Cadence 2010-2011?"
       │
       ▼
[Embed the query]
       │
       ▼
Query Vector: [-0.02, 0.15, -0.08, ...] (384 numbers)
       │
       ▼
[ChromaDB similarity search]
       │  Compares query vector against ALL stored chunk vectors
       │  Returns top-K most similar
       ▼
Retrieved Chunks (top 10):
  1. "cadence design systems inc. — 1/2/2010: 100.00, ..." (distance: 0.12)
  2. "stockholder return performance graph..."              (distance: 0.18)
  3. "cash provided by operating activities..."            (distance: 0.45)
  ...
```

**Distance vs Similarity:**
- ChromaDB returns **distance** (lower = more similar)
- Distance 0.12 = very similar to query
- Distance 0.45 = less similar

### Configuration

`src/config/settings.py`:
```python
TOP_K_DENSE = 10   # Retrieve top 10 initially
TOP_K_RERANK = 5   # Keep top 5 after reranking
```

### The Key Insight from Our Testing

When we printed retrieved chunks vs gold documents for the Cadence question:

```
Gold Doc 2      = Retrieved 1  ✓ (exact match — the table with 137.90)
Gold Doc 1      = Retrieved 2  ✓ (exact match — stockholder return text)
Retrieved 3     = Wrong company data ✗
```

**Retrieval was working correctly** — the right chunks were found. The problem was not retrieval but what came after (the LLM couldn't parse raw JSON tables).

---

## 12. Component 8 — Reranking

### Why Reranking?

The initial retrieval uses embedding similarity — it's fast and works well for getting roughly relevant documents. However, it's not perfect because:

- It embeds the query and documents **separately** — doesn't read them together
- It can miss nuanced relevance (the query "rate of return 2010-2011" might match a chunk about "2010 annual report" even if it doesn't contain the answer)

Reranking is a **second pass** that reads the query and each document **together** for a more accurate relevance score.

### Bi-encoder vs Cross-encoder

```
BI-ENCODER (Retrieval Stage):
  Query → [Encoder] → Query Vector
  Doc 1 → [Encoder] → Doc1 Vector
  similarity = cosine(Query Vector, Doc1 Vector)
  Fast! But encodes independently.

CROSS-ENCODER (Reranking Stage):
  [Query + Doc 1] → [Encoder] → Relevance Score
  Reads them together — much more accurate.
  But slow — can't precompute (must run at query time).
```

### The Model Used

**Model:** `BAAI/bge-reranker-base`
- Cross-encoder model from BAAI
- Outputs a relevance score (higher = more relevant)
- Takes (query, document) pairs as input

### How It Works

**File:** `src/reranking/reranker.py`

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(RERANK_MODEL)  # loaded once at module level

def rerank(query, retrieved_docs_dict):
    docs = retrieved_docs_dict['documents'][0]

    # Create (query, document) pairs
    pairs = [(query, doc) for doc in docs]

    # Score each pair
    scores = reranker.predict(pairs)

    # Sort by score (highest first)
    ranked = sorted(zip(scores, ids, docs), reverse=True)

    return [doc for score, id, doc in ranked[:TOP_K_RERANK]]
```

### Before vs After Reranking

```
Before Reranking (raw retrieval order by distance):
  1. Finance table (distance 0.12) ← most similar embedding
  2. Stockholder return text (distance 0.18)
  3. Irrelevant cash flow data (distance 0.45)

After Reranking (by cross-encoder relevance score):
  1. Stockholder return text (score 0.92) ← most relevant to question
  2. Finance table (score 0.87)
  3. Another relevant chunk (score 0.71)
```

---

## 13. Component 9 — Answer Generation

### What It Does

Takes the user question + top reranked chunks → asks the LLM to generate an answer.

### The LLM Used

**Model:** `llama3.1:8b` via **Ollama**
- Llama 3.1 with 8 billion parameters
- Runs locally on your machine (no API key, no cost)
- Temperature = 0 (deterministic — same question always gives same answer)
- Served by Ollama on port 11434

**File:** `src/llm/model.py`
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1:8b", temperature=0)
```

### The Answer Prompt — Evolution

#### Version 1 (Original — Too Simple)
```
You are a helpful RAG assistant.
Answer ONLY from the provided context.
If the answer is not present, say: "I couldn't find enough evidence."

Context: {context}
Question: {query}
```

**Problem:** LLM received a raw JSON table and said "I couldn't find enough evidence" because it couldn't parse `[["", "1/2/2010"...]]`.

#### Version 2 (Fixed — Table-Aware)
```
You are a financial analyst assistant. Answer using ONLY the provided context.

Rules:
- Context may contain tables formatted as readable text. Extract numbers directly.
- If arithmetic is needed (rate of return, percentage), compute step by step.
- Show your calculation briefly, then give the final answer.
- Use ONLY facts from the context. Do NOT add outside knowledge.
- If answer is not in context, say: "I couldn't find enough evidence."

Context: {context}
Question: {query}
```

**Result:** LLM correctly answered:
```
From the context:
Initial value on January 2, 2010: $100.00
Final value on January 1, 2011: $137.90

Calculation:
$137.90 - $100.00 = $37.90
($37.90 ÷ $100.00) × 100% ≈ 37.9%

The rate of return from 2010 to 2011 is 37.9%.
```

### How generate_answer Works

**File:** `src/llm/answer_generation.py`
```python
def generate_answer(query, context):
    prompt = ANSWER_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt)
    return response.content.strip()
```

Simple but powerful — the quality of the answer depends almost entirely on the quality of the prompt and the quality of the retrieved context.

---

## 14. Component 10 — Evaluation

### The LLM-as-Judge Approach

Traditional NLP evaluation compares text similarity (BLEU, ROUGE scores). For RAG, we need to evaluate whether the answer is correct, grounded, and complete — which requires understanding, not just text matching.

**Solution:** Use another LLM instance as a judge.

```
Question + Retrieved Context + Generated Answer
                    │
                    ▼
           [Judge LLM (llama3.1:8b)]
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  total_chunks: 3                 │
    │  relevant_chunks: 2              │
    │  total_context_facts: 8          │
    │  used_context_facts: 4           │
    │  total_relevant_facts: 5         │
    │  covered_relevant_facts: 3       │
    │  total_answer_statements: 4      │
    │  supported_answer_statements: 3  │
    │  reasoning: "..."                │
    └──────────────────────────────────┘
```

### The Evaluation Schema

**File:** `src/evaluation/evaluation_schema.py`
```python
class RAGEvaluationCounts(BaseModel):
    total_chunks: int
    relevant_chunks: int
    total_context_facts: int
    used_context_facts: int
    total_relevant_facts: int
    covered_relevant_facts: int
    total_answer_statements: int
    supported_answer_statements: int
    reasoning: str
```

### Metric Calculation

**File:** `src/evaluation/evaluation_metrics.py`
```python
def calculate_metrics(result):
    return {
        "context_relevance":   result.relevant_chunks / max(result.total_chunks, 1),
        "context_utilization": result.used_context_facts / max(result.total_context_facts, 1),
        "completeness":        result.covered_relevant_facts / max(result.total_relevant_facts, 1),
        "adherence":           result.supported_answer_statements / max(result.total_answer_statements, 1)
    }
```

**Note:** `max(..., 1)` prevents division by zero when counts are 0.

### Structured Output

The judge LLM returns structured JSON (not free text) using LangChain's `with_structured_output`:

```python
judge_llm = llm.with_structured_output(RAGEvaluationCounts)
result = judge_llm.invoke(prompt)
# result is automatically parsed into RAGEvaluationCounts object
```

---

## 15. Problems Found and Fixed

### Problem 1 — JSON Tables in Chunks

**Discovery:** Running the pipeline on a finance question and getting "I couldn't find enough evidence" even though the right chunks were retrieved.

**Root Cause:** Financial tables stored as raw JSON strings:
```
[["", "1/2/2010", "1/1/2011"], ["cadence design systems inc.", "100.00", "137.90"]]
```

LLM couldn't parse this format.

**Fix Applied:** Two-part solution:

1. **Fix 1 — Better Prompt** (immediate, no re-ingestion):
   Added explicit instructions to parse tables and compute arithmetic.

2. **Fix 2 — Table Preprocessing** (requires re-ingestion):
   Convert JSON tables to readable text BEFORE storing in ChromaDB.

**Impact:**
| Metric | Before | After Fix 1 | After Fix 1+2 |
|---|---|---|---|
| Context Utilization | 0.14 | 0.14 | 0.44 |
| Completeness | 0.20 | 0.33 | 0.60 |
| Adherence | 0.00 | 0.75 | 0.50 |

### Problem 2 — LLM Hallucination in Evaluation

**Discovery:** After Fix 1+2, Adherence dropped from 0.75 to 0.50. The judge said the answer referenced "expected volatility and risk-free interest rates" — concepts not in the context.

**Root Cause:** The LLM (llama3.1:8b) was applying domain knowledge about finance from its training data, going beyond what was in the retrieved context.

**Fix Applied:** Tightened the prompt:
```
"Use ONLY facts, numbers, and values explicitly present in the context.
Do NOT add financial concepts, domain knowledge, or assumptions from outside the context."
```

### Problem 3 — Module Caching

**Discovery:** After updating `preprocess_text.py`, the rebuild still used the old version.

**Root Cause:** Python caches imported modules. When the rebuild cell ran, it used the already-cached (old) version of `preprocess_text`.

**Fix Applied:** Always restart the kernel before running rebuild scripts to ensure all modules reload from disk.

**Lesson:** In Python, `import module` is cached after first import. To reload: restart kernel OR use `importlib.reload(module)`.

### Problem 4 — Empty Chunks from Table Conversion

**Discovery:** After adding table conversion, some documents produced empty chunks, causing ChromaDB's `collection.add()` to throw `ValueError: Non-empty lists required`.

**Root Cause:** Some documents were purely JSON tables with no data rows (only headers). `_table_to_text()` returned `""` for these → `chunk_documents("")` returned `[]`.

**Fix Applied:** Added a guard in `build_vectordb.py`:
```python
chunks = create_chunks(text)
if not chunks:
    logger.warning(f"Document {idx+1} produced no chunks, skipping.")
    continue
```

Also added empty string guard in `chunk_documents()`:
```python
if not preprocessed_docs or not preprocessed_docs.strip():
    return []
```

---

## 16. Results — Improvement Journey

### Single Question Evaluation (Cadence Design Systems Rate of Return)

| Stage | Context Rel. | Context Util. | Completeness | Adherence |
|---|---|---|---|---|
| Baseline (demo DB) | 0.40 | 0.25 | 0.00 | 0.00 |
| Real DB (full finqa) | 0.44 | 0.14 | 0.20 | 0.00 |
| + Better Prompt | 0.50 | 0.14 | 0.33 | 0.75 |
| + Table Preprocessing | 0.43 | 0.44 | 0.60 | 0.50 |
| + Tighter Grounding Prompt | 0.43 | 0.44 | 0.60 | 0.50 |

### Key Insight from Testing

When we compared retrieved chunks against gold documents:
- Retrieved 1 = Gold Doc 2 ✓ (the exact table with the answer)
- Retrieved 2 = Gold Doc 1 ✓ (the narrative text)

**Retrieval was working correctly from the start.** The bottleneck was:
1. LLM couldn't parse JSON tables → Fixed by prompt + preprocessing
2. LLM adding outside knowledge → Partially fixed by tighter prompt

### What Improvements Remain

1. **Semantic Chunking** — now implemented, needs evaluation
2. **Better Embedding Model** — `BAAI/bge-large-en-v1.5` or finance-specific FinBERT
3. **Multi-question Benchmark** — evaluate over 20+ questions for reliable averages
4. **Larger LLM** — llama3.1:70b or GPT-4 for better table reasoning

---

## 17. Code Structure

```
code_repo/
├── src/
│   ├── main.py                          ← Entry point: interactive CLI
│   ├── config/
│   │   └── settings.py                  ← Model names, paths, TOP_K values
│   ├── constants/
│   │   └── datasets_names.py            ← Domain → dataset mapping
│   ├── ingestion/
│   │   ├── build_vectordb.py            ← VectorDBBuilder_RAGbench class
│   │   ├── preprocess_text.py           ← Clean text + table conversion
│   │   └── dedup_docs.py                ← Remove duplicate documents
│   ├── chunking/
│   │   └── text_chunking.py             ← Recursive + Semantic chunking
│   ├── embeddings/
│   │   └── embedder.py                  ← get_embedding_function, get_embedder
│   ├── classification/
│   │   ├── query_classifier.py          ← LangChain chain for classification
│   │   ├── query_classification_schema.py ← QueryClassification Pydantic model
│   │   └── query_strategy.py            ← decide retrieve vs direct_llm
│   ├── retrieval/
│   │   ├── chroma_dense_retrieval.py    ← ChromaDB query wrapper
│   │   ├── context_compressor.py        ← (future use)
│   │   └── query_rewriter.py            ← (future use)
│   ├── reranking/
│   │   └── reranker.py                  ← CrossEncoder reranking
│   ├── llm/
│   │   ├── model.py                     ← ChatOllama instance
│   │   └── answer_generation.py         ← generate_answer function
│   ├── prompts/
│   │   ├── answer_prompt.py             ← ANSWER_PROMPT template
│   │   ├── evaluation_prompt.py         ← RAG_EVALUATION_PROMPT template
│   │   └── query_classification_prompt.py ← CLASSIFICATION_TEMPLATE
│   ├── evaluation/
│   │   ├── evaluation_metrics.py        ← evaluate_rag_response, calculate_metrics
│   │   └── evaluation_schema.py         ← RAGEvaluationCounts Pydantic model
│   └── utils/
│       └── logger.py                    ← Structured logging setup
├── notebooks/
│   ├── rag_pipeline_walkthrough.ipynb   ← Step-by-step exploration notebook
│   └── load_ragbench_dataset.ipynb      ← Dataset exploration
├── benchmarks/
│   └── evaluate_finance.py             ← Multi-question benchmark evaluation
├── tests/
│   └── __init__.py                      ← (test files to be added)
├── vectordb/
│   └── chroma_db/                       ← Persistent ChromaDB storage
├── requirements.txt
└── .env.example
```

---

## 18. Glossary

| Term | Simple Explanation |
|---|---|
| **RAG** | AI that retrieves relevant documents before answering |
| **Embedding** | Converting text to a list of numbers that captures meaning |
| **Vector** | A list of numbers representing text meaning |
| **Cosine Similarity** | Math score (0 to 1) measuring how similar two vectors are |
| **Vector Database** | Database that stores and searches by vector similarity |
| **ChromaDB** | The open-source vector database used in this project |
| **HNSW** | Fast graph-based algorithm for searching vectors |
| **Chunk** | A small piece of a larger document |
| **Chunking** | Breaking a large document into smaller pieces |
| **Semantic Chunking** | Chunking based on meaning changes, not character count |
| **Reranking** | Second-pass filtering of retrieved documents for better accuracy |
| **Cross-encoder** | Model that reads query + document together for relevance scoring |
| **Bi-encoder** | Model that encodes query and document separately (used in retrieval) |
| **LLM** | Large Language Model — the AI that generates text |
| **Ollama** | Tool for running LLMs locally on your computer |
| **Hallucination** | When an LLM makes up facts not present in the context |
| **Prompt Engineering** | Crafting instructions to get better LLM outputs |
| **LLM-as-Judge** | Using an LLM to evaluate the quality of another LLM's output |
| **Context Relevance** | How relevant the retrieved chunks are to the question |
| **Context Utilization** | How much of the retrieved context the LLM actually used |
| **Completeness** | Whether the answer covered all relevant facts |
| **Adherence** | Whether every claim in the answer is backed by the context |
| **Gold Answer** | The pre-verified correct answer in a benchmark dataset |
| **RAGBench** | The benchmark dataset used for evaluation |
| **finqa** | Finance QA sub-dataset in RAGBench (earnings reports) |
| **tatqa** | Table+Text QA sub-dataset in RAGBench |
| **Ingestion Pipeline** | The one-time process of loading and storing documents |
| **Query Pipeline** | The per-request process of retrieve → generate → evaluate |
| **Persistent DB** | Database saved to disk, survives session restarts |
| **EphemeralDB** | In-memory database, lost when Python session ends |

---

*Document generated during capstone project implementation — IIIT Hyderabad, June 2026*
