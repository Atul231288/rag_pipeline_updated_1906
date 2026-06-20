# RAG Pipeline - IIITH AIML Group 11

A comprehensive Retrieval Augmented Generation (RAG) pipeline implementation with support for multiple embeddings, vector databases, and LLMs across diverse domains (Finance, Legal, Biomedical, Customer Support, General Knowledge).

## Project Structure

```
rag-capstone/
├── configs/              # Configuration files for different setups
├── data/                 # Domain-specific datasets
├── src/                  # Core source code
├── benchmarks/           # Benchmark datasets and evaluators
├── experiments/          # Experimental scripts
├── results/              # Results storage
├── notebooks/            # Jupyter notebooks for analysis
├── deployment/           # FastAPI and Streamlit apps
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Features

- **Multiple Document Ingestion**: PDF, DOCX, TXT support
- **Advanced Chunking**: Fixed, Semantic, Sliding Window, Small2Big strategies
- **Multiple Embeddings**: MPNet, BGE, E5, Jina
- **Vector Databases**: Chroma, FAISS, Qdrant, Milvus
- **Retrieval Methods**: BM25, Dense, Hybrid
- **Reranking**: MonoT5, MonoBERT, RankLLaMA
- **LLM Support**: Llama3, Mistral, Qwen
- **Benchmarking**: RAGBench and RGB evaluation

## Installation

See [Installation Guide](#installation) below.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python experiments/exp_01_baseline.py
```

## Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

1. Clone the repository
```bash
git clone https://github.com/Atul231288/rag-pipeline-iiith-aiml-group-11.git
cd rag-capstone
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Configurations are stored in `configs/` directory:
- `baseline.yaml` - Baseline RAG setup
- `semantic_bge.yaml` - Semantic chunking with BGE embeddings
- `hybrid_retrieval.yaml` - Hybrid retrieval strategy

## Documentation

- [Ingestion Guide](docs/ingestion.md)
- [Chunking Strategies](docs/chunking.md)
- [Embeddings](docs/embeddings.md)
- [Vector Databases](docs/vectordb.md)
- [Retrieval Methods](docs/retrieval.md)
- [Reranking](docs/reranking.md)

## Contributing

Please follow the contribution guidelines in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

IIITH AIML Group 11

## Contact

For questions or issues, please open a GitHub issue.
