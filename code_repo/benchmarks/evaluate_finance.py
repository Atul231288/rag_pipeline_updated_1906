"""
Finance domain evaluation script.

Evaluates the RAG pipeline on the finance domain (finqa + tatqa from RAGBench)
and reports Context Relevance, Context Utilization, Completeness, and Adherence.

Usage:
    cd code_repo
    python benchmarks/evaluate_finance.py --num-samples 20
    python benchmarks/evaluate_finance.py --num-samples 20 --gold-only
    python benchmarks/evaluate_finance.py --dataset finqa --num-samples 10
"""

import sys
import os
import argparse
import csv
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from tqdm import tqdm

FINANCE_DATASETS = ["finqa", "tatqa"]
# RAGBench actual column names → our display names
RAGBENCH_METRIC_MAP = {
    "relevance_score":    "context_relevance",
    "utilization_score":  "context_utilization",
    "completeness_score": "completeness",
    "adherence_score":    "adherence",
}
METRIC_KEYS = list(RAGBENCH_METRIC_MAP.values())


# ---------------------------------------------------------------------------
# Mode A: Gold labels from RAGBench (no LLM needed — instant baseline)
# ---------------------------------------------------------------------------

def evaluate_gold_labels(dataset_names: list[str], num_samples: int) -> dict:
    """
    Reads pre-computed gold metric labels directly from RAGBench.
    No LLM or vector DB required. Gives the upper-bound reference score.
    """
    all_rows = []
    for name in dataset_names:
        print(f"\nLoading gold labels: {name}")
        ds = load_dataset("rungalileo/ragbench", name, split="test")
        for row in ds:
            all_rows.append({"dataset": name, **row})
        if num_samples and len(all_rows) >= num_samples:
            break

    rows = all_rows[:num_samples] if num_samples else all_rows

    results = []
    for row in rows:
        entry = {
            "question": row.get("question", ""),
            "gold_answer": row.get("response", ""),   # correct column
            "dataset": row["dataset"],
        }
        for ragbench_col, display_name in RAGBENCH_METRIC_MAP.items():
            val = row.get(ragbench_col)
            entry[display_name] = float(val) if val is not None else None
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Mode B: Full pipeline evaluation (retrieval → rerank → generate → judge)
# ---------------------------------------------------------------------------

def evaluate_full_pipeline(dataset_names: list[str], num_samples: int) -> dict:
    """
    Runs each question through the live RAG pipeline:
    retrieve → rerank → generate → LLM judge → metrics.

    Requires: Ollama running with llama3.1:8b, ChromaDB built for finance.
    """
    # Lazy imports so --gold-only skips loading heavy models
    from chromadb.utils import embedding_functions
    from ingestion.build_vectordb import VectorDBBuilder_RAGbench
    from reranking.reranker import rerank
    from llm.answer_generation import generate_answer
    from evaluation.evaluation_metrics import evaluate_rag_response, calculate_metrics
    from config.settings import CHROMA_PATH, EMBED_MODEL, TOP_K_DENSE, TOP_K_RERANK

    print("\nLoading embedding model and vector DB...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    vector_db = VectorDBBuilder_RAGbench(
        collection_name="finance",
        embedding_function=embedding_function,
        path=CHROMA_PATH,
    )

    if vector_db.is_empty():
        print("Finance vector DB is empty — running ingestion first...")
        vector_db.initialize()

    print(f"Finance collection: {vector_db.count()} chunks\n")

    # Collect test questions
    all_questions = []
    for name in dataset_names:
        print(f"Loading questions: {name}")
        ds = load_dataset("rungalileo/ragbench", name, split="test")
        for row in ds:
            all_questions.append({
                "question": row["question"],
                "gold_answer": row.get("response", ""),
                "dataset": name,
            })

    questions = all_questions[:num_samples] if num_samples else all_questions
    print(f"\nEvaluating {len(questions)} questions...\n")

    results = []
    errors = []

    for item in tqdm(questions, desc="Evaluating"):
        query = item["question"]
        try:
            # 1. Retrieve from ChromaDB
            retrieved = vector_db.query(query, TOP_K_DENSE)

            # 2. Rerank with cross-encoder
            reranked_docs = rerank(query, retrieved)[:TOP_K_RERANK]

            # 3. Generate answer via LLM
            answer = generate_answer(query, reranked_docs)

            # 4. LLM-as-judge evaluation
            eval_result = evaluate_rag_response(
                question=query, contexts=reranked_docs, answer=answer
            )
            metrics = calculate_metrics(eval_result)

            results.append({
                "question": query,
                "dataset": item["dataset"],
                "answer": answer[:200],
                "reasoning": eval_result.reasoning[:300] if hasattr(eval_result, "reasoning") else "",
                **metrics,
            })

        except Exception as e:
            errors.append({"question": query, "error": str(e)})
            print(f"\n[ERROR] {query[:60]}: {e}")

    if errors:
        print(f"\n{len(errors)} questions failed — see errors in output.")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list[dict]) -> dict:
    valid = {k: [r[k] for r in results if r.get(k) is not None] for k in METRIC_KEYS}
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in valid.items()}


def print_report(results: list[dict], label: str):
    agg = aggregate_metrics(results)
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Finance Domain — {len(results)} questions")
    print(f"{'='*55}")
    print(f"  Context Relevance    : {agg['context_relevance']:.4f}")
    print(f"  Context Utilization  : {agg['context_utilization']:.4f}")
    print(f"  Completeness         : {agg['completeness']:.4f}")
    print(f"  Adherence            : {agg['adherence']:.4f}")
    print(f"{'='*55}\n")

    # Per-dataset breakdown
    datasets = sorted(set(r["dataset"] for r in results))
    if len(datasets) > 1:
        for ds in datasets:
            sub = [r for r in results if r["dataset"] == ds]
            sub_agg = aggregate_metrics(sub)
            print(f"  [{ds}] ({len(sub)} questions)")
            for k in METRIC_KEYS:
                print(f"    {k:<26}: {sub_agg[k]:.4f}")
        print()


def save_results(results: list[dict], out_path: str):
    if not results:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on finance domain")
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of questions to evaluate (default: 20)"
    )
    parser.add_argument(
        "--dataset", choices=["finqa", "tatqa", "both"], default="both",
        help="Which finance dataset to use (default: both)"
    )
    parser.add_argument(
        "--gold-only", action="store_true",
        help="Only compute gold-label baseline (no LLM, no vector DB needed)"
    )
    parser.add_argument(
        "--output-dir", default="./benchmarks/results",
        help="Directory to save CSV results (default: ./benchmarks/results)"
    )
    args = parser.parse_args()

    dataset_names = (
        FINANCE_DATASETS if args.dataset == "both"
        else [args.dataset]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.gold_only:
        print("\n[Mode A] Gold label baseline (RAGBench pre-computed scores)")
        results = evaluate_gold_labels(dataset_names, args.num_samples)
        print_report(results, "Gold Label Baseline")
        save_results(results, os.path.join(args.output_dir, f"finance_gold_{ts}.csv"))
    else:
        print("\n[Mode B] Full pipeline evaluation")
        results = evaluate_full_pipeline(dataset_names, args.num_samples)
        print_report(results, "Full Pipeline Evaluation")
        save_results(results, os.path.join(args.output_dir, f"finance_pipeline_{ts}.csv"))


if __name__ == "__main__":
    main()
