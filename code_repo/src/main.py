# from retrieval.query_rewriter import rewrite_query
# from retrieval.chroma_dense_retrieval import chroma_dense_retriever
from reranking.reranker import rerank
from config.settings import CHROMA_PATH, EMBED_MODEL, RERANK_MODEL, TOP_K_DENSE, TOP_K_RERANK
from embeddings.embedder import get_embedding_function
from llm.answer_generation import generate_answer
from llm.model import llm
from utils.logger import get_logger
from ingestion.build_vectordb import VectorDBBuilder_RAGbench
from constants.datasets_names import DATASETS
from classification.query_classifier import build_classifier
from classification.query_strategy import determine_strategy
from evaluation.evaluation_metrics import evaluate_rag_response, calculate_metrics
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)
embedding_function = get_embedding_function()



if __name__ == "__main__":
    logger.info("Starting the RAG pipeline...")
    logger.info("Initializing the vector database...")
    #---------------------------------------------------------
    vector_db_collection = {}
    for domain in DATASETS.keys():
        vector_db_builder = VectorDBBuilder_RAGbench(
                                                    collection_name=domain,
                                                    embedding_function=embedding_function,
                                                    path=CHROMA_PATH
                                                )
        vector_db_builder.initialize()
        vector_db_builder.summary()
        vector_db_collection[domain] = vector_db_builder
    #----------------------------------------------------------
    while True:
        try:
            #Ask for Domain
            print("\nAvailable Domains:\n")
            for idx, domain in enumerate(DATASETS.keys(), start=1):
                print(f"{idx}. {domain}")
            choice = int(input("\nSelect Domain: "))
            selected_domain = list(
                DATASETS.keys()
            )[choice - 1]
            print(f"\nSelected Domain: {selected_domain}")

            # Query input
            query = input("\nEnter your query (type 'exit' to quit): ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                logger.info("Exiting RAG pipeline...")
                print("Goodbye!")
                break

            # rewrite the query using a query rewriter
            # rewritten_query = rewrite_query(query)
            rewritten_query = query
            # Query classification
            classifier = build_classifier(llm=llm)
            classification = classifier.invoke({"query":query})
            print(classification)
            query_strategy = determine_strategy(classification=classification)
            if not query_strategy['retrieve']:
                answer = llm.invoke(rewritten_query).content.strip()
            else:
                vector_db = vector_db_collection[selected_domain]
                # retrieve relevant documents using a dense retriever (ChromaDB)
                # retrieved_docs_dict = vector_db_builder.query(rewritten_query, CHROMA_PATH, EMBED_MODEL, TOP_K_DENSE)
                retrieved_docs_dict = vector_db.query(rewritten_query, TOP_K_DENSE)
                # rerank the retrieved documents using a cross-encoder model
                reranked_docs = rerank(rewritten_query, retrieved_docs_dict)[:TOP_K_RERANK]
                # generate an answer using the top reranked documents as context
                answer = generate_answer(query, reranked_docs)
                logger.info(f"Generated answer: {answer[:100]}...")  # Log only the first 100 characters of the answer
                print("Answer:", answer)

                # Evaluation
                evaluation = evaluate_rag_response(question=query,contexts=reranked_docs,answer=answer)
                metrics = calculate_metrics(evaluation)
                print(metrics)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
            print("\nGoodbye!")
            break

        except Exception as e:
            logger.exception(f"Error while processing query: {e}")
            print(f"\nError: {e}")
    logger.info("RAG pipeline completed successfully.")

