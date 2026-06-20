# Creating functions to build the vector database for the RAG pipeline and delete the collection after verification to clean up the database. viewign the available collections in the ChromaDB client to verify the collection was created and documents were added.
import chromadb

from chromadb.utils import embedding_functions
import chromadb

from datasets import load_dataset
from chromadb.utils import embedding_functions
from ingestion.dedup_docs import dedup_docs
from ingestion.preprocess_text import preprocess_documents
from chunking.text_chunking import chunk_documents as create_chunks
from constants.datasets_names import DATASETS
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorDBBuilder_RAGbench:
    def __init__(
        self,
        collection_name: str,
        embedding_function,
        path: str = "./chroma_db"
    ):

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection_name = collection_name

        self.embedding_function = embedding_function

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        )
        self.DATASETS = DATASETS

    # ----------------------------------
    # Initialization
    # ----------------------------------

    def initialize(self):

        if not self.is_empty():

            print(
                f"Collection '{self.collection_name}' "
                f"already contains {self.count()} chunks."
            )

            return

        print(
            f"Collection '{self.collection_name}' "
            f"is empty. Starting ingestion..."
        )

        # for domain, dataset_list in self.DATASETS.items():
        for dataset_name in self.DATASETS.get(self.collection_name, None):

            # for dataset_name in dataset_list:

            logger.info(f"Loading Dataset: {dataset_name}")

            subset_data = load_dataset(
                "rungalileo/ragbench",
                dataset_name,
                split="test"
            )
            dataset = dedup_docs(subset_data)
            logger.info(f"Total records in {dataset_name} are {len(subset_data)}. Deduped docs are {len(dataset)}")
            total_chunks = 0
            for idx, doc in enumerate(dataset):
                text = preprocess_documents(doc)
                chunks = create_chunks(text)
                if not chunks:
                    logger.warning(f"Document {idx+1} produced no chunks, skipping.")
                    continue
                print(f"Document {idx+1} split into {len(chunks)} chunks.")
                self.build_vector_db(
                    chunks=chunks,
                    dataset_name=dataset_name,
                    domain=self.collection_name
                )

                total_chunks += len(chunks)

            logger.info(f"Finished {dataset_name} with {total_chunks} chunks")

        logger.info(f"Ingestion Complete. Total Documents = {self.count()}")

    # -----------------------------
    # Collection Utilities
    # -----------------------------

    def list_collections(self):

        return [
            collection.name
            for collection in self.client.list_collections()
        ]

    def collection_exists(self):

        return (
            self.collection_name
            in self.list_collections()
        )

    def count(self):

        return self.collection.count()

    def is_empty(self):

        return self.count() == 0

    # -----------------------------
    # Insert
    # -----------------------------

    def build_vector_db(
        self,
        chunks: list,
        dataset_name: str,
        domain: str
    ):

        ids = []
        metadatas = []

        start_count = self.count()

        for idx, chunk in enumerate(chunks):

            ids.append(
                f"{domain}_{dataset_name}_{start_count + idx}"
            )

            metadatas.append(
                {
                    "dataset": dataset_name,
                    "domain": domain
                }
            )

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    # -----------------------------
    # Query
    # -----------------------------

    def query(
        self,
        query: str,
        top_k: int = 10
    ):

        return self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

    # -----------------------------
    # Delete Collection
    # -----------------------------

    def delete_collection(self):

        self.client.delete_collection(
            name=self.collection_name
        )

        print(
            f"Deleted collection: "
            f"{self.collection_name}"
        )

    # -----------------------------
    # Show Summary
    # -----------------------------

    def summary(self):

        print("\nCollection Name:")
        print(self.collection_name)

        print("\nDocument Count:")
        print(self.count())

        print("\nAvailable Collections:")
        print(self.list_collections())


# class VectorDBBuilder:
#     DATASETS = {
#         "legal": ["cuad"],

#         "finance": [
#             "finqa",
#             "tatqa"
#         ],

#         "biological": [
#             "pubmedqa",
#             "covidqa"
#         ],

#         "customer_support": [
#             "delusionqa",
#             "emanual",
#             "techqa"
#         ],

#         "general_knowledge": [
#             "hotpotqa",
#             "msmarco",
#             "hagrid",
#             "expertqa"
#         ]
#     }

#     def __init__(
#         self,
#         collection_name: str,
#         embedding_function,
#         path: str = "./chroma_db"
#     ):

#         self.client = chromadb.PersistentClient(
#             path=path
#         )

#         self.collection_name = collection_name

#         self.embedding_function = embedding_function

#         self.collection = (
#             self.client.get_or_create_collection(
#                 name=self.collection_name,
#                 embedding_function=self.embedding_function
#             )
#         )

#     # -----------------------------
#     # Collection Utilities
#     # -----------------------------

#     def list_collections(self):

#         return [
#             collection.name
#             for collection in self.client.list_collections()
#         ]

#     def collection_exists(self):

#         return (
#             self.collection_name
#             in self.list_collections()
#         )

#     def count(self):

#         return self.collection.count()

#     def is_empty(self):

#         return self.count() == 0

#     # -----------------------------
#     # Insert
#     # -----------------------------

#     def build_vector_db(
#         self,
#         chunks: list,
#         dataset_name: str,
#         domain: str
#     ):

#         ids = []
#         metadatas = []

#         start_count = self.count()

#         for idx, chunk in enumerate(chunks):

#             ids.append(
#                 f"{domain}_{dataset_name}_{start_count + idx}"
#             )

#             metadatas.append(
#                 {
#                     "dataset": dataset_name,
#                     "domain": domain
#                 }
#             )

#         self.collection.add(
#             documents=chunks,
#             ids=ids,
#             metadatas=metadatas
#         )

#         print(
#             f"Inserted {len(chunks)} chunks "
#             f"into {self.collection_name}"
#         )

#     # -----------------------------
#     # Query
#     # -----------------------------

#     def query(
#         self,
#         query: str,
#         top_k: int = 10
#     ):

#         results = self.collection.query(
#             query_texts=[query],
#             n_results=top_k
#         )

#         return results

#     # -----------------------------
#     # Delete Collection
#     # -----------------------------

#     def delete_collection(self):

#         self.client.delete_collection(
#             name=self.collection_name
#         )

#         print(
#             f"Deleted collection: "
#             f"{self.collection_name}"
#         )

#     # -----------------------------
#     # Show Summary
#     # -----------------------------

#     def summary(self):

#         print("\nCollection Name:")
#         print(self.collection_name)

#         print("\nDocument Count:")
#         print(self.count())

#         print("\nAvailable Collections:")
#         print(self.list_collections())




# import chromadb
# from chromadb.utils import embedding_functions

# class VectorDBBuilder:
#     def __init__(self, collection_name: str, embedding_function: embedding_functions, path: str = None):
#         self.client = chromadb.PersistentClient(path=path)
#         self.collection_name = collection_name
#         self.embedding_function = embedding_function
        

#     def build_vector_db(self, chunks: list, dataset_name: str, domain: str):
#         # Create a new collection
#         collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.embedding_function)

#         # Add documents to the collection
#         ids = []
#         metadatas = []
#         for idx, chunk in enumerate(chunks):
#             ids.append(f"{domain}_{dataset_name}_{idx}")
#             metadatas.append(
#                 {"dataset": dataset_name,
#                  "domain": domain
#                 })
#         collection.add(
#             documents=chunks,
#             ids=ids, 
#             metadatas=metadatas)
        

#     def list_collections(self):
#         return self.client.list_collections()

#     def delete_collection(self):
#         self.client.delete_collection(name=self.collection_name)


