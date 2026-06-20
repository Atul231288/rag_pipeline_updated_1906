# Chat model
from config.settings import RAG_MODEL
from langchain_ollama import ChatOllama
llm = ChatOllama(model=RAG_MODEL, temperature=0)