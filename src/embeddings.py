# from langchain_chroma import Chroma
# #from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from uuid import uuid4

# def initialize_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
#     """Initialize the embedding model"""
#     return HuggingFaceEmbeddings(model_name=model_name)

# def create_vector_store(chunks, embeddings, persist_directory="./chroma_langchain_db"):
#     """Create and populate the vector store"""
#     vector_store = Chroma(
#         collection_name="kcc_collection",
#         embedding_function=embeddings,
#         persist_directory=persist_directory,
#     )
    
#     uuids = [str(uuid4()) for _ in range(len(chunks))]
#     vector_store.add_documents(documents=chunks, ids=uuids)
#     return vector_store

from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
