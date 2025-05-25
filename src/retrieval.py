from langchain_chroma import Chroma

def vector_store_retrival(chunks,embeddings):
    vector_store = Chroma(
        collection_name="kcc_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

# def create_vector_store(chunks, embeddings, persist_dir="./chroma_langchain_db"):
#     vector_store = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=persist_dir,
#         collection_name="kcc_collection"
#     )
#     vector_store.persist()
#     return vector_store

# def get_retriever(embeddings, persist_dir="./chroma_langchain_db"):
#     vector_store = Chroma(
#         persist_directory=persist_dir,
#         embedding_function=embeddings,
#         collection_name="kcc_collection"
#     )
#     return vector_store.as_retriever(search_kwargs={"k": 3})
