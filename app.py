import streamlit as st
from dotenv import load_dotenv
from src.data_processing import *
from src.embeddings import get_embeddings
from src.retrieval import  vector_store_retrival
from src.generation import get_llm_chain, format_docs, search_web
import os

load_dotenv()

st.set_page_config(page_title="KCC Query Assistant", layout="wide")
st.title("🌾 KCC Query Assistant")

# Setup
csv_path = "data/questionsv3.csv"
embeddings = get_embeddings()

# Create vector DB if not already
if not os.path.exists("./chroma_langchain_db/index"):
    chunks = create_chunks(csv_path)
    

retriever = vector_store_retrival(chunks, embeddings)
llm_chain = get_llm_chain()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:")

if query:
    context_docs = retriever.get_relevant_documents(query)
    if context_docs:
        context = format_docs(context_docs)
        response = llm_chain.invoke({"context": context, "question": query})
    else:
        response = "⚠️ No local context found. Searching the web...\n\n"
        web_result = search_web(query)
        response += f"🌐 **Internet Search Result:**\n\n{web_result}"

    st.session_state.history.append({"query": query, "response": response})

# Display history
if st.session_state.history:
    st.subheader("📜 Chat History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"**Q:** {chat['query']}")
        st.markdown(f"**A:** {chat['response']}")
        st.markdown("---")
