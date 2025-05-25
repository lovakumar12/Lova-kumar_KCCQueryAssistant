import streamlit as st
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.render import render_text_description


from dotenv import load_dotenv
load_dotenv()

import os
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")


# Set threshold for similarity score
RELEVANCE_THRESHOLD = 0.5  # You can tune this as needed

# Load Vector Store and Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    collection_name="kcc_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = ChatOllama(model="gemma3:1b", temperature=0)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based only on the provided context."),
    ("human", "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}")
])

# Web Search Tool
search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

# Format context
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Chain (vector + LLM)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI Setup
st.set_page_config(page_title="🌾 Kisan QA Assistant", layout="wide")
st.title("🌿 Kisan Question Answering Assistant")

# Session state to store Q&A history
if "history" not in st.session_state:
    st.session_state.history = []

# Input from user
question = st.text_input("❓ Ask your question:", key="question_input")

# On submit
if st.button("Get Answer") and question.strip():
    with st.spinner("Fetching the best answer..."):
        docs = retriever.get_relevant_documents(question)
        if docs and embeddings.embed_query(question):
            # Check relevance (crude heuristic — presence of content)
            context_text = format_docs(docs)
            if context_text.strip():
                answer = chain.invoke(question)
                source = "🔍 From local knowledge base."
            else:
                docs = []
        else:
            docs = []

        # If no relevant docs, fallback to web
        if not docs:
            web_results = search_tool.invoke({"query": question})
            context_text = "\n\n".join([r["content"] for r in web_results[:3]])
            web_prompt = f"Use this online content to answer:\n\n{context_text}\n\nQuestion: {question}"
            answer = llm.invoke(web_prompt)
            source = "🌐 Answered via live internet search (no relevant local context found)."

        # Save Q&A in session state
        st.session_state.history.insert(0, {
            "question": question,
            "answer": answer,
            "source": source
        })

# Display latest Q&A
if st.session_state.history:
    st.subheader("💬 Latest Answer")
    st.markdown(f"**Q:** {st.session_state.history[0]['question']}")
    st.markdown(f"**A:** {st.session_state.history[0]['answer']}")
    st.markdown(f"**📌 Source:** {st.session_state.history[0]['source']}")

# Display previous history
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.subheader("🕘 Previous Q&A")
    for pair in st.session_state.history[1:]:
        st.markdown(f"**Q:** {pair['question']}")
        st.markdown(f"**A:** {pair['answer']}")
        st.markdown(f"**📌 Source:** {pair['source']}")
        st.markdown("---")
