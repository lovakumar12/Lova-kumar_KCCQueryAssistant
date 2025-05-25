import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_llm_chain():
    llm = ChatOllama(model="gemma3:1b", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly using the context."),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}")
    ])

    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def search_web(query):
    search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    return search_tool.run(query)
