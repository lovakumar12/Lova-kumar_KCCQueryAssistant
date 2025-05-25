
import pandas as pd
import re
import json
from typing import List, Dict
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-z0-9@:\.\,\-\/\? ]', '', text)  # Remove special chars
    return text

def format_question(question: str) -> str:
    """Standardize question formatting"""
    if not question:
        return "No question provided"
    replacements = {
        "asking about": "What is",
        "query regarding": "What is",
        "how to": "What is the best way to",
        "?": ""  # Remove existing question marks to avoid double ??
    }
    for old, new in replacements.items():
        question = question.replace(old, new)
    return question.strip().capitalize() + "?"

def format_answer(answer: str) -> str:
    """Standardize answer formatting"""
    if not answer:
        return "No answer provided"
    replacements = {
        "suggested him to": "You can",
        "recommended to": "You should",
        "advise to": "You should",
        "advice him to": "You should",
        "ask to": "Please"
    }
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    return answer.strip().capitalize()

def preprocess_data(file_path: str = "data/questionsv3.csv") -> pd.DataFrame:
    """Process raw data into cleaned Q&A pairs"""
    try:
        df = pd.read_csv(file_path)
        df["clean_question"] = df["questions"].apply(clean_text).apply(format_question)
        df["clean_answer"] = df["answers"].apply(clean_text).apply(format_answer)
        
        # Save cleaned data
        df.to_csv("data/cleaned_kcc.csv", index=False)
        qa_data = [{"query": q, "response": a} for q, a in zip(df["clean_question"], df["clean_answer"])]
        with open("data/kcc_qa_pairs.json", "w") as f:
            json.dump(qa_data, f, indent=2)
        
        return df
    except Exception as e:
        raise Exception(f"Data processing failed: {str(e)}")

def create_chunks(file_path: str = "data/cleaned_kcc.csv", 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks for embedding"""
    try:
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return text_splitter.split_documents(loader.load())
    except Exception as e:
        raise Exception(f"Chunking failed: {str(e)}")
    
# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def load_and_split_data(csv_path):
#     loader = CSVLoader(file_path=csv_path, encoding='utf-8')
#     data = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(data)

#     return chunks
