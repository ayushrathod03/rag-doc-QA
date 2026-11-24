import os
import tempfile
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a strict, accurate document assistant.
Answer the user's question using ONLY the provided context passages below.
Rules:
1. Do not use outside knowledge or make assumptions.
2. If the answer cannot be found in the context, state: "I cannot find the answer to this question in the uploaded documents."
3. Cite your sources inline using bracketed numbers like [1], [2] corresponding to the Context IDs provided.

Context:
{context}
"""

def _chunk_and_index(
    all_docs: List[Document],
    api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[FAISS, List[Document]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks


def process_uploaded_files(
    uploaded_files, 
    api_key: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> Tuple[FAISS, List[Document]]:
    """Loads PDFs/TXT/MD files, splits them into chunks, and builds a FAISS index."""
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
            all_docs.extend(docs)
        finally:
            os.remove(tmp_path)

    return _chunk_and_index(all_docs, api_key, chunk_size, chunk_overlap)


def process_sample_file(
    sample_path: str,
    api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[FAISS, List[Document]]:
    """Loads the bundled sample document so demo visitors can test in one click."""
    loader = TextLoader(sample_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(sample_path)
        doc.metadata["page"] = 1
    return _chunk_and_index(docs, api_key, chunk_size, chunk_overlap)


def query_rag(
    vector_store: FAISS, 
    question: str, 
    api_key: str, 
    model_name: str = "gemini-3.8-flash",
    top_k: int = 4
) -> Dict[str, Any]:
    """Retrieves top_k chunks and generates a grounded, cited response."""
    results_with_scores = vector_store.similarity_search_with_score(question, k=top_k)
    
    context_blocks = []
    sources = []
    
    for idx, (doc, score) in enumerate(results_with_scores, start=1):
        source_name = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", "N/A")
        header = f"[{idx}] Source: {source_name} (Page {page_num})"
        context_blocks.append(f"{header}\n{doc.page_content}")
        sources.append({
            "id": idx,
            "source": source_name,
            "page": page_num,
            "score": round(float(score), 4),
            "content": doc.page_content
        })

    formatted_context = "\n\n---\n\n".join(context_blocks)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    llm = ChatGoogleGenerativeAI(
        model=model_name.strip(),
        temperature=0,
        google_api_key=api_key
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": formatted_context, "question": question})
    
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    
    return {
        "answer": content,
        "sources": sources
    }


def evaluate_retrieval(
    vector_store: FAISS, 
    eval_cases: List[Dict[str, str]], 
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Calculates Retrieval Hit Rate@K and Mean Reciprocal Rank (MRR)
    given a list of {"question": ..., "expected_substring": ...} pairs.
    """
    hits = 0
    reciprocal_ranks = []
    detailed_results = []

    for case in eval_cases:
        q = case["question"]
        expected = case["expected_substring"].lower().strip()
        retrieved_docs = vector_store.similarity_search(q, k=top_k)
        
        hit_rank = None
        for rank, doc in enumerate(retrieved_docs, start=1):
            if expected in doc.page_content.lower():
                hit_rank = rank
                break
        
        if hit_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)
            
        detailed_results.append({
            "question": q,
            "expected_keyword": case["expected_substring"],
            "hit": hit_rank is not None,
            "rank": hit_rank if hit_rank else "-"
        })

    total = len(eval_cases)
    hit_rate = (hits / total) * 100 if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    return {
        "hit_rate": round(hit_rate, 1),
        "mrr": round(mrr, 3),
        "total": total,
        "hits": hits,
        "details": detailed_results
    }