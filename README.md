# Grounded RAG Document Q&A

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-0467DF)](https://faiss.ai/)
[![Gemini API](https://img.shields.io/badge/Gemini%20API-gemini--3.8--flash-4285F4?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A Retrieval-Augmented Generation (RAG) document Q&A application built with **Streamlit**, **LangChain**, **FAISS**, and the **Google Gemini API**. Upload multi-page PDFs, TXT, or Markdown files and ask questions answered **strictly from retrieved passages** with inline bracketed citations (`[1]`, `[2]`), expandable source provenance (document name, page number, and L2 distance score), and an interactive **Retrieval Evaluation** benchmark calculating **Hit Rate@K** and **Mean Reciprocal Rank (MRR)**.

---

## Live Demo

**App URL:** [https://ayush-rag-doc.streamlit.app/](https://ayush-rag-doc.streamlit.app/)

No local files required to try it out:
1. Open the live app (pre-configured via Streamlit Secrets, or paste your own Gemini API key in the sidebar).
2. Click **"Load Sample Document"** in the sidebar to index the bundled [`sample_doc.md`](sample_doc.md) in one click.
3. Ask questions in the **Document Q&A** tab or run the **Retrieval Evaluation (Hit Rate)** benchmark across the retrieved FAISS chunks.

---

## System Architecture

```text
+------------------------------------------------------------------------------+
|                            1. DOCUMENT INGESTION                             |
|   Uploaded PDFs (.pdf) -----------------> PyPDFLoader (1-indexed pages)      |
|   Uploaded TXT/MD or sample_doc.md -----> TextLoader (UTF-8)                 |
+------------------------------------------------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                       2. CHUNKING & VECTOR INDEXING                          |
|   RecursiveCharacterTextSplitter                                             |
|     - Default: chunk_size=500, chunk_overlap=100                             |
|     - Separators: ["\n\n", "\n", ". ", " ", ""]                              |
|                                      |                                       |
|                                      v                                       |
|   GoogleGenerativeAIEmbeddings (models/gemini-embedding-001)                 |
|                                      |                                       |
|                                      v                                       |
|   FAISS In-Memory Vector Store                                               |
+------------------------------------------------------------------------------+
                                       |
           +---------------------------+---------------------------+
           |                                                       |
           v                                                       v
+----------------------------------------+   +---------------------------------+
|      3A. CITATION-GROUNDED Q&A         |   |    3B. RETRIEVAL EVALUATION     |
|  FAISS similarity_search_with_score    |   |  FAISS similarity_search(k=K)   |
|    - Top-K chunks (default K=4)        |   |  Compare against JSON ground-   |
|    - Tracks source, page, L2 distance  |   |  truth expected substrings      |
|                  |                     |   |                |                |
|                  v                     |   |                v                |
|  ChatGoogleGenerativeAI                |   |  Metrics:                       |
|    - Model: gemini-3.8-flash           |   |    - Hit Rate@K                 |
|    - Temperature: 0                    |   |    - Mean Reciprocal Rank (MRR) |
|    - Strict grounding + [1], [2] cites |   |    - Per-query hit & rank table |
+----------------------------------------+   +---------------------------------+
```

---

## Retrieval Evaluation & Hyperparameter Tuning

To validate retrieval quality rather than relying on subjective chat outputs alone, the built-in evaluation tab benchmarks `FAISS` retrieval on a multi-PDF resume dataset (`ResumeFullStack.pdf` and `ResumeAI_ML.pdf`) using **Hit Rate@4** and **Mean Reciprocal Rank (MRR)**:

| Configuration | `chunk_size` | `chunk_overlap` | `top_k` | Hit Rate@4 | Mean Reciprocal Rank (MRR) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | `800` | `150` | `4` | `75.0%` | `0.688` | Larger chunks diluted specific project keywords across dense multi-PDF resumes |
| **Tuned (Default)** | `500` | `100` | `4` | **`100.0%`** | **`1.000`** | Tighter semantic boundaries improved top-1 passage ranking across all benchmark queries |

---

## Screenshots

### Multi-PDF Retrieval Evaluation Benchmark (`75.0% Hit Rate@4`, `0.688 MRR`)
![Hit Rate and MRR Evaluation Benchmark](assests/sample%20doc.png)

### Retrieval Benchmark Run (`ResumeFullStack.pdf` & `ResumeAI_ML.pdf`)
![Resume Retrieval Evaluation Run](assests/resume%20OP.png)

---

## Project Structure

```text
rag-doc-QA/
├── app.py               # Streamlit UI (sidebar controls, Document Q&A, Retrieval Evaluation)
├── rag_engine.py        # Ingestion, chunking, FAISS indexing, grounded Gemini Q&A, Hit Rate/MRR eval
├── sample_doc.md        # Bundled sample document for one-click live demo testing
├── requirements.txt     # Python dependencies
├── assests/             # Application screenshots
│   ├── sample doc.png
│   └── resume OP.png
└── .gitignore
```

---

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushrathod03/rag-doc-QA.git
   cd rag-doc-QA
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Gemini API key (optional)**
   You can paste your key directly in the Streamlit sidebar, or set an environment variable:
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="AIza..."
   # macOS / Linux
   export GEMINI_API_KEY="AIza..."
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
3. Select the repository (`ayushrathod03/rag-doc-QA`), branch (`main`), and main file path (`app.py`).
4. *(Optional, recommended for public demos)* In **App settings -> Secrets**, add your Gemini API key so visitors can test **"Load Sample Document"** immediately:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy**.
