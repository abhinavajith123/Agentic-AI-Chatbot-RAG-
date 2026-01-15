## Short Architecture:
### PDF Ingestion & Chunking:

PDFs are loaded using pdfplumber, with each page’s text extracted and stored with page metadata.

### Embedding & Vector Store (FAISS):

Text chunks are converted into embeddings using SentenceTransformers (all-MiniLM-L6-v2).

Embeddings are normalized and stored in a FAISS index for fast similarity search.

Chunk metadata is saved alongside the index for retrieval.

### RAG Pipeline (LangGraph):

Retrieve Node: Retrieves top K most relevant chunks from FAISS based on query embeddings.

Conditional Node: Checks if retrieved chunks are grounded using is_grounded.

Generate Node: If grounded, generates an answer with gen_ans and calculates confidence based on chunk scores.

Fallback Node: Returns a default response when no grounded information is found.

Flow Control: LangGraph orchestrates execution with conditional edges, starting from retrieve and terminating at either generate or fallback.

### API Layer (FastAPI):

Exposes endpoints to accept user queries and return RAG-based responses.

### UI Layer (Streamlit):

Interactive frontend for submitting questions and displaying answers with confidence scores.

## Instruction 
1. Clone the repository

2. Create a virtual environment (optional)

3. Create a .env file in the project root and add your API key
Format:
 GROQ_API_KEY=your_api_key_here

4. Install required dependencies
```
pip install -r requirement.txt
```

6. Ingest the PDF and build the FAISS vector index
```
python scripts/ingest.py
```
7. Start the FastAPI backend server
```
uvicorn app.main:app --reload
```
9. In a new terminal, run the Streamlit frontend
```
streamlit run app/ui.py
```

## Sample queries :
What is Agentic AI
What are agents
What are the characteristics of an Agent 
What are the categories and Types of Agentic Systems
Whatis the capital of India





