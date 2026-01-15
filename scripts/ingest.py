from app.ingestion.load_pdf import load_pdf
from app.ingestion.embed_store import build_faiss_index
from app.ingestion.chunker import chunk_text

pdf_path = "data/Ebook-Agentic-AI.pdf"

pages =load_pdf(pdf_path)
chunks = chunk_text(pages)
build_faiss_index(chunks)

print("PDF ingestion success")