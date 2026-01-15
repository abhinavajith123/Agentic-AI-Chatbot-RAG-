import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, FAISS_INDEX_PATH 

# This function convert chunks to vector and store in vector db
def build_faiss_index(chunks):
    # The same embedding model is used for retreival also 
    # initializes the  pre-trained sentence-transformers model all-MiniLM-L6-v2
    model = SentenceTransformer(EMBEDDING_MODEL) 
    texts = [c["text"] for c in chunks] # To access only the text part

    embeddings = model.encode(texts, normalize_embeddings=True)
    dim = embeddings.shape[1] 

    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings)) # stores all  chunk vectors into FAISS.

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(f"{FAISS_INDEX_PATH}_meta.pkl", "wb") as f:
        pickle.dump(chunks, f) 
