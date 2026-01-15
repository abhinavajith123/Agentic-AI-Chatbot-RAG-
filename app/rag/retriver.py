import pickle
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL,FAISS_INDEX_PATH,TOP_K
import faiss
import numpy as np
model = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(f"{FAISS_INDEX_PATH}_meta.pkl", "rb") as f:
    metadata = pickle.load(f)
# metadata is list of dictionaries of chunk,page no. pair

def retrive(query:str):
    q_ebd = model.encode([query],normalize_embeddings=True)
    # getting the similarity score and corresponding index
    scores, inds = index.search(np.array(q_ebd), TOP_K) # returns 2D array (list of list)

    results=[]
    for score,idx in zip(scores[0],inds[0]): #
        results.append({
            "text" : metadata[idx]["text"],
            "page": metadata[idx]["page"],
            "score": float(score)
        })
    return results

