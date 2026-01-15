from app.config import SIMILARITY_THRESHOLD

def is_grounded(chunks):
    for i in chunks:
        if i["score"] >= SIMILARITY_THRESHOLD:
            return True
    return False