from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages):
    # pages is a list of dictionary values containing text and page number which is the result we get from load_pdf.py
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []
    # taking each page from the pages list
    for page in pages:
        splits = splitter.split_text(page["text"]) # Long sentences are split into a list of chunks
        for chunk in splits:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })
    return chunks
