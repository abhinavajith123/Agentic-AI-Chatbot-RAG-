import pdfplumber


# This function 
def load_pdf(path: str):
    pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            # pdf.pages is a list  page object
            text = page.extract_text() # extract text from the page object
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1
                })

    return pages
