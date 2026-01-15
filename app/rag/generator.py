from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def gen_ans(question, chunks): # chunks is the result we obtain from retreiver.py
    context = "\n\n".join(
        [f"(Page {c['page']}) {c['text']}" for c in chunks]
    )

    prompt = f"""
You are an AI assistant.
Answer ONLY using the context below.Answer must contain alteast 30 words
If the answer is not present, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content
