import streamlit as st
import requests


st.set_page_config(page_title="Agentic AI RAG Chatbot" )
st.title(" Agentic AI Chatbot")
st.write("Ask questions about the Agentic AI eBook ")


ques = st.text_input("Enter your question:", "")

if st.button("Ask") and ques.strip():
    
        try:
            # Calling  FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/chat", 
                json={"question": ques}
            ).json()

            answer = response.get("answer", "No answer returned.")
            context = response.get("context", [])
            confidence = response.get("confidence", 0.0)

            st.markdown("Answer:")
            st.write(answer)

            
            
            st.write(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
