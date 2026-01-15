1. Clone the repository

2. Create a virtual environment (optional)

3. Create a .env file in the project root and add your API key
Format:
 GROQ_API_KEY=your_api_key_here

4. Install required dependencies
pip install -r requirements.txt

5. Ingest the PDF and build the FAISS vector index
python scripts/ingest.py

6. Start the FastAPI backend server
uvicorn app.main:app --reload

7. In a new terminal, run the Streamlit frontend
streamlit run app/ui.py


