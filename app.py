import numpy as np
import faiss
import os
from dotenv import load_dotenv
from pdfprocessing import pdf_to_text, clean_text
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pickle  

load_dotenv()
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
#gemini api key
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

def load_faiss_index(file_name='faiss_index'):
    """Load FAISS index from a file."""
    return faiss.read_index(file_name)

def load_chunks(file_name='chunks.pkl'):
    """Load chunks from a file if it exists."""
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def retrieve_relevant_chunks(question, index, chunks, top_k=5):
    """Retrieve top-k most relevant chunks for the given question."""
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

def generate_answer_with_gemini(question, relevant_chunks):
    """Generate an answer by augmenting with retrieved chunks using the Gemini model."""
    context = ' '.join(relevant_chunks)  # Combine relevant chunks
    prompt = f"Given Context: {context}\nQuestion: {question}\nAnswer: if the answer is not there in the document don't answer. Let the answer be precise to the question"
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content([prompt])

    return response.text

if __name__ == "__main__":
    # Load FAISS index and chunks
    faiss_index = load_faiss_index()
    chunks = load_chunks()

    # Example usage: Ask a question
    question = "What are some of the key challenges and growth opportunities identified in India’s diagnostics sector?"
    relevant_chunks = retrieve_relevant_chunks(question, faiss_index, chunks)
    
    # Generate the answer using the Gemini model
    answer = generate_answer_with_gemini(question, relevant_chunks)
    print("Answer:", answer)
