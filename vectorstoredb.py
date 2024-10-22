import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pdfprocessing import pdf_to_text, clean_text
import os
import pickle  

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    """Generate embeddings for text chunks."""
    embeddings = model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    """Create FAISS index for storing embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(embeddings)  # Add embeddings to index
    return index

def save_faiss_index(index, file_name='faiss_index'):
    """Save FAISS index to a file."""
    faiss.write_index(index, file_name)

def load_faiss_index(file_name='faiss_index'):
    """Load FAISS index from a file if it exists."""
    if os.path.exists(file_name):
        return faiss.read_index(file_name)
    return None

def save_chunks(chunks, file_name='chunks.pkl'):
    """Save chunks to a file for later use."""
    with open(file_name, 'wb') as f:
        pickle.dump(chunks, f)

def load_chunks(file_name='chunks.pkl'):
    """Load chunks from a file if it exists."""
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    return None

if __name__ == "__main__":
    # Load and preprocess the PDF
    pdf_text = pdf_to_text('investmentopportunities_healthcare.pdf')
    chunks = clean_text(pdf_text)

    # Load or create FAISS index
    faiss_index = load_faiss_index()
    if faiss_index is None:
        # Create embeddings and FAISS index
        embeddings = create_embeddings(chunks)
        faiss_index = create_faiss_index(embeddings)
        # Save the index and chunks for later use
        save_faiss_index(faiss_index)
        save_chunks(chunks)
        print("FAISS index and chunks created and saved.")
    else:
        print("FAISS index loaded.")
