import fitz  
import re

def pdf_to_text(pdf_file):
    """Extract text from PDF file."""
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    """Clean and split text into smaller chunks."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    chunks = text.split('. ')  # Split by sentence or paragraph
    return chunks

if __name__ == "__main__":
    pdf_text = pdf_to_text('investmentopportunities_healthcare.pdf')
    chunks = clean_text(pdf_text)
    print(chunks[:10])  