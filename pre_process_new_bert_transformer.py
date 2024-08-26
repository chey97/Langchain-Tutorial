import fitz  # PyMuPDF
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Directory and files
pdf_files = [
    # "pdf_doc/2023 List of Exempt Machinery and Equipment Steel Products.pdf",
    # "pdf_doc/10_RFI_Answered.pdf",
    # "pdf_doc/05_RFI_Answered.pdf",
    "pdf_doc/glass_factory_twin_sensors.pdf"
]
json_file = "glass_factory_paragraph_chunks.json"

# Function to extract paragraphs from each page of the PDF
def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            page_paragraphs = text.split('\n\n')  # Split by double newline, paragraphs are separated by it
            for para in page_paragraphs:
                if para.strip():
                    paragraphs.append({
                        'page': page_num + 1,
                        'text': para.strip(),
                        'pdf': os.path.basename(pdf_path)
                    })
    return paragraphs

# Extract paragraphs from all PDFs
all_paragraph_chunks = []
for pdf_file in pdf_files:
    all_paragraph_chunks.extend(extract_paragraphs_from_pdf(pdf_file))

print(all_paragraph_chunks)

# Function to chunk text into smaller pieces
def chunk_text(text, max_tokens=512):
    """Splits text into chunks that fit within the maximum token limit."""
    words = text.split()
    current_chunk = []
    current_length = 0
    chunks = []
    
    for word in words:
        if current_length + len(word) + 1 > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings for text chunks
def generate_embeddings(texts):
    embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Prepare texts and generate embeddings
texts = [chunk['text'] for chunk in all_paragraph_chunks]

embeddings = generate_embeddings(texts)

if embeddings.shape[0] == 0:
    raise ValueError("No embeddings were generated.")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "gf-faiss-index.bin")

data = {
    'paragraph_chunks': all_paragraph_chunks,
    'embeddings': embeddings.tolist()
}

with open(json_file, 'w') as f:
    json.dump(data, f, indent=2)

print("FAISS index and JSON file saved successfully.")
