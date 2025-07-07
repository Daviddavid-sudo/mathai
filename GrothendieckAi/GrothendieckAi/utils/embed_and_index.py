import os
import re
import json
import torch
import faiss
import PyPDF2
from transformers import AutoTokenizer, AutoModel

# --------- Config ---------
PDFS_DIR = "GrothendieckAi/data/pdfs/"
CHUNKS_DIR = "GrothendieckAi/data/pdf_chunks/"
os.makedirs(CHUNKS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------- PDF text extraction ---------
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages

# --------- Chunk text with overlap ---------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = re.split(r'(?<=[\.\?!])\s+(?=[A-Z\\])', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap between chunks
    final_chunks = []
    for i in range(len(chunks)):
        start = max(i - 1, 0)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined.strip())

    return final_chunks

# --------- Save chunked text file ---------
def save_chunks_file(pdf_name, pages_chunks):
    filename = os.path.join(CHUNKS_DIR, f"{pdf_name}_chunks.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for page_num, chunks in pages_chunks:
            for chunk in chunks:
                f.write(f"--- Page {page_num} ---\n")
                f.write(chunk + "\n\n")
    return filename

# --------- Load chunks from txt file ---------
def load_chunks(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    entries = text.split("--- Page ")
    chunks = []
    for entry in entries[1:]:
        page_str, content = entry.split("---", 1)
        chunks.append({
            "page": int(page_str.strip()),
            "text": content.strip()
        })
    return chunks

# --------- Mean pooling for embeddings ---------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # last hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --------- Embed chunks ---------
def embed_chunks(chunks, batch_size=32):
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_prefixed = [f"passage: {text}" for text in batch_texts]

        inputs = tokenizer(batch_prefixed, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs['attention_mask'])

        all_embeddings.append(embeddings.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return final_embeddings

# --------- Build FAISS index ---------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --------- Save FAISS index ---------
def save_index(index, path):
    faiss.write_index(index, path)

# --------- Save chunks JSON ---------
def save_chunks_with_pages(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

# --------- Process one PDF ---------
def process_pdf(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\nProcessing PDF: {pdf_name}")

    # Extract text
    pages = extract_text_from_pdf(pdf_path)

    # Chunk each page text
    pages_chunks = []
    for page_num, text in pages:
        if not text.strip():
            continue
        chunks = chunk_text(text)
        pages_chunks.append((page_num, chunks))

    # Save chunked text file
    chunk_file = save_chunks_file(pdf_name, pages_chunks)
    print(f"Saved chunk file: {chunk_file}")

    # Load chunks for embedding
    chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks for embedding.")

    # Embed chunks
    embeddings = embed_chunks(chunks)
    print(f"Embedded chunks shape: {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save index and chunks json
    index_path = chunk_file.replace(".txt", ".index")
    json_path = chunk_file.replace(".txt", ".json")
    save_index(index, index_path)
    save_chunks_with_pages(chunks, json_path)
    print(f"Saved FAISS index: {index_path}")
    print(f"Saved chunks JSON: {json_path}")
