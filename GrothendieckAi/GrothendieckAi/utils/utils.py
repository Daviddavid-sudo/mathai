# GrothendieckAi/utils.py

from .utils.embed_and_index import (
    load_chunks,
    embed_chunks as embed_chunks_advanced,
    build_faiss_index,
    save_index as save_faiss_index,
    save_texts,
)
from .utils.extract_and_chunk import (
    load_text,
    clean_text,
    split_into_chunks,
    embed_chunks as embed_chunks_simple,
    build_faiss_index as build_faiss_index_simple,
    save_index as save_faiss_index_simple,
    save_chunks,
)
from .utils.query_llama import (
    load_index_and_texts,
    embed_query,
    search_index,
    query_ollama_llama32,
    fix_math_latex_with_llama,
    save_math_answer_as_html,
)

# Optionally, you can define wrapper functions or aliases for convenience:

def embed_and_index_chunks(txt_path="output_chunks.txt"):
    chunks = load_chunks(txt_path)
    embeddings, texts = embed_chunks_advanced(chunks)
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    save_texts(texts)
    return index, texts

def extract_chunk_and_index(text_path="output_chunks.txt"):
    raw_text = load_text(text_path)
    chunks = split_into_chunks(raw_text)
    embeddings, cleaned_chunks = embed_chunks_simple(chunks)
    index = build_faiss_index_simple(embeddings)
    save_faiss_index_simple(index)
    save_chunks(cleaned_chunks)
    return index, cleaned_chunks

def query_with_llama(question, index=None, texts=None, k=10):
    if index is None or texts is None:
        index, texts = load_index_and_texts()
    query_embedding = embed_query(question)
    indices = search_index(query_embedding, index, k)
    relevant_chunks = [texts[i] for i in indices if i < len(texts)]
    answer_raw = query_ollama_llama32(relevant_chunks, question)
    answer_fixed = fix_math_latex_with_llama(answer_raw)
    return answer_fixed

# You can also export all functions if needed
__all__ = [
    # embed_and_index.py exports
    "load_chunks",
    "embed_chunks_advanced",
    "build_faiss_index",
    "save_faiss_index",
    "save_texts",

    # extract_and_chunk.py exports
    "load_text",
    "clean_text",
    "split_into_chunks",
    "embed_chunks_simple",
    "build_faiss_index_simple",
    "save_faiss_index_simple",
    "save_chunks",

    # query_llama.py exports
    "load_index_and_texts",
    "embed_query",
    "search_index",
    "query_ollama_llama32",
    "fix_math_latex_with_llama",
    "save_math_answer_as_html",

    # convenience wrappers
    "embed_and_index_chunks",
    "extract_chunk_and_index",
    "query_with_llama",
]
