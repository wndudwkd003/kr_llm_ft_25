import faiss
import json
import numpy as np
import os


def build_faiss_index(
    embeddings: np.ndarray,
    output_dir: str,
    index_name: str = "index.faiss"
):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, index_name))


def save_corpus(
    corpus: list[str],
    output_dir: str,
    corpus_name: str = "corpus.json"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, corpus_name), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
