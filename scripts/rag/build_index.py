from src.rag.embedder import TextEmbedder
from src.rag.indexer import build_faiss_index, save_corpus
from src.utils.pdf_parser import extract_pdf
from src.utils.rag_utils import chunk_texts
from src.rag.embedder import TextEmbedder
from src.configs.config_manager import ConfigManager
import os
import numpy as np
from argparse import ArgumentParser


def main(config: ConfigManager):
    rag_cfg = config.rag

    paragraphs = extract_pdf(
        rag_cfg.source_files,
        rag_cfg.min_length
    )
    print(f"총 문단 수: {len(paragraphs)}")

    # ✅ 문단 → 청크
    chunks = chunk_texts(
        paragraphs,
        chunk_size=rag_cfg.chunk_size,
        chunk_overlap=rag_cfg.chunk_overlap
    )

    print(f"청크 수: {len(chunks)}")

    # ✅ 청크를 임베딩
    embedder = TextEmbedder(model_name=rag_cfg.model_id)
    embeddings = embedder.encode(chunks)


    # ✅ 인덱스와 코퍼스 저장
    build_faiss_index(np.array(embeddings), rag_cfg.index_dir, rag_cfg.index_name)
    save_corpus(chunks, rag_cfg.index_dir, rag_cfg.corpus_name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Build RAG index from PDF files")
    parser.add_argument("--config", type=str, default="configs", help="Path to configuration directory")
    args = parser.parse_args()

    config = ConfigManager()
    config.load_all_configs(args.config)
    main(config)
