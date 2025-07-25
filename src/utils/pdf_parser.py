import fitz
import os
import json


def extract_pdf(files: list[str], min_len: 200):
    all_paragraphs = []

    for pdf_path in files:
        doc = fitz.open(pdf_path)
        paragraphs = []

        buffer = ""
        for page in doc:
            text = page.get_text("text")
            print(text)
            exit()

    all_paragraphs.append(paragraphs)



