import fitz
from src.utils.rag_utils import preprocess_text


def extract_pdf(files: list[str], min_length: int = 100) -> list[str]:
    all_paragraphs = []

    for pdf_path in files:
        doc = fitz.open(pdf_path)

        buffer = ""
        for page in doc:
            text = page.get_text("text")
            lines = text.splitlines()

            for line in lines:
                line = preprocess_text(line)
                if not line:
                    continue

                if len(buffer) > 0:
                    buffer += " " + line
                else:
                    buffer = line

                # 한 문장이 너무 짧으면 다음 줄과 합치기 위해 계속 누적
                if len(buffer) < 100:
                    continue
                else:
                    all_paragraphs.append(buffer.strip())
                    buffer = ""

        # 마지막 buffer 처리
        if buffer:
            all_paragraphs.append(buffer.strip())

    return all_paragraphs
