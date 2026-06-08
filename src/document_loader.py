from rapidocr_onnxruntime import RapidOCR
from pypdf import PdfReader


ocr_engine = RapidOCR()


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages_text.append(
                {
                    "page": page_number,
                    "text": text,
                    "source_type": "pdf",
                }
            )

    return pages_text


def extract_text_from_image(uploaded_file):
    image_bytes = uploaded_file.read()
    result, _ = ocr_engine(image_bytes)

    if not result:
        return []

    extracted_lines = []

    for item in result:
        text = item[1]

        if text:
            extracted_lines.append(text)

    extracted_text = "\n".join(extracted_lines)

    if not extracted_text.strip():
        return []

    return [
        {
            "page": 1,
            "text": extracted_text,
            "source_type": "image",
        }
    ]


def extract_document(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        text = extract_text_from_txt(uploaded_file)
        return [
            {
                "page": 1,
                "text": text,
                "source_type": "txt",
            }
        ]

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith((".jpg", ".jpeg", ".png")):
        return extract_text_from_image(uploaded_file)

    return []