from pypdf import PdfReader


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
                }
            )

    return pages_text


def extract_document(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        text = extract_text_from_txt(uploaded_file)
        return [
            {
                "page": 1,
                "text": text,
            }
        ]

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    return []