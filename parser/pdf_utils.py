import pdfplumber

def extract_text_from_pdf(pdf_path_or_file):
    """
    Extracts text from a given PDF file or file-like object.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path_or_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()
