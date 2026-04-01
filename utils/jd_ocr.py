import pdfplumber
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_jd_text(pdf_file) -> Optional[str]:
    """
    Extracts text from a Job Description PDF.
    First tries digital extraction with pdfplumber.
    If the text is too short (likely a scanned image), it falls back to OCR via pytesseract.

    Args:
        pdf_file: A file-like object or path representing the PDF.

    Returns:
        The extracted text as a string.
    """
    text_content = []
    try:
        # First attempt: standard pdfplumber extraction
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
        full_text = "\n".join(text_content).strip()
        
        # If we got meaningful digital text, return it
        if len(full_text) > 50:
            return full_text
            
        # Fallback to OCR if text is very short (meaning it's probably an image/scanned PDF)
        logger.info("Minimal digital text found. Attempting OCR fallback...")
        return _extract_with_ocr(pdf_file)
        
    except Exception as e:
        logger.error(f"Error extracting text with pdfplumber: {e}")
        # Try OCR as a fallback anyway
        return _extract_with_ocr(pdf_file)


def _extract_with_ocr(pdf_file) -> str:
    """
    Helper function to perform OCR on a PDF file using pdf2image and pytesseract.
    Note: Requires Poppler and Tesseract to be installed on the system.
    """
    try:
        from pdf2image import convert_from_bytes, convert_from_path
        import pytesseract
        
        # Determine if it's a file path or a file-like object (like from Streamlit upload)
        if hasattr(pdf_file, "read"):
            # Move cursor to start
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            images = convert_from_bytes(pdf_bytes)
        else:
            images = convert_from_path(pdf_file)
            
        ocr_text = []
        for img in images:
            text = pytesseract.image_to_string(img)
            ocr_text.append(text)
            
        return "\n".join(ocr_text).strip()
        
    except ImportError:
        logger.error("pdf2image or pytesseract not installed. Cannot perform OCR fallback.")
        return "Error: Could not extract Jd text. Scanned PDFs require `pdf2image` and `pytesseract`."
    except Exception as e:
        logger.error(f"OCR Extraction failed: {e}")
        return f"Error during OCR extraction: {str(e)}"
