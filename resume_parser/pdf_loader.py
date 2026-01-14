import pdfplumber

def extract_text_from_pdf(file)->str:
    text=[]
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text.append(page_text)
    
    return "\n".join(text)

def extract_text_from_multiple_pdfs(files:list)->list[dict]:
    extracted=[]
    for file in files:
        try:
            text=extract_text_from_pdf(file)
            extracted.append({
                "filename":file.name,
                "text":text
            })
        except Exception as e:
            extracted.append(
                {
                    "filename":file.name,
                    "text":"",
                    "error":str(e)
                }
            )
    return extracted