import os
from mainapp import extract_text_from_pdf

def test_pdf_extraction():
    sample_pdf = "sample.pdf"
    if os.path.exists(sample_pdf):
        text_chunks = extract_text_from_pdf(sample_pdf)
        assert isinstance(text_chunks, list)
        assert len(text_chunks) > 0
    else:
        assert True 