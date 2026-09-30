import sys
import PyPDF2

pdf_path = r"C:\\Users\\Aayush Kushwaha\\Downloads\\BITS\\Acads\\3-1\\AI\\Lab\\Lab(30_09_2026)\\BN_lab.pdf"

reader = PyPDF2.PdfReader(pdf_path)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"
with open(r"C:\\Users\\Aayush Kushwaha\\Downloads\\BITS\\Acads\\3-1\\AI\\Lab\\Lab(30_09_2026)\\bn_lab_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
