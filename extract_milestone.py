import re, zipfile
from pathlib import Path

base = Path('.')
doc = base / 'Milestone 1.docx'

try:
    with zipfile.ZipFile(doc) as z:
        xml = z.read('word/document.xml')
    text = xml.decode('utf-8', 'ignore')
    text = re.sub(r'<.*?>', '\n', text, flags=re.S)
    text = re.sub(r'\s+', ' ', text).strip()
    print(text[:2000])
except Exception as e:
    print(f'Error: {e}')
