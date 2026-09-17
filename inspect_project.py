import os, re, zipfile, json
from pathlib import Path

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')
doc = base / 'Milestone 1.docx'
print('DOC EXISTS:', doc.exists())
if doc.exists():
    with zipfile.ZipFile(doc) as z:
        xml = z.read('word/document.xml')
    text = re.sub(r'<.*?>', '\n', xml.decode('utf-8', 'ignore'))
    text = re.sub(r'\s+', ' ', text).strip()
    print('\n--- MILESTONE START ---\n')
    print(text[:12000])
    print('\n--- MILESTONE END ---\n')
    (base / 'milestone1_text.txt').write_text(text, encoding='utf-8')

xlsx = base / 'Samples EJP Probefield.xlsx'
print('XLSX EXISTS:', xlsx.exists())
if xlsx.exists():
    import pandas as pd
    df = pd.read_excel(xlsx)
    print('\n--- DATASET START ---\n')
    print('SHAPE:', df.shape)
    print('COLUMNS:', list(df.columns))
    print(df.head(10).to_string(index=False))
    print('\n--- DATASET END ---\n')
    summary = {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'dtypes': {c: str(dt) for c, dt in df.dtypes.items()},
        'head': df.head(5).to_dict(orient='records')
    }
    (base / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
