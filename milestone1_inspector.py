import json
import re
import zipfile
from pathlib import Path

import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')

# 1) Extract milestone text from Milestone 1.docx
milestone_path = base / 'Milestone 1.docx'
text_path = base / 'milestone1_text.txt'
if milestone_path.exists():
    with zipfile.ZipFile(milestone_path) as z:
        xml = z.read('word/document.xml')
    text = xml.decode('utf-8', 'ignore')
    text = re.sub(r'<.*?>', '\n', text, flags=re.S)
    text = re.sub(r'\s+', ' ', text).strip()
    text_path.write_text(text, encoding='utf-8')
    print('MILESTONE_TEXT_SAVED')
    print(text[:4000])
else:
    print('MILESTONE_NOT_FOUND')

# 2) Inspect dataset
xlsx_path = base / 'Samples EJP Probefield.xlsx'
if xlsx_path.exists():
    df = pd.read_excel(xlsx_path)
    summary = {
        'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
        'columns': df.columns.tolist(),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
        'missing_values': {str(k): int(v) for k, v in df.isna().sum().items()},
        'head': df.head(10).to_dict(orient='records'),
    }
    (base / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nDATASET_SUMMARY_SAVED')
    print(json.dumps({'shape': summary['shape'], 'columns': summary['columns']}, indent=2))
    print('\nHEAD:')
    print(df.head(10).to_string(index=False))
else:
    print('DATASET_NOT_FOUND')
