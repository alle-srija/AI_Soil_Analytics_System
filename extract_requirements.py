import os, re, zipfile, json
base = r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory'
path = os.path.join(base, 'Milestone 1.docx')
print('FILE EXISTS', os.path.exists(path))
if os.path.exists(path):
    z = zipfile.ZipFile(path)
    xml = z.read('word/document.xml')
    text = re.sub(r'<.*?>', '\n', xml.decode('utf-8', 'ignore'))
    text = re.sub(r'\s+', ' ', text)
    print(text[:12000])
    out = os.path.join(base, 'milestone1_text.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print('WROTE', out)

xlsx = os.path.join(base, 'Samples EJP Probefield.xlsx')
print('XLSX EXISTS', os.path.exists(xlsx))
if os.path.exists(xlsx):
    import pandas as pd
    df = pd.read_excel(xlsx)
    print('SHAPE', df.shape)
    print(df.head().to_string())
    print('COLUMNS', list(df.columns))
