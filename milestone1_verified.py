import json
from pathlib import Path
import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')
path = base / 'Samples EJP Probefield.xlsx'
out = base / 'milestone1_result.json'

df = pd.read_excel(path)
result = {
    'status': 'dataset_loaded',
    'rows': int(df.shape[0]),
    'columns': int(df.shape[1]),
    'column_names': df.columns.tolist(),
    'numeric_count': int(df.select_dtypes(include='number').shape[1]),
    'missing_total': int(df.isna().sum().sum()),
    'preview': df.head(5).to_dict(orient='records')
}
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'rows': result['rows'], 'columns': result['columns'], 'numeric_count': result['numeric_count'], 'missing_total': result['missing_total']}, indent=2))
