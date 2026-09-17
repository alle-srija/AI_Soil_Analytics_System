import os, json
from pathlib import Path
import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')
path = base / 'Samples EJP Probefield.xlsx'

df = pd.read_excel(path)
summary = {
    'shape': list(df.shape),
    'columns': df.columns.tolist(),
    'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
    'missing_values': {str(k): int(v) for k, v in df.isna().sum().items()},
    'head': df.head(10).to_dict(orient='records'),
}
(base / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Dataset summary saved to', base / 'dataset_summary.json')
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head(10).to_string(index=False))
