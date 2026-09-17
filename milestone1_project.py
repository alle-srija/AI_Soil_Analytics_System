import json
from pathlib import Path

import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')
excel_path = base / 'Samples EJP Probefield.xlsx'
summary_path = base / 'dataset_summary.json'

if not excel_path.exists():
    raise FileNotFoundError(f'Dataset not found: {excel_path}')

df = pd.read_excel(excel_path)
summary = {
    'rows': int(df.shape[0]),
    'columns': int(df.shape[1]),
    'column_names': df.columns.tolist(),
    'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
    'missing_values': {str(k): int(v) for k, v in df.isna().sum().items()},
    'sample_rows': df.head(10).to_dict(orient='records'),
}
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Dataset summary saved to:', summary_path)
print('Row count:', summary['rows'])
print('Column count:', summary['columns'])
print('Columns:', summary['column_names'])
print('\nPreview:')
print(df.head(10).to_string(index=False))
